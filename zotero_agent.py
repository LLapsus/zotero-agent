#!/usr/bin/env python3
"""
zotero_agent.py  -  Agentic RAG with ONE unified retrieval tool.

The two-tool version (zotero_agent_v0.py) gave Claude a semantic search tool and
a separate author-lookup tool. That taught routing well, but it hit a wall on
"what did Smith write about VAEs?": two independent tools can't do a real AND.
The author tool returns no abstracts, and the topic tool searches the whole
library, not just Smith's papers -- so the intersection only happened, weakly,
in the model's head.

This version replaces both with a single tool:

    search_library(author=?, topic=?)

The model no longer chooses between overlapping tools -- it just fills in
whichever parameters the question implies:

    "papers by Novak"            -> author="Novak"
    "papers about phase changes" -> topic="phase changes"
    "Novak's papers on VAEs"     -> author="Novak", topic="VAEs"

The branching moves INSIDE the tool, where it belongs:

    author only   -> exact SQL lookup over Zotero (lists all their papers)
    topic only    -> cosine search over the whole index (plain RAG)
    author+topic  -> filter to the author's papers in SQL, THEN cosine-rank
                     only that subset -- a genuine "filter + rank" join, the
                     thing neither vectors nor SQL can do alone.

The agent loop is still hand-written so you can watch each tool_use/tool_result
step, and it reuses zotero_rag.py for embeddings/index + Zotero's SQLite.

Run it:
    python zotero_agent.py                                # interactive
    python zotero_agent.py -q "Medbouhi's work on VAEs"   # one question
    python zotero_agent.py --scripted                     # author / topic / both

COSTS MONEY: every question calls Claude (topic embedding stays local/free).
Needs a built index (python zotero_rag.py --reindex), Ollama running, and
ANTHROPIC_API_KEY (zotero_rag loads .env for you).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

# Reuse the from-scratch pipeline: config, embeddings, index, and the Zotero DB
# path. Importing it also runs load_dotenv().
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zotero_rag  # noqa: E402
import ui          # noqa: E402  cosmetic ANSI styling for the interactive CLI

MODEL = zotero_rag.CLAUDE_MODEL     # same model as the rest of the project
TOP_K = zotero_rag.TOP_K
MAX_TURNS = 8                       # safety cap on the tool-use loop

SYSTEM = (
    "You help the user query their personal Zotero library. You have ONE tool, "
    "search_library, with two optional inputs:\n"
    "  - topic:  a concept/theme to rank papers by (semantic search).\n"
    "  - author: a person's surname to filter by (exact database match).\n"
    "Use `author` alone to list someone's papers, `topic` alone to find papers "
    "about a concept, and BOTH together to find a given author's papers about a "
    "given topic. Fill in whichever inputs the question implies. Answer using "
    "ONLY what the tool returns, cite paper titles, and if it returns nothing "
    "say so plainly instead of guessing."
)

TOOLS = [
    {
        "name": "search_library",
        "description": (
            "Search the user's Zotero library. Provide `author` to filter to a "
            "specific person's papers (exact, case-insensitive surname match in "
            "the database), `topic` to rank papers by semantic similarity to a "
            "concept, or BOTH to find a given author's papers about a given "
            "topic. At least one of author/topic must be given. Examples: "
            "author='Novak'; topic='phase transitions'; author='Novak' + "
            "topic='variational autoencoders'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Concept or theme to rank papers by (optional).",
                },
                "author": {
                    "type": "string",
                    "description": "Author surname to filter by (optional).",
                },
            },
        },
    },
]


# ----------------------------------------------------------------------------
# DATABASE side: look up an author's papers directly in Zotero's SQLite.
#
# Returns one dict per matched paper: key (Zotero item key, used to line up with
# the vector index), title, year, and the FULL author list (not just first
# author -- that's the detail the corpus threw away). Same read-only temp-copy
# pattern as load_corpus.
# ----------------------------------------------------------------------------

def author_papers(surname: str, db_path: Path | None = None) -> list[dict]:
    surname = surname.strip()
    if not surname:
        return []
    db_path = db_path or zotero_rag.ZOTERO_DB
    if not db_path.exists():
        return []

    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "zotero.sqlite"
        shutil.copy2(db_path, db_copy)  # snapshot; never touch the live DB
        con = sqlite3.connect(f"file:{db_copy}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        # 1. items with an author matching the surname (the WHERE clause vectors
        #    can never express).
        id_rows = con.execute(
            """
            SELECT DISTINCT i.itemID AS item_id, i.key AS key
            FROM items i
            JOIN itemCreators ic ON ic.itemID = i.itemID
            JOIN creators     c  ON c.creatorID = ic.creatorID
            WHERE LOWER(c.lastName) LIKE LOWER(:pat)
              AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            """,
            {"pat": f"%{surname}%"},
        ).fetchall()
        if not id_rows:
            con.close()
            return []
        keys = {r["item_id"]: r["key"] for r in id_rows}
        ids = list(keys)
        qmarks = ",".join("?" * len(ids))

        # 2. title + date for those items (the field pivot from load_corpus).
        meta = {
            r["item_id"]: (r["title"], r["date"])
            for r in con.execute(
                f"""
                SELECT i.itemID AS item_id,
                       MAX(CASE WHEN f.fieldName='title' THEN v.value END) AS title,
                       MAX(CASE WHEN f.fieldName='date'  THEN v.value END) AS date
                FROM items i
                JOIN itemData       d ON d.itemID  = i.itemID
                JOIN itemDataValues v ON v.valueID = d.valueID
                JOIN fields         f ON f.fieldID = d.fieldID
                WHERE i.itemID IN ({qmarks})
                GROUP BY i.itemID
                HAVING title IS NOT NULL
                """,
                ids,
            )
        }

        # 3. full author list per item.
        authors: dict[int, list[str]] = {}
        for a in con.execute(
            f"""
            SELECT ic.itemID AS item_id, c.lastName AS last
            FROM itemCreators ic
            JOIN creators c ON c.creatorID = ic.creatorID
            WHERE ic.itemID IN ({qmarks})
            ORDER BY ic.itemID, ic.orderIndex
            """,
            ids,
        ):
            if a["last"]:
                authors.setdefault(a["item_id"], []).append(a["last"])
        con.close()

    papers = []
    for item_id, (title, date) in meta.items():
        papers.append({
            "key": keys[item_id],
            "title": title,
            "year": (date or "")[:4],
            "authors": ", ".join(authors.get(item_id, [])) or "unknown",
        })
    papers.sort(key=lambda p: p["year"], reverse=True)
    return papers


# ----------------------------------------------------------------------------
# VECTOR side: cosine-rank a set of candidate corpus rows against a topic.
#
# `cand` is a list of indices into corpus/matrix. For a topic-only search that's
# every row (plain RAG); for author+topic it's just the author's rows -- so the
# ranking runs over the filtered subset, which is the whole point.
# ----------------------------------------------------------------------------

def rank_by_topic(topic: str, matrix, cand: list[int], k: int = TOP_K) -> list[tuple[int, float]]:
    q = zotero_rag.embed(topic)          # local Ollama call
    sims = matrix @ q                    # cosine similarity (unit vectors)
    top = sorted(cand, key=lambda i: -float(sims[i]))[:k]
    return [(i, float(sims[i])) for i in top]


# ---- formatting -----------------------------------------------------------

def _format_ranked(header: str, ranked: list[tuple[int, float]], corpus: list[dict]) -> str:
    blocks = [header]
    for n, (i, score) in enumerate(ranked, 1):
        h = corpus[i]
        tag = " ".join(x for x in [h.get("authors"), h.get("year")] if x)
        blocks.append(f"[{n}] ({score:.3f}) {h['title']} ({tag})\n{h['abstract']}")
    return "\n\n".join(blocks)


def _format_listing(header: str, papers: list[dict]) -> str:
    lines = [header]
    for n, p in enumerate(papers, 1):
        lines.append(f"[{n}] {p['title']} ({p['year']}) -- {p['authors']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# THE UNIFIED TOOL: branch on which inputs were filled.
# ----------------------------------------------------------------------------

def tool_search_library(topic: str, author: str, matrix, corpus) -> str:
    topic = (topic or "").strip()
    author = (author or "").strip()
    if not topic and not author:
        return "Provide at least a topic or an author."

    if author:
        papers = author_papers(author)
        if not papers:
            return f"No papers found with an author matching '{author}'."

        if not topic:
            # author only -> exact DB listing of everything they wrote.
            return _format_listing(
                f"Found {len(papers)} paper(s) by an author matching '{author}':",
                papers,
            )

        # author + topic -> rank ONLY this author's indexed papers by the topic.
        # Only papers with an abstract are in the index (they have vectors), so
        # we match by Zotero key against the corpus.
        keys = {p["key"] for p in papers}
        cand = [i for i, c in enumerate(corpus) if c["key"] in keys]
        if not cand:
            return (
                f"Found {len(papers)} paper(s) by '{author}', but none are in the "
                f"abstract index, so I can't rank them by topic. Their titles:\n"
                + _format_listing("", papers)
            )
        ranked = rank_by_topic(topic, matrix, cand)
        return _format_ranked(
            f"Papers by '{author}' ranked by relevance to '{topic}':", ranked, corpus
        )

    # topic only -> plain RAG over the whole library.
    ranked = rank_by_topic(topic, matrix, list(range(len(corpus))))
    return _format_ranked(f"Papers most related to '{topic}':", ranked, corpus)


# ----------------------------------------------------------------------------
# THE AGENT LOOP  -  hand-written, so each tool_use/tool_result step is visible.
# ----------------------------------------------------------------------------

def run_agent(question: str, client, matrix, corpus) -> None:
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        for block in resp.content:
            if block.type == "tool_use":
                arg = json.dumps(block.input, ensure_ascii=False)
                print(f"  {ui.tool('tool call')} {ui.cyan(block.name)}({arg})")

        if resp.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": resp.content})

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "search_library":
                out = tool_search_library(
                    block.input.get("topic", ""),
                    block.input.get("author", ""),
                    matrix, corpus,
                )
            else:
                out = f"Unknown tool: {block.name}"
            first = out.splitlines()[0] if out else ""
            print(f"  {ui.tool('tool result')} {ui.dim(first[:78])}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": out,
            })
        messages.append({"role": "user", "content": results})

    text = "".join(b.text for b in resp.content if b.type == "text")
    print("\n" + ui.bold("Answer:") + "\n")
    print(text or "(no text answer)")
    print()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _sample_author(corpus: list[dict]) -> str:
    """Pick a real author from the loaded library, so --scripted works on ANY
    library -- not just the one it was written against."""
    for c in corpus:
        name = (c.get("authors") or "").replace(" et al.", "").strip()
        if name:
            return name
    return "Smith"


def _scripted_questions(corpus: list[dict]) -> list[str]:
    who = _sample_author(corpus)
    return [
        f"Which papers in my library are by {who}?",               # author only -> SQL
        "What research do I have on machine learning?",            # topic only  -> vectors
        f"Of {who}'s papers, which is most about neural networks?",  # both -> filter + rank
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic RAG (unified tool) over a Zotero library.")
    ap.add_argument("-q", "--query", help="ask one question and exit")
    ap.add_argument("--scripted", action="store_true",
                    help="run a fixed demo: author-only, topic-only, and both")
    args = ap.parse_args()

    print("zotero_agent -- one unified search_library(author?, topic?) tool (COSTS MONEY)")

    try:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (zotero_rag loaded .env)
    except Exception as e:  # noqa: BLE001 - surface the real problem
        print(f"\nCould not create the Anthropic client: {e}")
        print("(Is ANTHROPIC_API_KEY set? zotero_rag loads .env automatically.)")
        return

    matrix, corpus = zotero_rag.load_index()

    if args.scripted:
        print("(example questions are built from your own library)")
        for q in _scripted_questions(corpus):
            print("\n" + "=" * 78)
            print(f"Q: {q}")
            print("=" * 78)
            run_agent(q, client, matrix, corpus)
        print("Notice the third question: the model fills BOTH author and topic,")
        print("and the tool filters to Medbouhi's papers first, then ranks them by")
        print("topic -- a real filter+rank join, not an intersection guessed in the")
        print("model's head (which is where the two-tool v0 fell down).")
        return

    if args.query:
        run_agent(args.query, client, matrix, corpus)
        return

    print("Ask about your library. Commands:  /exit (or Ctrl-D)  quit")
    while True:
        try:
            q = ui.ask("\n> ").strip()
        except EOFError:
            break
        if q in ("/exit", "/quit", "exit", "quit"):
            break
        if not q:
            continue
        run_agent(q, client, matrix, corpus)


if __name__ == "__main__":
    main()
