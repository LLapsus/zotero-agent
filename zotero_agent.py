#!/usr/bin/env python3
"""
zotero_agent.py  -  Where RAG ends and an agent begins.

zotero_rag.py answers every question by semantic retrieval: embed the question,
find the nearest abstracts, hand them to Claude. That is the right tool for
"what papers are about X" -- but it is the WRONG tool for "what papers are by
person Y". Author lookup is an exact database query (WHERE lastName = ...), not
a fuzzy similarity search, and author names aren't even in the embedded text.

So instead of picking the retrieval strategy ourselves, we give Claude two
tools and let it choose per question:

    search_by_topic(query)   -> semantic vector search   (the RAG path)
    find_by_author(surname)  -> exact SQL over Zotero     (the database path)

This is the shift from "stuff context into the prompt" (RAG) to "retrieval is a
tool the model calls" (agent). Claude reads the question, decides which tool(s)
to call, we run them, feed the results back, and it answers -- possibly after
calling BOTH (e.g. "what did Smith write about VAEs?").

The loop below is written by hand, not with the SDK's tool runner, so you can
see every step: the model emits a `tool_use` block, we execute it, we return a
`tool_result`, and we repeat until it stops calling tools. Each turn prints
which tool the model picked, so the routing is visible.

Reuses zotero_rag.py for the topic path (its retrieve/load_index) and reads
Zotero's SQLite directly for the author path -- two genuinely different data
sources for two genuinely different kinds of question.

Run it:
    python zotero_agent.py                          # interactive
    python zotero_agent.py -q "papers by Medbouhi"  # one question
    python zotero_agent.py --scripted               # a fixed routing demo

COSTS MONEY: every question calls Claude (topic embedding is still local/free).
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

# Reuse the from-scratch pipeline: its config, its SQLite path, and -- for the
# topic tool -- its retrieve()/load_index(). Importing it also runs load_dotenv().
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zotero_rag  # noqa: E402

MODEL = zotero_rag.CLAUDE_MODEL     # same model as the rest of the project
TOP_K = zotero_rag.TOP_K
MAX_TURNS = 8                       # safety cap on the tool-use loop

# The system prompt tells Claude WHICH tool fits WHICH question. Good tool
# descriptions (below) do most of the routing; this reinforces it and sets the
# grounding rule -- answer from tool results only.
SYSTEM = (
    "You help the user query their personal Zotero library. You have two tools:\n"
    "  - search_by_topic: semantic search for papers ABOUT a concept or theme.\n"
    "  - find_by_author:  exact database lookup of papers BY a named person.\n"
    "Pick the tool that matches the question. A question can need both -- e.g. "
    "'what did Smith write about diffusion models?' means find_by_author('Smith') "
    "and then reason over the topic. Answer using ONLY what the tools return, and "
    "cite paper titles. If a tool returns nothing, say so plainly instead of "
    "guessing."
)

# The descriptions are the load-bearing part of routing: Claude reads them to
# decide when each tool applies. Be explicit about the boundary between them.
TOOLS = [
    {
        "name": "search_by_topic",
        "description": (
            "Semantic (vector) search over the abstracts in the user's Zotero "
            "library. Returns the papers whose CONTENT is closest in meaning to "
            "a topic or concept. Use this for questions about WHAT papers are "
            "about -- themes, methods, ideas, findings. Examples: 'papers about "
            "phase transitions', 'work on variational autoencoders', 'anything "
            "on molecular dynamics'. Do NOT use this to find papers by a "
            "person's name -- author names are not part of the searchable text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or concept to search for.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_by_author",
        "description": (
            "Exact database lookup of papers written by a specific person, "
            "matched by surname against Zotero's author records (NOT semantic "
            "search). Use this whenever the user asks about a named author's "
            "work. Examples: 'papers by Novak', 'what did Medbouhi publish', "
            "'articles co-authored by Smith'. Matching is case-insensitive and "
            "substring-based on the surname."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "surname": {
                    "type": "string",
                    "description": "The author surname to look up.",
                },
            },
            "required": ["surname"],
        },
    },
]


# ----------------------------------------------------------------------------
# TOOL 1: search_by_topic  -  the RAG path (vectors)
#
# This is literally zotero_rag.retrieve(). We format the hits with abstracts so
# Claude has enough to actually answer a content question, same as the RAG
# pipeline's answer() does.
# ----------------------------------------------------------------------------

def tool_search_by_topic(query: str, matrix, corpus) -> str:
    hits = zotero_rag.retrieve(query, matrix, corpus, k=TOP_K)
    if not hits:
        return "No matching papers."
    blocks = []
    for n, h in enumerate(hits, 1):
        tag = " ".join(x for x in [h.get("authors"), h.get("year")] if x)
        blocks.append(f"[{n}] ({h['score']:.3f}) {h['title']} ({tag})\n{h['abstract']}")
    return "\n\n".join(blocks)


# ----------------------------------------------------------------------------
# TOOL 2: find_by_author  -  the DATABASE path (SQL)
#
# No vectors at all. We read Zotero's SQLite directly (same read-only temp-copy
# pattern as load_corpus) so we get ALL authors, always current, and matched
# exactly. This is the concrete demonstration that author lookup is a database
# query, not retrieval.
# ----------------------------------------------------------------------------

def find_papers_by_author(surname: str, db_path: Path | None = None) -> str:
    surname = surname.strip()
    if not surname:
        return "No surname given."
    db_path = db_path or zotero_rag.ZOTERO_DB
    if not db_path.exists():
        return f"Zotero DB not found at {db_path}."

    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "zotero.sqlite"
        shutil.copy2(db_path, db_copy)  # snapshot; never touch the live DB
        con = sqlite3.connect(f"file:{db_copy}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        # 1. item IDs that have an author matching the surname (case-insensitive,
        #    substring). This is the WHERE clause vectors can never express.
        id_rows = con.execute(
            """
            SELECT DISTINCT ic.itemID AS item_id
            FROM itemCreators ic
            JOIN creators c ON c.creatorID = ic.creatorID
            WHERE LOWER(c.lastName) LIKE LOWER(:pat)
              AND ic.itemID NOT IN (SELECT itemID FROM deletedItems)
            """,
            {"pat": f"%{surname}%"},
        ).fetchall()
        ids = [r["item_id"] for r in id_rows]
        if not ids:
            con.close()
            return f"No papers found with an author surname matching '{surname}'."

        qmarks = ",".join("?" * len(ids))

        # 2. title + date for those items (the same field pivot as load_corpus).
        meta_rows = con.execute(
            f"""
            SELECT i.itemID AS item_id,
                   MAX(CASE WHEN f.fieldName = 'title' THEN v.value END) AS title,
                   MAX(CASE WHEN f.fieldName = 'date'  THEN v.value END) AS date
            FROM items i
            JOIN itemData       d ON d.itemID  = i.itemID
            JOIN itemDataValues v ON v.valueID = d.valueID
            JOIN fields         f ON f.fieldID = d.fieldID
            WHERE i.itemID IN ({qmarks})
            GROUP BY i.itemID
            HAVING title IS NOT NULL
            """,
            ids,
        ).fetchall()

        # 3. the FULL author list per item (not just first-author -- that is the
        #    detail the corpus threw away).
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

    if not meta_rows:
        return f"No papers found with an author surname matching '{surname}'."

    lines = [f"Found {len(meta_rows)} paper(s) with an author matching '{surname}':"]
    for n, r in enumerate(meta_rows, 1):
        who = ", ".join(authors.get(r["item_id"], [])) or "unknown"
        year = (r["date"] or "")[:4]
        lines.append(f"[{n}] {r['title']} ({year}) -- {who}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# THE AGENT LOOP  -  hand-written tool-use loop, so every step is visible.
#
# call the model -> if it wants a tool, run it and return a tool_result ->
# repeat until it stops calling tools. We print each tool the model picks so
# you can watch the routing happen.
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

        # Show what the model decided this turn.
        for block in resp.content:
            if block.type == "tool_use":
                arg = json.dumps(block.input, ensure_ascii=False)
                print(f"  [tool call] {block.name}({arg})")

        if resp.stop_reason != "tool_use":
            break

        # Preserve the assistant turn verbatim (incl. any thinking/tool_use
        # blocks) -- the API needs it to match up the tool results.
        messages.append({"role": "assistant", "content": resp.content})

        # Execute every tool the model asked for, collect all results into ONE
        # user message (the API expects them batched together).
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "search_by_topic":
                out = tool_search_by_topic(block.input.get("query", ""), matrix, corpus)
            elif block.name == "find_by_author":
                out = find_papers_by_author(block.input.get("surname", ""))
            else:
                out = f"Unknown tool: {block.name}"
            first = out.splitlines()[0] if out else ""
            print(f"  [tool result] {first[:78]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": out,
            })
        messages.append({"role": "user", "content": results})

    text = "".join(b.text for b in resp.content if b.type == "text")
    print("\nAnswer:\n")
    print(text or "(no text answer)")
    print()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

SCRIPT = [
    "Which papers in my library were written by Medbouhi?",             # -> author (SQL)
    "What research do I have on phase transitions in machine learning?", # -> topic (vectors)
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic RAG over a Zotero library.")
    ap.add_argument("-q", "--query", help="ask one question and exit")
    ap.add_argument("--scripted", action="store_true",
                    help="run a fixed routing demo (author question + topic question)")
    args = ap.parse_args()

    print("zotero_agent -- Claude picks the retrieval tool per question (COSTS MONEY)")

    try:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (zotero_rag loaded .env)
    except Exception as e:  # noqa: BLE001 - surface the real problem
        print(f"\nCould not create the Anthropic client: {e}")
        print("(Is ANTHROPIC_API_KEY set? zotero_rag loads .env automatically.)")
        return

    matrix, corpus = zotero_rag.load_index()  # for the topic tool

    if args.scripted:
        for q in SCRIPT:
            print("\n" + "=" * 78)
            print(f"Q: {q}")
            print("=" * 78)
            run_agent(q, client, matrix, corpus)
        print("Notice the routing: the author question went to find_by_author (SQL),")
        print("the topic question to search_by_topic (vectors). Same interface, two")
        print("completely different data paths -- and the model chose.")
        return

    if args.query:
        run_agent(args.query, client, matrix, corpus)
        return

    print("Ask about your library (empty line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        run_agent(q, client, matrix, corpus)


if __name__ == "__main__":
    main()
