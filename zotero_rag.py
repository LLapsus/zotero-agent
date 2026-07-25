#!/usr/bin/env python3
"""
zotero_rag.py  -  A RAG pipeline over your Zotero library, built from scratch.

No LangChain. Every stage is explicit so you can see exactly what a RAG system
does under the hood:

    Zotero SQLite  ->  corpus (title + abstract per item)
    corpus         ->  embeddings  (LOCAL, via Ollama)
    embeddings     ->  vector index (a numpy matrix on disk)
    question       ->  retrieve top-k nearest abstracts (cosine similarity)
    top-k + question -> answer  (Anthropic Claude)

The only "magic" is the embedding model and the LLM. Everything between them
is a dot product and some string formatting. That's the whole point.

Deps:  pip install anthropic numpy requests
Needs: - Ollama running with an embedding model:  ollama pull nomic-embed-text
       - export ANTHROPIC_API_KEY=sk-ant-...

Usage: python zotero_rag.py --reindex           # build the index (first run)
       python zotero_rag.py                     # interactive chat (remembers the
                                                #   conversation; /reset starts over)
       python zotero_rag.py -q "your question"  # single question (one-shot, no memory)
"""

from __future__ import annotations

# load .env file if present, so you can put ANTHROPIC_API_KEY in there instead of your shell
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

# ----------------------------------------------------------------------------
# 0. CONFIG  -  change these to match your machine
# ----------------------------------------------------------------------------

ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"    # default Linux location
OLLAMA_URL = "http://localhost:11434/api/embed"         # embed via Ollama running locally
EMBED_MODEL = "nomic-embed-text"                        # ollama pull nomic-embed-text
CLAUDE_MODEL = "claude-sonnet-5"                        # or "claude-haiku-4-5-20251001" (cheaper)
REWRITE_MODEL = "claude-haiku-4-5-20251001"             # cheap model: rewrites follow-ups for retrieval
INDEX_DIR = Path.home() / ".zotero_rag"                 # where we cache the vectors
TOP_K = 6                                               # how many abstracts to feed Claude


# ----------------------------------------------------------------------------
# 1. LOAD CORPUS  -  read title + abstract straight out of Zotero's SQLite.
#
# Zotero stores field values in a normalized 3-table shape:
#   items        (itemID, itemTypeID, key)
#   itemData     (itemID, fieldID, valueID)     -- which field of which item
#   itemDataValues (valueID, value)             -- the actual string
#   fields       (fieldID, fieldName)           -- 'title', 'abstractNote', ...
# We pivot that back into one row per item. Zotero locks the DB while running,
# so we copy it to a temp file and read the copy (read-only, no lock fights).
# ----------------------------------------------------------------------------

def load_corpus(db_path: Path) -> list[dict]:
    if not db_path.exists():
        sys.exit(f"Zotero DB not found at {db_path} -- fix ZOTERO_DB in the config.")

    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "zotero.sqlite"
        shutil.copy2(db_path, db_copy)  # snapshot so we never touch the live DB
        con = sqlite3.connect(f"file:{db_copy}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        rows = con.execute(
            """
            SELECT i.itemID AS item_id,
                   i.key     AS key,
                   MAX(CASE WHEN f.fieldName = 'title'        THEN v.value END) AS title,
                   MAX(CASE WHEN f.fieldName = 'abstractNote' THEN v.value END) AS abstract,
                   MAX(CASE WHEN f.fieldName = 'date'         THEN v.value END) AS date
            FROM items i
            JOIN itemData       d ON d.itemID  = i.itemID
            JOIN itemDataValues v ON v.valueID = d.valueID
            JOIN fields         f ON f.fieldID = d.fieldID
            WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
                AND i.libraryID IN (SELECT libraryID FROM libraries WHERE type != 'feed')
            GROUP BY i.itemID
            HAVING title IS NOT NULL
               AND abstract IS NOT NULL
               AND TRIM(abstract) <> ''
            """
        ).fetchall()

        # Best-effort authors: attach last names. Wrapped so a schema quirk in a
        # different Zotero version can't kill the whole run.
        authors: dict[int, list[str]] = {}
        try:
            for r in con.execute(
                """
                SELECT ic.itemID AS item_id, c.lastName AS last
                FROM itemCreators ic
                JOIN creators c ON c.creatorID = ic.creatorID
                ORDER BY ic.itemID, ic.orderIndex
                """
            ):
                if r["last"]:
                    authors.setdefault(r["item_id"], []).append(r["last"])
        except sqlite3.Error:
            pass
        con.close()

    corpus = []
    for r in rows:
        auth = authors.get(r["item_id"], [])
        who = ""
        if auth:
            who = auth[0] + (" et al." if len(auth) > 1 else "")
        corpus.append(
            {
                "key": r["key"],
                "title": r["title"].strip(),
                "abstract": r["abstract"].strip(),
                "year": (r["date"] or "")[:4],
                "authors": who,
                # This is the exact string we embed AND later show to Claude.
                # One item = one chunk. Abstracts are short, so no chunking needed.
                "text": f"{r['title'].strip()}\n\n{r['abstract'].strip()}",
            }
        )
    return corpus


# ----------------------------------------------------------------------------
# 2. EMBED  -  turn text into a vector, locally, via Ollama.
#
# This is the one function you'd swap to change embedding provider. Right now
# it's a raw HTTP POST to Ollama so you can literally see the vector come back.
# To use Voyage instead: pip install voyageai and return
#   voyageai.Client().embed([text], model="voyage-3.5").embeddings[0]
# ----------------------------------------------------------------------------

# def embed(text: str) -> np.ndarray:
#     resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "input": text}, timeout=60)
#     if not resp.ok:
#         # ukaz realnu Ollama hlasku, nie holy 404
#         raise RuntimeError(f"Ollama {resp.status_code}: {resp.text}")
#     vec = np.asarray(resp.json()["embeddings"][0], dtype=np.float32)
#     return vec / (np.linalg.norm(vec) + 1e-8)

def embed_batch(texts: list[str]) -> np.ndarray:
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"Ollama {resp.status_code}: {resp.text}")
    mat = np.asarray(resp.json()["embeddings"], dtype=np.float32)
    # normalizuj kazdy riadok na jednotkovu dlzku
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)

def embed(text: str) -> np.ndarray:
    return embed_batch([text])[0]        # query stale ide po jednom


# ----------------------------------------------------------------------------
# 3. BUILD / LOAD INDEX  -  embed every item once, cache to disk.
#
# The "vector store" is just: a (N x dim) float matrix + a parallel list of
# metadata. No database required. We persist so you don't re-embed 1400 items
# on every run -- rebuild only with --reindex.
# ----------------------------------------------------------------------------

# def build_index(corpus: list[dict]) -> np.ndarray:
#     print(f"Embedding {len(corpus)} items via Ollama/{EMBED_MODEL} ...")
#     vectors = []
#     for i, item in enumerate(corpus, 1):
#         vectors.append(embed(item["text"]))
#         if i % 50 == 0 or i == len(corpus):
#             print(f"  {i}/{len(corpus)}")
#     matrix = np.vstack(vectors)

#     INDEX_DIR.mkdir(exist_ok=True)
#     np.save(INDEX_DIR / "vectors.npy", matrix)
#     (INDEX_DIR / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False))
#     print(f"Index saved to {INDEX_DIR}")
#     return matrix

def build_index(corpus: list[dict]) -> np.ndarray:
    print(f"Embedding {len(corpus)} items via Ollama/{EMBED_MODEL} ...")
    B = 64
    parts = []
    for i in range(0, len(corpus), B):
        parts.append(embed_batch([it["text"] for it in corpus[i:i + B]]))
        print(f"  {min(i + B, len(corpus))}/{len(corpus)}")
    matrix = np.vstack(parts)

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", matrix)
    (INDEX_DIR / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False))
    print(f"Index saved to {INDEX_DIR}")
    return matrix


def load_index() -> tuple[np.ndarray, list[dict]]:
    vec_path = INDEX_DIR / "vectors.npy"
    cor_path = INDEX_DIR / "corpus.json"
    if not (vec_path.exists() and cor_path.exists()):
        sys.exit("No index found. Run once with --reindex first.")
    matrix = np.load(vec_path)
    corpus = json.loads(cor_path.read_text())
    return matrix, corpus


# ----------------------------------------------------------------------------
# 4. RETRIEVE  -  find the k items whose vectors point most like the question.
#
# matrix is (N x dim) and already unit-normalized. We normalize the query the
# same way, then similarities = matrix @ query is an (N,) vector of cosine
# scores. argsort picks the top-k. That's the entire "semantic search".
# ----------------------------------------------------------------------------

def retrieve(question: str, matrix: np.ndarray, corpus: list[dict], k: int = TOP_K) -> list[dict]:
    q = embed(question)
    sims = matrix @ q                       # cosine similarity, one dot product each
    top = np.argsort(-sims)[:k]             # indices of the k highest scores
    hits = []
    for idx in top:
        item = dict(corpus[int(idx)])
        item["score"] = float(sims[idx])
        hits.append(item)
    return hits


# ----------------------------------------------------------------------------
# 4b. REWRITE QUERY  -  make a follow-up self-contained before retrieving.
#
# The interactive chat keeps a conversation, but retrieve() only ever sees the
# words of the latest question. "explain that more simply" has no searchable
# content, so we first ask a cheap model to fold the conversation into ONE
# standalone query, then retrieve on that. This is the retriever's half of
# "memory". One-shot -q has no history, so it never calls this.
# ----------------------------------------------------------------------------

def rewrite_query(question: str, history: list[dict] | None) -> str:
    if not history:                       # nothing to fold in -> already standalone
        return question
    import anthropic

    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    prompt = (
        "Given this conversation, rewrite the user's follow-up into a single "
        "standalone search query that stands on its own, with no pronouns like "
        "'that' or 'this' referring back. Output ONLY the query, nothing else.\n\n"
        f"Conversation so far:\n{transcript}\n\n"
        f"Follow-up: {question}"
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ----------------------------------------------------------------------------
# 5. GENERATE  -  hand the retrieved abstracts to Claude and ask the question.
#
# The prompt does two honest things: (a) label each source [1], [2]... so the
# answer can cite, and (b) instruct Claude to say when the library doesn't
# contain the answer instead of inventing one. That grounding instruction is
# what separates RAG from "just ask the model".
# ----------------------------------------------------------------------------

def answer(question: str, hits: list[dict], history: list[dict] | None = None) -> str:
    import anthropic

    context_blocks = []
    for n, h in enumerate(hits, 1):
        tag = " ".join(x for x in [h.get("authors"), h.get("year")] if x)
        context_blocks.append(f"[{n}] {h['title']} ({tag})\n{h['abstract']}")
    context = "\n\n".join(context_blocks)

    system = (
        "You answer questions using ONLY the provided paper abstracts from the "
        "user's Zotero library. Cite sources inline as [1], [2], etc. If the "
        "abstracts do not contain the answer, say so plainly -- do not use outside "
        "knowledge or guess."
    )
    user = f"Abstracts:\n\n{context}\n\n---\n\nQuestion: {question}"

    # In interactive chat `history` carries the earlier turns as CLEAN Q&A text
    # (not their abstracts -- so the prompt doesn't balloon by 6 abstracts each
    # turn). One-shot -q passes nothing, and this stays a single-message request,
    # exactly as before.
    messages = (history or []) + [{"role": "user", "content": user}]

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    # content is a list of typed blocks (v1 API); join the text ones.
    return "".join(b.text for b in resp.content if b.type == "text")


# ----------------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------------

def run_query(question: str, matrix: np.ndarray, corpus: list[dict],
              history: list[dict] | None = None) -> None:
    # history is None -> one-shot (stateless), same as before. A list -> chat:
    # rewrite the question for retrieval, answer with the conversation, then
    # remember this turn (mutating the caller's `history`).
    conversational = history is not None

    search_q = rewrite_query(question, history) if conversational else question
    if search_q != question:
        print(f"  [rewritten for search] {search_q!r}")

    hits = retrieve(search_q, matrix, corpus)
    print("\nRetrieved:")
    for n, h in enumerate(hits, 1):
        print(f"  [{n}] ({h['score']:.3f}) {h['title'][:80]}")
    print("\nAnswer:\n")
    text = answer(question, hits, history if conversational else None)
    print(text)
    print()

    if conversational:
        # Store the CLEAN turn (no abstracts) -- keeps the running prompt small.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": text})


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw RAG over a Zotero library.")
    ap.add_argument("--reindex", action="store_true", help="(re)build the vector index")
    ap.add_argument("-q", "--query", help="ask one question and exit")
    args = ap.parse_args()

    if args.reindex:
        matrix = build_index(load_corpus(ZOTERO_DB))
        corpus = json.loads((INDEX_DIR / "corpus.json").read_text())
    else:
        matrix, corpus = load_index()

    if args.query:
        run_query(args.query, matrix, corpus)   # one-shot: no memory
        return

    print("Ask about your library. It remembers the conversation, so follow-ups")
    print("work. Commands:  /reset  new conversation   |   empty line or Ctrl-D  quit")
    history: list[dict] = []
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        if q in ("/reset", "/new"):
            history.clear()
            print("  [conversation reset -- starting fresh]")
            continue
        run_query(q, matrix, corpus, history)


if __name__ == "__main__":
    main()