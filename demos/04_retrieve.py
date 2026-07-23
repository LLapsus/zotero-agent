#!/usr/bin/env python3
"""
04_retrieve.py  -  How does a question turn into "the 6 most relevant papers"?

Stage 4 of the pipeline: the search itself. `retrieve()` is only five lines, and
this demo takes them apart one at a time:

    q    = embed(question)          # your question -> a (dim,) vector
    sims = matrix @ q               # (N x dim) @ (dim,) = (N,) cosine scores
    top  = np.argsort(-sims)[:k]    # indices of the k highest scores
    ...  attach the score to a copy of each corpus item and return

Things worth seeing here:
    - the `-sims` trick (numpy sorts ascending, so negate to get descending)
    - what the returned dicts look like (stage-1 item + a 'score' key)
    - how the top-k scores compare to the rest of the library, i.e. whether a
      hit is genuinely close or merely "least far away"
    - what happens on a question your library simply can't answer

Run it:
    python demos/04_retrieve.py

Free and local: one Ollama call for the query, then pure numpy. No API cost.
"""

from __future__ import annotations

import time

import numpy as np

from _common import zotero_rag, rule

QUESTION = "How do phase transitions show up in machine learning models?"
OFF_TOPIC = "What is the best recipe for Neapolitan pizza dough?"


def main() -> None:
    rule("STAGE 4 - retrieve()")

    try:
        matrix, corpus = zotero_rag.load_index()
    except SystemExit:
        print("No index found. Build it first:  python zotero_rag.py --reindex")
        return

    print(f"Index: {matrix.shape[0]} items x {matrix.shape[1]} dims")
    print(f"TOP_K (from config): {zotero_rag.TOP_K}")
    print(f"\nQuestion: {QUESTION!r}")

    # ---- 1. the whole function, as the pipeline calls it -------------------
    t0 = time.perf_counter()
    hits = zotero_rag.retrieve(QUESTION, matrix, corpus)
    dt = time.perf_counter() - t0

    rule("What retrieve() returns")
    print(f"  {len(hits)} hits in {dt * 1000:.0f} ms "
          "(most of it is embedding the query; the search itself is instant)")
    print(f"  each hit is a dict with keys: {sorted(hits[0].keys())}")
    print("  i.e. the stage-1 item, plus a 'score' key added by retrieve()\n")
    for n, h in enumerate(hits, 1):
        tag = " ".join(x for x in [h.get("authors"), h.get("year")] if x)
        print(f"  [{n}] {h['score']:.3f}  {h['title'][:62]}")
        print(f"          {tag or '(no metadata)'}")

    # ---- 2. the same thing, unrolled step by step -------------------------
    rule("The same search, one line at a time")
    q = zotero_rag.embed(QUESTION)
    print(f"  1. q    = embed(question)      -> shape {q.shape}")
    sims = matrix @ q
    print(f"  2. sims = matrix @ q           -> shape {sims.shape}  "
          "(one score per item)")
    order = np.argsort(-sims)
    print(f"  3. np.argsort(-sims)           -> item indices, best first")
    print(f"        note the MINUS: numpy sorts ascending, so negating the")
    print(f"        scores turns it into 'largest first'.")
    print(f"        first 6 indices: {order[:6].tolist()}")
    top = order[:zotero_rag.TOP_K]
    print(f"  4. [:k]                        -> keep {zotero_rag.TOP_K}: "
          f"{top.tolist()}")
    print(f"\n  Cross-check vs retrieve(): "
          f"{[corpus[int(i)]['key'] for i in top] == [h['key'] for h in hits]}")

    # ---- 3. are the hits actually CLOSE, or just least-far? ---------------
    # A cosine score means nothing in isolation. Compare the top-k against the
    # distribution over the whole library to see if they really stand out.
    rule("Is the top-k genuinely close? (scores vs the whole library)")
    print(f"  whole library : min={sims.min():.3f}  median={np.median(sims):.3f}  "
          f"max={sims.max():.3f}")
    print(f"  the {zotero_rag.TOP_K} hits  : "
          + ", ".join(f"{s:.3f}" for s in sims[top]))
    gap = sims[top].mean() - np.median(sims)
    print(f"  top-k mean is {gap:.3f} above the median -> "
          f"{'a clear signal' if gap > 0.1 else 'a weak signal'}")
    print("\n  Rule of thumb: retrieval ALWAYS returns k items, even when none")
    print("  are relevant. There is no threshold in this pipeline -- grounding")
    print("  is left to the prompt in stage 5, which tells Claude to say so.")

    # ---- 4. a question the library cannot answer --------------------------
    rule("An off-topic question still returns k hits")
    print(f"  Question: {OFF_TOPIC!r}\n")
    off = zotero_rag.retrieve(OFF_TOPIC, matrix, corpus)
    for n, h in enumerate(off, 1):
        print(f"  [{n}] {h['score']:.3f}  {h['title'][:62]}")
    print("\n  Scores are lower and the titles are unrelated -- but they still")
    print("  come back. This is exactly why stage 5's system prompt insists on")
    print("  'if the abstracts don't answer it, say so'.")

    # ---- 5. k is just a slice ---------------------------------------------
    rule("k is nothing but how much of the sorted list you keep")
    for k in (1, 3, 6, 12):
        got = zotero_rag.retrieve(QUESTION, matrix, corpus, k=k)
        print(f"  k={k:<3} -> {len(got)} hits, "
              f"scores {np.round([g['score'] for g in got], 3).tolist()}")
    print("\n  Bigger k = more context for Claude = more tokens = more cost,")
    print("  and more chance of dragging in irrelevant abstracts.")

    rule("Next stage")
    print("Stage 5 (answer) formats these hits into a prompt and asks Claude.")
    print("See: demos/05_generate.py   (this one DOES cost money)")


if __name__ == "__main__":
    main()
