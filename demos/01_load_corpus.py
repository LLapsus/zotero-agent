#!/usr/bin/env python3
"""
01_load_corpus.py  -  What does `load_corpus()` actually give you?

Stage 1 of the pipeline. This demo does NOT re-implement anything: it imports
`load_corpus` straight from zotero_rag.py, runs it against your real Zotero DB,
and then shows you the shape of what comes back:

    - how many items survived the SQL filter (must have a title AND an abstract)
    - the exact dict layout of a single item (the keys the rest of the pipeline
      relies on)
    - one full item, pretty-printed
    - the "text" field verbatim -- this is the string that stage 2 will embed
    - a couple of aggregate stats, so the corpus feels concrete, not abstract

Run it:
    python demos/01_load_corpus.py

Nothing here costs money or calls a network service. It only reads a temp copy
of ~/Zotero/zotero.sqlite. If the DB isn't where zotero_rag expects it, the
underlying function exits with a clear message pointing at ZOTERO_DB.
"""

from __future__ import annotations

import time
from collections import Counter

from _common import zotero_rag, rule

# The config values live in zotero_rag.py -- we read them, we don't redefine them.
DB_PATH = zotero_rag.ZOTERO_DB


def main() -> None:
    rule("STAGE 1 - load_corpus()")
    print(f"Reading from: {DB_PATH}")
    print("(zotero_rag copies the DB to a temp file first, so the live Zotero")
    print(" library is never locked or touched.)")

    t0 = time.perf_counter()
    corpus = zotero_rag.load_corpus(DB_PATH)
    dt = time.perf_counter() - t0

    print(f"\nLoaded {len(corpus)} items in {dt:.2f}s.")
    if not corpus:
        print("Corpus is empty -- do you have items with abstracts in Zotero?")
        return

    # ---- 1. the SHAPE of one item ------------------------------------------
    # Every item is a plain dict. These keys are the contract the rest of the
    # pipeline depends on: 'text' gets embedded; 'title'/'authors'/'year' get
    # shown to Claude as a citation label; 'abstract' is the body Claude reads.
    rule("The dict layout (keys of one item)")
    sample = corpus[0]
    for k, v in sample.items():
        preview = str(v).replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:60] + " ..."
        print(f"  {k:10} ({type(v).__name__:4}) -> {preview}")

    # ---- 2. one FULL item, readable ----------------------------------------
    rule("One full item, expanded")
    _print_item(sample)

    # ---- 3. the exact string that gets embedded ----------------------------
    # This is the single most important thing to see: stage 2 embeds `text`
    # exactly as-is (title + blank line + abstract). One item == one chunk.
    rule("item['text'] -- the exact string stage 2 will embed")
    print(repr(sample["text"][:400]) + (" ..." if len(sample["text"]) > 400 else ""))

    # ---- 4. a few aggregate stats ------------------------------------------
    rule("A feel for the whole corpus")
    with_authors = sum(1 for it in corpus if it["authors"])
    lengths = [len(it["abstract"]) for it in corpus]
    years = Counter(it["year"] for it in corpus if it["year"])

    print(f"  items total ............. {len(corpus)}")
    print(f"  with an author name ..... {with_authors}")
    print(f"  abstract length (chars) . min={min(lengths)}  "
          f"avg={sum(lengths) // len(lengths)}  max={max(lengths)}")
    print("  most common years ....... "
          + ", ".join(f"{y}:{n}" for y, n in years.most_common(5)))

    rule("Next stage")
    print("Stage 2 (embed) turns each item['text'] into a vector.")
    print("See: demos/02_embed.py")


def _print_item(item: dict) -> None:
    print(f"  key      : {item['key']}")
    print(f"  title    : {item['title']}")
    print(f"  authors  : {item['authors'] or '(none listed)'}")
    print(f"  year     : {item['year'] or '(no date)'}")
    print(f"  abstract : {_wrap(item['abstract'])}")


def _wrap(text: str, width: int = 70, indent: str = " " * 13) -> str:
    """Cheap word-wrap so long abstracts don't blow past the terminal."""
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    out.append(line)
    return ("\n" + indent).join(out)


if __name__ == "__main__":
    main()
