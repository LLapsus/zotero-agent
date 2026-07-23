#!/usr/bin/env python3
"""
03_build_index.py  -  What is the "vector store", concretely?

Stage 3 of the pipeline. `build_index()` embeds the WHOLE corpus once and caches
the result to disk, so you never re-embed 1314 items on every run. The "vector
store" everyone talks about is, here, just two plain files:

    ~/.zotero_rag/vectors.npy   an (N x dim) float32 matrix  (row i = item i)
    ~/.zotero_rag/corpus.json   the list of item dicts       (item i)

No database, no server. Row i of the matrix is the embedding of corpus[i]. That
1:1 alignment is the only thing that makes retrieval work later.

This demo is deliberately READ-ONLY. It inspects your EXISTING index rather than
rebuilding it, because a real rebuild would overwrite ~/.zotero_rag and re-embed
everything. To see the batching logic, it embeds a tiny slice in memory only.

Run it:
    python demos/03_build_index.py

To actually (re)build the index, that's the pipeline's own command, not a demo:
    python zotero_rag.py --reindex
"""

from __future__ import annotations

import json

import numpy as np

from _common import zotero_rag, rule

INDEX_DIR = zotero_rag.INDEX_DIR
VEC_PATH = INDEX_DIR / "vectors.npy"
COR_PATH = INDEX_DIR / "corpus.json"


def main() -> None:
    rule("STAGE 3 - build_index()  (inspected, not rebuilt)")
    print(f"Index directory: {INDEX_DIR}")

    if not (VEC_PATH.exists() and COR_PATH.exists()):
        print("\nNo index on disk yet. Build it once with:")
        print("    python zotero_rag.py --reindex")
        print("then re-run this demo.")
        return

    # ---- 1. the two files that ARE the vector store ------------------------
    rule("The vector store is two files on disk")
    print(f"  vectors.npy : {VEC_PATH.stat().st_size / 1e6:6.2f} MB")
    print(f"  corpus.json : {COR_PATH.stat().st_size / 1e6:6.2f} MB")

    matrix, corpus = zotero_rag.load_index()
    print(f"\n  matrix : shape {matrix.shape}, dtype {matrix.dtype}")
    print(f"  corpus : {len(corpus)} items (a list of the dicts from stage 1)")
    print(f"\n  N rows in matrix == N items in corpus ?  "
          f"{matrix.shape[0] == len(corpus)}")
    print("  That alignment (row i <-> corpus[i]) is the whole 'index'.")

    # ---- 2. rows are already unit vectors ----------------------------------
    # build_index stored what embed_batch returned, and that was normalized.
    # So no normalization is needed at query time -- a dot product is cosine.
    rule("Every stored row is already unit length")
    norms = np.linalg.norm(matrix, axis=1)
    print(f"  norm  min={norms.min():.4f}  mean={norms.mean():.4f}  "
          f"max={norms.max():.4f}   (all ~1.0)")
    print(f"  memory footprint in RAM: {matrix.nbytes / 1e6:.1f} MB "
          f"({matrix.shape[0]} x {matrix.shape[1]} x 4 bytes)")

    # ---- 3. prove row i really is the embedding of corpus[i] ---------------
    # Re-embed one item's text and compare to its stored row. Same model, same
    # text -> the vectors should be (near) identical, i.e. self-similarity ~1.
    rule("Row i is the embedding of corpus[i] (spot check)")
    i = 0
    stored = matrix[i]
    fresh = zotero_rag.embed(corpus[i]["text"])
    print(f"  item[{i}]: {corpus[i]['title'][:64]}")
    print(f"  cosine(stored row, freshly re-embedded text) = {float(stored @ fresh):.4f}")
    print("  ~1.0 confirms the stored row corresponds to this exact item.")

    # ---- 4. how build_index fills that matrix: batches ---------------------
    # The real function loops in batches of B=64 and vstacks the pieces. We
    # reproduce ONLY the batching on the first few items, in memory, no disk.
    rule("How the matrix gets built: embed_batch, then vstack")
    demo_texts = [it["text"] for it in corpus[:5]]
    part = zotero_rag.embed_batch(demo_texts)
    print(f"  embed_batch(first 5 texts) -> {part.shape}  (5 rows at once)")
    print("  build_index() does this in chunks of 64 across the whole corpus,")
    print("  then np.vstack's the chunks into the final (N x dim) matrix.")
    print("  (Batching is why indexing 1314 items is a handful of HTTP calls,")
    print("   not 1314 of them.)")

    rule("Next stage")
    print("Stage 4 (retrieve) uses this matrix to find the top-k items for a query.")
    print("See: demos/04_retrieve.py")


if __name__ == "__main__":
    main()
