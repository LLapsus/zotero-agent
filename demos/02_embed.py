#!/usr/bin/env python3
"""
02_embed.py  -  What does `embed()` actually produce?

Stage 2 of the pipeline. Again we import the real functions from zotero_rag.py
and just watch what they return. An embedding is the whole "semantic" trick of
RAG, so it's worth seeing one concretely:

    - embed() takes a string and returns ONE vector (a 1-D numpy array)
    - that vector has a fixed length (the model's "dimension"), no matter how
      long the input text is
    - zotero_rag normalizes every vector to unit length, so cosine similarity
      later collapses into a single dot product
    - similar texts get vectors that point in similar directions; unrelated
      texts point elsewhere -- we measure that here so it isn't just a claim

Run it:
    python demos/02_embed.py

Needs Ollama running locally with the embedding model:
    ollama pull nomic-embed-text
This is FREE and LOCAL -- nothing leaves your machine, no API key involved.
"""

from __future__ import annotations

import time

import numpy as np

from _common import zotero_rag, rule


def main() -> None:
    rule("STAGE 2 - embed()")
    print(f"Embedding via Ollama model: {zotero_rag.EMBED_MODEL}")
    print(f"Ollama endpoint:            {zotero_rag.OLLAMA_URL}")

    # ---- 1. one string -> one vector ---------------------------------------
    text = "Variational autoencoders and the topology of latent space."
    print(f"\nInput text:\n  {text!r}")

    try:
        t0 = time.perf_counter()
        vec = zotero_rag.embed(text)
        dt = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - demo: show the real error, keep going
        print(f"\nCould not reach Ollama: {e}")
        print("Start it and pull the model:  ollama pull nomic-embed-text")
        return

    rule("The vector")
    print(f"  type            : {type(vec).__name__}")
    print(f"  dtype           : {vec.dtype}")
    print(f"  shape           : {vec.shape}   <- this is the 'dimension'")
    print(f"  first 8 numbers : {np.round(vec[:8], 4)}")
    print(f"  L2 norm (length): {np.linalg.norm(vec):.6f}   <- ~1.0 on purpose")
    print(f"  took            : {dt * 1000:.0f} ms")

    # ---- 2. the dimension is fixed, the text length is not -----------------
    # Feed a 3-word string and a long paragraph; both come back the same size.
    rule("Dimension is fixed, regardless of input length")
    short = zotero_rag.embed("cat")
    long = zotero_rag.embed("cat " * 200)
    print(f"  embed('cat')        -> shape {short.shape}")
    print(f"  embed('cat'*200)    -> shape {long.shape}")
    print("  Same length. The model compresses any text into a fixed vector.")

    # ---- 3. similar meaning -> higher cosine similarity --------------------
    # Because vectors are unit-normalized, cosine similarity == dot product.
    # Higher = more alike in "meaning" (as this model sees it).
    rule("Meaning shows up as similarity (dot product of unit vectors)")
    anchor = "deep generative models for images"
    candidates = [
        "generative adversarial networks synthesize photos",  # related
        "variational autoencoders learn latent representations",  # related
        "a recipe for sourdough bread",  # unrelated
        "quarterly tax filing deadlines",  # unrelated
    ]
    a = zotero_rag.embed(anchor)
    print(f"  anchor: {anchor!r}\n")
    scored = [(float(a @ zotero_rag.embed(c)), c) for c in candidates]
    for score, c in sorted(scored, reverse=True):
        bar = "#" * int(round(score * 40))
        print(f"  {score:.3f} {bar:<40} {c}")
    print("\n  The two ML sentences score highest; bread and taxes fall away.")
    print("  That gap is the entire basis of retrieval in stage 4.")

    # ---- 4. embed() is just embed_batch() for one string -------------------
    rule("embed() vs embed_batch()")
    pair = zotero_rag.embed_batch(["first text", "second text"])
    print(f"  embed_batch([a, b]) -> shape {pair.shape}  (N rows, one per text)")
    print(f"  embed(a)            -> shape {zotero_rag.embed('first text').shape}"
          "  (just row 0 of a batch of 1)")
    print("  Indexing uses batches (fast); a single query uses embed().")

    rule("Next stage")
    print("Stage 3 (build_index) embeds the WHOLE corpus and caches it to disk.")
    print("See: demos/03_build_index.py")


if __name__ == "__main__":
    main()
