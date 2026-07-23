#!/usr/bin/env python3
"""
05_generate.py  -  How do retrieved abstracts become a grounded answer?

Stage 5, the last one: `answer()` takes the top-k hits from stage 4, formats
them into one prompt, and asks Claude. There is no magic here either -- the
"RAG" part is entirely string formatting plus one instruction in the system
prompt.

What this demo shows:
    - the exact context block built from the hits ([1], [2], ... labels)
    - the system prompt, and which sentence in it is what makes this RAG
      rather than "just ask the model"
    - the real token count of the request, via the Messages API
    - an estimated cost, before you spend anything
    - (optionally) the actual call, the answer with its [n] citations, and the
      token usage that came back

Run it:
    python demos/05_generate.py          # FREE: prompt + token count only
    python demos/05_generate.py --run    # COSTS MONEY: also calls Claude

Needs ANTHROPIC_API_KEY (zotero_rag loads .env for you). Token counting is
free; only the final --run call is billed.
"""

from __future__ import annotations

import argparse

from _common import zotero_rag, rule

QUESTION = "How do phase transitions show up in machine learning models?"

# Published rates for the model configured in zotero_rag.py, USD per 1M tokens.
# Claude Sonnet 5 lists at $3 / $15; an introductory $2 / $10 runs through
# 2026-08-31. We quote the introductory rate and say so.
PRICE_IN_PER_MTOK = 2.00
PRICE_OUT_PER_MTOK = 10.00
PRICE_NOTE = "Claude Sonnet 5 introductory rate ($2/$10 per 1M tok, through 2026-08-31)"

# Mirrors the system prompt inside zotero_rag.answer() so we can display it.
# (answer() builds its own copy; this demo does not change what it sends.)
SYSTEM = (
    "You answer questions using ONLY the provided paper abstracts from the "
    "user's Zotero library. Cite sources inline as [1], [2], etc. If the "
    "abstracts do not contain the answer, say so plainly -- do not use outside "
    "knowledge or guess."
)


def build_user_message(question: str, hits: list[dict]) -> str:
    """Same formatting answer() does -- reproduced here so we can inspect it."""
    blocks = []
    for n, h in enumerate(hits, 1):
        tag = " ".join(x for x in [h.get("authors"), h.get("year")] if x)
        blocks.append(f"[{n}] {h['title']} ({tag})\n{h['abstract']}")
    context = "\n\n".join(blocks)
    return f"Abstracts:\n\n{context}\n\n---\n\nQuestion: {question}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo of the generate stage.")
    ap.add_argument("--run", action="store_true",
                    help="actually call the Anthropic API (this costs money)")
    args = ap.parse_args()

    rule("STAGE 5 - answer()")
    print(f"Model: {zotero_rag.CLAUDE_MODEL}")
    print(f"Question: {QUESTION!r}")

    matrix, corpus = zotero_rag.load_index()
    hits = zotero_rag.retrieve(QUESTION, matrix, corpus)
    print(f"\nRetrieved {len(hits)} hits from stage 4:")
    for n, h in enumerate(hits, 1):
        print(f"  [{n}] ({h['score']:.3f}) {h['title'][:64]}")

    # ---- 1. the system prompt: where the "grounding" lives -----------------
    rule("The system prompt")
    print(SYSTEM)
    print("\n  The last sentence is the whole difference between RAG and just")
    print("  asking Claude. Retrieval put the abstracts in front of the model;")
    print("  THIS is what stops it answering from its own training data when")
    print("  the abstracts don't cover the question.")

    # ---- 2. the user message: labelled context + the question --------------
    user = build_user_message(QUESTION, hits)
    rule("The user message (first 900 chars)")
    print(user[:900] + ("\n  ... [truncated for display]" if len(user) > 900 else ""))
    print(f"\n  full length: {len(user)} characters")
    print("  Note the [1] [2] ... labels: that is what lets Claude cite, and")
    print("  what lets YOU trace a claim back to a specific paper.")

    # ---- 3. how big is this request, really? ------------------------------
    # count_tokens is a free endpoint -- always cheaper than guessing.
    rule("Token count (free API call, nothing generated)")
    try:
        import anthropic
        client = anthropic.Anthropic()
        counted = client.messages.count_tokens(
            model=zotero_rag.CLAUDE_MODEL,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        n_in = counted.input_tokens
        print(f"  input tokens: {n_in}")
        est_in = n_in / 1e6 * PRICE_IN_PER_MTOK
        est_out = 400 / 1e6 * PRICE_OUT_PER_MTOK  # assume a ~400-token answer
        print(f"  estimated cost: ${est_in:.5f} in + ~${est_out:.5f} out "
              f"= ~${est_in + est_out:.5f}")
        print(f"  ({PRICE_NOTE})")
        print("\n  This is why TOP_K matters: every extra abstract is more")
        print("  input tokens on every single question you ask.")
    except Exception as e:  # noqa: BLE001 - demo: surface the real problem
        print(f"  Could not count tokens: {e}")
        print("  (Is ANTHROPIC_API_KEY set? zotero_rag loads .env automatically.)")
        return

    # ---- 4. the actual call, only if asked --------------------------------
    if not args.run:
        rule("Not calling the API")
        print("  Everything above was free. To actually ask Claude:")
        print("      python demos/05_generate.py --run")
        return

    rule("Calling Claude ...")
    text = zotero_rag.answer(QUESTION, hits)
    print(text)

    rule("What just happened")
    print("  1. retrieve() picked 6 abstracts by cosine similarity")
    print("  2. answer() pasted them into one prompt with [n] labels")
    print("  3. the system prompt told Claude to use ONLY those abstracts")
    print("  4. Claude's reply cites [n] -> you can check every claim")
    print("\n  Look for [1]/[2]/... in the answer above, then match them")
    print("  against the retrieved list at the top of this output.")

    rule("That's the whole pipeline")
    print("  SQLite -> corpus -> vectors -> cosine search -> prompt -> answer.")
    print("  Next: the same thing written with LangChain, for comparison.")


if __name__ == "__main__":
    main()
