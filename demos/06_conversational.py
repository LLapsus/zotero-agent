#!/usr/bin/env python3
"""
06_conversational.py  -  Turning the one-shot pipeline into a real conversation.

zotero_rag.py answers each question from a blank slate: run_query() retrieves
fresh and sends Claude a single message with no memory of what came before. So a
follow-up like "explain that in simpler terms" fails twice over -- the retriever
embeds only those five words (semantic noise), and Claude never saw the previous
answer. This demo fixes BOTH, from scratch, reusing zotero_rag's functions.

The two independent "amnesias", and the fix for each:

    LAYER A  (the LLM forgets)     ->  accumulate a `history` list of messages
                                       and send it every turn, so Claude sees
                                       the whole conversation so far.

    LAYER B  (the retriever forgets) ->  before searching, ask a cheap model to
                                         rewrite the follow-up into a STANDALONE
                                         query using the history ("query
                                         rewriting"), then retrieve on that.

Each turn prints the rewritten query (so you can watch LAYER B work) and the
retrieved titles (so you can see the context change), then the answer.

Run it:
    python demos/06_conversational.py             # interactive, remembers turns
    python demos/06_conversational.py --scripted  # a fixed 3-turn demo

COSTS MONEY. Unlike demos 1-4 this always calls the API -- a conversation is
inherently interactive. Each turn is one answer call, plus (from turn 2 on) one
cheap rewrite call. Ballpark a few cents for the scripted run.

Nothing here modifies zotero_rag.py: we import its retrieve()/load_index() and
add the conversation layer on top.
"""

from __future__ import annotations

import argparse

from _common import zotero_rag, rule

import ui  # cosmetic ANSI styling for the interactive prompt

# Same grounding instruction zotero_rag.answer() uses -- kept identical so the
# only thing this demo changes is the *memory*, not the answering behaviour.
SYSTEM = (
    "You answer questions using ONLY the provided paper abstracts from the "
    "user's Zotero library. Cite sources inline as [1], [2], etc. If the "
    "abstracts do not contain the answer, say so plainly -- do not use outside "
    "knowledge or guess."
)

# The rewrite in LAYER B is a small, easy task, so we give it a cheap, fast
# model instead of the main one. Using the right-sized model for each sub-task
# is a habit worth forming early.
REWRITE_MODEL = "claude-haiku-4-5-20251001"

# A canned conversation for --scripted. Turn 1 is self-contained; turns 2 and 3
# only make sense *because* of the turns before them -- exactly what a stateless
# pipeline cannot handle.
SCRIPT = [
    "How do phase transitions show up in machine learning models?",
    "Can you explain that first idea in simpler terms?",
    "And how does any of this relate to generalization?",
]


def rewrite_standalone(client, question: str, history: list[dict]) -> str:
    """LAYER B: fold the conversation into one self-contained search query.

    With no history there is nothing to fold in, so we skip the call and save
    the money -- the question is already standalone.
    """
    if not history:
        return question

    # Flatten the prior turns into plain text for the rewriter to read.
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    prompt = (
        "Given this conversation, rewrite the user's follow-up into a single "
        "standalone search query that would make sense on its own, with no "
        "pronouns like 'that' or 'this' referring back. Output ONLY the query, "
        "nothing else.\n\n"
        f"Conversation so far:\n{transcript}\n\n"
        f"Follow-up: {question}"
    )
    resp = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def format_context(hits: list[dict]) -> str:
    """The same [1] [2] labelling zotero_rag.answer() builds -- unchanged."""
    blocks = []
    for n, h in enumerate(hits, 1):
        tag = " ".join(x for x in [h.get("authors"), h.get("year")] if x)
        blocks.append(f"[{n}] {h['title']} ({tag})\n{h['abstract']}")
    return "\n\n".join(blocks)


def answer_with_history(client, question: str, hits: list[dict],
                        history: list[dict]) -> str:
    """LAYER A: send the whole conversation, not just this question.

    Note the token-budget trick: `history` holds only the *clean* Q&A text, not
    the giant retrieved abstracts. The abstracts go only into the current turn's
    message. So history grows with the conversation, not with 6 abstracts per
    turn. Claude still remembers what it said; it just doesn't re-read old
    context it no longer needs.
    """
    # Format the retrieved abstracts into the same [1] [2] style zotero_rag.answer()
    context = format_context(hits)
    current = f"Abstracts:\n\n{context}\n\n---\n\nQuestion: {question}"

    # Build the message list for this turn: the prior conversation plus the
    # current turn's abstracts and question. The prior conversation is just the
    # clean Q&A text, not the abstracts, so it doesn't balloon with every turn.
    messages = history + [{"role": "user", "content": current}]
    resp = client.messages.create(
        model=zotero_rag.CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def one_turn(client, question: str, matrix, corpus, history: list[dict]) -> None:
    """Run a single conversational turn, printing both layers as they happen."""
    # ---- LAYER B: what are we actually going to search for? ----------------
    # rewrite_standalone() is cheap, so we do it every turn except the first.
    standalone = rewrite_standalone(client, question, history)
    if standalone != question:
        print(f"  [rewritten for search] {standalone!r}")

    # retrieve() is expensive, so we do it only once per turn. The rewritten
    # query is what the retriever sees, so it can find relevant abstracts even
    # if the user asked a follow-up like "explain that in simpler terms."
    hits = zotero_rag.retrieve(standalone, matrix, corpus)
    print("  retrieved:")
    for n, h in enumerate(hits, 1):
        print(f"    [{n}] ({h['score']:.3f}) {h['title'][:66]}")

    # ---- LAYER A: answer with the whole conversation in context ------------
    text = answer_with_history(client, question, hits, history)
    print(f"\n{text}\n")

    # Record ONLY the clean turn (no abstracts) -- see answer_with_history().
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": text})
    print(f"  [history now holds {len(history) // 2} turn(s)]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Conversational RAG with memory.")
    ap.add_argument("--scripted", action="store_true",
                    help="run a fixed 3-turn conversation instead of prompting")
    args = ap.parse_args()

    rule("STAGE 6 - conversation with memory (COSTS MONEY)")
    print("Every question below is one paid answer call, plus a cheap rewrite")
    print("call from the second turn onward. Ctrl-C to stop.")

    try:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (zotero_rag loaded .env)
    except Exception as e:  # noqa: BLE001 - demo: surface the real problem
        print(f"\nCould not create the Anthropic client: {e}")
        print("(Is ANTHROPIC_API_KEY set? zotero_rag loads .env automatically.)")
        return

    matrix, corpus = zotero_rag.load_index()
    history: list[dict] = []  # LAYER A lives here: it persists across turns

    if args.scripted:
        for i, q in enumerate(SCRIPT, 1):
            rule(f"Turn {i}:  {q}")
            one_turn(client, q, matrix, corpus, history)

        rule("What to notice")
        print("  - Turn 1 had no history, so no rewrite happened: the question")
        print("    went to the retriever as-is.")
        print("  - Turns 2-3 were vague on their own ('that first idea', 'any")
        print("    of this'). The rewrite turned each into a standalone query --")
        print("    that is LAYER B making retrieval follow the thread.")
        print("  - The answers refer back to earlier ones because `history` was")
        print("    sent every turn -- that is LAYER A.")
        print("\n  Compare: zotero_rag.py rebuilds `messages` from scratch each")
        print("  call, so it can do neither. This file only adds the memory;")
        print("  retrieve() and the prompt are unchanged.")
        return

    print("\nAsk about your library. Commands:  /exit (or Ctrl-D)  quit")
    while True:
        try:
            q = ui.ask("\n> ").strip()
        except EOFError:
            break
        if q in ("/exit", "/quit", "exit", "quit"):
            break
        if not q:
            continue
        one_turn(client, q, matrix, corpus, history)


if __name__ == "__main__":
    main()
