#!/usr/bin/env python3
"""
run_routing_eval.py  -  Does the agent fill the RIGHT tool arguments?

This is the cheapest, most deterministic layer of evaluation. It does NOT run
retrieval, the DB, Ollama, or answer generation. For each question it asks one
thing: given your real tool schema + system prompt, which of {author, topic}
does Claude fill in? Then it compares that against a labelled expectation.

Why a separate API call instead of running the agent?
  - We're testing the ROUTING decision, nothing downstream. One call, tiny
    max_tokens, stop at the tool call. Cheap (~$0.001/case) and fast.
  - It isolates the thing under test: no retrieval noise, no generation noise.

Why import from your agent instead of copying the prompt/schema?
  - So the eval tests the REAL config. A copied prompt drifts the moment you
    tweak the agent, and then your "passing" eval is testing a ghost.

Usage:
  python run_routing_eval.py                       # run once
  python run_routing_eval.py --repeat 3            # stability check (the LLM
                                                   #   isn't deterministic)
  python run_routing_eval.py --cases my_cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Pull the REAL tool schema + system prompt + model straight from the agent, so
# this eval tests the ACTUAL config -- a copy would silently drift the moment you
# tweak the agent. zotero_agent.py lives one dir up, so put the repo root on
# sys.path first.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zotero_agent import TOOLS, SYSTEM, MODEL   # noqa: E402  the real config under test

TOOL_NAME = "search_library"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The one piece with real logic -> kept pure so it's unit-testable without any
# network. Given what Claude actually filled and what we expected, pass or fail.
# ---------------------------------------------------------------------------
def _present(value) -> bool:
    """A field 'counts' only if it's there AND non-empty."""
    return bool(value and str(value).strip())


def grade(actual: dict | None, expect: dict) -> tuple[bool, str]:
    """
    actual: the tool input Claude produced, e.g. {"author": "Novak", "topic": "VAE"},
            or None if it never called the tool.
    expect: {"author": bool, "topic": bool, optional "author_contains": str}
    Returns (passed, reason_if_failed).
    """
    if actual is None:
        return False, "no tool call"

    got_author = _present(actual.get("author"))
    got_topic = _present(actual.get("topic"))
    want_author = bool(expect.get("author"))
    want_topic = bool(expect.get("topic"))

    if got_author != want_author:
        return False, f"author: wanted {want_author}, got {got_author}"
    if got_topic != want_topic:
        return False, f"topic: wanted {want_topic}, got {got_topic}"

    # Optional: catch the classic failure where the name lands in `topic`.
    needle = expect.get("author_contains")
    if needle and needle.lower() not in str(actual.get("author", "")).lower():
        return False, f"author should contain '{needle}', got {actual.get('author')!r}"

    return True, ""


# ---------------------------------------------------------------------------
# The part that talks to the API. One call per question, forced to be as
# deterministic as we can make it (temperature=0), stopped at the tool call.
# ---------------------------------------------------------------------------
def route_once(client, question: str) -> dict | None:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,                     # room for adaptive thinking + the tool call
        # No temperature: claude-sonnet-5 (the whole 4.7+/5 line) REJECTS
        # temperature/top_p/top_k with a 400 -- determinism isn't a knob anymore.
        # Use --repeat to check stability instead (temperature=0 never guaranteed
        # identical output anyway). This also mirrors how the real agent calls the API.
        system=SYSTEM,
        tools=TOOLS,
        tool_choice={"type": "auto"},        # also tests "did it call the tool at all"
        messages=[{"role": "user", "content": question}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            return block.input
    return None


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def label(expect: dict) -> str:
    a, t = bool(expect.get("author")), bool(expect.get("topic"))
    return "both" if a and t else "author" if a else "topic"


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval: does the agent route tool args correctly?")
    ap.add_argument("--cases", type=Path, default=Path(__file__).parent / "routing_cases.jsonl")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case (stability check)")
    args = ap.parse_args()

    load_dotenv()
    import anthropic
    client = anthropic.Anthropic()

    cases = load_cases(args.cases)
    total_pass = 0          # cases that passed ALL repeats (strict)
    total_runs_pass = 0     # individual routing attempts that passed
    total_runs = 0
    # lang -> [runs_pass, runs_total, cases_pass, cases_total]
    by_lang: dict[str, list[int]] = {}

    for c in cases:
        passes = 0
        last_reason = ""
        last_actual: dict | None = None
        for _ in range(args.repeat):
            actual = route_once(client, c["q"])
            ok, reason = grade(actual, c["expect"])
            passes += ok
            if not ok:
                last_reason = reason
                last_actual = actual

        # Two views of the same result:
        #   per-RUN  -- how many of the `repeat` attempts routed correctly.
        #   per-CASE -- strict: the case "passes" only if EVERY repeat did
        #               (flaky == fail). Good for a CI gate; harsh as a headline.
        case_ok = passes == args.repeat
        total_pass += case_ok
        total_runs_pass += passes
        total_runs += args.repeat
        st = by_lang.setdefault(c.get("lang", "??"), [0, 0, 0, 0])
        st[0] += passes            # runs passed
        st[1] += args.repeat       # runs total
        st[2] += case_ok           # cases passed (strict)
        st[3] += 1                 # cases total
        mark = "PASS" if case_ok else "FAIL"
        rep = f" [{passes}/{args.repeat}]" if args.repeat > 1 else ""
        # On failure, show what Claude actually filled -- so you can SEE the
        # failure mode (e.g. the name folded into `topic`) instead of guessing.
        detail = "" if case_ok else f"   ({last_reason}; got {last_actual})"
        print(f"{mark}{rep}  {label(c['expect']):6}  {c['q'][:60]}{detail}")

    n = len(cases)
    print()
    if args.repeat > 1:
        # The honest headline: fraction of ALL individual attempts that routed
        # right (n cases x repeat), so a 2/3 case counts as 2 wins, not 0.
        print(f"{total_runs_pass}/{total_runs} runs passed "
              f"({100 * total_runs_pass / total_runs:.1f}%)")
    print(f"{total_pass}/{n} cases passed"
          + (f" all {args.repeat} repeats" if args.repeat > 1 else "")
          + f" ({100 * total_pass / n:.1f}%)")
    print("by language:")
    for lang in sorted(by_lang):
        rp, rt, cp, ct = by_lang[lang]
        if args.repeat > 1:
            print(f"  {lang}:  {rp}/{rt} runs ({100 * rp / rt:.0f}%)   {cp}/{ct} cases")
        else:
            print(f"  {lang}:  {cp}/{ct}  ({100 * cp / ct:.0f}%)")

    # Conservative CI gate: non-zero exit if ANY case was flaky or failed.
    # (Switch to a run-rate threshold if you'd rather tolerate occasional flakiness.)
    sys.exit(0 if total_pass == n else 1)


if __name__ == "__main__":
    main()