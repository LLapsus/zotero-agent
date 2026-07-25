"""
ui.py  -  tiny ANSI styling for the interactive CLIs.

Pure cosmetics for the interactive loops: coloured retrieval indices, distinct
tool-activity lines, and a subtly coloured input prompt so the user's line
stands out from the program's output.

Every helper degrades to plain text when stdout is NOT a terminal (piped output,
captured logs) or when NO_COLOR is set -- so redirected output stays clean and
the demos' captured examples don't fill up with escape codes.
"""

from __future__ import annotations

import os
import sys

# Colour only for a real terminal; honour the NO_COLOR convention.
_ON = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _sgr(code: str):
    def paint(s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if _ON else s
    return paint


bold = _sgr("1")
dim = _sgr("2")
cyan = _sgr("36")
green = _sgr("32")
yellow = _sgr("33")
grey = _sgr("90")


def idx(n: int) -> str:
    """A retrieval index like [3], coloured to stand out from the title."""
    return cyan(f"[{n}]")


def tool(label: str) -> str:
    """A tool-activity marker like [tool call], set apart from normal output."""
    return yellow(f"[{label}]")


def ask(marker: str = "> ") -> str:
    """input() with a subtly coloured prompt marker (no background)."""
    return input(cyan(marker))  # cyan() is a no-op when not a TTY -> plain marker
