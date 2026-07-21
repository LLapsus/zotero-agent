"""
_common.py  -  tiny shared helper for the demo scripts.

The demos live in demos/ but the pipeline they explain (zotero_rag.py) lives one
directory up. Importing it from here would normally fail, so every demo starts
with `from _common import zotero_rag` and this file does two things:

  1. puts the repo root on sys.path, so `import zotero_rag` works, and
  2. re-exports the module, so demos get it in one line.

That's the whole trick. There is no magic here - open zotero_rag.py to see the
real code; these demos only *call* it and print what comes back.
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo root = the folder that contains zotero_rag.py, i.e. demos/..
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zotero_rag  # noqa: E402  (import must come after the sys.path tweak)


def rule(title: str) -> None:
    """Print a labelled horizontal rule, just to keep demo output readable."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


__all__ = ["zotero_rag", "rule", "REPO_ROOT"]
