"""Pull a Python code blob out of a model's free-form response.

The strategy is a small fixed stack: prefer fenced ```python blocks (last one
wins, since models tend to put the answer after their reasoning), fall back to
unmarked fenced blocks, and finally fall back to parsing the whole text as
Python. Anything fancier (heuristic stitching, multi-block merging, etc.) is
explicitly out of scope.
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Final

logger = logging.getLogger(__name__)

_FENCED_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"```(\w+)?\s*\n(.*?)\n```",
    re.DOTALL,
)
_PYTHON_LANG_TAGS: Final[frozenset[str]] = frozenset({"python", "py"})

ExtractionMethod = str  # "fenced_python" | "fenced_unmarked" | "raw_parse"


def extract_code(text: str) -> tuple[str | None, str | None]:
    """Extract a Python code blob from a model response.

    Args:
        text: The raw response text.

    Returns:
        A ``(code, method)`` tuple. ``code`` is the extracted Python source
        with surrounding whitespace stripped; ``method`` is one of
        ``"fenced_python"``, ``"fenced_unmarked"``, or ``"raw_parse"``.
        Returns ``(None, None)`` when no extractable code is found.
    """
    if not text:
        return (None, None)

    blocks = _FENCED_BLOCK_RE.findall(text)
    python_blocks = [
        body for tag, body in blocks if tag and tag.lower() in _PYTHON_LANG_TAGS
    ]
    if python_blocks:
        return _finalize(python_blocks[-1], "fenced_python")

    if blocks:
        return _finalize(blocks[-1][1], "fenced_unmarked")

    stripped = text.strip()
    if stripped:
        try:
            ast.parse(stripped)
        except SyntaxError:
            return (None, None)
        return (stripped, "raw_parse")

    return (None, None)


def _finalize(body: str, method: ExtractionMethod) -> tuple[str | None, str | None]:
    """Strip whitespace and reject empties."""
    code = body.strip()
    if not code:
        return (None, None)
    return (code, method)
