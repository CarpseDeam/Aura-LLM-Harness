"""Frozen dataclasses for the critic layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Critique:
    """Structured review of a candidate against a task spec.

    Attributes:
        passed: Whether the reviewer believes the candidate satisfies the
            spec's contract requirements.
        violations: Concrete, mechanical contract issues found in the
            candidate (e.g. ``"function `find_duplicates` is not at module
            level"``). Empty when ``passed`` is true.
        suggestions: Terse fixes that address each violation
            (e.g. ``"use Path(p).relative_to(directory).as_posix()"``).
            May be empty.
        raw_response: Full critic model response text, kept for diagnostics.
            ``None`` when not available (e.g. transport failure).
    """

    passed: bool
    violations: tuple[str, ...]
    suggestions: tuple[str, ...]
    raw_response: str | None
