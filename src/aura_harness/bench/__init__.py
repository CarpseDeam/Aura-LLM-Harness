"""Benchmark layer — run a task against the candidate generator and score it.

Public entry points:

- :func:`run_bench` — run a task end-to-end and return a session summary.
- The dataclasses in :mod:`aura_harness.bench.models`.
"""
from __future__ import annotations

from aura_harness.bench.models import (
    BenchConfig,
    CandidateResult,
    RunResult,
    SessionSummary,
)
from aura_harness.bench.runner import run_bench

__all__ = [
    "BenchConfig",
    "CandidateResult",
    "RunResult",
    "SessionSummary",
    "run_bench",
]
