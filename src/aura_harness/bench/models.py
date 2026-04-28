"""Frozen dataclasses for the benchmark layer.

These are the structured types written to ``config.json``, ``results.json``
and ``summary.json`` inside a session folder, plus the in-memory aggregate
that the runner returns to its caller.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    """The static configuration for a single bench session.

    Attributes:
        task: The bench task name (matches ``bench/tasks/<task>/``).
        runs: How many independent batches were requested.
        n: How many candidates per batch were requested.
        model: Model name used for every candidate.
        timeout_seconds: Per-candidate verifier subprocess timeout.
        harness_git_sha: ``git rev-parse HEAD`` at session start (or
            ``"unknown"`` if git is unavailable).
        start_time: ISO 8601 UTC timestamp at session start.
    """

    task: str
    runs: int
    n: int
    model: str
    timeout_seconds: float
    harness_git_sha: str
    start_time: str


@dataclass(frozen=True)
class CandidateResult:
    """The bench outcome for one candidate within one run.

    Attributes:
        run_index: Which run this candidate belongs to (``0..runs-1``).
        candidate_index: Position within its batch (``0..n-1``).
        seed: Sampling seed used for this candidate.
        passed: Whether the verifier subprocess exited 0.
        extraction_method: Method used by the extractor, or ``None`` if no
            code was extracted (or generation itself failed).
        candidate_error: Generation error message, or ``None`` on success.
        verify_stderr: Subprocess stderr captured on failure (truncated),
            or ``None`` on pass / no run.
        latency_ms: Wall-clock time spent generating this candidate.
    """

    run_index: int
    candidate_index: int
    seed: int
    passed: bool
    extraction_method: str | None
    candidate_error: str | None
    verify_stderr: str | None
    latency_ms: float


@dataclass(frozen=True)
class RunResult:
    """The bench outcome for one full run (one batch of N candidates).

    Attributes:
        run_index: Position of this run in the session (``0..runs-1``).
        batch_id: The :class:`CandidateBatch` id from the lab layer.
        candidates: One :class:`CandidateResult` per candidate, in batch order.
        run_duration_ms: Wall-clock time for generation + verification of
            this entire run.
    """

    run_index: int
    batch_id: str
    candidates: tuple[CandidateResult, ...]
    run_duration_ms: float


@dataclass(frozen=True)
class SessionSummary:
    """The aggregate result of a bench session.

    Attributes:
        task: The bench task name.
        model: Model name used.
        total_candidates: Total candidates produced across all runs.
        passes: Total candidates that passed the verifier.
        pass_rate: ``passes / total_candidates`` (``0.0`` if no candidates).
        mean_latency_ms: Mean per-candidate generation latency, in ms.
    """

    task: str
    model: str
    total_candidates: int
    passes: int
    pass_rate: float
    mean_latency_ms: float
