"""Frozen dataclasses for the benchmark layer.

These are the structured types written to ``config.json``, ``results.json``
and ``summary.json`` inside a session folder, plus the in-memory aggregate
that the runner returns to its caller.

A bench session has the shape:

    session
    └── runs (one per --runs)
        └── candidates (one per --n)
            └── rounds (round 0 = original generation, rounds 1+ = reflexion)
"""
from __future__ import annotations

from dataclasses import dataclass

from aura_harness.critic import Critique


@dataclass(frozen=True)
class BenchConfig:
    """The static configuration for a single bench session.

    Attributes:
        task: The bench task name (matches ``bench/tasks/<task>/``).
        runs: How many independent batches were requested.
        n: How many candidates per batch were requested.
        model: Coder model name used for every candidate.
        timeout_seconds: Per-candidate verifier subprocess timeout.
        critic_rounds: Maximum reflexion rounds per failed candidate
            (``0`` disables the critic loop).
        critic_model: Reasoning model used by the critic (only meaningful
            when ``critic_rounds > 0``).
        harness_git_sha: ``git rev-parse HEAD`` at session start (or
            ``"unknown"`` if git is unavailable).
        start_time: ISO 8601 UTC timestamp at session start.
    """

    task: str
    runs: int
    n: int
    model: str
    timeout_seconds: float
    critic_rounds: int
    critic_model: str
    harness_git_sha: str
    start_time: str


@dataclass(frozen=True)
class RoundResult:
    """The bench outcome for a single generation+verify attempt at a slot.

    Round 0 is the original candidate generated from the spec alone.
    Rounds 1+ are reflexion rounds, each triggered by the previous round's
    critique.

    Attributes:
        round_index: ``0`` for the original generation, ``1+`` for
            reflexion rounds.
        seed: Sampling seed used for this round.
        passed: Whether the verifier subprocess exited 0 on this round.
        extraction_method: Method used by the extractor, or ``None`` if no
            code was extracted (or generation itself failed).
        candidate_error: Generation error message, or ``None`` on success.
        verify_stderr: Subprocess stderr captured on failure (truncated),
            or ``None`` on pass / no run.
        latency_ms: Wall-clock time spent generating the candidate for this
            round.
        raw_text: Full raw model output for this round, or ``None`` if
            generation failed.
        extracted_code: The extracted Python source for this round, or
            ``None`` if extraction failed.
        critique: The :class:`Critique` from the previous round that
            triggered this round. ``None`` for round 0.
    """

    round_index: int
    seed: int
    passed: bool
    extraction_method: str | None
    candidate_error: str | None
    verify_stderr: str | None
    latency_ms: float
    raw_text: str | None
    extracted_code: str | None
    critique: Critique | None


@dataclass(frozen=True)
class CandidateResult:
    """The bench outcome for one candidate slot within one run.

    A "slot" is a single (run, candidate_index) position. With the critic
    loop disabled it has exactly one round; with the loop enabled it may
    have up to ``critic_rounds + 1`` rounds.

    Attributes:
        run_index: Which run this slot belongs to (``0..runs-1``).
        candidate_index: Position within its batch (``0..n-1``).
        seed: Sampling seed of round 0 (the original candidate). Preserved
            for reproducibility regardless of how many reflexion rounds run.
        rounds: One :class:`RoundResult` per generation attempt at this
            slot. ``rounds[0]`` is always the original generation.
    """

    run_index: int
    candidate_index: int
    seed: int
    rounds: tuple[RoundResult, ...]

    @property
    def passed(self) -> bool:
        """Whether any round at this slot passed the verifier."""
        return any(r.passed for r in self.rounds)

    @property
    def round_zero_passed(self) -> bool:
        """Whether the original (round 0) candidate passed the verifier."""
        return bool(self.rounds) and self.rounds[0].passed

    @property
    def total_latency_ms(self) -> float:
        """Sum of generation latency across every round at this slot."""
        return sum(r.latency_ms for r in self.rounds)


@dataclass(frozen=True)
class RunResult:
    """The bench outcome for one full run (one batch of N candidate slots).

    Attributes:
        run_index: Position of this run in the session (``0..runs-1``).
        batch_id: The :class:`CandidateBatch` id from the lab layer (the
            id of the round-0 batch; reflexion rounds spawn their own
            single-candidate batches with their own ids).
        candidates: One :class:`CandidateResult` per slot, in batch order.
        run_duration_ms: Wall-clock time for generation + verification of
            this entire run, including any reflexion rounds.
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
        model: Coder model name used.
        critic_rounds: Maximum reflexion rounds configured for this session
            (``0`` if the critic loop was disabled).
        critic_model: Reasoning model used by the critic.
        total_candidates: Total candidate slots produced across all runs.
        passes: Total slots that passed the verifier on any round.
        pass_rate: ``passes / total_candidates`` (``0.0`` if no candidates).
        one_shot_passes: Slots that passed on round 0 alone.
        one_shot_pass_rate: ``one_shot_passes / total_candidates`` —
            equivalent to ``pass_rate`` when ``critic_rounds == 0``.
        mean_latency_ms: Mean per-slot generation latency, summed across
            all rounds at the slot, in ms.
    """

    task: str
    model: str
    critic_rounds: int
    critic_model: str
    total_candidates: int
    passes: int
    pass_rate: float
    one_shot_passes: int
    one_shot_pass_rate: float
    mean_latency_ms: float
