"""Dataclasses for parallel candidate generation results."""
from __future__ import annotations

from dataclasses import dataclass

from aura_harness.backend import CompletionResult


@dataclass(frozen=True)
class Candidate:
    """A single attempt produced by the candidate generator.

    Attributes:
        index: Position in the batch, ``0..n-1``.
        seed: Sampling seed passed to the model for this attempt.
        result: The successful completion, or ``None`` if generation failed.
        error: Error message string, or ``None`` on success.
        wall_duration_ms: Wall-clock time spent on this candidate, in milliseconds.
    """

    index: int
    seed: int
    result: CompletionResult | None
    error: str | None
    wall_duration_ms: float

    @property
    def is_success(self) -> bool:
        """Whether this candidate produced a usable completion."""
        return self.result is not None

    @property
    def text(self) -> str | None:
        """The completion text, or ``None`` if the candidate failed."""
        if self.result is None:
            return None
        return self.result.text


@dataclass(frozen=True)
class CandidateBatch:
    """A collection of N candidate attempts produced from a single prompt.

    Attributes:
        batch_id: Unique hex identifier for this batch.
        prompt: The user prompt that was sent.
        model: Model name used for every candidate in the batch.
        n_requested: Number of candidates requested.
        candidates: Always length ``n_requested``, ordered by ``index``.
        total_wall_duration_ms: Wall-clock time for the whole batch.
    """

    batch_id: str
    prompt: str
    model: str
    n_requested: int
    candidates: list[Candidate]
    total_wall_duration_ms: float

    @property
    def successful(self) -> list[Candidate]:
        """Candidates that produced a completion, preserving index order."""
        return [c for c in self.candidates if c.is_success]

    @property
    def failed(self) -> list[Candidate]:
        """Candidates that raised an error, preserving index order."""
        return [c for c in self.candidates if not c.is_success]

    @property
    def success_count(self) -> int:
        """Number of candidates that produced a completion."""
        return sum(1 for c in self.candidates if c.is_success)
