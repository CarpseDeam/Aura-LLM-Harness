"""Dataclasses for the scoring pipeline (extract -> validate -> score)."""
from __future__ import annotations

from dataclasses import dataclass

from aura_harness.lab.models import Candidate, CandidateBatch


@dataclass(frozen=True)
class ValidationSpec:
    """What it takes for extracted code to count as 'passing'.

    Attributes:
        expected_symbols: Top-level names that must be defined in the code.
        test_code: Python source appended to the candidate; must run without
            raising. ``None`` skips the exec step.
        test_timeout_seconds: Wall-clock cap on the test subprocess.
    """

    expected_symbols: tuple[str, ...] = ()
    test_code: str | None = None
    test_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a single extracted code blob against a spec."""

    passed: bool
    parse_ok: bool
    parse_error: str | None
    missing_symbols: tuple[str, ...]
    test_ran: bool
    test_ok: bool | None
    test_error: str | None
    duration_ms: float


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate paired with its extraction + validation outcome."""

    candidate: Candidate
    extracted_code: str | None
    extraction_method: str | None
    validation: ValidationResult | None

    @property
    def passed(self) -> bool:
        """Whether this candidate produced code that passed validation."""
        return self.validation is not None and self.validation.passed


@dataclass(frozen=True)
class ScoredBatch:
    """A scored batch — each candidate annotated with extraction + validation."""

    batch: CandidateBatch
    spec: ValidationSpec
    scored: tuple[ScoredCandidate, ...]

    @property
    def passing(self) -> tuple[ScoredCandidate, ...]:
        """Scored candidates whose validation passed, in batch order."""
        return tuple(s for s in self.scored if s.passed)

    @property
    def pass_rate(self) -> float:
        """Fraction of requested candidates that passed (0.0 - 1.0)."""
        if self.batch.n_requested == 0:
            return 0.0
        return len(self.passing) / self.batch.n_requested

    @property
    def first_passing(self) -> ScoredCandidate | None:
        """First candidate in batch order whose validation passed, or ``None``."""
        passing = self.passing
        return passing[0] if passing else None
