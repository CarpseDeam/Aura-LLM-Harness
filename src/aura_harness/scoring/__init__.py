"""Scoring layer — extract code, validate against a spec, score a batch."""
from __future__ import annotations

from aura_harness.scoring.extractor import extract_code
from aura_harness.scoring.models import (
    ScoredBatch,
    ScoredCandidate,
    ValidationResult,
    ValidationSpec,
)
from aura_harness.scoring.scorer import score_batch
from aura_harness.scoring.validator import validate

__all__ = [
    "ScoredBatch",
    "ScoredCandidate",
    "ValidationResult",
    "ValidationSpec",
    "extract_code",
    "score_batch",
    "validate",
]
