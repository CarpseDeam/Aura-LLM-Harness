"""Score a :class:`CandidateBatch` end-to-end (extract -> validate -> aggregate)."""
from __future__ import annotations

import logging

from aura_harness.lab.models import Candidate, CandidateBatch
from aura_harness.scoring.extractor import extract_code
from aura_harness.scoring.models import (
    ScoredBatch,
    ScoredCandidate,
    ValidationSpec,
)
from aura_harness.scoring.validator import validate

logger = logging.getLogger(__name__)


def score_batch(batch: CandidateBatch, spec: ValidationSpec) -> ScoredBatch:
    """Score every candidate in ``batch`` against ``spec``.

    Sequential by design for v1. Each ``validate`` call is independent, so
    parallelizing this loop is a straightforward future optimization.

    Args:
        batch: The output of :meth:`CandidateGenerator.generate`.
        spec: Validation requirements applied to each candidate's code.

    Returns:
        A :class:`ScoredBatch` whose ``scored`` field is parallel to
        ``batch.candidates`` (same order, same length).
    """
    scored: list[ScoredCandidate] = [_score_one(c, spec) for c in batch.candidates]
    pass_count = sum(1 for s in scored if s.passed)
    logger.info(
        "Scored batch %s: %d/%d passed",
        batch.batch_id[:8],
        pass_count,
        batch.n_requested,
    )
    return ScoredBatch(batch=batch, spec=spec, scored=tuple(scored))


def _score_one(candidate: Candidate, spec: ValidationSpec) -> ScoredCandidate:
    """Extract + validate a single candidate."""
    if not candidate.is_success or candidate.text is None:
        return ScoredCandidate(
            candidate=candidate,
            extracted_code=None,
            extraction_method=None,
            validation=None,
        )
    code, method = extract_code(candidate.text)
    if code is None:
        return ScoredCandidate(
            candidate=candidate,
            extracted_code=None,
            extraction_method=None,
            validation=None,
        )
    result = validate(code, spec)
    return ScoredCandidate(
        candidate=candidate,
        extracted_code=code,
        extraction_method=method,
        validation=result,
    )
