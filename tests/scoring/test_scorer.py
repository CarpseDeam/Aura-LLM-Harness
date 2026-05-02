"""Tests for the batch scorer."""
from __future__ import annotations

from aura_harness.backend import CompletionResult
from aura_harness.lab.models import Candidate, CandidateBatch
from aura_harness.scoring.models import ValidationSpec
from aura_harness.scoring.scorer import score_batch


def _make_candidate(index: int, *, text: str | None = None, error: str | None = None) -> Candidate:
    result: CompletionResult | None = None
    if text is not None:
        result = CompletionResult(
            text=text,
            model="test-model",
            prompt_tokens=1,
            completion_tokens=1,
            total_duration_ms=1.0,
            raw={"response": text},
        )
    return Candidate(
        index=index,
        seed=1000 + index,
        result=result,
        error=error,
        wall_duration_ms=1.0,
    )


def _make_batch(candidates: list[Candidate]) -> CandidateBatch:
    return CandidateBatch(
        batch_id="abcdef0123456789",
        prompt="dummy prompt",
        model="test-model",
        n_requested=len(candidates),
        candidates=candidates,
        total_wall_duration_ms=10.0,
    )


def test_mixed_batch_score() -> None:
    valid_text = "```python\ndef add(a, b):\n    return a + b\n```\n"
    syntax_text = "```python\ndef bad(:\n    pass\n```\n"
    no_code_text = "I cannot write code today, sorry."
    candidates = [
        _make_candidate(0, text=valid_text),
        _make_candidate(1, text=syntax_text),
        _make_candidate(2, text=no_code_text),
        _make_candidate(3, error="model exploded"),
    ]
    batch = _make_batch(candidates)
    spec = ValidationSpec(expected_symbols=("add",), test_code="assert add(1, 2) == 3")

    scored = score_batch(batch, spec)

    assert len(scored.scored) == batch.n_requested
    assert [s.candidate.index for s in scored.scored] == [0, 1, 2, 3]

    s0, s1, s2, s3 = scored.scored

    assert s0.extracted_code is not None
    assert s0.extraction_method == "fenced_python"
    assert s0.validation is not None
    assert s0.validation.passed is True
    assert s0.passed is True

    assert s1.extracted_code is not None
    assert s1.extraction_method == "fenced_python"
    assert s1.validation is not None
    assert s1.validation.parse_ok is False
    assert s1.passed is False

    assert s2.extracted_code is None
    assert s2.extraction_method is None
    assert s2.validation is None
    assert s2.passed is False

    assert s3.extracted_code is None
    assert s3.validation is None
    assert s3.passed is False

    assert scored.pass_rate == 1 / 4
    assert scored.first_passing is not None
    assert scored.first_passing.candidate.index == 0
    assert scored.passing == (s0,)


def test_first_passing_picks_earliest_index() -> None:
    code_text = "```python\ndef add(a, b):\n    return a + b\n```\n"
    candidates = [
        _make_candidate(0, error="boom"),
        _make_candidate(1, text=code_text),
        _make_candidate(2, text=code_text),
    ]
    batch = _make_batch(candidates)
    spec = ValidationSpec(expected_symbols=("add",))

    scored = score_batch(batch, spec)

    assert scored.first_passing is not None
    assert scored.first_passing.candidate.index == 1
    assert len(scored.passing) == 2


def test_all_failed_batch() -> None:
    candidates = [_make_candidate(i, error="nope") for i in range(3)]
    batch = _make_batch(candidates)
    scored = score_batch(batch, ValidationSpec())

    assert scored.pass_rate == 0.0
    assert scored.first_passing is None
    assert scored.passing == ()
    assert all(s.validation is None for s in scored.scored)


def test_ordering_preserved_with_input_ordering() -> None:
    candidates = [
        _make_candidate(0, text="```python\nx = 1\n```"),
        _make_candidate(1, text="```python\ny = 2\n```"),
        _make_candidate(2, text="```python\nz = 3\n```"),
    ]
    batch = _make_batch(candidates)
    scored = score_batch(batch, ValidationSpec())
    assert [s.candidate.index for s in scored.scored] == [0, 1, 2]


def test_scored_batch_carries_spec() -> None:
    spec = ValidationSpec(expected_symbols=("add",))
    batch = _make_batch([_make_candidate(0, error="x")])
    scored = score_batch(batch, spec)
    assert scored.spec is spec
    assert scored.batch is batch
