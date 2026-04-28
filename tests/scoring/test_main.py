"""Tests for the scoring CLI entry point.

These tests stub out the network-dependent pieces (``CandidateGenerator``,
``OllamaClient``) so the CLI can be exercised in-process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import aura_harness.scoring.__main__ as cli
from aura_harness.lab.models import Candidate, CandidateBatch
from aura_harness.llm import CompletionResult
from aura_harness.scoring.models import (
    ScoredBatch,
    ScoredCandidate,
    ValidationResult,
    ValidationSpec,
)


def _stub_main_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the network + scoring path with capturing stubs.

    Returns a dict the test can inspect to see what spec the CLI built.
    """
    captured: dict[str, Any] = {}

    monkeypatch.setattr(cli, "OllamaClient", MagicMock())

    def fake_generator(_client: Any) -> Any:
        result = CompletionResult(
            text="```python\ndef add(a, b):\n    return a + b\n```",
            model="test-model",
            prompt_eval_count=1,
            eval_count=1,
            total_duration_ms=1.0,
            raw={},
        )
        candidate = Candidate(
            index=0, seed=0, result=result, error=None, wall_duration_ms=1.0
        )
        batch = CandidateBatch(
            batch_id="deadbeef" * 4,
            prompt="dummy",
            model="test-model",
            n_requested=1,
            candidates=[candidate],
            total_wall_duration_ms=1.0,
        )
        gen = MagicMock()
        gen.generate.return_value = batch
        return gen

    monkeypatch.setattr(cli, "CandidateGenerator", fake_generator)

    def fake_score(batch: CandidateBatch, spec: ValidationSpec) -> ScoredBatch:
        captured["spec"] = spec
        scored_one = ScoredCandidate(
            candidate=batch.candidates[0],
            extracted_code="def add(a, b):\n    return a + b",
            extraction_method="fenced_python",
            validation=ValidationResult(
                passed=True,
                parse_ok=True,
                parse_error=None,
                missing_symbols=(),
                test_ran=spec.test_code is not None,
                test_ok=True if spec.test_code is not None else None,
                test_error=None,
                duration_ms=1.0,
            ),
        )
        return ScoredBatch(batch=batch, spec=spec, scored=(scored_one,))

    monkeypatch.setattr(cli, "score_batch", fake_score)
    return captured


def test_test_file_is_read_and_used_as_spec_test_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    test_path = tmp_path / "tests.py"
    payload = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n"
    test_path.write_text(payload, encoding="utf-8")
    captured = _stub_main_pipeline(monkeypatch)

    exit_code = cli.main(["a prompt", "--test-file", str(test_path)])

    assert exit_code == 0
    spec = captured["spec"]
    assert spec.test_code == payload


def test_test_flag_still_works(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = _stub_main_pipeline(monkeypatch)

    exit_code = cli.main(["a prompt", "--test", "assert add(1, 2) == 3"])

    assert exit_code == 0
    assert captured["spec"].test_code == "assert add(1, 2) == 3"


def test_passing_both_test_and_test_file_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    test_path = tmp_path / "tests.py"
    test_path.write_text("assert True\n", encoding="utf-8")
    _stub_main_pipeline(monkeypatch)

    exit_code = cli.main(
        ["a prompt", "--test", "assert True", "--test-file", str(test_path)]
    )

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err
    assert "--test" in err and "--test-file" in err


def test_missing_test_file_errors_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does_not_exist.py"
    _stub_main_pipeline(monkeypatch)

    exit_code = cli.main(["a prompt", "--test-file", str(missing)])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "could not read --test-file" in err
    assert str(missing) in err


def test_no_test_means_spec_test_code_is_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = _stub_main_pipeline(monkeypatch)

    exit_code = cli.main(["a prompt"])

    assert exit_code == 0
    assert captured["spec"].test_code is None
