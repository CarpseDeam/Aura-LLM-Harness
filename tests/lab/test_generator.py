"""Tests for the parallel candidate generator (no real network)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aura_harness.lab.generator import CandidateGenerator
from aura_harness.lab.models import Candidate, CandidateBatch
from aura_harness.llm import CompletionResult, OllamaClient, OllamaError


def _make_result(text: str = "ok", model: str = "test-model") -> CompletionResult:
    return CompletionResult(
        text=text,
        model=model,
        prompt_eval_count=1,
        eval_count=1,
        total_duration_ms=1.0,
        raw={"response": text},
    )


def _mock_client(default_model: str = "test-model") -> MagicMock:
    client = MagicMock(spec=OllamaClient)
    client.default_model = default_model
    return client


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "batches.jsonl"


def test_generate_returns_n_candidates_with_seeds(log_path: Path) -> None:
    client = _mock_client()
    client.generate.return_value = _make_result()
    gen = CandidateGenerator(client, log_path=log_path)

    batch = gen.generate("hello", n=5, base_seed=100)

    assert isinstance(batch, CandidateBatch)
    assert batch.n_requested == 5
    assert len(batch.candidates) == 5
    assert [c.index for c in batch.candidates] == [0, 1, 2, 3, 4]
    assert [c.seed for c in batch.candidates] == [100, 101, 102, 103, 104]
    assert batch.success_count == 5
    assert batch.failed == []
    assert all(c.is_success and c.text == "ok" for c in batch.candidates)


def test_generate_passes_through_call_arguments(log_path: Path) -> None:
    client = _mock_client(default_model="default-m")
    client.generate.return_value = _make_result()
    gen = CandidateGenerator(client, log_path=log_path)

    gen.generate(
        "p",
        n=3,
        model="override-m",
        system="sys",
        temperature=0.9,
        base_seed=10,
        num_predict=64,
    )

    assert client.generate.call_count == 3
    seen_seeds: list[int] = []
    for call in client.generate.call_args_list:
        args, kwargs = call
        assert args == ("p",)
        assert kwargs["model"] == "override-m"
        assert kwargs["system"] == "sys"
        assert kwargs["temperature"] == 0.9
        assert kwargs["num_predict"] == 64
        seen_seeds.append(kwargs["seed"])
    assert sorted(seen_seeds) == [10, 11, 12]


def test_one_failure_does_not_break_batch(log_path: Path) -> None:
    client = _mock_client()

    def side_effect(prompt: str, **kwargs: Any) -> CompletionResult:
        if kwargs["seed"] == 2:
            raise OllamaError("simulated failure")
        return _make_result(text=f"seed={kwargs['seed']}")

    client.generate.side_effect = side_effect
    gen = CandidateGenerator(client, log_path=log_path)

    batch = gen.generate("p", n=5, base_seed=0)

    assert batch.success_count == 4
    assert len(batch.failed) == 1
    failed = batch.failed[0]
    assert failed.index == 2
    assert failed.seed == 2
    assert failed.error is not None
    assert "simulated failure" in failed.error
    assert failed.result is None
    # ordering preserved
    assert [c.index for c in batch.candidates] == [0, 1, 2, 3, 4]


def test_all_failures_no_propagation(log_path: Path) -> None:
    client = _mock_client()
    client.generate.side_effect = OllamaError("nope")
    gen = CandidateGenerator(client, log_path=log_path)

    batch = gen.generate("p", n=5, base_seed=0)

    assert batch.success_count == 0
    assert len(batch.failed) == 5
    assert batch.successful == []
    assert all(c.error == "nope" for c in batch.candidates)


def test_total_wall_duration_is_reasonable(log_path: Path) -> None:
    sleep_s = 0.05
    client = _mock_client()

    def side_effect(prompt: str, **kwargs: Any) -> CompletionResult:
        time.sleep(sleep_s)
        return _make_result()

    client.generate.side_effect = side_effect
    gen = CandidateGenerator(client, max_workers=5, log_path=log_path)

    batch = gen.generate("p", n=5, base_seed=0)

    # Total wall time should be > one candidate's sleep but well under serial time
    # (5 * sleep_s = 250ms). Allow generous slack on CI.
    assert batch.total_wall_duration_ms >= sleep_s * 1000.0 * 0.5
    assert batch.total_wall_duration_ms < sleep_s * 1000.0 * 5.0
    for candidate in batch.candidates:
        assert candidate.wall_duration_ms > 0.0


def test_candidates_ordered_by_index_when_completion_order_differs(log_path: Path) -> None:
    """Slowest seed completes first via inverted delays; ordering must hold."""
    client = _mock_client()

    def side_effect(prompt: str, **kwargs: Any) -> CompletionResult:
        # earlier indices sleep longer → finish later
        delay = (5 - (kwargs["seed"] - 100)) * 0.02
        time.sleep(delay)
        return _make_result(text=f"seed={kwargs['seed']}")

    client.generate.side_effect = side_effect
    gen = CandidateGenerator(client, max_workers=5, log_path=log_path)

    batch = gen.generate("p", n=5, base_seed=100)

    assert [c.index for c in batch.candidates] == [0, 1, 2, 3, 4]
    assert [c.seed for c in batch.candidates] == [100, 101, 102, 103, 104]
    assert [c.text for c in batch.candidates] == [
        "seed=100",
        "seed=101",
        "seed=102",
        "seed=103",
        "seed=104",
    ]


def test_batch_summary_appended_to_log(log_path: Path) -> None:
    client = _mock_client(default_model="m1")
    client.generate.return_value = _make_result()
    gen = CandidateGenerator(client, log_path=log_path)

    batch = gen.generate("a long prompt about debugging", n=3, base_seed=7)

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["batch_id"] == batch.batch_id
    assert record["model"] == "m1"
    assert record["prompt_preview"] == "a long prompt about debugging"
    assert record["n_requested"] == 3
    assert record["success_count"] == 3
    assert record["seeds"] == [7, 8, 9]
    assert "timestamp" in record
    assert "total_wall_duration_ms" in record
    assert "mean_candidate_wall_ms" in record


def test_log_directory_auto_created(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "dir" / "batches.jsonl"
    assert not nested.parent.exists()
    client = _mock_client()
    client.generate.return_value = _make_result()
    gen = CandidateGenerator(client, log_path=nested)

    gen.generate("p", n=2, base_seed=0)

    assert nested.parent.is_dir()
    assert nested.exists()


def test_n_must_be_at_least_one(log_path: Path) -> None:
    gen = CandidateGenerator(_mock_client(), log_path=log_path)
    with pytest.raises(ValueError):
        gen.generate("p", n=0)


def test_default_model_falls_back_to_client(log_path: Path) -> None:
    client = _mock_client(default_model="client-default")
    client.generate.return_value = _make_result(model="client-default")
    gen = CandidateGenerator(client, log_path=log_path)

    batch = gen.generate("p", n=2, base_seed=0)

    assert batch.model == "client-default"
    for call in client.generate.call_args_list:
        assert call.kwargs["model"] == "client-default"


def test_candidate_text_property_handles_failure() -> None:
    failed = Candidate(index=0, seed=0, result=None, error="boom", wall_duration_ms=1.0)
    assert failed.text is None
    assert failed.is_success is False

    succeeded = Candidate(
        index=1, seed=1, result=_make_result(text="hi"), error=None, wall_duration_ms=1.0
    )
    assert succeeded.text == "hi"
    assert succeeded.is_success is True
