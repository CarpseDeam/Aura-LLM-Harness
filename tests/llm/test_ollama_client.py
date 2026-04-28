"""Tests for the Ollama client (no real network)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from aura_harness.llm.exceptions import OllamaError
from aura_harness.llm.ollama_client import CompletionResult, OllamaClient

BASE_URL = "http://localhost:11434"


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "calls.jsonl"


def _patched_httpx(transport: httpx.MockTransport) -> Any:
    """Patch httpx.Client inside the client module to inject a MockTransport."""
    real_client_cls = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    return patch("aura_harness.llm.ollama_client.httpx.Client", side_effect=factory)


def test_generate_returns_completion_result(log_path: Path) -> None:
    response_payload = {
        "response": "hi",
        "model": "test-model",
        "prompt_eval_count": 5,
        "eval_count": 2,
        "total_duration": 1_500_000,  # nanoseconds → 1.5 ms
        "done": True,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["prompt"] == "hello"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.2
        return httpx.Response(200, json=response_payload)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, default_model="test-model", log_path=log_path)

    with _patched_httpx(transport):
        result = client.generate("hello")

    assert isinstance(result, CompletionResult)
    assert result.text == "hi"
    assert result.model == "test-model"
    assert result.prompt_eval_count == 5
    assert result.eval_count == 2
    assert result.total_duration_ms == pytest.approx(1.5)
    assert result.raw == response_payload


def test_generate_passes_system_seed_and_num_predict(log_path: Path) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"response": "", "prompt_eval_count": 0, "eval_count": 0, "total_duration": 0})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, default_model="m", log_path=log_path)

    with _patched_httpx(transport):
        client.generate("p", system="s", seed=42, num_predict=64, temperature=0.7)

    assert captured["system"] == "s"
    assert captured["options"] == {"temperature": 0.7, "seed": 42, "num_predict": 64}


def test_generate_connection_error_raises_ollama_error(log_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, log_path=log_path)

    with _patched_httpx(transport), pytest.raises(OllamaError) as exc_info:
        client.generate("x")

    assert "could not reach Ollama" in str(exc_info.value)
    assert BASE_URL in str(exc_info.value)
    # failure path still logs
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["error"] is not None


def test_generate_non_2xx_raises_with_status_and_body(log_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom internal")

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, log_path=log_path)

    with _patched_httpx(transport), pytest.raises(OllamaError) as exc_info:
        client.generate("x")

    msg = str(exc_info.value)
    assert "500" in msg
    assert "boom internal" in msg


def test_list_models_parses_tags(log_path: Path) -> None:
    payload = {
        "models": [
            {"name": "qwen3.5:latest", "size": 1},
            {"name": "llama3:8b", "size": 2},
            {"size": 3},  # missing name; should be skipped
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, log_path=log_path)

    with _patched_httpx(transport):
        names = client.list_models()

    assert names == ["qwen3.5:latest", "llama3:8b"]


def test_health_check_true_on_success(log_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, log_path=log_path)

    with _patched_httpx(transport):
        assert client.health_check() is True


def test_health_check_false_on_failure(log_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, log_path=log_path)

    with _patched_httpx(transport):
        assert client.health_check() is False


def test_generate_writes_one_jsonl_line(log_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "response": "hello there",
                "prompt_eval_count": 3,
                "eval_count": 4,
                "total_duration": 2_000_000,
            },
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, default_model="m", log_path=log_path)

    with _patched_httpx(transport):
        client.generate("a prompt")

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["model"] == "m"
    assert record["prompt_preview"] == "a prompt"
    assert record["response_preview"] == "hello there"
    assert record["prompt_eval_count"] == 3
    assert record["eval_count"] == 4
    assert record["total_duration_ms"] == pytest.approx(2.0)
    assert record["error"] is None
    assert "timestamp" in record


def test_log_directory_auto_created(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "dir" / "calls.jsonl"
    assert not nested.parent.exists()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"response": "ok", "prompt_eval_count": 0, "eval_count": 0, "total_duration": 0},
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(base_url=BASE_URL, log_path=nested)

    with _patched_httpx(transport):
        client.generate("p")

    assert nested.parent.is_dir()
    assert nested.exists()
