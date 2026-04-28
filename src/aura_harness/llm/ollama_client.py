"""Synchronous client for a local Ollama server."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import httpx

from aura_harness.llm.exceptions import OllamaError

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL: Final[str] = "http://localhost:11434"
_DEFAULT_MODEL: Final[str] = "qwen3.5:latest"
_DEFAULT_TIMEOUT_S: Final[float] = 120.0
_HEALTH_TIMEOUT_S: Final[float] = 2.0
_PREVIEW_CHARS: Final[int] = 200
_BODY_TRUNCATE_CHARS: Final[int] = 500
_NS_PER_MS: Final[float] = 1_000_000.0


@dataclass(frozen=True)
class CompletionResult:
    """Structured result of a single ``generate`` call."""

    text: str
    model: str
    prompt_eval_count: int
    eval_count: int
    total_duration_ms: float
    raw: dict[str, Any] = field(repr=False)


class OllamaClient:
    """Thin synchronous wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        default_model: str = _DEFAULT_MODEL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_S,
        log_path: Path | None = None,
    ) -> None:
        """Construct a client.

        Args:
            base_url: Root URL of the Ollama server.
            default_model: Model name used when ``generate`` is called without one.
            timeout_seconds: Per-request timeout for non-health requests.
            log_path: JSONL file to append call records to. Defaults to
                ``./.aura/calls.jsonl`` under the current working directory.
        """
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout_seconds
        self._log_path = log_path if log_path is not None else Path.cwd() / ".aura" / "calls.jsonl"

    @property
    def default_model(self) -> str:
        """Model name used when ``generate`` is called without an explicit model."""
        return self._default_model

    @property
    def base_url(self) -> str:
        """Root URL of the Ollama server (no trailing slash)."""
        return self._base_url

    @property
    def timeout_seconds(self) -> float:
        """Per-request timeout for non-health requests, in seconds."""
        return self._timeout

    @property
    def log_path(self) -> Path:
        """Path to the JSONL call log."""
        return self._log_path

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> CompletionResult:
        """Run a non-streaming completion against ``/api/generate``.

        Args:
            prompt: User prompt text.
            model: Override the client's default model.
            system: Optional system prompt.
            temperature: Sampling temperature.
            seed: Optional deterministic sampling seed.
            num_predict: Max tokens to generate (Ollama's ``num_predict``).

        Returns:
            A :class:`CompletionResult`.

        Raises:
            OllamaError: On connection failure or non-2xx response.
        """
        chosen_model = model or self._default_model
        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if num_predict is not None:
            options["num_predict"] = num_predict

        payload: dict[str, Any] = {
            "model": chosen_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system is not None:
            payload["system"] = system

        try:
            data = self._post_json("/api/generate", payload, timeout=self._timeout)
        except OllamaError as exc:
            self._log_call(
                model=chosen_model,
                prompt=prompt,
                response_text="",
                prompt_eval_count=0,
                eval_count=0,
                total_duration_ms=0.0,
                error=str(exc),
            )
            raise

        text = str(data.get("response", ""))
        prompt_eval_count = int(data.get("prompt_eval_count", 0))
        eval_count = int(data.get("eval_count", 0))
        total_duration_ms = float(data.get("total_duration", 0)) / _NS_PER_MS

        self._log_call(
            model=chosen_model,
            prompt=prompt,
            response_text=text,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            total_duration_ms=total_duration_ms,
            error=None,
        )

        return CompletionResult(
            text=text,
            model=chosen_model,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            total_duration_ms=total_duration_ms,
            raw=data,
        )

    def list_models(self) -> list[str]:
        """Return the names of locally installed models via ``/api/tags``."""
        data = self._get_json("/api/tags", timeout=self._timeout)
        models = data.get("models", [])
        return [str(entry["name"]) for entry in models if "name" in entry]

    def health_check(self) -> bool:
        """Return True if the server responds to ``/api/tags`` quickly.

        Never raises; returns False on any error.
        """
        try:
            with httpx.Client(base_url=self._base_url, timeout=_HEALTH_TIMEOUT_S) as client:
                response = client.get("/api/tags")
            return 200 <= response.status_code < 300
        except Exception as exc:
            logger.debug("health_check failed: %s", exc)
            return False

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self._base_url, timeout=timeout) as client:
                response = client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        return self._parse_response(response)

    def _get_json(self, path: str, *, timeout: float) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self._base_url, timeout=timeout) as client:
                response = client.get(path)
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        if not (200 <= response.status_code < 300):
            body = response.text[:_BODY_TRUNCATE_CHARS]
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code}: {body}"
            )
        try:
            return response.json()
        except ValueError as exc:
            body = response.text[:_BODY_TRUNCATE_CHARS]
            raise OllamaError(f"invalid JSON from Ollama: {body}") from exc

    def _log_call(
        self,
        *,
        model: str,
        prompt: str,
        response_text: str,
        prompt_eval_count: int,
        eval_count: int,
        total_duration_ms: float,
        error: str | None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "prompt_preview": prompt[:_PREVIEW_CHARS],
            "response_preview": response_text[:_PREVIEW_CHARS],
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_duration_ms": total_duration_ms,
            "error": error,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("failed to write call log to %s: %s", self._log_path, exc)
