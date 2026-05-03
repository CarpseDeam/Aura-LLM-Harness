"""Local Ollama backend — wraps :class:`OllamaClient` as a :class:`Backend`."""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from aura_harness.backend.protocol import (
    BackendError,
    CompletionResult,
    StreamChunk,
)
from aura_harness.llm import OllamaClient, OllamaError

_NS_PER_MS: float = 1_000_000.0
_BODY_TRUNCATE_CHARS: int = 500


class LocalOllamaBackend:
    """:class:`Backend` implementation backed by a local Ollama server.

    Delegates :meth:`chat` to :meth:`OllamaClient.chat` and translates
    the raw Ollama response into the provider-neutral
    :class:`CompletionResult` shape (``prompt_eval_count`` →
    ``prompt_tokens``, ``eval_count`` → ``completion_tokens``,
    preserving ``total_duration_ms`` and ``raw``). Any
    :class:`OllamaError` raised by the underlying client is re-raised
    as :class:`BackendError` so consumers can catch a single
    provider-agnostic error type.

    The ``response_format`` and ``reasoning`` kwargs on :meth:`chat`
    are part of the cross-backend protocol but have no effect here:
    Ollama's ``/api/chat`` endpoint does not surface a structured-
    output toggle or a thinking-mode flag. They are accepted silently
    so the same call sites work against the cloud backend (which will
    honor them) without rewriting.
    """

    def __init__(self, client: OllamaClient) -> None:
        """Wrap an existing :class:`OllamaClient`.

        Args:
            client: Configured Ollama client. The backend never
                replaces or reconfigures it; the client retains
                ownership of its base URL, default model, JSONL log
                path, and timeouts.
        """
        self._client = client

    @property
    def name(self) -> str:
        """Stable backend identifier used for genealogy stamping."""
        return "local_ollama"

    @property
    def client(self) -> OllamaClient:
        """The wrapped :class:`OllamaClient`.

        Exposed so callers that still need transport-specific
        operations (``health_check``, ``list_models``, ``unload_model``)
        can reach them without holding a separate reference.
        """
        return self._client

    @property
    def default_model(self) -> str:
        """Default model name, delegated to the wrapped client."""
        return self._client.default_model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        seed: int | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        reasoning: bool = False,
    ) -> CompletionResult:
        """Dispatch a chat completion through the wrapped Ollama client.

        ``response_format`` and ``reasoning`` are accepted for protocol
        compatibility but have no effect for Ollama; see the class
        docstring.

        Raises:
            BackendError: When the underlying :class:`OllamaClient`
                raises :class:`OllamaError`.
        """
        del response_format, reasoning
        try:
            raw = self._client.chat(
                messages,
                model=model,
                temperature=temperature,
                seed=seed,
                num_predict=max_tokens,
            )
        except OllamaError as exc:
            raise BackendError(str(exc)) from exc
        return CompletionResult(
            text=raw.text,
            model=raw.model,
            prompt_tokens=raw.prompt_eval_count,
            completion_tokens=raw.eval_count,
            total_duration_ms=raw.total_duration_ms,
            raw=raw.raw,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        seed: int | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        reasoning: bool = False,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion from Ollama's ``/api/chat``.

        Yields a :class:`StreamChunk` per server-emitted JSON line. The
        terminal line (``done=true``) carries the accumulated
        :class:`CompletionResult` in ``StreamChunk.final``.

        ``response_format`` and ``reasoning`` are accepted for protocol
        compatibility but have no effect for Ollama; see the class
        docstring.

        Raises:
            BackendError: On transport failure, non-2xx response, or
                JSON decoding errors.
        """
        del response_format, reasoning
        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        accumulated: list[str] = []
        try:
            with httpx.Client(
                base_url=self._client.base_url,
                timeout=self._client.timeout_seconds,
            ) as http_client:
                with http_client.stream("POST", "/api/chat", json=payload) as response:
                    if not (200 <= response.status_code < 300):
                        body = response.read().decode("utf-8", errors="replace")
                        raise BackendError(
                            f"Ollama returned HTTP {response.status_code}: "
                            f"{body[:_BODY_TRUNCATE_CHARS]}"
                        )
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError as exc:
                            raise BackendError(
                                f"invalid JSON from Ollama stream: {line[:_BODY_TRUNCATE_CHARS]}"
                            ) from exc

                        message = data.get("message") or {}
                        delta = str(message.get("content", ""))
                        if data.get("done"):
                            text = "".join(accumulated)
                            result = CompletionResult(
                                text=text,
                                model=str(data.get("model", model)),
                                prompt_tokens=int(data.get("prompt_eval_count", 0)),
                                completion_tokens=int(data.get("eval_count", 0)),
                                total_duration_ms=float(data.get("total_duration", 0))
                                / _NS_PER_MS,
                                raw=data,
                            )
                            yield StreamChunk(delta="", final=result)
                            return
                        accumulated.append(delta)
                        yield StreamChunk(delta=delta, final=None)
        except httpx.HTTPError as exc:
            raise BackendError(
                f"could not reach Ollama at {self._client.base_url}: {exc}"
            ) from exc
        except OllamaError as exc:
            raise BackendError(str(exc)) from exc
