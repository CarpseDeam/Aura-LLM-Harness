"""Cloud HTTP backend — :class:`Backend` impl over the OpenAI Python SDK.

The OpenAI SDK speaks the OpenAI chat-completions wire format, which is
the lingua franca of cloud LLM providers (DeepSeek, Together, OpenRouter,
Anthropic via shims, etc.). :class:`CloudHTTPBackend` is generic over
``base_url`` and model strings: per-provider defaults live in sibling
``cloud_*_constants`` modules.

Provider-specific knobs (DeepSeek's ``thinking`` mode parameter,
structured-output toggles) are mapped from the protocol's ``reasoning``
and ``response_format`` kwargs onto the SDK's ``extra_body`` /
``response_format`` pass-throughs.
"""
from __future__ import annotations

from typing import Any, Iterator

from openai import APIError, OpenAI

from aura_harness.backend.protocol import (
    BackendError,
    CompletionResult,
    StreamChunk,
)

_DEFAULT_TIMEOUT_S: float = 900.0


class CloudHTTPBackend:
    """:class:`Backend` over an OpenAI-compatible cloud endpoint.

    Wraps an :class:`openai.OpenAI` client pointed at ``base_url`` and
    translates the unified protocol kwargs into the corresponding SDK
    arguments. Streaming uses ``stream_options={"include_usage": True}``
    so the terminal SSE chunk carries token counts that can be folded
    into :class:`CompletionResult`.

    Errors raised by the SDK (:class:`openai.APIError` and subclasses)
    are wrapped in :class:`BackendError` so consumers can write
    provider-agnostic ``except`` clauses.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        default_model: str,
        name: str = "cloud",
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Construct a cloud backend.

        Args:
            base_url: Root URL of the OpenAI-compatible API.
            api_key: Bearer credential.
            default_model: Model name used when callers do not pin one.
            name: Stable identifier for genealogy stamping. Defaults to
                ``"cloud"``; provider-specific call sites should pass
                a more specific name (e.g. ``"deepseek"``).
            timeout: Per-request timeout in seconds. Defaults to 900s
                because some providers (DeepSeek under load) hold the
                connection open for queueing well past the SDK's
                built-in 600s default.
        """
        self._name = name
        self._default_model = default_model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        """Stable backend identifier used for genealogy stamping."""
        return self._name

    @property
    def default_model(self) -> str:
        """Model name used when callers do not pin one explicitly."""
        return self._default_model

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
        """Run a non-streaming chat completion via the OpenAI SDK.

        Raises:
            BackendError: Wraps any :class:`openai.APIError`.
        """
        kwargs = self._build_kwargs(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning=reasoning,
            stream=False,
        )
        try:
            response = self._client.chat.completions.create(**kwargs)
        except APIError as exc:
            raise BackendError(self._format_error(exc)) from exc

        raw = response.model_dump()
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        usage = raw.get("usage") or {}
        return CompletionResult(
            text=text,
            model=str(raw.get("model", model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_duration_ms=0.0,
            raw=raw,
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
        """Stream a chat completion via the OpenAI SDK.

        Yields one :class:`StreamChunk` per SDK chunk. The SDK iterator
        transparently filters SSE keep-alives, so empty deltas in the
        middle of a stream are real (rare) and forwarded as ``delta=""``.

        The terminal chunk carries the accumulated
        :class:`CompletionResult` built from the ``include_usage`` block
        on the final SSE event.

        Raises:
            BackendError: Wraps any :class:`openai.APIError`.
        """
        kwargs = self._build_kwargs(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning=reasoning,
            stream=True,
        )
        kwargs["stream_options"] = {"include_usage": True}

        accumulated: list[str] = []
        usage: dict[str, Any] = {}
        last_model = model
        last_raw: dict[str, Any] = {}
        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                last_raw = chunk.model_dump()
                last_model = str(last_raw.get("model") or last_model)
                chunk_usage = last_raw.get("usage")
                if chunk_usage:
                    usage = chunk_usage

                choices = last_raw.get("choices") or []
                if not choices:
                    continue
                delta_obj = choices[0].get("delta") or {}
                delta = str(delta_obj.get("content") or "")
                if delta:
                    accumulated.append(delta)
                yield StreamChunk(delta=delta, final=None)
        except APIError as exc:
            raise BackendError(self._format_error(exc)) from exc

        text = "".join(accumulated)
        result = CompletionResult(
            text=text,
            model=last_model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_duration_ms=0.0,
            raw=last_raw,
        )
        yield StreamChunk(delta="", final=result)

    def _build_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
        max_tokens: int | None,
        response_format: str | None,
        reasoning: bool,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return kwargs

    @staticmethod
    def _format_error(exc: APIError) -> str:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return f"cloud HTTP {status}: {exc}"
        return str(exc)
