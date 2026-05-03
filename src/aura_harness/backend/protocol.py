"""Provider-agnostic chat backend protocol.

Consumers (``CriticClient``, ``CandidateGenerator``) talk to a
:class:`Backend`, not to a specific provider client. Concrete backends live
in sibling modules — :class:`aura_harness.backend.local_ollama.LocalOllamaBackend`
wraps the local Ollama transport, and a future cloud backend will wrap
``openai`` for DeepSeek / OpenAI-compatible endpoints.

The :class:`CompletionResult` dataclass is the unified, provider-neutral
shape every backend returns. :class:`BackendError` is the unified error
type every backend raises so consumers can write provider-agnostic
``except`` clauses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


class BackendError(Exception):
    """Unified error type raised by every :class:`Backend` implementation.

    Concrete backends translate provider-specific errors (transport,
    HTTP, decoding) into ``BackendError`` so consumer code can stay
    provider-agnostic.
    """


@dataclass(frozen=True)
class CompletionResult:
    """Structured result of a single :meth:`Backend.chat` call.

    Attributes:
        text: Assistant message content.
        model: Name of the model that produced ``text``.
        prompt_tokens: Number of tokens in the prompt portion of the
            request, as reported by the provider.
        completion_tokens: Number of tokens generated for ``text``.
        total_duration_ms: Wall-clock duration of the call as reported
            by the provider, in milliseconds.
        raw: The provider's raw response payload, unchanged. Kept for
            diagnostics and provider-specific fields not mapped above.
    """

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ms: float
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class StreamChunk:
    """One chunk produced by :meth:`Backend.chat_stream`.

    Attributes:
        delta: Incremental text for this chunk. May be empty (e.g. on a
            terminal chunk that only carries ``final``, or on keep-alive
            chunks that the backend has not filtered out).
        final: ``None`` while streaming. On the terminal chunk, the full
            accumulated :class:`CompletionResult` — equivalent to what
            :meth:`Backend.chat` would have returned for the same args.

    A terminal chunk may carry both a non-empty ``delta`` and a populated
    ``final``, or it may carry ``final`` with ``delta=""``. The caller
    pattern is: iterate chunks, accumulate or display ``delta``, capture
    ``final`` when ``chunk.final is not None``.
    """

    delta: str
    final: CompletionResult | None


class Backend(Protocol):
    """Provider-agnostic chat backend.

    Implementations must expose a ``name`` property, a ``default_model``
    property, and a :meth:`chat` method. Errors must be raised as
    :class:`BackendError` (or a subclass) so consumer code can catch one
    error type regardless of provider.
    """

    @property
    def name(self) -> str:
        """Stable backend identifier for genealogy stamping."""
        ...

    @property
    def default_model(self) -> str:
        """Model name used when callers do not pin one explicitly."""
        ...

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
        """Run a non-streaming chat completion.

        Args:
            messages: List of ``{"role", "content"}`` dicts. Roles are
                ``"system"``, ``"user"``, or ``"assistant"``.
            model: Model name to dispatch against.
            temperature: Sampling temperature.
            seed: Optional deterministic sampling seed. Backends that do
                not support seeding should accept and ignore this.
            max_tokens: Optional cap on tokens generated. Maps to
                provider-specific names (e.g. Ollama's ``num_predict``).
            response_format: Optional structured-output toggle. The
                value ``"json"`` requests JSON-only output where the
                provider supports it. Backends that do not should
                accept and ignore this.
            reasoning: When ``True``, request the provider's thinking
                / extended-reasoning mode where supported. Backends
                that do not should accept and ignore this.

        Returns:
            A :class:`CompletionResult`.

        Raises:
            BackendError: On any provider failure (transport, HTTP,
                decoding). Concrete backends translate their internal
                errors into ``BackendError``.
        """
        ...

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
        """Run a streaming chat completion.

        Yields :class:`StreamChunk` instances as the provider produces
        text. The terminal chunk carries a populated ``final`` field
        with the full accumulated :class:`CompletionResult`.

        Semantic contract: consuming ``chat_stream`` to completion and
        taking the terminal chunk's ``final`` must produce a
        :class:`CompletionResult` equivalent to what :meth:`chat` would
        return for the same args. Implementations are free to share
        code between the two methods or not — the contract is what
        matters.

        Args:
            messages: List of ``{"role", "content"}`` dicts.
            model: Model name to dispatch against.
            temperature: Sampling temperature.
            seed: Optional deterministic sampling seed.
            max_tokens: Optional cap on tokens generated.
            response_format: Optional structured-output toggle.
            reasoning: When ``True``, request the provider's thinking
                / extended-reasoning mode where supported.

        Yields:
            :class:`StreamChunk` per provider-emitted chunk.

        Raises:
            BackendError: On any provider failure (transport, HTTP,
                decoding). Concrete backends translate their internal
                errors into ``BackendError``.
        """
        ...
