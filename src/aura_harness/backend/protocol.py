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
from typing import Any, Protocol


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
