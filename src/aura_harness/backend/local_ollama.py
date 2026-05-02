"""Local Ollama backend — wraps :class:`OllamaClient` as a :class:`Backend`."""
from __future__ import annotations

from aura_harness.backend.protocol import BackendError, CompletionResult
from aura_harness.llm import OllamaClient, OllamaError


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
