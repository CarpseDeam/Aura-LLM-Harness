"""Provider-agnostic chat backend layer for Aura."""
from __future__ import annotations

from aura_harness.backend.local_ollama import LocalOllamaBackend
from aura_harness.backend.protocol import Backend, BackendError, CompletionResult

__all__ = [
    "Backend",
    "BackendError",
    "CompletionResult",
    "LocalOllamaBackend",
]
