"""Provider-agnostic chat backend layer for Aura."""
from __future__ import annotations

from aura_harness.backend.cloud_http import CloudHTTPBackend
from aura_harness.backend.local_ollama import LocalOllamaBackend
from aura_harness.backend.protocol import (
    Backend,
    BackendError,
    CompletionResult,
    StreamChunk,
)

__all__ = [
    "Backend",
    "BackendError",
    "CloudHTTPBackend",
    "CompletionResult",
    "LocalOllamaBackend",
    "StreamChunk",
]
