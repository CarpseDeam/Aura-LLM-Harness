"""Exceptions raised by the Ollama client."""
from __future__ import annotations


class OllamaError(RuntimeError):
    """Raised when an Ollama request fails (transport or HTTP error)."""
