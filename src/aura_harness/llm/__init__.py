"""Local LLM client package for Aura."""
from __future__ import annotations

from aura_harness.llm.exceptions import OllamaError
from aura_harness.llm.ollama_client import CompletionResult, OllamaClient

__all__ = ["CompletionResult", "OllamaClient", "OllamaError"]
