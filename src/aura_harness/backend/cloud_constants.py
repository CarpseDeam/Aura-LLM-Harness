"""Default endpoints and model identifiers for cloud backends.

:class:`aura_harness.backend.cloud_http.CloudHTTPBackend` is generic over
any OpenAI-compatible HTTP API. This module collects the per-provider
defaults so call sites do not have to hard-code them.

Future providers (Anthropic via OpenAI-compat shim, Together, OpenRouter,
Groq, etc.) will get their own constants modules in this package and
re-use the same :class:`CloudHTTPBackend` class.
"""
from __future__ import annotations

from typing import Final

DEEPSEEK_BASE_URL: Final[str] = "https://api.deepseek.com"
DEEPSEEK_FLASH: Final[str] = "deepseek-v4-flash"
DEEPSEEK_PRO: Final[str] = "deepseek-v4-pro"
