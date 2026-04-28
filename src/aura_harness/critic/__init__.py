"""Critic layer — review a candidate against its spec and emit a structured critique."""
from __future__ import annotations

from aura_harness.critic.critic_client import (
    CRITIC_SYSTEM_PROMPT,
    DEFAULT_CRITIC_MODEL,
    CriticClient,
)
from aura_harness.critic.models import Critique

__all__ = [
    "CRITIC_SYSTEM_PROMPT",
    "DEFAULT_CRITIC_MODEL",
    "CriticClient",
    "Critique",
]
