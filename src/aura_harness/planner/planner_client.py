"""Stateless planner service backed by Ollama's chat endpoint.

Given a :class:`ConversationState`, :class:`PlannerClient` builds the
messages list and asks the model for the next assistant turn. The state
itself is owned by the caller — this client is purely a function-shaped
service over the chat API.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from aura_harness.llm import OllamaClient
from aura_harness.planner.conversation import (
    ROLE_SYSTEM,
    ConversationState,
    Turn,
)

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT: Final[str] = (
    "You are Aura's planning collaborator. Your job is to help the user "
    "shape a clear, buildable spec for a small piece of software through "
    "conversation. Ask focused clarifying questions, surface assumptions "
    "explicitly, and confirm the shape of the work as it firms up. "
    "Do NOT generate code — code generation is a separate downstream step. "
    "Keep replies concise; favor structured bullets when listing assumptions "
    "or open questions."
)

_DEFAULT_TEMPERATURE: Final[float] = 0.4
_DEFAULT_LOG_RELPATH: Final[Path] = Path(".aura") / "planner.jsonl"
_PROMPT_PREVIEW_CHARS: Final[int] = 200
_RESPONSE_PREVIEW_CHARS: Final[int] = 200


class PlannerClient:
    """Turn a :class:`ConversationState` into the next assistant turn."""

    def __init__(
        self,
        client: OllamaClient,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        log_path: Path | None = None,
    ) -> None:
        """Construct a planner client.

        Args:
            client: The :class:`OllamaClient` used for chat calls.
            temperature: Sampling temperature for planner replies.
            log_path: JSONL file to append planner-call records to. Defaults
                to ``./.aura/planner.jsonl`` under the current working
                directory.
        """
        self._client = client
        self._temperature = temperature
        self._log_path = log_path if log_path is not None else Path.cwd() / _DEFAULT_LOG_RELPATH

    @property
    def log_path(self) -> Path:
        """Path to the JSONL planner-call log."""
        return self._log_path

    def reply(self, state: ConversationState) -> Turn:
        """Ask the model for the next assistant turn given ``state``.

        Args:
            state: The conversation so far. Must end with a user turn.

        Returns:
            A new :class:`Turn` with role ``"assistant"``.

        Raises:
            ValueError: If ``state`` has no turns or does not end with a
                user turn.
            OllamaError: On transport or HTTP failure.
        """
        if not state.turns:
            raise ValueError("conversation has no turns")
        if state.turns[-1].role != "user":
            raise ValueError("last turn must be a user turn")

        messages = self._build_messages(state)
        result = self._client.chat(
            messages,
            model=state.model,
            temperature=self._temperature,
        )
        self._log_turn(
            session_id=state.session_id,
            model=state.model,
            user_content=state.turns[-1].content,
            assistant_content=result.text,
            prompt_eval_count=result.prompt_eval_count,
            eval_count=result.eval_count,
            total_duration_ms=result.total_duration_ms,
        )
        return Turn.assistant(result.text)

    @staticmethod
    def _build_messages(state: ConversationState) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if state.system_prompt:
            messages.append({"role": ROLE_SYSTEM, "content": state.system_prompt})
        for turn in state.turns:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def _log_turn(
        self,
        *,
        session_id: str,
        model: str,
        user_content: str,
        assistant_content: str,
        prompt_eval_count: int,
        eval_count: int,
        total_duration_ms: float,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "model": model,
            "user_preview": user_content[:_PROMPT_PREVIEW_CHARS],
            "assistant_preview": assistant_content[:_RESPONSE_PREVIEW_CHARS],
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_duration_ms": total_duration_ms,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("failed to write planner log to %s: %s", self._log_path, exc)
