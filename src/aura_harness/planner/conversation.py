"""Pure data layer for planner conversations.

The planner's chat state is fully described by a sequence of immutable turns
plus the session metadata (id, model, system prompt). All mutation goes
through :meth:`ConversationState.appended`, which returns a new state.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

ROLE_USER: Final[str] = "user"
ROLE_ASSISTANT: Final[str] = "assistant"
ROLE_SYSTEM: Final[str] = "system"

_VALID_ROLES: Final[frozenset[str]] = frozenset({ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_session_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Turn:
    """A single message exchanged in a planner conversation.

    Attributes:
        role: One of ``"user"``, ``"assistant"``, ``"system"``.
        content: The message text.
        timestamp: When the turn was created (UTC).
    """

    role: str
    content: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"invalid role: {self.role!r}")

    @classmethod
    def user(cls, content: str) -> "Turn":
        """Construct a user turn stamped with the current UTC time."""
        return cls(role=ROLE_USER, content=content, timestamp=_now())

    @classmethod
    def assistant(cls, content: str) -> "Turn":
        """Construct an assistant turn stamped with the current UTC time."""
        return cls(role=ROLE_ASSISTANT, content=content, timestamp=_now())


@dataclass(frozen=True)
class ConversationState:
    """The full state of a single planner chat session.

    Attributes:
        session_id: Unique hex identifier for this conversation.
        model: Model name used to drive the planner.
        system_prompt: System prompt prepended to every chat call. Stored
            separately from :attr:`turns` so it can be swapped without
            rewriting history.
        turns: All user/assistant turns exchanged so far, in order.
    """

    model: str
    system_prompt: str
    session_id: str = field(default_factory=_new_session_id)
    turns: tuple[Turn, ...] = ()

    def appended(self, turn: Turn) -> "ConversationState":
        """Return a new state with ``turn`` appended to :attr:`turns`."""
        return ConversationState(
            session_id=self.session_id,
            model=self.model,
            system_prompt=self.system_prompt,
            turns=self.turns + (turn,),
        )

    def with_model(self, model: str) -> "ConversationState":
        """Return a new state with a different model name."""
        return ConversationState(
            session_id=self.session_id,
            model=model,
            system_prompt=self.system_prompt,
            turns=self.turns,
        )

    @property
    def has_assistant_turn(self) -> bool:
        """Whether at least one assistant turn has been recorded."""
        return any(t.role == ROLE_ASSISTANT for t in self.turns)

    @property
    def last(self) -> Turn | None:
        """The most recent turn, or ``None`` if the conversation is empty."""
        return self.turns[-1] if self.turns else None
