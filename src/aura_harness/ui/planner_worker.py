"""QThread worker that runs a planner reply off the GUI thread."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal

from aura_harness.planner import ConversationState, PlannerClient, Turn

logger = logging.getLogger(__name__)


class PlannerWorker(QThread):
    """Run a single :meth:`PlannerClient.reply` call on a worker thread.

    Emits :attr:`turn_ready` with the new assistant :class:`Turn` on success,
    or :attr:`turn_error` with a string message on failure.
    """

    turn_ready = Signal(object)
    turn_error = Signal(str)

    def __init__(
        self,
        planner: PlannerClient,
        state: ConversationState,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._planner = planner
        self._state = state

    def run(self) -> None:
        try:
            turn: Turn = self._planner.reply(self._state)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the GUI
            logger.exception("planner reply failed")
            self.turn_error.emit(str(exc))
            return
        self.turn_ready.emit(turn)
