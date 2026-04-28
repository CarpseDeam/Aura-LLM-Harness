"""QThread worker that runs a candidate batch off the GUI thread."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal

from aura_harness.lab import CandidateGenerator

logger = logging.getLogger(__name__)


class BatchWorker(QThread):
    """Run a single :meth:`CandidateGenerator.generate` call on a worker thread.

    Per-candidate failures are captured by the generator and returned as part
    of the batch, so :attr:`batch_complete` is the normal path. :attr:`batch_error`
    only fires on something truly catastrophic (validation errors, etc.).
    """

    batch_complete = Signal(object)
    batch_error = Signal(str)

    def __init__(
        self,
        generator: CandidateGenerator,
        prompt: str,
        n: int,
        model: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._generator = generator
        self._prompt = prompt
        self._n = n
        self._model = model

    def run(self) -> None:
        try:
            batch = self._generator.generate(self._prompt, self._n, model=self._model)
        except Exception as exc:  # noqa: BLE001 — surface any catastrophic failure to the GUI
            logger.exception("batch generation failed")
            self.batch_error.emit(str(exc))
            return
        self.batch_complete.emit(batch)
