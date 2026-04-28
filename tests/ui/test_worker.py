"""Tests for the BatchWorker QThread."""
from __future__ import annotations

from unittest.mock import MagicMock

from pytestqt.qtbot import QtBot

from aura_harness.lab import CandidateBatch, CandidateGenerator
from aura_harness.ui.worker import BatchWorker


def _empty_batch() -> CandidateBatch:
    return CandidateBatch(
        batch_id="abcdef0123456789",
        prompt="hi",
        model="m",
        n_requested=0,
        candidates=[],
        total_wall_duration_ms=12.5,
    )


def test_worker_emits_batch_complete_on_success(qtbot: QtBot) -> None:
    generator = MagicMock(spec=CandidateGenerator)
    batch = _empty_batch()
    generator.generate.return_value = batch

    worker = BatchWorker(generator, prompt="hi", n=3, model="m")

    with qtbot.waitSignal(worker.batch_complete, timeout=5000) as blocker:
        worker.start()

    assert blocker.args == [batch]
    assert worker.wait(5000)
    generator.generate.assert_called_once_with("hi", 3, model="m")


def test_worker_emits_batch_error_on_exception(qtbot: QtBot) -> None:
    generator = MagicMock(spec=CandidateGenerator)
    generator.generate.side_effect = ValueError("boom")

    worker = BatchWorker(generator, prompt="hi", n=3, model="m")

    completed: list[object] = []
    worker.batch_complete.connect(lambda batch: completed.append(batch))

    with qtbot.waitSignal(worker.batch_error, timeout=5000) as blocker:
        worker.start()

    assert blocker.args == ["boom"]
    assert worker.wait(5000)
    assert completed == []
