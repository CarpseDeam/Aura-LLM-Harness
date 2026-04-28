"""Tests for the main window shell."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytestqt.qtbot import QtBot

from aura_harness.lab import CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.planner import PlannerClient
from aura_harness.ui.main_window import INPUT_PLACEHOLDER, WINDOW_TITLE, MainWindow
from aura_harness.workspace import WorkspaceManager


@pytest.fixture
def client() -> MagicMock:
    mock = MagicMock(spec=OllamaClient)
    mock.default_model = "test-model"
    mock.health_check.return_value = True
    mock.list_models.return_value = ["test-model", "other-model"]
    mock._base_url = "http://localhost:11434"
    mock.base_url = "http://localhost:11434"
    return mock


@pytest.fixture
def generator() -> MagicMock:
    return MagicMock(spec=CandidateGenerator)


@pytest.fixture
def planner() -> MagicMock:
    return MagicMock(spec=PlannerClient)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(tmp_path / "workspace")


def _patch_worker_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the worker thread from actually running during UI submit tests."""
    monkeypatch.setattr(
        "aura_harness.ui.main_window.BatchWorker.start", lambda self: None
    )


def test_window_title_and_placeholder(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
) -> None:
    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)
    assert window.windowTitle() == WINDOW_TITLE
    assert window.input.placeholderText() == INPUT_PLACEHOLDER


def test_submit_appends_input_to_output(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_start(monkeypatch)
    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    window.input.setPlainText("hello world")
    window.send_button.click()

    assert "hello world" in window._output_text()
    assert window.input.toPlainText() == ""


def test_submit_ignores_blank_input(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
) -> None:
    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    window.input.setPlainText("   \n\t  ")
    window.send_button.click()

    assert window._output_text() == ""
    assert window.send_button.isEnabled()


def test_submit_escapes_html(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_start(monkeypatch)
    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    window.input.setPlainText("<script>alert(1)</script>")
    window.send_button.click()

    text = window._output_text()
    assert "&lt;script&gt;" in text
    assert "<script>alert(1)</script>" not in text


def test_combo_box_populated_when_healthy(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
) -> None:
    client.list_models.return_value = ["m1", "test-model", "m2"]

    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    items = [window.model_combo.itemText(i) for i in range(window.model_combo.count())]
    assert items == ["m1", "test-model", "m2"]
    assert window.model_combo.currentText() == "test-model"


def test_combo_box_selects_index_zero_when_default_missing(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
) -> None:
    client.list_models.return_value = ["m1", "m2"]

    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    assert window.model_combo.currentIndex() == 0
    assert window.model_combo.currentText() == "m1"


def test_combo_box_falls_back_when_unhealthy(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
) -> None:
    client.health_check.return_value = False

    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    items = [window.model_combo.itemText(i) for i in range(window.model_combo.count())]
    assert items == ["test-model"]
    assert "Could not reach Ollama" in window._output_text()


def test_combo_box_falls_back_when_list_models_raises(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
) -> None:
    client.list_models.side_effect = RuntimeError("nope")

    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    items = [window.model_combo.itemText(i) for i in range(window.model_combo.count())]
    assert items == ["test-model"]
    assert "Could not reach Ollama" in window._output_text()


def test_n_spin_default_and_range(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
) -> None:
    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    assert window.n_spin.value() == 3
    assert window.n_spin.minimum() == 1
    assert window.n_spin.maximum() == 10


def test_send_disabled_during_run(
    qtbot: QtBot,
    client: MagicMock,
    generator: MagicMock,
    workspace: WorkspaceManager,
    planner: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_start(monkeypatch)
    window = MainWindow(client, generator, workspace, planner)
    qtbot.addWidget(window)

    window.input.setPlainText("hello")
    window.send_button.click()

    assert not window.send_button.isEnabled()
