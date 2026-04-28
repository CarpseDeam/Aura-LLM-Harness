"""Tests for the main window shell."""
from __future__ import annotations

from pytestqt.qtbot import QtBot

from aura_harness.ui.main_window import INPUT_PLACEHOLDER, WINDOW_TITLE, MainWindow


def test_window_title_and_placeholder(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == WINDOW_TITLE
    assert window.input.placeholderText() == INPUT_PLACEHOLDER
    assert window.output.isReadOnly()


def test_submit_appends_input_to_output(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    window.input.setPlainText("hello world")
    window.send_button.click()

    assert "hello world" in window.output.toPlainText()
    assert window.input.toPlainText() == ""


def test_submit_ignores_blank_input(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    window.input.setPlainText("   \n\t  ")
    window.send_button.click()

    assert window.output.toPlainText() == ""


def test_submit_escapes_html(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    window.input.setPlainText("<script>alert(1)</script>")
    window.send_button.click()

    html_text = window.output.toHtml()
    assert "&lt;script&gt;" in html_text
    assert "<script>alert(1)</script>" not in html_text
