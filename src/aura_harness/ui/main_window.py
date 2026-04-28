"""Main window for the Aura LLM Harness."""
from __future__ import annotations

import html
import logging
from typing import Final

from PySide6.QtGui import QFont, QFontDatabase, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aura_harness.ui import theme

logger = logging.getLogger(__name__)

WINDOW_TITLE: Final[str] = "Aura LLM Harness"
INPUT_PLACEHOLDER: Final[str] = "Describe what you want built..."


class MainWindow(QMainWindow):
    """Minimal shell window: output area, multiline input, send button."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(960, 720)
        self.setStyleSheet(theme.STYLESHEET)

        mono = self._mono_font()

        self.output: QTextEdit = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setAcceptRichText(True)
        self.output.setFont(mono)

        self.input: QPlainTextEdit = QPlainTextEdit()
        self.input.setFont(mono)
        self.input.setPlaceholderText(INPUT_PLACEHOLDER)
        self.input.setMinimumHeight(theme.INPUT_MIN_HEIGHT)
        self.input.setMaximumHeight(theme.INPUT_MIN_HEIGHT * 2)

        self.send_button: QPushButton = QPushButton("Send")
        self.send_button.clicked.connect(self._on_submit)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.send_button)

        layout = QVBoxLayout()
        margin = theme.PADDING_LG
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(theme.PADDING_MD)
        layout.addWidget(self.output, 1)
        layout.addWidget(self.input)
        layout.addLayout(button_row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        for sequence in (QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")):
            shortcut = QShortcut(sequence, self)
            shortcut.activated.connect(self._on_submit)

        logger.debug("MainWindow initialized")

    @staticmethod
    def _mono_font() -> QFont:
        families = QFontDatabase.families()
        chosen = theme.MONO_FAMILY_PRIMARY if theme.MONO_FAMILY_PRIMARY in families else ""
        if not chosen:
            for fallback in theme.MONO_FAMILY_FALLBACKS:
                if fallback in families:
                    chosen = fallback
                    break
        font = QFont(chosen) if chosen else QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(theme.FONT_SIZE_PT)
        return font

    def _on_submit(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        logger.info("Submit: %d chars", len(text))
        self._append_user_block(text)
        self.input.clear()

    def _append_user_block(self, text: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        block = (
            '<div style="margin: 6px 0;">'
            f'<span style="color: {theme.ACCENT}; font-weight: 600;">&gt; </span>'
            f'<span style="color: {theme.TEXT};">{escaped}</span>'
            "</div>"
        )
        self.output.append(block)
