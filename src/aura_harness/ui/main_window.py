"""Main window for the Aura LLM Harness."""
from __future__ import annotations

import html
import logging
from typing import Final

from PySide6.QtGui import QFont, QFontDatabase, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aura_harness.lab import Candidate, CandidateBatch, CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.ui import theme
from aura_harness.ui.worker import BatchWorker

logger = logging.getLogger(__name__)

WINDOW_TITLE: Final[str] = "Aura LLM Harness"
INPUT_PLACEHOLDER: Final[str] = "Describe what you want built..."
DEFAULT_N: Final[int] = 3
N_MIN: Final[int] = 1
N_MAX: Final[int] = 10


class MainWindow(QMainWindow):
    """Shell window: output area, multiline input, controls, send button."""

    def __init__(self, client: OllamaClient, generator: CandidateGenerator) -> None:
        super().__init__()
        self._client = client
        self._generator = generator
        self._worker: BatchWorker | None = None

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

        self.model_combo: QComboBox = QComboBox()
        self.n_spin: QSpinBox = QSpinBox()
        self.n_spin.setRange(N_MIN, N_MAX)
        self.n_spin.setValue(DEFAULT_N)

        self.send_button: QPushButton = QPushButton("Send")
        self.send_button.clicked.connect(self._on_submit)

        control_row = QHBoxLayout()
        control_row.addWidget(self.model_combo)
        control_row.addWidget(QLabel("N:"))
        control_row.addWidget(self.n_spin)
        control_row.addStretch(1)
        control_row.addWidget(self.send_button)

        layout = QVBoxLayout()
        margin = theme.PADDING_LG
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(theme.PADDING_MD)
        layout.addWidget(self.output, 1)
        layout.addWidget(self.input)
        layout.addLayout(control_row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        for sequence in (QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")):
            shortcut = QShortcut(sequence, self)
            shortcut.activated.connect(self._on_submit)

        self._populate_models()
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

    def _populate_models(self) -> None:
        default_model = self._client.default_model
        models: list[str] = []
        if self._client.health_check():
            try:
                models = self._client.list_models()
            except Exception as exc:  # noqa: BLE001 — fall back gracefully if listing fails
                logger.warning("list_models failed: %s", exc)
                models = []
        if not models:
            self.model_combo.addItem(default_model)
            self.model_combo.setCurrentIndex(0)
            self._append_muted_line(
                f"⚠ Could not reach Ollama at {self._client._base_url}. "
                "Using default model only."
            )
            return
        self.model_combo.addItems(models)
        if default_model in models:
            self.model_combo.setCurrentIndex(models.index(default_model))
        else:
            self.model_combo.setCurrentIndex(0)

    def _on_submit(self) -> None:
        if self._worker is not None:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        n = self.n_spin.value()
        model = self.model_combo.currentText()
        logger.info("Submit: %d chars, n=%d, model=%s", len(text), n, model)
        self._append_user_block(text)
        self._append_muted_line(f"Generating {n} candidates with {model}...")
        self.send_button.setEnabled(False)
        self.input.clear()
        self._worker = BatchWorker(self._generator, text, n, model, parent=self)
        self._worker.batch_complete.connect(self._on_batch_complete)
        self._worker.batch_error.connect(self._on_batch_error)
        self._worker.start()

    def _on_batch_complete(self, batch: CandidateBatch) -> None:
        self._render_batch(batch)
        self._cleanup_worker()

    def _on_batch_error(self, message: str) -> None:
        self._append_styled_line(f"Batch failed: {message}", theme.ERROR)
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        worker = self._worker
        self._worker = None
        self.send_button.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def _render_batch(self, batch: CandidateBatch) -> None:
        short_id = batch.batch_id[:8]
        header = (
            f"Batch {short_id}: {batch.success_count}/{batch.n_requested} "
            f"succeeded in {batch.total_wall_duration_ms:.0f}ms"
        )
        self._append_muted_line(header)
        self._append_separator()
        for candidate in batch.candidates:
            self._append_candidate(candidate)
            self._append_separator()

    def _append_candidate(self, candidate: Candidate) -> None:
        if candidate.is_success:
            color = theme.SUCCESS
            symbol = "✓"
            body = (
                f'<pre style="color: {theme.TEXT}; margin: 4px 0; '
                'white-space: pre-wrap;">'
                f"{html.escape(candidate.text or '')}"
                "</pre>"
            )
        else:
            color = theme.ERROR
            symbol = "✗"
            body = (
                f'<div style="color: {theme.ERROR}; margin: 4px 0;">'
                f"{html.escape(candidate.error or '')}"
                "</div>"
            )
        header = (
            f'<div style="color: {color}; margin: 6px 0 2px 0;">'
            f"[{candidate.index}] {symbol} {candidate.wall_duration_ms:.0f}ms"
            "</div>"
        )
        self.output.append(header + body)

    def _append_user_block(self, text: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        block = (
            '<div style="margin: 6px 0;">'
            f'<span style="color: {theme.ACCENT}; font-weight: 600;">&gt; </span>'
            f'<span style="color: {theme.TEXT};">{escaped}</span>'
            "</div>"
        )
        self.output.append(block)

    def _append_styled_line(self, text: str, color: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        self.output.append(
            f'<div style="color: {color}; margin: 4px 0;">{escaped}</div>'
        )

    def _append_muted_line(self, text: str) -> None:
        self._append_styled_line(text, theme.MUTED)

    def _append_separator(self) -> None:
        self.output.append(
            f'<hr style="border: none; border-top: 1px solid {theme.MUTED};">'
        )
