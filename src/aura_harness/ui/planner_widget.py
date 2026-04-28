"""Planner chat widget — owns its conversation and talks to a PlannerClient."""
from __future__ import annotations

import html
import logging
from typing import Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aura_harness.llm import OllamaClient
from aura_harness.planner import (
    PLANNER_SYSTEM_PROMPT,
    ConversationState,
    PlannerClient,
    Turn,
)
from aura_harness.ui import theme
from aura_harness.ui.planner_worker import PlannerWorker

logger = logging.getLogger(__name__)

PLANNER_INPUT_PLACEHOLDER: Final[str] = "Describe what you want to plan..."
PLANNER_HEADER_TEXT: Final[str] = "Planner"
PLANNER_INPUT_MIN_HEIGHT: Final[int] = 72
PLANNER_INPUT_MAX_HEIGHT: Final[int] = 144
TRANSCRIPT_MIN_HEIGHT: Final[int] = 200
USER_PREFIX: Final[str] = "You"
ASSISTANT_PREFIX: Final[str] = "Planner"
COMMIT_BUTTON_LABEL: Final[str] = "Commit & Generate"
SEND_BUTTON_LABEL: Final[str] = "Send"


class PlannerWidget(QWidget):
    """Owns the planner conversation state and renders the chat UI.

    The widget is self-contained: it builds the :class:`ConversationState`
    internally, drives :class:`PlannerWorker` for replies, and renders user
    and assistant turns into a scrollable transcript. The host is expected
    only to listen for :attr:`commit_requested`.
    """

    commit_requested = Signal(object)

    def __init__(
        self,
        client: OllamaClient,
        planner: PlannerClient,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._planner = planner
        self._worker: PlannerWorker | None = None
        self._mono = self._mono_font()

        self._state = ConversationState(
            model=client.default_model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

        self._build_ui()
        self._populate_models()
        self._update_commit_enabled()

    # -------------------------------------------------------------- build

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

    def _build_ui(self) -> None:
        header = QLabel(PLANNER_HEADER_TEXT)
        header.setStyleSheet(
            f"color: {theme.ACCENT}; font-weight: 600; font-size: {theme.FONT_SIZE_PT + 1}pt;"
        )

        self._transcript_container: QWidget = QWidget()
        self._transcript_layout: QVBoxLayout = QVBoxLayout(self._transcript_container)
        self._transcript_layout.setContentsMargins(
            theme.PADDING_MD, theme.PADDING_MD, theme.PADDING_MD, theme.PADDING_MD
        )
        self._transcript_layout.setSpacing(theme.PADDING_SM)
        self._transcript_layout.addStretch(1)

        self._transcript: QScrollArea = QScrollArea()
        self._transcript.setWidget(self._transcript_container)
        self._transcript.setWidgetResizable(True)
        self._transcript.setMinimumHeight(TRANSCRIPT_MIN_HEIGHT)
        self._transcript.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.input: QPlainTextEdit = QPlainTextEdit()
        self.input.setFont(self._mono)
        self.input.setPlaceholderText(PLANNER_INPUT_PLACEHOLDER)
        self.input.setMinimumHeight(PLANNER_INPUT_MIN_HEIGHT)
        self.input.setMaximumHeight(PLANNER_INPUT_MAX_HEIGHT)

        self.model_combo: QComboBox = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        self.send_button: QPushButton = QPushButton(SEND_BUTTON_LABEL)
        self.send_button.clicked.connect(self._on_send)

        self.commit_button: QPushButton = QPushButton(COMMIT_BUTTON_LABEL)
        self.commit_button.clicked.connect(self._on_commit)

        controls = QHBoxLayout()
        controls.setSpacing(theme.PADDING_SM)
        controls.addWidget(QLabel("Model:"))
        controls.addWidget(self.model_combo, 1)
        controls.addWidget(self.send_button)
        controls.addWidget(self.commit_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.PADDING_SM)
        layout.addWidget(header)
        layout.addWidget(self._transcript, 1)
        layout.addWidget(self.input)
        layout.addLayout(controls)

        for sequence in (QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")):
            shortcut = QShortcut(sequence, self.input)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._on_send)

    def _populate_models(self) -> None:
        default_model = self._client.default_model
        models: list[str] = []
        if self._client.health_check():
            try:
                models = self._client.list_models()
            except Exception as exc:  # noqa: BLE001 — fall back gracefully
                logger.warning("planner list_models failed: %s", exc)
                models = []
        if not models:
            self.model_combo.addItem(default_model)
            self.model_combo.setCurrentIndex(0)
            self._append_muted_line(
                f"⚠ Could not reach Ollama at {self._client.base_url}. "
                "Using default model only."
            )
            return
        self.model_combo.addItems(models)
        if default_model in models:
            self.model_combo.setCurrentIndex(models.index(default_model))
        else:
            self.model_combo.setCurrentIndex(0)

    # ------------------------------------------------------------ public

    @property
    def state(self) -> ConversationState:
        """The current conversation state owned by this widget."""
        return self._state

    # ------------------------------------------------------------ signals

    def _on_model_changed(self, model: str) -> None:
        if not model or model == self._state.model:
            return
        self._state = self._state.with_model(model)

    def _on_send(self) -> None:
        if self._worker is not None:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self._state = self._state.appended(Turn.user(text))
        self._append_user_turn(text)
        self.input.clear()
        self._set_busy(True)
        self._append_muted_line(f"Planner thinking ({self._state.model})...")

        self._worker = PlannerWorker(self._planner, self._state, parent=self)
        self._worker.turn_ready.connect(self._on_turn_ready)
        self._worker.turn_error.connect(self._on_turn_error)
        self._worker.start()

    def _on_turn_ready(self, turn: Turn) -> None:
        self._state = self._state.appended(turn)
        self._append_assistant_turn(turn.content)
        self._cleanup_worker()
        self._update_commit_enabled()

    def _on_turn_error(self, message: str) -> None:
        self._append_styled_line(f"Planner failed: {message}", theme.ERROR)
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        worker = self._worker
        self._worker = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.input.setReadOnly(busy)
        if busy:
            self.commit_button.setEnabled(False)
        else:
            self._update_commit_enabled()

    def _update_commit_enabled(self) -> None:
        self.commit_button.setEnabled(self._state.has_assistant_turn)

    def _on_commit(self) -> None:
        if not self._state.has_assistant_turn:
            return
        logger.info(
            "planner commit_requested: session=%s turns=%d",
            self._state.session_id,
            len(self._state.turns),
        )
        self.commit_requested.emit(self._state)

    # ----------------------------------------------------------- render

    def _append_user_turn(self, text: str) -> None:
        self._add_transcript_widget(self._make_turn_block(USER_PREFIX, text, theme.ACCENT))

    def _append_assistant_turn(self, text: str) -> None:
        self._add_transcript_widget(self._make_turn_block(ASSISTANT_PREFIX, text, theme.SUCCESS))

    def _make_turn_block(self, prefix: str, text: str, prefix_color: str) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.NoFrame)
        frame.setStyleSheet(
            f"background-color: {theme.SURFACE}; border: 1px solid {theme.BORDER}; "
            f"border-radius: {theme.RADIUS}px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(
            theme.PADDING_MD, theme.PADDING_SM, theme.PADDING_MD, theme.PADDING_SM
        )
        layout.setSpacing(theme.PADDING_SM // 2)

        prefix_label = QLabel()
        prefix_label.setTextFormat(Qt.TextFormat.RichText)
        prefix_label.setText(
            f'<span style="color: {prefix_color}; font-weight: 600;">{html.escape(prefix)}</span>'
        )

        body = QLabel()
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setText(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(prefix_label)
        layout.addWidget(body)
        return frame

    def _append_styled_line(self, text: str, color: str) -> None:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        escaped = html.escape(text).replace("\n", "<br>")
        label.setText(f'<span style="color: {color};">{escaped}</span>')
        label.setWordWrap(True)
        self._add_transcript_widget(label)

    def _append_muted_line(self, text: str) -> None:
        self._append_styled_line(text, theme.MUTED)

    def _add_transcript_widget(self, widget: QWidget) -> None:
        index = max(self._transcript_layout.count() - 1, 0)
        self._transcript_layout.insertWidget(index, widget)
        self._transcript.ensureWidgetVisible(widget)
