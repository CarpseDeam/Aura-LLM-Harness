"""Main window for the Aura LLM Harness."""
from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Final

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aura_harness.lab import Candidate, CandidateBatch, CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.planner import ConversationState, PlannerClient
from aura_harness.scaffold import get_template
from aura_harness.scoring import ScoredBatch, ScoredCandidate, ValidationSpec
from aura_harness.ui import theme
from aura_harness.ui.planner_widget import PlannerWidget
from aura_harness.ui.scaffold_dialog import ScaffoldDialog
from aura_harness.ui.worker import BatchWorker
from aura_harness.workspace import WorkspaceError, WorkspaceManager

logger = logging.getLogger(__name__)

WINDOW_TITLE: Final[str] = "Aura LLM Harness"
INPUT_PLACEHOLDER: Final[str] = (
    "Describe the function or feature to build — Run Worker generates N candidates."
)
WORKER_PROMPT_LABEL: Final[str] = "Worker prompt"
WORKER_OUTPUT_LABEL: Final[str] = "Worker output"
WORKER_OUTPUT_EMPTY_TEXT: Final[str] = (
    "Generated candidates appear here once Run Worker produces a batch."
)
WORKER_MODEL_LABEL: Final[str] = "Worker model:"
WORKSPACE_LABEL: Final[str] = "Workspace"
VALIDATE_LABEL: Final[str] = "Validate output (parse, expected symbols, optional test code)"
SYMBOLS_PLACEHOLDER: Final[str] = "comma-separated, e.g. add, helper"
TEST_PLACEHOLDER: Final[str] = "assert add(1,2) == 3"
DEFAULT_N: Final[int] = 3
N_MIN: Final[int] = 1
N_MAX: Final[int] = 10
TEST_TIMEOUT_SECONDS: Final[float] = 10.0
ERROR_TRUNCATE: Final[int] = 200
SIDEBAR_WIDTH: Final[int] = 250
WORKSPACE_PATH_ELIDE_WIDTH: Final[int] = 230
CONTEXT_FILE_BYTE_LIMIT: Final[int] = 100 * 1024
DEFAULT_APPLY_FILENAME: Final[str] = "candidate.py"


class MainWindow(QMainWindow):
    """Shell window: sidebar + (output area, validation, input, controls)."""

    def __init__(
        self,
        client: OllamaClient,
        generator: CandidateGenerator,
        workspace: WorkspaceManager,
        planner: PlannerClient,
    ) -> None:
        super().__init__()
        self._client = client
        self._generator = generator
        self._workspace = workspace
        self._planner = planner
        self._worker: BatchWorker | None = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1280, 720)
        self.setStyleSheet(theme.STYLESHEET)

        mono = self._mono_font()
        self._mono = mono

        sidebar = self._build_sidebar()
        right_column = self._build_right_column(mono)

        root_layout = QHBoxLayout()
        margin = theme.PADDING_LG
        root_layout.setContentsMargins(margin, margin, margin, margin)
        root_layout.setSpacing(theme.PADDING_MD)
        root_layout.addWidget(sidebar)
        root_layout.addWidget(right_column, 1)

        container = QWidget()
        container.setLayout(root_layout)
        self.setCentralWidget(container)

        for sequence in (QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")):
            shortcut = QShortcut(sequence, self)
            shortcut.activated.connect(self._on_submit)

        self._populate_models()
        self._refresh_workspace_view()
        logger.debug("MainWindow initialized")

    # ------------------------------------------------------------------ build

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

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)

        header_label = QLabel(WORKSPACE_LABEL)
        header_label.setProperty("role", "section")

        self.workspace_path_label: QLabel = QLabel()
        self.workspace_path_label.setProperty("role", "path")
        self.workspace_path_label.setToolTip(str(self._workspace.root))
        self.workspace_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        pick_button = QPushButton("Choose…")
        pick_button.clicked.connect(self._on_pick_workspace)

        new_button = QPushButton("New project")
        new_button.clicked.connect(self._on_new_project)

        action_row = QHBoxLayout()
        action_row.setSpacing(theme.PADDING_SM)
        action_row.addWidget(pick_button)
        action_row.addWidget(new_button)
        action_row.addStretch(1)

        self.file_list: QListWidget = QListWidget()

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_workspace_view)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.PADDING_SM)
        layout.addWidget(header_label)
        layout.addWidget(self.workspace_path_label)
        layout.addLayout(action_row)
        layout.addWidget(self.file_list, 1)
        layout.addWidget(refresh_button)
        return sidebar

    def _build_right_column(self, mono: QFont) -> QWidget:
        right = QWidget()

        self.planner_widget: PlannerWidget = PlannerWidget(self._client, self._planner)
        self.planner_widget.commit_requested.connect(self._on_planner_commit)

        output_header = QLabel(WORKER_OUTPUT_LABEL)
        output_header.setProperty("role", "section")

        self._output_container: QWidget = QWidget()
        self._output_layout: QVBoxLayout = QVBoxLayout(self._output_container)
        self._output_layout.setContentsMargins(
            theme.PADDING_MD, theme.PADDING_MD, theme.PADDING_MD, theme.PADDING_MD
        )
        self._output_layout.setSpacing(theme.PADDING_SM)
        self._output_empty: QLabel = QLabel(WORKER_OUTPUT_EMPTY_TEXT)
        self._output_empty.setProperty("role", "empty")
        self._output_empty.setWordWrap(True)
        self._output_layout.addWidget(self._output_empty)
        self._output_layout.addStretch(1)

        self.output: QScrollArea = QScrollArea()
        self.output.setWidget(self._output_container)
        self.output.setWidgetResizable(True)
        self.output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        worker_header = QLabel("Worker")
        worker_header.setProperty("role", "section")

        prompt_label = QLabel(WORKER_PROMPT_LABEL)
        prompt_label.setProperty("role", "inline")

        self.input: QPlainTextEdit = QPlainTextEdit()
        self.input.setFont(mono)
        self.input.setPlaceholderText(INPUT_PLACEHOLDER)
        self.input.setMinimumHeight(theme.INPUT_MIN_HEIGHT)
        self.input.setMaximumHeight(theme.INPUT_MIN_HEIGHT * 2)

        self.validation_group, self._validation_details = self._build_validation_section(mono)

        model_label = QLabel(WORKER_MODEL_LABEL)
        model_label.setProperty("role", "inline")
        n_label = QLabel("N:")
        n_label.setProperty("role", "inline")

        self.model_combo: QComboBox = QComboBox()
        self.n_spin: QSpinBox = QSpinBox()
        self.n_spin.setRange(N_MIN, N_MAX)
        self.n_spin.setValue(DEFAULT_N)
        self.n_spin.setFixedWidth(64)

        self.send_button: QPushButton = QPushButton("Run Worker")
        self.send_button.setProperty("primary", True)
        self.send_button.clicked.connect(self._on_submit)

        control_row = QHBoxLayout()
        control_row.setSpacing(theme.PADDING_SM)
        control_row.addWidget(model_label)
        control_row.addWidget(self.model_combo, 1)
        control_row.addWidget(n_label)
        control_row.addWidget(self.n_spin)
        control_row.addStretch(1)
        control_row.addWidget(self.send_button)

        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.PADDING_MD)
        layout.addWidget(self.planner_widget, 1)
        layout.addWidget(self._make_hrule())
        layout.addWidget(output_header)
        layout.addWidget(self.output, 1)
        layout.addWidget(self._make_hrule())
        layout.addWidget(worker_header)
        layout.addWidget(prompt_label)
        layout.addWidget(self.input)
        layout.addWidget(self.validation_group)
        layout.addWidget(self._validation_details)
        layout.addLayout(control_row)
        return right

    @staticmethod
    def _make_hrule() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.NoFrame)
        line.setProperty("role", "hline")
        line.setFixedHeight(1)
        return line

    def _build_validation_section(self, mono: QFont) -> tuple[QCheckBox, QWidget]:
        """Build the validation toggle checkbox and its (initially hidden) details panel."""
        checkbox = QCheckBox(VALIDATE_LABEL)
        checkbox.setChecked(False)

        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(
            theme.PADDING_LG, 0, 0, 0
        )
        details_layout.setSpacing(theme.PADDING_SM)

        symbols_label = QLabel("Expected symbols:")
        symbols_label.setProperty("role", "inline")
        self.symbols_edit: QLineEdit = QLineEdit()
        self.symbols_edit.setPlaceholderText(SYMBOLS_PLACEHOLDER)
        self.symbols_edit.setFont(mono)

        symbols_row = QHBoxLayout()
        symbols_row.setSpacing(theme.PADDING_SM)
        symbols_row.addWidget(symbols_label)
        symbols_row.addWidget(self.symbols_edit, 1)

        test_label = QLabel("Test code:")
        test_label.setProperty("role", "inline")
        test_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.test_edit: QPlainTextEdit = QPlainTextEdit()
        self.test_edit.setFont(mono)
        self.test_edit.setPlaceholderText(TEST_PLACEHOLDER)
        line_height = self.test_edit.fontMetrics().lineSpacing()
        self.test_edit.setFixedHeight(line_height * 4 + theme.PADDING_LG)

        test_row = QHBoxLayout()
        test_row.setSpacing(theme.PADDING_SM)
        test_row.addWidget(test_label)
        test_row.addWidget(self.test_edit, 1)

        details_layout.addLayout(symbols_row)
        details_layout.addLayout(test_row)
        details.setVisible(False)

        checkbox.toggled.connect(self._on_validation_toggled)
        return checkbox, details

    def _on_validation_toggled(self, checked: bool) -> None:
        self._validation_details.setVisible(checked)

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
                f"⚠ Could not reach Ollama at {self._client.base_url}. "
                "Using default model only."
            )
            return
        self.model_combo.addItems(models)
        if default_model in models:
            self.model_combo.setCurrentIndex(models.index(default_model))
        else:
            self.model_combo.setCurrentIndex(0)

    # ----------------------------------------------------------- workspace ui

    def _refresh_workspace_view(self) -> None:
        """Reload the sidebar's path label and file list from the workspace."""
        self._update_workspace_path_label()
        self.file_list.clear()
        try:
            files = self._workspace.list_files()
        except OSError as exc:
            logger.warning("workspace listing failed: %s", exc)
            files = []
        for rel in files:
            item = QListWidgetItem(rel.as_posix())
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, rel.as_posix())
            self.file_list.addItem(item)

    def _update_workspace_path_label(self) -> None:
        text = str(self._workspace.root)
        metrics = QFontMetrics(self.workspace_path_label.font())
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, WORKSPACE_PATH_ELIDE_WIDTH)
        self.workspace_path_label.setText(elided)
        self.workspace_path_label.setToolTip(text)

    def _on_pick_workspace(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose workspace", str(self._workspace.root)
        )
        if not chosen:
            return
        self._workspace.set_root(Path(chosen))
        self._refresh_workspace_view()
        self._append_muted_line(f"Workspace: {self._workspace.root}")

    def _on_new_project(self) -> None:
        default_parent = self._workspace.root.parent
        dialog = ScaffoldDialog(default_parent, parent=self)
        if dialog.exec() != ScaffoldDialog.DialogCode.Accepted:
            return
        target = dialog.created_path
        if target is None:
            return
        template_id = dialog.template_combo.currentData()
        self._workspace.set_root(target)
        self._refresh_workspace_view()
        try:
            label = get_template(template_id).label if template_id else "project"
        except KeyError:
            label = "project"
        self._append_muted_line(f"Scaffolded {label} at {target}")

    def _checked_context_files(self) -> list[Path]:
        paths: list[Path] = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            stored = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(stored, str):
                paths.append(Path(stored))
        return paths

    def _build_prompt_with_context(self, user_prompt: str) -> tuple[str, int]:
        """Prepend file-context blocks to ``user_prompt``. Returns (prompt, count)."""
        rels = self._checked_context_files()
        attached: list[str] = []
        for rel in rels:
            try:
                content = self._workspace.read_file(rel)
            except (WorkspaceError, OSError) as exc:
                logger.warning("skipping %s: %s", rel, exc)
                continue
            if len(content.encode("utf-8")) > CONTEXT_FILE_BYTE_LIMIT:
                logger.warning("skipping %s (>100KB)", rel)
                continue
            attached.append(f"# File: {rel.as_posix()}\n{content}")
        if not attached:
            return user_prompt, 0
        joined = "\n\n".join(attached)
        return f"{joined}\n\n---\n\n{user_prompt}", len(attached)

    # ---------------------------------------------------------------- submit

    def _build_spec(self) -> ValidationSpec | None:
        """Build a :class:`ValidationSpec` from the validation widgets, or ``None``."""
        if not self.validation_group.isChecked():
            return None
        raw = self.symbols_edit.text()
        symbols = tuple(s for s in (part.strip() for part in raw.split(",")) if s)
        test_text = self.test_edit.toPlainText().strip()
        test_code = test_text if test_text else None
        return ValidationSpec(
            expected_symbols=symbols,
            test_code=test_code,
            test_timeout_seconds=TEST_TIMEOUT_SECONDS,
        )

    def _on_submit(self) -> None:
        if self._worker is not None:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        n = self.n_spin.value()
        model = self.model_combo.currentText()
        spec = self._build_spec()
        prompt, context_count = self._build_prompt_with_context(text)
        logger.info(
            "Submit: %d chars (+%d context), n=%d, model=%s, scoring=%s",
            len(text),
            context_count,
            n,
            model,
            spec is not None,
        )
        self._append_user_block(text)
        if context_count:
            self._append_muted_line(f"Attaching {context_count} file(s) as context.")
        suffix = " + scoring" if spec is not None else ""
        self._append_muted_line(f"Generating {n} candidates with {model}{suffix}...")
        self.send_button.setEnabled(False)
        self.input.clear()
        self._worker = BatchWorker(self._generator, prompt, n, model, spec=spec, parent=self)
        self._worker.batch_complete.connect(self._on_batch_complete)
        self._worker.scored_complete.connect(self._on_scored_complete)
        self._worker.batch_error.connect(self._on_batch_error)
        self._worker.start()

    def _on_batch_complete(self, batch: CandidateBatch) -> None:
        self._render_batch(batch)
        self._cleanup_worker()

    def _on_scored_complete(self, scored: ScoredBatch) -> None:
        self._render_scored_batch(scored)
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

    # ---------------------------------------------------------------- planner

    def _on_planner_commit(self, state: ConversationState) -> None:
        """Stub handler — real wiring to extractor/decomposer is the next dispatch."""
        logger.info(
            "planner commit received: session=%s model=%s turns=%d",
            state.session_id,
            state.model,
            len(state.turns),
        )
        self._append_muted_line(
            f"Planner committed: session {state.session_id[:8]} "
            f"({len(state.turns)} turn(s), model={state.model})."
        )

    # ----------------------------------------------------------------- apply

    def _on_apply_candidate(self, index: int, code: str) -> None:
        default = str(self._workspace.root / DEFAULT_APPLY_FILENAME)
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Apply candidate",
            default,
            "Python files (*.py);;All files (*)",
        )
        if not chosen:
            return
        target = Path(chosen).resolve()
        try:
            rel = target.relative_to(self._workspace.root)
        except ValueError:
            QMessageBox.warning(
                self,
                "Outside workspace",
                "Files must be saved inside the workspace.",
            )
            return
        try:
            self._workspace.write_file(rel, code)
        except (WorkspaceError, OSError) as exc:
            QMessageBox.warning(self, "Write failed", str(exc))
            return
        self._refresh_workspace_view()
        self._append_muted_line(f"Wrote candidate [{index}] to {rel.as_posix()}")

    # ----------------------------------------------------------------- render

    def _render_batch(self, batch: CandidateBatch) -> None:
        short_id = batch.batch_id[:8]
        header = (
            f"Batch {short_id}: {batch.success_count}/{batch.n_requested} "
            f"succeeded in {batch.total_wall_duration_ms:.0f}ms"
        )
        self._append_muted_line(header)
        self._append_separator()
        for candidate in batch.candidates:
            self._add_output_widget(self._make_candidate_widget(candidate))
            self._append_separator()

    def _render_scored_batch(self, scored: ScoredBatch) -> None:
        short_id = scored.batch.batch_id[:8]
        pass_count = len(scored.passing)
        pct = scored.pass_rate * 100
        header = (
            f"Batch {short_id}: {pass_count}/{scored.batch.n_requested} "
            f"passed ({pct:.0f}%) in {scored.batch.total_wall_duration_ms:.0f}ms"
        )
        self._append_muted_line(header)
        self._append_separator()
        for sc in scored.scored:
            self._add_output_widget(self._make_scored_widget(sc))
            self._append_separator()

    def _make_candidate_widget(self, candidate: Candidate) -> QWidget:
        frame = self._candidate_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.PADDING_SM // 2)

        if candidate.is_success:
            color = theme.SUCCESS
            symbol = "✓"
            code_text = candidate.text or ""
            status_html = (
                f'<span style="color: {color};">[{candidate.index}] {symbol} '
                f"{candidate.wall_duration_ms:.0f}ms</span>"
            )
            status_label = self._make_rich_label(status_html)
            layout.addWidget(status_label)
            layout.addWidget(self._make_apply_button(candidate.index, code_text))
            layout.addWidget(self._make_code_block(code_text, theme.TEXT))
        else:
            status_html = (
                f'<span style="color: {theme.ERROR};">[{candidate.index}] ✗ '
                f"{candidate.wall_duration_ms:.0f}ms</span>"
            )
            layout.addWidget(self._make_rich_label(status_html))
            error_html = (
                f'<span style="color: {theme.ERROR};">'
                f"{html.escape(candidate.error or '')}</span>"
            )
            layout.addWidget(self._make_rich_label(error_html))
        return frame

    def _make_scored_widget(self, sc: ScoredCandidate) -> QWidget:
        candidate = sc.candidate
        frame = self._candidate_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.PADDING_SM // 2)

        if not candidate.is_success:
            status_html = (
                f'<span style="color: {theme.ERROR};">[{candidate.index}] ✗ '
                f"{candidate.wall_duration_ms:.0f}ms FAIL</span>"
            )
            layout.addWidget(self._make_rich_label(status_html))
            error_html = (
                f'<span style="color: {theme.ERROR};">'
                f"{html.escape(candidate.error or '')}</span>"
            )
            layout.addWidget(self._make_rich_label(error_html))
            return frame

        if sc.validation is None:
            status_html = (
                f'<span style="color: {theme.ERROR};">[{candidate.index}] ✗ '
                "extracted=None FAIL</span>"
            )
            layout.addWidget(self._make_rich_label(status_html))
            note_html = f'<span style="color: {theme.MUTED};">no code extracted</span>'
            layout.addWidget(self._make_rich_label(note_html))
            return frame

        validation = sc.validation
        glyph_color = theme.SUCCESS if validation.passed else theme.ERROR
        glyph = "✓" if validation.passed else "✗"
        verdict = "PASS" if validation.passed else "FAIL"
        parse_token = "ok" if validation.parse_ok else "fail"
        if validation.missing_symbols:
            symbols_token = f"missing:[{','.join(validation.missing_symbols)}]"
        else:
            symbols_token = "ok"
        if validation.test_ran and validation.test_ok:
            test_token = "ok"
        elif validation.test_ran and not validation.test_ok:
            test_token = "fail"
        else:
            test_token = "skip"
        method = sc.extraction_method or ""
        status_html = (
            f'<span style="color: {theme.TEXT};">[{candidate.index}] </span>'
            f'<span style="color: {glyph_color};">{glyph}</span> '
            f'<span style="color: {theme.TEXT};">'
            f"extracted={html.escape(method)} parse={parse_token} "
            f"symbols={html.escape(symbols_token)} test={test_token} "
            "</span>"
            f'<span style="color: {glyph_color};">{verdict}</span>'
            f'<span style="color: {theme.TEXT};"> ({validation.duration_ms:.0f}ms)</span>'
        )
        layout.addWidget(self._make_rich_label(status_html))

        code = sc.extracted_code or ""
        layout.addWidget(self._make_apply_button(candidate.index, code))
        layout.addWidget(self._make_code_block(code, theme.TEXT))

        if validation.parse_error:
            truncated = validation.parse_error[:ERROR_TRUNCATE]
            err_html = (
                f'<span style="color: {theme.ERROR};">'
                f"parse_error: {html.escape(truncated)}</span>"
            )
            layout.addWidget(self._make_rich_label(err_html))
        if validation.test_error:
            truncated = validation.test_error[:ERROR_TRUNCATE]
            layout.addWidget(self._make_code_block(f"test_error: {truncated}", theme.ERROR))
        return frame

    def _make_apply_button(self, index: int, code: str) -> QPushButton:
        button = QPushButton("Apply")
        button.setFixedWidth(80)
        button.clicked.connect(lambda _checked=False, i=index, c=code: self._on_apply_candidate(i, c))
        return button

    def _make_code_block(self, text: str, color: str) -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        body = (
            f'<pre style="color: {color}; margin: 4px 0; white-space: pre-wrap;">'
            f"{html.escape(text)}"
            "</pre>"
        )
        label.setText(body)
        label.setWordWrap(True)
        label.setFont(self._mono)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _make_rich_label(self, html_text: str) -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setText(html_text)
        label.setWordWrap(True)
        label.setFont(self._mono)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _candidate_frame(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.NoFrame)
        return frame

    # --------------------------------------------------- output append helpers

    def _append_user_block(self, text: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        body = (
            f'<span style="color: {theme.ACCENT}; font-weight: 600;">&gt; </span>'
            f'<span style="color: {theme.TEXT};">{escaped}</span>'
        )
        self._add_output_widget(self._make_rich_label(body))

    def _append_styled_line(self, text: str, color: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        body = f'<span style="color: {color};">{escaped}</span>'
        self._add_output_widget(self._make_rich_label(body))

    def _append_muted_line(self, text: str) -> None:
        self._append_styled_line(text, theme.MUTED)

    def _append_separator(self) -> None:
        self._add_output_widget(self._make_hrule())

    def _add_output_widget(self, widget: QWidget) -> None:
        """Insert ``widget`` before the trailing stretch and scroll to it."""
        if self._output_empty.isVisible():
            self._output_empty.setVisible(False)
        index = max(self._output_layout.count() - 1, 0)
        self._output_layout.insertWidget(index, widget)
        self.output.ensureWidgetVisible(widget)

    # -------------------------------------------------------- test helpers

    def _output_text(self) -> str:
        """Return concatenated plain text of all output widgets (test helper)."""
        parts: list[str] = []
        for i in range(self._output_layout.count()):
            item = self._output_layout.itemAt(i)
            widget = item.widget() if item is not None else None
            if widget is self._output_empty:
                continue
            if isinstance(widget, QLabel):
                parts.append(widget.text())
        return "\n".join(parts)
