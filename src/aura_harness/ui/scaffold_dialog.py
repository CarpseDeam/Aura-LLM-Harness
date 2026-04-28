"""Modal dialog for creating a new workspace from a scaffold template."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aura_harness.scaffold import TEMPLATES, scaffold
from aura_harness.ui import theme

logger = logging.getLogger(__name__)

NAME_PLACEHOLDER: Final[str] = "my_project"


class ScaffoldDialog(QDialog):
    """Collect a template + name + parent dir, then scaffold on Create."""

    def __init__(self, default_parent_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setModal(True)
        self._created_path: Path | None = None

        self.template_combo: QComboBox = QComboBox()
        for template in TEMPLATES:
            display = f"{template.label} — {template.description}"
            self.template_combo.addItem(display, template.id)

        self.name_edit: QLineEdit = QLineEdit()
        self.name_edit.setPlaceholderText(NAME_PLACEHOLDER)

        self.parent_edit: QLineEdit = QLineEdit(str(default_parent_dir))
        pick_button = QPushButton("...")
        pick_button.setFixedWidth(36)
        pick_button.setMinimumWidth(36)
        pick_button.clicked.connect(self._on_pick_parent)

        parent_row = QHBoxLayout()
        parent_row.setSpacing(theme.PADDING_SM)
        parent_row.addWidget(self.parent_edit, 1)
        parent_row.addWidget(pick_button)

        form = QFormLayout()
        form.addRow("Template:", self.template_combo)
        form.addRow("Name:", self.name_edit)
        form.addRow("Parent directory:", parent_row)

        self.error_label: QLabel = QLabel()
        self.error_label.setStyleSheet(f"color: {theme.ERROR};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Create")
        buttons.accepted.connect(self._on_create)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.PADDING_LG, theme.PADDING_LG, theme.PADDING_LG, theme.PADDING_LG
        )
        layout.setSpacing(theme.PADDING_MD)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)

    @property
    def created_path(self) -> Path | None:
        """Resolved path of the newly scaffolded project, once Create succeeded."""
        return self._created_path

    def _on_pick_parent(self) -> None:
        current = self.parent_edit.text().strip() or str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "Choose parent directory", current)
        if chosen:
            self.parent_edit.setText(chosen)

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(True)

    def _on_create(self) -> None:
        self.error_label.setVisible(False)
        template_id = self.template_combo.currentData()
        name = self.name_edit.text().strip()
        parent_text = self.parent_edit.text().strip()

        if not template_id:
            self._show_error("Pick a template.")
            return
        if not name:
            self._show_error("Enter a project name.")
            return
        if not parent_text:
            self._show_error("Choose a parent directory.")
            return

        parent_dir = Path(parent_text)
        try:
            target = scaffold(template_id, name, parent_dir)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        except FileExistsError as exc:
            QMessageBox.warning(self, "Already exists", str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(self, "Scaffold failed", str(exc))
            return

        self._created_path = target
        self.accept()
