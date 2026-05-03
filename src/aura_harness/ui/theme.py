"""Dark theme constants for the Aura LLM Harness UI."""
from __future__ import annotations

from typing import Final

BACKGROUND: Final[str] = "#1e1e1e"
SURFACE: Final[str] = "#2c2c2c"
SURFACE_RAISED: Final[str] = "#363636"
TEXT: Final[str] = "#e0e0e0"
TEXT_DIM: Final[str] = "#b8b8b8"
ACCENT: Final[str] = "#64B5F6"
ACCENT_HOVER: Final[str] = "#90CAF9"
ACCENT_PRESSED: Final[str] = "#42A5F5"
BORDER: Final[str] = "#3a3a3a"
PLACEHOLDER: Final[str] = "#7a7a7a"
SUCCESS: Final[str] = "#66BB6A"
ERROR: Final[str] = "#E57373"
MUTED: Final[str] = "#888888"

MONO_FAMILY_PRIMARY: Final[str] = "JetBrains Mono"
MONO_FAMILY_FALLBACKS: Final[tuple[str, ...]] = ("Consolas", "Menlo", "Monaco", "Courier New")
FONT_SIZE_PT: Final[int] = 11
FONT_SIZE_HEADER_PT: Final[int] = FONT_SIZE_PT + 1

PADDING_SM: Final[int] = 6
PADDING_MD: Final[int] = 10
PADDING_LG: Final[int] = 16
RADIUS: Final[int] = 6

INPUT_MIN_HEIGHT: Final[int] = 96

STYLESHEET: Final[str] = f"""
QMainWindow, QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}

QPlainTextEdit, QTextEdit, QLineEdit {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: {PADDING_SM}px {PADDING_MD}px;
    selection-background-color: {ACCENT};
    selection-color: {BACKGROUND};
}}
QPlainTextEdit:focus, QTextEdit:focus, QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox, QSpinBox {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: 4px {PADDING_SM}px;
    min-height: 18px;
}}
QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {BACKGROUND};
    border: 1px solid {BORDER};
    outline: 0;
}}

QListWidget {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: 2px;
    outline: 0;
}}
QListWidget::item {{
    padding: 3px 6px;
}}
QListWidget::item:selected {{
    background-color: {BORDER};
    color: {TEXT};
}}

QCheckBox {{
    color: {TEXT};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {SURFACE};
}}
QCheckBox::indicator:hover {{
    border-color: {PLACEHOLDER};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* Default (secondary) buttons — flat with subtle border */
QPushButton {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: 5px {PADDING_MD}px;
    font-weight: 500;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {SURFACE_RAISED};
    border-color: {PLACEHOLDER};
}}
QPushButton:pressed {{
    background-color: {BACKGROUND};
}}
QPushButton:disabled {{
    color: {PLACEHOLDER};
    background-color: {SURFACE};
    border-color: {BORDER};
}}

/* Primary action buttons (Run Planner / Run Worker) — filled accent */
QPushButton[primary="true"] {{
    background-color: {ACCENT};
    color: {BACKGROUND};
    border: 1px solid {ACCENT};
    font-weight: 600;
    padding: 5px {PADDING_LG}px;
}}
QPushButton[primary="true"]:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[primary="true"]:pressed {{
    background-color: {ACCENT_PRESSED};
    border-color: {ACCENT_PRESSED};
}}
QPushButton[primary="true"]:disabled {{
    background-color: {BORDER};
    color: {PLACEHOLDER};
    border-color: {BORDER};
}}

/* Section header label */
QLabel[role="section"] {{
    color: {TEXT};
    font-weight: 700;
    font-size: {FONT_SIZE_HEADER_PT}pt;
}}

/* Inline label next to dropdowns / spinners */
QLabel[role="inline"] {{
    color: {TEXT_DIM};
    font-weight: 500;
}}

/* Subtle path text in workspace header */
QLabel[role="path"] {{
    color: {MUTED};
}}

/* Empty-state placeholder text */
QLabel[role="empty"] {{
    color: {PLACEHOLDER};
    font-style: italic;
}}

/* Thin neutral horizontal rule between sections */
QFrame[role="hline"] {{
    background-color: {BORDER};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}

QScrollBar:vertical {{
    background: {BACKGROUND};
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {PLACEHOLDER};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BACKGROUND};
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {PLACEHOLDER};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
"""
