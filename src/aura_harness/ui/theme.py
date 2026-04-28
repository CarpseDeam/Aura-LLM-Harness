"""Dark theme constants for the Aura LLM Harness UI."""
from __future__ import annotations

from typing import Final

BACKGROUND: Final[str] = "#1e1e1e"
SURFACE: Final[str] = "#2c2c2c"
TEXT: Final[str] = "#e0e0e0"
ACCENT: Final[str] = "#64B5F6"
ACCENT_HOVER: Final[str] = "#90CAF9"
ACCENT_PRESSED: Final[str] = "#42A5F5"
BORDER: Final[str] = "#3a3a3a"
PLACEHOLDER: Final[str] = "#7a7a7a"

MONO_FAMILY_PRIMARY: Final[str] = "JetBrains Mono"
MONO_FAMILY_FALLBACKS: Final[tuple[str, ...]] = ("Consolas", "Menlo", "Monaco", "Courier New")
FONT_SIZE_PT: Final[int] = 11

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
QTextEdit, QPlainTextEdit {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: {PADDING_MD}px;
    selection-background-color: {ACCENT};
    selection-color: {BACKGROUND};
}}
QPlainTextEdit[placeholderText] {{
    color: {TEXT};
}}
QPushButton {{
    background-color: {ACCENT};
    color: {BACKGROUND};
    border: none;
    border-radius: {RADIUS}px;
    padding: {PADDING_SM}px {PADDING_LG}px;
    font-weight: 600;
    min-width: 80px;
}}
QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
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
"""
