"""Entry point: ``python -m aura_harness``."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from aura_harness.ui.main_window import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
