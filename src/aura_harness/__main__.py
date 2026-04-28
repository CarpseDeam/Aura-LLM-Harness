"""Entry point: ``python -m aura_harness``."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from aura_harness.lab import CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.ui.main_window import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = QApplication(sys.argv)
    client = OllamaClient()
    generator = CandidateGenerator(client)
    window = MainWindow(client, generator)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
