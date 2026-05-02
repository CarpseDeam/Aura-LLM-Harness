"""Entry point: ``python -m aura_harness``."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from aura_harness.backend import LocalOllamaBackend
from aura_harness.lab import CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.planner import PlannerClient
from aura_harness.ui.main_window import MainWindow
from aura_harness.workspace import WorkspaceManager

_DEFAULT_WORKSPACE_README: str = "Aura LLM Harness workspace\n"


def _bootstrap_workspace(root: Path) -> WorkspaceManager:
    workspace = WorkspaceManager(root)
    readme = workspace.root / "README.md"
    if not readme.exists():
        workspace.write_file(Path("README.md"), _DEFAULT_WORKSPACE_README)
    return workspace


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = QApplication(sys.argv)
    client = OllamaClient()
    backend = LocalOllamaBackend(client)
    generator = CandidateGenerator(backend)
    planner = PlannerClient(client)
    workspace = _bootstrap_workspace(Path.cwd() / "workspace")
    window = MainWindow(client, generator, workspace, planner)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
