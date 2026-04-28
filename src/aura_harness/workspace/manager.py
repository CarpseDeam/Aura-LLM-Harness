"""Workspace manager: list/read/write files within a sandboxed root directory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

_SKIP_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {"__pycache__", ".venv", ".git", "node_modules"}
)


class WorkspaceError(Exception):
    """Raised when a workspace path operation is rejected (e.g. root escape)."""


class WorkspaceManager:
    """Sandboxed view of a workspace directory.

    All read/write operations resolve their target path and verify it stays
    within ``root``. Listings skip dotfiles, dotdirs, and a small set of
    well-known build/cache directories.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Currently active workspace root (always resolved/absolute)."""
        return self._root

    def set_root(self, root: Path) -> None:
        """Switch to a new workspace root, creating it if missing."""
        resolved = root.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self._root = resolved
        logger.debug("workspace root set to %s", resolved)

    def list_files(self) -> list[Path]:
        """Return sorted list of files in the workspace, relative to root.

        Skips dotfiles, dotdirs, ``__pycache__``, ``.venv``, ``.git``,
        ``node_modules``.
        """
        results: list[Path] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(self._root)
            except ValueError:
                continue
            if any(self._should_skip(part) for part in rel.parts):
                continue
            results.append(rel)
        results.sort(key=lambda p: p.as_posix())
        return results

    def read_file(self, rel_path: Path) -> str:
        """Read ``rel_path`` (relative to root) as UTF-8 text."""
        target = self._safe_resolve(rel_path)
        return target.read_text(encoding="utf-8")

    def write_file(self, rel_path: Path, content: str) -> None:
        """Write ``content`` to ``rel_path`` as UTF-8, creating parents."""
        target = self._safe_resolve(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.debug("wrote %d bytes to %s", len(content), target)

    def _safe_resolve(self, rel_path: Path) -> Path:
        """Resolve ``rel_path`` against root and reject any escape."""
        candidate = (self._root / rel_path).resolve()
        if not candidate.is_relative_to(self._root):
            raise WorkspaceError(
                f"path {rel_path!s} escapes workspace root {self._root!s}"
            )
        return candidate

    @staticmethod
    def _should_skip(part: str) -> bool:
        if part.startswith("."):
            return True
        return part in _SKIP_DIR_NAMES
