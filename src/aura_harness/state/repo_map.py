"""Build a :class:`RepoMap` by AST-walking a workspace.

The planner consumes a :class:`RepoMap` to know what symbols already
exist in the workspace and which modules are imported where. This module
is the only place in :mod:`aura_harness.state` that contains behavior;
everything else is types.

Why ``ast`` and not tree-sitter: stdlib, no extra dependency, and
sufficient for Python-only workspaces. Multi-language support is a
future concern that will earn its own builder.
"""
from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

from aura_harness.state.types import FileSymbols, RepoMap

logger = logging.getLogger(__name__)

_SKIP_DIR_NAMES: frozenset[str] = frozenset({"__pycache__", ".venv", "node_modules", ".git"})


def build_repo_map(workspace_path: Path) -> RepoMap:
    """Walk ``workspace_path`` and return a :class:`RepoMap`.

    Walks recursively, collecting every ``.py`` file. Hidden directories
    (anything whose name begins with ``"."``), ``__pycache__``, ``.venv``,
    ``node_modules`` and ``.git`` are skipped.

    Each file is parsed with :func:`ast.parse`. Files with
    :class:`SyntaxError` are skipped (a partial work-in-progress should
    not break the planner) and a debug-level log message is emitted.

    The returned :class:`RepoMap` is deterministic: ``files`` is sorted
    by path, and within each :class:`FileSymbols` both ``symbols`` and
    ``imports`` are sorted alphabetically.

    Args:
        workspace_path: Root directory to walk. Paths in the resulting
            :class:`FileSymbols` are made relative to this root.

    Returns:
        A populated :class:`RepoMap`. Empty when no parseable Python
        files are found.
    """
    collected: list[FileSymbols] = []

    for dirpath, dirnames, filenames in os.walk(workspace_path):
        dirnames[:] = sorted(
            d for d in dirnames if not _should_skip_dir(d)
        )
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            file_path = Path(dirpath) / filename
            symbols = _parse_file(file_path)
            if symbols is None:
                continue
            relative = file_path.relative_to(workspace_path)
            collected.append(
                FileSymbols(
                    path=relative,
                    symbols=symbols.symbols,
                    imports=symbols.imports,
                )
            )

    collected.sort(key=lambda f: f.path.as_posix())
    return RepoMap(files=tuple(collected))


def _should_skip_dir(name: str) -> bool:
    """Whether to prune a directory during the walk."""
    if name.startswith("."):
        return True
    return name in _SKIP_DIR_NAMES


def _parse_file(file_path: Path) -> FileSymbols | None:
    """Parse one ``.py`` file. Returns ``None`` on read or parse failure.

    The returned :class:`FileSymbols` carries the absolute path; the
    caller is responsible for rewriting it relative to the workspace.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("repo_map: skip unreadable %s: %s", file_path, exc)
        return None

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        logger.debug("repo_map: skip unparseable %s: %s", file_path, exc)
        return None

    symbols = tuple(sorted(_extract_symbols(tree)))
    imports = tuple(sorted(_extract_imports(tree)))
    return FileSymbols(path=file_path, symbols=symbols, imports=imports)


def _extract_symbols(tree: ast.Module) -> list[str]:
    """Top-level ``def`` / ``async def`` / ``class`` names."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def _extract_imports(tree: ast.Module) -> list[str]:
    """Module-scope imported names, in the dotted convention.

    - ``import foo.bar`` → ``"foo.bar"``.
    - ``from foo import bar`` → ``"foo.bar"``;
      ``from foo import bar, baz`` → ``"foo.bar"``, ``"foo.baz"``.
    - Relative imports keep their leading dots so they remain
      recognizable: ``from . import x`` → ``".x"``;
      ``from .pkg import y`` → ``".pkg.y"``.
    """
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            module = node.module or ""
            base = f"{prefix}{module}"
            for alias in node.names:
                names.append(f"{base}.{alias.name}" if base else alias.name)
    return names
