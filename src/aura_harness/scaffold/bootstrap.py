"""Scaffold a new project on disk from a :class:`Template`."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

from aura_harness.scaffold.templates import get_template

logger = logging.getLogger(__name__)

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-zA-Z0-9_-]+")


def scaffold(template_id: str, name: str, parent_dir: Path) -> Path:
    """Materialize ``template_id`` as ``parent_dir / name`` on disk.

    Args:
        template_id: ID of a registered template (e.g. ``"blank"``).
        name: Project name; must match ``[a-zA-Z0-9_-]+`` and be non-empty.
        parent_dir: Directory under which the new project is created.

    Returns:
        The absolute path to the created project directory.

    Raises:
        ValueError: ``name`` is empty or contains disallowed characters.
        FileExistsError: target directory already exists and is non-empty.
        OSError: filesystem error while creating directories or files.
    """
    if not name or not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"invalid project name {name!r}: must match [a-zA-Z0-9_-]+ and be non-empty"
        )
    template = get_template(template_id)
    target = (parent_dir / name).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"target directory {target!s} already exists and is non-empty")
    target.mkdir(parents=True, exist_ok=True)
    for rel_path, content in template.files.items():
        rendered_path = rel_path.format(name=name)
        rendered_content = content.format(name=name)
        file_path = target / rendered_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(rendered_content, encoding="utf-8")
    logger.info("scaffolded %s at %s", template.id, target)
    return target
