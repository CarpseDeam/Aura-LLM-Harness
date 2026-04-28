"""Project scaffold template definitions and registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Template:
    """A project scaffold template.

    ``files`` maps a relative path to file content. Both keys and values may
    contain ``{name}`` placeholders that are substituted with the project name.
    """

    id: str
    label: str
    description: str
    files: dict[str, str]


_BLANK_README: Final[str] = "# {name}\n\nAura LLM Harness workspace.\n"

_PYTHON_CLI_PYPROJECT: Final[str] = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
description = "A Python CLI scaffolded by Aura LLM Harness."
requires-python = ">=3.10"

[project.scripts]
{name} = "{name}.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]
"""

_PYTHON_CLI_MAIN: Final[str] = '''\
"""Entry point for the {name} CLI."""
from __future__ import annotations


def main() -> None:
    print("hello from {name}")


if __name__ == "__main__":
    main()
'''

_PYTHON_CLI_TEST: Final[str] = '''\
"""Smoke test for {name}."""
from __future__ import annotations


def test_import() -> None:
    import {name}  # noqa: F401
'''

_PYTHON_CLI_GITIGNORE: Final[str] = """\
.venv/
__pycache__/
*.pyc
build/
dist/
*.egg-info/
.pytest_cache/
.coverage
"""

_PYTHON_CLI_README: Final[str] = "# {name}\n\nA Python CLI scaffolded by Aura LLM Harness.\n"


BLANK_TEMPLATE: Final[Template] = Template(
    id="blank",
    label="Blank workspace",
    description="An empty workspace with just a README.",
    files={"README.md": _BLANK_README},
)

PYTHON_CLI_TEMPLATE: Final[Template] = Template(
    id="python_cli",
    label="Python CLI",
    description="A minimal Python CLI with pyproject, src layout, and a smoke test.",
    files={
        "pyproject.toml": _PYTHON_CLI_PYPROJECT,
        "src/{name}/__init__.py": "",
        "src/{name}/__main__.py": _PYTHON_CLI_MAIN,
        "tests/test_smoke.py": _PYTHON_CLI_TEST,
        ".gitignore": _PYTHON_CLI_GITIGNORE,
        "README.md": _PYTHON_CLI_README,
    },
)


TEMPLATES: Final[tuple[Template, ...]] = (BLANK_TEMPLATE, PYTHON_CLI_TEMPLATE)

_TEMPLATE_BY_ID: Final[dict[str, Template]] = {t.id: t for t in TEMPLATES}


def get_template(template_id: str) -> Template:
    """Return the template with ``template_id``, or raise ``KeyError``."""
    return _TEMPLATE_BY_ID[template_id]
