"""Project scaffold layer: create new workspaces from templates."""
from __future__ import annotations

from aura_harness.scaffold.bootstrap import scaffold
from aura_harness.scaffold.templates import TEMPLATES, Template, get_template

__all__ = ["TEMPLATES", "Template", "get_template", "scaffold"]
