"""Typed, immutable run-state schema for the harness.

See ``docs/STATE_SCHEMA.md`` for the design rationale. This package
exports the type set every station reads from and writes to, plus the
single behavior helper :func:`build_repo_map`.
"""
from __future__ import annotations

from aura_harness.state.repo_map import build_repo_map
from aura_harness.state.types import (
    CodeArtifact,
    CritiqueReport,
    FileSymbols,
    FileWrite,
    Plan,
    RepoMap,
    RunState,
    Slice,
    SliceContract,
    TaskSpec,
)

__all__ = [
    "CodeArtifact",
    "CritiqueReport",
    "FileSymbols",
    "FileWrite",
    "Plan",
    "RepoMap",
    "RunState",
    "Slice",
    "SliceContract",
    "TaskSpec",
    "build_repo_map",
]
