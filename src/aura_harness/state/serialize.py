"""JSON serialization for :class:`RunState` run records.

The orchestrator writes a single ``run_record.json`` per pipeline run so a
human (or a downstream comparison tool) can reconstruct what happened
without re-running anything: task, plan, slice contracts, every artifact's
files and genealogy, every critique, and the integration result.

Why a hand-rolled serializer instead of :func:`dataclasses.asdict`:

- ``CompletionResult.raw`` is a provider-specific blob and shows up nested
  under no fields the run record actually carries — but to stay safe we
  drop ``raw`` if it sneaks in via any future field. (``RunState`` itself
  does not hold a ``CompletionResult`` today.)
- ``Path`` values must serialize to ``str`` for JSON.
- ``datetime`` values must serialize to ISO-8601 strings.
- Tuples must become lists.

Round-tripping is a non-goal here: this format is a human-readable record,
not a serialization protocol. If reading runs back becomes a need, that's
a separate dispatch.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aura_harness.state.types import RunState

_DROPPED_FIELDS: frozenset[str] = frozenset({"raw"})


def run_record_to_dict(state: RunState) -> dict[str, Any]:
    """Convert ``state`` into a JSON-serializable dict.

    Every dataclass becomes a dict, every tuple becomes a list,
    :class:`pathlib.Path` becomes its string form, :class:`datetime`
    becomes its ISO-8601 representation. Unknown / non-serializable
    values fall through to ``repr``.
    """
    encoded = _encode(state)
    if not isinstance(encoded, dict):
        raise TypeError("RunState did not encode to a dict")
    return encoded


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items() if k not in _DROPPED_FIELDS}
    if is_dataclass(value):
        out: dict[str, Any] = {}
        for field in fields(value):
            if field.name in _DROPPED_FIELDS:
                continue
            out[field.name] = _encode(getattr(value, field.name))
        return out
    return repr(value)
