"""Common base for harness stations.

A :class:`Station` wraps a :class:`Backend` and stamps every value it emits
with the five-field genealogy that ``aura_harness.state.types`` uses to
trace provenance back through the pipeline. Subclasses define their own
``run`` signature — the shape varies (a worker takes a slice and a state,
a critic takes an artifact and a slice) so the base does not try to
enforce one. It exists to share the genealogy-stamping plumbing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aura_harness.backend import Backend


class Station:
    """Base class providing genealogy stamping for station outputs."""

    def __init__(self, name: str, backend: Backend) -> None:
        self._name = name
        self._backend = backend

    @property
    def name(self) -> str:
        """Station name used for ``produced_by_station`` genealogy."""
        return self._name

    @property
    def backend(self) -> Backend:
        """The backend this station dispatches against."""
        return self._backend

    def _genealogy(
        self,
        *,
        seed: int | None = None,
        input_ref: str = "",
    ) -> dict[str, Any]:
        """Return the five genealogy fields ready to spread into a constructor."""
        return {
            "produced_by_station": self._name,
            "produced_by_backend": self._backend.name,
            "produced_at": datetime.now(timezone.utc),
            "seed": seed,
            "input_ref": input_ref,
        }
