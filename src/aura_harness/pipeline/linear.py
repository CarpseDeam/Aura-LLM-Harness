"""Linear (no-topo-sort) executor: walk a :class:`Plan`'s slices in input order.

Each slice is dispatched to the configured :class:`WorkerStation`; the
returned :class:`CodeArtifact` is appended to the threaded :class:`RunState`
and the next slice is processed. The integrator hook is reserved for a
future dispatch and currently no-ops when ``integrator_station`` is ``None``.
"""
from __future__ import annotations

from dataclasses import replace

from aura_harness.state import Plan, RunState
from aura_harness.stations import WorkerStation


class LinearExecutor:
    """Run a :class:`Plan` slice-by-slice through a :class:`WorkerStation`."""

    def __init__(
        self,
        worker_station: WorkerStation,
        integrator_station: object | None = None,
    ) -> None:
        self._worker_station = worker_station
        self._integrator_station = integrator_station

    def run(self, state: RunState, plan: Plan) -> RunState:
        """Process every slice in ``plan`` and return the extended :class:`RunState`."""
        # TODO: topo-sort by Slice.depends_on once a multi-slice plan needs it.
        current = state if state.plan is plan else replace(state, plan=plan)
        for slice_ in plan.slices:
            artifact = self._worker_station.run(slice_, current)
            current = replace(current, artifacts=current.artifacts + (artifact,))

        if self._integrator_station is not None:
            # TODO: call integrator_station.run(plan, current) once IntegratorStation lands.
            pass

        return current
