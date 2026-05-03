"""End-to-end pipeline orchestrator.

Wires :class:`PlannerStation` → :class:`LinearExecutor` (which already drives
the worker / critic loop) → :class:`IntegratorStation` and threads a single
:class:`RunState` through every stage. Backends and models are chosen at
construction time by the caller; the orchestrator itself is provider-
agnostic.

Failure policy:

- :meth:`PlannerStation.run` raises :class:`BackendError` on parse failures
  the planner could not recover from. The orchestrator does not catch it —
  there is no recovery without a plan and the user needs to see the error.
- :meth:`LinearExecutor.run` does not raise on per-slice worker failure: a
  worker that exhausts critic rounds still emits its last
  :class:`CodeArtifact`, the executor appends it, and the next slice runs.
  The orchestrator preserves that behavior so every run produces maximum
  information regardless of individual slice outcomes.
- :meth:`IntegratorStation.run` is contract-bound to never raise on a
  contract or compile failure; it returns an :class:`IntegrationResult`
  with ``success=False`` and a populated ``compile_errors`` tuple.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from aura_harness.pipeline.linear import LinearExecutor
from aura_harness.state import RunState, TaskSpec
from aura_harness.stations import IntegratorStation, PlannerStation


class Pipeline:
    """Run a :class:`TaskSpec` through planner → executor → integrator."""

    def __init__(
        self,
        planner: PlannerStation,
        executor: LinearExecutor,
        integrator: IntegratorStation,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._integrator = integrator

    @property
    def planner(self) -> PlannerStation:
        """The planner station; exposed so callers can drive stages individually."""
        return self._planner

    @property
    def executor(self) -> LinearExecutor:
        """The linear executor wrapping the worker station."""
        return self._executor

    @property
    def integrator(self) -> IntegratorStation:
        """The integrator station."""
        return self._integrator

    def run(self, task_spec: TaskSpec, workspace_path: str) -> RunState:
        """Execute the full pipeline and return the final :class:`RunState`."""
        run_id = _derive_run_id(task_spec)
        state = RunState(run_id=run_id, task=task_spec)

        plan = self._planner.run(task_spec)
        state = replace(state, plan=plan)

        state = self._executor.run(state, plan)

        integration = self._integrator.run(plan, state.artifacts, workspace_path)
        state = replace(state, integration=integration)
        return state


def _derive_run_id(task_spec: TaskSpec) -> str:
    """A run id stamped from the task id; uniqueness comes from the timestamp.

    Run ids in this orchestrator are not used for collision-resistant lookup
    — the genealogy fields on each produced value carry their own ids — so
    a task-derived run id keeps run records self-explanatory at a glance.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{task_spec.task_id}-{stamp}"
