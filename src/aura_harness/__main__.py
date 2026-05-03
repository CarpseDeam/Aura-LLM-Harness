"""Top-level CLI: ``python -m aura_harness --task-file <path> --workspace <path>``.

Runs the full planner → worker (with critic) → integrator pipeline against
configurable backends per stage, writes the integrated workspace to disk,
and saves a ``run_record.json`` containing the entire :class:`RunState` for
later inspection. Cloud planner + local worker is the default configuration
because it matches the project thesis: a strong remote planner pairs with
cheap local execution.

The :class:`Pipeline` orchestrator is the canonical programmatic entry; this
CLI drives the same three stages directly so it can narrate per-stage
progress without baking print statements into the orchestrator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

from aura_harness.backend import (
    Backend,
    BackendError,
    CloudHTTPBackend,
    LocalOllamaBackend,
)
from aura_harness.backend.cloud_constants import DEEPSEEK_BASE_URL, DEEPSEEK_FLASH
from aura_harness.critic import CriticClient, DEFAULT_CRITIC_MODEL
from aura_harness.lab.generator import CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.pipeline.linear import LinearExecutor
from aura_harness.pipeline.orchestrator import Pipeline
from aura_harness.state import Plan, RunState, Slice, TaskSpec
from aura_harness.state.serialize import run_record_to_dict
from aura_harness.stations import (
    CriticStation,
    IntegratorStation,
    PlannerStation,
    WorkerStation,
)

_DEEPSEEK_API_KEY_ENV: Final[str] = "DEEPSEEK_API_KEY"
_RUN_RECORD_FILENAME: Final[str] = "run_record.json"

_DEFAULT_PLANNER_BACKEND: Final[str] = "cloud"
_DEFAULT_PLANNER_MODEL: Final[str] = DEEPSEEK_FLASH
_DEFAULT_WORKER_BACKEND: Final[str] = "local"
_DEFAULT_WORKER_MODEL: Final[str] = "qwen2.5-coder:14b"
_DEFAULT_CRITIC_ROUNDS: Final[int] = 2
_DEFAULT_N: Final[int] = 3

_EXIT_USAGE_ERROR: Final[int] = 2
_EXIT_CONFIG_ERROR: Final[int] = 3


def _ensure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aura_harness",
        description=(
            "Run the full Aura pipeline (planner → worker/critic → integrator) "
            "against a TaskSpec file and write the integrated code plus a run "
            "record to a workspace."
        ),
    )
    p.add_argument(
        "--task-file",
        required=True,
        help="Path to a TaskSpec spec file (e.g. bench/tasks/<name>/spec.md).",
    )
    p.add_argument(
        "--workspace",
        required=True,
        help="Output directory for integrated code and run_record.json. Created if missing.",
    )
    p.add_argument(
        "--planner-backend",
        choices=("cloud", "local"),
        default=_DEFAULT_PLANNER_BACKEND,
    )
    p.add_argument("--planner-model", default=_DEFAULT_PLANNER_MODEL)
    p.add_argument(
        "--worker-backend",
        choices=("local", "cloud"),
        default=_DEFAULT_WORKER_BACKEND,
    )
    p.add_argument("--worker-model", default=_DEFAULT_WORKER_MODEL)
    p.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL)
    p.add_argument(
        "--critic-rounds",
        type=int,
        default=_DEFAULT_CRITIC_ROUNDS,
        help="Maximum critic-driven retry rounds per slice.",
    )
    p.add_argument(
        "-n",
        "--candidates",
        type=int,
        default=_DEFAULT_N,
        help="Candidate count per worker generation pass.",
    )
    return p


def _build_backend(kind: str, *, default_model: str) -> Backend:
    """Construct a backend; cloud requires DEEPSEEK_API_KEY in the environment."""
    if kind == "local":
        return LocalOllamaBackend(OllamaClient(default_model=default_model))
    if kind == "cloud":
        api_key = os.environ.get(_DEEPSEEK_API_KEY_ENV)
        if not api_key:
            raise SystemExit(
                f"error: {_DEEPSEEK_API_KEY_ENV} is required for the cloud backend "
                "but is not set"
            )
        return CloudHTTPBackend(
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key,
            default_model=default_model,
            name="deepseek",
        )
    raise SystemExit(f"unknown backend: {kind}")


def _read_task_spec(task_file: Path, workspace_path: Path) -> TaskSpec:
    if not task_file.is_file():
        raise SystemExit(f"error: task file not found: {task_file}")
    description = task_file.read_text(encoding="utf-8")
    task_id = task_file.parent.name or task_file.stem
    return TaskSpec(
        task_id=task_id,
        description=description,
        workspace_path=workspace_path,
    )


def _slice_target(slice_: Slice) -> str:
    if slice_.target_path:
        return slice_.target_path
    if slice_.target_files:
        return str(slice_.target_files[0])
    return "<unknown>"


def _artifact_passed_summary(artifact) -> bool:
    """Best-effort 'did the slice come out clean?' for the worker progress line.

    WorkerStation only returns the final artifact for a slice and discards
    intermediate critique reports, so the CLI cannot inspect the actual
    last critique. Heuristic: a non-empty artifact that did not consume any
    critic-retry rounds is considered passing; one that exhausted retries
    (``critic_round > 0``) is flagged so the operator notices a slice that
    may still be broken even though the run continued.
    """
    if not artifact.files or not artifact.files[0].content.strip():
        return False
    return artifact.critic_round == 0


def _run_executor_with_progress(
    executor: LinearExecutor,
    state: RunState,
    plan: Plan,
) -> RunState:
    """Replay the executor's loop with a per-slice progress line."""
    current = state if state.plan is plan else replace(state, plan=plan)
    total = len(plan.slices)
    for index, slice_ in enumerate(plan.slices, start=1):
        print(
            f"[worker {index}/{total}] slice={slice_.slice_id} "
            f"target={_slice_target(slice_)}",
            flush=True,
        )
        artifact = executor.worker_station.run(slice_, current)
        current = replace(current, artifacts=current.artifacts + (artifact,))
        passed = _artifact_passed_summary(artifact)
        print(
            f"[worker {index}/{total}] done "
            f"critic_rounds_used={artifact.critic_round} passed={passed}",
            flush=True,
        )
    return current


def _print_compile_errors(state: RunState) -> None:
    """Print each compile error on its own line; no-op when there are none."""
    integration = state.integration
    if integration is None or not integration.compile_errors:
        return
    print("[integrator] compile errors:", flush=True)
    for err in integration.compile_errors:
        line_part = f":{err.line}" if err.line is not None else ""
        target = err.file_path or "<plan>"
        print(
            f"  - [{err.error_type}] {target}{line_part}: {err.message}",
            flush=True,
        )


def _write_run_record(state: RunState, path: Path) -> None:
    payload = run_record_to_dict(state)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = _build_parser().parse_args(argv)
    if args.candidates < 1:
        print("error: --candidates must be >= 1", file=sys.stderr)
        return _EXIT_USAGE_ERROR
    if args.critic_rounds < 0:
        print("error: --critic-rounds must be >= 0", file=sys.stderr)
        return _EXIT_USAGE_ERROR

    workspace_path = Path(args.workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    task_file = Path(args.task_file).resolve()
    task_spec = _read_task_spec(task_file, workspace_path)

    try:
        planner_backend = _build_backend(
            args.planner_backend, default_model=args.planner_model
        )
        worker_backend = _build_backend(
            args.worker_backend, default_model=args.worker_model
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_CONFIG_ERROR

    candidate_generator = CandidateGenerator(worker_backend)
    critic_client = CriticClient(worker_backend, model=args.critic_model)
    critic_station = CriticStation(worker_backend, critic_client)
    worker_station = WorkerStation(
        backend=worker_backend,
        candidate_generator=candidate_generator,
        critic_station=critic_station,
        max_critic_rounds=args.critic_rounds,
        n=args.candidates,
        model=args.worker_model,
    )
    integrator_station = IntegratorStation(backend=worker_backend)
    planner_station = PlannerStation(
        backend=planner_backend,
        model=args.planner_model,
    )
    executor = LinearExecutor(worker_station=worker_station)
    pipeline = Pipeline(
        planner=planner_station,
        executor=executor,
        integrator=integrator_station,
    )

    print(
        f"[planner] running on {planner_backend.name}/{args.planner_model}...",
        flush=True,
    )
    try:
        plan = pipeline.planner.run(task_spec)
    except BackendError as exc:
        print(f"\nplanner error: {exc}", file=sys.stderr)
        return 1
    print(
        f"[planner] plan with {len(plan.slices)} slices: "
        + ", ".join(s.slice_id for s in plan.slices),
        flush=True,
    )

    state = RunState(run_id=task_spec.task_id, task=task_spec, plan=plan)
    state = _run_executor_with_progress(pipeline.executor, state, plan)

    print(
        f"[integrator] writing {len(state.artifacts)} files to {workspace_path}...",
        flush=True,
    )
    integration = pipeline.integrator.run(plan, state.artifacts, str(workspace_path))
    state = replace(state, integration=integration)
    print(
        f"[integrator] success={integration.success} "
        f"compile_errors={len(integration.compile_errors)}",
        flush=True,
    )

    record_path = workspace_path / _RUN_RECORD_FILENAME
    _write_run_record(state, record_path)
    _print_compile_errors(state)
    print(f"[done] run_record at {record_path}", flush=True)
    return 0 if integration.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
