"""End-to-end smoke CLI for :class:`PlannerStation`.

Used by hand to verify that a backend + model + prompt template can
produce a parseable :class:`Plan` end-to-end before any downstream
station consumes one. Hits a real provider — no mocking.

Examples (run from a shell with ``DEEPSEEK_API_KEY`` exported when using
the cloud backend):

    python -m aura_harness.stations.planner_smoke --backend cloud \\
        --model deepseek-v4-flash \\
        --task-file bench/tasks/pygame_smallgame/spec.md

    python -m aura_harness.stations.planner_smoke --backend local \\
        --model gpt-oss:20b \\
        --task "Build a CLI calculator with add/sub/mul/div."
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Final

from aura_harness.backend import (
    Backend,
    BackendError,
    CloudHTTPBackend,
    LocalOllamaBackend,
)
from aura_harness.backend.cloud_constants import DEEPSEEK_BASE_URL
from aura_harness.llm import OllamaClient
from aura_harness.state import Plan, TaskSpec
from aura_harness.stations.planner import PlannerStation

_DEEPSEEK_API_KEY_ENV: Final[str] = "DEEPSEEK_API_KEY"


def _build_backend(kind: str) -> Backend:
    if kind == "local":
        return LocalOllamaBackend(OllamaClient())
    if kind == "cloud":
        api_key = os.environ.get(_DEEPSEEK_API_KEY_ENV)
        if not api_key:
            raise SystemExit(
                f"missing {_DEEPSEEK_API_KEY_ENV} environment variable"
            )
        return CloudHTTPBackend(
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key,
            default_model="deepseek-v4-flash",
            name="deepseek",
        )
    raise SystemExit(f"unknown backend: {kind}")


def _resolve_task_text(args: argparse.Namespace) -> str:
    if args.task and args.task_file:
        raise SystemExit("pass --task or --task-file, not both")
    if args.task:
        return args.task
    if args.task_file:
        path = Path(args.task_file)
        if not path.is_file():
            raise SystemExit(f"task file not found: {path}")
        return path.read_text(encoding="utf-8")
    raise SystemExit("one of --task or --task-file is required")


def _print_plan(plan: Plan, task_spec: TaskSpec) -> None:
    print("=" * 72)
    print(f"task_id:    {task_spec.task_id}")
    print(f"plan_id:    {plan.plan_id}")
    print(f"backend:    {plan.produced_by_backend}")
    print(f"station:    {plan.produced_by_station}")
    print(f"produced:   {plan.produced_at.isoformat()}")
    print(f"input_ref:  {plan.input_ref}")
    print()
    print("Description:")
    print(_indent(task_spec.description.strip(), 2))
    print()
    if plan.rationale:
        print("Rationale:")
        print(_indent(plan.rationale, 2))
        print()
    print(f"Slices ({len(plan.slices)}):")
    for index, slice_ in enumerate(plan.slices, start=1):
        print(f"  [{index}] id={slice_.slice_id}")
        print(f"      target_path:       {slice_.target_path}")
        print(f"      depends_on:        {list(slice_.depends_on)}")
        print(f"      expected_symbols:  {list(slice_.contract.expected_symbols)}")
        print(f"      forbidden_imports: {list(slice_.contract.forbidden_imports)}")
        print("      description:")
        print(_indent(slice_.description, 8))
    print("=" * 72)


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PlannerStation smoke CLI.")
    parser.add_argument("--backend", choices=("local", "cloud"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", help="Inline task description.")
    parser.add_argument("--task-file", help="Path to a file containing the task description.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    args = parser.parse_args(argv)

    description = _resolve_task_text(args)
    backend = _build_backend(args.backend)

    task_spec = TaskSpec(
        task_id=f"smoke-{uuid.uuid4().hex[:8]}",
        description=description,
        workspace_path=Path.cwd(),
    )

    station = PlannerStation(
        backend=backend,
        model=args.model,
        temperature=args.temperature,
        max_parse_retries=args.max_parse_retries,
    )

    try:
        plan = station.run(task_spec)
    except BackendError as exc:
        print(f"\nplanner error: {exc}", file=sys.stderr)
        return 1

    _print_plan(plan, task_spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
