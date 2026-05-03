"""End-to-end smoke CLI for :class:`IntegratorStation`.

Exercises the integrator in isolation against an in-memory
:class:`Plan` and inline-stub :class:`CodeArtifact` set, with no
upstream planner or worker. The full planner -> worker -> integrator
chain is dispatch 7's concern; this CLI only proves that
:class:`IntegratorStation` writes files and compile-checks correctly.

Two cases are baked in:

- ``--case valid``: a three-slice pygame stub plan whose stub source
  is syntactically valid Python. Expected outcome: ``success=True``,
  three written files, no compile errors.
- ``--case broken``: the same plan, except the engine slice's stub
  contains a deliberate syntax error. Expected outcome:
  ``success=False`` with one ``SyntaxError`` entry in
  ``compile_errors`` and the README slice still written.

Examples:

    python -m aura_harness.stations.integrator_smoke --case valid \\
        --workspace ./_smoke_ws

    python -m aura_harness.stations.integrator_smoke --case broken \\
        --workspace ./_smoke_ws_broken
"""
from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from aura_harness.backend import Backend, CompletionResult
from aura_harness.state import (
    CodeArtifact,
    FileWrite,
    IntegrationResult,
    Plan,
    Slice,
    SliceContract,
)
from aura_harness.stations.integrator import IntegratorStation

_VALID_ENGINE: Final[str] = '''"""Stub engine module."""
from __future__ import annotations


class Engine:
    def __init__(self) -> None:
        self.tick = 0

    def step(self) -> None:
        self.tick += 1
'''

_BROKEN_ENGINE: Final[str] = '''"""Stub engine with deliberate syntax error."""
class Engine
    def __init__(self) -> None:
        self.tick = 0
'''

_VALID_MAIN: Final[str] = '''"""Stub entry point."""
from __future__ import annotations

from game.engine import Engine


def main() -> int:
    engine = Engine()
    engine.step()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_VALID_README: Final[str] = """# Pygame Smoke Stub

Run with `python main.py`. This is a smoke-test stub, not a real game.
"""


class _NullBackend:
    """Backend stub used purely for genealogy stamping in this smoke CLI.

    The integrator never calls ``chat`` or ``chat_stream``; the backend
    exists so :class:`IntegratorStation`'s base class can stamp
    ``produced_by_backend`` on the result.
    """

    @property
    def name(self) -> str:
        return "null-smoke"

    @property
    def default_model(self) -> str:
        return "n/a"

    def chat(self, *args: object, **kwargs: object) -> CompletionResult:
        raise RuntimeError("integrator must not call backend.chat")

    def chat_stream(self, *args: object, **kwargs: object):  # noqa: ANN202
        raise RuntimeError("integrator must not call backend.chat_stream")


def _build_plan() -> Plan:
    slices = (
        Slice(
            slice_id="engine",
            description="Pygame engine stub.",
            target_files=(Path("game/engine.py"),),
            depends_on=(),
            contract=SliceContract(
                expected_symbols=("Engine",),
                forbidden_imports=(),
            ),
            target_path="game/engine.py",
        ),
        Slice(
            slice_id="main",
            description="Entry point stub.",
            target_files=(Path("main.py"),),
            depends_on=("engine",),
            contract=SliceContract(
                expected_symbols=("main",),
                forbidden_imports=(),
            ),
            target_path="main.py",
        ),
        Slice(
            slice_id="readme",
            description="README stub.",
            target_files=(Path("README.md"),),
            depends_on=(),
            contract=SliceContract(),
            target_path="README.md",
        ),
    )
    return Plan(
        plan_id=f"smoke-plan-{uuid.uuid4().hex[:8]}",
        slices=slices,
        rationale="Three-slice stub for integrator smoke runs.",
        produced_by_station="planner",
        produced_by_backend="null-smoke",
        produced_at=datetime.now(timezone.utc),
        seed=None,
        input_ref="smoke-task",
    )


def _build_artifacts(plan: Plan, *, case: str) -> tuple[CodeArtifact, ...]:
    sources: dict[str, str] = {
        "engine": _BROKEN_ENGINE if case == "broken" else _VALID_ENGINE,
        "main": _VALID_MAIN,
        "readme": _VALID_README,
    }
    artifacts: list[CodeArtifact] = []
    for slice_ in plan.slices:
        artifacts.append(
            CodeArtifact(
                artifact_id=uuid.uuid4().hex,
                slice_id=slice_.slice_id,
                files=(
                    FileWrite(
                        path=Path(slice_.target_path or slice_.slice_id),
                        content=sources[slice_.slice_id],
                    ),
                ),
                parent_artifact_ids=(),
                critic_round=0,
                produced_by_station="worker",
                produced_by_backend="null-smoke",
                produced_at=datetime.now(timezone.utc),
                seed=None,
                input_ref=plan.plan_id,
            )
        )
    return tuple(artifacts)


def _print_result(result: IntegrationResult) -> None:
    print("=" * 72)
    print(f"integration_id:    {result.integration_id}")
    print(f"success:           {result.success}")
    print(f"workspace_path:    {result.workspace_path}")
    print(f"station:           {result.produced_by_station}")
    print(f"backend:           {result.produced_by_backend}")
    print(f"produced_at:       {result.produced_at.isoformat()}")
    print(f"input_ref:         {result.input_ref}")
    print(f"written_files ({len(result.written_files)}):")
    for path in result.written_files:
        print(f"  - {path}")
    print(f"compile_errors ({len(result.compile_errors)}):")
    for err in result.compile_errors:
        suffix = f" (line {err.line})" if err.line is not None else ""
        target = err.file_path or "<plan>"
        print(f"  - [{err.error_type}] {target}{suffix}: {err.message}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IntegratorStation smoke CLI.")
    parser.add_argument(
        "--case",
        choices=("valid", "broken"),
        default="valid",
        help="Which stub set to integrate.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Output workspace directory. Will be created if missing.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the workspace directory before integrating.",
    )
    args = parser.parse_args(argv)

    workspace_path = Path(args.workspace).resolve()
    if args.clean and workspace_path.exists():
        shutil.rmtree(workspace_path)

    plan = _build_plan()
    artifacts = _build_artifacts(plan, case=args.case)

    backend: Backend = _NullBackend()
    station = IntegratorStation(backend=backend)
    result = station.run(plan, artifacts, str(workspace_path))

    _print_result(result)

    expected_success = args.case == "valid"
    if result.success != expected_success:
        print(
            f"\nunexpected success={result.success} for case={args.case}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
