"""CLI entry: ``python -m aura_harness.pipeline --task <name> --runs N --n M ...``.

Stands the new station / executor pipeline up against the same bench tasks
the legacy ``aura_harness.bench`` runner consumes. For each run a fresh
``RunState`` is built from the task's ``spec.md``, the executor produces a
``CodeArtifact`` for the single slice, and the artifact is scored by the
task's ``verify.py`` subprocess so results are directly comparable to the
legacy runner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from aura_harness.backend import LocalOllamaBackend
from aura_harness.critic import CriticClient, DEFAULT_CRITIC_MODEL
from aura_harness.lab.generator import CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.pipeline.linear import LinearExecutor
from aura_harness.state import (
    CodeArtifact,
    Plan,
    RunState,
    Slice,
    SliceContract,
    TaskSpec,
)
from aura_harness.stations import CriticStation, WorkerStation

_DEFAULT_RUNS: Final[int] = 3
_DEFAULT_N: Final[int] = 3
_DEFAULT_MODEL: Final[str] = "qwen2.5-coder:7b"
_DEFAULT_CRITIC_ROUNDS: Final[int] = 0
_DEFAULT_VERIFY_TIMEOUT: Final[float] = 30.0
_TASKS_SUBDIR: Final[str] = "bench/tasks"
_CANDIDATE_MODULE_FILENAME: Final[str] = "candidate_module.py"
_CANDIDATE_REPORT_FILENAME: Final[str] = "report.json"
_EXIT_USAGE_ERROR: Final[int] = 2

_TASK_EXPECTED_SYMBOLS: Final[dict[str, tuple[str, ...]]] = {
    "duplicate_finder": ("find_duplicates",),
}


def _ensure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aura_harness.pipeline",
        description="Run a bench task through the station / executor pipeline.",
    )
    p.add_argument("--task", required=True, help="Task name under bench/tasks/.")
    p.add_argument("--runs", type=int, default=_DEFAULT_RUNS)
    p.add_argument("--n", type=int, default=_DEFAULT_N)
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--critic-rounds", type=int, default=_DEFAULT_CRITIC_ROUNDS)
    p.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL)
    p.add_argument("--timeout", type=float, default=_DEFAULT_VERIFY_TIMEOUT)
    return p


def _project_root() -> Path:
    """Layout: ``<root>/src/aura_harness/pipeline/__main__.py`` → parents[3] is root."""
    return Path(__file__).resolve().parents[3]


def _build_plan(task: str, spec_text: str, task_id: str) -> Plan:
    """Wrap the task's prose spec in a single-slice Plan."""
    expected = _TASK_EXPECTED_SYMBOLS.get(task, ())
    contract = SliceContract(expected_symbols=expected, forbidden_imports=())
    slice_ = Slice(
        slice_id=f"{task}-slice-0",
        description=spec_text,
        target_files=(Path(_CANDIDATE_MODULE_FILENAME),),
        depends_on=(),
        contract=contract,
    )
    return Plan(
        plan_id=uuid.uuid4().hex,
        slices=(slice_,),
        rationale=f"single-slice plan synthesized from {task}/spec.md",
        produced_by_station="manual",
        produced_by_backend="manual",
        produced_at=datetime.now(timezone.utc),
        seed=None,
        input_ref=task_id,
    )


def _score_artifact(
    artifact: CodeArtifact,
    *,
    verify_path: Path,
    fixtures_dir: Path,
    timeout_seconds: float,
) -> tuple[int, int, tuple[str, ...]]:
    """Run ``verify.py`` against the artifact's code; return ``(passed, total, failures)``."""
    code = artifact.files[0].content if artifact.files else ""
    if not code.strip():
        return (0, 1, ("artifact had no code",))

    with tempfile.TemporaryDirectory() as tmp:
        candidate_path = Path(tmp) / _CANDIDATE_MODULE_FILENAME
        candidate_path.write_text(code, encoding="utf-8")
        report_path = Path(tmp) / _CANDIDATE_REPORT_FILENAME
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(verify_path),
                    str(candidate_path),
                    str(fixtures_dir),
                    str(report_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return (0, 1, (f"verify timed out after {timeout_seconds}s",))

        if not report_path.is_file():
            failure = "verify exited non-zero" if proc.returncode != 0 else "no report"
            return (0, 1, (failure,))
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return (0, 1, ("verify report unreadable",))

    if not isinstance(payload, dict):
        return (0, 1, ("verify report malformed",))
    tests_passed = payload.get("tests_passed")
    tests_total = payload.get("tests_total")
    raw_failures = payload.get("failures", [])
    if not isinstance(tests_passed, int) or not isinstance(tests_total, int):
        return (0, 1, ("verify report missing counts",))
    failures: tuple[str, ...] = (
        tuple(raw_failures)
        if isinstance(raw_failures, list)
        and all(isinstance(f, str) for f in raw_failures)
        else ()
    )
    return (tests_passed, tests_total, failures)


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = _build_parser().parse_args(argv)
    if args.runs < 1 or args.n < 1:
        print("error: --runs and --n must both be >= 1", file=sys.stderr)
        return _EXIT_USAGE_ERROR
    if args.critic_rounds < 0:
        print("error: --critic-rounds must be >= 0", file=sys.stderr)
        return _EXIT_USAGE_ERROR

    root = _project_root()
    task_dir = root / _TASKS_SUBDIR / args.task
    spec_path = task_dir / "spec.md"
    verify_path = task_dir / "verify.py"
    fixtures_dir = task_dir / "fixtures"
    for required in (spec_path, verify_path, fixtures_dir):
        if not required.exists():
            print(f"error: missing {required}", file=sys.stderr)
            return _EXIT_USAGE_ERROR

    spec_text = spec_path.read_text(encoding="utf-8")
    workspace_path = task_dir
    task_id = f"{args.task}-task"
    task_spec = TaskSpec(
        task_id=task_id,
        description=spec_text,
        workspace_path=workspace_path,
    )

    client = OllamaClient(default_model=args.model)
    backend = LocalOllamaBackend(client)
    candidate_generator = CandidateGenerator(backend)
    critic_client = CriticClient(backend, model=args.critic_model)
    critic_station = CriticStation(backend, critic_client)
    worker_station = WorkerStation(
        backend=backend,
        candidate_generator=candidate_generator,
        critic_station=critic_station,
        max_critic_rounds=args.critic_rounds,
        n=args.n,
        model=args.model,
    )
    executor = LinearExecutor(worker_station=worker_station)

    runs_passed = 0
    print(
        f"pipeline: task={args.task} runs={args.runs} n={args.n} "
        f"model={args.model} critic_rounds={args.critic_rounds}",
        flush=True,
    )
    for run_index in range(args.runs):
        plan = _build_plan(args.task, spec_text, task_id)
        run_state = RunState(
            run_id=uuid.uuid4().hex,
            task=task_spec,
            plan=plan,
        )
        final_state = executor.run(run_state, plan)
        artifact = final_state.artifacts[-1]
        tests_passed, tests_total, failures = _score_artifact(
            artifact,
            verify_path=verify_path,
            fixtures_dir=fixtures_dir,
            timeout_seconds=args.timeout,
        )
        passed = tests_total > 0 and tests_passed == tests_total
        if passed:
            runs_passed += 1
        verdict = "PASS" if passed else "FAIL"
        failure_summary = (
            f" failures={list(failures)}" if failures and not passed else ""
        )
        print(
            f"[run {run_index + 1}/{args.runs}] {verdict} "
            f"{tests_passed}/{tests_total}{failure_summary}",
            flush=True,
        )

    pct = (runs_passed / args.runs * 100.0) if args.runs else 0.0
    print(
        f"Summary: {runs_passed}/{args.runs} runs passed ({pct:.0f}%)",
        flush=True,
    )
    return 0 if runs_passed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
