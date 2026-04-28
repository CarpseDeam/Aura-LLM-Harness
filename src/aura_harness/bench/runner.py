"""Core bench runner.

Given a task name and a (runs, n, model) configuration, runs the existing
:class:`CandidateGenerator` against the task's spec, extracts code from each
candidate, and invokes the task's ``verify.py`` as a subprocess. Pass/fail
verdicts and metadata are written into a session folder under ``sessions/``.

The runner intentionally re-uses :func:`extract_code` from the scoring layer
rather than duplicating extraction logic.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from aura_harness.bench.models import (
    BenchConfig,
    CandidateResult,
    RunResult,
    SessionSummary,
)
from aura_harness.bench.session import (
    create_session_dir,
    write_batch,
    write_config,
    write_results,
    write_summary,
)
from aura_harness.lab.generator import CandidateGenerator
from aura_harness.lab.models import Candidate
from aura_harness.llm import OllamaClient
from aura_harness.scoring import extract_code

logger = logging.getLogger(__name__)

DEFAULT_RUNS: Final[int] = 3
DEFAULT_N: Final[int] = 3
DEFAULT_MODEL: Final[str] = "qwen2.5-coder:7b"
DEFAULT_VERIFY_TIMEOUT_SECONDS: Final[float] = 30.0

_GIT_TIMEOUT_SECONDS: Final[float] = 5.0
_STDERR_PREVIEW_CHARS: Final[int] = 4000
_TASKS_SUBDIR: Final[str] = "bench/tasks"
_SESSIONS_SUBDIR: Final[str] = "sessions"
_CANDIDATE_MODULE_FILENAME: Final[str] = "candidate_module.py"
_CANDIDATE_REPORT_FILENAME: Final[str] = "report.json"
_GIT_UNKNOWN: Final[str] = "unknown"


def run_bench(
    task: str,
    runs: int = DEFAULT_RUNS,
    n: int = DEFAULT_N,
    model: str = DEFAULT_MODEL,
    *,
    timeout_seconds: float = DEFAULT_VERIFY_TIMEOUT_SECONDS,
    project_root: Path | None = None,
    on_session_created: Callable[[Path], None] | None = None,
) -> tuple[Path, SessionSummary]:
    """Run a bench task and return ``(session_dir, summary)``.

    Args:
        task: Task name; must match a folder under ``bench/tasks/``.
        runs: How many independent batches to generate. Each run uses a
            fresh batch (and a fresh random base seed).
        n: How many candidates per batch.
        model: Ollama model name to use.
        timeout_seconds: Per-candidate verifier subprocess timeout.
        project_root: Project root override (mostly for tests). Defaults to
            the repo root inferred from this file's location.
        on_session_created: Called with the session folder path immediately
            after it is created on disk, before any candidates are generated.
            The CLI uses this to surface the path to the user up front.

    Returns:
        A ``(session_dir, summary)`` pair. The session folder is fully
        materialized on disk before this returns.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if n < 1:
        raise ValueError("n must be >= 1")

    root = project_root if project_root is not None else _project_root()
    task_dir = root / _TASKS_SUBDIR / task
    spec_path = task_dir / "spec.md"
    verify_path = task_dir / "verify.py"
    fixtures_dir = task_dir / "fixtures"

    if not spec_path.is_file():
        raise FileNotFoundError(f"spec not found for task {task!r}: {spec_path}")
    if not verify_path.is_file():
        raise FileNotFoundError(f"verify.py not found for task {task!r}: {verify_path}")
    if not fixtures_dir.is_dir():
        raise FileNotFoundError(f"fixtures dir not found for task {task!r}: {fixtures_dir}")

    spec_text = spec_path.read_text(encoding="utf-8")

    sessions_root = root / _SESSIONS_SUBDIR
    session_dir = create_session_dir(task, sessions_root)
    config = BenchConfig(
        task=task,
        runs=runs,
        n=n,
        model=model,
        timeout_seconds=timeout_seconds,
        harness_git_sha=_git_sha(root),
        start_time=datetime.now(timezone.utc).isoformat(),
    )
    write_config(session_dir, config)
    if on_session_created is not None:
        on_session_created(session_dir)

    client = OllamaClient(default_model=model)
    generator = CandidateGenerator(client)

    run_results: list[RunResult] = []
    for run_index in range(runs):
        run_started = time.perf_counter()
        batch = generator.generate(spec_text, n=n, model=model)
        candidate_results = tuple(
            _verify_candidate(
                run_index=run_index,
                candidate=cand,
                verify_path=verify_path,
                fixtures_dir=fixtures_dir,
                timeout_seconds=timeout_seconds,
            )
            for cand in batch.candidates
        )
        run_duration_ms = (time.perf_counter() - run_started) * 1000.0
        run_result = RunResult(
            run_index=run_index,
            batch_id=batch.batch_id,
            candidates=candidate_results,
            run_duration_ms=run_duration_ms,
        )
        run_results.append(run_result)
        write_batch(session_dir, run_index, batch, run_result)

    write_results(session_dir, run_results)
    summary = _build_summary(task=task, model=model, runs=run_results)
    write_summary(session_dir, summary)
    return session_dir, summary


def _verify_candidate(
    *,
    run_index: int,
    candidate: Candidate,
    verify_path: Path,
    fixtures_dir: Path,
    timeout_seconds: float,
) -> CandidateResult:
    """Extract code from a candidate and run the verifier subprocess."""
    if not candidate.is_success:
        return CandidateResult(
            run_index=run_index,
            candidate_index=candidate.index,
            seed=candidate.seed,
            passed=False,
            extraction_method=None,
            candidate_error=candidate.error,
            verify_stderr=None,
            latency_ms=candidate.wall_duration_ms,
        )

    code, method = extract_code(candidate.text or "")
    if code is None:
        return CandidateResult(
            run_index=run_index,
            candidate_index=candidate.index,
            seed=candidate.seed,
            passed=False,
            extraction_method=None,
            candidate_error=None,
            verify_stderr="extraction failed: no code block found",
            latency_ms=candidate.wall_duration_ms,
        )

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
            return CandidateResult(
                run_index=run_index,
                candidate_index=candidate.index,
                seed=candidate.seed,
                passed=False,
                extraction_method=method,
                candidate_error=None,
                verify_stderr=f"verify timed out after {timeout_seconds}s",
                latency_ms=candidate.wall_duration_ms,
            )

    passed = proc.returncode == 0
    stderr_preview = None if passed else (proc.stderr or "")[:_STDERR_PREVIEW_CHARS]
    return CandidateResult(
        run_index=run_index,
        candidate_index=candidate.index,
        seed=candidate.seed,
        passed=passed,
        extraction_method=method,
        candidate_error=None,
        verify_stderr=stderr_preview,
        latency_ms=candidate.wall_duration_ms,
    )


def _build_summary(
    *, task: str, model: str, runs: list[RunResult]
) -> SessionSummary:
    """Aggregate the per-run results into a single summary."""
    total = sum(len(r.candidates) for r in runs)
    passes = sum(1 for r in runs for c in r.candidates if c.passed)
    pass_rate = passes / total if total else 0.0
    mean_latency = (
        sum(c.latency_ms for r in runs for c in r.candidates) / total if total else 0.0
    )
    return SessionSummary(
        task=task,
        model=model,
        total_candidates=total,
        passes=passes,
        pass_rate=pass_rate,
        mean_latency_ms=mean_latency,
    )


def _project_root() -> Path:
    """Infer the repo root from this file's location.

    Layout: ``<root>/src/aura_harness/bench/runner.py`` → parents[3] is root.
    """
    return Path(__file__).resolve().parents[3]


def _git_sha(root: Path) -> str:
    """Return ``git rev-parse HEAD`` at ``root``, or ``"unknown"``."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("git rev-parse failed: %s", exc)
        return _GIT_UNKNOWN
    return proc.stdout.strip() or _GIT_UNKNOWN
