"""Core bench runner.

Given a task name and a (runs, n, model) configuration, runs the existing
:class:`CandidateGenerator` against the task's spec, extracts code from each
candidate, and invokes the task's ``verify.py`` as a subprocess. Pass/fail
verdicts and metadata are written into a session folder under ``sessions/``.

When ``critic_rounds > 0`` the runner adds a reflexion loop on top: every
candidate slot that fails round 0 is reviewed by a :class:`CriticClient`,
and the resulting violations + suggestions are folded into a follow-up
prompt that the :class:`CandidateGenerator` retries. Up to ``critic_rounds``
retries happen per slot. The slot's final pass/fail is whether ANY round
passed.

The runner intentionally re-uses :func:`extract_code` from the scoring layer
rather than duplicating extraction logic.
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from aura_harness.bench.models import (
    BenchConfig,
    CandidateResult,
    RoundResult,
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
from aura_harness.critic import CriticClient, Critique, DEFAULT_CRITIC_MODEL
from aura_harness.lab.generator import CandidateGenerator
from aura_harness.lab.models import Candidate
from aura_harness.llm import OllamaClient
from aura_harness.scoring import extract_code

logger = logging.getLogger(__name__)

DEFAULT_RUNS: Final[int] = 3
DEFAULT_N: Final[int] = 3
DEFAULT_MODEL: Final[str] = "qwen2.5-coder:7b"
DEFAULT_VERIFY_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_CRITIC_ROUNDS: Final[int] = 0

_GIT_TIMEOUT_SECONDS: Final[float] = 5.0
_STDERR_PREVIEW_CHARS: Final[int] = 4000
_TASKS_SUBDIR: Final[str] = "bench/tasks"
_SESSIONS_SUBDIR: Final[str] = "sessions"
_CANDIDATE_MODULE_FILENAME: Final[str] = "candidate_module.py"
_CANDIDATE_REPORT_FILENAME: Final[str] = "report.json"
_PROGRESS_FILENAME: Final[str] = "progress.json"
_GIT_UNKNOWN: Final[str] = "unknown"
_REFLEXION_SEED_MAX: Final[int] = 2**31 - 1

_REFLEXION_PROMPT_TEMPLATE: Final[str] = (
    "{spec}\n"
    "\n"
    "---\n"
    "\n"
    "## Previous attempt\n"
    "\n"
    "You previously produced this candidate, which failed verification:\n"
    "\n"
    "```python\n"
    "{code}\n"
    "```\n"
    "\n"
    "## Reviewer feedback\n"
    "\n"
    "A reviewer identified the following contract violations:\n"
    "{violations}\n"
    "\n"
    "Suggested fixes:\n"
    "{suggestions}\n"
    "\n"
    "## Instruction\n"
    "\n"
    "Produce a corrected version of the module that addresses every "
    "violation above. Output the entire corrected module in a single "
    "fenced ```python``` block. Do not include explanations outside the "
    "code block."
)
_NO_ITEMS_PLACEHOLDER: Final[str] = "- (none provided)"


@dataclass
class _ProgressTracker:
    """Owns stdout progress lines and the ``progress.json`` file.

    Best-effort observability layer: any failure to write ``progress.json``
    is logged and swallowed so the bench keeps running.
    """

    session_dir: Path
    total_runs: int
    candidates_per_run: int
    started_at: str
    started_perf: float
    verbose: bool = False
    status: str = "running"
    current_run: int = 0
    current_candidate: int = 0
    current_round: int = 0
    current_phase: str = "idle"
    passes_so_far: int = 0
    candidates_completed: int = 0
    ratchets_accepted: int = 0
    ratchets_rejected: int = 0
    last_event: str = ""
    _cand_prefix: str = field(default="", init=False)
    _run_prefix: str = field(default="", init=False)

    @property
    def total_candidates(self) -> int:
        return self.total_runs * self.candidates_per_run

    def write(self) -> None:
        payload: dict[str, Any] = {
            "status": self.status,
            "started_at": self.started_at,
            "elapsed_seconds": round(time.perf_counter() - self.started_perf, 2),
            "current_run": self.current_run,
            "total_runs": self.total_runs,
            "current_candidate": self.current_candidate,
            "candidates_per_run": self.candidates_per_run,
            "current_round": self.current_round,
            "current_phase": self.current_phase,
            "passes_so_far": self.passes_so_far,
            "candidates_completed": self.candidates_completed,
            "total_candidates": self.total_candidates,
            "ratchets_accepted": self.ratchets_accepted,
            "ratchets_rejected": self.ratchets_rejected,
            "last_event": self.last_event,
        }
        path = self.session_dir / _PROGRESS_FILENAME
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("progress.json write failed: %s", exc)

    def session_started(self) -> None:
        self.last_event = "session started"
        self.write()

    def start_run(self, run_index: int) -> None:
        self.current_run = run_index + 1
        self.current_candidate = 0
        self.current_round = 0
        self.current_phase = "generating"
        self._run_prefix = f"[run {self.current_run}/{self.total_runs}]"
        self._cand_prefix = ""
        message = "starting batch generation..."
        self.last_event = message
        print(f"{self._run_prefix} {message}", flush=True)
        self.write()

    def start_candidate(self, candidate_index: int) -> None:
        self.current_candidate = candidate_index + 1
        self.current_round = 0
        self.current_phase = "verifying"
        self._cand_prefix = (
            f"[run {self.current_run}/{self.total_runs} "
            f"cand {self.current_candidate}/{self.candidates_per_run}]"
        )
        self.write()

    def round_zero_done(
        self, *, tests_passed: int, tests_total: int, elapsed_s: float
    ) -> None:
        self.current_round = 0
        self.current_phase = "idle"
        message = (
            f"round 0: {tests_passed}/{tests_total} tests passed "
            f"({_fmt_short(elapsed_s)})"
        )
        self.last_event = message
        print(f"{self._cand_prefix} {message}", flush=True)
        self.write()

    def start_critic(self, round_index: int) -> None:
        self.current_round = round_index
        self.current_phase = "critiquing"
        message = f"critic round {round_index}..."
        self.last_event = message
        print(f"{self._cand_prefix} {message}", flush=True)
        self.write()

    def critic_done(
        self, *, round_index: int, elapsed_s: float, violations: int
    ) -> None:
        self.current_phase = "idle"
        message = (
            f"critic round {round_index} done "
            f"({_fmt_short(elapsed_s)}, {violations} violations)"
        )
        self.last_event = message
        print(f"{self._cand_prefix} {message}", flush=True)
        self.write()

    def start_retry(self, round_index: int) -> None:
        self.current_round = round_index
        self.current_phase = "retrying"
        message = f"retry round {round_index}..."
        self.last_event = message
        print(f"{self._cand_prefix} {message}", flush=True)
        self.write()

    def retry_done(
        self,
        *,
        round_index: int,
        tests_passed: int,
        tests_total: int,
        accepted: bool,
        elapsed_s: float,
    ) -> None:
        if accepted:
            self.ratchets_accepted += 1
            verdict = "RATCHET"
        else:
            self.ratchets_rejected += 1
            verdict = "rejected"
        self.current_phase = "idle"
        message = (
            f"retry round {round_index}: {tests_passed}/{tests_total} tests passed "
            f"({verdict}) ({_fmt_short(elapsed_s)})"
        )
        self.last_event = message
        print(f"{self._cand_prefix} {message}", flush=True)
        self.write()

    def candidate_done(self, *, passed: bool) -> None:
        self.candidates_completed += 1
        if passed:
            self.passes_so_far += 1
        self.current_phase = "idle"
        self.write()

    def run_complete(
        self, *, run_passes: int, run_total: int, elapsed_s: float
    ) -> None:
        message = (
            f"complete: {run_passes}/{run_total} passed in {_fmt_long(elapsed_s)}"
        )
        self.last_event = message
        print(f"{self._run_prefix} {message}", flush=True)
        self.current_phase = "idle"
        self.write()

    def session_complete(
        self,
        *,
        passes: int,
        total: int,
        elapsed_s: float,
    ) -> None:
        pct = (passes / total * 100.0) if total else 0.0
        message = (
            f"{passes}/{total} passed ({pct:.0f}%) in {_fmt_long(elapsed_s)}, "
            f"written to {self.session_dir}"
        )
        self.last_event = message
        self.current_phase = "idle"
        self.status = "complete"
        print(f"[done] {message}", flush=True)
        self.write()

    def session_errored(self, exc: BaseException) -> None:
        self.status = "error"
        self.last_event = f"error: {type(exc).__name__}: {exc}"
        self.write()

    def unload(self, client: OllamaClient, model: str) -> None:
        client.unload_model(model)
        if self.verbose:
            print(f"[unload] {model}", flush=True)


def _fmt_short(seconds: float) -> str:
    """Format a sub-minute duration as ``"12.4s"``."""
    return f"{seconds:.1f}s"


def _fmt_long(seconds: float) -> str:
    """Format a long duration as ``"14m21s"`` or ``"1h05m12s"``."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def run_bench(
    task: str,
    runs: int = DEFAULT_RUNS,
    n: int = DEFAULT_N,
    model: str = DEFAULT_MODEL,
    *,
    timeout_seconds: float = DEFAULT_VERIFY_TIMEOUT_SECONDS,
    critic_rounds: int = DEFAULT_CRITIC_ROUNDS,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    project_root: Path | None = None,
    on_session_created: Callable[[Path], None] | None = None,
    verbose: bool = False,
) -> tuple[Path, SessionSummary]:
    """Run a bench task and return ``(session_dir, summary)``.

    Args:
        task: Task name; must match a folder under ``bench/tasks/``.
        runs: How many independent batches to generate. Each run uses a
            fresh batch (and a fresh random base seed).
        n: How many candidates per batch.
        model: Coder model name to use for candidate generation.
        timeout_seconds: Per-candidate verifier subprocess timeout.
        critic_rounds: Maximum number of reflexion rounds per failed
            candidate. ``0`` (default) disables the critic entirely and
            preserves the original single-shot runner behavior.
        critic_model: Reasoning model name used by the critic when
            ``critic_rounds > 0``.
        project_root: Project root override (mostly for tests). Defaults to
            the repo root inferred from this file's location.
        on_session_created: Called with the session folder path immediately
            after it is created on disk, before any candidates are generated.
            The CLI uses this to surface the path to the user up front.
        verbose: When ``True`` the runner prints noisy ``[unload] <model>``
            lines on every model eviction. Default ``False``.

    Returns:
        A ``(session_dir, summary)`` pair. The session folder is fully
        materialized on disk before this returns.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if n < 1:
        raise ValueError("n must be >= 1")
    if critic_rounds < 0:
        raise ValueError("critic_rounds must be >= 0")

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
        critic_rounds=critic_rounds,
        critic_model=critic_model,
        harness_git_sha=_git_sha(root),
        start_time=datetime.now(timezone.utc).isoformat(),
    )
    write_config(session_dir, config)
    if on_session_created is not None:
        on_session_created(session_dir)

    session_started_perf = time.perf_counter()
    tracker = _ProgressTracker(
        session_dir=session_dir,
        total_runs=runs,
        candidates_per_run=n,
        started_at=config.start_time,
        started_perf=session_started_perf,
        verbose=verbose,
    )
    tracker.session_started()

    client = OllamaClient(default_model=model)
    generator = CandidateGenerator(client)
    critic: CriticClient | None = (
        CriticClient(client, model=critic_model) if critic_rounds > 0 else None
    )

    run_results: list[RunResult] = []
    last_model_used: str = model
    try:
        for run_index in range(runs):
            tracker.start_run(run_index)
            run_started = time.perf_counter()
            batch = generator.generate(spec_text, n=n, model=model)
            tracker.unload(client, model)
            last_model_used = model

            slot_results: list[CandidateResult] = []
            for cand in batch.candidates:
                slot_result, slot_last_model = _resolve_candidate_slot(
                    run_index=run_index,
                    candidate=cand,
                    spec_text=spec_text,
                    verify_path=verify_path,
                    fixtures_dir=fixtures_dir,
                    timeout_seconds=timeout_seconds,
                    critic=critic,
                    critic_rounds=critic_rounds,
                    generator=generator,
                    model=model,
                    client=client,
                    tracker=tracker,
                )
                slot_results.append(slot_result)
                last_model_used = slot_last_model
            candidate_results = tuple(slot_results)

            run_duration_ms = (time.perf_counter() - run_started) * 1000.0
            run_result = RunResult(
                run_index=run_index,
                batch_id=batch.batch_id,
                candidates=candidate_results,
                run_duration_ms=run_duration_ms,
            )
            run_results.append(run_result)
            write_batch(session_dir, run_index, batch, run_result)
            run_passes = sum(1 for c in candidate_results if c.passed)
            tracker.run_complete(
                run_passes=run_passes,
                run_total=len(candidate_results),
                elapsed_s=run_duration_ms / 1000.0,
            )

        write_results(session_dir, run_results)
        summary = _build_summary(
            task=task,
            model=model,
            critic_rounds=critic_rounds,
            critic_model=critic_model,
            runs=run_results,
        )
        write_summary(session_dir, summary)
        tracker.unload(client, last_model_used)
    except BaseException as exc:
        tracker.session_errored(exc)
        raise

    tracker.session_complete(
        passes=summary.passes,
        total=summary.total_candidates,
        elapsed_s=time.perf_counter() - session_started_perf,
    )
    return session_dir, summary


def _resolve_candidate_slot(
    *,
    run_index: int,
    candidate: Candidate,
    spec_text: str,
    verify_path: Path,
    fixtures_dir: Path,
    timeout_seconds: float,
    critic: CriticClient | None,
    critic_rounds: int,
    generator: CandidateGenerator,
    model: str,
    client: OllamaClient,
    tracker: _ProgressTracker,
) -> tuple[CandidateResult, str]:
    """Run round 0 and any reflexion rounds for a single candidate slot.

    Reflexion rounds use a one-way ratchet on ``tests_passed``: a retry
    only replaces the prior-best (and feeds the next prompt) if it
    strictly improves the pass-count. Rejected retries are still kept in
    the rounds list with ``ratchet_accepted=False`` for inspection.

    Returns the resolved :class:`CandidateResult` plus the name of the last
    LLM model touched while resolving this slot. The caller uses that name
    to issue a final eviction once the run finishes, keeping VRAM clean.
    """
    tracker.start_candidate(candidate.index)
    round_zero_started = time.perf_counter()
    round_zero = _verify_candidate_round(
        round_index=0,
        candidate=candidate,
        verify_path=verify_path,
        fixtures_dir=fixtures_dir,
        timeout_seconds=timeout_seconds,
        critique=None,
        ratchet_accepted=True,
    )
    round_zero_elapsed = (
        candidate.wall_duration_ms / 1000.0
        + (time.perf_counter() - round_zero_started)
    )
    tracker.round_zero_done(
        tests_passed=round_zero.tests_passed,
        tests_total=round_zero.tests_total,
        elapsed_s=round_zero_elapsed,
    )

    rounds: list[RoundResult] = [round_zero]
    last_llm_model: str = model
    prior_best = round_zero
    prior_critique: Critique | None = None

    if critic is None or critic_rounds <= 0 or round_zero.passed:
        tracker.candidate_done(passed=round_zero.passed)
        return (
            CandidateResult(
                run_index=run_index,
                candidate_index=candidate.index,
                seed=candidate.seed,
                rounds=tuple(rounds),
            ),
            last_llm_model,
        )

    for next_round_index in range(1, critic_rounds + 1):
        if prior_best.extracted_code is None:
            # Nothing for the critic to chew on (extraction failed or
            # generation itself failed); reflexion can't recover from this.
            break

        if prior_critique is None:
            tracker.start_critic(next_round_index)
            critic_started = time.perf_counter()
            critique = critic.review(
                spec_text,
                prior_best.extracted_code,
                failures=prior_best.failures,
                verifier_stderr=prior_best.verify_stderr,
                tests_passed=prior_best.tests_passed,
                tests_total=prior_best.tests_total,
            )
            tracker.unload(client, critic.model)
            last_llm_model = critic.model
            tracker.critic_done(
                round_index=next_round_index,
                elapsed_s=time.perf_counter() - critic_started,
                violations=len(critique.violations),
            )
            if critique.passed or not critique.violations:
                # Critic sees nothing actionable; further rounds would just
                # re-prompt with empty feedback. Stop and keep the slot failed.
                break
            prior_critique = critique

        follow_up_prompt = _build_reflexion_prompt(
            spec=spec_text,
            previous_code=prior_best.extracted_code,
            critique=prior_critique,
        )
        new_seed = random.randint(0, _REFLEXION_SEED_MAX)
        tracker.start_retry(next_round_index)
        retry_started = time.perf_counter()
        retry_batch = generator.generate(
            follow_up_prompt,
            n=1,
            model=model,
            base_seed=new_seed,
        )
        tracker.unload(client, model)
        last_llm_model = model
        retry_candidate = retry_batch.candidates[0]
        next_round = _verify_candidate_round(
            round_index=next_round_index,
            candidate=retry_candidate,
            verify_path=verify_path,
            fixtures_dir=fixtures_dir,
            timeout_seconds=timeout_seconds,
            critique=prior_critique,
            ratchet_accepted=False,
        )
        accepted = next_round.tests_passed > prior_best.tests_passed
        tracker.retry_done(
            round_index=next_round_index,
            tests_passed=next_round.tests_passed,
            tests_total=next_round.tests_total,
            accepted=accepted,
            elapsed_s=time.perf_counter() - retry_started,
        )
        if accepted:
            next_round = _with_ratchet(next_round, accepted=True)
            prior_best = next_round
            # The next round needs a fresh critique against the new
            # prior-best, since the prior critique targeted older code.
            prior_critique = None
        rounds.append(next_round)
        if prior_best.passed:
            break

    slot_result = CandidateResult(
        run_index=run_index,
        candidate_index=candidate.index,
        seed=candidate.seed,
        rounds=tuple(rounds),
    )
    tracker.candidate_done(passed=slot_result.passed)
    return slot_result, last_llm_model


def _with_ratchet(round_result: RoundResult, *, accepted: bool) -> RoundResult:
    """Return a copy of ``round_result`` with ``ratchet_accepted`` flipped."""
    return RoundResult(
        round_index=round_result.round_index,
        seed=round_result.seed,
        passed=round_result.passed,
        tests_passed=round_result.tests_passed,
        tests_total=round_result.tests_total,
        failures=round_result.failures,
        ratchet_accepted=accepted,
        extraction_method=round_result.extraction_method,
        candidate_error=round_result.candidate_error,
        verify_stderr=round_result.verify_stderr,
        latency_ms=round_result.latency_ms,
        raw_text=round_result.raw_text,
        extracted_code=round_result.extracted_code,
        critique=round_result.critique,
    )


def _verify_candidate_round(
    *,
    round_index: int,
    candidate: Candidate,
    verify_path: Path,
    fixtures_dir: Path,
    timeout_seconds: float,
    critique: Critique | None,
    ratchet_accepted: bool,
) -> RoundResult:
    """Extract code from a candidate and run the verifier subprocess."""
    if not candidate.is_success:
        return RoundResult(
            round_index=round_index,
            seed=candidate.seed,
            passed=False,
            tests_passed=0,
            tests_total=1,
            failures=("generation failed",),
            ratchet_accepted=ratchet_accepted,
            extraction_method=None,
            candidate_error=candidate.error,
            verify_stderr=None,
            latency_ms=candidate.wall_duration_ms,
            raw_text=None,
            extracted_code=None,
            critique=critique,
        )

    raw_text = candidate.text or ""
    code, method = extract_code(raw_text)
    if code is None:
        return RoundResult(
            round_index=round_index,
            seed=candidate.seed,
            passed=False,
            tests_passed=0,
            tests_total=1,
            failures=("extraction failed",),
            ratchet_accepted=ratchet_accepted,
            extraction_method=None,
            candidate_error=None,
            verify_stderr="extraction failed: no code block found",
            latency_ms=candidate.wall_duration_ms,
            raw_text=raw_text,
            extracted_code=None,
            critique=critique,
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
            return RoundResult(
                round_index=round_index,
                seed=candidate.seed,
                passed=False,
                tests_passed=0,
                tests_total=1,
                failures=(f"verify timed out after {timeout_seconds}s",),
                ratchet_accepted=ratchet_accepted,
                extraction_method=method,
                candidate_error=None,
                verify_stderr=f"verify timed out after {timeout_seconds}s",
                latency_ms=candidate.wall_duration_ms,
                raw_text=raw_text,
                extracted_code=code,
                critique=critique,
            )
        gradient = _read_gradient_report(report_path, returncode=proc.returncode)

    passed = gradient.passed
    stderr_preview = None if passed else (proc.stderr or "")[:_STDERR_PREVIEW_CHARS]
    return RoundResult(
        round_index=round_index,
        seed=candidate.seed,
        passed=passed,
        tests_passed=gradient.tests_passed,
        tests_total=gradient.tests_total,
        failures=gradient.failures,
        ratchet_accepted=ratchet_accepted,
        extraction_method=method,
        candidate_error=None,
        verify_stderr=stderr_preview,
        latency_ms=candidate.wall_duration_ms,
        raw_text=raw_text,
        extracted_code=code,
        critique=critique,
    )


@dataclass(frozen=True)
class _Gradient:
    """The pass-count signal extracted from a verifier's report file."""

    tests_passed: int
    tests_total: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.tests_total > 0 and self.tests_passed == self.tests_total


def _read_gradient_report(report_path: Path, *, returncode: int) -> _Gradient:
    """Read the verifier's structured report, falling back to binary signal.

    Verify scripts emit a JSON report with ``tests_passed``, ``tests_total``,
    and ``failures``. If the file is missing, malformed, or has the wrong
    shape (e.g., a verifier that hasn't been updated to the new protocol),
    we synthesize a 1/1 pass / 0/1 fail from the subprocess returncode so
    the ratchet still has a meaningful signal.
    """
    fallback = (
        _Gradient(tests_passed=1, tests_total=1, failures=())
        if returncode == 0
        else _Gradient(
            tests_passed=0,
            tests_total=1,
            failures=("verify exited non-zero",),
        )
    )
    if not report_path.is_file():
        return fallback
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    tests_passed = payload.get("tests_passed")
    tests_total = payload.get("tests_total")
    raw_failures = payload.get("failures", [])
    if not isinstance(tests_passed, int) or not isinstance(tests_total, int):
        return fallback
    if tests_total < 0 or tests_passed < 0 or tests_passed > tests_total:
        return fallback
    failures: tuple[str, ...]
    if isinstance(raw_failures, list) and all(isinstance(f, str) for f in raw_failures):
        failures = tuple(raw_failures)
    else:
        failures = ()
    return _Gradient(
        tests_passed=tests_passed,
        tests_total=tests_total,
        failures=failures,
    )


def _build_reflexion_prompt(
    *, spec: str, previous_code: str, critique: Critique
) -> str:
    """Compose the follow-up prompt sent to the coder for a reflexion round."""
    violations = _format_bullets(critique.violations)
    suggestions = _format_bullets(critique.suggestions)
    return _REFLEXION_PROMPT_TEMPLATE.format(
        spec=spec,
        code=previous_code,
        violations=violations,
        suggestions=suggestions,
    )


def _format_bullets(items: tuple[str, ...]) -> str:
    if not items:
        return _NO_ITEMS_PLACEHOLDER
    return "\n".join(f"- {item}" for item in items)


def _build_summary(
    *,
    task: str,
    model: str,
    critic_rounds: int,
    critic_model: str,
    runs: list[RunResult],
) -> SessionSummary:
    """Aggregate the per-run results into a single summary."""
    total = sum(len(r.candidates) for r in runs)
    passes = sum(1 for r in runs for c in r.candidates if c.passed)
    one_shot_passes = sum(
        1 for r in runs for c in r.candidates if c.round_zero_passed
    )
    pass_rate = passes / total if total else 0.0
    one_shot_pass_rate = one_shot_passes / total if total else 0.0
    mean_latency = (
        sum(c.total_latency_ms for r in runs for c in r.candidates) / total
        if total
        else 0.0
    )
    kept_rounds = [
        c.prior_best
        for r in runs
        for c in r.candidates
        if c.prior_best is not None
    ]
    if kept_rounds:
        mean_tests_passed = sum(rr.tests_passed for rr in kept_rounds) / len(kept_rounds)
        mean_tests_total = sum(rr.tests_total for rr in kept_rounds) / len(kept_rounds)
    else:
        mean_tests_passed = 0.0
        mean_tests_total = 0.0
    return SessionSummary(
        task=task,
        model=model,
        critic_rounds=critic_rounds,
        critic_model=critic_model,
        total_candidates=total,
        passes=passes,
        pass_rate=pass_rate,
        one_shot_passes=one_shot_passes,
        one_shot_pass_rate=one_shot_pass_rate,
        mean_latency_ms=mean_latency,
        mean_tests_passed=mean_tests_passed,
        mean_tests_total=mean_tests_total,
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
