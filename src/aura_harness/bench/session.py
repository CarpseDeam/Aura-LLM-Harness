"""Bench session output management.

A "session" is a single invocation of the bench runner. Each session gets its
own folder under ``sessions/`` named ``run-<NNN>-<task>-<timestamp>`` where
``NNN`` is a zero-padded incrementing index based on existing folders. The
folder layout is:

    sessions/run-NNN-task-ts/
        config.json              # the BenchConfig
        batches/
            batch-000.jsonl      # one JSONL record per candidate slot in run 0,
                                 # with all reflexion rounds nested under "rounds"
            batch-001.jsonl      # ... etc
        results.json             # all RunResults
        summary.json             # the SessionSummary

This module owns folder creation and serialization. It does not run the bench;
that's :mod:`aura_harness.bench.runner`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
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
from aura_harness.lab.models import Candidate, CandidateBatch

_SESSION_DIR_RE: Final[re.Pattern[str]] = re.compile(r"^run-(\d+)-")
_TIMESTAMP_FMT: Final[str] = "%Y%m%dT%H%M%SZ"
_INDEX_WIDTH: Final[int] = 3
_BATCH_FILENAME_FMT: Final[str] = "batch-{idx:03d}.jsonl"


def create_session_dir(task: str, sessions_root: Path) -> Path:
    """Create and return the folder for a new bench session.

    The folder is ``sessions_root / "run-<NNN>-<task>-<timestamp>"`` with
    a ``batches/`` subfolder pre-created.
    """
    sessions_root.mkdir(parents=True, exist_ok=True)
    next_idx = _next_session_index(sessions_root)
    timestamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)
    folder = sessions_root / f"run-{next_idx:0{_INDEX_WIDTH}d}-{task}-{timestamp}"
    folder.mkdir()
    (folder / "batches").mkdir()
    return folder


def _next_session_index(sessions_root: Path) -> int:
    """Compute the next session index by scanning sibling folders."""
    max_idx = -1
    for entry in sessions_root.iterdir():
        if not entry.is_dir():
            continue
        m = _SESSION_DIR_RE.match(entry.name)
        if m is None:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def write_config(session_dir: Path, config: BenchConfig) -> None:
    """Write the session :class:`BenchConfig` to ``config.json``."""
    path = session_dir / "config.json"
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def write_batch(
    session_dir: Path,
    run_index: int,
    batch: CandidateBatch,
    run_result: RunResult,
) -> None:
    """Write the per-slot raw output + per-round outcomes for one run.

    One JSONL record per candidate slot, in batch order. Each record carries
    the round-0 raw model text plus a ``rounds`` array containing every
    reflexion round (its raw text, extraction outcome, verifier verdict, and
    the critique that triggered it).
    """
    path = session_dir / "batches" / _BATCH_FILENAME_FMT.format(idx=run_index)
    with path.open("w", encoding="utf-8") as fh:
        for cand, cresult in zip(batch.candidates, run_result.candidates):
            record = _build_batch_record(cand, cresult)
            fh.write(json.dumps(record) + "\n")


def write_results(session_dir: Path, runs: list[RunResult]) -> None:
    """Write the per-slot, per-round verdicts to ``results.json``."""
    payload = {"runs": [asdict(r) for r in runs]}
    path = session_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(session_dir: Path, summary: SessionSummary) -> None:
    """Write the aggregate :class:`SessionSummary` to ``summary.json``."""
    path = session_dir / "summary.json"
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def _build_batch_record(cand: Candidate, cresult: CandidateResult) -> dict[str, Any]:
    """Build the JSONL record for one candidate slot in a batch file."""
    return {
        "run_index": cresult.run_index,
        "candidate_index": cand.index,
        "seed": cand.seed,
        "is_success": cand.is_success,
        "round_zero_raw_text": cand.text,
        "round_zero_error": cand.error,
        "round_zero_wall_duration_ms": cand.wall_duration_ms,
        "passed": cresult.passed,
        "round_zero_passed": cresult.round_zero_passed,
        "round_count": len(cresult.rounds),
        "rounds": [_round_to_record(r) for r in cresult.rounds],
    }


def _round_to_record(round_result: RoundResult) -> dict[str, Any]:
    """Serialize one :class:`RoundResult` for the batch JSONL."""
    critique_payload = (
        asdict(round_result.critique) if round_result.critique is not None else None
    )
    return {
        "round_index": round_result.round_index,
        "seed": round_result.seed,
        "passed": round_result.passed,
        "tests_passed": round_result.tests_passed,
        "tests_total": round_result.tests_total,
        "failures": list(round_result.failures),
        "ratchet_accepted": round_result.ratchet_accepted,
        "extraction_method": round_result.extraction_method,
        "candidate_error": round_result.candidate_error,
        "verify_stderr": round_result.verify_stderr,
        "latency_ms": round_result.latency_ms,
        "raw_text": round_result.raw_text,
        "extracted_code": round_result.extracted_code,
        "critique": critique_payload,
    }
