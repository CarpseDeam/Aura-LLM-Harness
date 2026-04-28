"""Bench session output management.

A "session" is a single invocation of the bench runner. Each session gets its
own folder under ``sessions/`` named ``run-<NNN>-<task>-<timestamp>`` where
``NNN`` is a zero-padded incrementing index based on existing folders. The
folder layout is:

    sessions/run-NNN-task-ts/
        config.json              # the BenchConfig
        batches/
            batch-000.jsonl      # one JSONL record per candidate in run 0
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
from typing import Final

from aura_harness.bench.models import (
    BenchConfig,
    RunResult,
    SessionSummary,
)
from aura_harness.lab.models import CandidateBatch

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
    """Write the per-candidate raw output + extraction outcome for one run.

    One JSONL record per candidate, in batch order. Each record carries the
    raw model text (so we can re-extract / re-verify offline later) plus the
    extraction method and pass/fail recorded by the runner.
    """
    path = session_dir / "batches" / _BATCH_FILENAME_FMT.format(idx=run_index)
    with path.open("w", encoding="utf-8") as fh:
        for cand, cresult in zip(batch.candidates, run_result.candidates):
            record = {
                "run_index": run_index,
                "candidate_index": cand.index,
                "seed": cand.seed,
                "is_success": cand.is_success,
                "raw_text": cand.text,
                "error": cand.error,
                "wall_duration_ms": cand.wall_duration_ms,
                "extraction_method": cresult.extraction_method,
                "passed": cresult.passed,
            }
            fh.write(json.dumps(record) + "\n")


def write_results(session_dir: Path, runs: list[RunResult]) -> None:
    """Write the per-candidate verdicts to ``results.json``."""
    payload = {"runs": [asdict(r) for r in runs]}
    path = session_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(session_dir: Path, summary: SessionSummary) -> None:
    """Write the aggregate :class:`SessionSummary` to ``summary.json``."""
    path = session_dir / "summary.json"
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
