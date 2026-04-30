"""CLI entry: ``python -m aura_harness.bench --task <name> --runs N --n M --model M``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from aura_harness.bench.models import SessionSummary
from aura_harness.bench.runner import (
    DEFAULT_CRITIC_ROUNDS,
    DEFAULT_MODEL,
    DEFAULT_N,
    DEFAULT_RUNS,
    DEFAULT_VERIFY_TIMEOUT_SECONDS,
    run_bench,
)
from aura_harness.critic import DEFAULT_CRITIC_MODEL

_EXIT_USAGE_ERROR: Final[int] = 2


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout to UTF-8 so non-ASCII glyphs render on Windows."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aura_harness.bench",
        description="Run a bench task against the candidate generator and score the results.",
    )
    p.add_argument(
        "--task", required=True,
        help="Task name; must match a folder under bench/tasks/.",
    )
    p.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Number of independent batches (default: {DEFAULT_RUNS}).",
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Candidates per batch (default: {DEFAULT_N}).",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Coder model name (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--timeout", type=float, default=DEFAULT_VERIFY_TIMEOUT_SECONDS,
        help=f"Per-candidate verify subprocess timeout in seconds "
             f"(default: {DEFAULT_VERIFY_TIMEOUT_SECONDS}).",
    )
    p.add_argument(
        "--critic-rounds", type=int, default=DEFAULT_CRITIC_ROUNDS,
        help=f"Maximum reflexion rounds per failed candidate "
             f"(default: {DEFAULT_CRITIC_ROUNDS}; 0 disables the critic loop).",
    )
    p.add_argument(
        "--critic-model", default=DEFAULT_CRITIC_MODEL,
        help=f"Reasoning model for the critic (default: {DEFAULT_CRITIC_MODEL}).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print noisy [unload] lines on every model eviction.",
    )
    return p


def _announce_session(session_dir: Path) -> None:
    print(f"Session: {session_dir}", flush=True)


def _print_summary(summary: SessionSummary) -> None:
    pct = summary.pass_rate * 100.0
    one_shot_pct = summary.one_shot_pass_rate * 100.0
    if summary.critic_rounds > 0:
        print(
            f"Summary: {summary.passes}/{summary.total_candidates} passed "
            f"({pct:.0f}% final, {one_shot_pct:.0f}% one-shot), "
            f"mean latency {summary.mean_latency_ms:.0f}ms, "
            f"model={summary.model}, "
            f"critic={summary.critic_model} x{summary.critic_rounds}"
        )
    else:
        print(
            f"Summary: {summary.passes}/{summary.total_candidates} passed "
            f"({pct:.0f}%), mean latency {summary.mean_latency_ms:.0f}ms, "
            f"model={summary.model}"
        )


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = _build_parser().parse_args(argv)
    if args.runs < 1 or args.n < 1:
        print("error: --runs and --n must both be >= 1", file=sys.stderr)
        return _EXIT_USAGE_ERROR
    if args.critic_rounds < 0:
        print("error: --critic-rounds must be >= 0", file=sys.stderr)
        return _EXIT_USAGE_ERROR

    try:
        _, summary = run_bench(
            task=args.task,
            runs=args.runs,
            n=args.n,
            model=args.model,
            timeout_seconds=args.timeout,
            critic_rounds=args.critic_rounds,
            critic_model=args.critic_model,
            on_session_created=_announce_session,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USAGE_ERROR

    _print_summary(summary)
    return 0 if summary.passes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
