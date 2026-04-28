"""CLI smoke test: ``python -m aura_harness.scoring "PROMPT" --n 5 --expect add``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from aura_harness.lab.generator import CandidateGenerator
from aura_harness.llm import OllamaClient
from aura_harness.scoring.models import ScoredBatch, ScoredCandidate, ValidationSpec
from aura_harness.scoring.scorer import score_batch

_ERROR_PREVIEW_CHARS: Final[int] = 200
_EXIT_USAGE_ERROR: Final[int] = 2


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout to UTF-8 so non-ASCII status glyphs render on Windows."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aura_harness.scoring",
        description="Generate N candidates and score them against a validation spec.",
    )
    p.add_argument("prompt", help="Prompt to send to the model.")
    p.add_argument("--n", type=int, default=3, help="Number of candidates (default: 3).")
    p.add_argument("--model", default=None, help="Model name. Defaults to OllamaClient default.")
    p.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    p.add_argument("--seed", type=int, default=None, help="Base seed; default: random.")
    p.add_argument(
        "--expect", action="append", default=[], metavar="SYMBOL",
        help="Top-level symbol that must be defined. May be repeated.",
    )
    p.add_argument(
        "--test", default=None,
        help="Inline Python test code appended to each candidate. Best for short, "
             "single-line tests; multiline strings get mangled by cmd.exe — use "
             "--test-file instead.",
    )
    p.add_argument(
        "--test-file", default=None, metavar="PATH",
        help="UTF-8 Python file appended to each candidate. Recommended for "
             "non-trivial multiline tests. Mutually exclusive with --test.",
    )
    p.add_argument("--timeout", type=float, default=10.0, help="Test timeout in seconds.")
    return p


def _resolve_test_code(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve test source from ``--test`` / ``--test-file``.

    Returns ``(test_code, error)`` — exactly one is non-None, or both are
    None when the user passed neither flag.
    """
    if args.test is not None and args.test_file is not None:
        return (None, "--test and --test-file are mutually exclusive; pass at most one.")
    if args.test_file is not None:
        path = Path(args.test_file)
        try:
            return (path.read_text(encoding="utf-8"), None)
        except OSError as exc:
            return (None, f"could not read --test-file {path}: {exc}")
    return (args.test, None)


def _print_report(scored: ScoredBatch) -> None:
    b = scored.batch
    print(
        f"Batch {b.batch_id[:8]}: {b.n_requested} candidates from {b.model} "
        f"in {b.total_wall_duration_ms:.0f}ms"
    )
    for s in scored.scored:
        _print_candidate_line(s)
    pass_count = len(scored.passing)
    print()
    print(f"Summary: {pass_count}/{b.n_requested} passed ({scored.pass_rate * 100.0:.0f}%)")


def _print_candidate_line(s: ScoredCandidate) -> None:
    idx = s.candidate.index
    if s.validation is None:
        print(f"[{idx}] ✗ extracted=None FAIL")
        detail = s.candidate.error if s.candidate.error else "no code extracted"
        print(f"    {detail[:_ERROR_PREVIEW_CHARS]}")
        return

    v = s.validation
    mark = "✓" if v.passed else "✗"
    parse_str = "ok" if v.parse_ok else "fail"
    if not v.parse_ok:
        symbols_str = "skip"
    elif v.missing_symbols:
        symbols_str = f"missing:{list(v.missing_symbols)}"
    else:
        symbols_str = "ok"
    test_str = "skip" if not v.test_ran else ("ok" if v.test_ok else "fail")
    verdict = "PASS" if v.passed else "FAIL"
    print(
        f"[{idx}] {mark} extracted={s.extraction_method} parse={parse_str} "
        f"symbols={symbols_str} test={test_str} {verdict} ({v.duration_ms:.0f}ms)"
    )
    detail = v.test_error or v.parse_error
    if detail:
        print(f"    {detail[:_ERROR_PREVIEW_CHARS]}")


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = _build_parser().parse_args(argv)
    test_code, err = _resolve_test_code(args)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return _EXIT_USAGE_ERROR
    client = OllamaClient(default_model=args.model) if args.model else OllamaClient()
    generator = CandidateGenerator(client)
    batch = generator.generate(
        args.prompt,
        n=args.n,
        model=args.model,
        temperature=args.temperature,
        base_seed=args.seed,
    )
    spec = ValidationSpec(
        expected_symbols=tuple(args.expect),
        test_code=test_code,
        test_timeout_seconds=args.timeout,
    )
    scored = score_batch(batch, spec)
    _print_report(scored)
    return 0 if scored.passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
