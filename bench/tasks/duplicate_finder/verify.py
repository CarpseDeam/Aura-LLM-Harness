"""Verifier for the duplicate_finder benchmark task.

Invoked as a subprocess by the bench runner:

    python verify.py <candidate_module_path> <fixtures_dir> <output_json_path>

Loads the candidate module, calls its ``find_duplicates`` function against the
fixtures directory, then compares the produced JSON report to the committed
``expected_report.json``. Exits 0 on a structural match; exits non-zero with a
diff written to stderr on any mismatch.

This script is the source of truth for "did the candidate work?". It is
intentionally strict, deterministic, and self-contained: standard library
only, no imports from the harness.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Final

_EXPECTED_REPORT_FILENAME: Final[str] = "expected_report.json"
_EXIT_USAGE_ERROR: Final[int] = 2
_EXIT_LOAD_ERROR: Final[int] = 3
_EXIT_RUN_ERROR: Final[int] = 4
_EXIT_OUTPUT_ERROR: Final[int] = 5
_EXIT_MISMATCH: Final[int] = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="Verify a duplicate_finder candidate against fixtures.",
    )
    parser.add_argument("candidate_module_path", help="Path to the candidate .py file.")
    parser.add_argument("fixtures_dir", help="Directory of fixture files to scan.")
    parser.add_argument("output_json_path", help="Path the candidate must write its report to.")
    return parser


def _load_candidate(module_path: Path) -> Any:
    """Import the candidate module from a file path."""
    spec = importlib.util.spec_from_file_location("candidate_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize(report: Any) -> list[list[str]]:
    """Validate the parsed JSON document is shaped as ``list[list[str]]``.

    Order is preserved as-is: the spec requires the candidate to emit groups
    and within-group paths in lexicographic order, and the verifier enforces
    that strictly via the equality check.
    """
    if not isinstance(report, list):
        raise ValueError(f"top-level value must be a list, got {type(report).__name__}")
    groups: list[list[str]] = []
    for i, group in enumerate(report):
        if not isinstance(group, list):
            raise ValueError(f"group {i} must be a list, got {type(group).__name__}")
        for j, path in enumerate(group):
            if not isinstance(path, str):
                raise ValueError(f"group {i} entry {j} must be a string, got {type(path).__name__}")
        groups.append(list(group))
    return groups


def _diff(expected: list[list[str]], actual: list[list[str]]) -> str:
    lines = ["report mismatch:"]
    lines.append(f"  expected ({len(expected)} groups):")
    for group in expected:
        lines.append(f"    {group}")
    lines.append(f"  actual ({len(actual)} groups):")
    for group in actual:
        lines.append(f"    {group}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidate_path = Path(args.candidate_module_path)
    fixtures_dir = Path(args.fixtures_dir)
    output_path = Path(args.output_json_path)
    expected_path = Path(__file__).resolve().parent / _EXPECTED_REPORT_FILENAME

    if not candidate_path.is_file():
        print(f"verify error: candidate module not found: {candidate_path}", file=sys.stderr)
        return _EXIT_USAGE_ERROR
    if not fixtures_dir.is_dir():
        print(f"verify error: fixtures dir not found: {fixtures_dir}", file=sys.stderr)
        return _EXIT_USAGE_ERROR
    if not expected_path.is_file():
        print(f"verify error: expected report not found: {expected_path}", file=sys.stderr)
        return _EXIT_USAGE_ERROR

    try:
        module = _load_candidate(candidate_path)
    except Exception as exc:  # noqa: BLE001 — surface any import-time failure
        print(f"verify error: failed to import candidate: {exc!r}", file=sys.stderr)
        return _EXIT_LOAD_ERROR

    fn = getattr(module, "find_duplicates", None)
    if not callable(fn):
        print("verify error: candidate has no callable 'find_duplicates'", file=sys.stderr)
        return _EXIT_LOAD_ERROR

    try:
        fn(str(fixtures_dir), str(output_path))
    except Exception as exc:  # noqa: BLE001 — candidate failure must not crash verifier
        print(f"verify error: find_duplicates raised: {exc!r}", file=sys.stderr)
        return _EXIT_RUN_ERROR

    if not output_path.is_file():
        print(f"verify error: candidate did not write {output_path}", file=sys.stderr)
        return _EXIT_OUTPUT_ERROR

    try:
        actual_raw = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"verify error: could not parse output JSON: {exc}", file=sys.stderr)
        return _EXIT_OUTPUT_ERROR

    try:
        expected_raw = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"verify error: could not parse expected JSON: {exc}", file=sys.stderr)
        return _EXIT_OUTPUT_ERROR

    try:
        expected = _normalize(expected_raw)
        actual = _normalize(actual_raw)
    except ValueError as exc:
        print(f"verify error: report shape invalid: {exc}", file=sys.stderr)
        return _EXIT_MISMATCH

    if actual != expected:
        print(_diff(expected, actual), file=sys.stderr)
        return _EXIT_MISMATCH

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
