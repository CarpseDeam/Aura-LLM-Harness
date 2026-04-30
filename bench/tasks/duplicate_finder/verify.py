"""Verifier for the duplicate_finder benchmark task.

Invoked as a subprocess by the bench runner:

    python verify.py <candidate_module_path> <fixtures_dir> <report_path>

The third positional argument is dual-purpose: it is both the path the
candidate is asked to write its duplicates JSON to AND the path the verifier
overwrites with its structured assertion report once evaluation is complete.
The runner reads the structured report after the subprocess exits.

Report shape:

    {
        "tests_passed": <int>,
        "tests_total": <int>,
        "failures": [<assertion-name>, ...]
    }

The verifier emits this report regardless of pass/fail so the runner can
aggregate the gradient signal. Exit status 0 means every assertion passed,
non-zero means at least one failed (or there was a usage error).

Standard library only, no imports from the harness.
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
_EXIT_FAIL: Final[int] = 1

_ASSERT_IMPORTS: Final[str] = "module imports"
_ASSERT_HAS_ENTRYPOINT: Final[str] = "exposes find_duplicates"
_ASSERT_RUNS_CLEANLY: Final[str] = "runs without raising"
_ASSERT_OUTPUT_EXISTS: Final[str] = "produces output file"
_ASSERT_VALID_JSON: Final[str] = "output is valid JSON"
_ASSERT_SHAPE: Final[str] = "output is list[list[str]]"
_ASSERT_RELATIVE: Final[str] = "all paths are relative"
_ASSERT_FORWARD_SLASHES: Final[str] = "all paths use forward slashes"
_ASSERT_GROUPS_CORRECT: Final[str] = "duplicate groups are correct"

_ALL_ASSERTIONS: Final[tuple[str, ...]] = (
    _ASSERT_IMPORTS,
    _ASSERT_HAS_ENTRYPOINT,
    _ASSERT_RUNS_CLEANLY,
    _ASSERT_OUTPUT_EXISTS,
    _ASSERT_VALID_JSON,
    _ASSERT_SHAPE,
    _ASSERT_RELATIVE,
    _ASSERT_FORWARD_SLASHES,
    _ASSERT_GROUPS_CORRECT,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="Verify a duplicate_finder candidate against fixtures.",
    )
    parser.add_argument("candidate_module_path", help="Path to the candidate .py file.")
    parser.add_argument("fixtures_dir", help="Directory of fixture files to scan.")
    parser.add_argument(
        "report_path",
        help=(
            "Dual-purpose path: candidate writes its duplicates JSON here, "
            "verifier then overwrites with the structured assertion report."
        ),
    )
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
    """Validate the parsed JSON document is shaped as ``list[list[str]]``."""
    if not isinstance(report, list):
        raise ValueError(f"top-level value must be a list, got {type(report).__name__}")
    groups: list[list[str]] = []
    for i, group in enumerate(report):
        if not isinstance(group, list):
            raise ValueError(f"group {i} must be a list, got {type(group).__name__}")
        for j, path in enumerate(group):
            if not isinstance(path, str):
                raise ValueError(
                    f"group {i} entry {j} must be a string, got {type(path).__name__}"
                )
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


def _write_report(report_path: Path, passed: list[str], failures: list[str]) -> None:
    """Emit the structured assertion report consumed by the bench runner."""
    payload = {
        "tests_passed": len(passed),
        "tests_total": len(passed) + len(failures),
        "failures": failures,
    }
    try:
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"verify warning: could not write report at {report_path}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidate_path = Path(args.candidate_module_path)
    fixtures_dir = Path(args.fixtures_dir)
    report_path = Path(args.report_path)
    expected_path = Path(__file__).resolve().parent / _EXPECTED_REPORT_FILENAME

    if not candidate_path.is_file():
        print(f"verify error: candidate module not found: {candidate_path}", file=sys.stderr)
        _write_report(report_path, [], list(_ALL_ASSERTIONS))
        return _EXIT_USAGE_ERROR
    if not fixtures_dir.is_dir():
        print(f"verify error: fixtures dir not found: {fixtures_dir}", file=sys.stderr)
        _write_report(report_path, [], list(_ALL_ASSERTIONS))
        return _EXIT_USAGE_ERROR
    if not expected_path.is_file():
        print(f"verify error: expected report not found: {expected_path}", file=sys.stderr)
        _write_report(report_path, [], list(_ALL_ASSERTIONS))
        return _EXIT_USAGE_ERROR

    passed: list[str] = []
    failures: list[str] = []

    def record(name: str, ok: bool) -> bool:
        (passed if ok else failures).append(name)
        return ok

    def remaining_fail(starting_with: str) -> None:
        idx = _ALL_ASSERTIONS.index(starting_with)
        for name in _ALL_ASSERTIONS[idx:]:
            if name not in passed and name not in failures:
                failures.append(name)

    try:
        module = _load_candidate(candidate_path)
        record(_ASSERT_IMPORTS, True)
    except Exception as exc:  # noqa: BLE001 — surface any import-time failure
        print(f"verify error: failed to import candidate: {exc!r}", file=sys.stderr)
        record(_ASSERT_IMPORTS, False)
        remaining_fail(_ASSERT_HAS_ENTRYPOINT)
        _write_report(report_path, passed, failures)
        return _EXIT_FAIL

    fn = getattr(module, "find_duplicates", None)
    if not callable(fn):
        print("verify error: candidate has no callable 'find_duplicates'", file=sys.stderr)
        record(_ASSERT_HAS_ENTRYPOINT, False)
        remaining_fail(_ASSERT_RUNS_CLEANLY)
        _write_report(report_path, passed, failures)
        return _EXIT_FAIL
    record(_ASSERT_HAS_ENTRYPOINT, True)

    try:
        fn(str(fixtures_dir), str(report_path))
        record(_ASSERT_RUNS_CLEANLY, True)
    except Exception as exc:  # noqa: BLE001 — candidate failure must not crash verifier
        print(f"verify error: find_duplicates raised: {exc!r}", file=sys.stderr)
        record(_ASSERT_RUNS_CLEANLY, False)
        remaining_fail(_ASSERT_OUTPUT_EXISTS)
        _write_report(report_path, passed, failures)
        return _EXIT_FAIL

    if not report_path.is_file():
        print(f"verify error: candidate did not write {report_path}", file=sys.stderr)
        record(_ASSERT_OUTPUT_EXISTS, False)
        remaining_fail(_ASSERT_VALID_JSON)
        _write_report(report_path, passed, failures)
        return _EXIT_FAIL
    record(_ASSERT_OUTPUT_EXISTS, True)

    try:
        actual_raw = json.loads(report_path.read_text(encoding="utf-8"))
        record(_ASSERT_VALID_JSON, True)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"verify error: could not parse output JSON: {exc}", file=sys.stderr)
        record(_ASSERT_VALID_JSON, False)
        remaining_fail(_ASSERT_SHAPE)
        _write_report(report_path, passed, failures)
        return _EXIT_FAIL

    try:
        expected_raw = json.loads(expected_path.read_text(encoding="utf-8"))
        expected = _normalize(expected_raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"verify error: could not parse expected JSON: {exc}", file=sys.stderr)
        remaining_fail(_ASSERT_SHAPE)
        _write_report(report_path, passed, failures)
        return _EXIT_USAGE_ERROR

    try:
        actual = _normalize(actual_raw)
        record(_ASSERT_SHAPE, True)
    except ValueError as exc:
        print(f"verify error: report shape invalid: {exc}", file=sys.stderr)
        record(_ASSERT_SHAPE, False)
        remaining_fail(_ASSERT_RELATIVE)
        _write_report(report_path, passed, failures)
        return _EXIT_FAIL

    all_paths = [p for group in actual for p in group]

    relative_offenders = [p for p in all_paths if Path(p).is_absolute() or p.startswith("/")]
    if relative_offenders:
        print(
            f"verify error: paths must be relative, found absolute: {relative_offenders}",
            file=sys.stderr,
        )
    record(_ASSERT_RELATIVE, not relative_offenders)

    backslash_offenders = [p for p in all_paths if "\\" in p]
    if backslash_offenders:
        print(
            f"verify error: paths must use forward slashes, found backslashes: {backslash_offenders}",
            file=sys.stderr,
        )
    record(_ASSERT_FORWARD_SLASHES, not backslash_offenders)

    groups_match = actual == expected
    if not groups_match:
        print(_diff(expected, actual), file=sys.stderr)
    record(_ASSERT_GROUPS_CORRECT, groups_match)

    _write_report(report_path, passed, failures)
    return 0 if not failures else _EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
