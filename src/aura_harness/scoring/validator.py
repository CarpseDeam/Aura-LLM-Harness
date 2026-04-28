"""Validate extracted Python code against a :class:`ValidationSpec`.

Three checks, in order: AST parse, expected-symbol presence, and an optional
test-code subprocess execution. Any failure short-circuits the rest in a way
that matches user intent: a syntax error skips the test, but a missing symbol
does not (since the user might still want to see the test failure).

Safety note: ``test_code`` runs in a subprocess with the calling user's full
permissions and no sandboxing. Acceptable for v1 (you're running prompts you
wrote yourself); proper sandboxing is a later concern.
"""
from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Final

from aura_harness.scoring.models import ValidationResult, ValidationSpec

logger = logging.getLogger(__name__)

_TEST_ERROR_TRUNCATE: Final[int] = 2000
_CANDIDATE_FILENAME: Final[str] = "_candidate.py"


def validate(code: str, spec: ValidationSpec) -> ValidationResult:
    """Validate ``code`` against ``spec``.

    Args:
        code: Python source to validate (already extracted from the model
            response).
        spec: Validation requirements.

    Returns:
        A :class:`ValidationResult`. This function never raises — every
        failure mode (parse error, missing symbol, test failure, timeout,
        unexpected internal error) surfaces as fields on the result.
    """
    started = time.perf_counter()

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(
            passed=False,
            parse_ok=False,
            parse_error=str(exc),
            missing_symbols=spec.expected_symbols,
            test_ran=False,
            test_ok=None,
            test_error=None,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    defined_names = _collect_top_level_names(tree)
    missing_symbols = tuple(s for s in spec.expected_symbols if s not in defined_names)

    test_ran = False
    test_ok: bool | None = None
    test_error: str | None = None
    if spec.test_code is not None:
        test_ran = True
        test_ok, test_error = _run_test(code, spec.test_code, spec.test_timeout_seconds)

    passed = (not missing_symbols) and (test_ok if test_ran else True)

    return ValidationResult(
        passed=bool(passed),
        parse_ok=True,
        parse_error=None,
        missing_symbols=missing_symbols,
        test_ran=test_ran,
        test_ok=test_ok,
        test_error=test_error,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def _collect_top_level_names(tree: ast.Module) -> set[str]:
    """Names defined at module top level via def/async def/class/Name=...."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _run_test(
    code: str, test_code: str, timeout_seconds: float
) -> tuple[bool, str | None]:
    """Execute ``code + test_code`` in a subprocess; return ``(ok, error)``."""
    try:
        with tempfile.TemporaryDirectory() as td:
            candidate_path = Path(td) / _CANDIDATE_FILENAME
            candidate_path.write_text(code + "\n\n" + test_code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, _CANDIDATE_FILENAME],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=td,
            )
    except subprocess.TimeoutExpired:
        return (False, f"timed out after {timeout_seconds}s")
    except Exception as exc:  # noqa: BLE001 — never let validator bugs crash callers
        logger.debug("validator subprocess error: %s", exc)
        return (False, f"validator error: {exc}")

    if completed.returncode == 0:
        return (True, None)
    detail = completed.stderr if completed.stderr else completed.stdout
    return (False, detail[:_TEST_ERROR_TRUNCATE])
