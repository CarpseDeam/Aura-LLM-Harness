"""Tests for the code validator."""
from __future__ import annotations

from aura_harness.scoring.models import ValidationSpec
from aura_harness.scoring.validator import validate


def test_valid_trivial_code_empty_spec() -> None:
    result = validate("x = 1\n", ValidationSpec())
    assert result.passed is True
    assert result.parse_ok is True
    assert result.parse_error is None
    assert result.missing_symbols == ()
    assert result.test_ran is False
    assert result.test_ok is None
    assert result.test_error is None
    assert result.duration_ms >= 0.0


def test_syntax_error_short_circuits() -> None:
    result = validate("def broken(:\n    pass", ValidationSpec(expected_symbols=("foo",)))
    assert result.passed is False
    assert result.parse_ok is False
    assert result.parse_error is not None
    assert "broken" in result.parse_error or "syntax" in result.parse_error.lower() or len(result.parse_error) > 0
    assert result.missing_symbols == ("foo",)
    assert result.test_ran is False
    assert result.test_ok is None
    assert result.test_error is None


def test_expected_symbol_present() -> None:
    code = "def foo():\n    return 1\n"
    result = validate(code, ValidationSpec(expected_symbols=("foo",)))
    assert result.passed is True
    assert result.missing_symbols == ()


def test_expected_symbol_missing() -> None:
    code = "def bar():\n    return 1\n"
    result = validate(code, ValidationSpec(expected_symbols=("foo",)))
    assert result.passed is False
    assert result.missing_symbols == ("foo",)
    assert result.parse_ok is True


def test_class_and_assign_count_as_top_level_symbols() -> None:
    code = "class Thing:\n    pass\n\nVALUE = 42\n"
    result = validate(code, ValidationSpec(expected_symbols=("Thing", "VALUE")))
    assert result.missing_symbols == ()
    assert result.passed is True


def test_async_function_counts_as_top_level_symbol() -> None:
    code = "async def runner():\n    return 1\n"
    result = validate(code, ValidationSpec(expected_symbols=("runner",)))
    assert result.missing_symbols == ()


def test_nested_def_does_not_count_as_top_level() -> None:
    code = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    result = validate(code, ValidationSpec(expected_symbols=("inner",)))
    assert result.missing_symbols == ("inner",)


def test_test_code_passes() -> None:
    code = "def add(a, b):\n    return a + b\n"
    spec = ValidationSpec(expected_symbols=("add",), test_code="assert add(1, 2) == 3")
    result = validate(code, spec)
    assert result.test_ran is True
    assert result.test_ok is True
    assert result.test_error is None
    assert result.passed is True


def test_test_code_fails_on_wrong_value() -> None:
    code = "def add(a, b):\n    return a - b\n"
    spec = ValidationSpec(expected_symbols=("add",), test_code="assert add(1, 2) == 3")
    result = validate(code, spec)
    assert result.test_ran is True
    assert result.test_ok is False
    assert result.test_error is not None
    assert len(result.test_error) > 0
    assert result.passed is False


def test_test_code_timeout() -> None:
    code = "def loop():\n    pass\n"
    spec = ValidationSpec(test_code="while True:\n    pass", test_timeout_seconds=0.5)
    result = validate(code, spec)
    assert result.test_ran is True
    assert result.test_ok is False
    assert result.test_error is not None
    assert "timed out" in result.test_error
    assert result.passed is False


def test_test_code_runtime_error_captured() -> None:
    code = "def f():\n    return 1\n"
    spec = ValidationSpec(test_code="raise RuntimeError('nope')")
    result = validate(code, spec)
    assert result.test_ran is True
    assert result.test_ok is False
    assert result.test_error is not None
    assert "nope" in result.test_error


def test_validator_never_raises_on_weird_input() -> None:
    # Whatever we throw at it, it must produce a ValidationResult.
    weird_inputs = ["", "    ", "def x(:", "\x00\x01\x02"]
    for inp in weird_inputs:
        result = validate(inp, ValidationSpec(expected_symbols=("foo",)))
        assert result is not None
        assert isinstance(result.passed, bool)


def test_passed_requires_no_missing_symbols_even_when_test_passes() -> None:
    code = "def add(a, b):\n    return a + b\n"
    spec = ValidationSpec(
        expected_symbols=("add", "missing_thing"),
        test_code="assert add(1, 2) == 3",
    )
    result = validate(code, spec)
    assert result.test_ok is True
    assert result.missing_symbols == ("missing_thing",)
    assert result.passed is False
