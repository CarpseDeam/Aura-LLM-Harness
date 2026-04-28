"""Tests for the code extractor."""
from __future__ import annotations

from aura_harness.scoring.extractor import extract_code


def test_single_python_fenced_block() -> None:
    text = "Here is some code:\n```python\ndef f():\n    return 1\n```\n"
    code, method = extract_code(text)
    assert method == "fenced_python"
    assert code == "def f():\n    return 1"


def test_multiple_python_blocks_returns_last() -> None:
    text = (
        "First attempt:\n```python\ndef f():\n    return 1\n```\n"
        "Better answer:\n```python\ndef f():\n    return 2\n```\n"
    )
    code, method = extract_code(text)
    assert method == "fenced_python"
    assert code == "def f():\n    return 2"


def test_python_block_preferred_over_unmarked() -> None:
    text = (
        "Random:\n```\nnot the answer\n```\n"
        "Real:\n```python\ndef f():\n    return 1\n```\n"
        "Trailing:\n```\nalso not it\n```\n"
    )
    code, method = extract_code(text)
    assert method == "fenced_python"
    assert code == "def f():\n    return 1"


def test_only_unmarked_fenced_blocks() -> None:
    text = "Stuff:\n```\ndef g():\n    return 99\n```\n"
    code, method = extract_code(text)
    assert method == "fenced_unmarked"
    assert code == "def g():\n    return 99"


def test_only_unmarked_returns_last() -> None:
    text = "```\nfirst\n```\n```\ndef h():\n    return 7\n```\n"
    code, method = extract_code(text)
    assert method == "fenced_unmarked"
    assert code == "def h():\n    return 7"


def test_raw_parse_when_no_fences() -> None:
    text = "def f():\n    return 1\n"
    code, method = extract_code(text)
    assert method == "raw_parse"
    assert code == "def f():\n    return 1"


def test_no_fences_and_gibberish() -> None:
    text = "this is not code at all !!! @@@@"
    assert extract_code(text) == (None, None)


def test_empty_string() -> None:
    assert extract_code("") == (None, None)


def test_realistic_deepseek_preamble() -> None:
    text = (
        "# Solution\n\n"
        "We need a function that adds two numbers. Here's my approach:\n\n"
        "First, we define the function signature, then return the sum.\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "This handles the basic case. Let me know if you need edge cases.\n"
    )
    code, method = extract_code(text)
    assert method == "fenced_python"
    assert code == "def add(a, b):\n    return a + b"


def test_whitespace_around_extracted_code_is_stripped() -> None:
    text = "```python\n\n\n   def f():\n       return 1   \n\n\n```\n"
    code, method = extract_code(text)
    assert method == "fenced_python"
    assert code == "def f():\n       return 1"


def test_python_lang_tag_case_insensitive() -> None:
    text = "```Python\ndef f():\n    return 1\n```\n"
    code, method = extract_code(text)
    assert method == "fenced_python"
    assert code == "def f():\n    return 1"


def test_py_short_tag_recognized() -> None:
    text = "```py\ndef f():\n    return 1\n```\n"
    code, method = extract_code(text)
    assert method == "fenced_python"


def test_empty_python_block_returns_none() -> None:
    text = "```python\n   \n```\n"
    assert extract_code(text) == (None, None)
