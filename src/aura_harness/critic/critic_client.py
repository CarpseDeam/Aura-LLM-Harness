"""Stateless critic service backed by a local reasoning model via Ollama.

Given a task spec and a candidate's code, :class:`CriticClient` asks the
reasoning model for a structured contract review and returns a
:class:`Critique`. Parse failures and transport failures are captured as a
``passed=False`` critique with an explanatory violation; the client never
raises.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from aura_harness.critic.models import Critique
from aura_harness.llm import OllamaClient, OllamaError

logger = logging.getLogger(__name__)

DEFAULT_CRITIC_MODEL: Final[str] = "deepseek-r1:14b"

CRITIC_SYSTEM_PROMPT: Final[str] = (
    "You are a strict code reviewer. You will be shown a task specification "
    "and a candidate Python module that attempts to fulfill it. Your job is "
    "to find concrete, mechanical contract violations — wrong function "
    "names, wrong signatures, wrong return shapes, missing requirements, "
    "string vs Path mistakes, platform-specific path separators, and so on. "
    "Be specific and terse. Do NOT critique style or naming preferences; "
    "only flag issues that would cause the verifier to reject the candidate.\n"
    "\n"
    "Output a single JSON object and nothing else, with this exact shape:\n"
    "{\n"
    '  "passed": <bool>,\n'
    '  "violations": [<string>, ...],\n'
    '  "suggestions": [<string>, ...]\n'
    "}\n"
    "\n"
    "- ``passed`` is true only when you find no contract violations.\n"
    "- ``violations`` are short concrete strings naming each issue.\n"
    "- ``suggestions`` are short concrete fixes, ideally one per violation.\n"
    "- Output JSON only. No prose, no markdown fences, no commentary."
)

_DEFAULT_TEMPERATURE: Final[float] = 0.1
_DEFAULT_LOG_RELPATH: Final[Path] = Path(".aura") / "critic.jsonl"
_PROMPT_PREVIEW_CHARS: Final[int] = 200
_RESPONSE_PREVIEW_CHARS: Final[int] = 400
_RAW_RESPONSE_TRUNCATE_CHARS: Final[int] = 8000
_VERIFIER_STDERR_CAP_CHARS: Final[int] = 2000

_FENCED_JSON_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_USER_PROMPT_BASE: Final[str] = (
    "## Task specification\n"
    "\n"
    "{spec}\n"
    "\n"
    "## Candidate module\n"
    "\n"
    "```python\n"
    "{code}\n"
    "```\n"
)

_USER_PROMPT_TAIL_DEFAULT: Final[str] = (
    "Review the candidate against the specification. Emit JSON only."
)

_USER_PROMPT_TAIL_FOCUSED: Final[str] = (
    "Focus your review on the failures listed above. The passing tests "
    "already validate the rest of the candidate's behavior — do not "
    "re-critique that. Identify the specific contract violations causing "
    "the failed tests and propose the minimal changes that would flip "
    "them to passing without regressing the passing tests. Emit JSON only."
)


class CriticClient:
    """Turn a (spec, candidate code) pair into a structured :class:`Critique`."""

    def __init__(
        self,
        client: OllamaClient,
        *,
        model: str = DEFAULT_CRITIC_MODEL,
        temperature: float = _DEFAULT_TEMPERATURE,
        log_path: Path | None = None,
    ) -> None:
        """Construct a critic client.

        Args:
            client: The :class:`OllamaClient` used for chat calls.
            model: Reasoning model name. Defaults to
                :data:`DEFAULT_CRITIC_MODEL`.
            temperature: Sampling temperature for the reviewer.
            log_path: JSONL file to append critic-call records to. Defaults
                to ``./.aura/critic.jsonl`` under the current working
                directory.
        """
        self._client = client
        self._model = model
        self._temperature = temperature
        self._log_path = (
            log_path if log_path is not None else Path.cwd() / _DEFAULT_LOG_RELPATH
        )

    @property
    def model(self) -> str:
        """Reasoning model used for critic calls."""
        return self._model

    @property
    def log_path(self) -> Path:
        """Path to the JSONL critic-call log."""
        return self._log_path

    def review(
        self,
        spec: str,
        candidate_code: str,
        *,
        failures: tuple[str, ...] = (),
        verifier_stderr: str | None = None,
        tests_passed: int | None = None,
        tests_total: int | None = None,
    ) -> Critique:
        """Ask the reasoning model to review ``candidate_code`` against ``spec``.

        Optional failure-context kwargs let the runner thread the prior
        round's verifier signal into the prompt so the critic can focus on
        the specific tests that failed instead of re-analyzing the whole
        module from scratch:

        - ``failures``: failed test names from the prior round.
        - ``verifier_stderr``: stderr captured by the verifier subprocess
            (truncated to a 2000 char cap before being included).
        - ``tests_passed`` / ``tests_total``: the gradient signal. Both
            must be provided for the gradient line to appear.

        When none of the failure-context fields are provided the prompt
        falls back to the legacy generic-review wording, so existing
        callers using ``review(spec, code)`` are unaffected.

        Never raises. Transport failures and JSON parse failures are returned
        as a ``passed=False`` :class:`Critique` with a single violation
        explaining the cause.
        """
        user_content = _build_user_prompt(
            spec=spec,
            code=candidate_code,
            failures=failures,
            verifier_stderr=verifier_stderr,
            tests_passed=tests_passed,
            tests_total=tests_total,
        )
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            result = self._client.chat(
                messages,
                model=self._model,
                temperature=self._temperature,
            )
        except OllamaError as exc:
            critique = Critique(
                passed=False,
                violations=(f"critic transport error: {exc}",),
                suggestions=(),
                raw_response=None,
            )
            self._log_call(
                user_content=user_content,
                response_text="",
                critique=critique,
                prompt_eval_count=0,
                eval_count=0,
                total_duration_ms=0.0,
                error=str(exc),
            )
            return critique

        critique = _parse_response(result.text)
        self._log_call(
            user_content=user_content,
            response_text=result.text,
            critique=critique,
            prompt_eval_count=result.prompt_eval_count,
            eval_count=result.eval_count,
            total_duration_ms=result.total_duration_ms,
            error=None,
        )
        return critique

    def _log_call(
        self,
        *,
        user_content: str,
        response_text: str,
        critique: Critique,
        prompt_eval_count: int,
        eval_count: int,
        total_duration_ms: float,
        error: str | None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self._model,
            "prompt_preview": user_content[:_PROMPT_PREVIEW_CHARS],
            "response_preview": response_text[:_RESPONSE_PREVIEW_CHARS],
            "passed": critique.passed,
            "violation_count": len(critique.violations),
            "suggestion_count": len(critique.suggestions),
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_duration_ms": total_duration_ms,
            "error": error,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("failed to write critic log to %s: %s", self._log_path, exc)


def _build_user_prompt(
    *,
    spec: str,
    code: str,
    failures: tuple[str, ...],
    verifier_stderr: str | None,
    tests_passed: int | None,
    tests_total: int | None,
) -> str:
    """Compose the user message for the critic.

    When any failure-context field is populated, append a "Failure context"
    section listing the gradient, failed test names, and (capped) verifier
    stderr, plus a focus directive in place of the generic review wording.
    Otherwise, return the legacy prompt unchanged.
    """
    base = _USER_PROMPT_BASE.format(spec=spec, code=code)
    failure_section = _build_failure_context_section(
        failures=failures,
        verifier_stderr=verifier_stderr,
        tests_passed=tests_passed,
        tests_total=tests_total,
    )
    if not failure_section:
        return f"{base}\n{_USER_PROMPT_TAIL_DEFAULT}"
    return f"{base}\n{failure_section}\n{_USER_PROMPT_TAIL_FOCUSED}"


def _build_failure_context_section(
    *,
    failures: tuple[str, ...],
    verifier_stderr: str | None,
    tests_passed: int | None,
    tests_total: int | None,
) -> str:
    """Render the optional "Failure context" section, or ``""`` if empty."""
    lines: list[str] = []
    if tests_passed is not None and tests_total is not None:
        lines.append(
            f"- Gradient: passed {tests_passed} of {tests_total} named tests."
        )
    if failures:
        lines.append("- Failed tests:")
        lines.extend(f"  - {name}" for name in failures)
    stderr_block = _format_verifier_stderr(verifier_stderr)
    if stderr_block is not None:
        lines.append("- Verifier stderr:")
        lines.append(stderr_block)
    if not lines:
        return ""
    body = "\n".join(lines)
    return f"## Failure context\n\nThe verifier reported:\n\n{body}\n"


def _format_verifier_stderr(stderr: str | None) -> str | None:
    """Wrap stderr in a fenced block, truncating to the 2000 char cap."""
    if stderr is None:
        return None
    text = stderr.strip()
    if not text:
        return None
    if len(text) > _VERIFIER_STDERR_CAP_CHARS:
        omitted = len(text) - _VERIFIER_STDERR_CAP_CHARS
        text = (
            text[:_VERIFIER_STDERR_CAP_CHARS]
            + f"\n... (truncated, {omitted} more chars)"
        )
    return f"```\n{text}\n```"


def _parse_response(text: str) -> Critique:
    """Parse the model's response into a :class:`Critique`.

    Tolerates surrounding prose and reasoning-model ``<think>`` blocks by
    extracting the first JSON object found. On parse failure, returns a
    ``passed=False`` critique whose single violation explains the failure.
    """
    raw = text[:_RAW_RESPONSE_TRUNCATE_CHARS] if text else ""
    payload = _extract_json_object(text)
    if payload is None:
        return Critique(
            passed=False,
            violations=("critic parse error: no JSON object in response",),
            suggestions=(),
            raw_response=raw,
        )

    passed_value = payload.get("passed")
    if not isinstance(passed_value, bool):
        return Critique(
            passed=False,
            violations=(
                "critic parse error: missing or non-boolean 'passed' field",
            ),
            suggestions=(),
            raw_response=raw,
        )

    violations = _coerce_string_tuple(payload.get("violations"))
    suggestions = _coerce_string_tuple(payload.get("suggestions"))
    return Critique(
        passed=passed_value,
        violations=violations,
        suggestions=suggestions,
        raw_response=raw,
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in ``text``, or ``None``.

    Tries fenced ```json blocks first, then falls back to the slice between
    the first ``{`` and the last ``}``.
    """
    if not text:
        return None

    fenced = _FENCED_JSON_RE.search(text)
    if fenced is not None:
        parsed = _try_loads(fenced.group(1))
        if parsed is not None:
            return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return _try_loads(text[start : end + 1])


def _try_loads(blob: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a JSON list-of-strings field into a tuple, dropping non-strings."""
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())
