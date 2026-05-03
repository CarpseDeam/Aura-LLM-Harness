"""Planner station — turn a :class:`TaskSpec` into a structured :class:`Plan`.

The planner is a thin stage: it renders a prompt from the task spec, asks
the backend for a JSON plan, validates the parsed JSON against the
:class:`Plan` / :class:`Slice` / :class:`SliceContract` schema, and stamps
genealogy onto the resulting :class:`Plan`. Plan-level critique is not in
scope here — that arrives in a later dispatch and will change the return
type when it does.

Backend selection (cloud vs. local) and model choice are pushed to the
call site so the same station works against DeepSeek via
:class:`CloudHTTPBackend` and against ``gpt-oss:20b`` via
:class:`LocalOllamaBackend` without subclassing.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Final

from aura_harness.backend import Backend, BackendError
from aura_harness.state import Plan, Slice, SliceContract, TaskSpec
from aura_harness.stations.base import Station

_PROMPTS_DIR: Final[Path] = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_PATH: Final[Path] = _PROMPTS_DIR / "planner_system.md"
_USER_PROMPT_PATH: Final[Path] = _PROMPTS_DIR / "planner_user.md"

_FENCED_JSON_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_PARSE_RETRY_FEEDBACK: Final[str] = (
    "Your previous response could not be parsed as JSON: {error}\n"
    "\n"
    "Reply again with a single JSON object only. No prose, no markdown "
    "fences, no commentary. Match the schema in the system message exactly."
)

_VALIDATION_RETRY_FEEDBACK: Final[str] = (
    "Your previous response parsed as JSON but did not match the required "
    "schema: {error}\n"
    "\n"
    "Reply again with a single JSON object that fixes the listed issue. "
    "Output JSON only."
)


class PlannerStation(Station):
    """Decompose a :class:`TaskSpec` into a stamped :class:`Plan`.

    The constructor takes the backend and the model string explicitly:
    the same station instance can be repointed at a different model by
    constructing it again with a different ``model``, and a different
    backend by passing a different :class:`Backend`. No backend or model
    selection lives inside this class.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        model: str,
        name: str = "planner",
        temperature: float = 0.2,
        max_parse_retries: int = 1,
    ) -> None:
        if max_parse_retries < 0:
            raise ValueError("max_parse_retries must be >= 0")
        super().__init__(name=name, backend=backend)
        self._model = model
        self._temperature = temperature
        self._max_parse_retries = max_parse_retries
        self._system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self._user_template = _USER_PROMPT_PATH.read_text(encoding="utf-8")

    @property
    def model(self) -> str:
        """Model name dispatched against on every call."""
        return self._model

    def run(self, task_spec: TaskSpec) -> Plan:
        """Produce a :class:`Plan` for ``task_spec``.

        Raises:
            BackendError: When the backend repeatedly returns output that
                cannot be parsed as JSON, or that parses but fails Plan
                schema validation, after ``max_parse_retries`` retries.
        """
        user_content = self._render_user_prompt(task_spec)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_error: str = ""
        attempts = self._max_parse_retries + 1
        for _ in range(attempts):
            result = self._backend.chat(
                messages,
                model=self._model,
                temperature=self._temperature,
                response_format="json",
            )
            messages.append({"role": "assistant", "content": result.text})

            payload, parse_error = _extract_json_object(result.text)
            if payload is None:
                last_error = parse_error or "no JSON object found in response"
                messages.append(
                    {
                        "role": "user",
                        "content": _PARSE_RETRY_FEEDBACK.format(error=last_error),
                    }
                )
                continue

            validation_error = _validate_plan_payload(payload)
            if validation_error is not None:
                last_error = validation_error
                messages.append(
                    {
                        "role": "user",
                        "content": _VALIDATION_RETRY_FEEDBACK.format(error=last_error),
                    }
                )
                continue

            return self._build_plan(payload=payload, task_spec=task_spec)

        raise BackendError(
            f"planner could not produce a valid plan after {attempts} attempt(s): "
            f"{last_error}"
        )

    def _render_user_prompt(self, task_spec: TaskSpec) -> str:
        constraints_section = _render_constraints_section(task_spec.constraints)
        return self._user_template.format(
            description=task_spec.description.strip(),
            constraints_section=constraints_section,
        )

    def _build_plan(self, *, payload: dict[str, Any], task_spec: TaskSpec) -> Plan:
        slices = tuple(_build_slice(item) for item in payload["slices"])
        rationale = str(payload.get("rationale", "")).strip()
        return Plan(
            plan_id=uuid.uuid4().hex,
            slices=slices,
            rationale=rationale,
            **self._genealogy(seed=None, input_ref=task_spec.task_id),
        )


def _render_constraints_section(constraints: tuple[str, ...]) -> str:
    if not constraints:
        return ""
    bullets = "\n".join(f"- {item}" for item in constraints)
    return f"\n# Constraints\n\n{bullets}\n"


def _build_slice(item: dict[str, Any]) -> Slice:
    contract_payload = item.get("contract") or {}
    contract = SliceContract(
        expected_symbols=_string_tuple(contract_payload.get("expected_symbols")),
        forbidden_imports=_string_tuple(contract_payload.get("forbidden_imports")),
    )
    target_path = str(item["target_path"]).strip()
    target_files: tuple[Path, ...] = (Path(target_path),)
    return Slice(
        slice_id=str(item["id"]).strip(),
        description=str(item["description"]).strip(),
        target_files=target_files,
        depends_on=_string_tuple(item.get("depends_on")),
        contract=contract,
        target_path=target_path,
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        str(item).strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )


def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(payload, None)`` on success or ``(None, error)`` on failure."""
    if not text or not text.strip():
        return None, "empty response"

    fenced = _FENCED_JSON_RE.search(text)
    if fenced is not None:
        parsed, error = _try_loads(fenced.group(1))
        if parsed is not None:
            return parsed, None
        # fall through to brace-slice fallback if the fenced block itself
        # was not a valid object
        last_error: str | None = error
    else:
        last_error = None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, last_error or "no JSON object delimiters found"

    parsed, error = _try_loads(text[start : end + 1])
    if parsed is None:
        return None, error or last_error or "could not decode JSON"
    return parsed, None


def _try_loads(blob: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc.msg} (line {exc.lineno}, col {exc.colno})"
    if not isinstance(parsed, dict):
        return None, "top-level JSON value is not an object"
    return parsed, None


def _validate_plan_payload(payload: dict[str, Any]) -> str | None:
    """Return ``None`` if ``payload`` matches the Plan schema, else an error message."""
    slices = payload.get("slices")
    if not isinstance(slices, list) or not slices:
        return "'slices' must be a non-empty array"

    seen_ids: set[str] = set()
    for index, item in enumerate(slices):
        prefix = f"slices[{index}]"
        if not isinstance(item, dict):
            return f"{prefix} is not an object"

        slice_id = item.get("id")
        if not isinstance(slice_id, str) or not slice_id.strip():
            return f"{prefix}.id must be a non-empty string"
        if slice_id in seen_ids:
            return f"{prefix}.id duplicates an earlier slice id ('{slice_id}')"
        seen_ids.add(slice_id)

        if not isinstance(item.get("description"), str) or not item["description"].strip():
            return f"{prefix}.description must be a non-empty string"

        target_path = item.get("target_path")
        if not isinstance(target_path, str) or not target_path.strip():
            return f"{prefix}.target_path must be a non-empty string"

        contract = item.get("contract")
        if not isinstance(contract, dict):
            return f"{prefix}.contract must be an object"
        if not isinstance(contract.get("expected_symbols"), list):
            return f"{prefix}.contract.expected_symbols must be an array"
        if not isinstance(contract.get("forbidden_imports"), list):
            return f"{prefix}.contract.forbidden_imports must be an array"

    return None
