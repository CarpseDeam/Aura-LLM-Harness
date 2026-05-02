"""Worker station — generate code for one slice, then loop through the critic.

The worker drives :class:`CandidateGenerator` to produce ``n`` candidates,
picks the first one whose extracted code clears the slice contract (AST
parse, expected symbols present, no forbidden imports), and emits it as a
:class:`CodeArtifact`. With ``max_critic_rounds > 0`` it then runs the
:class:`CriticStation` on that artifact and, on a failing critique,
regenerates with a reflexion prompt — repeating until the critic passes
or the round budget is exhausted. Only the final artifact is returned;
intermediate retry artifacts are constructed (so ``parent_artifact_ids``
on later rounds carries a real id) and discarded.
"""
from __future__ import annotations

import ast
import uuid
from pathlib import Path
from typing import Final

from aura_harness.backend import Backend
from aura_harness.lab.generator import CandidateGenerator
from aura_harness.lab.models import Candidate, CandidateBatch
from aura_harness.scoring import ValidationSpec, extract_code, validate
from aura_harness.state import CodeArtifact, CritiqueReport, FileWrite, RunState, Slice, SliceContract
from aura_harness.stations.base import Station
from aura_harness.stations.critic import CriticStation

_DEFAULT_TARGET_FILENAME: Final[str] = "candidate_module.py"

_REFLEXION_PROMPT_TEMPLATE: Final[str] = (
    "{spec}\n"
    "\n"
    "---\n"
    "\n"
    "## Previous attempt\n"
    "\n"
    "You previously produced this candidate, which a reviewer rejected:\n"
    "\n"
    "```python\n"
    "{code}\n"
    "```\n"
    "\n"
    "## Reviewer feedback\n"
    "\n"
    "Violations:\n"
    "{violations}\n"
    "\n"
    "Suggestions:\n"
    "{suggestions}\n"
    "\n"
    "## Instruction\n"
    "\n"
    "Produce a corrected version of the module that addresses every "
    "violation above. Output the entire corrected module in a single "
    "fenced ```python``` block. Do not include explanations outside the "
    "code block."
)
_NO_ITEMS_PLACEHOLDER: Final[str] = "- (none provided)"


class WorkerStation(Station):
    """Generate code for one :class:`Slice`, looping a single-pass critic."""

    def __init__(
        self,
        backend: Backend,
        candidate_generator: CandidateGenerator,
        critic_station: CriticStation,
        *,
        max_critic_rounds: int,
        n: int,
        model: str | None = None,
        name: str = "worker",
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if max_critic_rounds < 0:
            raise ValueError("max_critic_rounds must be >= 0")
        super().__init__(name=name, backend=backend)
        self._candidate_generator = candidate_generator
        self._critic_station = critic_station
        self._max_critic_rounds = max_critic_rounds
        self._n = n
        self._model = model

    def run(self, slice: Slice, state: RunState) -> CodeArtifact:
        """Produce a final :class:`CodeArtifact` for ``slice``."""
        del state  # state isn't read here; LinearExecutor handles append
        spec_text = slice.description

        batch = self._candidate_generator.generate(
            spec_text,
            n=self._n,
            model=self._model,
        )
        artifact = self._build_artifact(
            batch=batch,
            slice=slice,
            critic_round=0,
            parent_artifact_ids=(),
        )

        for round_index in range(1, self._max_critic_rounds + 1):
            report = self._critic_station.run(artifact, slice)
            if report.passed:
                return artifact
            reflexion_prompt = _build_reflexion_prompt(
                spec=spec_text,
                code=_artifact_code(artifact),
                report=report,
            )
            batch = self._candidate_generator.generate(
                reflexion_prompt,
                n=self._n,
                model=self._model,
            )
            artifact = self._build_artifact(
                batch=batch,
                slice=slice,
                critic_round=round_index,
                parent_artifact_ids=(artifact.artifact_id,),
            )

        return artifact

    def _build_artifact(
        self,
        *,
        batch: CandidateBatch,
        slice: Slice,
        critic_round: int,
        parent_artifact_ids: tuple[str, ...],
    ) -> CodeArtifact:
        """Pick the best candidate from ``batch`` and wrap it as a :class:`CodeArtifact`."""
        picked = _pick_best_candidate(batch, slice.contract)
        code = picked.code if picked is not None else ""
        seed = picked.candidate.seed if picked is not None else None
        target_path = (
            slice.target_files[0]
            if slice.target_files
            else Path(_DEFAULT_TARGET_FILENAME)
        )
        files: tuple[FileWrite, ...] = (
            (FileWrite(path=target_path, content=code),) if code else ()
        )
        return CodeArtifact(
            artifact_id=uuid.uuid4().hex,
            slice_id=slice.slice_id,
            files=files,
            parent_artifact_ids=parent_artifact_ids,
            critic_round=critic_round,
            **self._genealogy(seed=seed, input_ref=slice.slice_id),
        )


class _PickedCandidate:
    """A candidate paired with its extracted code, kept for genealogy stamping."""

    __slots__ = ("candidate", "code")

    def __init__(self, candidate: Candidate, code: str) -> None:
        self.candidate = candidate
        self.code = code


def _pick_best_candidate(
    batch: CandidateBatch, contract: SliceContract
) -> _PickedCandidate | None:
    """Return the first candidate whose code clears ``contract``.

    Falls back to the first candidate that at least produced extractable
    code so the critic still has something to review on a retry. Returns
    ``None`` only if no candidate produced extractable code.
    """
    extracted: list[_PickedCandidate] = []
    for candidate in batch.candidates:
        text = candidate.text
        if not text:
            continue
        code, _ = extract_code(text)
        if code is None:
            continue
        extracted.append(_PickedCandidate(candidate=candidate, code=code))

    for picked in extracted:
        if _passes_contract(picked.code, contract):
            return picked
    return extracted[0] if extracted else None


def _passes_contract(code: str, contract: SliceContract) -> bool:
    """AST parse + expected_symbols (via scoring.validate) + forbidden_imports."""
    spec = ValidationSpec(expected_symbols=contract.expected_symbols)
    result = validate(code, spec)
    if not result.passed:
        return False
    if not contract.forbidden_imports:
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    forbidden = set(contract.forbidden_imports)
    for module_name in _top_level_imports(tree):
        if module_name in forbidden:
            return False
    return True


def _top_level_imports(tree: ast.Module) -> list[str]:
    """Module names from top-level ``import`` / ``from ... import`` statements."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def _artifact_code(artifact: CodeArtifact) -> str:
    """Pull the candidate code blob out of an artifact, or ``""`` if empty."""
    if not artifact.files:
        return ""
    return artifact.files[0].content


def _build_reflexion_prompt(*, spec: str, code: str, report: CritiqueReport) -> str:
    """Compose the follow-up prompt sent to the coder for a reflexion round."""
    return _REFLEXION_PROMPT_TEMPLATE.format(
        spec=spec,
        code=code,
        violations=_format_bullets(report.violations),
        suggestions=_format_bullets(report.suggestions),
    )


def _format_bullets(items: tuple[str, ...]) -> str:
    if not items:
        return _NO_ITEMS_PLACEHOLDER
    return "\n".join(f"- {item}" for item in items)
