"""Integrator station — materialize a :class:`Plan`'s artifacts into files.

The integrator is the last stage before runtime. It takes the
:class:`CodeArtifact` set the workers produced for a :class:`Plan`'s
:class:`Slice` set and writes each artifact's source to its slice's
``target_path`` under a workspace directory, in dependency order.
After every write, ``.py`` files are compile-checked with
:mod:`py_compile`. The whole thing produces an
:class:`IntegrationResult`.

What this station explicitly does *not* do: invoke the model (the
``backend`` arg is held only for genealogy stamping), import the
written modules, run the resulting program, or re-validate slice
contracts (``expected_symbols`` / ``forbidden_imports``) — those are
worker-time concerns. If a worker emits an artifact that passed
worker validation but is structurally broken, that is a worker bug,
surfaced here as a compile error.
"""
from __future__ import annotations

import py_compile
import uuid
from pathlib import Path
from typing import Final

from aura_harness.backend import Backend
from aura_harness.state import (
    CodeArtifact,
    CompileError,
    IntegrationResult,
    Plan,
    Slice,
)
from aura_harness.stations.base import Station

_PY_SUFFIX: Final[str] = ".py"


class IntegratorStation(Station):
    """Write per-slice artifacts to disk and compile-check the result.

    The ``backend`` constructor arg is required by the :class:`Station`
    base for genealogy stamping but never receives a model call from
    this station. Pass the same backend the workers used so the
    integration result's ``produced_by_backend`` traces back to the
    same provider.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        name: str = "integrator",
    ) -> None:
        super().__init__(name=name, backend=backend)

    def run(
        self,
        plan: Plan,
        artifacts: tuple[CodeArtifact, ...],
        workspace_path: str,
    ) -> IntegrationResult:
        """Materialize ``artifacts`` for ``plan`` under ``workspace_path``.

        Never raises on a contract or compile failure — every problem is
        captured as a :class:`CompileError` inside the returned
        :class:`IntegrationResult`. Only unexpected I/O failures (disk
        full, permission denied on the workspace itself) propagate.
        """
        artifact_by_slice_id, contract_errors = _match_artifacts(plan, artifacts)
        if contract_errors:
            return self._build_result(
                plan=plan,
                workspace_path=workspace_path,
                written=(),
                errors=contract_errors,
            )

        ordered_slices, topo_error = _stable_topo_sort(plan.slices)
        if topo_error is not None:
            return self._build_result(
                plan=plan,
                workspace_path=workspace_path,
                written=(),
                errors=(topo_error,),
            )

        workspace_root = Path(workspace_path)
        workspace_root.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        errors: list[CompileError] = []
        for slice_ in ordered_slices:
            artifact = artifact_by_slice_id[slice_.slice_id]
            target_relpath = slice_.target_path
            if not target_relpath:
                errors.append(
                    CompileError(
                        file_path="",
                        error_type="ContractViolation",
                        message=(
                            f"slice '{slice_.slice_id}' has no target_path; "
                            "the planner must populate it"
                        ),
                        line=None,
                    )
                )
                continue

            content = _artifact_content(artifact)
            if content is None:
                errors.append(
                    CompileError(
                        file_path=target_relpath,
                        error_type="ContractViolation",
                        message=(
                            f"artifact for slice '{slice_.slice_id}' has no "
                            "files to write"
                        ),
                        line=None,
                    )
                )
                continue

            absolute_path = workspace_root / target_relpath
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            absolute_path.write_text(content, encoding="utf-8")
            written.append(target_relpath)

            if absolute_path.suffix == _PY_SUFFIX:
                compile_error = _compile_check(absolute_path, target_relpath)
                if compile_error is not None:
                    errors.append(compile_error)

        return self._build_result(
            plan=plan,
            workspace_path=workspace_path,
            written=tuple(written),
            errors=tuple(errors),
        )

    def _build_result(
        self,
        *,
        plan: Plan,
        workspace_path: str,
        written: tuple[str, ...],
        errors: tuple[CompileError, ...],
    ) -> IntegrationResult:
        return IntegrationResult(
            integration_id=uuid.uuid4().hex,
            success=not errors,
            workspace_path=workspace_path,
            written_files=written,
            compile_errors=errors,
            **self._genealogy(seed=None, input_ref=plan.plan_id),
        )


def _match_artifacts(
    plan: Plan, artifacts: tuple[CodeArtifact, ...]
) -> tuple[dict[str, CodeArtifact], tuple[CompileError, ...]]:
    """Pair each slice with its artifact, or return contract errors."""
    by_slice_id: dict[str, CodeArtifact] = {}
    duplicates: list[str] = []
    for artifact in artifacts:
        if artifact.slice_id in by_slice_id:
            duplicates.append(artifact.slice_id)
        else:
            by_slice_id[artifact.slice_id] = artifact

    errors: list[CompileError] = []
    if duplicates:
        errors.append(
            CompileError(
                file_path="",
                error_type="ContractViolation",
                message=(
                    "multiple artifacts target the same slice id(s): "
                    + ", ".join(sorted(set(duplicates)))
                ),
                line=None,
            )
        )

    plan_slice_ids = {s.slice_id for s in plan.slices}
    missing = sorted(plan_slice_ids - by_slice_id.keys())
    if missing:
        errors.append(
            CompileError(
                file_path="",
                error_type="ContractViolation",
                message=(
                    "no artifact provided for slice id(s): "
                    + ", ".join(missing)
                ),
                line=None,
            )
        )

    extra = sorted(by_slice_id.keys() - plan_slice_ids)
    if extra:
        errors.append(
            CompileError(
                file_path="",
                error_type="ContractViolation",
                message=(
                    "artifact slice id(s) not present in plan: "
                    + ", ".join(extra)
                ),
                line=None,
            )
        )

    if len(artifacts) != len(plan.slices) and not errors:
        errors.append(
            CompileError(
                file_path="",
                error_type="ContractViolation",
                message=(
                    f"artifact count ({len(artifacts)}) does not match "
                    f"slice count ({len(plan.slices)})"
                ),
                line=None,
            )
        )

    return by_slice_id, tuple(errors)


def _stable_topo_sort(
    slices: tuple[Slice, ...],
) -> tuple[tuple[Slice, ...], CompileError | None]:
    """Return ``slices`` topo-sorted by ``depends_on``, preserving input order."""
    id_to_slice = {s.slice_id: s for s in slices}
    if len(id_to_slice) != len(slices):
        return (), CompileError(
            file_path="",
            error_type="ContractViolation",
            message="plan contains duplicate slice ids",
            line=None,
        )
    for slice_ in slices:
        for dep in slice_.depends_on:
            if dep not in id_to_slice:
                return (), CompileError(
                    file_path="",
                    error_type="ContractViolation",
                    message=(
                        f"slice '{slice_.slice_id}' depends on unknown "
                        f"slice id '{dep}'"
                    ),
                    line=None,
                )

    emitted: set[str] = set()
    ordered: list[Slice] = []
    pending = list(slices)
    while pending:
        progressed = False
        next_pending: list[Slice] = []
        for slice_ in pending:
            if all(dep in emitted for dep in slice_.depends_on):
                ordered.append(slice_)
                emitted.add(slice_.slice_id)
                progressed = True
            else:
                next_pending.append(slice_)
        if not progressed:
            cycle = sorted(s.slice_id for s in next_pending)
            return (), CompileError(
                file_path="",
                error_type="ContractViolation",
                message=(
                    "plan slices form a dependency cycle involving: "
                    + ", ".join(cycle)
                ),
                line=None,
            )
        pending = next_pending
    return tuple(ordered), None


def _artifact_content(artifact: CodeArtifact) -> str | None:
    """Return the artifact's source text, or ``None`` if it has no files."""
    if not artifact.files:
        return None
    return artifact.files[0].content


def _compile_check(absolute_path: Path, relative_path: str) -> CompileError | None:
    """Run ``py_compile`` on ``absolute_path``; return ``None`` on success."""
    try:
        py_compile.compile(str(absolute_path), doraise=True)
    except py_compile.PyCompileError as exc:
        line: int | None = None
        underlying = getattr(exc, "exc_value", None)
        if isinstance(underlying, SyntaxError) and underlying.lineno is not None:
            line = int(underlying.lineno)
        message = exc.msg.strip() if getattr(exc, "msg", None) else str(exc).strip()
        return CompileError(
            file_path=relative_path,
            error_type="SyntaxError",
            message=message or "compile failed",
            line=line,
        )
    return None
