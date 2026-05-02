"""Frozen dataclasses for the run-state schema.

Every type here is immutable and hashable: collection fields are tuples,
not lists, so a value cannot drift after construction. Stations consume a
:class:`RunState` (or a piece of it) and produce a new typed value with
genealogy fields filled in. The :mod:`aura_harness.state` module is types-
and-one-builder; consumer behavior lives in the stations that read these
types in later dispatches.

See ``docs/STATE_SCHEMA.md`` for the design rationale, the genealogy
pattern, and the ``parent_artifact_ids`` cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FileSymbols:
    """Top-level symbols and imports parsed from a single Python file.

    Attributes:
        path: File path relative to the workspace root.
        symbols: Names of module-scope ``def`` / ``async def`` / ``class``
            definitions, sorted alphabetically.
        imports: Module-scope imported names, sorted alphabetically. See
            :func:`aura_harness.state.repo_map.build_repo_map` for the
            naming convention (dotted modules, relative imports, etc).
    """

    path: Path
    symbols: tuple[str, ...]
    imports: tuple[str, ...]


@dataclass(frozen=True)
class RepoMap:
    """A static, AST-derived snapshot of a workspace's Python files.

    Attributes:
        files: One :class:`FileSymbols` per ``.py`` file under the
            workspace, sorted by path. Files that fail to parse are
            silently skipped at build time.
    """

    files: tuple[FileSymbols, ...]


@dataclass(frozen=True)
class TaskSpec:
    """A user-facing task to be solved by the harness.

    Attributes:
        task_id: Stable identifier for the task; used as the ``input_ref``
            on any :class:`Plan` produced from this spec.
        description: Free-form prose description of what the task wants.
        workspace_path: Filesystem root the task operates against.
        constraints: Optional hard requirements (e.g. forbidden imports,
            required public symbols) the planner must respect.
        context_files: Optional pinned files the planner should treat as
            primary context. Paths are absolute or relative to
            ``workspace_path``; consumers decide.
        repo_map: Optional pre-computed repository snapshot. ``None``
            when the task is being constructed before the workspace has
            been walked.
    """

    task_id: str
    description: str
    workspace_path: Path
    constraints: tuple[str, ...] = ()
    context_files: tuple[Path, ...] = ()
    repo_map: RepoMap | None = None


@dataclass(frozen=True)
class SliceContract:
    """Static, mechanical expectations for a single slice's output.

    Attributes:
        expected_symbols: Symbol names the slice's :class:`CodeArtifact`
            must expose at module scope.
        forbidden_imports: Modules the slice's code must not import.
            Matched by full dotted name.
    """

    expected_symbols: tuple[str, ...] = ()
    forbidden_imports: tuple[str, ...] = ()


@dataclass(frozen=True)
class Slice:
    """One unit of work in a :class:`Plan`.

    Attributes:
        slice_id: Stable identifier, unique within the parent plan.
        description: Free-form prose for the coder.
        target_files: Files this slice is allowed to write.
        depends_on: ``slice_id`` values that must be completed first.
        contract: Mechanical acceptance criteria; defaults to an empty
            contract.
    """

    slice_id: str
    description: str
    target_files: tuple[Path, ...]
    depends_on: tuple[str, ...] = ()
    contract: SliceContract = SliceContract()


@dataclass(frozen=True)
class Plan:
    """A planner's decomposition of a :class:`TaskSpec` into slices.

    Genealogy fields (``produced_by_*``, ``produced_at``, ``seed``,
    ``input_ref``) trace this plan back to the planner station, the
    backend that ran the model, and the :class:`TaskSpec` consumed.

    Attributes:
        plan_id: Stable identifier; used as the ``input_ref`` on any
            :class:`CodeArtifact` produced from this plan.
        slices: Ordered slices the coder will execute.
        rationale: Human-readable summary of why the plan is shaped this
            way; not consumed mechanically.
        produced_by_station: Name of the station that emitted the plan
            (e.g. ``"planner"``).
        produced_by_backend: Name of the backend used (e.g.
            ``"local-ollama"``, ``"openai-deepseek"``).
        produced_at: UTC timestamp at the moment of emission.
        seed: Sampling seed used by the backend, if any.
        input_ref: ``task_id`` of the consumed :class:`TaskSpec`.
    """

    plan_id: str
    slices: tuple[Slice, ...]
    rationale: str
    produced_by_station: str
    produced_by_backend: str
    produced_at: datetime
    seed: int | None
    input_ref: str


@dataclass(frozen=True)
class FileWrite:
    """A single file's path and full contents, as produced by a coder.

    Attributes:
        path: Destination path. Absolute, or relative to the workspace.
        content: Full file content the coder intends to write.
    """

    path: Path
    content: str


@dataclass(frozen=True)
class CodeArtifact:
    """The coder station's output for one slice (or one integration).

    ``parent_artifact_ids`` distinguishes three production modes:

    - Round 0 (first attempt at a slice): empty tuple.
    - Critic retry: one parent — the prior round's artifact.
    - Integration: many parents — the per-slice artifacts being merged.

    Attributes:
        artifact_id: Stable identifier; used as the ``input_ref`` on any
            :class:`CritiqueReport` produced for this artifact.
        slice_id: The :class:`Slice` this artifact targets. For an
            integration artifact this is conventionally the integrating
            slice's id; consumers decide the convention.
        files: Files the coder wants to write.
        parent_artifact_ids: Provenance — see the three cases above.
        critic_round: ``0`` for the first attempt; incremented per
            critic-driven retry.
        produced_by_station: Name of the station that emitted the
            artifact (e.g. ``"coder"``, ``"integrator"``).
        produced_by_backend: Name of the backend used.
        produced_at: UTC timestamp at the moment of emission.
        seed: Sampling seed used by the backend, if any.
        input_ref: ``plan_id`` of the consumed :class:`Plan`.
    """

    artifact_id: str
    slice_id: str
    files: tuple[FileWrite, ...]
    parent_artifact_ids: tuple[str, ...]
    critic_round: int
    produced_by_station: str
    produced_by_backend: str
    produced_at: datetime
    seed: int | None
    input_ref: str


@dataclass(frozen=True)
class CritiqueReport:
    """The critic station's structured review of a :class:`CodeArtifact`.

    Attributes:
        critique_id: Stable identifier for this review.
        artifact_id: The artifact under review. Mirrors ``input_ref``;
            kept as its own field so consumers do not need to know the
            genealogy convention to find the reviewed artifact.
        passed: Whether the critic believes the artifact satisfies its
            slice contract.
        violations: Concrete, mechanical contract issues found.
        suggestions: Terse fixes addressing each violation.
        produced_by_station: Name of the station that emitted the
            critique (e.g. ``"critic"``).
        produced_by_backend: Name of the backend used.
        produced_at: UTC timestamp at the moment of emission.
        seed: Sampling seed used by the backend, if any.
        input_ref: ``artifact_id`` of the consumed :class:`CodeArtifact`.
    """

    critique_id: str
    artifact_id: str
    passed: bool
    violations: tuple[str, ...]
    suggestions: tuple[str, ...]
    produced_by_station: str
    produced_by_backend: str
    produced_at: datetime
    seed: int | None
    input_ref: str


@dataclass(frozen=True)
class RunState:
    """The full state threaded through a single harness run.

    A run begins with only ``task`` populated; later stations append to
    ``plan``, ``artifacts``, and ``critiques`` by constructing a new
    :class:`RunState` (the type is frozen — every update is a copy).

    Attributes:
        run_id: Stable identifier for the run.
        task: The originating :class:`TaskSpec`.
        plan: Set once the planner has emitted a :class:`Plan`.
        artifacts: Every :class:`CodeArtifact` produced so far, in
            production order. Includes round-0, retry, and integration
            artifacts.
        critiques: Every :class:`CritiqueReport` produced so far.
    """

    run_id: str
    task: TaskSpec
    plan: Plan | None = None
    artifacts: tuple[CodeArtifact, ...] = ()
    critiques: tuple[CritiqueReport, ...] = ()
