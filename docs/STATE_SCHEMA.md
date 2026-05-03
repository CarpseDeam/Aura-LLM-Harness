# State Schema

This document describes the typed object graph that every Station in the
harness reads from and writes to. The implementation lives in
`aura_harness/state/`; this is the prose explanation of what is there
and why.

## 1. Why this schema exists

Stations talk to each other through values, not through ad-hoc
dictionaries or kwargs. A planner emits a `Plan`, a coder consumes that
`Plan` and emits `CodeArtifact`s, a critic consumes a `CodeArtifact` and
emits a `CritiqueReport`. If those handoffs happen through `dict`s you
end up debugging shape drift forever — a key gets renamed in one
station, the consumer keeps reading the old key, nothing fails loudly,
and the run silently goes sideways. A typed schema turns those drifts
into `mypy` errors before the run starts. Stations become composable
because their inputs and outputs are concrete types, and the types
double as documentation: anyone reading a Station signature knows
exactly what it consumes and produces.

Two design choices make this schema more than a toy. The first is
**immutability**: every dataclass is `@dataclass(frozen=True)` and every
collection field is a tuple. A station cannot mutate an input it
received; if it wants to "update" the run state it has to construct a
new `RunState` with the new values. That removes a whole category of
distributed-debugging bugs where a downstream station's behavior depends
on whether some upstream station happened to mutate a shared list. The
second is **genealogy**: every produced thing carries enough metadata to
trace it back to the station, the backend, the seed, and the input
state element it consumed. When a critic flags a `CodeArtifact` as bad,
you can answer "which plan produced this, against which task, with
which seed, by which model" by walking pointers — no log archaeology
required. State is the substrate the rest of the architecture rests on;
spending the time to get its shape right pays off every time a Station
gets added.

## 2. The types

### `FileSymbols`

A static, AST-derived view of one Python file's top-level symbols and
imports. Produced by `build_repo_map`; consumed by the planner so it can
reason about what the workspace already has before slicing the work.

```python
@dataclass(frozen=True)
class FileSymbols:
    path: Path                  # relative to the workspace root
    symbols: tuple[str, ...]    # top-level def / async def / class names
    imports: tuple[str, ...]    # dotted module names, see repo_map.py
```

### `RepoMap`

A bag of `FileSymbols`, one per parseable `.py` file in the workspace.
Produced by `build_repo_map`; consumed by the planner and attached to a
`TaskSpec` as `repo_map`.

```python
@dataclass(frozen=True)
class RepoMap:
    files: tuple[FileSymbols, ...]   # sorted by path
```

### `TaskSpec`

The user-facing description of what to do. Produced upstream of the
harness (CLI, UI, or bench loader); consumed by the planner.

```python
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    description: str
    workspace_path: Path
    constraints: tuple[str, ...] = ()
    context_files: tuple[Path, ...] = ()
    repo_map: RepoMap | None = None
```

### `SliceContract`

Mechanical acceptance criteria for one slice's output. Static, set by
the planner when it emits the slice, read by the critic when it reviews
the resulting `CodeArtifact`.

```python
@dataclass(frozen=True)
class SliceContract:
    expected_symbols: tuple[str, ...] = ()
    forbidden_imports: tuple[str, ...] = ()
```

### `Slice`

One unit of coding work inside a `Plan`. Produced by the planner;
consumed by the coder station and (via its contract) the critic.

```python
@dataclass(frozen=True)
class Slice:
    slice_id: str
    description: str
    target_files: tuple[Path, ...]
    depends_on: tuple[str, ...] = ()
    contract: SliceContract = SliceContract()
    target_path: str | None = None
```

`target_path` is the slice's single primary destination as a workspace-relative
string — populated by the planner so a one-file-per-slice executor can write
the artifact without parsing `target_files`; optional so older `Slice` values
that only set `target_files` keep working.

### `Plan`

The full decomposition of a `TaskSpec` into ordered slices, with
genealogy. Produced by the planner station; consumed by the coder.

```python
@dataclass(frozen=True)
class Plan:
    plan_id: str
    slices: tuple[Slice, ...]
    rationale: str
    # genealogy
    produced_by_station: str
    produced_by_backend: str
    produced_at: datetime
    seed: int | None
    input_ref: str   # task_id
```

### `FileWrite`

A path-and-content pair: what the coder wants on disk for one file. Not
itself produced by a station — it is the payload inside a
`CodeArtifact`.

```python
@dataclass(frozen=True)
class FileWrite:
    path: Path
    content: str
```

### `CodeArtifact`

The coder station's output. One artifact per slice per round; one
artifact per integration step. The `parent_artifact_ids` tuple
encodes which mode produced it (see section 4). Produced by the coder
or integrator; consumed by the verifier and the critic.

```python
@dataclass(frozen=True)
class CodeArtifact:
    artifact_id: str
    slice_id: str
    files: tuple[FileWrite, ...]
    parent_artifact_ids: tuple[str, ...]
    critic_round: int
    # genealogy
    produced_by_station: str
    produced_by_backend: str
    produced_at: datetime
    seed: int | None
    input_ref: str   # plan_id
```

### `CritiqueReport`

The critic station's structured review of one `CodeArtifact`. Produced
by the critic; consumed by the runner to decide whether to retry, and
by the coder retry to focus its fix.

```python
@dataclass(frozen=True)
class CritiqueReport:
    critique_id: str
    artifact_id: str       # mirrors input_ref; lets consumers find the artifact
    passed: bool
    violations: tuple[str, ...]
    suggestions: tuple[str, ...]
    # genealogy
    produced_by_station: str
    produced_by_backend: str
    produced_at: datetime
    seed: int | None
    input_ref: str   # artifact_id
```

`artifact_id` is duplicated with `input_ref` on purpose. `input_ref` is
the genealogy field, present on every produced thing in the same shape.
`artifact_id` is the typed pointer: a consumer that knows it has a
`CritiqueReport` and wants the reviewed artifact should not have to
know "for critiques the input is an artifact." Both fields hold the
same value, but they document different intents.

### `RunState`

The whole state of one run, threaded through stations. Each station
that produces something returns a new `RunState` with the new value
appended.

```python
@dataclass(frozen=True)
class RunState:
    run_id: str
    task: TaskSpec
    plan: Plan | None = None
    artifacts: tuple[CodeArtifact, ...] = ()
    critiques: tuple[CritiqueReport, ...] = ()
```

## 3. Genealogy

Every produced thing — `Plan`, `CodeArtifact`, `CritiqueReport` —
carries the same five fields:

- `produced_by_station`: which station emitted it (`"planner"`,
  `"coder"`, `"critic"`, `"integrator"`, ...).
- `produced_by_backend`: which backend served the model call
  (`"local_ollama"`, `"openai-deepseek"`, ...). This is separate from
  the model name on purpose; a station can change models without
  changing backends, and a backend can swap models internally.
- `produced_at`: UTC timestamp of emission.
- `seed`: sampling seed if the backend supports seeding, else `None`.
  Combined with the input ref this is what makes a result reproducible.
- `input_ref`: the id of the state element this thing was produced
  *from*. For a `Plan` that's a `task_id`; for a `CodeArtifact` that's
  a `plan_id`; for a `CritiqueReport` that's an `artifact_id`.

Carrying these fields on every produced thing makes provenance walks
trivial. Worked example: a `CritiqueReport` says `passed=False` for
artifact `art-9c2`. To trace it:

1. The report's `input_ref` is `art-9c2`. Find that `CodeArtifact` in
   `RunState.artifacts`.
2. The artifact's `input_ref` is the plan, say `plan-04a`. Find that
   `Plan` in `RunState.plan` (or in a future plan history if we track
   replans).
3. The plan's `input_ref` is the task, say `task-duplicate-finder`.
   That's `RunState.task.task_id`.

At each hop you also know the station and backend that produced the
value — so when a critic flags a regression you can immediately answer
"did the coder change models on round 1?" or "was the planner using a
seeded run?". Without genealogy fields you would be cross-referencing
log files; with them, the answer is on the value itself.

## 4. The `parent_artifact_ids` pattern

`CodeArtifact.parent_artifact_ids` is a tuple, not a single optional
id. The tuple shape encodes three production modes:

**Round 0 — first attempt at a slice.** No prior artifact for this
slice; the tuple is empty.

```
plan-04a ──► slice-1 ──► artifact A   parent_artifact_ids = ()
                                      critic_round       = 0
```

**Critic retry.** A prior artifact for the same slice failed critic
review; the new artifact has exactly one parent.

```
artifact A (round 0) ──► critique (failed) ──► artifact B
                                               parent_artifact_ids = (A.id,)
                                               critic_round        = 1
```

**Integration.** Multiple per-slice artifacts are merged into one
combined artifact. Every input artifact appears as a parent.

```
artifact A (slice 0) ─┐
artifact B (slice 1) ─┼──► artifact M  parent_artifact_ids = (A.id, B.id, C.id)
artifact C (slice 2) ─┘                 critic_round        = 0
```

Using a tuple in all three cases means a consumer that wants the
provenance graph never has to special-case "is this a retry or an
integration?" — it walks `parent_artifact_ids` and the shape falls out.
A single optional parent id would force every integration consumer to
look at a different field; a list would lose immutability.

## 5. What's deferred

- **`prompt_version`**. We considered adding a `prompt_version` field to
  every produced thing so prompt regressions could be tracked alongside
  model regressions. Punted: it earns its place when we have enough
  prompt churn that we actually need to bisect. Today a prompt change
  is a code change and `git blame` is fine.
- **`IntegrationReport` as a separate type**. We considered a dedicated
  type for the integration step. Punted: the integration output is
  shaped exactly like a `CodeArtifact` (files plus genealogy), and
  `parent_artifact_ids` already encodes "this is an integration" via
  the multi-parent case. Adding a second type now would be premature.
- **Serialization helpers**. No `to_dict` / `from_dict` /  JSON
  round-trip in this dispatch. Bench persistence will need them; they
  go in when bench actually consumes them, not before.
- **Methods on the dataclasses**. None. Derived views, validation, and
  filtering all live outside the type. The state module is types-and-
  one-builder by design; adding methods invites the schema to become a
  framework.

## 6. The repo map

A `RepoMap` is a static snapshot of a workspace's Python files and what
they declare. `build_repo_map(workspace_path)` walks the workspace,
parses every `.py` file with `ast`, and emits one `FileSymbols` per
parseable file with top-level `def`/`class` names and module-scope
imports. Files that fail to parse are skipped (a partial work-in-
progress should not break the planner) with a debug-level log line.
Hidden directories, `__pycache__`, `.venv`, `node_modules`, and `.git`
are pruned during the walk.

The planner reads this map to know what symbols already exist before it
slices the work — useful for avoiding duplicate definitions, for
choosing which files a slice should target, and for surfacing
constraints back to the user. Today no station consumes it; the type
and the builder land first so the planner dispatch can wire them in
without churn.

We use the stdlib `ast` rather than tree-sitter on purpose. `ast` is
zero dependencies, ships with the interpreter, and is enough for the
Python-only workspaces we run today. Tree-sitter would buy us multi-
language support, which is a real but future concern; when we add a
second language the right move is to introduce a sibling builder
(`build_repo_map_ts`?) rather than retrofit one builder to handle
both.
