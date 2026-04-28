# API Reference

## `aura_harness.llm`

### `OllamaClient`

#### Methods

- `generate(prompt: str, model: str | None = None, ...)`: Runs a non-streaming generation against `/api/generate`.
- `chat(messages: list[dict[str, str]], model: str | None = None, ...)`: Runs a non-streaming chat completion against `/api/chat`.
- `list_models()`: Returns names of locally installed models.
- `health_check()`: Returns True if the server is reachable.

#### Properties

- `default_model`: (read-only) The model name used when `generate` is called without an explicit model.
- `base_url`: (read-only) Root URL of the Ollama server.
- `timeout_seconds`: (read-only) Per-request timeout for non-health requests.
- `log_path`: (read-only) Path to the JSONL call log.

## `aura_harness.lab`

### `Candidate`

A single generation attempt.

- `index`: Position in the batch.
- `text`: The completion text (if success).
- `error`: Error message (if failure).
- `is_success`: Boolean indicating success.
- `wall_duration_ms`: Time taken for this candidate.

### `CandidateBatch`

A collection of candidates from a single prompt.

- `batch_id`: Unique identifier.
- `candidates`: List of `Candidate` objects.
- `success_count`: Number of successful candidates.
- `total_wall_duration_ms`: Time taken for the whole batch.

## `aura_harness.workspace`

### `WorkspaceManager`

Manages the project workspace.

- `__init__(root: Path)`: Initializes the manager with the given root path.
- `root`: (read-only) The current workspace root path.
- `set_root(root: Path)`: Changes the active workspace root.
- `list_files()`: Returns a list of relative paths to files in the workspace (excluding ignored directories).
- `read_file(rel_path: Path)`: Reads a file's content as a UTF-8 string.
- `write_file(rel_path: Path, content: str)`: Writes content to a file in the workspace.

## `aura_harness.planner`

### `Turn`

A single message in a conversation.
- `role`: "system", "user", or "assistant".
- `content`: The message text.
- `timestamp`: Creation time.

### `ConversationState`

The full state of a planning session.
- `session_id`: Unique identifier.
- `model`: The model used for this session.
- `system_prompt`: The instructions for the planner.
- `turns`: A tuple of `Turn` objects.

### `PlannerClient`

Stateless service for chatting with a planner model.
- `__init__(client: OllamaClient)`: Initializes the planner.
- `chat(state: ConversationState) -> Turn`: Sends the conversation history to Ollama and returns the next assistant turn.

## `aura_harness.ui`

### `MainWindow`

The main application window.

- `__init__(client: OllamaClient, generator: CandidateGenerator, workspace: WorkspaceManager, planner: PlannerClient)`: Initializes the window and wires it to its collaborators.

### `PlannerWidget`

Conversational interface for spec planning.
- `commit_requested`: Signal emitted with a `ConversationState` when "Commit & Generate" is clicked.

### `PlannerWorker`

A `QThread` for running planner chat completions asynchronously.
- `__init__(planner: PlannerClient, state: ConversationState)`: Initializes the worker.
- `finished`: Signal emitted with the new `Turn` on success.
- `error`: Signal emitted with an error message string on failure.

### `ScaffoldDialog`

A modal dialog for creating a new project from a template.

- `__init__(default_parent_dir: Path, parent: QWidget | None = None)`: Initializes the dialog.
- `created_path`: (read-only) The `Path` of the successfully created project, or `None`.

### BatchWorker

A `QThread` for running candidate generation and scoring asynchronously.

- `__init__(generator, prompt, n, model, spec=None)`: Initializes the worker.
- `batch_complete`: Signal emitted with a `CandidateBatch` object on success (when `spec` is `None`).
- `scored_complete`: Signal emitted with a `ScoredBatch` object on success (when `spec` is provided).
- `batch_error`: Signal emitted with an error message string on catastrophic failure.

## `aura_harness.scoring`

### `ValidationSpec`

Defines requirements for code validation.
- `expected_symbols`: Tuple of names that must be defined in the code.
- `test_code`: Optional Python source code to execute with the candidate.
- `test_timeout_seconds`: Timeout for test execution.

### `ValidationResult`

Result of a single validation attempt.
- `passed`: Boolean indicating overall success.
- `parse_ok`: Boolean indicating if the code parsed successfully.
- `test_ok`: Boolean or None indicating if tests passed.

### `ScoredCandidate`

A candidate paired with its extraction and validation results.
- `candidate`: The original `Candidate`.
- `extracted_code`: The code extracted from the candidate.
- `validation`: The `ValidationResult` for the candidate.

### `ScoredBatch`

A batch of scored candidates.
- `batch`: The original `CandidateBatch`.
- `spec`: The `ValidationSpec` used for scoring.
- `scored`: Tuple of `ScoredCandidate` objects.
- `pass_rate`: Percentage of candidates that passed validation.

### Functions

- `extract_code(text: str)`: Returns `(code, method)` or `(None, None)`.
- `validate(code: str, spec: ValidationSpec)`: Performs parsing, symbol check, and test execution.
- `score_batch(batch: CandidateBatch, spec: ValidationSpec)`: Orchestrates extraction and validation for a whole batch.

## `aura_harness.scaffold`

### `Template`

A project scaffold template definition.

- `id`: Unique identifier.
- `label`: Human-readable name.
- `description`: Short summary of the template.
- `files`: Dictionary mapping relative paths to file contents.

### Functions

- `scaffold(template_id: str, name: str, parent_dir: Path) -> Path`: Materializes a template on disk.
- `get_template(template_id: str) -> Template`: Retrieves a template by its ID.

### Constants

- `TEMPLATES`: A tuple of all registered `Template` objects.

## `aura_harness.bench`

### `BenchConfig`

Static configuration for a bench session.
- `task`: Task name.
- `runs`: Number of batches.
- `n`: Candidates per batch.
- `model`: Model name.
- `timeout_seconds`: Verifier timeout.
- `harness_git_sha`: Git SHA of the harness.
- `start_time`: ISO timestamp.

### `CandidateResult`

Outcome for a single candidate.
- `passed`: Boolean indicating if the verifier passed.
- `extraction_method`: Method used for code extraction.
- `verify_stderr`: Captured stderr on failure.
- `latency_ms`: Generation time.

### `RunResult`

Outcome for a full run (batch).
- `run_index`: Index of the run.
- `batch_id`: ID of the candidate batch.
- `candidates`: Tuple of `CandidateResult`.

### `SessionSummary`

Aggregate results for a session.
- `total_candidates`: Total count.
- `passes`: Number of passing candidates.
- `pass_rate`: Ratio of passes to total.
- `mean_latency_ms`: Average generation time.

### Functions

- `run_bench(task: str, runs: int, n: int, model: str, ...) -> tuple[Path, SessionSummary]`: Executes a benchmark session and returns the session directory and summary.

