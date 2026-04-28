# API Reference

## `aura_harness.llm`

### `OllamaClient`

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

## `aura_harness.ui`

### `MainWindow`

The main application window.

- `__init__(client: OllamaClient, generator: CandidateGenerator, workspace: WorkspaceManager)`: Initializes the window and wires it to the provided client, generator, and workspace manager.

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

