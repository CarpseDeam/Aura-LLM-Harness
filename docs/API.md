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

## `aura_harness.ui`

### `MainWindow`

The main application window.

- `__init__(client: OllamaClient, generator: CandidateGenerator)`: Initializes the window and wires it to the provided client and generator.

### `BatchWorker`

A `QThread` for running candidate generation asynchronously.

- `batch_complete`: Signal emitted with a `CandidateBatch` object on success.
- `batch_error`: Signal emitted with an error message string on catastrophic failure.
