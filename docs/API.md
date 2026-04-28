# API Reference

## `aura_harness.llm`

### `OllamaClient`

#### Properties

- `default_model`: (read-only) The model name used when `generate` is called without an explicit model.

## `aura_harness.ui`

### `MainWindow`

The main application window.

- `__init__(client: OllamaClient, generator: CandidateGenerator)`: Initializes the window and wires it to the provided client and generator.

### `BatchWorker`

A `QThread` for running candidate generation asynchronously.

- `batch_complete`: Signal emitted with a `CandidateBatch` object on success.
- `batch_error`: Signal emitted with an error message string on catastrophic failure.
