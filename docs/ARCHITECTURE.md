# Architecture

## Component Overview

- **`OllamaClient`**: Low-level client for interacting with the Ollama API. Manages defaults such as the `default_model`.
- **`CandidateGenerator` (the "lab")**: Orchestrates multiple parallel calls to `OllamaClient` to generate multiple candidates for a single prompt.
- **`MainWindow` (UI)**: The main graphical interface, providing controls for model selection, candidate count (N), and prompt submission.
- **`BatchWorker` (UI Threading)**: A `QThread` that handles the execution of the `CandidateGenerator` off the main GUI thread, preventing UI freezes during generation.
- **Scoring Layer**:
    - **`Extractor`**: Extracts code blocks from raw LLM responses using regex and AST parsing.
    - **`Validator`**: Executes extracted code in a subprocess to check for syntax, required symbols, and test pass/fail results.
    - **`Scorer`**: High-level orchestrator that takes a `CandidateBatch` and produces a `ScoredBatch`.

## Data Flow

1. User enters a prompt and selects parameters in the `MainWindow`.
2. `MainWindow` constructs and starts a `BatchWorker`.
3. `BatchWorker` calls `CandidateGenerator.generate()`.
4. `CandidateGenerator` uses `OllamaClient` to fetch completions.
5. (CLI-only for now) The scoring layer takes the `CandidateBatch`, extracts code, and validates it against a `ValidationSpec`.
6. Results are returned to the caller (UI or CLI) for rendering or final output.

## CLI Interface

The scoring layer can be invoked directly via the CLI for batch evaluation:

```bash
python -m aura_harness.scoring "PROMPT" \
    --n 5 \
    --model qwen2.5-coder:7b \
    --expect add \
    --test "assert add(1,2)==3"
```

For more complex tests, use `--test-file` to load Python code from a file:

```bash
python -m aura_harness.scoring "PROMPT" \
    --expect add \
    --test-file tests/my_tests.py
```

`--test` and `--test-file` are mutually exclusive. Use `--test-file` for multiline tests to avoid shell mangling.

This performs generation, extraction, and validation in one command.
