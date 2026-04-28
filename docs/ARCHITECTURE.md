# Architecture

## Component Overview

- **`OllamaClient`**: Low-level client for interacting with the Ollama API. Manages defaults such as the `default_model`.
- **`CandidateGenerator` (the "lab")**: Orchestrates multiple parallel calls to `OllamaClient` to generate multiple candidates for a single prompt.
- **`MainWindow` (UI)**: The main graphical interface, providing controls for model selection, workspace management, candidate count (N), and prompt submission.
- **`WorkspaceManager`**: Manages a local directory for file operations. Provides file listing, reading (for context injection), and writing (for applying candidates).
- **`BatchWorker` (UI Threading)**: A `QThread` that handles the execution of the `CandidateGenerator` and optional `Scorer` off the main GUI thread.
- **Scoring Layer**:
    - **`Extractor`**: Extracts code blocks from raw LLM responses using regex and AST parsing.
    - **`Validator`**: Executes extracted code in a subprocess to check for syntax, required symbols, and test pass/fail results.
    - **`Scorer`**: High-level orchestrator that takes a `CandidateBatch` and produces a `ScoredBatch`.

## Data Flow

1. User selects context files in the sidebar and enters a prompt in the `MainWindow`.
2. `MainWindow` reads selected files via `WorkspaceManager` and prepends them to the prompt.
3. `MainWindow` constructs and starts a `BatchWorker`, optionally passing a `ValidationSpec`.
4. `BatchWorker` calls `CandidateGenerator.generate()`.
5. `CandidateGenerator` uses `OllamaClient` to fetch completions.
6. If a `ValidationSpec` was provided, `BatchWorker` calls `score_batch()` on the results.
7. Results (raw or scored) are returned to `MainWindow` via signals and rendered as interactive widgets.
8. User can "Apply" a candidate, writing its code back to the workspace via `WorkspaceManager`.

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
