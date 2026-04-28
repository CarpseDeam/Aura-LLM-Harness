# Architecture

## Component Overview

- **`OllamaClient`**: Low-level client for interacting with the Ollama API. Manages defaults such as the `default_model`. Supports both completion and chat endpoints.
- **`PlannerClient`**: Stateless service for managing conversational interactions with a planner model.
- **`CandidateGenerator` (the "lab")**: Orchestrates multiple parallel calls to `OllamaClient` to generate multiple candidates for a single prompt.
- **`MainWindow` (UI)**: The main graphical interface, providing controls for model selection, workspace management, candidate count (N), and prompt submission.
- **`PlannerWidget` (UI)**: A conversational interface for interacting with the `PlannerClient` to refine a project specification before generation.
- **`WorkspaceManager`**: Manages a local directory for file operations. Provides file listing, reading (for context injection), and writing (for applying candidates).
- **`Scaffold Layer`**: Provides templates and logic to bootstrap new projects. Includes `scaffold()` function and `Template` definitions.
- **`BatchWorker` (UI Threading)**: A `QThread` that handles the execution of the `CandidateGenerator` and optional `Scorer` off the main GUI thread.
- **`PlannerWorker` (UI Threading)**: A `QThread` that handles asynchronous calls to the `PlannerClient`.
- **Benchmark Harness**:
    - **`Runner`**: Orchestrates generation against a task spec and invokes a standalone `verify.py` script for each candidate.
    - **Session Manager**: Manages session-specific directories and persists configuration, raw batches, and aggregate summaries.
- **Scoring Layer**:
    - **`Extractor`**: Extracts code blocks from raw LLM responses using regex and AST parsing.
    - **`Validator`**: Executes extracted code in a subprocess to check for syntax, required symbols, and test pass/fail results.
    - **`Scorer`**: High-level orchestrator that takes a `CandidateBatch` and produces a `ScoredBatch`.

## Data Flow

1. **Planning Phase**:
    - User chats with a planner model via the `PlannerWidget`.
    - `PlannerWidget` uses `PlannerWorker` to fetch assistant turns from `PlannerClient`.
    - Once satisfied, the user clicks "Commit & Generate".
2. **Generation Phase**:
    - User selects context files in the sidebar and enters a prompt in the `MainWindow`.
    - `MainWindow` reads selected files via `WorkspaceManager` and prepends them to the prompt.
    - `MainWindow` constructs and starts a `BatchWorker`, optionally passing a `ValidationSpec`.
    - `BatchWorker` calls `CandidateGenerator.generate()`.
    - `CandidateGenerator` uses `OllamaClient` to fetch completions.
3. **Validation & Application**:
    - If a `ValidationSpec` was provided, `BatchWorker` calls `score_batch()` on the results.
    - Results (raw or scored) are returned to `MainWindow` via signals and rendered as interactive widgets.
    - User can "Apply" a candidate, writing its code back to the workspace via `WorkspaceManager`.

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

## Benchmark Harness

The benchmark harness provides a systematic way to measure model performance against fixed tasks.

### Data Flow

1. **Task Loading**:
    - Runner loads `spec.md`, `verify.py`, and `fixtures/` from `bench/tasks/<task_name>/`.
2. **Execution Loop**:
    - For each run (batch), `CandidateGenerator` produces N candidates.
    - Each candidate is extracted via `Extractor`.
    - Extracted code is written to a temporary file and tested by the task's `verify.py` subprocess.
3. **Persistence**:
    - Config, raw batches, result details, and aggregate summaries are written to a timestamped folder in `sessions/`.

### CLI Interface

```bash
python -m aura_harness.bench --task duplicate_finder --runs 3 --n 3 --model qwen2.5-coder:7b
```
