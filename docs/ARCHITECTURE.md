# Architecture

## Component Overview

- **`OllamaClient`**: Low-level client for interacting with the Ollama API. Manages defaults such as the `default_model`. Supports both completion and chat endpoints, as well as explicit model unloading to manage VRAM.
- **`PlannerClient`**: Stateless service for managing conversational interactions with a planner model.
- **`CandidateGenerator` (the "lab")**: Orchestrates multiple parallel calls to `OllamaClient` to generate multiple candidates for a single prompt.
- **`MainWindow` (UI)**: The main graphical interface, providing controls for model selection, workspace management, candidate count (N), and prompt submission.
- **`PlannerWidget` (UI)**: A conversational interface for interacting with the `PlannerClient` to refine a project specification before generation.
- **`WorkspaceManager`**: Manages a local directory for file operations. Provides file listing, reading (for context injection), and writing (for applying candidates).
- **`Scaffold Layer`**: Provides templates and logic to bootstrap new projects. Includes `scaffold()` function and `Template` definitions.
- **`BatchWorker` (UI Threading)**: A `QThread` that handles the execution of the `CandidateGenerator` and optional `Scorer` off the main GUI thread.
- **`PlannerWorker`** (UI Threading): A `QThread` that handles asynchronous calls to the `PlannerClient`.
- **`CriticClient`**: Stateless service that reviews generated code against a specification. Produces structured critiques (violations and suggestions) using a reasoning model.
- **Benchmark Harness**:
    - **`Runner`**: Orchestrates generation against a task spec, optionally invokes the `CriticClient` for reflexion, and invokes a standalone `verify.py` script for each candidate.
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
    - **Reflexion (Optional)**: If a candidate fails verification and `--critic-rounds > 0`:
        - `CriticClient` reviews the code against `spec.md`.
        - A follow-up prompt with critique is sent back to `CandidateGenerator`.
        - **One-Way Ratchet**: The runner tracks the "prior-best" round based on a gradient signal (pass-count). A retry only replaces the prior-best if it strictly improves the number of passed tests.
        - This repeats up to `max_rounds`.
    - Each candidate is extracted via `Extractor`.
    - Extracted code is written to a temporary file and tested by the task's `verify.py` subprocess.
    - **Verifier Protocol**: The `verify.py` script receives a report path as its third argument. It writes a structured JSON report (`tests_passed`, `tests_total`, `failures`) which the runner uses for the gradient signal. A fallback is provided for backward compatibility.
3. **Persistence**:
    - Config, raw batches, result details (per round), and aggregate summaries are written to a timestamped folder in `sessions/`.

### VRAM Management

To prevent VRAM spillover and maintain high generation throughput, the Benchmark Runner enforces a "one model resident" invariant. It explicitly unloads models from Ollama's memory after each phase:
- After batch generation (unloads the coder model).
- After each reflexion review (unloads the critic model).
- After reflexion retries (unloads the coder model).
- At the end of the session (cleans the GPU).

This is achieved via `OllamaClient.unload_model()`, which utilizes Ollama's eager eviction pattern (empty prompt + `keep_alive: 0`).

### CLI Interface

```bash
python -m aura_harness.bench --task duplicate_finder --runs 3 --n 3 --model qwen2.5-coder:7b --critic-rounds 2
```
