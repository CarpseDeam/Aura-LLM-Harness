# Architecture

## Component Overview

- **`OllamaClient`**: Low-level client for interacting with the Ollama API. Manages defaults such as the `default_model`.
- **`CandidateGenerator` (the "lab")**: Orchestrates multiple parallel calls to `OllamaClient` to generate multiple candidates for a single prompt.
- **`MainWindow` (UI)**: The main graphical interface, providing controls for model selection, candidate count (N), and prompt submission.
- **`BatchWorker` (UI Threading)**: A `QThread` that handles the execution of the `CandidateGenerator` off the main GUI thread, preventing UI freezes during generation.

## Data Flow

1. User enters a prompt and selects parameters in the `MainWindow`.
2. `MainWindow` constructs and starts a `BatchWorker`.
3. `BatchWorker` calls `CandidateGenerator.generate()`.
4. `CandidateGenerator` uses `OllamaClient` to fetch completions.
5. `BatchWorker` emits signals back to `MainWindow` upon completion or failure.
6. `MainWindow` renders the results or errors in the output area.
