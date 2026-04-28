# Architecture

## Component Overview

- **`OllamaClient`**: Low-level client for interacting with the Ollama API. Manages defaults such as the `default_model`.
- **`CandidateGenerator` (the "lab")**: Orchestrates multiple parallel calls to `OllamaClient` to generate multiple candidates for a single prompt. It utilizes the client's `default_model` for consistency when a specific model is not specified.
