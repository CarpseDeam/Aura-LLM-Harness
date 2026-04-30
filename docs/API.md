# API Reference

## `aura_harness.llm`

### `OllamaClient`

#### Methods

- `generate(prompt: str, model: str | None = None, ...)`: Runs a non-streaming generation against `/api/generate`.
- `chat(messages: list[dict[str, str]], model: str | None = None, ...)`: Runs a non-streaming chat completion against `/api/chat`.
- `unload_model(model_name: str)`: Evicts `model_name` from Ollama's resident set immediately.
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

## `aura_harness.critic`

### `Critique`

Structured review of a candidate.
- `passed`: Boolean indicating if the candidate passes review.
- `violations`: Tuple of strings describing contract violations.
- `suggestions`: Tuple of strings with suggested fixes.

### `CriticClient`

Stateless service for reviewing code.
- `__init__(client: OllamaClient, model: str = DEFAULT_CRITIC_MODEL)`: Initializes the critic.
- `review(spec: str, code: str, *, failures: tuple[str, ...] = (), verifier_stderr: str | None = None, tests_passed: int | None = None, tests_total: int | None = None) -> Critique`: Reviews the code against the spec, optionally using failure context (failed tests, verifier stderr, and gradient signal) to provide focused feedback. Returns a critique.

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
- `model`: Coder model name.
- `timeout_seconds`: Verifier timeout.
- `critic_rounds`: Maximum reflexion rounds (0 disables critic).
- `critic_model`: Reasoning model for the critic.
- `harness_git_sha`: Git SHA of the harness.
- `start_time`: ISO timestamp.

### `RoundResult`

Outcome for a single generation+verify attempt at a slot.
- `round_index`: 0 for original, 1+ for reflexion.
- `seed`: Sampling seed for this round.
- `passed`: Boolean indicating if every assertion passed.
- `tests_passed`: Number of assertions that passed.
- `tests_total`: Total number of assertions evaluated.
- `failures`: Tuple of names of failing assertions.
- `ratchet_accepted`: Boolean indicating if this round improved on prior-best.
- `extraction_method`: Method used for code extraction.
- `candidate_error`: Generation error (if any).
- `verify_stderr`: Captured stderr on failure.
- `latency_ms`: Generation time for this round.
- `raw_text`: Full raw model output.
- `extracted_code`: Extracted Python source.
- `critique`: The `Critique` that triggered this round (None for round 0).

### `CandidateResult`

Outcome for a single candidate slot (potentially multiple rounds).
- `run_index`: Index of the run.
- `candidate_index`: Position in the batch.
- `seed`: Seed of round 0.
- `rounds`: Tuple of `RoundResult`.
- `prior_best`: (property) The round kept under the ratchet (highest `tests_passed`).
- `passed`: (property) Whether the `prior_best` round passed.
- `round_zero_passed`: (property) Whether the original candidate passed.
- `total_latency_ms`: (property) Sum of latency across all rounds.

### `RunResult`

Outcome for a full run (batch of slots).
- `run_index`: Index of the run.
- `batch_id`: ID of the round-0 candidate batch.
- `candidates`: Tuple of `CandidateResult`.
- `run_duration_ms`: Total time for the run (including reflexion).

### `SessionSummary`

Aggregate results for a session.
- `task`: Task name.
- `model`: Coder model name.
- `critic_rounds`: Max reflexion rounds configured.
- `critic_model`: Critic model name.
- `total_candidates`: Total slots count.
- `passes`: Number of slots that passed on their prior-best round.
- `pass_rate`: Ratio of passes to total slots.
- `one_shot_passes`: Slots that passed on round 0.
- `one_shot_pass_rate`: Ratio of one-shot passes to total.
- `mean_latency_ms`: Average latency per slot (summed across rounds).
- `mean_tests_passed`: Mean number of passed tests across all prior-best rounds.
- `mean_tests_total`: Mean total number of tests across all prior-best rounds.

### Functions

- `run_bench(task: str, runs: int, n: int, model: str, *, timeout_seconds: float, critic_rounds: int, critic_model: str, verbose: bool = False, ...) -> tuple[Path, SessionSummary]`: Executes a benchmark session and returns the session directory and summary.

