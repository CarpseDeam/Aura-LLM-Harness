# Changelog

All notable changes to this project.

- 2026-04-30: refactor: restructure Benchmark Runner to use phase-based orchestration, reducing model swap overhead and enabling parallel critic calls.
- 2026-04-30: feat: thread failure context (gradient, failed tests, stderr) into CriticClient for focused reflexion reviews.
- 2026-04-30: feat: add stdout progress logging and real-time `progress.json` updates to the Benchmark Runner.
- 2026-04-30: feat: implement gradient evaluation and one-way ratchet logic in Benchmark Runner to ensure monotonic improvement during reflexion.
- 2026-04-30: feat: implement explicit model unloading in OllamaClient and Bench Runner to optimize VRAM usage.
- 2026-04-28: feat: implement critic + reflexion loop for the bench to improve candidate success rates.
- 2026-04-28: docs: sharpen duplicate_finder task spec with explicit contract requirements.
- 2026-04-28: feat: implement benchmark harness with automated verifier and session logging.
- 2026-04-28: feat: implement chat-style planner layer for conversational spec shaping.
- 2026-04-28: feat: implement project scaffolding with templates (Blank, Python CLI).
- 2026-04-28: feat: implement workspace management, file context injection, and candidate application in GUI.
- 2026-04-28: feat: implement scoring foundations (Extractor, Validator, Scorer) and CLI.
- 2026-04-28: refactor: expose base_url, timeout_seconds, and log_path as public properties on OllamaClient.
- 2026-04-28: feat: wire GUI to candidate generator with asynchronous BatchWorker.
- 2026-04-28: refactor: expose OllamaClient.default_model as a public property and update CandidateGenerator to use it.
- 2026-04-28: feat: add --test-file argument to scoring CLI and make it mutually exclusive with --test.
