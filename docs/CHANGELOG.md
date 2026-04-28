# Changelog

All notable changes to this project.

- 2026-04-28: feat: implement scoring foundations (Extractor, Validator, Scorer) and CLI.
- 2026-04-28: refactor: expose base_url, timeout_seconds, and log_path as public properties on OllamaClient.
- 2026-04-28: feat: wire GUI to candidate generator with asynchronous BatchWorker.
- 2026-04-28: refactor: expose OllamaClient.default_model as a public property and update CandidateGenerator to use it.
