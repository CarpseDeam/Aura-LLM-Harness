"""Parallel candidate generation against a chat-capable :class:`Backend`.

The :class:`CandidateGenerator` fans out N independent chat completions
through a thread pool, each with its own seed, and returns them as a
:class:`CandidateBatch`. Failures on individual candidates are captured
rather than raised so the rest of the batch can complete.

Concurrency note: when the backend is a local Ollama server, the server
caps how many requests it processes concurrently per model via the
``OLLAMA_NUM_PARALLEL`` environment variable (default 1 on older builds,
4 on newer ones). When that cap is below ``max_workers`` the server
queues the extras; results are still correct, but the speedup flattens.
Bump ``OLLAMA_NUM_PARALLEL`` on the server side if you want full
throughput. Cloud backends have their own concurrency limits.
"""
from __future__ import annotations

import json
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from aura_harness.backend import Backend, CompletionResult
from aura_harness.lab.models import Candidate, CandidateBatch

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS: Final[int] = 5
_DEFAULT_TEMPERATURE: Final[float] = 0.5
_PROMPT_PREVIEW_CHARS: Final[int] = 200
_SEED_MAX: Final[int] = 2**31 - 1


class CandidateGenerator:
    """Produce N parallel candidate completions from a single prompt."""

    def __init__(
        self,
        backend: Backend,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        log_path: Path | None = None,
    ) -> None:
        """Construct a generator.

        Args:
            backend: The :class:`Backend` used to issue chat calls.
            max_workers: Upper bound on concurrent in-flight requests.
            log_path: JSONL file to append batch summaries to. Defaults to
                ``./.aura/batches.jsonl`` under the current working directory.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._backend = backend
        self._max_workers = max_workers
        self._log_path = (
            log_path if log_path is not None else Path.cwd() / ".aura" / "batches.jsonl"
        )

    def generate(
        self,
        prompt: str,
        n: int,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        base_seed: int | None = None,
        num_predict: int | None = None,
    ) -> CandidateBatch:
        """Generate ``n`` candidate completions in parallel.

        ``prompt`` becomes the user message. When ``system`` is provided
        it is prepended as a system message; otherwise the chat is
        single-turn user-only.

        Args:
            prompt: User prompt; identical for every candidate.
            n: Number of candidates to generate. Must be ``>= 1``.
            model: Override the backend's default model.
            system: Optional system prompt; identical for every candidate.
            temperature: Sampling temperature. Defaults higher than the
                backend's chat default (0.2) to encourage diverse candidates.
            base_seed: Seed for candidate ``0``; candidate ``i`` uses
                ``base_seed + i``. If ``None``, a random base is picked.
            num_predict: Optional cap on tokens generated per candidate.
                Maps to the backend's ``max_tokens`` parameter.

        Returns:
            A :class:`CandidateBatch` with exactly ``n`` candidates ordered
            by index. Failures are captured on individual candidates; this
            method does not raise on per-candidate errors.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        chosen_seed_base = base_seed if base_seed is not None else random.randint(0, _SEED_MAX)
        seeds = [chosen_seed_base + i for i in range(n)]
        chosen_model = model or self._backend.default_model
        messages = _build_messages(prompt=prompt, system=system)
        batch_id = uuid.uuid4().hex

        logger.debug(
            "starting batch %s: n=%d model=%s base_seed=%d",
            batch_id,
            n,
            chosen_model,
            chosen_seed_base,
        )

        batch_started = time.perf_counter()
        completed: list[Candidate] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                pool.submit(
                    self._run_one,
                    index=i,
                    seed=seeds[i],
                    messages=messages,
                    model=chosen_model,
                    temperature=temperature,
                    num_predict=num_predict,
                )
                for i in range(n)
            ]
            for future in as_completed(futures):
                completed.append(future.result())
        total_wall_ms = (time.perf_counter() - batch_started) * 1000.0
        candidates = sorted(completed, key=lambda c: c.index)

        batch = CandidateBatch(
            batch_id=batch_id,
            prompt=prompt,
            model=chosen_model,
            n_requested=n,
            candidates=candidates,
            total_wall_duration_ms=total_wall_ms,
        )
        self._log_batch(batch, seeds=seeds)
        return batch

    def _run_one(
        self,
        *,
        index: int,
        seed: int,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        num_predict: int | None,
    ) -> Candidate:
        """Run a single ``backend.chat`` call and wrap the outcome."""
        started = time.perf_counter()
        result: CompletionResult | None = None
        error: str | None = None
        try:
            result = self._backend.chat(
                messages,
                model=model,
                temperature=temperature,
                seed=seed,
                max_tokens=num_predict,
            )
        except Exception as exc:  # noqa: BLE001 — capture all per-candidate failures
            error = str(exc)
            logger.debug("candidate %d failed: %s", index, exc)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return Candidate(
            index=index,
            seed=seed,
            result=result,
            error=error,
            wall_duration_ms=wall_ms,
        )

    def _log_batch(self, batch: CandidateBatch, *, seeds: list[int]) -> None:
        """Append a one-line JSON summary of ``batch`` to the batch log."""
        successful = batch.successful
        mean_wall = (
            sum(c.wall_duration_ms for c in successful) / len(successful)
            if successful
            else 0.0
        )
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch.batch_id,
            "model": batch.model,
            "prompt_preview": batch.prompt[:_PROMPT_PREVIEW_CHARS],
            "n_requested": batch.n_requested,
            "success_count": batch.success_count,
            "total_wall_duration_ms": batch.total_wall_duration_ms,
            "mean_candidate_wall_ms": mean_wall,
            "seeds": seeds,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("failed to write batch log to %s: %s", self._log_path, exc)


def _build_messages(*, prompt: str, system: str | None) -> list[dict[str, str]]:
    """Compose the chat messages list from a raw prompt and optional system."""
    if system is not None:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]
