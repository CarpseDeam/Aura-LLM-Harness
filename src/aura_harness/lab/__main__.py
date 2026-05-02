"""CLI smoke test: ``python -m aura_harness.lab "prompt" --n 5``."""
from __future__ import annotations

import argparse
import sys

from aura_harness.backend import LocalOllamaBackend
from aura_harness.lab.generator import CandidateGenerator
from aura_harness.lab.models import CandidateBatch
from aura_harness.llm import OllamaClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aura_harness.lab",
        description="Generate N candidates in parallel for a single prompt.",
    )
    parser.add_argument("prompt", help="Prompt to send to the model.")
    parser.add_argument("--n", type=int, default=3, help="Number of candidates (default: 3).")
    parser.add_argument("--model", default="qwen3.5:latest", help="Model name.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; candidate i uses base_seed + i. Default: random.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Optional cap on tokens generated per candidate.",
    )
    return parser


def _print_batch(batch: CandidateBatch) -> None:
    print(
        f"Batch {batch.batch_id}: {batch.n_requested} candidates, "
        f"{batch.success_count} succeeded in {batch.total_wall_duration_ms:.0f}ms"
    )
    for candidate in batch.candidates:
        marker = "OK" if candidate.is_success else "FAIL"
        print(f"[{candidate.index}] {marker} {candidate.wall_duration_ms:.0f}ms")
        print(candidate.text if candidate.is_success else candidate.error)
        print("---")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    client = OllamaClient(default_model=args.model)
    backend = LocalOllamaBackend(client)
    generator = CandidateGenerator(backend)

    batch = generator.generate(
        args.prompt,
        n=args.n,
        model=args.model,
        system=args.system,
        temperature=args.temperature,
        base_seed=args.seed,
        num_predict=args.num_predict,
    )
    _print_batch(batch)
    return 0 if batch.success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
