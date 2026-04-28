"""CLI smoke test: ``python -m aura_harness.llm``."""
from __future__ import annotations

import argparse
import sys

from aura_harness.llm.exceptions import OllamaError
from aura_harness.llm.ollama_client import OllamaClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aura_harness.llm",
        description="Smoke-test the local Ollama client.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send to the model.")
    parser.add_argument("--model", default="qwen3.5:latest", help="Model name.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--list", action="store_true", help="List installed models and exit.")
    parser.add_argument("--health", action="store_true", help="Check server reachability and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    client = OllamaClient(default_model=args.model)

    if args.health:
        if client.health_check():
            print("ok")
            return 0
        print("unreachable")
        return 1

    if args.list:
        try:
            for name in client.list_models():
                print(name)
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if not args.prompt:
        print("error: prompt is required (or pass --list / --health)", file=sys.stderr)
        return 2

    try:
        result = client.generate(args.prompt, system=args.system)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
