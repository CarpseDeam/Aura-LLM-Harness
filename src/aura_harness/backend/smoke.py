"""End-to-end smoke CLI for the :class:`Backend` layer.

Used by hand to verify a backend is wired up correctly end-to-end before
any station code touches it. Hits a real provider — no mocking. Prints
the response text (streamed chunk-by-chunk if ``--stream``) followed by
token counts and reported duration.

Examples (run from a shell with ``DEEPSEEK_API_KEY`` exported):

    python -m aura_harness.backend.smoke --backend local --model qwen3.5:latest \\
        --prompt "Say hi in one word." --stream

    python -m aura_harness.backend.smoke --backend cloud --model deepseek-v4-flash \\
        --prompt "Say hi in one word." --stream

    python -m aura_harness.backend.smoke --backend cloud --model deepseek-v4-pro \\
        --prompt "Reason through 17 * 23." --reasoning
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Final

from aura_harness.backend.cloud_constants import DEEPSEEK_BASE_URL
from aura_harness.backend.cloud_http import CloudHTTPBackend
from aura_harness.backend.local_ollama import LocalOllamaBackend
from aura_harness.backend.protocol import Backend, BackendError, CompletionResult
from aura_harness.llm import OllamaClient

_DEEPSEEK_API_KEY_ENV: Final[str] = "DEEPSEEK_API_KEY"


def _build_backend(kind: str) -> Backend:
    if kind == "local":
        return LocalOllamaBackend(OllamaClient())
    if kind == "cloud":
        api_key = os.environ.get(_DEEPSEEK_API_KEY_ENV)
        if not api_key:
            raise SystemExit(
                f"missing {_DEEPSEEK_API_KEY_ENV} environment variable"
            )
        return CloudHTTPBackend(
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key,
            default_model="deepseek-v4-flash",
            name="deepseek",
        )
    raise SystemExit(f"unknown backend: {kind}")


def _print_summary(result: CompletionResult) -> None:
    print()
    print("---")
    print(f"model:             {result.model}")
    print(f"prompt_tokens:     {result.prompt_tokens}")
    print(f"completion_tokens: {result.completion_tokens}")
    print(f"total_duration_ms: {result.total_duration_ms:.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backend smoke CLI.")
    parser.add_argument("--backend", choices=("local", "cloud"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    args = parser.parse_args(argv)

    backend = _build_backend(args.backend)
    messages = [{"role": "user", "content": args.prompt}]

    try:
        if args.stream:
            final: CompletionResult | None = None
            for chunk in backend.chat_stream(
                messages,
                model=args.model,
                reasoning=args.reasoning,
            ):
                if chunk.delta:
                    sys.stdout.write(chunk.delta)
                    sys.stdout.flush()
                if chunk.final is not None:
                    final = chunk.final
            if final is None:
                print("\n[no terminal chunk received]", file=sys.stderr)
                return 1
            _print_summary(final)
        else:
            result = backend.chat(
                messages,
                model=args.model,
                reasoning=args.reasoning,
            )
            sys.stdout.write(result.text)
            sys.stdout.flush()
            _print_summary(result)
    except BackendError as exc:
        print(f"\nbackend error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
