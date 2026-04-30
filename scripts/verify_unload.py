"""Sanity-check OllamaClient.unload_model against a live Ollama server.

Loads a model with a tiny ``generate`` call, queries ``/api/ps`` to confirm
it is resident, calls ``unload_model``, then queries ``/api/ps`` again to
confirm the model is no longer listed. Prints PASS or FAIL.

Run from the project root::

    python scripts/verify_unload.py

The script is hand-run; it intentionally avoids CLI args and hard-codes the
model so the wiring path matches what the bench runner uses in practice.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from aura_harness.llm import OllamaClient  # noqa: E402

MODEL_NAME = "qwen2.5-coder:7b"
SETTLE_SECONDS = 2.0
PS_TIMEOUT_S = 5.0


def fetch_ps(base_url: str) -> dict[str, Any]:
    """Return the parsed JSON body of ``GET /api/ps``."""
    with httpx.Client(base_url=base_url, timeout=PS_TIMEOUT_S) as client:
        response = client.get("/api/ps")
    response.raise_for_status()
    return response.json()


def model_is_loaded(ps_payload: dict[str, Any], name: str) -> bool:
    models = ps_payload.get("models") or []
    return any(entry.get("name") == name for entry in models)


def main() -> int:
    client = OllamaClient(default_model=MODEL_NAME)
    base_url = client.base_url

    print(f"Step 1: warming model {MODEL_NAME!r} via /api/generate")
    client.generate("hello", model=MODEL_NAME, num_predict=1)

    print("Step 2: GET /api/ps after load")
    loaded_payload = fetch_ps(base_url)
    print(loaded_payload)
    if not model_is_loaded(loaded_payload, MODEL_NAME):
        print(f"FAIL: model {MODEL_NAME!r} not present in /api/ps after load")
        return 1

    print(f"Step 3: client.unload_model({MODEL_NAME!r})")
    client.unload_model(MODEL_NAME)

    print(f"Step 4: sleeping {SETTLE_SECONDS}s for eviction to settle")
    time.sleep(SETTLE_SECONDS)

    print("Step 5: GET /api/ps after unload")
    after_payload = fetch_ps(base_url)
    print(after_payload)

    if model_is_loaded(after_payload, MODEL_NAME):
        print(f"FAIL: model {MODEL_NAME!r} still present in /api/ps after unload")
        return 1

    print(f"PASS: model {MODEL_NAME!r} evicted successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
