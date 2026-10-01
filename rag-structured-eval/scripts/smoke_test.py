"""Stage 0 smoke test: raw chat-completion reliability triage across all 7
candidate models, BEFORE investing in the full structured/agentic harness.

Plain chat, no tools, no structured output — isolates "does this model even
reliably respond on this hardware" from every other variable. See
DESIGN.md / PROGRESS.md Stage 0.

python -m scripts.smoke_test [--attempts 3]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from metrics import InferenceClient  # noqa: E402

# Ordered smallest-first, not largest-first: this machine has only ~7.7GB
# total RAM (discovered when a largest-first run let models stack in memory
# and got OOM-killed before logging a single result — see DESIGN.md section
# 3). Run the server with OLLAMA_MAX_LOADED_MODELS=1 / a short
# OLLAMA_KEEP_ALIVE, and this script explicitly `ollama stop`s each model
# before moving to the next as a second layer of protection.
MODELS = [
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q5_K_M",
    "phi4-mini:3.8b-q4_K_M",
    "phi4-mini:3.8b-q8_0",
    "mistral:7b-instruct-q4_K_M",
    "mistral:7b-instruct-q5_K_M",
    "qwen2.5:7b-instruct",
]


def _stop_model(model: str) -> None:
    """Explicitly unload a model from the Ollama server before switching to
    the next one, rather than relying on keep_alive to expire in time."""
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)

SMOKE_QUESTION = "In one sentence, what is model risk management?"

RUN_ID = "smoke"
METRICS_PATH = Path(__file__).resolve().parents[1] / "eval" / "runs" / RUN_ID / "metrics.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 raw reliability smoke test.")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of MODELS to run (e.g. to resume after a crash). Default: all 7.",
    )
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds to wait after stopping a model.")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else MODELS

    client = InferenceClient(run_id=RUN_ID, metrics_path=METRICS_PATH, timeout=args.timeout)

    for model in models:
        for attempt in range(args.attempts):
            print(f"[run] model={model} attempt={attempt}")
            result, record = client.chat(
                model=model,
                messages=[{"role": "user", "content": SMOKE_QUESTION}],
                temperature=0.0,
                prompt_version="smoke",
                question_id=None,
                round_index=0,
                attempt=attempt,
                call_kind="plain",
                quant=model,
            )
            status = "OK" if record.success else f"FAIL ({record.error})"
            print(f"    -> {status}, latency={record.total_latency_s:.1f}s, ttft={record.ttft_s}")
        print(f"[stop] unloading {model}")
        _stop_model(model)
        time.sleep(args.cooldown)  # give the server a moment to actually free memory before loading the next model

    df = pd.read_json(METRICS_PATH, lines=True)
    summary = df.groupby("model").agg(
        success_rate=("success", "mean"),
        mean_latency_s=("total_latency_s", "mean"),
        mean_ttft_s=("ttft_s", "mean"),
        n=("success", "count"),
    ).sort_values("success_rate", ascending=False)
    print("\n=== Stage 0 smoke test summary ===")
    print(summary.round(2).to_string())
    summary.round(3).to_markdown(METRICS_PATH.parent / "smoke_summary.md")
    print(f"\nWrote {METRICS_PATH.parent / 'smoke_summary.md'}")


if __name__ == "__main__":
    main()
