"""Phase 3 model comparison study — DESIGN.md section 9.

Runs the same golden set, same prompt version, at one fixed temperature
(the best one found in Stage 2) across all 7 models. Per-model try/except
isolation: one model's total failure doesn't abort the sweep (expected,
given qwen2.5:7b-instruct's documented ~1-in-9 historical success rate).

python -m compare_models --prompt-id system_v1 --temperature 0.0 --run-id <name>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.embeddings import get_embedder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.faithfulness import score_answer  # noqa: E402
from eval.run_eval import load_golden, load_completed_keys, write_summary  # noqa: E402
from generate import (  # noqa: E402
    DEFAULT_NUM_CTX,
    GenerationFailure,
    run_agentic_structured,
    run_retrieve_then_read,
)
from metrics import InferenceClient  # noqa: E402
from prompt_store import load_prompt  # noqa: E402
from retrieval.hybrid_retriever import build_default_hybrid_retriever  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "eval" / "runs"

# Per-model call timeout, sized to model weight so a hung small model doesn't
# eat the same ceiling as the 7B baseline. See DESIGN.md section 9.
#
# mistral:7b-instruct-q4_K_M / q5_K_M were dropped from this comparison after
# Stage 0: this machine's ~7.7GB total RAM couldn't reliably load either
# variant even after the context-length fix (repeated OOM kills — see
# PROGRESS.md). Both are still pulled locally if this ever gets revisited on
# more capable hardware; qwen2.5:7b-instruct remains as the sole 7B-class
# comparison point since it was already confirmed loadable pre-Stage-0.
MODEL_TIMEOUTS = {
    "llama3.2:3b-instruct-q4_K_M": 120.0,
    "llama3.2:3b-instruct-q5_K_M": 150.0,
    "phi4-mini:3.8b-q4_K_M": 150.0,
    "phi4-mini:3.8b-q8_0": 180.0,
    "qwen2.5:7b-instruct": 900.0,
}

MODELS = list(MODEL_TIMEOUTS.keys())


def _stop_model(model: str) -> None:
    """Explicitly unload a model before switching to the next one. This
    machine has only ~7.7GB total RAM (see DESIGN.md section 3) — letting
    models stack in memory across the sweep risks an OOM kill mid-run."""
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)


def run_comparison(
    prompt_id: str,
    temperature: float,
    run_id: str,
    embed_provider: str = "ollama",
    mode: str = "retrieve_first",
    tool_prompt_id: str = "system_v1",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    trace_path = run_dir / "trace.jsonl"
    manifest_path = run_dir / "manifest.json"

    manifest = {
        "models": MODELS,
        "prompt_id": prompt_id,
        "temperature": temperature,
        "run_id": run_id,
        "mode": mode,
        "num_ctx": num_ctx,
    }
    if mode == "agentic":
        manifest["tool_prompt_id"] = tool_prompt_id
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    embedder = get_embedder(embed_provider)
    retriever = build_default_hybrid_retriever(embed_provider)
    prompt = load_prompt(prompt_id)
    tool_prompt = load_prompt(tool_prompt_id) if mode == "agentic" else None
    golden = load_golden()
    completed = load_completed_keys(results_path)

    for model in MODELS:
        timeout = MODEL_TIMEOUTS.get(model, 900.0)
        client = InferenceClient(run_id=run_id, metrics_path=metrics_path, timeout=timeout)
        model_success_count = 0
        model_total_count = 0

        for item in golden:
            key = (item["id"], model, temperature)
            if key in completed:
                print(f"[skip] {key} already completed")
                continue

            model_total_count += 1
            print(f"[run] question={item['id']} model={model} temperature={temperature}")
            try:
                common = dict(
                    question=item["question"],
                    retriever=retriever,
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    run_id=run_id,
                    question_id=item["id"],
                    quant=model,
                    num_ctx=num_ctx,
                    trace_path=trace_path,
                )
                if mode == "agentic":
                    answer, records, error_kind = run_agentic_structured(tool_prompt=tool_prompt, **common)
                else:
                    answer, records, error_kind = run_retrieve_then_read(**common)
                quality_report = retriever.last_quality_report  # capture before the post-hoc re-query below overwrites it
                retrieved_chunks = [c.to_retrieved_chunk() for c in retriever.retrieve(item["question"], top_k=5)]
                fr = score_answer(answer, retrieved_chunks, item, embedder, item["id"])
                if answer is not None:
                    model_success_count += 1

                row = {
                    "question_id": item["id"],
                    "model": model,
                    "temperature": temperature,
                    "prompt_version": prompt.id,
                    "valid_structured_output": answer is not None,
                    "error_kind": error_kind,
                    "attempts_used": len(records),
                    "faithfulness_pass": fr.passed,
                    "citation_existence_ok": fr.citation_existence_ok,
                    "grounding_ok": fr.grounding_ok,
                    "abstention_ok": fr.abstention_ok,
                    "hallucinated_citations": fr.hallucinated_citations,
                    "expected_citation_recall": fr.expected_citation_recall,
                    "retrieval_gate_blocked": error_kind == GenerationFailure.RETRIEVAL_QUALITY_GATE_BLOCKED,
                    "retrieval_top1_rerank_score": quality_report.top1_rerank_score if quality_report else None,
                    "retrieval_score_gap": quality_report.score_gap if quality_report else None,
                    "retrieval_dense_lexical_agreement": quality_report.dense_lexical_agreement if quality_report else None,
                }
            except Exception as e:  # noqa: BLE001 - one model's crash must not abort the whole sweep
                print(f"[error] model={model} question={item['id']} failed: {e}")
                traceback.print_exc()
                row = {
                    "question_id": item["id"],
                    "model": model,
                    "temperature": temperature,
                    "prompt_version": prompt.id,
                    "valid_structured_output": False,
                    "error_kind": f"exception: {e}",
                    "attempts_used": 0,
                    "faithfulness_pass": False,
                    "citation_existence_ok": False,
                    "grounding_ok": False,
                    "abstention_ok": False,
                    "hallucinated_citations": [],
                    "expected_citation_recall": None,
                    "retrieval_gate_blocked": False,
                    "retrieval_top1_rerank_score": None,
                    "retrieval_score_gap": None,
                    "retrieval_dense_lexical_agreement": None,
                }

            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        if model_total_count:
            print(f"[model done] {model}: {model_success_count}/{model_total_count} succeeded this run")
        print(f"[stop] unloading {model}")
        _stop_model(model)
        time.sleep(3)  # give the server a moment to actually free memory before loading the next model

    write_summary(run_dir, results_path, metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 full model comparison.")
    parser.add_argument("--mode", default="retrieve_first", choices=["retrieve_first", "agentic"])
    parser.add_argument("--prompt-id", default="system_v3", help="Reader prompt for the structured-final call.")
    parser.add_argument("--tool-prompt-id", default="system_v1", help="System prompt for agentic tool rounds.")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--embed-provider", default="ollama", choices=["openai", "ollama", "fake"])
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Must match the server's OLLAMA_CONTEXT_LENGTH.")
    args = parser.parse_args()

    run_comparison(
        args.prompt_id,
        args.temperature,
        args.run_id,
        args.embed_provider,
        mode=args.mode,
        tool_prompt_id=args.tool_prompt_id,
        num_ctx=args.num_ctx,
    )


if __name__ == "__main__":
    main()
