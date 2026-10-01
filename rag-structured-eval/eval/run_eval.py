"""Phase 2 temperature-sweep harness — DESIGN.md sections 8 and 10.

python -m eval.run_eval --model <model> --temperatures 0.0,0.3,0.7,1.0 --run-id <name>

--mode retrieve_first (default) retrieves in code; --mode agentic lets the
model drive retrieval through the retrieve tool (generate.py). Each run also
writes trace.jsonl: every retrieval / tool call, the exact final prompt, and
the raw final response.

Resumable by construction: before each (model, question, temperature) combo,
skips it if a completed row for that exact key already exists in
results.jsonl. Required because full sweeps can take hours on this hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from ingest.embeddings import get_embedder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.faithfulness import score_answer  # noqa: E402
from generate import (  # noqa: E402
    DEFAULT_NUM_CTX,
    GenerationFailure,
    run_agentic_structured,
    run_retrieve_then_read,
)
from metrics import InferenceClient  # noqa: E402
from prompt_store import load_prompt  # noqa: E402
from retrieval.hybrid_retriever import build_default_hybrid_retriever  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runs"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_qa.jsonl"


def load_golden(golden_path: Path = GOLDEN_PATH) -> list[dict]:
    with golden_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_completed_keys(results_path: Path) -> set[tuple[str, str, float]]:
    if not results_path.exists():
        return set()
    keys = set()
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["question_id"], row["model"], row["temperature"]))
    return keys


def run_sweep(
    model: str,
    prompt_id: str,
    temperatures: list[float],
    run_id: str,
    embed_provider: str = "ollama",
    golden_path: Path = GOLDEN_PATH,
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
        "model": model,
        "prompt_id": prompt_id,
        "temperatures": temperatures,
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
    client = InferenceClient(run_id=run_id, metrics_path=metrics_path)
    golden = load_golden(golden_path)
    completed = load_completed_keys(results_path)

    for temperature in temperatures:
        for item in golden:
            key = (item["id"], model, temperature)
            if key in completed:
                print(f"[skip] {key} already completed")
                continue

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
            except Exception as e:  # noqa: BLE001 - one question's crash must not abort the whole sweep
                print(f"[error] question={item['id']} failed: {e}")
                import traceback

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

    write_summary(run_dir, results_path, metrics_path)


def write_summary(run_dir: Path, results_path: Path, metrics_path: Path) -> None:
    results = pd.read_json(results_path, lines=True) if results_path.exists() else pd.DataFrame()
    metrics = pd.read_json(metrics_path, lines=True) if metrics_path.exists() else pd.DataFrame()

    lines = ["# Temperature sweep summary\n"]
    if not results.empty:
        grouped = results.groupby("temperature").agg(
            validity_rate=("valid_structured_output", "mean"),
            faithfulness_rate=("faithfulness_pass", "mean"),
            n_questions=("question_id", "count"),
        )
        if not metrics.empty:
            lat = metrics.groupby("temperature").agg(
                mean_ttft_s=("ttft_s", "mean"),
                mean_tokens_per_sec=("tokens_per_sec", "mean"),
                mean_total_latency_s=("total_latency_s", "mean"),
                success_rate=("success", "mean"),
            )
            grouped = grouped.join(lat, how="left")
        lines.append(grouped.to_markdown())
    else:
        lines.append("No results yet.")

    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {run_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 temperature-sweep harness.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="retrieve_first", choices=["retrieve_first", "agentic"])
    parser.add_argument("--prompt-id", default="system_v3", help="Reader prompt for the structured-final call.")
    parser.add_argument("--tool-prompt-id", default="system_v1", help="System prompt for agentic tool rounds.")
    parser.add_argument("--temperatures", default="0.0,0.3,0.7,1.0")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--embed-provider", default="ollama", choices=["openai", "ollama", "fake"])
    parser.add_argument("--golden-path", default=str(GOLDEN_PATH), help="Path to a golden_qa.jsonl-format file; defaults to the full golden set.")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Must match the server's OLLAMA_CONTEXT_LENGTH.")
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]
    run_sweep(
        args.model,
        args.prompt_id,
        temperatures,
        args.run_id,
        args.embed_provider,
        Path(args.golden_path),
        mode=args.mode,
        tool_prompt_id=args.tool_prompt_id,
        num_ctx=args.num_ctx,
    )


if __name__ == "__main__":
    main()
