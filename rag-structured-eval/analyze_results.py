"""Phase 3 analysis — DESIGN.md section 9.

Reads a comparison run's results.jsonl + metrics.jsonl and writes:
  - comparison_table.md: one row per model, sorted by faithfulness desc.
  - quality_vs_speed.png: one scatter chart (latency vs faithfulness),
    points directly labeled by model name instead of color-coded, since a
    7-way categorical palette isn't needed when every point can just be
    labeled with text (see dataviz skill: identity via color is unnecessary
    when direct labels already carry it, and avoids an un-validated ramp).

python -m analyze_results --run-id <name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "eval" / "runs"

MODEL_SIZE_GB = {
    "qwen2.5:7b-instruct": 4.7,
    "llama3.2:3b-instruct-q4_K_M": 2.0,
    "llama3.2:3b-instruct-q5_K_M": 2.3,
    "phi4-mini:3.8b-q4_K_M": 2.5,
    "phi4-mini:3.8b-q8_0": 4.1,
    "mistral:7b-instruct-q4_K_M": 4.3,
    "mistral:7b-instruct-q5_K_M": 5.2,
}

# phi4-mini:3.8b-q8_0 drops from 128K to 4K context vs its q4 sibling — flagged
# explicitly wherever this model appears in output, per DESIGN.md section 4.
CONTEXT_NOTE = {
    "phi4-mini:3.8b-q8_0": "4K context (vs 128K on q4_K_M) — may truncate agentic transcripts",
}


def analyze(run_id: str) -> None:
    run_dir = RUNS_DIR / run_id
    results_path = run_dir / "results.jsonl"
    metrics_path = run_dir / "metrics.jsonl"

    results = pd.read_json(results_path, lines=True)
    metrics = pd.read_json(metrics_path, lines=True)

    # Total latency per question = sum across all rounds/attempts for that
    # question — the real end-to-end cost of the agentic flow, not just the
    # final call. See DESIGN.md section 6.
    per_question_latency = metrics.groupby(["model", "question_id"])["total_latency_s"].sum().reset_index()
    mean_latency_by_model = per_question_latency.groupby("model")["total_latency_s"].mean()

    per_model_metrics = metrics.groupby("model").agg(
        mean_ttft_s=("ttft_s", "mean"),
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        success_rate=("success", "mean"),
    )

    per_model_results = results.groupby("model").agg(
        faithfulness_rate=("faithfulness_pass", "mean"),
        validity_rate=("valid_structured_output", "mean"),
        n_questions=("question_id", "count"),
    )

    table = per_model_results.join(per_model_metrics, how="left")
    table["mean_total_latency_s"] = mean_latency_by_model
    table["model_size_gb"] = table.index.map(MODEL_SIZE_GB)
    table["notes"] = table.index.map(lambda m: CONTEXT_NOTE.get(m, ""))
    table = table.sort_values("faithfulness_rate", ascending=False)

    table_path = run_dir / "comparison_table.md"
    table_path.write_text(
        f"# Model comparison — run {run_id}\n\n" + table.round(3).to_markdown(),
        encoding="utf-8",
    )
    print(f"Wrote {table_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    x = table["mean_total_latency_s"]
    y = table["faithfulness_rate"]
    ax.scatter(x, y, s=60, color="#3B6FA0", zorder=3)
    for model, row in table.iterrows():
        ax.annotate(
            model,
            (row["mean_total_latency_s"], row["faithfulness_rate"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Mean total latency per question (s)")
    ax.set_ylabel("Faithfulness rate")
    ax.set_title(f"Quality vs. speed — run {run_id}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    chart_path = run_dir / "quality_vs_speed.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Wrote {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 analysis: comparison table + quality-vs-speed chart.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    analyze(args.run_id)


if __name__ == "__main__":
    main()
