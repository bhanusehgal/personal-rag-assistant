"""LLM-free retrieval-quality evaluation — DESIGN.md section 11.

Measures retrieval alone (precision@k, recall@k, MRR, nDCG@k) against
golden_qa.jsonl's expected_citation_chunks, without spending any LLM
inference time — the fast iteration loop for tuning retrieval independent
of the slow, memory-constrained generation pipeline.

Precision/recall/MRR/nDCG are computed ONLY over the golden set's
`unanswerable: false` rows — they're undefined against an empty expected-
citations set. For the `unanswerable: true` rows there's nothing to measure
recall against; instead (in hybrid mode) their retrieval-quality-gate score
is reported separately — the direct retrieval-side complement to
eval/faithfulness.py's abstention_ok check, and the fastest way to sanity-
check RETRIEVAL_GATE_THRESHOLD before spending any LLM call.

IMPORTANT CALIBRATION NOTE: the current sample corpus is 5 documents = 5
total chunks. Every retrieve() call at the default top_k=5 already returns
every chunk in the corpus, so precision@5/recall@5 will read ~1.0 for both
dense and hybrid modes here — there's nothing to filter out of 5 chunks
yet. The informative comparison at this corpus size is MRR / nDCG@1 (does
the right chunk land at rank 1?) and the unanswerable-questions' score
distribution — not precision/recall@5. This stops being a limitation
automatically once a larger corpus is ingested.

Usage:
    python -m eval.retrieval_eval --mode {dense,hybrid,both} \
        [--golden-path eval/golden_qa.jsonl] [--k-values 1,3,5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tools import Retriever  # noqa: E402
from ingest.embeddings import get_embedder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.hybrid_retriever import build_default_hybrid_retriever  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_qa.jsonl"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_K_VALUES = [1, 3, 5]


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dcg(relevances: list[int]) -> float:
    # rank = i+1 (1-indexed); standard formula: rel_i / log2(rank + 1)
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(ranked_keys: list[tuple[str, int]], expected: set[tuple[str, int]], k: int) -> float:
    top_k = ranked_keys[:k]
    relevances = [1 if key in expected else 0 for key in top_k]
    dcg = _dcg(relevances)
    ideal_relevances = [1] * min(len(expected), k) + [0] * max(0, k - len(expected))
    idcg = _dcg(ideal_relevances)
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalEvalReport:
    mode: str
    n_answerable: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    per_question: list[dict] = field(default_factory=list)
    unanswerable_scores: list[dict] = field(default_factory=list)  # only populated in hybrid mode


def evaluate_retrieval(
    golden: list[dict],
    retrieve_fn,
    k_values: list[int] = DEFAULT_K_VALUES,
    mode: str = "hybrid",
    quality_fn=None,
) -> RetrievalEvalReport:
    """retrieve_fn(question, top_k) -> ranked list of (source, chunk_index) keys.
    quality_fn(question), if given, is called right after retrieve_fn for the
    same question and should return the retriever's RetrievalQualityReport
    for that call (or None) — used only for the unanswerable-question score
    reporting below, not for precision/recall/MRR/nDCG."""
    max_k = max(k_values)
    answerable = [g for g in golden if not g.get("unanswerable", False)]
    unanswerable = [g for g in golden if g.get("unanswerable", False)]

    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    per_question: list[dict] = []

    for item in answerable:
        expected = {(e["source"], e["chunk_index"]) for e in item.get("expected_citation_chunks", [])}
        ranked_keys = retrieve_fn(item["question"], max_k)

        first_relevant_rank = next((i + 1 for i, key in enumerate(ranked_keys) if key in expected), None)
        mrr_sum += 1.0 / first_relevant_rank if first_relevant_rank else 0.0

        row: dict = {"question_id": item["id"], "first_relevant_rank": first_relevant_rank}
        for k in k_values:
            top_k = ranked_keys[:k]
            hit = len(set(top_k) & expected)
            precision = hit / k if k else 0.0
            recall = hit / len(expected) if expected else 0.0
            ndcg = _ndcg_at_k(ranked_keys, expected, k)
            precision_sums[k] += precision
            recall_sums[k] += recall
            ndcg_sums[k] += ndcg
            row[f"precision@{k}"] = round(precision, 4)
            row[f"recall@{k}"] = round(recall, 4)
            row[f"ndcg@{k}"] = round(ndcg, 4)
        per_question.append(row)

    n = len(answerable) or 1
    report = RetrievalEvalReport(
        mode=mode,
        n_answerable=len(answerable),
        precision_at_k={k: precision_sums[k] / n for k in k_values},
        recall_at_k={k: recall_sums[k] / n for k in k_values},
        mrr=mrr_sum / n,
        ndcg_at_k={k: ndcg_sums[k] / n for k in k_values},
        per_question=per_question,
    )

    if quality_fn is not None:
        for item in unanswerable:
            retrieve_fn(item["question"], max_k)  # populates the retriever's last_quality_report
            qr = quality_fn(item["question"])
            report.unanswerable_scores.append(
                {
                    "question_id": item["id"],
                    "top1_rerank_score": qr.top1_rerank_score if qr else None,
                    "passes_gate": qr.passes_gate() if qr else None,
                }
            )

    return report


def write_report_markdown(report: RetrievalEvalReport, out_path: Path) -> None:
    lines = [f"# Retrieval eval — mode={report.mode}\n", f"n_answerable = {report.n_answerable}\n"]
    lines.append("| k | precision@k | recall@k | nDCG@k |")
    lines.append("|---|---|---|---|")
    for k in sorted(report.precision_at_k):
        lines.append(
            f"| {k} | {report.precision_at_k[k]:.3f} | {report.recall_at_k[k]:.3f} | {report.ndcg_at_k[k]:.3f} |"
        )
    lines.append(f"\nMRR = {report.mrr:.3f}\n")

    if report.unanswerable_scores:
        lines.append("## Unanswerable-question scores (retrieval-gate sanity check)\n")
        lines.append("| question_id | top1_rerank_score | passes_gate |")
        lines.append("|---|---|---|")
        for row in report.unanswerable_scores:
            lines.append(f"| {row['question_id']} | {row['top1_rerank_score']} | {row['passes_gate']} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-free retrieval-quality eval.")
    parser.add_argument("--mode", default="both", choices=["dense", "hybrid", "both"])
    parser.add_argument("--golden-path", default=str(GOLDEN_PATH))
    parser.add_argument("--k-values", default="1,3,5")
    parser.add_argument("--embed-provider", default="ollama", choices=["openai", "ollama", "fake"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    golden = load_golden(Path(args.golden_path))
    modes = ["dense", "hybrid"] if args.mode == "both" else [args.mode]

    for mode in modes:
        if mode == "dense":
            embedder = get_embedder(args.embed_provider)
            dense_retriever = Retriever(embedder)

            def retrieve_fn(question: str, top_k: int, _r=dense_retriever) -> list[tuple[str, int]]:
                return [(c.source, c.chunk_index) for c in _r.retrieve(question, top_k=top_k)]

            quality_fn = None
        else:
            hybrid_retriever = build_default_hybrid_retriever(args.embed_provider)

            def retrieve_fn(question: str, top_k: int, _r=hybrid_retriever) -> list[tuple[str, int]]:
                return [(c.source, c.chunk_index) for c in _r.retrieve(question, top_k=top_k)]

            def quality_fn(question: str, _r=hybrid_retriever):
                return _r.last_quality_report

        report = evaluate_retrieval(golden, retrieve_fn, k_values=k_values, mode=mode, quality_fn=quality_fn)
        write_report_markdown(report, RUNS_DIR / f"retrieval-eval-{mode}" / "retrieval_eval.md")

        summary = f"[{mode}] MRR={report.mrr:.3f}  " + "  ".join(
            f"P@{k}={report.precision_at_k[k]:.3f} R@{k}={report.recall_at_k[k]:.3f} nDCG@{k}={report.ndcg_at_k[k]:.3f}"
            for k in k_values
        )
        print(summary)


if __name__ == "__main__":
    main()
