"""Retrieval-quality signal + gate — DESIGN.md section 11.

Deliberately mirrors eval/faithfulness.py's pattern: a dataclass that always
computes and exposes every underlying signal, plus a single pass/fail
property/method the caller actually gates on. Computed on every retrieve()
call regardless of outcome, per the explicit "always log, sometimes block"
design decision — so the threshold below can be tuned later from real
observed score distributions instead of guessed at twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.tools import RetrievedChunk

    from .types import LexicalHit, RankedChunk

RETRIEVAL_GATE_THRESHOLD = 0.5
"""Validated against real score distributions (see DESIGN.md section 11 /
PROGRESS.md Stage 2.5) — not a guess. On the 5-sample-doc corpus's golden
set: all 10 answerable questions scored top1_rerank_score in [0.969, 0.9994]
(tightly clustered near 1.0); 3 of the 4 deliberately-unanswerable questions
scored below 0.03. 0.5 sits comfortably in that gap.

The 4th unanswerable question (q013, an "adjacent-topic trap": a real
document exists on a related subject but never states the specific fact
asked about) scored 0.755 — ABOVE this threshold, so the gate will not
catch it, at any threshold that doesn't also reject genuinely answerable
questions (the answerable floor is 0.969). This is an honest, expected
limitation, not a tuning problem: a cross-encoder measures topical
relevance, not whether a passage actually contains the specific fact
asked about — it cannot distinguish "this passage is about the same
subject" from "this passage answers the question." Re-run
`python -m eval.retrieval_eval --mode both` and re-derive this value if the
corpus changes meaningfully."""


@dataclass
class RetrievalQualityReport:
    question: str
    top1_rerank_score: float | None
    score_gap: float | None  # top1 - top2 rerank_score; None if <2 reranked candidates
    dense_lexical_agreement: bool  # do dense-rank-1 and lexical-rank-1 point at the same chunk?
    top1_source_signal: Literal["dense_only", "lexical_only", "both", "none"]
    num_dense_hits: int
    num_lexical_hits: int
    num_fused_candidates: int
    num_reranked: int

    def passes_gate(self, threshold: float = RETRIEVAL_GATE_THRESHOLD) -> bool:
        """v1 rule, deliberately simple: pass iff a top1 rerank score exists
        and clears the threshold. score_gap and dense_lexical_agreement are
        always computed and logged but not yet part of the pass/fail rule —
        left available for a smarter multi-signal rule once there's real
        data to tune one from."""
        if self.top1_rerank_score is None:
            return False
        return self.top1_rerank_score >= threshold


def assess_quality(
    query: str,
    dense_hits: list["RetrievedChunk"],
    lexical_hits: list["LexicalHit"],
    fused: list["RankedChunk"],
    reranked: list["RankedChunk"],
) -> RetrievalQualityReport:
    top1_rerank_score = reranked[0].rerank_score if reranked else None

    score_gap = None
    if len(reranked) >= 2 and reranked[0].rerank_score is not None and reranked[1].rerank_score is not None:
        score_gap = reranked[0].rerank_score - reranked[1].rerank_score

    dense_top1_key = (dense_hits[0].source, dense_hits[0].chunk_index) if dense_hits else None
    lexical_top1_key = (lexical_hits[0].source, lexical_hits[0].chunk_index) if lexical_hits else None
    dense_lexical_agreement = dense_top1_key is not None and dense_top1_key == lexical_top1_key

    top1_source_signal: Literal["dense_only", "lexical_only", "both", "none"] = "none"
    if reranked:
        top1 = reranked[0]
        has_dense = top1.dense_rank is not None
        has_lexical = top1.lexical_rank is not None
        if has_dense and has_lexical:
            top1_source_signal = "both"
        elif has_dense:
            top1_source_signal = "dense_only"
        elif has_lexical:
            top1_source_signal = "lexical_only"

    return RetrievalQualityReport(
        question=query,
        top1_rerank_score=top1_rerank_score,
        score_gap=score_gap,
        dense_lexical_agreement=dense_lexical_agreement,
        top1_source_signal=top1_source_signal,
        num_dense_hits=len(dense_hits),
        num_lexical_hits=len(lexical_hits),
        num_fused_candidates=len(fused),
        num_reranked=len(reranked),
    )
