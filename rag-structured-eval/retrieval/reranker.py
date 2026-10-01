"""Cross-encoder reranking — DESIGN.md section 11.

A cross-encoder scores (query, passage) jointly through one transformer
forward pass, unlike dense embeddings which score query and passage
independently and lose interaction information. That makes it a much more
decision-useful relevance signal than cosine similarity alone — this is the
real, learned confidence signal the retrieval-quality gate (quality.py)
is built on.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, CPU-friendly, a
standard MS MARCO passage-reranking cross-encoder). If this proves too
heavy on this machine's documented tight RAM budget (DESIGN.md section 3),
cross-encoder/ms-marco-TinyBERT-L-2-v2 (~4M params) is a same-interface
fallback — swap FALLBACK_RERANK_MODEL in as model_name, nothing else changes.

Score normalization: this model was trained with a regression objective and
its raw CrossEncoder.predict() output is an UNBOUNDED logit, not a
probability — verified empirically at setup time against the installed
sentence-transformers version (see PROGRESS.md "Stage 2.5" verification
notes), not assumed. A sigmoid is applied explicitly here to normalize into
[0, 1] so the quality gate's threshold has a stable, bounded scale to
compare against.
"""
from __future__ import annotations

import numpy as np

from .types import RankedChunk

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FALLBACK_RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[RankedChunk]) -> list[RankedChunk]:
        """Scores each (query, candidate.text) pair, sets rerank_score on each
        candidate in place, and returns the same list re-sorted descending.
        Mutates and returns for convenience; callers should not rely on the
        input list's original order surviving."""
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        raw_scores = self._model.predict(pairs)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype="float64")))

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates
