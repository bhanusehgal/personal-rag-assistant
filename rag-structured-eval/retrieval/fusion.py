"""Reciprocal Rank Fusion (RRF) — combines dense (FAISS cosine) and lexical
(SQLite FTS5 bm25) rankings — DESIGN.md section 11.

RRF only uses rank *position*, never raw score magnitude, which is exactly
why it's the right fusion method here: dense cosine similarity and BM25
live on completely incomparable scales, so averaging or weighting their raw
scores directly would be meaningless without careful calibration. A
candidate that ranks #2 or #3 in *both* lists can legitimately outscore
something that's #1 in only one — that's the intended behavior, not a bug.

k=60 is the standard default from the original RRF paper (Cormack, Clarke &
Buettcher, 2009) and the default used by OpenSearch/Elasticsearch/Weaviate
hybrid search — not tuned for this project specifically, but a reasonable,
well-precedented starting point.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import RankedChunk

if TYPE_CHECKING:
    from agent.tools import RetrievedChunk

    from .types import LexicalHit

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    dense_hits: list["RetrievedChunk"],
    lexical_hits: list["LexicalHit"],
    k: int = DEFAULT_RRF_K,
) -> dict[tuple[str, int], float]:
    """score(chunk) = sum over each input list containing it of 1/(k + rank),
    rank 1-indexed. dense_hits is assumed already ranked in list order
    (agent.tools.Retriever.retrieve()'s return order); lexical_hits carries
    its own explicit .rank field."""
    scores: dict[tuple[str, int], float] = {}

    for rank, chunk in enumerate(dense_hits, start=1):
        key = (chunk.source, chunk.chunk_index)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    for hit in lexical_hits:
        key = (hit.source, hit.chunk_index)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + hit.rank)

    return scores


def fuse(
    dense_hits: list["RetrievedChunk"],
    lexical_hits: list["LexicalHit"],
    top_n: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RankedChunk]:
    """Builds one RankedChunk per unique (source, chunk_index) present in
    either input list, fills in whichever of dense/lexical score+rank apply
    (None if a candidate came from only one side), sorts by rrf_score
    descending, and returns the top_n. rerank_score is left None here —
    reranker.py fills it in as a separate stage."""
    dense_by_key = {(c.source, c.chunk_index): c for c in dense_hits}
    dense_rank_by_key = {(c.source, c.chunk_index): i for i, c in enumerate(dense_hits, start=1)}
    lexical_by_key = {(h.source, h.chunk_index): h for h in lexical_hits}

    rrf_scores = reciprocal_rank_fusion(dense_hits, lexical_hits, k=rrf_k)

    ranked: list[RankedChunk] = []
    for key in set(dense_by_key) | set(lexical_by_key):
        dense_chunk = dense_by_key.get(key)
        lexical_hit = lexical_by_key.get(key)
        meta = dense_chunk if dense_chunk is not None else lexical_hit

        ranked.append(
            RankedChunk(
                source=meta.source,
                doc_type=meta.doc_type,
                topic=meta.topic,
                date=meta.date,
                chunk_index=meta.chunk_index,
                text=meta.text,
                dense_score=dense_chunk.score if dense_chunk is not None else None,
                dense_rank=dense_rank_by_key.get(key),
                lexical_score=lexical_hit.lexical_score if lexical_hit is not None else None,
                lexical_rank=lexical_hit.rank if lexical_hit is not None else None,
                rrf_score=rrf_scores[key],
            )
        )

    ranked.sort(key=lambda r: r.rrf_score, reverse=True)
    return ranked[:top_n]
