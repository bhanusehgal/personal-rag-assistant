"""Shared dataclasses for the hybrid retrieval pipeline — DESIGN.md section 11.

Fusion and reranking key on (source, chunk_index) throughout, matching the
composite key already used everywhere downstream: generate.py's
_retrieved_key_set, eval/faithfulness.py's cited_keys, and golden_qa.jsonl's
expected_citation_chunks. No other file's key convention needs to change.

All scores in this package follow one convention: higher = more relevant.
(SQLite's bm25() is the opposite by default — see lexical_index.py.)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tools import RetrievedChunk  # noqa: E402


@dataclass
class LexicalHit:
    source: str
    doc_type: str
    topic: str
    date: str
    chunk_index: int
    text: str
    lexical_score: float  # -bm25(): higher = more relevant, see lexical_index.py
    rank: int  # 1-indexed position within this lexical result list


@dataclass
class RankedChunk:
    source: str
    doc_type: str
    topic: str
    date: str
    chunk_index: int
    text: str
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None  # None until CrossEncoderReranker.rerank() runs

    @property
    def key(self) -> tuple[str, int]:
        return (self.source, self.chunk_index)

    def best_score(self) -> float:
        """rerank_score once reranked, else the RRF fusion score — used wherever
        a single scalar 'how relevant is this' is needed (e.g. RetrievedChunk.score)."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return 0.0

    def to_retrieved_chunk(self) -> RetrievedChunk:
        """Backward-compat conversion for call sites that only know the original
        dense-only RetrievedChunk shape (eval/faithfulness.py's score_answer)."""
        return RetrievedChunk(
            source=self.source,
            doc_type=self.doc_type,
            topic=self.topic,
            date=self.date,
            chunk_index=self.chunk_index,
            text=self.text,
            score=self.best_score(),
        )

    def as_dict(self) -> dict:
        """Same shape as RetrievedChunk.as_dict() plus rerank_score, for the
        retrieve tool's JSON result the model actually sees."""
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "topic": self.topic,
            "date": self.date,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "score": round(self.best_score(), 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
        }
