"""Rules-based faithfulness scoring — DESIGN.md section 8.

Deliberately no second LLM-judge call: it would double the cost of an
already slow/unreliable chat model and inherit the same unreliability it's
meant to measure. Uses only the structured answer, the exact chunks
retrieved for that question, and the existing (proven-reliable) OllamaEmbedder.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from agent.tools import RetrievedChunk  # noqa: E402
from ingest.embeddings import Embedder  # noqa: E402
from schema import StructuredAnswer  # noqa: E402

GROUNDING_SIMILARITY_THRESHOLD = 0.5


@dataclass
class FaithfulnessResult:
    question_id: str
    citation_existence_ok: bool
    grounding_ok: bool
    abstention_ok: bool
    hallucinated_citations: list[tuple[str, int]]
    ungrounded_citations: list[tuple[str, int]]
    expected_citation_recall: float | None  # diagnostic only, not gating

    @property
    def passed(self) -> bool:
        return self.citation_existence_ok and self.grounding_ok and self.abstention_ok


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def score_answer(
    answer: StructuredAnswer | None,
    retrieved_chunks: list[RetrievedChunk],
    golden: dict,
    embedder: Embedder,
    question_id: str,
) -> FaithfulnessResult:
    chunk_by_key = {(c.source, c.chunk_index): c for c in retrieved_chunks}

    if answer is None:
        return FaithfulnessResult(
            question_id=question_id,
            citation_existence_ok=False,
            grounding_ok=False,
            abstention_ok=False,
            hallucinated_citations=[],
            ungrounded_citations=[],
            expected_citation_recall=0.0 if golden.get("expected_citation_chunks") else None,
        )

    cited_keys = [(c.source, c.chunk_index) for c in answer.citations]
    hallucinated = [k for k in cited_keys if k not in chunk_by_key]
    citation_existence_ok = not hallucinated

    ungrounded: list[tuple[str, int]] = []
    if citation_existence_ok and cited_keys:
        [answer_vec] = embedder.embed([answer.answer])
        for key in cited_keys:
            chunk_text = chunk_by_key[key].text
            [chunk_vec] = embedder.embed([chunk_text])
            sim = _cosine_sim(answer_vec, chunk_vec)
            if sim < GROUNDING_SIMILARITY_THRESHOLD:
                ungrounded.append(key)
    grounding_ok = citation_existence_ok and not ungrounded

    unanswerable = bool(golden.get("unanswerable", False))
    if unanswerable:
        abstention_ok = (answer.has_sufficient_context is False) and (len(answer.citations) == 0)
    else:
        abstention_ok = (answer.has_sufficient_context is True) and (len(answer.citations) > 0)

    expected = golden.get("expected_citation_chunks")
    recall = None
    if expected:
        expected_keys = {(e["source"], e["chunk_index"]) for e in expected}
        matched = expected_keys & set(cited_keys)
        recall = len(matched) / len(expected_keys) if expected_keys else None

    return FaithfulnessResult(
        question_id=question_id,
        citation_existence_ok=citation_existence_ok,
        grounding_ok=grounding_ok,
        abstention_ok=abstention_ok,
        hallucinated_citations=hallucinated,
        ungrounded_citations=ungrounded,
        expected_citation_recall=recall,
    )
