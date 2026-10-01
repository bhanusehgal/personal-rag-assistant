"""HybridRetriever — orchestrates dense + lexical + RRF fusion + cross-encoder
reranking + a retrieval-quality report into the one retrieve() call sites
across this project already know how to use — DESIGN.md section 11.

RETRIEVE_TOOL_SCHEMA (the model-facing tool contract: name, params,
description) is reused from agent.tools UNCHANGED — only what happens
behind the tool dispatch differs from the dense-only version.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tools import Retriever  # noqa: E402
from ingest.embeddings import get_embedder  # noqa: E402
from ingest.metadata import VALID_DOC_TYPES, VALID_TOPICS  # noqa: E402

from .fusion import DEFAULT_RRF_K, fuse  # noqa: E402
from .lexical_index import LexicalStore  # noqa: E402
from .quality import RetrievalQualityReport, assess_quality  # noqa: E402
from .reranker import CrossEncoderReranker  # noqa: E402
from .types import RankedChunk  # noqa: E402

# The model-facing tool contract for this project's agentic arm: `query` only.
# agent.tools.RETRIEVE_TOOL_SCHEMA also offers doc_type / topic / top_k, and its
# filters are filter-then-search — one wrong guess by a small model returns zero
# chunks on a corpus this size (see PROGRESS.md, 2026-09-30). The reranker
# already puts the right chunk first without them, so they're not offered here.
RETRIEVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the user's personal MRM document corpus (validation reports, "
            "regulatory guidance, closure packs, project docs, exam responses) for "
            "passages relevant to a query. Use this whenever answering requires "
            "specific facts, figures, or wording from those documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
            },
            "required": ["query"],
        },
    },
}

DEFAULT_FUSE_POOL = 10  # how many dense / lexical hits each side contributes into fusion
DEFAULT_RERANK_POOL = 10  # how many fused candidates the cross-encoder actually scores


class HybridRetriever:
    """dense.retrieve() -> lexical.search() -> fuse() (RRF) -> reranker.rerank()
    -> RetrievalQualityReport, on every call. last_quality_report is always
    set after retrieve() runs, regardless of whether a caller later gates on it."""

    def __init__(
        self,
        dense: Retriever,
        lexical: LexicalStore,
        reranker: CrossEncoderReranker,
        rrf_k: int = DEFAULT_RRF_K,
        fuse_pool: int = DEFAULT_FUSE_POOL,
        rerank_pool: int = DEFAULT_RERANK_POOL,
    ):
        self.dense = dense
        self.lexical = lexical
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.fuse_pool = fuse_pool
        self.rerank_pool = rerank_pool
        self.last_quality_report: RetrievalQualityReport | None = None

    def retrieve(
        self,
        query: str,
        doc_type: str | None = None,
        topic: str | None = None,
        top_k: int = 5,
    ) -> list[RankedChunk]:
        dense_hits = self.dense.retrieve(query, doc_type=doc_type, topic=topic, top_k=self.fuse_pool)
        lexical_hits = self.lexical.search(query, doc_type=doc_type, topic=topic, top_k=self.fuse_pool)

        fused = fuse(dense_hits, lexical_hits, top_n=self.rerank_pool, rrf_k=self.rrf_k)
        reranked = self.reranker.rerank(query, fused) if fused else []

        self.last_quality_report = assess_quality(query, dense_hits, lexical_hits, fused, reranked)
        return reranked[:top_k]


def make_hybrid_retrieve_tool(hybrid: HybridRetriever):
    """Same call signature / same {"results": [...]} | {"error": ...} |
    {"results": [], "note": ...} return shape as agent.tools.make_retrieve_tool,
    so generate.py's swap is a one-line import change, nothing else."""

    def _retrieve(
        query: str,
        doc_type: str | None = None,
        topic: str | None = None,
        top_k: int = 5,
    ) -> dict:
        if doc_type is not None and doc_type not in VALID_DOC_TYPES:
            return {"error": f"Invalid doc_type {doc_type!r}. Valid: {sorted(VALID_DOC_TYPES)}"}
        if topic is not None and topic not in VALID_TOPICS:
            return {"error": f"Invalid topic {topic!r}. Valid: {sorted(VALID_TOPICS)}"}
        # Tool-call arguments are model-generated JSON, not schema-enforced —
        # some models emit top_k as a string. Coerce defensively at this
        # boundary, matching agent/tools.py's make_retrieve_tool.
        try:
            top_k = int(top_k) if top_k else 5
        except (TypeError, ValueError):
            return {"error": f"Invalid top_k {top_k!r}: must be an integer."}

        chunks = hybrid.retrieve(query, doc_type=doc_type, topic=topic, top_k=top_k)
        if not chunks:
            return {"results": [], "note": "No matching chunks found in the corpus."}
        return {"results": [c.as_dict() for c in chunks]}

    return _retrieve


def build_default_hybrid_retriever(embed_provider: str = "ollama") -> HybridRetriever:
    """One-time construction helper. Fails fast (via LexicalIndexNotFoundError,
    propagated from LexicalStore) if `python -m scripts.build_lexical_index`
    hasn't been run yet — mirrors agent.tools.Retriever's IndexNotFoundError
    pattern for the dense side."""
    embedder = get_embedder(embed_provider)
    dense = Retriever(embedder)
    lexical = LexicalStore()
    reranker = CrossEncoderReranker()
    return HybridRetriever(dense=dense, lexical=lexical, reranker=reranker)
