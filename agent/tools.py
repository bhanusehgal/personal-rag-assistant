"""Tool definitions for the agent loop.

Currently one tool: retrieve(). It's a plain function wrapped in an OpenAI-style
tool schema (TOOLS) — the agent loop decides whether/when to call it; retrieval
is never forced. Filtering is filter-then-search: doc_type/topic constraints are
turned into a FAISS IDSelector built from the SQLite sidecar, so the vector
search only ever ranks candidates that already pass the metadata filter.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from ingest.embeddings import Embedder
from ingest.metadata import VALID_DOC_TYPES, VALID_TOPICS

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "faiss_index"
STATE_PATH = INDEX_DIR / "state.json"
DB_PATH = INDEX_DIR / "metadata.db"
INDEX_PATH = INDEX_DIR / "index.faiss"


@dataclass
class RetrievedChunk:
    source: str
    doc_type: str
    topic: str
    date: str
    chunk_index: int
    text: str
    score: float

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "topic": self.topic,
            "date": self.date,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "score": round(self.score, 4),
        }


class IndexNotFoundError(RuntimeError):
    pass


class Retriever:
    """Wraps the FAISS index + SQLite sidecar built by ingest/indexer.py.

    Holds one embedder instance, used only to embed queries — it must be the
    same provider the index was built with (checked at load time via
    state.json), or similarity scores are meaningless.
    """

    def __init__(self, embedder: Embedder):
        if not INDEX_PATH.exists() or not DB_PATH.exists():
            raise IndexNotFoundError(
                f"No index found at {INDEX_DIR}. Run `python -m ingest.indexer` "
                "(see README) before starting the agent."
            )
        import json

        state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
        indexed_dim = state.get("embedding_dim")
        if indexed_dim is not None and indexed_dim != embedder.dim:
            raise RuntimeError(
                f"Index was built with a {indexed_dim}-dim embedder "
                f"(provider={state.get('embedding_provider')!r}) but the agent's "
                f"embedder produces {embedder.dim}-dim vectors. Re-run "
                f"`python -m ingest.indexer` with a matching --provider, or point "
                f"the agent at the provider the index was built with."
            )

        self.embedder = embedder
        self.index = faiss.read_index(str(INDEX_PATH))
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    def retrieve(
        self,
        query: str,
        doc_type: str | None = None,
        topic: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        selector = None
        if doc_type or topic:
            clauses, params = [], []
            if doc_type:
                clauses.append("doc_type = ?")
                params.append(doc_type)
            if topic:
                clauses.append("topic = ?")
                params.append(topic)
            rows = self.conn.execute(
                f"SELECT id FROM chunks WHERE {' AND '.join(clauses)}", params
            ).fetchall()
            candidate_ids = [r[0] for r in rows]
            if not candidate_ids:
                return []  # filter matched nothing — no point searching at all
            selector = faiss.IDSelectorBatch(np.array(candidate_ids, dtype="int64"))

        [qvec] = self.embedder.embed([query])
        qarr = np.array([qvec], dtype="float32")
        faiss.normalize_L2(qarr)

        search_k = min(top_k, self.index.ntotal) or 1
        if selector is not None:
            D, I = self.index.search(qarr, search_k, params=faiss.SearchParameters(sel=selector))
        else:
            D, I = self.index.search(qarr, search_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            row = self.conn.execute(
                "SELECT source, doc_type, topic, date, chunk_index, text FROM chunks WHERE id = ?",
                (int(idx),),
            ).fetchone()
            if row is None:
                continue
            source, dt, tp, date, chunk_index, text = row
            results.append(
                RetrievedChunk(
                    source=source,
                    doc_type=dt,
                    topic=tp,
                    date=date,
                    chunk_index=chunk_index,
                    text=text,
                    score=float(score),
                )
            )
        return results


RETRIEVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the user's personal MRM document corpus (validation reports, "
            "regulatory guidance, closure packs, project docs, exam responses) for "
            "chunks relevant to a query. Use this whenever answering requires "
            "specific facts, figures, or wording from those documents rather than "
            "general knowledge. Returns the most relevant chunks with their source "
            "filename and chunk index for citation — if nothing relevant comes back, "
            "say so instead of guessing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "doc_type": {
                    "type": "string",
                    "enum": sorted(VALID_DOC_TYPES),
                    "description": "Optional filter to a single document type.",
                },
                "topic": {
                    "type": "string",
                    "enum": sorted(VALID_TOPICS),
                    "description": "Optional filter to a single topic.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def make_retrieve_tool(retriever: Retriever):
    """Returns a callable matching the retrieve tool's JSON schema, for the agent loop
    to dispatch tool calls to by name."""

    def _retrieve(query: str, doc_type: str | None = None, topic: str | None = None, top_k: int = 5) -> dict:
        if doc_type is not None and doc_type not in VALID_DOC_TYPES:
            return {"error": f"Invalid doc_type {doc_type!r}. Valid: {sorted(VALID_DOC_TYPES)}"}
        if topic is not None and topic not in VALID_TOPICS:
            return {"error": f"Invalid topic {topic!r}. Valid: {sorted(VALID_TOPICS)}"}
        # Tool-call arguments come from model-generated JSON (not schema-enforced
        # the way structured/response_format output is) — some models emit
        # top_k as a string (e.g. "5"). Coerce defensively at this boundary.
        try:
            top_k = int(top_k) if top_k else 5
        except (TypeError, ValueError):
            return {"error": f"Invalid top_k {top_k!r}: must be an integer."}
        chunks = retriever.retrieve(query, doc_type=doc_type, topic=topic, top_k=top_k)
        if not chunks:
            return {"results": [], "note": "No matching chunks found in the corpus."}
        return {"results": [c.as_dict() for c in chunks]}

    return _retrieve
