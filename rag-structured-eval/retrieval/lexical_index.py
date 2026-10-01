"""Lexical (BM25) retrieval via SQLite FTS5 — DESIGN.md section 11.

Uses SQLite's built-in FTS5 module rather than a rank_bm25 pip dependency,
matching this codebase's existing SQLite-sidecar pattern (agent/tools.py's
metadata.db). Builds a NEW, separate database
(rag-structured-eval/data/lexical_index.db) by reading the parent project's
../data/faiss_index/metadata.db `chunks` table READ-ONLY and mirroring it
into an FTS5 virtual table — the parent's index is never modified or
rebuilt by this project (see DESIGN.md section 2's reuse boundary).

Two correctness details, load-bearing, not incidental:
- SQLite's bm25() returns LOWER = more relevant (the opposite of FAISS
  cosine's "higher = better"). We store lexical_score = -bm25(...) so every
  score in this package follows one "higher = more relevant" convention.
- Natural-language questions contain characters ('?', "'", ':', '-') that
  are meaningful FTS5 query syntax and raise sqlite3.OperationalError on
  MATCH if passed raw. Queries are sanitized: tokenized, non-alphanumeric
  characters stripped per token, joined with OR (a permissive "any
  distinctive term matches" default appropriate for a recall-oriented
  lexical stage feeding into fusion + reranking, not a final answer).
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .types import LexicalHit

RAG_EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RAG_EVAL_ROOT.parent

LEXICAL_DB_PATH = RAG_EVAL_ROOT / "data" / "lexical_index.db"
SOURCE_DB_PATH = REPO_ROOT / "data" / "faiss_index" / "metadata.db"

_FTS_TOKENIZE = "porter unicode61 remove_diacritics 2"


class LexicalIndexNotFoundError(RuntimeError):
    pass


def _sanitize_fts_query(query: str) -> str | None:
    """Tokenize + strip to alphanumeric-only tokens, quote each, join with OR.
    Returns None if the query has no usable tokens at all."""
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def build_lexical_index(
    source_db_path: Path = SOURCE_DB_PATH,
    dest_db_path: Path = LEXICAL_DB_PATH,
) -> int:
    """Always a full drop-and-rebuild — unlike ingest/indexer.py's incremental
    FAISS logic, lexical indexing is cheap (no embedding calls, pure SQL),
    so there's no cost reason to track per-file hashes here. Reads the
    source read-only; writes the destination fresh. Returns the row count."""
    if not source_db_path.exists():
        raise LexicalIndexNotFoundError(
            f"No source index found at {source_db_path}. Run `python -m ingest.indexer` "
            "in the parent project first (see ../README.md)."
        )

    dest_db_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_db_path.exists():
        dest_db_path.unlink()

    src_uri = source_db_path.resolve().as_uri() + "?mode=ro"
    src_conn = sqlite3.connect(src_uri, uri=True)
    dest_conn = sqlite3.connect(str(dest_db_path))
    try:
        dest_conn.execute(
            f"""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                source UNINDEXED,
                doc_type UNINDEXED,
                topic UNINDEXED,
                date UNINDEXED,
                chunk_index UNINDEXED,
                tokenize='{_FTS_TOKENIZE}'
            )
            """
        )
        rows = src_conn.execute(
            "SELECT id, source, doc_type, topic, date, chunk_index, text FROM chunks"
        ).fetchall()
        dest_conn.executemany(
            "INSERT INTO chunks_fts(rowid, text, source, doc_type, topic, date, chunk_index) "
            "VALUES (?,?,?,?,?,?,?)",
            [(cid, text, source, doc_type, topic, date, chunk_index)
             for cid, source, doc_type, topic, date, chunk_index, text in rows],
        )
        dest_conn.commit()
        return len(rows)
    finally:
        src_conn.close()
        dest_conn.close()


class LexicalStore:
    """Query side. Raises LexicalIndexNotFoundError if the index hasn't been
    built yet — mirrors agent.tools.Retriever's IndexNotFoundError pattern."""

    def __init__(self, db_path: Path = LEXICAL_DB_PATH):
        if not Path(db_path).exists():
            raise LexicalIndexNotFoundError(
                f"No lexical index at {db_path}. Run `python -m scripts.build_lexical_index` first."
            )
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)

    def search(
        self,
        query: str,
        doc_type: str | None = None,
        topic: str | None = None,
        top_k: int = 10,
    ) -> list[LexicalHit]:
        match_expr = _sanitize_fts_query(query)
        if match_expr is None:
            return []  # nothing tokenizable in the query — no lexical signal possible

        clauses = ["chunks_fts MATCH ?"]
        params: list = [match_expr]
        if doc_type:
            clauses.append("doc_type = ?")
            params.append(doc_type)
        if topic:
            clauses.append("topic = ?")
            params.append(topic)
        params.append(top_k)

        sql = f"""
            SELECT source, doc_type, topic, date, chunk_index, text, bm25(chunks_fts) AS raw_score
            FROM chunks_fts
            WHERE {' AND '.join(clauses)}
            ORDER BY bm25(chunks_fts) ASC
            LIMIT ?
        """
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Malformed MATCH expression (e.g. an FTS5 reserved token slipped
            # through sanitization) — treat as "no lexical signal", not a crash.
            return []

        hits: list[LexicalHit] = []
        for rank, (source, dt, tp, date, chunk_index, text, raw_score) in enumerate(rows, start=1):
            hits.append(
                LexicalHit(
                    source=source,
                    doc_type=dt,
                    topic=tp,
                    date=date,
                    chunk_index=chunk_index,
                    text=text,
                    lexical_score=-raw_score,
                    rank=rank,
                )
            )
        return hits
