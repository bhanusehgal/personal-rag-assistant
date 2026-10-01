"""Build or incrementally update the FAISS index from data/raw_docs.

Incremental strategy: each source file's content is hashed (sha256). On each
run, unchanged files are skipped, changed files have their old chunks removed
(by FAISS vector id, tracked in state.json) and re-added, and new files are
added fresh. There is no separate "re-index everything" step needed for the
common case of dropping a few new documents in — that's the whole point of
tracking file hashes instead of just always rebuilding.

Metadata (source, doc_type, topic, date, chunk_index, chunk text) lives in a
SQLite sidecar (metadata.db) keyed by the same integer ids used in the FAISS
IndexIDMap, so a vector search hit maps straight back to its row.

Usage:
    python -m ingest.indexer                    # incremental update, real OpenAI embeddings
    python -m ingest.indexer --provider ollama   # incremental update, local Ollama embeddings
    python -m ingest.indexer --provider fake     # pipeline test, no API key/cost
    python -m ingest.indexer --rebuild           # wipe and rebuild from scratch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

from ingest.chunker import chunk_text
from ingest.embeddings import get_embedder
from ingest.loaders import iter_source_files, load_document
from ingest.metadata import get_metadata, load_manifest

ROOT = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = ROOT / "data" / "raw_docs"
INDEX_DIR = ROOT / "data" / "faiss_index"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
STATE_PATH = INDEX_DIR / "state.json"
DB_PATH = INDEX_DIR / "metadata.db"
INDEX_PATH = INDEX_DIR / "index.faiss"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            topic TEXT NOT NULL,
            date TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic)")
    conn.commit()


def _empty_state() -> dict:
    return {"files": {}, "next_id": 0, "embedding_dim": None, "embedding_provider": None}


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return _empty_state()


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_or_update_index(provider: str = "openai", rebuild: bool = False) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    state = _empty_state() if rebuild else _load_state()
    manifest = load_manifest(MANIFEST_PATH)
    embedder = get_embedder(provider)

    if rebuild and INDEX_PATH.exists():
        INDEX_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    if rebuild:
        conn.execute("DROP TABLE IF EXISTS chunks")
    _init_db(conn)

    if INDEX_PATH.exists() and not rebuild:
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedder.dim))
        state["embedding_dim"] = embedder.dim
        state["embedding_provider"] = provider

    if state.get("embedding_provider") not in (None, provider):
        conn.close()
        raise RuntimeError(
            f"Existing index was built with embedding provider "
            f"'{state['embedding_provider']}' but this run is using '{provider}'. "
            f"Mixing embedding spaces silently breaks similarity search — "
            f"pass --rebuild to start the index over with the new provider."
        )

    files = list(iter_source_files(RAW_DOCS_DIR))
    if not files:
        print(f"No documents found in {RAW_DOCS_DIR}. Add PDF/DOCX/MD files and re-run.")
        conn.close()
        return

    added_chunks = 0
    for path in files:
        rel_name = path.name
        current_hash = _file_hash(path)
        prev = state["files"].get(rel_name)

        if prev and prev["hash"] == current_hash:
            print(f"  [skip] {rel_name} (unchanged)")
            continue

        if prev:
            old_ids = prev["chunk_ids"]
            print(f"  [update] {rel_name} ({len(old_ids)} old chunks removed)")
            index.remove_ids(np.array(old_ids, dtype="int64"))
            conn.execute(
                "DELETE FROM chunks WHERE id IN (%s)" % ",".join("?" * len(old_ids)),
                old_ids,
            )

        meta = get_metadata(manifest, rel_name)
        doc = load_document(path)
        chunks = chunk_text(doc.text, doc_type=meta.doc_type)
        if not chunks:
            print(f"  [warn] {rel_name} produced 0 chunks (empty or unreadable) — skipping")
            continue

        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
        vecs = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vecs)  # so inner product == cosine similarity

        ids = list(range(state["next_id"], state["next_id"] + len(chunks)))
        state["next_id"] += len(chunks)
        index.add_with_ids(vecs, np.array(ids, dtype="int64"))

        conn.executemany(
            "INSERT INTO chunks (id, source, doc_type, topic, date, chunk_index, text) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (cid, rel_name, meta.doc_type, meta.topic, meta.date, c.chunk_index, c.text)
                for cid, c in zip(ids, chunks)
            ],
        )

        state["files"][rel_name] = {"hash": current_hash, "chunk_ids": ids}
        added_chunks += len(chunks)
        print(f"  [ok] {rel_name}: {len(chunks)} chunks ({meta.doc_type}/{meta.topic})")

    conn.commit()
    conn.close()
    faiss.write_index(index, str(INDEX_PATH))
    _save_state(state)

    print(
        f"\nDone. {added_chunks} chunks embedded/updated this run. "
        f"Index now has {index.ntotal} total vectors."
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build or incrementally update the FAISS index from data/raw_docs."
    )
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the index from scratch.")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "ollama", "fake"],
        help="Embedding provider. 'ollama' runs fully local (needs `ollama serve` + model "
        "pulled). 'fake' is deterministic/offline, for pipeline testing only.",
    )
    args = parser.parse_args()
    build_or_update_index(provider=args.provider, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
