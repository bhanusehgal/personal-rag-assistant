"""Build (or rebuild) the lexical (SQLite FTS5) index used by hybrid
retrieval — DESIGN.md section 11.

Reads the parent project's ../data/faiss_index/metadata.db `chunks` table
READ-ONLY and mirrors it into a new, separate database at
rag-structured-eval/data/lexical_index.db. Always a full rebuild — no
per-file hashing needed, since lexical indexing is cheap (pure SQL, no
embedding calls).

Usage:
    python -m scripts.build_lexical_index
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.lexical_index import LEXICAL_DB_PATH, SOURCE_DB_PATH, build_lexical_index  # noqa: E402

STATE_PATH = LEXICAL_DB_PATH.parent / "lexical_index_state.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/rebuild the lexical (FTS5) index from the parent project's FAISS metadata.db."
    )
    parser.parse_args()  # no flags yet — always a full rebuild; kept for a future --source-db override

    print(f"Reading chunks from {SOURCE_DB_PATH} (read-only)...")
    row_count = build_lexical_index()
    print(f"Built {LEXICAL_DB_PATH} with {row_count} chunk(s).")

    state = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "source_db": str(SOURCE_DB_PATH),
        "source_db_mtime": SOURCE_DB_PATH.stat().st_mtime,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Wrote {STATE_PATH}.")
    print(
        "\nIf you re-run ingest.indexer in the parent project (add/change documents), "
        "re-run this script afterward to keep the lexical index in sync — "
        "there is no automatic incremental sync between the two."
    )


if __name__ == "__main__":
    main()
