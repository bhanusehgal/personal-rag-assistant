"""One-time fix for this machine: the Ollama desktop app persists its own
models-path and context-length settings in a local SQLite DB, which override
plain environment variables for `ollama serve`. See DESIGN.md section 3.

Run with Ollama stopped. Restart the server afterward.

python scripts/fix_ollama_settings_db.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path.home() / "AppData" / "Local" / "Ollama" / "db.sqlite"
MODELS_PATH = r"D:\ollama-models"
CONTEXT_LENGTH = 1024


def main() -> None:
    for suffix in ("-wal", "-shm"):
        stale = DB_PATH.with_name(DB_PATH.name + suffix)
        if stale.exists():
            stale.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE settings SET models = ?, context_length = ? WHERE id = 1",
            (MODELS_PATH, CONTEXT_LENGTH),
        )
        conn.commit()
        cur.execute("SELECT id, models, context_length FROM settings")
        print("Updated settings:", cur.fetchall())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
