"""Per-document metadata via a manifest file rather than auto-classification.

Auto-inferring doc_type/topic/date from PDF content is unreliable enough on
mixed regulatory + internal docs that it's not worth it for v1. Instead,
data/manifest.json maps filename -> {doc_type, topic, date}. Missing entries
fall back to defaults with a warning printed at index time, so ingestion
never hard-fails on an unmapped file — it just won't filter well until you
add the entry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

VALID_DOC_TYPES = {
    "validation_report",
    "regulatory_guidance",
    "closure_pack",
    "project_doc",
    "exam_response",
}
VALID_TOPICS = {"ALM", "CECL", "AML", "IRRBB", "other"}


@dataclass
class DocMetadata:
    doc_type: str
    topic: str
    date: str  # ISO format (YYYY-MM-DD)


DEFAULT_METADATA = DocMetadata(doc_type="unknown", topic="other", date=_date.today().isoformat())


def load_manifest(manifest_path: Path) -> dict[str, DocMetadata]:
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, DocMetadata] = {}
    for filename, entry in raw.items():
        out[filename] = DocMetadata(
            doc_type=entry.get("doc_type", DEFAULT_METADATA.doc_type),
            topic=entry.get("topic", DEFAULT_METADATA.topic),
            date=entry.get("date", DEFAULT_METADATA.date),
        )
    return out


def get_metadata(manifest: dict[str, DocMetadata], filename: str, warn: bool = True) -> DocMetadata:
    if filename in manifest:
        return manifest[filename]
    if warn:
        print(
            f"  [warn] no manifest entry for '{filename}' — using defaults "
            f"(doc_type={DEFAULT_METADATA.doc_type}, topic={DEFAULT_METADATA.topic}). "
            f"Add an entry to data/manifest.json for accurate filtering."
        )
    return DEFAULT_METADATA
