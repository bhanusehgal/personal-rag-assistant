"""Recursive token-based chunking, with per-document-type size/overlap.

Regulatory guidance tends to be dense and section-numbered, so it gets larger
chunks. Internal project docs and closure packs are more conversational/bulleted
and chunk better smaller. All of it still uses the same recursive-split
strategy (paragraph -> line -> sentence -> word) rather than a fundamentally
different splitter per type — revisit only if retrieval quality on the
regulatory docs turns out to need section-aware splitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ENC = tiktoken.get_encoding("cl100k_base")

# doc_type -> (chunk_size_tokens, overlap_ratio)
CHUNK_PARAMS = {
    "regulatory_guidance": (900, 0.15),
    "validation_report": (700, 0.15),
    "closure_pack": (700, 0.15),
    "project_doc": (600, 0.15),
    "exam_response": (700, 0.15),
    "default": (700, 0.15),
}


@dataclass
class Chunk:
    text: str
    chunk_index: int


def _token_len(text: str) -> int:
    return len(_ENC.encode(text))


def chunk_text(text: str, doc_type: str = "default") -> list[Chunk]:
    chunk_size, overlap_ratio = CHUNK_PARAMS.get(doc_type, CHUNK_PARAMS["default"])
    overlap = int(chunk_size * overlap_ratio)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]
    return [Chunk(text=p, chunk_index=i) for i, p in enumerate(pieces)]
