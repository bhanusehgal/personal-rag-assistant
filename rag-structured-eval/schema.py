"""Structured final-answer schema for the agentic generation pipeline.

Kept deliberately flat (no unions/optionals) to minimize the chance of
hitting JSON-schema features Ollama's constrained-decoding engine can't
handle — see DESIGN.md section 7. Verify empirically per model in Stage 1.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

SCHEMA_VERSION = "answer_v1"


class Citation(BaseModel):
    source: str
    chunk_index: int


class StructuredAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    has_sufficient_context: bool
    confidence: Literal["low", "medium", "high"]


def structured_answer_json_schema() -> dict:
    """Wraps StructuredAnswer's schema for Ollama's OpenAI-compat response_format param."""
    schema = StructuredAnswer.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_answer",
            "schema": schema,
            "strict": True,
        },
    }
