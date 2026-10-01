"""Embedding providers, swappable behind a common interface.

- OpenAIEmbedder: real embeddings via text-embedding-3-small, costs money/calls the API.
- OllamaEmbedder: real embeddings via a local Ollama server (default nomic-embed-text) —
  no network call leaves the machine, no API key needed. Requires `ollama serve` running
  and the model pulled (`ollama pull nomic-embed-text`).
- FakeEmbedder: deterministic hash-based vectors, zero cost, zero network — exists so the
  ingestion pipeline (chunking, FAISS add/remove, incremental re-index, metadata storage)
  can be exercised and tested without an API key. Never use it to judge retrieval quality.
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBEDDING_DIM = 768
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


class Embedder:
    @property
    def dim(self) -> int:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = OPENAI_EMBEDDING_MODEL):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in, "
                "or run with --provider fake to test the pipeline without an API key."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dim = OPENAI_EMBEDDING_DIM

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=list(batch))
            out.extend(d.embedding for d in resp.data)
        return out


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        model: str = OLLAMA_EMBEDDING_MODEL,
        base_url: str | None = None,
        dim: int = OLLAMA_EMBEDDING_DIM,
    ):
        import httpx

        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self._dim = dim
        # Embedding requests haven't shown the multi-minute stalls seen on the
        # chat model (nomic-embed-text is far smaller than qwen2.5:7b), but the
        # timeout is generous rather than tight in case the same underlying
        # local-CPU-inference flakiness (see agent/loop.py) ever reaches this path.
        self.client = httpx.Client(base_url=self.base_url, timeout=300.0)

        try:
            resp = self.client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url} ({e}). "
                "Start it with `ollama serve` (or launch the Ollama app), "
                "or run with --provider fake to test the pipeline without it."
            ) from e

        available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
        if self.model.split(":")[0] not in available:
            raise RuntimeError(
                f"Ollama model '{self.model}' not found locally. "
                f"Pull it first: `ollama pull {self.model}`."
            )

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = self.client.post("/api/embed", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            out.extend(resp.json()["embeddings"])
        return out


class FakeEmbedder(Embedder):
    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            v = rng.normal(size=self._dim).astype("float32")
            v /= np.linalg.norm(v) + 1e-8
            vecs.append(v.tolist())
        return vecs


def get_embedder(provider: str = "openai") -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "ollama":
        return OllamaEmbedder()
    if provider == "fake":
        return FakeEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider!r}")
