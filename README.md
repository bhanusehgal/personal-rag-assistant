# Personal RAG Assistant

Agentic RAG over a personal model risk management (MRM) knowledge base —
validation reports, regulatory guidance, closure packs, and internal project
docs. See the original build plan for full scope; this README covers what's
actually built so far.

**Status: Phase 1 (ingestion pipeline) complete. Phase 2 (agent loop) code
complete, running fully local via Ollama** — see "Known issues" below for a
real caveat on local chat latency/reliability before you rely on it.
Phases 3–5 (memory, eval, UI) are not yet built.

## Repo structure

```
ingest/
  loaders.py     # PDF / DOCX / Markdown text extraction
  chunker.py     # per-doc-type recursive token chunking
  metadata.py    # manifest-driven doc_type/topic/date lookup
  embeddings.py  # OpenAI / local-Ollama / fake-offline embedders, same interface
  indexer.py     # main pipeline: load -> chunk -> embed -> FAISS + SQLite
scripts/
  create_sample_docs.py   # generates 5 synthetic test docs (PDF/DOCX/MD)
agent/
  tools.py       # retrieve() tool: FAISS+SQLite filter-then-search
  loop.py        # tool-calling REPL against a local Ollama chat model
eval/            # Phase 4 (not built yet)
data/
  raw_docs/      # your source documents — gitignored, never committed
  faiss_index/   # index.faiss + metadata.db + state.json — gitignored
  manifest.json  # filename -> {doc_type, topic, date} — gitignored
```

## Setup

```powershell
# from the project root
C:\Python314\python.exe -m venv .venv        # note: use the real interpreter, not
                                              # the `python` Windows Store alias, which
                                              # silently fails inside venv/subprocess calls
.venv\Scripts\python.exe -m pip install -r requirements.txt

copy .env.example .env
# then edit .env and set OPENAI_API_KEY=...
```

## Ingesting documents

1. Drop PDF/DOCX/MD files into `data/raw_docs/`.
2. Add a corresponding entry to `data/manifest.json`:
   ```json
   {
     "my_validation_report.pdf": {
       "doc_type": "validation_report",
       "topic": "ALM",
       "date": "2024-11-15"
     }
   }
   ```
   Valid `doc_type`: `validation_report`, `regulatory_guidance`, `closure_pack`,
   `project_doc`, `exam_response`. Valid `topic`: `ALM`, `CECL`, `AML`, `IRRBB`,
   `other`. Files without a manifest entry still get indexed (with a printed
   warning) under `doc_type=unknown` — they just won't be filterable until you
   add the entry.
3. Build/update the index:
   ```powershell
   .venv\Scripts\python.exe -m ingest.indexer
   ```
   This is **incremental** — already-indexed, unchanged files (by content
   hash) are skipped. Changed files have their old chunks removed and
   re-embedded. Use `--rebuild` to wipe and start over.

   To smoke-test the pipeline without spending on embeddings or needing an
   API key, use `--provider fake` (deterministic hash-based vectors — fine
   for checking that loading/chunking/indexing runs end to end, useless for
   judging actual retrieval quality).

### Trying it with sample data first

```powershell
.venv\Scripts\python.exe scripts\create_sample_docs.py   # writes 5 synthetic docs + manifest entries
.venv\Scripts\python.exe -m ingest.indexer --provider fake # pipeline smoke test
.venv\Scripts\python.exe -m ingest.indexer                 # real index, needs OPENAI_API_KEY
```

This was run during setup and verified end-to-end: all 5 loaders (2×MD,
1×DOCX, 1×PDF, plus the regulatory-guidance MD) extracted clean text,
chunking produced correctly-sized/overlapping chunks on longer text, and
incremental re-runs correctly skip unchanged files. Delete the
`sample_*` files from `data/raw_docs/` (and their manifest entries) whenever
you're ready to switch to your real corpus — `--rebuild` at that point is the
clean way to drop them from the index too.

## Running the agent (Phase 2)

Fully local, no API key: embeddings via Ollama's `nomic-embed-text`, chat +
tool-calling via Ollama's `qwen2.5:7b-instruct` (chosen for reliable native
tool-calling, needed for the `retrieve()` tool). Both are pulled once via
`ollama pull <model>`.

```powershell
# start the Ollama server if it isn't already running
"C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe" serve

# rebuild the index with local embeddings (only needed once, or after doc changes)
.venv\Scripts\python.exe -m ingest.indexer --provider ollama --rebuild

# interactive REPL
.venv\Scripts\python.exe -m agent.loop
```

`--timeout <seconds>` (or `OLLAMA_CHAT_TIMEOUT` env var) overrides the chat
client's per-request timeout — see "Known issues" for why the default is a
generous 900s rather than something interactive-feeling.

### Known issues

**Local chat inference is slow and occasionally stalls/errors outright, on
this hardware specifically** (weak CPU, 2GB MX330 GPU — no real GPU
acceleration). Investigated across two sessions (2026-08-08/09):

- An older Ollama build (v0.16.3) hung forever on generation — fixed by
  upgrading to v0.32.6.
- On v0.32.6, generation itself completes (confirmed via Ollama's own runner
  timing logs) but the HTTP response sometimes doesn't come back for several
  minutes afterward — sometimes as a client-side timeout, sometimes as an
  outright HTTP 500 arriving ~4-5 minutes in. Root cause was never pinned
  down. Ruled out, in order: Windows Defender scanning the Ollama process
  (excluded it, no change), Vulkan GPU-discovery on the Intel iGPU (disabled
  it, made things *worse* — model load itself started crash-looping), flash
  attention specifically (`OLLAMA_FLASH_ATTENTION=0`, no change, still hit
  the same stall + a 500).
- **Current mitigation, not a fix**: `agent/loop.py`'s chat client uses a
  900s timeout and `max_retries=0` (see `DEFAULT_CHAT_TIMEOUT_SECONDS` in
  that file) — long enough to usually ride out the slowness, single-attempt
  so a real failure surfaces in ~15min instead of silently retrying up to
  ~45min. A query that fails is worth just re-running by hand.
- If this becomes too painful for real use, the documented fallback is
  switching **only** the chat model to a cloud API (OpenAI/Anthropic) while
  keeping embeddings local via `OllamaEmbedder` — unblocks immediately at
  the cost of chat turns (and any retrieved excerpts included in context)
  leaving the machine. Deliberately not done yet; revisit if the slowness
  proves unworkable.

## Design notes / open questions from the build plan

**FAISS vs. Chroma:** staying on FAISS. Metadata lives in a SQLite sidecar
keyed by the same integer IDs as the FAISS `IndexIDMap`; Phase 2's retrieval
tool will filter candidate chunk IDs via SQL first, then restrict the FAISS
search to that ID subset. Revisit only if filtering becomes a real bottleneck.

**Chunking:** per-`doc_type` chunk size/overlap (see `CHUNK_PARAMS` in
`ingest/chunker.py`) rather than one global setting — regulatory guidance
gets larger chunks (900 tokens) than internal project docs (600 tokens).
Same splitter strategy (recursive paragraph → line → sentence) for all types.

**Agentic retrieval (Phase 2, not yet built):** retrieval will be exposed as
a callable tool (`retrieve(query, doc_type=None, topic=None, top_k=5)`) via
native function-calling, not a forced always-on RAG step — the model decides
whether to call it, answer from context, or ask a clarifying question.

**Data handling:** `data/raw_docs/`, `data/manifest.json`, and
`data/faiss_index/` are all gitignored — never committed. The only place
document content leaves your machine is the embeddings API call (and later,
chat completion calls) to whichever provider you configure in `.env`.
Decide per-document, before ingesting anything real, whether it should go
through a third-party API at all.

## Known gaps / next steps

- No section-aware PDF chunking — long regulatory PDFs with complex layout
  may chunk worse than the plain-paragraph splitter handles well. Watch for
  this once real docs go in; revisit with a per-type splitter if needed.
- No dedup/near-dup detection across documents.
- Phase 2 end-to-end verification is still pending a clean run — see "Known
  issues" above. Once local chat is confirmed reliable enough (or the cloud
  fallback is wired in), next is Phase 4 (`eval/`, still an empty
  placeholder) and indexing real documents instead of the 5 sample docs.
