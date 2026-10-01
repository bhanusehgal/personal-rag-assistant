# DESIGN.md — rag-structured-eval

Technical design for the structured, benchmarked RAG generation layer built on top of the existing "Personal RAG Assistant" project. See `../README.md` and `../agent/loop.py` for the parent project this reuses retrieval/embeddings from. See `PROGRESS.md` for current build status.

## 1. Why this project exists

The parent project's Phase 2 chat agent (`agent/loop.py`, `qwen2.5:7b-instruct` via local Ollama) is unreliable on this hardware — roughly 1-in-9 successful completions historically, multi-minute-to-14-minute latencies, and undiagnosed stalls (Ollama upgrade v0.16→v0.32 fixed one infinite hang; Defender exclusion, disabling Vulkan discovery, and disabling flash attention were all tried and ruled out as fixes for the rest). That's an anecdotal, ad-hoc finding. This project turns "is this reliable/fast/accurate enough" into something measured:

- Prompts become versioned, inspectable artifacts instead of a hardcoded string.
- Generation is schema-constrained and validated instead of free text, with a bounded retry loop.
- A hard, rules-based quality gate stops further work the moment answer faithfulness drops below threshold, rather than letting a subtly-broken prompt/model silently ship.
- Every inference call is instrumented for TTFT, tokens/sec, and total latency — so "model X is fine" or "model X is too slow" is a number, not a feeling.
- A systematic quality-vs-speed comparison across smaller GGUF-quantized models (Llama 3.2 3B, Phi-4-mini, Mistral 7B) tests whether a smaller model is simply more *reliable* on this CPU-only hardware than the 7B baseline — not just faster.

## 2. Scope and reuse boundary

**Reused as-is, not rebuilt**: `ingest/` (loaders, chunker, indexer, `OllamaEmbedder`) and `agent/tools.py` (`Retriever`, `RetrievedChunk`, `RETRIEVE_TOOL_SCHEMA`, `make_retrieve_tool`). This project imports them via `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` since the parent repo has no packaging. The existing FAISS index at `../data/faiss_index/` is used unchanged — no re-ingestion.

**New in this project**: the generation layer (prompts, structured output, validation/retry, agentic tool-calling orchestration reusing `agent/tools.py`'s tool schema), the benchmarking harness (golden Q&A set, faithfulness scoring, quality gate), and the inference-metrics instrumentation.

**Generation scope decision**: this project uses the **full agentic tool-calling flow** (the model decides if/when to call `retrieve`, same as `agent/loop.py` today) rather than a simplified single-shot retrieve-then-generate. This was chosen deliberately over the simpler alternative, accepting more latency/variance per question, because the benchmark should reflect how the system will actually be used.

## 3. Environment constraints that shaped this design

- **System RAM is only 7.7GB total** (observed as low as 0.2GB free under normal desktop load — VS Code, Chrome, multiple Claude Code sessions) — discovered during the Stage 0 smoke test when the first two attempts got OOM-killed before logging a single result, even for the smallest candidate model. This is a tighter constraint than the CPU/GPU limits below: a 7B Q4 model alone needs ~4.5-5GB for weights before KV cache and OS/Ollama overhead.
  - **Root cause of a second, compounding problem found while fixing this**: the Ollama *desktop app* persists its own settings — model storage path and default context length — in a local SQLite database (`%LOCALAPPDATA%\Ollama\db.sqlite`, table `settings`), which **overrides plain environment variables** (`OLLAMA_MODELS`, `OLLAMA_CONTEXT_LENGTH`) for a `ollama serve` process started from the CLI on this machine/version. Setting `$env:OLLAMA_MODELS` or `$env:OLLAMA_CONTEXT_LENGTH` before starting the server — via `Start-Process`, `cmd /c "set ... && ollama serve"`, or a standalone `.bat` file — had **no effect**; the server logged the old defaults every time. The fix was to update the `settings` row directly (`UPDATE settings SET models=..., context_length=... WHERE id=1`, after stopping the server and clearing `db.sqlite-wal`/`db.sqlite-shm`) rather than relying on env vars for these two settings.
  - **Operational requirement for every stage, not just Stage 0**: (1) run the Ollama server with `OLLAMA_MAX_LOADED_MODELS=1` and a short `OLLAMA_KEEP_ALIVE` (e.g. `30s`) — env vars for these two still work; (2) keep `context_length` at `1024` in the settings DB (validated: let the smallest model complete successfully at 73s with no OOM, vs. an immediate kill at the default 4096); (3) never assume two models can be warm at once — explicitly `ollama stop <model>` before switching models in any script that loops over the model list (`scripts/smoke_test.py`, `compare_models.py`), with a short sleep after to let memory actually free.
- **CPU-only inference**: the GPU (MX330, 2GB VRAM) reports performance state `ERR!` — treated as unusable for acceleration. Every model comparison in this project is a CPU-inference comparison.
- **Disk**: `C:` had only 16GB free vs `D:`'s 917GB — Ollama's model store was moved to `D:\ollama-models` (see `scripts/setup_ollama_models.ps1`) before pulling ~20GB of new GGUF variants.
- **Ollama version**: upgraded from v0.32.6 to v0.34.2 as a first step, on the chance the unresolved chat-completion stalls are fixed upstream. Known-ruled-out fixes (don't re-try): Defender process exclusion, disabling Vulkan iGPU discovery (`GGML_VK_VISIBLE_DEVICES=""`, made it worse), `OLLAMA_FLASH_ATTENTION=0`.

## 4. Model inventory

| Model | Tag | Size | Context | Notes |
|---|---|---|---|---|
| Qwen 2.5 7B (baseline) | `qwen2.5:7b-instruct` | 4.7GB | — | Already pulled; the unreliable Phase-2 baseline this project benchmarks against. |
| Llama 3.2 3B | `llama3.2:3b-instruct-q4_K_M` | 2.0GB | — | |
| Llama 3.2 3B | `llama3.2:3b-instruct-q5_K_M` | 2.3GB | — | |
| Phi-4-mini | `phi4-mini:3.8b-q4_K_M` | 2.5GB | 128K | |
| Phi-4-mini | `phi4-mini:3.8b-q8_0` | 4.1GB | **4K** | No q5 tag exists in the Ollama library for this model — this pair is Q4-vs-Q8, not Q4-vs-Q5. The 4K context ceiling (vs 128K on q4) may truncate the agentic transcript + retrieved chunks on some questions; this is treated as its own recorded failure mode, not silently absorbed. |

Tags verified live against ollama.com/library at design time, not guessed.

**Mistral 7B dropped from the comparison after Stage 0**: `mistral:7b-instruct-q4_K_M` and `mistral:7b-instruct-q5_K_M` are pulled locally but excluded from `compare_models.py`'s model list — this machine's ~7.7GB total RAM couldn't reliably load either variant even after the Stage 0 context-length fix (repeated OOM kills, see PROGRESS.md). The comparison study is now 5 models: `qwen2.5:7b-instruct` (the sole 7B-class point, already confirmed loadable pre-Stage-0) plus the four Llama 3.2 / Phi-4-mini variants.

## 5. Prompt storage design

Prompts are first-class, versioned files — not hardcoded strings like `agent/loop.py`'s `SYSTEM_PROMPT`.

```
prompts/
  registry.json     # prompt_id -> {file, version, created, compatible_models, notes}
  system_v1.md       # raw prompt text, used verbatim as the system message
```

`prompt_store.py` exposes:
- `PromptVersion` dataclass: `id`, `version`, `text`, `compatible_models`, `notes`
- `load_prompt(prompt_id) -> PromptVersion`
- `list_prompts() -> list[PromptVersion]`

No YAML dependency — metadata (`registry.json`) and prompt body (plain `.md`) are kept separate on purpose.

`system_v1.md` adapts `agent/loop.py`'s two non-negotiable rules — cite only what's in retrieved context; say explicitly when context is insufficient rather than filling from general knowledge — but targets the structured schema (§7): the model must only cite `(source, chunk_index)` pairs that actually appeared in a `retrieve` tool result during that conversation, and must set `has_sufficient_context: false` with empty citations when nothing relevant came back.

**Reproducibility**: every eval run stamps `prompt_version`, `model`, `temperature`, and `schema_version` into that run's `manifest.json` and into every row of `metrics.jsonl`/`results.jsonl` — so results are comparable by filtering/grouping on these fields, never by relying on filenames.

## 6. Inference performance metrics

`metrics.py`'s `InferenceClient` is the single call site every chat completion goes through — nothing calls the raw `openai` client directly elsewhere. Every call (each tool-calling round, each structured-final attempt, each retry) produces one `InferenceRecord`, appended to that run's `metrics.jsonl`, success or failure:

- **TTFT (time to first token)**: streamed via `stream=True`; timestamp at the first content-or-tool-call-delta chunk minus request start.
- **Tokens/sec**: completion tokens ÷ (stream-end time − first-token time). Token counts come from the OpenAI-compat stream's `usage` field (`stream_options={"include_usage": True}`) when Ollama populates it; otherwise fall back to a `tiktoken cl100k_base` approximation (same encoder `ingest/chunker.py` already uses) — **not** each model's real tokenizer. `token_count_method` on every record says which was used.
- **Total latency**: full wall-clock request-to-completion time.

Per-question totals for model comparison (§9) sum latency/tokens across *all* rounds of that question's agentic round-trip, not just the final structured call — that's the real end-to-end cost a user would experience.

## 7. Structured generation, validation, and retry (Phase 2 design)

### Schema

`schema.py` — deliberately flat Pydantic models (no unions/optionals), to minimize the chance of hitting JSON-schema features Ollama's constrained-decoding engine can't handle:

```python
SCHEMA_VERSION = "answer_v1"

class Citation(BaseModel):
    source: str
    chunk_index: int

class StructuredAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    has_sufficient_context: bool
    confidence: Literal["low", "medium", "high"]
```

### Reconciling structured output with agentic tool-calling

> **Updated 2026-09-30 — see section 12.** The two-phase split below still holds for the agentic arm, but the final call's prompt is now built from scratch rather than replaying the tool transcript, the tool is query-only, and retrieve-first (no tool calling at all) is the default path.

Ollama's `response_format` (JSON-schema-constrained decoding) and `tools`/`tool_choice` are never combined in the same request. So `generate.py`'s `run_agentic_structured()` runs in two phases per question:

1. **Tool-calling phase** (mirrors `agent/loop.py`'s `run_agent_loop`, up to `max_tool_rounds=5`): each round calls the model with `tools=[RETRIEVE_TOOL_SCHEMA]`, `tool_choice="auto"`. If the model returns no tool calls, the phase ends (it believes it has enough context); if it does, dispatch via `make_retrieve_tool(retriever)` and continue.
2. **Structured-final phase** — **always** a separate, dedicated call with `tools` removed and `response_format=structured_answer_json_schema()` set, regardless of whether phase 1 ended naturally or was cut off at `max_tool_rounds`. This guarantees every final answer is schema-constrained, since the round that decided to stop calling tools was never itself schema-constrained.

### Validation and retry

On the structured-final call's response, three distinct failure modes are checked and logged separately (not collapsed into one bool), because Stage 0/1 needs to tell "the constraint mechanism doesn't work on this model" apart from "the model is just wrong":

- `schema_ignored` — model returned non-JSON despite `response_format`.
- `json_parse_error` — JSON parsed but doesn't match `StructuredAnswer`.
- `validation_error` — structurally valid but fails a business rule, chiefly: a cited `(source, chunk_index)` doesn't match any chunk actually retrieved in this conversation (hallucinated citation).

On any failure, the concrete error is appended as a message and the structured-final call is retried, up to `max_final_retries=2` (3 attempts total). If all attempts fail, the question is recorded as a failure with the final error type — never silently dropped.

**Biggest open technical risk**: whether `response_format` reliably constrains output when issued as a follow-up call after a tool-calling phase in the *same* conversation — this exact sequence is unverified until Stage 1. Fallback if it proves unreliable: bypass the OpenAI-compat endpoint for the structured-final call and hit Ollama's native `/api/chat` directly via `httpx` with a top-level `"format"` field instead.

## 8. Benchmark threshold / quality gate

### Golden set

`eval/golden_qa.jsonl` — one JSON object per line: `id`, `question`, `expected_citation_chunks` (diagnostic recall signal, not gating), `unanswerable` (bool, tests correct abstention), `notes`. Started with questions against the 5 existing synthetic sample docs (no real MRM corpus ingested yet); to be expanded once real documents are loaded.

### Faithfulness scoring — rules-based, deliberately no second LLM-judge call

An LLM-judge pass would double the cost of an already slow/unreliable chat model and inherit the same unreliability it's meant to measure. `eval/faithfulness.py` instead checks, per question, using only the structured answer, the exact chunks retrieved for that question, and the already-proven-reliable `OllamaEmbedder`:

1. **Citation existence** (hard gate): every cited `(source, chunk_index)` must be among that question's actually-retrieved chunks. A citation to anything else = hallucinated citation = automatic fail.
2. **Grounding overlap** (soft check): cosine similarity (via `OllamaEmbedder`) between the answer text and each cited chunk's text must be ≥ 0.5.
3. **Correct abstention**: `unanswerable: true` goldens require `has_sufficient_context=False` and empty citations; `unanswerable: false` goldens require the opposite.

A question passes iff all three hold. **Gate metric = % of golden questions that pass.**

### Gate procedure

`eval/gate.py`: `FAITHFULNESS_THRESHOLD = 0.80` — **explicitly a placeholder**, to be confirmed with the user after Stage 1's first real numbers, not treated as final. `python -m eval.gate --run-id <id>` loads that run's `results.jsonl`, prints a `PASS` or `STOP: faithfulness gate FAILED (…)` banner, and exits 0/1 accordingly.

**Process rule**: re-run and pass this gate after any change to prompt version, schema, or retrieval config, before continuing to the next stage. A failed gate means stop and fix, not proceed and note it.

## 9. Model comparison study (Phase 3 design)

`compare_models.py` runs the same golden set, same prompt version, at **one fixed temperature** (the best one found in Stage 2 — re-sweeping all 4 temperatures × 5 models is prohibitive on this hardware) across all 5 remaining models (see section 4 for why Mistral 7B was dropped). Per-model `try/except` isolation means one model's total failure doesn't abort the sweep — it's recorded as `success_rate: 0` and the run continues, which is expected given the baseline's documented ~1-in-9 historical rate. Per-model timeouts are sized to model weight (smaller/faster ceilings for the 3B/mini variants, 900s matching `DEFAULT_CHAT_TIMEOUT_SECONDS` for the `qwen2.5:7b-instruct` baseline).

`analyze_results.py` produces `comparison_table.md` (one row per model — faithfulness rate, validity rate, mean TTFT, mean tokens/sec, mean total latency, success rate, model size — sorted by faithfulness descending) and one `quality_vs_speed.png` scatter chart (x = mean total latency, y = faithfulness rate). One chart, no dashboard.

## 10. Temperature sweep design (Phase 2 evaluation)

`eval/run_eval.py` sweeps **0.0, 0.3, 0.7, 1.0**:
- 0.0 — deterministic baseline, expected best-case structured-output conformance.
- 0.3 — typical factual-QA setting.
- 0.7 — typical chat-UI default; tests whether it meaningfully degrades citation faithfulness.
- 1.0 — stress test for the retry/validation safety net.

**Resumable by construction**: before each (model, question, temperature) combination, the harness skips it if a completed row for that exact key already exists in `results.jsonl`. This is required because the agentic flow can multiply call counts well past a single-shot design's estimate, and full runs are expected to take multiple hours, potentially overnight — an interrupted run must be safely re-invokable without redoing completed work.

## 11. Hybrid retrieval architecture and the retrieval-quality gate (Stage 2.5)

### Why: retrieval was the one untested variable

Every eval run through Stage 2 (Section 8/PROGRESS.md) showed the same
signature — `citation_existence_ok` and `grounding_ok` at 100%, but heavy
over-abstention on genuinely answerable questions — across two different
model families and one prompt rewrite. That rules out "the model
hallucinates" and weakens "this model specifically is bad at this," leaving
retrieval itself (recall, or how retrieved chunks are *presented*) as the
remaining untested variable. Dense-only cosine similarity also has no way
to express "I'm not confident in this match" — every query gets 5 scored
chunks back with no calibrated notion of good vs. mediocre. This section
adds a hybrid retrieval stage with a real confidence signal, and a gate
that can act on that signal independent of the LLM's own self-reported
`has_sufficient_context`.

**Calibration caveat, stated plainly**: the current sample corpus is 5
documents = 5 total chunks. At `top_k=5`, retrieval already returns every
chunk in the corpus on every query, so precision@5/recall@5 read ~1.0 for
both dense-only and hybrid retrieval right now — there's nothing to filter
out of 5 chunks. The meaningful signal at this scale is **rank order**
(MRR, nDCG@1 — does the right chunk land at position 1?) and the
reranker's **absolute confidence score**, especially on the 4 deliberately
unanswerable golden questions. Precision/recall become genuinely
discriminating the moment a larger, real document set is ingested — this
is a known, temporary limitation of the sample corpus, not of the pipeline.

### Scope and reuse boundary (extends section 2)

`agent/tools.py`, `ingest/`, and `../data/faiss_index/` remain reused
**read-only, unmodified** — this stage never rebuilds or writes to the
parent's index. All new state (the lexical index, its build metadata)
lives under `rag-structured-eval/data/`, gitignored, owned entirely by this
project.

### Lexical retrieval — SQLite FTS5, not `rank_bm25`

A new, separate SQLite database (`data/lexical_index.db`) is built by
reading `../data/faiss_index/metadata.db`'s `chunks` table read-only and
mirroring it into an FTS5 virtual table (`scripts/build_lexical_index.py`,
always a full rebuild — cheap, no embedding calls). FTS5 was chosen over
adding a `rank_bm25` pip dependency because it fits the project's existing
SQLite-sidecar pattern and gets BM25 ranking for free from SQLite itself
(confirmed available: `sqlite3.sqlite_version` 3.50.4).

Two things handled explicitly, not left implicit: SQLite's `bm25()`
returns **lower = more relevant** (opposite of FAISS cosine), so
`retrieval/lexical_index.py` stores `lexical_score = -bm25(...)` — every
score in the `retrieval/` package follows one "higher = more relevant"
convention. And natural-language questions contain FTS5 query-syntax
characters (`?`, `'`, `:`) that raise on `MATCH` if passed raw — queries
are tokenized, stripped to alphanumeric characters, and joined with `OR`
before querying.

### Fusion — Reciprocal Rank Fusion, k=60

`retrieval/fusion.py` combines dense and lexical rankings via RRF:
`score(chunk) = Σ 1/(k + rank)` summed over whichever ranked list(s)
contain that chunk. RRF uses only rank position, never raw score
magnitude — the right choice here because dense cosine similarity and BM25
live on incomparable scales; averaging or weighting the raw scores
directly would need calibration this project has no basis for yet. `k=60`
is the standard default from the original RRF paper and from
OpenSearch/Elasticsearch/Weaviate's hybrid search defaults — a
well-precedented starting point, not tuned specifically for this corpus.

### Reranking — a real cross-encoder, chosen deliberately over RRF alone

`retrieval/reranker.py` uses `sentence-transformers`'s
`cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, CPU-friendly) to score each
(query, candidate) pair jointly — a materially stronger relevance signal
than either dense or lexical ranking alone, since a cross-encoder sees
query and passage together in one forward pass instead of scoring them
independently. This was a deliberate, informed tradeoff: it adds
`sentence-transformers` + `torch` to requirements — a heavy install on a
machine with documented tight disk/RAM history (section 3) — accepted
because the whole point of this stage is a *real* confidence signal, not
just a cheaper fused-rank approximation. `cross-encoder/ms-marco-TinyBERT-L-2-v2`
(~4M params) is a documented, same-interface fallback if the MiniLM model
proves too heavy in practice.

The model's raw `predict()` output is an **unbounded logit** (it was
trained with a regression objective, not BCE) — a sigmoid is applied
explicitly in `reranker.py` to normalize into `[0, 1]` so the gate below
has a stable, bounded scale to threshold against. Verified empirically
against the installed `sentence-transformers` version, not assumed.

### The retrieval-quality gate

`retrieval/quality.py`'s `RetrievalQualityReport` mirrors
`eval/faithfulness.py`'s `FaithfulnessResult` pattern deliberately: every
underlying signal (`top1_rerank_score`, `score_gap`, `dense_lexical_agreement`,
plus hit/candidate counts) is always computed and logged, on every
`HybridRetriever.retrieve()` call, regardless of outcome. `passes_gate()`
is a single, deliberately simple v1 rule (`top1_rerank_score >= threshold`)
— the other signals are logged specifically so a smarter multi-signal rule
can be built later from real observed data, the same way
`FAITHFULNESS_THRESHOLD` started as an explicit, unvalidated placeholder.

`RETRIEVAL_GATE_THRESHOLD` was validated, not guessed, from
`eval/retrieval_eval.py`'s real score distribution on this corpus: all 10
answerable questions scored `top1_rerank_score` in `[0.969, 0.9994]`; 3 of
the 4 deliberately-unanswerable questions scored below 0.03. `0.5` sits
comfortably in that gap. **One unanswerable question (q013) scored 0.755 —
above any threshold that wouldn't also reject genuinely answerable
questions.** It's an "adjacent-topic trap": a real document exists on a
related subject (the TII project doc discusses ALM/IRRBB) but never states
the specific fact asked about (capital relief). This is an honest, expected
gate limitation, not a tuning failure to fix — a cross-encoder measures
topical relevance, not whether a passage contains the specific fact asked
about, so it structurally cannot distinguish "same subject" from "answers
the question." Re-derive this threshold with `eval/retrieval_eval.py` if
the corpus changes meaningfully.

**Gate behavior, by design**: hard-gate generation by default, but always
compute and log the score regardless. In `generate.py`'s
`run_agentic_structured()`, the gate check runs immediately after the
tool-calling phase, before the (expensive) structured-final LLM call. If
the last `retrieve()` call's quality report fails the threshold, generation
is skipped entirely and a synthesized, valid `StructuredAnswer`
(`has_sufficient_context=False`, no citations) is returned with
`error_kind = GenerationFailure.RETRIEVAL_QUALITY_GATE_BLOCKED` — a real
answer object, not `None`, so `eval/faithfulness.py`'s `score_answer()`
runs its existing `abstention_ok` check completely unmodified. If phase 1
never called `retrieve()` at all, the gate is skipped (documented, not
silently guessed at).

### LLM-free retrieval evaluation

`eval/retrieval_eval.py` computes precision@k/recall@k/MRR/nDCG@k against
`golden_qa.jsonl`'s `expected_citation_chunks` (over the `unanswerable:
false` rows only — those metrics are undefined against an empty expected
set) with **zero LLM calls**, running in seconds. For the 4 `unanswerable:
true` rows, there's nothing to measure recall against; instead their
`top1_rerank_score`/gate outcome is reported separately — the direct
retrieval-side complement to `abstention_ok`, and the fastest way to
sanity-check the gate threshold before spending any LLM inference time.

### Known approximation carried over from the dense-only design

`eval/run_eval.py` and `compare_models.py` both re-query retrieval
*after* generation completes, to get "the chunks that were retrieved" for
faithfulness scoring, rather than threading the actual `tool_results` used
during generation back out of `run_agentic_structured()`. This was already
an approximation with the dense-only retriever (assumes retrieval is
stable for the same query string) and remains a reasonable one with the
hybrid pipeline — cross-encoder inference at eval time is a deterministic
forward pass, so re-querying the same string reproduces the same ranking.
Not fixed structurally in this pass; worth revisiting if that assumption
ever breaks (e.g. a future non-deterministic reranker).

## 12. Retrieve-first generation and the explicit final prompt (2026-09-30)

**Supersedes the prompt-construction parts of section 7.** The schema, the validation rules and the separate structured-final call are unchanged; how passages reach that call is not.

### Why

The "over-abstention" finding of Stages 1-2 was largely a plumbing artifact (evidence in `PROGRESS.md`, session 2026-09-30). In the failing questions the model had been shown little or no document text:

- The shared `agent.tools.RETRIEVE_TOOL_SCHEMA` offers optional `doc_type` / `topic` filters, applied filter-then-search. On a 5-document corpus one wrong guess returns zero chunks.
- `phi4-mini` never emitted a tool call the harness could dispatch, so it never retrieved anything.
- The server runs at `OLLAMA_CONTEXT_LENGTH=1024`. `system_v1` plus the question is ~380 tokens, the five chunks are 184-402 tokens each, and the answer needs room too. Replaying the tool transcript into the final call cannot fit; Ollama drops what doesn't, silently.

### What `generate.py` does now

- **`run_retrieve_then_read()` — the default.** The harness calls `HybridRetriever.retrieve(question)` itself, applies the retrieval-quality gate (section 11), then makes one structured-final call. No tool call, so no dependence on a model's tool-calling reliability, and one LLM call per question instead of three or more. A gate-blocked question makes no LLM call.
- **`run_agentic_structured()` — kept as a comparison arm** (`--mode agentic`). The model-facing tool is now query-only (`retrieval.hybrid_retriever.RETRIEVE_TOOL_SCHEMA`); any other argument a model adds is ignored. Tool results are short previews so the tool rounds stay small; the full passages are held in code. The gate uses the best rerank score across every retrieve call the question made.
- **`build_final_messages()` — one final prompt for both paths.** Short reader prompt (`prompts/system_v3.md`), then a user message of `CONTEXT` (passages labelled `[source #chunk_index]`, plain text, no metadata JSON), `QUESTION`, and the instruction. Passages are added in rank order while they fit `num_ctx - FINAL_MAX_TOKENS - overhead`; the last one that doesn't fit is cut to the remaining budget, and the top passage is always included. Up to `MAX_CONTEXT_CHUNKS = 3`.
- **Retries rebuild, they don't append.** Each retry is the same base prompt plus one short note about the most recent rejection, re-budgeted. The old append-the-error approach grew the prompt until the passages were pushed out.
- **Citations are validated against the passages actually shown**, not everything retrieved. One added rule: `has_sufficient_context: true` with empty `citations` is rejected and retried.
- **`trace.jsonl` per run**: every retrieval / tool call with its raw arguments, the exact final user message with its estimated token count, and the raw final response. `metrics.jsonl` only ever had token counts, which is why the cause above went unseen for two stages.

### Limits

- `--num-ctx` (default 1024) must match the server's `OLLAMA_CONTEXT_LENGTH`; nothing checks it. Token counts are a `tiktoken` estimate with a fixed overhead allowance — compare `est_prompt_tokens` in `trace.jsonl` with `prompt_tokens` in `metrics.jsonl`.
- At 1024 the passage budget is roughly 500 tokens: the largest chunk (402) fits alone, two mid-sized ones fit together, and a cross-document question whose two passages don't both fit gets the second one truncated or dropped. Raising the server context to 2048 lifts this with no code change.
- In the agentic arm the tool rounds themselves are not budgeted; with `system_v1` (~350 tokens) several rounds of previews can still exceed 1024.

## 13. Known limitations / non-goals

- No CI — the "gate" is a local script the developer is expected to run and honor manually, not an enforced pipeline.
- Faithfulness scoring is a proxy (citation existence + embedding grounding + abstention correctness), not a full semantic-correctness judge — it will miss answers that are wrong in ways that still happen to cite a real, topically-similar chunk.
- Token counts may be approximate for some models (see §6).
- This project does not re-ingest or re-index documents; it depends entirely on the parent project's existing FAISS index staying in sync with what the golden set expects.
