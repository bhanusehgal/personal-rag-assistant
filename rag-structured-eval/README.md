# rag-structured-eval

Structured, benchmarked generation layer for the "Personal RAG Assistant" project. Reuses the parent project's retrieval/embeddings (`../ingest/`, `../agent/tools.py`) unchanged; adds versioned prompts, schema-constrained + validated generation with retry, a rules-based faithfulness quality gate, per-call inference metrics (TTFT/tokens-per-sec/latency), and a quality-vs-speed comparison across Llama 3.2 3B, Phi-4-mini, Mistral 7B (GGUF Q4/Q5) against the existing `qwen2.5:7b-instruct` baseline.

Read `DESIGN.md` for the full technical design and rationale. Read `PROGRESS.md` for current build status — start there if you're picking this project back up after a pause.

## Setup (one-time)

1. Run `scripts/setup_ollama_models.ps1` — upgrades Ollama, moves the model store to `D:\ollama-models` (this machine's `C:` drive is nearly full), and pulls the 6 new model variants.
2. **Install the CPU-only torch wheel first** (hybrid retrieval's reranker needs `sentence-transformers`, which pulls in `torch` — on this GPU-less machine, installing torch before the main requirements file keeps pip from resolving a much larger CUDA build):
   ```
   .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
3. From the repo root (`Personal RAG Assistant/`): `.venv\Scripts\python.exe -m pip install -r rag-structured-eval\requirements.txt`
4. Verify the parent agent still works: `.venv\Scripts\python.exe -m agent.loop` and send one query.
5. Build the lexical (BM25) index used by hybrid retrieval: `cd rag-structured-eval` then `..\.venv\Scripts\python.exe -m scripts.build_lexical_index`. Re-run this any time the parent project's document index changes — there's no automatic sync between the two.

## Running each stage

`rag-structured-eval` has a hyphen in its name, so it can't be addressed as a Python package (`-m rag-structured-eval.x` is not valid syntax). Instead, `cd` into it first and invoke modules relative to that directory — each script inserts the repo root onto `sys.path` itself, so the shared parent `.venv` still resolves `ingest`/`agent` imports correctly.

```
cd rag-structured-eval
..\.venv\Scripts\python.exe -m pip install -r requirements.txt   # one-time, from repo root instead is equivalent
```

- **Stage 0 (smoke test)**: `..\.venv\Scripts\python.exe -m scripts.smoke_test`
- **Stage 1 (single-question structured pipeline)**: `..\.venv\Scripts\python.exe generate.py --question "..." --model <model> --prompt-id system_v1 --temperature 0.0`
- **Stage 2 (temperature sweep)**: `..\.venv\Scripts\python.exe -m eval.run_eval --model <model> --prompt-id system_v1 --temperatures 0.0,0.3,0.7,1.0 --run-id <name>`
- **Stage 2.5 (retrieval-only eval, no LLM calls, seconds not minutes)**: `..\.venv\Scripts\python.exe -m eval.retrieval_eval --mode both` — precision@k/recall@k/MRR/nDCG@k for dense-only vs. hybrid retrieval, plus a gate-threshold sanity check against the unanswerable golden questions. Run this before trusting any live LLM run against the hybrid pipeline. See `DESIGN.md` section 11 for the full design and an important calibration caveat about the current small sample corpus.
- **Quality gate**: `..\.venv\Scripts\python.exe -m eval.gate --run-id <name>`
- **Stage 3 (full model comparison)**: `..\.venv\Scripts\python.exe compare_models.py --prompt-id system_v1 --temperature <best> --run-id <name>` then `..\.venv\Scripts\python.exe analyze_results.py --run-id <name>`

`generate.py`, `eval.run_eval`, and `compare_models.py` all build a hybrid (dense + lexical + reranked) retriever automatically as of Stage 2.5 — no new flags needed for these existing commands, but they do require the lexical index to have been built first (setup step 5 above).

## Note on the parent project's `eval/`

The parent repo has its own empty `eval/.gitkeep`, reserved for a never-built "Phase 4." This project's `eval/` is a separate, unrelated folder inside `rag-structured-eval/` — no conflict, just don't confuse the two.
