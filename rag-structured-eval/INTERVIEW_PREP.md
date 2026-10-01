# System Design Interview Prep — RAG Evaluation Project

A study set built from this project's own history: running a structured,
agentic RAG pipeline against three local models under real memory
constraints, and evaluating them with a golden-question harness. Each
question is answered with the reasoning and the actual metrics behind it.
Where a question asks about production/business tradeoffs that this personal
project never had to answer for real (e.g. "who is the user"), the answer
gives you a reasoning framework and says so explicitly — treat those as
practice for articulating judgment, not as recited facts.

Source of truth for every number below: `PROGRESS.md` and `DESIGN.md` in this
directory, plus this session's raw run outputs under `eval/runs/`.

---

## 1. Problem Framing & Requirements

### 1. What's the actual failure mode you're optimizing against — hallucination, or under-answering?

**Answer:** Under-answering, decisively, based on the data. Across every run
in this project — Stage 1 (llama3.2, 14 questions), Stage 2 v2 (llama3.2,
12 comparable questions), and Stage 2 diagnostic #2 (phi4-mini, 14
questions) — `citation_existence_ok` and `grounding_ok` were **100% (zero
failures) in every single run**. The model never once cited a chunk that
wasn't retrieved, and never once made a claim unsupported by a retrieved
chunk. The *entire* faithfulness gate failure came from one dimension:
`abstention_ok` — the model saying "insufficient context" on genuinely
answerable questions (Stage 1: 8/10 answerable questions wrongly abstained;
phi4-mini: 10/10). A system tuned only against hallucination would look
"safe" by every naive metric while being useless in practice, because it
refuses to answer almost everything. The lesson: measure the two failure
modes separately, because optimizing for one can silently make the other
worse (this is close to a precision/recall tradeoff — a model that never
answers has perfect "precision" on the answers it does give, and zero
recall).

### 2. Who is the user, and what does a wrong answer cost versus a refusal?

**Answer (framework, not an established fact about this specific project):**
The golden set's content (SR 11-7 guidance, NMD behavioral models, stress
test outcomes, model risk management findings) points at a model-risk /
regulatory-compliance use case, where a wrong answer with a fabricated
citation could mislead someone relying on it for an actual filing or audit
response — a high blast-radius error. A refusal, by contrast, just means the
user goes back to reading the source document manually — annoying, costly in
time, but not misleading. In that specific domain, the asymmetry argues for
exactly the trade this project is currently making (never hallucinate, even
at the cost of over-abstaining) — **but** over-abstaining at the rate
observed (100% on answerable questions for phi4-mini) means the system adds
zero value over just reading the documents yourself. The real target isn't
"abstain less at any hallucination cost" — it's "find the point where
grounding stays at 100% and abstention drops to something like the 4
genuinely-unanswerable questions only." The project hasn't found that point
yet; that's the whole open question in Stage 3.

### 3. Why build a golden-set + gate harness instead of eyeballing outputs?

**Answer:** Because the failure mode here (over-abstention) is invisible to
casual spot-checking — a model that says "I don't have enough information"
sounds *appropriately cautious* on a single manual read, not obviously wrong.
It took a structured comparison against a labeled answerable/unanswerable
set (`golden_qa.jsonl`, 10 answerable + 4 unanswerable) to reveal that the
caution was miscalibrated: correct on 4/4 genuinely unanswerable questions,
but wrong on 8–10 out of 10 genuinely answerable ones. Without the labeled
set, "the model is being cautious" and "the model is broken" look identical
from the outside. The `FAITHFULNESS_THRESHOLD = 0.80` gate in `eval/gate.py`
turns that comparison into a binary go/no-go so the pipeline can't silently
ship past a broken configuration — but it's explicitly flagged in the code
as a **placeholder never validated against a real stakeholder's risk
tolerance**. Justifying a real number means answering question 2 first: what
fraction of wrong-or-missing answers is tolerable given the cost of each
error type in the actual deployment context?

---

## 2. RAG & Retrieval Architecture

### 4. Where could relevant context get lost between retrieval and generation?

**Answer:** Walking the path — query → `embedder.embed([query])` → FAISS
`index.search()` → SQLite `SELECT ... WHERE id = ?` per hit → assembled into
a tool-result JSON blob → appended to the conversation → read by the model —
there are three distinct places to lose signal:
1. **Embedding mismatch**: if the query embedding doesn't land near the
   right chunk vectors (embedding model quality, or a query phrased very
   differently from the source text), FAISS returns the wrong top-k before
   the LLM even sees anything. This project didn't find evidence of this
   (citations that *were* returned were always grounded), but it wasn't
   directly isolated from the other failure modes.
2. **top_k cutoff**: relevant content that ranks 6th when `top_k=5` is
   silently dropped, indistinguishable from "not in the corpus at all" from
   the model's point of view.
3. **Presentation to the model**: this is the project's current leading
   hypothesis. The retrieved chunks arrive as a *tool result* — a JSON blob
   returned from a function call — rather than as an explicit "here is your
   context" block. It's plausible the model's own training biases it to
   treat tool-call results with more skepticism (the way models are often
   trained to be cautious about function outputs, e.g. web search results)
   than it would a plainly labeled context section, causing it to
   under-trust content that is, in fact, sufficient and relevant.

### 5. Why would presentation format cause over-abstention? How would you isolate it?

**Answer:** The mechanism (hypothesized, not yet proven) is that models
trained on tool-use data learn a pattern like "treat retrieved/tool content
as evidence to be verified, not as ground truth to answer from directly,"
whereas a prompt that says `CONTEXT:\n<chunks>\n\nQUESTION:` primes a
different, more directly-generative mode. This project has ruled out two
alternative explanations already — prompt wording (v2 rewrite made things
*worse*: 25% vs. v1's 33% on a matched 12-question subset) and model choice
(phi4-mini fails identically to llama3.2, actually slightly worse: 0/10 vs.
2/10 answerable questions correct) — leaving presentation format and
temperature as the remaining live variables.
**To isolate presentation as the variable:** hold model, temperature,
prompt, and retrieval all fixed, and change only how the same retrieved
chunks are serialized into the final context — e.g., replace the tool-result
JSON with a plain `## Retrieved Context\n[1] ...\n[2] ...` block inserted
directly into the user or system message, and re-run against the same 10
answerable questions. If abstention drops substantially with retrieval
mechanically unchanged, that isolates presentation cleanly. This is
explicitly the next-recommended-step in `PROGRESS.md`.

### 6. How would you determine whether `top_k=5` is right?

**Answer:** Two failure signatures point in opposite directions, and both
are measurable from data this harness already collects. **Too few** shows up
as `abstention_ok: false` on questions whose `expected_citation_chunks`
exist in the corpus but weren't retrieved at all — you'd check this by
logging what was *actually* retrieved per question and diffing against the
golden set's expected chunks; if the right chunk simply never appears in the
top-5, that's a retrieval-recall problem, not a generation problem. **Too
many** shows up as `grounding_ok` failures or lower answer quality on
questions that *did* get correctly-abstained-or-answered before, because
irrelevant chunks dilute the prompt and can distract the model or push
useful content out of a tight context window (relevant here: phi4-mini's
1024-token context was already ~50–85% consumed by prompt tokens alone in
several rounds — see question 10). In this project specifically, the
evidence so far (100% grounding, 100% citation existence, but heavy
abstention) argues *against* "top_k too high" and is at least consistent
with "top_k too low or chunks not recognized as sufficient" — but this
hasn't been isolated from the presentation hypothesis yet.

### 7. Why split retrieval and final-answer generation into two calls?

**Answer:** Because `response_format` (schema-constrained JSON output) and
`tools` (function-calling) don't reliably compose in a single call across
providers/models — asking a model to both decide whether to call a tool
*and* emit schema-valid JSON in the same turn is a much less-tested code
path than a plain multi-turn tool-use loop followed by a dedicated
structured-output call. This was explicitly flagged as the single biggest
technical risk in `DESIGN.md` before Stage 1 started, with a named fallback
(call Ollama's native `/api/chat` with a top-level `format` field if the
OpenAI-compatible `response_format` didn't work after tool calls). It
**did** work — confirmed in Stage 1's manual smoke test — so the fallback
was never needed. **What it buys you:** a clean separation of concerns
(retrieval decision-making vs. answer formatting) and a schema-enforced
final answer you can parse without fragile string parsing. **What it
costs:** at least one extra full round-trip per question — under
`OLLAMA_MAX_LOADED_MODELS=1`, that's an extra potential model reload, which
measurably dominates latency (see question 9).

---

## 3. Model Selection Under Resource Constraints

### 8. How does a 7.7GB RAM ceiling dictate the model shortlist?

**Answer:** Directly and non-negotiably. Stage 0's smoke test showed the
actual cutoff empirically: **every Q4-quantized ~3–4B model tested succeeded
at 100% with 16–25s mean latency** (`llama3.2:3b-instruct-q4_K_M`:
16.1s/100%; `llama3.2:3b-instruct-q5_K_M`: 25.2s/100%;
`phi4-mini:3.8b-q4_K_M`: 23.4s/100%), while **every model at or above
~4.5–5GB of weights failed outright** (`phi4-mini:3.8b-q8_0`: 0/2, 620s
timeouts; `qwen2.5:7b-instruct`: 0/2, 621s timeouts; both Mistral 7B variants
dropped from the project entirely after repeated OOM kills even *after* the
context-length fix). The rule of thumb that emerged: **weights need to fit
in well under half of total RAM**, because KV cache, the embedding model
(also memory-resident, ~270MB for `nomic-embed-text`), the OS, and this
project's own tooling all compete for the same pool. "Biggest model I can
theoretically fit" stops being the right frame the moment you account for
**concurrent** memory needs — a model whose weights alone are 60–70% of
total RAM will OOM under real usage even if `ollama list` shows it loaded
successfully once.

### 9. Quantify the cost of `OLLAMA_MAX_LOADED_MODELS=1` forcing reloads.

**Answer:** This is directly measurable from this session's phi4-mini run.
A cold model load took **32.19 seconds** in one observed case (chat model,
`llama_server.go: "llama-server started in 32.19 seconds"`), and the
pipeline alternates between the embedding model and the chat model on
**every retrieval call within a single question** — meaning a question with
2 tool-calling rounds plus a final structured call could trigger 2–4 full
model swaps, each costing 30+ seconds of pure reload overhead before any
tokens are even generated. Concretely, looking at the phi4-mini full run's
per-question totals: mean latency was **50.4s per question** at a measured
generation throughput of only **~10.0 tokens/sec** — for questions that
generated on the order of 100–300 completion tokens, that's roughly 10–30s
of *actual* generation time, meaning **well over half the wall-clock time
per question was reload/orchestration overhead, not model compute.**
**Redesign options:** (a) run two Ollama instances on different ports, each
pinned to one model, so neither gets evicted — trades RAM headroom (you now
need both models resident simultaneously, which may not fit) for eliminated
reload cost; (b) batch all retrieval calls before any chat-model call within
a question, rather than interleaving them round-by-round, to minimize the
number of swaps; (c) accept the cost as a fixed tax of the memory-constrained
environment and optimize elsewhere.

### 10. Why does a 1024-token context matter more for phi4-mini than qwen2.5?

**Answer:** Because context pressure is about *fraction consumed before
generation starts*, not absolute size, and phi4-mini's actual usage in this
project ran right up against that ceiling. Observed server logs during the
phi4-mini run showed prompt token counts of **534, 587, 636, and — in one
later round — 870 out of the 1024-token slot**, meaning some rounds had as
little as ~150 tokens of headroom left for the model's own response before
`--context-shift` would start evicting earlier tokens. `qwen2.5:7b` was
never actually tested at the 1024-token cap in a completed run (it timed out
in Stage 0 before reaching this pipeline), so this specific pressure point
is really a `phi4-mini`-and-`llama3.2`-class-model observation, not a
`qwen2.5`-specific comparison — but the general point holds: a smaller
context window combined with a verbose tool schema + several retrieved
chunks can leave a model very little room to reason before its input is
already truncated or shifted, which is a plausible *additional* contributor
to over-abstention (a model given a half-truncated context has a legitimate
reason to say "insufficient information") separate from the presentation
hypothesis in question 5.

### 11. Why is "phi4-mini ≈ llama3.2, qwen2.5 unusable" not yet a fair conclusion?

**Answer:** Because `qwen2.5:7b-instruct` never completed a single question
through this project's actual RAG pipeline — it only ran through Stage 0's
raw chat-completion smoke test, where it hit **0/2 success, ~621s
timeouts**, and was excluded from the pipeline entirely without ever seeing
the structured-output-plus-tool-calling harness, the golden question set, or
a faithfulness score. "Unusable" here means "too slow/unreliable to even
attempt," which is a real and valid finding for a resource-constrained
deployment — but it is a *different kind* of finding from "tested and scored
28.6%." Comparing three models fairly requires all three to run the
*identical* harness; right now you have two comparable data points
(llama3.2: 42.9%, phi4-mini: 28.6%) and one non-comparable exclusion
(qwen2.5: never reached the eval). The honest framing is "two Q4 ~3–4B
models both fail the same way; a 7B model wasn't evaluable at all on this
hardware" — not "we compared three models and two were fine."

---

## 4. Structured Output & Agentic Reliability

### 12. Why was `response_format`-after-tool-calling the biggest a priori risk?

**Answer:** Because it's a much less common code path than either piece in
isolation. Plain schema-constrained JSON output (`response_format`) is
well-tested by providers; plain multi-round tool-calling is well-tested;
but *chaining* them — several rounds of `tools`/`tool_choice="auto"`
messages, then switching the same conversation to `response_format` for a
final call — exercises message-history handling and schema-enforcement
together in a way that's much less commonly validated, especially against a
local Ollama server rather than a first-party hosted API. Retrieval working
was a much lower-risk bet (FAISS + SQLite is deterministic, well-understood
infrastructure); "will the model be smart enough" was treated as a
*measurement* question (that's what the golden set and gate are for), not an
engineering risk to de-risk before writing code. The **fallback**, named
explicitly in `DESIGN.md` before Stage 1 started, was to bypass the
OpenAI-compatible endpoint entirely and call Ollama's native `/api/chat`
with a top-level `"format"` field instead of `response_format` — a lower-level,
more direct request that provider compatibility layers can't get in the way
of. It was never needed: Stage 1's manual smoke test confirmed the chained
approach worked (2 tool rounds + 1 structured final call, ~83s total, no
schema violations).

### 13. Why can a schema-enforced final answer still contain a malformed tool-call argument?

**Answer:** Because these are two different enforcement mechanisms operating
at two different points in the stack. `response_format` with a JSON schema
is validated server-side (or by the OpenAI-compatible layer) against the
*final* output before it's returned — the API contract guarantees the shape.
Tool-call *arguments*, by contrast, are just a string the model was trained
to produce that *looks like* JSON matching a function signature — there is
no schema validator sitting between "model emits text" and "your code reads
`args["top_k"]`." The model decided `top_k` should be `"5"` (a string) not
`5` (an int) because nothing forced it to respect the type; the tool
schema is a *hint* to the model, not a runtime contract. **The general
principle:** anything the model produces that flows into *your own code* as
a function argument is untyped, external input, exactly like user input from
a web form — it must be validated/coerced at the boundary
(`agent/tools.py`'s `_retrieve()` now does `int(top_k)` with a clear error on
failure), never trusted as already being the right type just because a
schema was declared. Only output that flows through a provider's own
schema-validated response path (`response_format`) can be trusted structurally
— and even then, only the *shape* is guaranteed, not that the content is
correct (that's what `faithfulness_pass` checks separately).

### 14. What does a rising retry rate tell you that pass/fail doesn't?

**Answer:** `attempts_used` in this project's results (mostly 2, occasionally
3) reflects retries triggered by the harness's own error-isolation logic —
for example, when a model cites a chunk that wasn't actually in the
retrieved set, `generate.py` feeds back "these citations were not among the
chunks actually retrieved... only cite pairs that appeared above" and
retries. A rising retry rate tells you the model is producing *recoverable*
errors (wrong format, unlisted citation) rather than *fundamental* ones
(wrong answer despite a clean structural response) — it's a leading
indicator of instability that a pure pass/fail number smooths over entirely.
Two runs with identical 28.6% faithfulness could look completely different
operationally: one where every question needed exactly 2 calls versus one
where half needed 3 retries — the second is burning 50% more compute and
latency per question for the *same* final accuracy, which matters directly
for the cost questions in section 7.

---

## 5. Evaluation Methodology & Statistical Rigor

### 15. What can't you conclude from a 14-question eval?

**Answer:** You cannot treat small percentage differences as a meaningful
model ranking. With `n=14` (or `n=12` for the v1/v2 comparison, or `n=10` for
the answerable-only subset), a single question flipping pass/fail moves the
headline number by roughly 7–10 percentage points. **llama3.2's 42.9%
(6/14)** versus **phi4-mini's 28.6% (4/14)** is a 2-question difference —
consistent with a real gap (both are clean, deterministic temp-0.0 runs
against the identical question set, and the *pattern* is consistent: both
models got 0–2 out of 10 answerable questions right and both got 4/4
unanswerable right), but nowhere near enough to say "phi4-mini is 33% worse
than llama3.2" as a general claim — it's enough to say "in this specific
14-question sample, phi4-mini did somewhat worse, in the same failure mode."
A rigorous confidence interval on a Bernoulli proportion at n=14 is wide
(roughly ±20–25 percentage points at typical confidence levels) — the real
value of this golden set at its current size is **detecting a consistent
qualitative failure pattern** (over-abstention, present in literally every
run so far), not **precisely ranking** two models that fail the same way.

### 16. What do you lose by collapsing three checks into one `faithfulness_pass` boolean?

**Answer:** You lose the ability to tell *which* failure mode is driving the
number without going back to the raw per-question rows — which is exactly
why this project kept all three (`citation_existence_ok`, `grounding_ok`,
`abstention_ok`) as separate fields in every result row even though the gate
only checks the combined boolean. That decision paid off directly: it's the
reason Stage 1 could report "citation existence and grounding are 14/14
(100%) — the *only* failure mode is abstention" instead of just "STOP, 43%,"
which would have left root-causing to guesswork. **When to gate on them
separately:** whenever the three checks have different costs (see question
2 — a hallucinated citation is a much worse failure than an unnecessary
abstention in a compliance context), you'd want independent thresholds —
e.g., "grounding must be ≥99%, abstention-accuracy only needs to clear 60%"
— rather than one blended number that could hide a small amount of
hallucination behind a good abstention rate, or vice versa.

### 17. What's the risk of a 4-question probe subset, and how do you pick probe questions well?

**Answer:** The risk is that a probe is a *sample*, and if it happens to
contain only "easy" or only "hard" questions relative to the full set, the
directional signal it gives can be wrong in either direction — a probe that
looks great can mask a full-set failure, and a probe that looks bad could
be an unlucky draw of the hardest questions. In this project, the probe
(`golden_qa_probe.jsonl`, q001–q004) was chosen as simply "the first 4
answerable questions" — a real limitation the exercise should acknowledge,
even though in this case the probe result (4/4 over-abstention) turned out
to fully match the eventual full-run result (10/10 over-abstention on
answerable questions), so it happened to generalize well. **A better
selection method:** stratify the probe across the dimensions you already
know matter — e.g., include at least one single-doc factual-recall question,
one multi-document cross-reference question (like q003, which spans two
source documents), and one from each topic area — so a probe pass/fail
reflects the full set's *diversity* of difficulty, not just its first four
rows in file order.

### 18. Why measure `expected_citation_recall` if existence/grounding already pass?

**Answer:** Because "every citation you gave is real and grounded" and "you
gave *all* the citations you should have" are independent properties. A
model could cite exactly one relevant chunk (passing existence and
grounding perfectly) while ignoring two other equally-relevant chunks the
question's `expected_citation_chunks` says it should have used — producing
a technically-correct-but-incomplete answer. `expected_citation_recall`
catches under-citation that existence/grounding checks structurally cannot,
because those two checks only ever look at citations the model *did*
produce, never at what it left out. In this project's data,
`expected_citation_recall: 0.0` appears on every failed (over-abstained)
answerable question — which makes sense mechanically (an abstained answer
has zero citations, so recall against any non-empty expected set is
necessarily 0), but the field would become genuinely informative the moment
abstention rates come down and you start getting non-empty, non-abstained
answers to actually measure completeness on.

### 19. Finish the temperature sweep, or abandon it early? Which did you do, and was it right?

**Answer:** This project actually did both, in sequence, and the reasoning
changed as new information arrived. Diagnostic #1 (llama3.2 at temperature
0.3) was **abandoned at 2/14 questions** — but the actual reason at the time
was a *mistaken pace estimate* (a real-time-tracking confusion during the
session made 40–50s/question look like 15–20 min/question), not a
considered cost/accuracy tradeoff; both sampled points showed no improvement
(`abstention_ok: false`), which is weak evidence but not a disproof. The
**better-reasoned decision** came at diagnostic #2: rather than re-running
the abandoned temperature sweep once the real pace was known to be cheap
(~10 minutes for all 14 questions), the project ran a *different* model at
the *same* temperature (0.0) instead, because that answered a higher-value
question — "is this llama3.2-specific?" — for the same cost as finishing the
temperature sweep would have. **The general principle:** early-stopping a
sweep is justified when (a) the cost of continuing is known and non-trivial,
and (b) the early samples are consistent with "no effect" *and* you have a
cheaper or higher-information alternative experiment available. Abandoning
based on a wrong cost estimate (what actually happened here) isn't a good
process even when, by luck, the conclusion it led to (test something else
instead) turned out to still be a reasonable one.

---

## 6. Root-Cause Debugging & Observability

### 20. Why was "the machine is degrading" the wrong diagnosis, and what would have shortened it?

**Answer:** It was an easy misdiagnosis because the *symptom* — high
`llama-server.exe` memory use, low free system RAM, calls taking far longer
than Stage 0/1's benchmarks — is genuinely consistent with real hardware
memory pressure, and this machine *does* have a real, previously-documented
7.7GB ceiling. The actual causes turned out to be two specific, fixable
bugs: (1) an **orphaned `llama-server.exe` process** from an earlier
misconfigured (`-c 4096` instead of `1024`) server that survived multiple
rounds of `Stop-Process -Name '*ollama*'`, because that filter matches the
`ollama.exe` parent process name but not its distinct `llama-server.exe`
child — the zombie sat there consuming several GB indefinitely; (2) a
**missing `max_tokens` cap** that let one generation run away to **4243
completion tokens over 463.5 seconds** (measured directly in
`metrics.jsonl`) before being manually killed. **The signal that would have
shortened the investigation:** comparing accumulated CPU-seconds on the
`llama-server.exe` process (visible via `Get-Process -Id <pid> | Select
CPU`) against wall-clock elapsed since the last observation. A process
genuinely computing shows CPU time climbing roughly in proportion to
elapsed time (confirmed healthy in this session — e.g., CPU climbing
steadily from 39.7s to 210s over successive checks on a normal question); a
process that's actually just sitting idle behind a hung client shows flat
CPU despite elapsed time passing. That single ratio — the same
compute-bound-vs-idle-wait distinction used everywhere in systems debugging
— separates "genuinely slow" from "actually stuck" without needing to touch
Ollama internals at all.

### 21. Why didn't a 900-second timeout catch the runaway generation?

**Answer:** Because it's implemented as a *streaming* read (`stream=True` in
`metrics.py`'s `InferenceClient.chat()`), and the 900-second value is passed
as the client's general request timeout, which for a streaming response
typically governs the gap between receiving *chunks*, not the total
duration of the whole response. As long as the server keeps emitting a new
token every few seconds — which it was, just very slowly, at the observed
~9.6–10 tokens/sec throughput — the connection never goes idle long enough
to trip a per-chunk/read timeout, and there was no separate **total request
duration** cap at all. This is the general and important distinction: a
**read/idle timeout** protects you against a connection that stalls
completely; a **total timeout** protects you against a connection that
*never stops*, however slowly. A model generating one token every 100ms
forever would sail through almost any read timeout while still eventually
exhausting memory, disk, or your patience. The actual fix applied here
wasn't a timeout at all — it was a `max_tokens=300` cap on the completion
request itself, which is the correct fix for this specific failure mode
(bounding the *amount of work the server is allowed to do*, rather than
trying to detect after the fact that it's doing too much).

### 22. What's the lesson about attributing a symptom to a broad cause vs. a narrow one?

**Answer:** The original Stage 0 finding — "`OLLAMA_CONTEXT_LENGTH` and
`OLLAMA_MODELS` don't work as environment variables on this machine" — was a
broad, mechanism-level claim (env vars are broken) built from a narrower,
confounded observation (a specific launch method, at a specific time, while
a competing process was also running and re-asserting its own settings from
a SQLite-backed config). Re-tested this session with the confound actually
removed — every `*ollama*`-named process stopped *first*, then the server
started fresh with the env var set directly — the env var worked exactly as
documented upstream (`n_ctx = 1024` confirmed in the server log). **The
general lesson:** a broad causal claim ("X doesn't work") is only as strong
as the isolation of the experiment that produced it. If you didn't
control for every other process/setting that could plausibly also be
writing to the same piece of state, the honest claim is the narrow one —
"X didn't work *in that specific launch sequence*" — not the broad one.
Broad claims are tempting because they're simpler to remember and pass on
(exactly why this one got written into `DESIGN.md` and believed for an
entire session), but they actively mislead the next debugging session
into skipping the one check (are there other processes touching this?)
that would have found the real, narrower cause immediately.

---

## 7. Cost, Latency, and Accuracy Tradeoffs

### 23. What's the actual cost unit you're optimizing?

**Answer:** In this project, the *stated* cost (local Ollama, no per-token
billing) is zero dollars per query — but the real cost paid was
overwhelmingly **engineering time debugging infrastructure**, not compute.
A rough accounting from this project's own history: multiple full sessions
spent on OOM triage, a settings-database reverse-engineering exercise, a
zombie-process investigation, and a runaway-generation bug — versus a
comparatively small amount of time actually iterating on the thing being
evaluated (prompts, retrieval, models). If you priced engineer-hours at even
a modest rate, "free" local inference has already cost far more than
plausibly years of a hosted API's per-token bill at this project's query
volume would have. **The general principle:** "cost" for a resource
-constrained local setup should be modeled as *(compute is free) + (latency
has a real cost to the user) + (engineering time to keep it running has a
real, often underestimated, cost)* — and the last term dominates whenever
the infrastructure is this fragile.

### 24. At what accuracy floor does "cheap but wrong" stop being acceptable — and is under-answering a cost or accuracy problem?

**Answer:** Given that hallucination/grounding failures were **zero across
every run**, and the entire accuracy shortfall is over-abstention, this is
better framed as an **availability/utility problem dressed up as an
accuracy metric**. The system is accurate whenever it answers — it just
declines to answer 100% of the time on the hardest configuration tested.
That reframing matters directly for "which lever to pull": if this were a
true accuracy problem (the model answers confidently but wrong), you'd
reach for a bigger/better model, more retrieval, or fine-tuning. Because
it's actually an availability problem with a *known, single-dimension*
root cause (`abstention_ok`), the fix search should stay narrowly scoped to
"why does the model think it doesn't have enough information" — which is
exactly why the project's next step is testing context presentation
(hypothesis c) rather than jumping to a bigger, more expensive model.

### 25. Given a fixed compute budget, where does the next dollar go?

**Answer:** Not on a bigger model — two different model families
(`llama3.2` and `phi4-mini`) already fail identically (42.9% and 28.6%,
same failure mode, same 100% grounding), which is evidence *against*
"model capability" being the bottleneck. Not on prompt engineering alone —
the v2 rewrite, explicitly designed to reduce hedging, made things *worse*
(25% vs. 33% on the matched subset), which is evidence against "just tell it
to be less cautious" as a fix. That leaves **retrieval/context presentation**
as the highest-expected-value place to spend the next unit of effort, and
it's also the *cheapest* to test: it requires no new model pulls, no GPU, no
new infrastructure — just a change to how `tools.py`'s retrieved chunks are
serialized into the prompt, followed by a re-run of the existing 14-question
harness (now confirmed to take only ~10 minutes end to end). The
cost/information ratio strongly favors testing hypothesis (c) before
spending any budget on model upgrades or infra scaling.

### 26. When would you move off local Ollama to a cloud API?

**Answer:** The instant the actual value of getting the RAG system *working
correctly* exceeds the cost of a hosted API call, given how much of this
project's wall-clock time and reliability budget has gone into a hardware
ceiling that a hosted API simply doesn't have — no `OLLAMA_MAX_LOADED_MODELS`
juggling, no context-length settings-database archaeology, no zombie-process
memory leaks, no 620-second timeouts on a 7B model. **What you gain:** speed
of iteration (the entire model-swap and orphaned-process debugging saga in
this session would not exist), and access to genuinely larger/more capable
models than a 7.7GB machine can run at all — directly relevant since it's
still an open question whether "bigger model" would fix over-abstention
(this project could only rule out *two ~3–4B models*, not the capability
axis in general). **What you lose:** the $0 marginal cost, data
locality/privacy for the presumably sensitive model-risk documents in the
corpus, and control over exact model/quantization/version — a hosted
model can change behavior under you without notice. The related memory note
from a sibling project (Personal RAG Assistant, ~1-in-9 local chat success
rate) suggests this tradeoff has already been hit once before in this
broader body of work and was left as an open decision, not resolved.

### 27. How do you build a cost/accuracy curve instead of a single number?

**Answer:** Run the *same* 14-question golden set, held fixed, across a
range of models spanning a cost axis — e.g., a free local Q4 model, a free
local larger-quantization model, and a paid cloud API model — and plot
**faithfulness gate pass rate (y-axis)** against **cost per 14-question run
(x-axis, in $, using $0 for local + an amortized engineering-time estimate,
and actual token-billed cost for the API)**. A secondary useful axis is
**mean latency per question**, since a stakeholder deciding on
productionization cares about user-facing wait time as much as raw accuracy.
This project already has the harness to produce two of the three points on
that curve for free (`compare_models.py`, `eval/run_eval.py` against
`golden_qa.jsonl`) — the missing piece is simply plugging in a cloud
provider and running the identical eval, which the existing `InferenceClient`
abstraction (OpenAI-compatible client) is already positioned to support with
a different `base_url`/API key.

### 28. Is the golden set itself cost-relevant — when do you upgrade from a small probe to a bigger, trustworthy eval?

**Answer:** Yes — the smaller the eval, the cheaper each iteration but the
noisier each result (see question 15's confidence-interval point). The
right moment to invest in a larger, more statistically trustworthy golden
set is **when you're about to make a decision you can't easily reverse** —
e.g., picking a model or prompt configuration to ship, or reporting a
specific accuracy number to a stakeholder who will hold you to it. Up to
that point, a small, cheap, fast-iterating set (this project's 14 questions,
or the even smaller 4-question probe) is the *correct* tool, because its job
is to cheaply reject bad ideas quickly (both the v2 prompt and the phi4-mini
swap were correctly identified as "no better" using the small set, at low
cost) — not to produce a publishable accuracy figure. The project hasn't yet
reached the "about to ship" decision point, so continuing to iterate cheaply
on the 14-question set is still the right call; expanding it would be
premature optimization of eval rigor before the underlying approach
(hypothesis c) has even been tried.

---

## 8. Productionization & Next Steps

### 29. How do you validate the context-presentation fix without re-running the whole comparison?

**Answer:** Hold every other variable fixed at its known-working
configuration — model (`llama3.2:3b-instruct-q4_K_M`, since it's the faster,
already-characterized Stage 0 winner), prompt (`system_v1`, the
already-tested baseline, not v2 which is confirmed worse), temperature
(0.0, matching every prior comparable run) — and change *only* how
retrieved chunks are serialized into the context before the final
generation call. Re-run against the same 10 answerable questions (or even
just the same 4-question probe subset first, given the probe already
matched the full-run pattern once in this project). If abstention drops
substantially with literally nothing else changed, that's a clean,
minimal, single-variable result — and only *then* would it be worth
re-running the full 14-question set and the phi4-mini comparison to confirm
the fix generalizes across models, rather than re-running everything
upfront on a still-unproven hypothesis.

### 30. Hard gate vs. soft gate — what decides that?

**Answer:** A **hard gate** (block deployment/rollback automatically) makes
sense when a failure is asymmetric and severe in the way this project's
domain suggests (a hallucinated regulatory citation shipped to a user is a
much worse outcome than a delayed rollout) — and specifically for the
*grounding/citation-existence* dimensions, which have been 100% clean every
single run so far, a hard gate at anything less than 100% would be a
justified, low-false-positive tripwire precisely because a regression there
would be a genuinely new, unprecedented failure. A **soft gate** (log +
alert, ship anyway) is more appropriate for the abstention dimension
specifically, *given what's already known* — the model abstains too much,
everyone already knows it, and blocking every deploy on a metric that is
currently failing 71–100% of the time makes the gate meaningless as a
signal (it would fail on literally every build until hypothesis (c) is
resolved). The general rule: hard-gate on regressions in dimensions that
have historically been reliable; soft-gate (with visible tracking) on
dimensions that are known, open problems you're actively iterating against,
so the gate still gives you a trend line without blocking all forward
progress.

### 31. What production monitoring would this offline harness not give you?

**Answer:** The offline harness only ever sees the 14 questions you wrote
down in advance — it cannot tell you whether the *distribution* of real
user questions in production looks anything like `golden_qa.jsonl`, or
whether over-abstention concentrates on question types you never
anticipated. Three things worth adding that the offline eval structurally
can't provide: (1) **implicit abstention-rate tracking in prod** — logging
the rate of `has_sufficient_context: false` responses over live traffic,
independent of any golden set, as a leading indicator (a sudden spike would
flag a regression or a shift in the kind of questions users are actually
asking, well before anyone manually notices); (2) **user behavior signals**
— e.g., immediate rephrasing or repeated queries after an abstention, which
proxies for "the system was wrong to abstain" without needing ground-truth
labels at all; (3) **retrieval-quality drift** — tracking the distribution
of FAISS similarity scores for top-1 hits over time, since a slow drop
could indicate the corpus has grown stale relative to the kinds of
questions being asked, a failure mode this static, fixed 5-document eval
corpus can never surface.

---

## Quick-reference: what actually happened (for self-checking)

- **Stage 0**: Environment triage on a 7.7GB-RAM, no-GPU machine. Winner:
  `llama3.2:3b-instruct-q4_K_M` (100% success, 16.1s mean latency, 13.0s
  TTFT). Both Mistral 7B variants dropped after repeated OOM kills;
  `qwen2.5:7b-instruct` kept only as an expected-to-struggle comparison
  point (confirmed: 0/2 success, ~621s timeouts).
- **Stage 1**: Structured + agentic pipeline validated end to end.
  `response_format` after tool-calling confirmed reliable. Gate: **STOP,
  42.9% (6/14)**. `citation_existence_ok` and `grounding_ok`: 100% (14/14).
  The only failure: over-abstention on 8/10 answerable questions; 4/4
  unanswerable correctly abstained.
- **Stage 2, prompt v2**: Rewrite intended to reduce hedging instead scored
  **worse** — 3/12 (25%) vs. v1's 4/12 (33%) on a matched subset. Grounding
  remained 12/12.
- **Stage 2, diagnostic #1 (temperature 0.3, llama3.2)**: Abandoned at 2/14
  questions on a mistaken pace estimate (real pace: ~40–50s/question, not
  15–20 min). Both sampled questions still over-abstained.
- **Stage 2, diagnostic #2 (phi4-mini, full 14 questions, temp 0.0)**: Gate
  **STOP, 28.6% (4/14)**. 0/10 answerable passed; 4/4 unanswerable correctly
  abstained. 100% validity/success rate. Mean TTFT 26.4s, ~10.0 tokens/sec,
  ~50.4s mean latency/question. Ruled out "llama3.2-specific limitation."
- **Bugs found and fixed mid-investigation**: a settings/env-var confound
  (a background app racing a manually-started server over context length —
  resolved, env var confirmed to work when isolated); an orphaned
  `llama-server.exe` process invisible to name-based process kills,
  consuming several GB indefinitely; a missing `max_tokens` cap that let one
  generation run away to 4243 tokens over 463.5 seconds before being killed
  (fixed with `max_tokens=300`).
- **Open question going into Stage 3**: is over-abstention driven by
  temperature, or by how retrieved chunks are presented to the model?
  Current evidence (two models fail identically, a friendlier prompt made
  it worse) favors presentation format as the more likely lever.
