"""Retrieval + structured-final answer generation — DESIGN.md sections 7 and 12.

Two ways to get passages, one way to answer:

  - run_retrieve_then_read (default): the harness runs the hybrid retriever on
    the user's question itself. No tool call, so nothing depends on the model
    emitting a well-formed one (phi4-mini never did — see PROGRESS.md).
  - run_agentic_structured (comparison arm): the model drives retrieval through
    a query-only `retrieve` tool for up to max_tool_rounds rounds. It sees short
    previews of what came back; the full passages are held here for the final call.

Both end in the same structured-final call. Its prompt is built from scratch by
build_final_messages() — short system prompt, the passages as a labelled CONTEXT
block, the question — and budgeted to fit num_ctx with room left for the answer.
The tool transcript is never replayed into it: on a 1024-token context Ollama
silently drops whatever doesn't fit, which is how earlier runs lost the passages.
Retries rebuild the prompt with one short correction note instead of appending.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiktoken  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from metrics import InferenceClient, InferenceRecord  # noqa: E402
from prompt_store import PromptVersion  # noqa: E402
from retrieval.hybrid_retriever import RETRIEVE_TOOL_SCHEMA, HybridRetriever  # noqa: E402
from retrieval.quality import RETRIEVAL_GATE_THRESHOLD  # noqa: E402
from retrieval.types import RankedChunk  # noqa: E402
from schema import SCHEMA_VERSION, StructuredAnswer, structured_answer_json_schema  # noqa: E402

MAX_TOOL_ROUNDS = 5
MAX_FINAL_RETRIES = 2

DEFAULT_NUM_CTX = 1024  # must match the server's OLLAMA_CONTEXT_LENGTH (scripts/start_ollama_tuned.bat)
FINAL_MAX_TOKENS = 256  # reserved for the structured answer; largest seen across all prior runs was 146
CHAT_TEMPLATE_OVERHEAD_TOKENS = 64  # role headers etc. the server adds, plus slack for tokenizer mismatch
MAX_CONTEXT_CHUNKS = 3
MIN_TRUNCATED_CHUNK_TOKENS = 100  # don't include a trailing passage cut shorter than this
RETRIEVE_TOP_K = 5
TOOL_TOP_K = 3
TOOL_PREVIEW_CHARS = 300

# Only an estimate of the server-side count (each model has its own tokenizer);
# CHAT_TEMPLATE_OVERHEAD_TOKENS absorbs the difference. Compare est_prompt_tokens
# in trace.jsonl against prompt_tokens in metrics.jsonl to check it.
_ENCODING = tiktoken.get_encoding("cl100k_base")


class GenerationFailure:
    """Distinguishes *why* the structured-final phase failed — see DESIGN.md section 7."""

    SCHEMA_IGNORED = "schema_ignored"  # model returned non-JSON despite response_format
    JSON_PARSE_ERROR = "json_parse_error"  # JSON parsed but doesn't match StructuredAnswer
    VALIDATION_ERROR = "validation_error"  # structurally valid but fails a business rule
    NO_RESPONSE = "no_response"  # the call itself failed (timeout, HTTP error, etc.)
    RETRIEVAL_QUALITY_GATE_BLOCKED = "retrieval_quality_gate_blocked"  # see DESIGN.md section 11


def _trace(trace_path: Path | None, question_id: str | None, event: str, **fields) -> None:
    """Appends one JSON line per pipeline event (retrieval, tool call + result,
    final prompt, final response) to a run's trace.jsonl. metrics.jsonl only has
    token counts; this is the only record of what the model asked for and was shown."""
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": question_id, "event": event, **fields}) + "\n")


def _ntok(text: str) -> int:
    return len(_ENCODING.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODING.decode(tokens[:max_tokens]).rstrip() + " ..."


def _chunk_summary(chunks: list[RankedChunk]) -> list[dict]:
    return [
        {
            "source": c.source,
            "chunk_index": c.chunk_index,
            "rerank_score": round(c.rerank_score, 4) if c.rerank_score is not None else None,
        }
        for c in chunks
    ]


def build_final_messages(
    question: str,
    chunks: list[RankedChunk],
    prompt: PromptVersion,
    num_ctx: int = DEFAULT_NUM_CTX,
    max_tokens: int = FINAL_MAX_TOKENS,
    correction: str | None = None,
    min_extra_rerank_score: float = RETRIEVAL_GATE_THRESHOLD,
) -> tuple[list[dict], list[dict], int]:
    """Builds the whole structured-final prompt so that it fits num_ctx with
    max_tokens left over for the answer. Passages go in rank order, whole while
    they fit; the last one that doesn't is cut to the remaining budget. The top
    passage is always included; later ones only if their rerank score clears
    min_extra_rerank_score (on the golden set every expected second passage
    scores >= 0.86 and every off-topic one <= 0.04).

    Returns (messages, included, est_prompt_tokens); included is one
    {source, chunk_index, truncated} per passage the model is actually shown.
    """
    head = "CONTEXT\n"
    tail = f"\nQUESTION\n{question}\n\nAnswer using only the CONTEXT above, as JSON matching the required schema."
    if correction:
        tail += f"\n\nYour previous reply was rejected: {correction} Reply again with corrected JSON."

    fixed = _ntok(prompt.text) + _ntok(head) + _ntok(tail) + CHAT_TEMPLATE_OVERHEAD_TOKENS
    remaining = num_ctx - max_tokens - fixed

    blocks: list[str] = []
    included: list[dict] = []
    for chunk in chunks[:MAX_CONTEXT_CHUNKS]:
        if included and (chunk.rerank_score or 0.0) < min_extra_rerank_score:
            break
        label =f"[{chunk.source} #{chunk.chunk_index}]\n"
        block = f"{label}{chunk.text.strip()}\n"
        cost = _ntok(block)
        truncated = False
        if cost > remaining:
            room = remaining - _ntok(label) - 2
            if included and room < MIN_TRUNCATED_CHUNK_TOKENS:
                break
            block = f"{label}{_truncate_to_tokens(chunk.text.strip(), max(room, MIN_TRUNCATED_CHUNK_TOKENS))}\n"
            cost = _ntok(block)
            truncated = True
        blocks.append(block)
        included.append({"source": chunk.source, "chunk_index": chunk.chunk_index, "truncated": truncated})
        remaining -= cost
        if truncated:
            break

    context = "\n".join(blocks) if blocks else "(no passages were retrieved)\n"
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": head + context + tail},
    ]
    est_prompt_tokens = sum(_ntok(m["content"]) for m in messages) + CHAT_TEMPLATE_OVERHEAD_TOKENS
    return messages, included, est_prompt_tokens


def _gate_abstain() -> StructuredAnswer:
    return StructuredAnswer(
        answer="Insufficient reliable context was retrieved to answer this question confidently.",
        citations=[],
        has_sufficient_context=False,
        confidence="low",
    )


def _structured_final(
    question: str,
    chunks: list[RankedChunk],
    client: InferenceClient,
    model: str,
    prompt: PromptVersion,
    temperature: float,
    question_id: str | None,
    num_ctx: int,
    max_final_retries: int,
    quant: str | None,
    trace_path: Path | None,
    round_index: int,
) -> tuple[StructuredAnswer | None, list[InferenceRecord], str | None]:
    schema = structured_answer_json_schema()
    records: list[InferenceRecord] = []
    final_error_kind: str | None = None
    correction: str | None = None

    for attempt in range(max_final_retries + 1):
        messages, included, est_prompt_tokens = build_final_messages(
            question, chunks, prompt, num_ctx=num_ctx, correction=correction
        )
        shown_keys = {(c["source"], c["chunk_index"]) for c in included}
        _trace(
            trace_path,
            question_id,
            "final_prompt",
            attempt=attempt,
            included=included,
            est_prompt_tokens=est_prompt_tokens,
            num_ctx=num_ctx,
            user_message=messages[-1]["content"],
        )

        result, record = client.chat(
            model=model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
            prompt_version=prompt.id,
            schema_version=SCHEMA_VERSION,
            question_id=question_id,
            round_index=round_index,
            attempt=attempt,
            call_kind="structured_final",
            quant=quant,
            max_tokens=FINAL_MAX_TOKENS,
        )
        records.append(record)
        _trace(
            trace_path,
            question_id,
            "final_response",
            attempt=attempt,
            content=result.content if result else None,
            prompt_tokens=record.prompt_tokens,
            error=record.error,
        )

        if result is None or result.content is None:
            final_error_kind = GenerationFailure.NO_RESPONSE
            correction = None
            continue

        try:
            parsed = json.loads(result.content)
        except json.JSONDecodeError:
            final_error_kind = GenerationFailure.SCHEMA_IGNORED
            correction = "it was not valid, complete JSON. Keep the answer brief."
            continue

        try:
            answer = StructuredAnswer.model_validate(parsed)
        except ValidationError as e:
            final_error_kind = GenerationFailure.JSON_PARSE_ERROR
            problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:3])
            correction = f"it did not match the schema ({problems})."
            continue

        not_shown = [(c.source, c.chunk_index) for c in answer.citations if (c.source, c.chunk_index) not in shown_keys]
        if not_shown:
            final_error_kind = GenerationFailure.VALIDATION_ERROR
            correction = (
                f"it cited {not_shown}, which is not a passage in the CONTEXT. Cite only the "
                "source and chunk_index of passages shown, or set has_sufficient_context to false."
            )
            continue

        if answer.has_sufficient_context and not answer.citations:
            final_error_kind = GenerationFailure.VALIDATION_ERROR
            correction = (
                "has_sufficient_context was true but citations was empty. Cite the passage(s) "
                "the answer came from, or set has_sufficient_context to false."
            )
            continue

        return answer, records, None

    return None, records, final_error_kind


def run_retrieve_then_read(
    question: str,
    retriever: HybridRetriever,
    client: InferenceClient,
    model: str,
    prompt: PromptVersion,
    temperature: float,
    run_id: str,
    question_id: str | None = None,
    max_final_retries: int = MAX_FINAL_RETRIES,
    quant: str | None = None,
    retrieval_gate_threshold: float = RETRIEVAL_GATE_THRESHOLD,
    num_ctx: int = DEFAULT_NUM_CTX,
    trace_path: Path | None = None,
) -> tuple[StructuredAnswer | None, list[InferenceRecord], str | None]:
    """Runs one question to completion with retrieval done in code. Returns
    (answer_or_None, all_records, final_error_kind_or_None). A gate-blocked
    question makes no LLM call at all, so all_records is empty for it."""
    retriever.last_quality_report = None
    chunks = retriever.retrieve(question, top_k=RETRIEVE_TOP_K)
    _trace(trace_path, question_id, "retrieval", query=question, results=_chunk_summary(chunks))

    # Retrieval-quality gate (DESIGN.md section 11).
    quality_report = retriever.last_quality_report
    if quality_report is None or not quality_report.passes_gate(retrieval_gate_threshold):
        _trace(
            trace_path,
            question_id,
            "gate_blocked",
            top1_rerank_score=quality_report.top1_rerank_score if quality_report else None,
            threshold=retrieval_gate_threshold,
        )
        return _gate_abstain(), [], GenerationFailure.RETRIEVAL_QUALITY_GATE_BLOCKED

    return _structured_final(
        question, chunks, client, model, prompt, temperature, question_id,
        num_ctx, max_final_retries, quant, trace_path, round_index=0,
    )


def run_agentic_structured(
    question: str,
    retriever: HybridRetriever,
    client: InferenceClient,
    model: str,
    prompt: PromptVersion,
    tool_prompt: PromptVersion,
    temperature: float,
    run_id: str,
    question_id: str | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    max_final_retries: int = MAX_FINAL_RETRIES,
    quant: str | None = None,
    retrieval_gate_threshold: float = RETRIEVAL_GATE_THRESHOLD,
    num_ctx: int = DEFAULT_NUM_CTX,
    trace_path: Path | None = None,
) -> tuple[StructuredAnswer | None, list[InferenceRecord], str | None]:
    """Runs one question to completion with the model driving retrieval.
    tool_prompt is the system prompt for the tool-calling rounds; prompt is the
    reader prompt for the structured-final call. Returns
    (answer_or_None, all_records, final_error_kind_or_None)."""
    retriever.last_quality_report = None
    messages: list[dict] = [{"role": "user", "content": question}]
    all_records: list[InferenceRecord] = []
    collected: dict[tuple[str, int], RankedChunk] = {}
    retrieve_called = False

    system_msg = {"role": "system", "content": tool_prompt.text}

    # Phase 1: tool-calling rounds.
    for round_index in range(max_tool_rounds):
        result, record = client.chat(
            model=model,
            messages=[system_msg] + messages,
            tools=[RETRIEVE_TOOL_SCHEMA],
            tool_choice="auto",
            temperature=temperature,
            prompt_version=tool_prompt.id,
            schema_version=None,
            question_id=question_id,
            round_index=round_index,
            attempt=0,
            call_kind="tool_round",
            quant=quant,
        )
        all_records.append(record)

        if result is None:
            return None, all_records, GenerationFailure.NO_RESPONSE

        if not result.tool_calls:
            _trace(trace_path, question_id, "no_tool_call", round=round_index, content=result.content)
            break

        messages.append(
            {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": result.tool_calls,
            }
        )
        for tc in result.tool_calls:
            raw_arguments = tc["function"]["arguments"]
            chunks: list[RankedChunk] = []
            if tc["function"]["name"] != "retrieve":
                tool_result = {"error": f"Unknown tool {tc['function']['name']!r}"}
            else:
                try:
                    args = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                # Only `query` is honoured. Tool-call arguments aren't
                # schema-enforced, so anything else the model adds is ignored.
                query = args.get("query") if isinstance(args, dict) else None
                if not isinstance(query, str) or not query.strip():
                    tool_result = {"error": "retrieve needs a non-empty 'query' string."}
                else:
                    retrieve_called = True
                    chunks = retriever.retrieve(query, top_k=TOOL_TOP_K)
                    for chunk in chunks:
                        best = collected.get(chunk.key)
                        if best is None or (chunk.rerank_score or 0.0) > (best.rerank_score or 0.0):
                            collected[chunk.key] = chunk
                    if chunks:
                        tool_result = {
                            "results": [
                                {
                                    "source": c.source,
                                    "chunk_index": c.chunk_index,
                                    "preview": c.text.strip()[:TOOL_PREVIEW_CHARS],
                                }
                                for c in chunks
                            ],
                            "note": "Previews only. The full passages are supplied for the final answer.",
                        }
                    else:
                        tool_result = {"results": [], "note": "No matching chunks found in the corpus."}
            _trace(
                trace_path,
                question_id,
                "tool_call",
                round=round_index,
                name=tc["function"]["name"],
                raw_arguments=raw_arguments,
                results=_chunk_summary(chunks),
                error=tool_result.get("error"),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_result),
                }
            )

    ranked = sorted(collected.values(), key=lambda c: c.rerank_score or 0.0, reverse=True)

    # Retrieval-quality gate (DESIGN.md section 11), on the best passage across
    # every retrieve call this question made. Skipped if the model never called
    # retrieve at all — the final call then runs with an empty CONTEXT.
    if retrieve_called:
        best_score = ranked[0].rerank_score if ranked else None
        if best_score is None or best_score < retrieval_gate_threshold:
            _trace(
                trace_path,
                question_id,
                "gate_blocked",
                top1_rerank_score=best_score,
                threshold=retrieval_gate_threshold,
            )
            return _gate_abstain(), all_records, GenerationFailure.RETRIEVAL_QUALITY_GATE_BLOCKED

    # Phase 2: structured-final call, built from scratch — see module docstring.
    answer, final_records, error_kind = _structured_final(
        question, ranked, client, model, prompt, temperature, question_id,
        num_ctx, max_final_retries, quant, trace_path, round_index=max_tool_rounds,
    )
    return answer, all_records + final_records, error_kind


def main() -> None:
    import argparse

    from prompt_store import load_prompt
    from retrieval.hybrid_retriever import build_default_hybrid_retriever

    parser = argparse.ArgumentParser(description="Manual single-question smoke test.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="retrieve_first", choices=["retrieve_first", "agentic"])
    parser.add_argument("--prompt-id", default="system_v3", help="Reader prompt for the structured-final call.")
    parser.add_argument("--tool-prompt-id", default="system_v1", help="System prompt for agentic tool rounds.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Must match the server's OLLAMA_CONTEXT_LENGTH.")
    parser.add_argument("--embed-provider", default="ollama", choices=["openai", "ollama", "fake"])
    args = parser.parse_args()

    retriever = build_default_hybrid_retriever(args.embed_provider)
    prompt = load_prompt(args.prompt_id)
    run_dir = Path(__file__).resolve().parent / "eval" / "runs" / "manual-smoke"
    client = InferenceClient(run_id="manual-smoke", metrics_path=run_dir / "metrics.jsonl")

    common = dict(
        question=args.question,
        retriever=retriever,
        client=client,
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        run_id="manual-smoke",
        question_id="manual",
        num_ctx=args.num_ctx,
        trace_path=run_dir / "trace.jsonl",
    )
    if args.mode == "agentic":
        answer, records, error_kind = run_agentic_structured(tool_prompt=load_prompt(args.tool_prompt_id), **common)
    else:
        answer, records, error_kind = run_retrieve_then_read(**common)

    print(f"\n{len(records)} inference call(s) made.")
    if answer:
        print("\nStructuredAnswer:")
        print(answer.model_dump_json(indent=2))
        if error_kind:
            print(f"\n({error_kind})")
    else:
        print(f"\nFailed: {error_kind}")


if __name__ == "__main__":
    main()
