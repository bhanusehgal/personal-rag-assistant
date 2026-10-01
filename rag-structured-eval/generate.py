"""Agentic tool-calling phase + structured-final phase, per DESIGN.md section 7.

Ollama's response_format (JSON-schema-constrained decoding) and tools/
tool_choice are never combined in the same request, so a question is answered
in two phases:

  1. Tool-calling phase: mirrors ../agent/loop.py's run_agent_loop, using the
     existing RETRIEVE_TOOL_SCHEMA / make_retrieve_tool from ../agent/tools.py
     unchanged. Runs up to max_tool_rounds rounds until the model stops
     calling tools.
  2. Structured-final phase: ALWAYS a separate call with tools removed and
     response_format=structured_answer_json_schema() set, regardless of how
     phase 1 ended. Validated against StructuredAnswer + a business rule (no
     hallucinated citations), retried up to max_final_retries times with the
     concrete error fed back into the conversation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from agent.tools import RETRIEVE_TOOL_SCHEMA  # noqa: E402
from metrics import InferenceClient, InferenceRecord  # noqa: E402
from prompt_store import PromptVersion  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever, make_hybrid_retrieve_tool  # noqa: E402
from retrieval.quality import RETRIEVAL_GATE_THRESHOLD  # noqa: E402
from schema import SCHEMA_VERSION, StructuredAnswer, structured_answer_json_schema  # noqa: E402

MAX_TOOL_ROUNDS = 5
MAX_FINAL_RETRIES = 2


class GenerationFailure:
    """Distinguishes *why* the structured-final phase failed — see DESIGN.md section 7."""

    SCHEMA_IGNORED = "schema_ignored"  # model returned non-JSON despite response_format
    JSON_PARSE_ERROR = "json_parse_error"  # JSON parsed but doesn't match StructuredAnswer
    VALIDATION_ERROR = "validation_error"  # structurally valid but fails a business rule
    NO_RESPONSE = "no_response"  # the call itself failed (timeout, HTTP error, etc.)
    RETRIEVAL_QUALITY_GATE_BLOCKED = "retrieval_quality_gate_blocked"  # see DESIGN.md section 11


def _trace(trace_path: Path | None, question_id: str | None, event: str, **fields) -> None:
    """Appends one JSON line per pipeline event (tool call + result, final prompt,
    final response) to a run's trace.jsonl. metrics.jsonl only has token counts;
    this is the only record of what the model actually asked for and was shown."""
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": question_id, "event": event, **fields}) + "\n")


def _retrieved_key_set(tool_results: list[dict]) -> set[tuple[str, int]]:
    keys = set()
    for result in tool_results:
        for chunk in result.get("results", []):
            keys.add((chunk["source"], chunk["chunk_index"]))
    return keys


def run_agentic_structured(
    question: str,
    retriever: HybridRetriever,
    client: InferenceClient,
    model: str,
    prompt: PromptVersion,
    temperature: float,
    run_id: str,
    question_id: str | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    max_final_retries: int = MAX_FINAL_RETRIES,
    quant: str | None = None,
    retrieval_gate_threshold: float = RETRIEVAL_GATE_THRESHOLD,
    trace_path: Path | None = None,
) -> tuple[StructuredAnswer | None, list[InferenceRecord], str | None]:
    """Runs one question to completion. Returns (answer_or_None, all_records, final_error_kind_or_None)."""
    retrieve_fn = make_hybrid_retrieve_tool(retriever)
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_results: list[dict] = []
    all_records: list[InferenceRecord] = []

    system_msg = {"role": "system", "content": prompt.text}

    # Phase 1: tool-calling rounds.
    for round_index in range(max_tool_rounds):
        result, record = client.chat(
            model=model,
            messages=[system_msg] + messages,
            tools=[RETRIEVE_TOOL_SCHEMA],
            tool_choice="auto",
            temperature=temperature,
            prompt_version=prompt.id,
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
            messages.append({"role": "assistant", "content": result.content or ""})
            break

        messages.append(
            {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": result.tool_calls,
            }
        )
        for tc in result.tool_calls:
            if tc["function"]["name"] != "retrieve":
                tool_result = {"error": f"Unknown tool {tc['function']['name']!r}"}
            else:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_result = retrieve_fn(**args)
            _trace(
                trace_path,
                question_id,
                "tool_call",
                round=round_index,
                name=tc["function"]["name"],
                raw_arguments=tc["function"]["arguments"],
                num_results=len(tool_result.get("results", [])),
                result_keys=[[c["source"], c["chunk_index"]] for c in tool_result.get("results", [])],
                error=tool_result.get("error"),
                note=tool_result.get("note"),
                result_chars=len(json.dumps(tool_result)),
            )
            tool_results.append(tool_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_result),
                }
            )

    retrieved_keys = _retrieved_key_set(tool_results)

    # Retrieval-quality gate (DESIGN.md section 11): computed on every
    # HybridRetriever.retrieve() call regardless of outcome; only *blocked*
    # here if the last call's report fails the threshold. If phase 1 never
    # called retrieve() at all (model answered directly), last_quality_report
    # is None and the gate is skipped — this path is unaffected.
    quality_report = retriever.last_quality_report
    if quality_report is not None and not quality_report.passes_gate(retrieval_gate_threshold):
        abstain = StructuredAnswer(
            answer="Insufficient reliable context was retrieved to answer this question confidently.",
            citations=[],
            has_sufficient_context=False,
            confidence="low",
        )
        return abstain, all_records, GenerationFailure.RETRIEVAL_QUALITY_GATE_BLOCKED

    schema = structured_answer_json_schema()
    final_error_kind: str | None = None

    # Phase 2: structured-final call, always separate from the tool-calling rounds.
    final_messages = messages + [
        {
            "role": "user",
            "content": (
                "Give your final answer now, based only on what was retrieved above, "
                "as JSON matching the required schema."
            ),
        }
    ]
    for attempt in range(max_final_retries + 1):
        result, record = client.chat(
            model=model,
            messages=[system_msg] + final_messages,
            response_format=schema,
            temperature=temperature,
            prompt_version=prompt.id,
            schema_version=SCHEMA_VERSION,
            question_id=question_id,
            round_index=max_tool_rounds,
            attempt=attempt,
            call_kind="structured_final",
            quant=quant,
        )
        all_records.append(record)
        _trace(
            trace_path,
            question_id,
            "final_response",
            attempt=attempt,
            content=result.content if result else None,
            prompt_tokens=record.prompt_tokens,
        )

        if result is None or result.content is None:
            final_error_kind = GenerationFailure.NO_RESPONSE
            final_messages.append(
                {"role": "user", "content": "No response was returned. Please answer again as JSON."}
            )
            continue

        try:
            parsed = json.loads(result.content)
        except json.JSONDecodeError as e:
            final_error_kind = GenerationFailure.SCHEMA_IGNORED
            final_messages.append({"role": "assistant", "content": result.content})
            final_messages.append(
                {"role": "user", "content": f"That was not valid JSON ({e}). Respond with JSON only, matching the schema."}
            )
            continue

        try:
            answer = StructuredAnswer.model_validate(parsed)
        except ValidationError as e:
            final_error_kind = GenerationFailure.JSON_PARSE_ERROR
            final_messages.append({"role": "assistant", "content": result.content})
            final_messages.append(
                {"role": "user", "content": f"That JSON didn't match the required schema: {e}. Try again."}
            )
            continue

        hallucinated = [c for c in answer.citations if (c.source, c.chunk_index) not in retrieved_keys]
        if hallucinated:
            final_error_kind = GenerationFailure.VALIDATION_ERROR
            final_messages.append({"role": "assistant", "content": result.content})
            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"These citations were not among the chunks actually retrieved: {hallucinated}. "
                        "Only cite (source, chunk_index) pairs that appeared in a retrieve result above, "
                        "or set has_sufficient_context to false. Try again."
                    ),
                }
            )
            continue

        return answer, all_records, None

    return None, all_records, final_error_kind


def main() -> None:
    import argparse

    from prompt_store import load_prompt
    from retrieval.hybrid_retriever import build_default_hybrid_retriever

    parser = argparse.ArgumentParser(description="Manual smoke test for run_agentic_structured.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-id", default="system_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--embed-provider", default="ollama", choices=["openai", "ollama", "fake"])
    args = parser.parse_args()

    retriever = build_default_hybrid_retriever(args.embed_provider)
    prompt = load_prompt(args.prompt_id)
    client = InferenceClient(
        run_id="manual-smoke",
        metrics_path=Path(__file__).resolve().parent / "eval" / "runs" / "manual-smoke" / "metrics.jsonl",
    )

    answer, records, error_kind = run_agentic_structured(
        question=args.question,
        retriever=retriever,
        client=client,
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        run_id="manual-smoke",
        question_id="manual",
        trace_path=Path(__file__).resolve().parent / "eval" / "runs" / "manual-smoke" / "trace.jsonl",
    )

    print(f"\n{len(records)} inference call(s) made.")
    if answer:
        print("\nStructuredAnswer:")
        print(answer.model_dump_json(indent=2))
    else:
        print(f"\nFailed: {error_kind}")


if __name__ == "__main__":
    main()
