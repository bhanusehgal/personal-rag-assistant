"""InferenceClient: the single call site every chat completion in this project
goes through. Wraps Ollama's OpenAI-compatible streaming chat endpoint and logs
one InferenceRecord (TTFT, tokens/sec, total latency, success/failure) per call
to a run's metrics.jsonl — see DESIGN.md section 6.

Mirrors ../agent/loop.py's client construction (max_retries=0, generous
per-request timeout) since that pattern is already proven against this
machine's slow/occasionally-stalling local Ollama server.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import tiktoken

DEFAULT_TIMEOUT_SECONDS = 900.0
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class ToolCallAccumulator:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class ChatResult:
    """Reconstructed final message from a streamed response."""

    content: str | None
    tool_calls: list[dict] | None  # [{"id", "type": "function", "function": {"name", "arguments"}}]


@dataclass
class InferenceRecord:
    run_id: str
    model: str
    quant: str | None
    temperature: float
    prompt_version: str
    schema_version: str | None
    question_id: str | None
    round_index: int
    attempt: int
    call_kind: Literal["tool_round", "structured_final", "plain"]
    ttft_s: float | None
    tokens_per_sec: float | None
    total_latency_s: float
    completion_tokens: int | None
    prompt_tokens: int | None
    token_count_method: str
    success: bool
    error: str | None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_json_line(self) -> str:
        return json.dumps(asdict(self))


def _make_openai_client(base_url: str, timeout: float):
    from openai import OpenAI

    return OpenAI(
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key="ollama",
        timeout=timeout,
        max_retries=0,
    )


class InferenceClient:
    def __init__(
        self,
        run_id: str,
        metrics_path: Path,
        base_url: str = "http://localhost:11434",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.run_id = run_id
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = _make_openai_client(base_url, timeout)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
        temperature: float = 0.0,
        prompt_version: str = "",
        schema_version: str | None = None,
        question_id: str | None = None,
        round_index: int = 0,
        attempt: int = 0,
        call_kind: Literal["tool_round", "structured_final", "plain"] = "plain",
        quant: str | None = None,
        max_tokens: int = 300,
    ) -> tuple[ChatResult | None, InferenceRecord]:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.monotonic()
        ttft_s: float | None = None
        content_parts: list[str] = []
        tool_call_accs: dict[int, ToolCallAccumulator] = {}
        usage_completion_tokens: int | None = None
        usage_prompt_tokens: int | None = None
        success = False
        error: str | None = None
        result: ChatResult | None = None

        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if ttft_s is None and chunk.choices:
                    delta = chunk.choices[0].delta
                    if (delta.content) or (getattr(delta, "tool_calls", None)):
                        ttft_s = time.monotonic() - t0

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                    if getattr(delta, "tool_calls", None):
                        for tc_delta in delta.tool_calls:
                            acc = tool_call_accs.setdefault(tc_delta.index, ToolCallAccumulator())
                            if tc_delta.id:
                                acc.id = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                acc.name = tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                acc.arguments += tc_delta.function.arguments

                if getattr(chunk, "usage", None):
                    usage_completion_tokens = chunk.usage.completion_tokens
                    usage_prompt_tokens = chunk.usage.prompt_tokens

            t_end = time.monotonic()
            content = "".join(content_parts) or None
            tool_calls = None
            if tool_call_accs:
                tool_calls = [
                    {
                        "id": acc.id,
                        "type": "function",
                        "function": {"name": acc.name, "arguments": acc.arguments},
                    }
                    for _, acc in sorted(tool_call_accs.items())
                ]
            result = ChatResult(content=content, tool_calls=tool_calls)
            success = True

        except Exception as e:  # noqa: BLE001 - deliberately broad: any failure must still be logged
            t_end = time.monotonic()
            error = f"{type(e).__name__}: {e}"

        total_latency_s = t_end - t0

        if usage_completion_tokens is not None:
            completion_tokens = usage_completion_tokens
            prompt_tokens = usage_prompt_tokens
            token_count_method = "usage_field"
        else:
            approx_text = "".join(content_parts)
            completion_tokens = len(_ENCODING.encode(approx_text)) if approx_text else None
            prompt_tokens = None
            token_count_method = "tiktoken_approx"

        tokens_per_sec = None
        if success and completion_tokens and ttft_s is not None and total_latency_s > ttft_s:
            tokens_per_sec = completion_tokens / (total_latency_s - ttft_s)

        record = InferenceRecord(
            run_id=self.run_id,
            model=model,
            quant=quant,
            temperature=temperature,
            prompt_version=prompt_version,
            schema_version=schema_version,
            question_id=question_id,
            round_index=round_index,
            attempt=attempt,
            call_kind=call_kind,
            ttft_s=ttft_s,
            tokens_per_sec=tokens_per_sec,
            total_latency_s=total_latency_s,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            token_count_method=token_count_method,
            success=success,
            error=error,
        )
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(record.as_json_line() + "\n")

        return result, record
