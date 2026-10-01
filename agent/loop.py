"""Phase 2: tool-calling agent loop.

Runs entirely against a local Ollama server via its OpenAI-compatible
/v1/chat/completions endpoint (openai>=1.40's client, pointed at a local
base_url — no API key, no data leaves the machine). retrieve() is exposed as
a real tool call: the model decides whether to search the corpus, answer from
its own knowledge, or ask a clarifying question — retrieval is never forced.

The system prompt requires two things on every turn that uses retrieved
context: (1) cite sources as [filename, chunk N] inline, and (2) say
explicitly when the corpus doesn't contain enough to answer, rather than
filling the gap from general knowledge.

Usage:
    python -m agent.loop                          # interactive REPL
    python -m agent.loop --model qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from agent.tools import RETRIEVE_TOOL_SCHEMA, IndexNotFoundError, Retriever, make_retrieve_tool
from ingest.embeddings import OLLAMA_DEFAULT_BASE_URL, get_embedder

DEFAULT_CHAT_MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = """You are a research assistant over the user's personal model risk \
management (MRM) document corpus: validation reports, regulatory guidance \
(e.g. SR 11-7, CECL, ALM), closure packs, exam responses, and internal project docs.

Rules:
- Use the retrieve tool whenever answering needs specific facts, figures, findings, \
or wording from those documents. Don't call it for general knowledge questions or \
small talk.
- Every claim drawn from a retrieved chunk must be cited inline as \
[source_filename, chunk N], immediately after the sentence it supports.
- If retrieval returns nothing relevant, or the returned chunks don't actually \
contain what's needed, say plainly that the corpus doesn't have enough to answer \
— do not fill the gap from general knowledge and present it as if it came from \
the documents.
- If a question is ambiguous (e.g. which document, which time period), ask a \
clarifying question instead of guessing.
"""


# Default per-request timeout for the chat client. This machine's local Ollama
# CPU inference is known to be slow AND occasionally stalls well past normal
# generation time before responding (see README "Known issues" — root cause was
# never pinned down; Defender exclusions, disabling Vulkan GPU discovery, and
# disabling flash attention were all tried and none fixed it). 900s is a
# deliberately generous ceiling to ride that out rather than a "should never
# take this long" budget — a request that hits it has almost certainly failed,
# not just been slow, based on the ~5min mark where stalls have surfaced before.
DEFAULT_CHAT_TIMEOUT_SECONDS = 900.0


def _make_client(base_url: str, timeout: float = DEFAULT_CHAT_TIMEOUT_SECONDS):
    from openai import OpenAI

    # Ollama's OpenAI-compatible endpoint ignores the API key but the client
    # requires a non-empty string.
    #
    # max_retries=0: the SDK's default (2) would silently re-run a multi-minute
    # stalled/failed request up to 2 more times, turning one ~15min wait into a
    # possible ~45min one with no visibility in between. One attempt, and the
    # failure (timeout or the intermittent HTTP 500 seen in testing) surfaces
    # immediately instead.
    return OpenAI(
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key="ollama",
        timeout=timeout,
        max_retries=0,
    )


def run_agent_loop(
    user_message: str,
    history: list[dict],
    client,
    model: str,
    retrieve_fn,
) -> tuple[str, list[dict]]:
    """Runs one user turn to completion, including any tool-call round-trips.

    Returns (assistant_reply_text, updated_history).
    """
    messages = history + [{"role": "user", "content": user_message}]

    max_rounds = 5
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=[RETRIEVE_TOOL_SCHEMA],
            tool_choice="auto",
        )
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "", messages

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            if tc.function.name != "retrieve":
                result = {"error": f"Unknown tool {tc.function.name!r}"}
            else:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = retrieve_fn(**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    # Model kept calling tools past max_rounds — force a final answer.
    messages.append(
        {
            "role": "user",
            "content": "Please give your final answer now based on what you've retrieved so far.",
        }
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )
    final = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": final})
    return final, messages


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Interactive REPL for the RAG agent (local Ollama).")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL))
    parser.add_argument("--embed-provider", default="ollama", choices=["openai", "ollama", "fake"])
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("OLLAMA_CHAT_TIMEOUT", DEFAULT_CHAT_TIMEOUT_SECONDS)),
        help=(
            "Per-request timeout in seconds for the chat client (default "
            f"{DEFAULT_CHAT_TIMEOUT_SECONDS:.0f}s). Local CPU inference on this "
            "machine is slow and occasionally stalls further — see README."
        ),
    )
    args = parser.parse_args()

    embedder = get_embedder(args.embed_provider)
    try:
        retriever = Retriever(embedder)
    except IndexNotFoundError as e:
        print(f"[error] {e}")
        return

    retrieve_fn = make_retrieve_tool(retriever)
    client = _make_client(args.base_url, timeout=args.timeout)

    print(
        f"RAG agent ready (model={args.model}, embed={args.embed_provider}, "
        f"timeout={args.timeout:.0f}s). Local CPU inference can take several "
        f"minutes per reply on this machine — that's expected. Ctrl+C to quit.\n"
    )
    history: list[dict] = []
    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        try:
            reply, history = run_agent_loop(user_message, history, client, args.model, retrieve_fn)
        except Exception as e:
            print(f"[error] {e}")
            continue
        print(f"\nagent> {reply}\n")


if __name__ == "__main__":
    main()
