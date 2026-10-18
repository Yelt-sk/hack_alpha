#!/usr/bin/env python
"""
Agent runner: drives the MLForge MCP server with an LLM.

Backends are OpenAI-compatible, which covers Ollama, llama.cpp's server, vLLM
and LM Studio -- so the same loop runs against a local model and against a
hosted one with nothing but a --base-url change.

    # local, via Ollama
    python scripts/run_agent.py --base-url http://localhost:11434/v1 \
        --model qwen2.5:14b-instruct --dataset synthetic_churn

    # any OpenAI-compatible endpoint
    python scripts/run_agent.py --base-url https://... --api-key $KEY --model <id>

The server is the same in both cases. That is the point of putting the ML
competence in the tools rather than in the prompt: swapping the brain does not
change what the agent is *able* to do, only how well it chooses.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SYSTEM = (ROOT / "prompts" / "system.md").read_text(encoding="utf-8")


def to_openai_tools(tools) -> list[dict[str, Any]]:
    return [{"type": "function",
             "function": {"name": t.name, "description": (t.description or "")[:1024],
                          "parameters": t.input_schema}}
            for t in tools]


def compact(text: str, limit: int) -> str:
    """Re-serialise JSON without indentation before truncating.

    The server pretty-prints, which costs roughly a third of the tokens in pure
    whitespace -- material when the local model has a 12k window.
    """
    try:
        text = json.dumps(json.loads(text), separators=(",", ":"), default=str)
    except (json.JSONDecodeError, TypeError):
        pass
    return text if len(text) <= limit else text[:limit] + f"...[truncated at {limit} chars]"


def extract_text(result) -> str:
    parts = []
    for c in getattr(result, "content", []) or []:
        if getattr(c, "type", None) == "text":
            parts.append(c.text)
    if not parts and getattr(result, "structuredContent", None) is not None:
        parts.append(json.dumps(result.structured_content, default=str))
    return "\n".join(parts) or "(no output)"


async def chat(client, model: str, messages: list[dict], tools: list[dict], temperature: float):
    return await client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto",
        temperature=temperature, max_tokens=2048,
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("MLFORGE_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--api-key", default=os.environ.get("MLFORGE_API_KEY", "not-needed"))
    ap.add_argument("--model", default=os.environ.get("MLFORGE_MODEL", "qwen2.5:14b-instruct"))
    ap.add_argument("--dataset", default="synthetic_churn")
    ap.add_argument("--target", default=None, help="Required when --dataset is a CSV path.")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tool-chars", type=int, default=3000,
                    help="Truncate each tool result. Lower it for small context windows.")
    ap.add_argument("--parallel-tools", action="store_true",
                    help="Execute every tool call in a turn. Off by default: small models "
                         "emit the whole plan at once, before any result exists.")
    args = ap.parse_args()
    serial = not args.parallel_tools

    try:
        from openai import AsyncOpenAI
    except ImportError:
        sys.exit("pip install openai  # the runner speaks the OpenAI-compatible protocol")

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server.py")], env=None)

    task = (
        f"Build the best honest model you can for the dataset '{args.dataset}'"
        + (f" (target column '{args.target}')" if args.target else "")
        + ". Follow the procedure. Finish with final_evaluate and export_solution, "
          "then summarise what you did and what the scores mean."
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            oai_tools = to_openai_tools(tools)
            print(f"Connected. {len(tools)} tools: {', '.join(t.name for t in tools)}\n")

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": task},
            ]

            for step in range(1, args.max_steps + 1):
                resp = await chat(client, args.model, messages, oai_tools, args.temperature)
                msg = resp.choices[0].message
                calls = msg.tool_calls or []
                messages.append({
                    "role": "assistant", "content": msg.content or "",
                    **({"tool_calls": [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name,
                                                     "arguments": c.function.arguments}}
                                       for c in calls]} if calls else {}),
                })

                if msg.content:
                    print(f"\n--- step {step} | assistant ---\n{msg.content}")
                if not calls:
                    print("\nAgent finished.")
                    break

                # Small models emit the whole workflow in one turn, before any
                # result exists. The protocol still demands a reply per call, so
                # answer the extras with an instruction instead of running them.
                if serial and len(calls) > 1:
                    print(f"[step {step}] model emitted {len(calls)} calls; running the first only")
                for idx, c in enumerate(calls):
                    if serial and idx > 0:
                        messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps({
                            "status": "not_executed",
                            "reason": "You called several tools at once.",
                            "hint": "Call exactly one tool per message and wait for its result "
                                    "before deciding the next one."})})
                        continue
                    name = c.function.name
                    try:
                        cargs = json.loads(c.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        out = f"Your arguments were not valid JSON ({e}). Re-send this call with valid JSON."
                        print(f"[step {step}] {name} -> malformed arguments")
                        messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
                        continue
                    print(f"[step {step}] -> {name}({json.dumps(cargs, default=str)[:220]})")
                    try:
                        res = await session.call_tool(name, cargs)
                        out = extract_text(res)
                    except Exception as e:  # keep the loop alive; the model can recover
                        out = json.dumps({"status": "error", "problem": str(e),
                                          "hint": "Check the argument names against the tool schema."})
                    print(f"           <- {out[:400]}")
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": compact(out, args.max_tool_chars)})
            else:
                print(f"\nStopped: reached the {args.max_steps}-step limit.")


if __name__ == "__main__":
    asyncio.run(main())
