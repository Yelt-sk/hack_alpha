"""The agent loop: offer tools, let the model choose, execute, repeat.

This is the part ``baseline.py`` demonstrates with MCP. The platform image has no
``mcp`` package, so the same contract is served in-process: the registry produces
the ``tools=[...]`` payload and dispatches the calls itself.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .prompt import build as build_prompt
from .session import Session
from .tools.registry import REGISTRY

#: Keep the transcript bounded. The oldest exchanges are the least useful ones --
#: anything the agent still needs is one `get_context` call away.
_KEEP_RECENT = 40


def _trim(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(messages) <= _KEEP_RECENT + 2:
        return messages
    head, tail = messages[:2], messages[-_KEEP_RECENT:]
    # A tool reply with no preceding call confuses the API, so drop a leading orphan.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    return head + tail


def _complete(client: Any, model: str, messages: list[dict], tools: list[dict]) -> Any:
    """One chat completion, retried on transient failures."""
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30),
           reraise=True)
    def call() -> Any:
        return client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            tool_choice="auto", parallel_tool_calls=False, temperature=0.2,
        )

    return call()


def run(session: Session, max_steps: int = 120, log=print) -> dict[str, Any]:
    """Drive the model until it stops, the steps run out, or the clock does."""
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_prompt(session)},
        {"role": "user", "content": "Solve the task. Begin with get_context."},
    ]
    announced: set[str] = set()
    steps = 0

    for steps in range(1, max_steps + 1):
        if session.clock.must_finalize:
            log("[loop] time reserve reached; stopping the agent")
            break

        phase = session.clock.phase
        if phase not in announced and phase in ("exploit", "finalize"):
            announced.add(phase)
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "phase_change": phase,
                    "minutes_left": round(session.clock.usable_remaining / 60, 1),
                    "instruction": (
                        "Stop exploring. Tune or finalise the best recipe you have."
                        if phase == "exploit"
                        else "Finalise now: confirm with leaderboard that the best result is saved, "
                             "then summarise and stop."
                    ),
                }, ensure_ascii=False),
            })

        try:
            response = _complete(client, model, _trim(messages), REGISTRY.schemas(session.regime))
        except Exception as err:  # noqa: BLE001
            log(f"[loop] the model call failed and could not be retried: {err}")
            session.note(f"loop aborted at step {steps}: {err}")
            break

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if message.content:
            log(f"\n--- step {steps} ---\n{message.content[:1500]}")
        if not message.tool_calls:
            log("[loop] the agent finished")
            break

        for call in message.tool_calls:
            name = call.function.name
            log(f"[step {steps}] {name}({(call.function.arguments or '')[:200]})")
            output = REGISTRY.dispatch(name, call.function.arguments)
            log(f"         -> {output[:300]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return {
        "steps": steps,
        "regime": session.regime,
        "experiments": len(session.ledger),
        "final_message": next(
            (m.get("content") for m in reversed(messages)
             if m.get("role") == "assistant" and m.get("content")),
            "",
        ),
    }
