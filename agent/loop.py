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

#: Usable context, in tokens. Overflowing it ends the run rather than degrading
#: it, so the whole request is budgeted -- system prompt and tool schemas
#: included, not just the recent exchanges.
_CONTEXT_TOKENS = int(os.environ.get("AGENT_CONTEXT_TOKENS", 14000))

#: Dense JSON tokenises far worse than prose. Measured on this system's own tool
#: output, close to three characters per token; four was optimistic enough to
#: overflow a 16k window on the fourth step.
_CHARS_PER_TOKEN = 3


def _size(message: dict[str, Any]) -> int:
    return len(str(message.get("content") or "")) + len(str(message.get("tool_calls") or ""))


def _trim(messages: list[dict[str, Any]], overhead_chars: int = 0,
          scale: float = 1.0) -> list[dict[str, Any]]:
    """Keep the opening and as much recent conversation as the window allows."""
    head, rest = messages[:2], messages[2:]
    budget = int(_CONTEXT_TOKENS * _CHARS_PER_TOKEN * scale) - overhead_chars
    budget -= sum(_size(m) for m in head)

    kept: list[dict[str, Any]] = []
    for message in reversed(rest):
        size = _size(message)
        if size > budget and kept:
            break
        budget -= size
        kept.append(message)
    kept.reverse()
    # A tool reply whose call was trimmed away is rejected by the API.
    while kept and kept[0].get("role") == "tool":
        kept = kept[1:]
    return head + kept


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
    stopped = "steps"
    #: Shrinks on an overflow and stays shrunk. Restoring the full budget would
    #: just overflow again a step later.
    scale = 1.0

    for steps in range(1, max_steps + 1):
        if session.clock.must_finalize:
            log("[loop] time reserve reached; stopping the agent")
            stopped = "time"
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

        tools = REGISTRY.schemas(session.regime)
        overhead = len(json.dumps(tools))
        try:
            response = _complete(client, model, _trim(messages, overhead, scale), tools)
        except Exception as err:  # noqa: BLE001
            # A window overflow is recoverable: keep less history and go on.
            # Giving up here would waste whatever time is left in the run.
            if "context" in str(err).lower() and scale > 0.3:
                scale *= 0.5
                log(f"[loop] context overflowed; history budget now {scale:.0%} and retrying")
                session.note(f"context overflowed; history budget reduced to {scale:.0%}")
                try:
                    response = _complete(client, model, _trim(messages, overhead, scale), tools)
                except Exception as retry_err:  # noqa: BLE001
                    log(f"[loop] still failing after the retry: {retry_err}")
                    stopped = "context"
                    break
            else:
                log(f"[loop] the model call failed and could not be retried: {err}")
                session.note(f"loop aborted at step {steps}: {err}")
                stopped = "error"
                break

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if message.content:
            log(f"\n--- step {steps} ---\n{message.content[:1500]}")
        if not message.tool_calls:
            log("[loop] the agent finished")
            stopped = "finished"
            break

        for call in message.tool_calls:
            name = call.function.name
            log(f"[step {steps}] {name}({(call.function.arguments or '')[:200]})")
            output = REGISTRY.dispatch(name, call.function.arguments)
            log(f"         -> {output[:300]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return {
        "steps": steps,
        "stopped": stopped,
        "regime": session.regime,
        "experiments": len(session.ledger),
        "final_message": next(
            (m.get("content") for m in reversed(messages)
             if m.get("role") == "assistant" and m.get("content")),
            "",
        ),
    }
