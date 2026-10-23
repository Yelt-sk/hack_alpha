"""The agent loop: offer tools, let the model choose, execute, repeat.

This is the part ``baseline.py`` demonstrates with MCP. The platform image has no
``mcp`` package, so the same contract is served in-process: the registry produces
the ``tools=[...]`` payload and dispatches the calls itself.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

from agent_prompt import build as build_prompt
from agent_session import Session
from agent_registry import REGISTRY

#: Usable context, in tokens. Overflowing it ends the run rather than degrading
#: it, so the whole request is budgeted -- system prompt and tool schemas
#: included, not just the recent exchanges.
#:
#: The platform's model is not the one this was developed against, and its window
#: is not documented, so the number is discovered rather than assumed: the server
#: is asked, and only if it will not say does this default apply.
#:
#: The platform's proxy does not advertise it, so the default is what the proxy
#: was measured to accept: a 60,057-token prompt went through, which puts the real
#: window at 64k or more. Budgeting 14k here -- the earlier guess -- threw away
#: three quarters of it and trimmed the experiment record the agent needs to avoid
#: repeating itself. Half of the proven floor leaves room for a smaller endpoint
#: and for tokenisation being worse than the estimate; an overflow still shrinks
#: the budget and retries, so being wrong here costs a step, not the run.
_CONTEXT_TOKENS = int(os.environ.get("AGENT_CONTEXT_TOKENS", 0))
_CONTEXT_FALLBACK = 32000


def discover_context(client: Any, model: str, log=print) -> int:
    """Ask the endpoint how much context it has; fall back to a safe guess.

    llama.cpp and vLLM both report it on /v1/models; OpenAI-compatible servers
    that do not simply leave the field out.
    """
    if _CONTEXT_TOKENS:
        return _CONTEXT_TOKENS
    for field in ("n_ctx", "context_length", "max_model_len", "max_context_length"):
        try:
            listed = client.models.list()
            for entry in listed.data:
                value = getattr(entry, field, None) or (entry.model_extra or {}).get(field)
                if isinstance(value, int) and value > 1024:
                    # Leave room for the reply and for tokenisation being worse
                    # than the estimate.
                    usable = int(value * 0.80)
                    log(f"[loop] endpoint reports {field}={value}; budgeting {usable} tokens")
                    return usable
        except Exception:  # noqa: BLE001
            break
    log(f"[loop] context size not advertised; budgeting {_CONTEXT_FALLBACK} tokens")
    return _CONTEXT_FALLBACK

#: Dense JSON tokenises far worse than prose. Measured on this system's own tool
#: output, close to three characters per token; four was optimistic enough to
#: overflow a 16k window on the fourth step.
_CHARS_PER_TOKEN = 3


def _size(message: dict[str, Any]) -> int:
    return len(str(message.get("content") or "")) + len(str(message.get("tool_calls") or ""))


def _trim(messages: list[dict[str, Any]], overhead_chars: int = 0,
          scale: float = 1.0, context_tokens: int = _CONTEXT_FALLBACK) -> list[dict[str, Any]]:
    """Keep the opening and as much recent conversation as the window allows."""
    head, rest = messages[:2], messages[2:]
    budget = int(context_tokens * _CHARS_PER_TOKEN * scale) - overhead_chars
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


#: How many of the last `_BARREN_WINDOW` tool calls may be fruitless before the
#: loop stops arguing. A *consecutive* count was the first attempt and it failed
#: in testing: the model interleaved duplicates with successful runs, so the
#: counter reset before it ever tripped. Oscillation is a property of the recent
#: window, not of an unbroken streak.
_BARREN_LIMIT = int(os.environ.get("AGENT_BARREN_LIMIT", 5))
_BARREN_WINDOW = int(os.environ.get("AGENT_BARREN_WINDOW", 8))

#: Statuses that mean nothing was learned and nothing was produced.
#: Minutes that have to be left for the loop to hand an early-stopping agent its
#: own list back. The feature exists because a full run finished at 143 minutes of
#: 180, having just written down three things it would try with more time; below a
#: quarter-hour the push would land on finalisation instead, and above it the
#: branch would never fire on a three-hour run.
#:
#: It was used at the only place that reads it and never defined -- in v1, v2 and
#: v3 alike -- so every agent that stopped on its own crashed the loop instead of
#: being asked once. Found by a live run, not by a test; there is now a test.
_PUSHBACK_MINUTES = float(os.environ.get("AGENT_PUSHBACK_MINUTES", 15))

#: Rehearsal only: a directory to receive every exchange untruncated. The console
#: log cuts tool output at 300 characters, which is enough to follow a run and
#: not enough to study one afterwards. Unset on the platform, so nothing is written.
_TRACE_DIR = os.environ.get("AGENT_TRACE_DIR", "")


def _trace(record: dict[str, Any]) -> None:
    if not _TRACE_DIR:
        return
    try:
        os.makedirs(_TRACE_DIR, exist_ok=True)
        with open(os.path.join(_TRACE_DIR, "trace.jsonl"), "a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # a trace is never worth a run

_BARREN_STATUSES = ('"status":"duplicate"', '"status":"invalid_recipe"',
                    '"status":"error"', '"status":"refused"', '"status":"unchanged"',
                    '"code":"unknown_tool"', '"code":"bad_arguments"', '"code":"bad_json"')


def _productive(output: str) -> bool:
    return not any(marker in output for marker in _BARREN_STATUSES)


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

    context_tokens = discover_context(client, model, log)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_prompt(session)},
        {"role": "user", "content": "Solve the task. Begin with get_context."},
    ]
    announced: set[str] = set()
    steps = 0
    stopped = "steps"
    #: Recent tool replies, as productive/fruitless. Judged structurally rather
    #: than trusted to the model: from inside the loop, a refusal it ignores
    #: looks exactly like one it acts on.
    recent: deque[bool] = deque(maxlen=_BARREN_WINDOW)
    #: Asked once, never twice: a second push would be nagging, and the model
    #: has already answered the question.
    pushed_back = False
    #: Shrinks on an overflow and stays shrunk. Restoring the full budget would
    #: just overflow again a step later.
    scale = 1.0
    #: What the run costs. The team shares one daily budget, so the price of an
    #: attempt decides how many attempts there are; unmeasured, it gets guessed.
    spend = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    try:
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
                response = _complete(client, model, _trim(messages, overhead, scale, context_tokens), tools)
            except Exception as err:  # noqa: BLE001
                # A window overflow is recoverable: keep less history and go on.
                # Giving up here would waste whatever time is left in the run.
                if "context" in str(err).lower() and scale > 0.3:
                    scale *= 0.5
                    log(f"[loop] context overflowed; history budget now {scale:.0%} and retrying")
                    session.note(f"context overflowed; history budget reduced to {scale:.0%}")
                    try:
                        response = _complete(client, model, _trim(messages, overhead, scale, context_tokens), tools)
                    except Exception as retry_err:  # noqa: BLE001
                        log(f"[loop] still failing after the retry: {retry_err}")
                        stopped = "context"
                        break
                else:
                    log(f"[loop] the model call failed and could not be retried: {err}")
                    session.note(f"loop aborted at step {steps}: {err}")
                    stopped = "error"
                    break

            usage = getattr(response, "usage", None)
            if usage:
                spend["prompt_tokens"] += usage.prompt_tokens or 0
                spend["completion_tokens"] += usage.completion_tokens or 0
                # Proxies that price the call report it here; others simply do not.
                spend["cost"] += float((usage.model_extra or {}).get("cost") or 0.0)

            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            _trace({"kind": "model", "step": steps, "elapsed": session.clock.elapsed,
                    "usable_remaining": session.clock.usable_remaining, "phase": phase,
                    "usage": usage.model_dump() if usage else None,
                    "message": messages[-1]})

            if message.content:
                log(f"\n--- step {steps} ---\n{message.content[:1500]}")
            if not message.tool_calls:
                # Stopping because the obvious moves ran out is not the same as
                # stopping because the time ran out. Observed in a full run: the
                # agent finished at 143 minutes of 180, having just written down
                # three things it would try with more time. Handing that list back
                # once costs one step and can buy a quarter of the budget.
                idle_minutes = session.clock.usable_remaining / 60
                if idle_minutes > _PUSHBACK_MINUTES and not pushed_back:
                    pushed_back = True
                    log(f"[loop] the agent stopped with {idle_minutes:.0f} min left; asking once")
                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "minutes_left": round(idle_minutes, 1),
                            "note": (
                                "You stopped, but the run has not. If you named anything "
                                "you would try with more time, this is that time."
                            ),
                            "instruction": (
                                "Either run the next experiment now, or reply again with "
                                "nothing to call and the run will finalise."
                            ),
                        }, ensure_ascii=False),
                    })
                    continue
                log("[loop] the agent finished")
                stopped = "finished"
                break

            for call in message.tool_calls:
                name = call.function.name
                # Checked before every call, not once per step. A step can hold several
                # calls, and one of them can be an engine fit or a tuning sweep that
                # runs for half an hour: the clock has to be consulted at the moment
                # the work would start. Observed on a real run: a 900s subprocess was
                # launched with 1.7 usable minutes left, overran, and threw away a
                # finished result worth more than everything else in the ledger.
                if session.clock.must_finalize:
                    log(f"[step {steps}] {name} not started: the reserve has been reached")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({
                        "status": "refused",
                        "problem": "the time reserve was reached before this call could start",
                        "hint": ("Nothing new can be started. Confirm with leaderboard that the "
                                 "best result is the saved one, then stop."),
                        "time": session.clock.summary(),
                    }, ensure_ascii=False)})
                    recent.append(False)
                    continue
                log(f"[step {steps}] {name}({(call.function.arguments or '')[:200]})")
                before = session.clock.usable_remaining
                output = REGISTRY.dispatch(name, call.function.arguments)
                spent = before - session.clock.usable_remaining
                if spent > before and before > 0:
                    # Worth saying out loud: a call that ate its way into the reserve
                    # is the thing that breaks finalisation, and it is invisible unless
                    # someone measures it.
                    log(f"         [warning] {name} ran {spent:.0f}s with {before:.0f}s usable")
                    session.note(f"{name} overran the usable budget by {spent - before:.0f}s")
                log(f"         -> {output[:300]}")
                _trace({"kind": "tool", "step": steps, "name": name,
                        "arguments": call.function.arguments, "seconds": spent,
                        "usable_remaining": session.clock.usable_remaining, "output": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                recent.append(_productive(output))

            barren = recent.count(False)
            if len(recent) == _BARREN_WINDOW and barren >= _BARREN_LIMIT:
                # Persuasion has already been tried -- the refusals carry hints and
                # name the untried axes. When they keep going unheeded, the remaining
                # minutes are worth more to the deterministic pipeline than to another
                # round of the same exchange.
                log(f"[loop] {barren} of the last {_BARREN_WINDOW} calls produced nothing; "
                    f"handing the rest to the autopilot")
                session.note(f"{barren}/{_BARREN_WINDOW} calls fruitless; agent stopped")
                stopped = "no_progress"
                break

    finally:
        # Written in a finally: this line is the only record of what the run
        # cost, and the run needs the number most when it ended badly. Observed
        # on a live run -- a NameError in this loop threw the totals away along
        # with everything else, so the attempt could not be priced afterwards.
        log(f"[loop] {steps} steps, {spend['prompt_tokens']} prompt + "
            f"{spend['completion_tokens']} completion tokens"
            + (f", cost {spend['cost']:.2f}" if spend["cost"] else ""))
        session.spend.update(spend)
        _trace({"kind": "end", "steps": steps, "stopped": stopped, "spend": spend,
                "messages": messages})


    return {
        "steps": steps,
        "stopped": stopped,
        "regime": session.regime,
        "experiments": len(session.ledger),
        "spend": spend,
        "final_message": next(
            (m.get("content") for m in reversed(messages)
             if m.get("role") == "assistant" and m.get("content")),
            "",
        ),
    }
