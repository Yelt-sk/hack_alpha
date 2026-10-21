"""Orientation tools: where the run stands, and which regime it is in."""

from __future__ import annotations

from typing import Any, Literal

from agent_session import session, summary
from agent_registry import tool


@tool
def get_context() -> dict[str, Any]:
    """Where the run stands: task, metric, time left, best result so far, saved submission.

    Call this first, and again whenever you are unsure what has already happened.
    It is cheap and it is the only tool that tells you how much time remains.
    """
    return summary()


@tool
def set_strategy(regime: Literal["standard", "custom"], rationale: str) -> dict[str, Any]:
    """Declare how you intend to work, and why. Do this early, after profiling the data.

    Choose `standard` for an ordinary tabular task: you describe experiments as
    recipes and proven algorithms execute them. This is faster, reproducible, and
    what you should default to.

    Choose `custom` only when the task genuinely needs code a recipe cannot
    express -- an unusual target, a bespoke validation requirement, a structure
    the recipe vocabulary does not cover. You keep `run_python` either way; the
    declaration sets which tools are offered and is recorded for the final report.

    Args:
        regime: The working mode you are adopting.
        rationale: One or two sentences on what in the data led you to choose it.
    """
    current = session()
    if len(rationale.strip()) < 20:
        return {
            "status": "error",
            "problem": "rationale is too short",
            "hint": "Say what about this task made the regime appropriate; it goes into the report.",
        }
    current.regime, current.regime_rationale = regime, rationale.strip()
    current.note(f"regime={regime}: {rationale.strip()[:120]}")
    return {"status": "ok", "regime": regime, "note": "Tool list updated for this regime."}


@tool
def leaderboard() -> dict[str, Any]:
    """Every scored experiment so far, ranked, with the fold-noise band.

    Use it before deciding what to try next, and before finalising, to check that
    the result you are about to submit is genuinely the best one.
    """
    current = session()
    best = current.ledger.best()
    return {
        "metric": current.spec.metric,
        "direction": current.spec.direction,
        "n_experiments": len(current.ledger),
        "best_id": best.id if best else None,
        "runs": current.ledger.rows(),
        "submission": current.submissions.describe(),
        "time": current.clock.summary(),
    }
