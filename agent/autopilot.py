"""Running without a language model.

Two reasons this exists. It is the harness that lets the whole system be tested
end to end without an API key. And it is what happens if the model endpoint is
unreachable mid-run: a submission still gets built, from a short fixed sequence
of recipes that are reasonable on any tabular task.

It is deliberately dull. The agent is supposed to beat it.
"""

from __future__ import annotations

from typing import Any

from .session import Session
from .tools.core.modeling import run_recipe, tune_recipe


def _plan(session: Session) -> list[dict[str, Any]]:
    spec, store = session.spec, session.store
    drop = [spec.id_column]
    categorical = [
        c for c in store.X.columns
        if c not in drop and not store.X[c].dtype.kind in "biufc"
    ]

    plan: list[dict[str, Any]] = [
        {"name": "dummy", "drop_columns": drop, "model": {"type": "dummy"}},
        {"name": "lightgbm_default", "drop_columns": drop, "model": {"type": "lightgbm"}},
    ]
    if categorical:
        plan.append({
            "name": "lightgbm_target_encoded", "drop_columns": drop,
            "categorical": {"encode": "target", "rare_threshold": 0.01},
            "model": {"type": "lightgbm"},
        })
    if spec.kind == "regression" and abs(float(store.y.astype("float64").skew())) > 1.5:
        plan.append({
            "name": "lightgbm_log_target", "drop_columns": drop,
            "target_transform": "log1p", "model": {"type": "lightgbm"},
        })
    if spec.is_classification:
        share = store.y.value_counts(normalize=True)
        if float(share.min()) < 0.05:
            plan.append({
                "name": "lightgbm_balanced", "drop_columns": drop,
                "class_weight": "balanced", "model": {"type": "lightgbm"},
            })
    plan.append({"name": "catboost_default", "drop_columns": drop, "model": {"type": "catboost"}})
    return plan


def run(session: Session, log=print, tune: bool = True) -> dict[str, Any]:
    """Work through the fixed plan, stopping when the clock says to."""
    executed: list[str] = []

    for recipe in _plan(session):
        if session.clock.must_finalize:
            log("[autopilot] out of usable time; stopping")
            break
        log(f"[autopilot] {recipe['name']}")
        outcome = run_recipe(recipe, note="autopilot")
        executed.append(recipe["name"])
        if outcome.get("status") == "ok":
            log(f"           {outcome['metric']}={outcome['score']} — {outcome['verdict']}")
        else:
            log(f"           {outcome.get('status')}: {outcome.get('problem') or outcome.get('errors')}")

    best = session.ledger.best()
    if tune and best is not None and best.detail.get("recipe") and not session.clock.must_finalize:
        budget = int(session.clock.usable_remaining * 0.5)
        if budget >= 120:
            log(f"[autopilot] tuning {best.id} for up to {budget}s")
            outcome = tune_recipe(best.detail["recipe"], time_budget_s=budget)
            if outcome.get("status") == "ok":
                log(f"           {outcome['metric']}={outcome['score']} — {outcome['verdict']}")

    final = session.ledger.best()
    return {
        "executed": executed,
        "experiments": len(session.ledger),
        "best": final.row() if final else None,
    }
