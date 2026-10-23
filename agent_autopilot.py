"""Running without a language model.

Two reasons this exists. It is the harness that lets the whole system be tested
end to end without an API key. And it is what happens if the model endpoint is
unreachable mid-run: a submission still gets built, from a short fixed sequence
of recipes that are reasonable on any tabular task.

It is deliberately dull. The agent is supposed to beat it.
"""

from __future__ import annotations

from typing import Any

from agent_session import Session
from agent_tool_modeling import run_recipe, tune_recipe


def _inherited(session: Session) -> dict[str, Any]:
    """Settings the floor already established by measurement, not by guesswork.

    The floor fits more than one candidate and records what each scored. Reusing
    its conclusion costs nothing and avoids the plan's worst habit: on the
    regression task every candidate but one was fitted on the untransformed
    target, and an untransformed target measured 54.5 RMSE against 48.8 for a
    transformed one -- on every fold. Four of six slots were spent proving
    something the floor had already disproved.
    """
    settings: dict[str, Any] = {}
    for entry in session.ledger.entries:
        if entry.source != "baseline":
            continue
        recipe = (entry.detail or {}).get("recipe") or {}
        # sample_weight joins the list for the same reason the other two are on
        # it: the floor now measures it (46.05 weighted against 48.46 plain on the
        # regression task) and the autopilot has six slots, which is too few to
        # spend re-deciding a question that has already been answered on the same
        # folds. It is a top-level field and model-agnostic, unlike a loss
        # objective, so it transfers to every candidate without translation.
        # The entity history is carried too: when the floor kept it, it was the
        # largest single gain on the task, and a candidate without it starts
        # several RMSE behind before its own idea is even measured.
        for field in ("target_transform", "class_weight", "sample_weight", "history"):
            value = recipe.get(field)
            if value and value != "none":
                settings[field] = value
    return settings


def _plan(session: Session) -> list[dict[str, Any]]:
    spec, store = session.spec, session.store
    drop = [spec.id_column]
    categorical = [
        c for c in store.X.columns
        if c not in drop and not store.X[c].dtype.kind in "biufc"
    ]
    carried = _inherited(session)
    forecast = getattr(store.plan, "name", "") == "forward_chain"

    def recipe(name: str, **extra: Any) -> dict[str, Any]:
        return {"name": name, "drop_columns": drop, **carried, **extra}

    plan: list[dict[str, Any]] = [
        # Kept as a reference point even though the floor has already beaten it:
        # knowing how far above a constant the run sits is worth nine seconds.
        {"name": "dummy", "drop_columns": drop, "model": {"type": "dummy"}},
        recipe("lightgbm_default", model={"type": "lightgbm"}),
    ]
    if categorical:
        plan.append(recipe(
            "lightgbm_target_encoded",
            categorical={"encode": "target", "rare_threshold": 0.01},
            model={"type": "lightgbm"},
        ))
    if spec.kind == "regression" and not carried.get("target_transform"):
        # Only when the floor did not already settle this -- otherwise it is
        # carried into every candidate above and a separate slot is waste.
        if abs(float(store.y.astype("float64").skew())) > 1.5:
            plan.append(recipe("lightgbm_log_target", target_transform="log1p",
                               model={"type": "lightgbm"}))
    if spec.is_classification and not carried.get("class_weight"):
        share = store.y.value_counts(normalize=True)
        if float(share.min()) < 0.05:
            plan.append(recipe("lightgbm_balanced", class_weight="balanced",
                               model={"type": "lightgbm"}))
    if forecast:
        # Extrapolation rewards less capacity: measured on the forecast folds, 300
        # trees beat 500 and the default leaf count beat a wider one. Worth one slot
        # on a task where the test set is a period the model has never seen.
        plan.append(recipe("lightgbm_restrained",
                           model={"type": "lightgbm",
                                  "params": {"n_estimators": 250, "num_leaves": 31,
                                             "min_child_samples": 50}}))
    # CatBoost earns its slot; XGBoost was tried here and removed. The argument for
    # adding it was diversity for the averaging step, and the argument was wrong
    # twice over. Measured on the forecast folds it scored 81.2 RMSE against 48.5
    # for LightGBM -- worse than a constant, with one period at 105.7, because it
    # does not extrapolate past the training periods at all. And the blend takes
    # the leading candidates by score, so a weak model cannot contribute diversity
    # to it by construction: only a candidate that is both different *and* good can.
    plan.append(recipe("catboost_default", model={"type": "catboost"}))
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

    # Finish by averaging the strongest candidates rather than trusting the single
    # best. The reason is measured, not stylistic: on a forecast task model
    # rankings do not survive the move to the next period -- rank correlation
    # between adjacent periods fell to -0.04 here -- so "the best on validation" is
    # a draw from a spread, not a choice. Averaging beat the typical single
    # candidate on all three backtested periods, and beat both selection rules on
    # the period most like the test set.
    #
    # The ladder still decides: the blend is scored on the same folds and only
    # replaces the submission if it wins. A blend that loses costs seconds.
    blended = _blend(session, log)

    final = session.ledger.best()
    return {
        "executed": executed,
        "blended": blended,
        "experiments": len(session.ledger),
        "best": final.row() if final else None,
    }


def _blend(session: Session, log=print) -> dict[str, Any] | None:
    """Average the leading candidates, if there are enough of them to average."""
    if session.clock.must_finalize:
        return None
    try:
        from agent_registry import REGISTRY

        if "stack_experiments" not in REGISTRY.tools:
            return None  # the mod is absent or disabled; nothing to do
        ranked = [
            e for e in session.ledger.entries
            if not e.tainted and (e.detail or {}).get("predictions")
            and "error" not in (e.detail or {}).get("predictions", {})
        ]
        if len(ranked) < 3:
            return None
        ranked.sort(key=lambda e: e.score if session.spec.direction == "maximize" else -e.score,
                    reverse=True)
        # The weakest candidates drag an average down, and a constant baseline
        # drags it a long way: take the leaders, not everything.
        chosen = [e.id for e in ranked[:5]]
        log(f"[autopilot] averaging {chosen}")
        import json

        outcome = json.loads(REGISTRY.dispatch(
            "stack_experiments",
            {"experiment_ids": chosen, "meta": "mean",
             "note": "autopilot: average of the leading candidates"},
        ))
        if outcome.get("status") == "ok":
            log(f"           {outcome['metric']}={outcome['score']} — {outcome['verdict']}")
        else:
            log(f"           {outcome.get('status')}: {outcome.get('problem')}")
        return outcome
    except Exception as err:  # noqa: BLE001 - a failed blend must not cost the run
        log(f"[autopilot] averaging failed: {type(err).__name__}: {err}")
        return None
