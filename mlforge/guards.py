"""
Guard rails.

Three tiers, chosen deliberately:

* **block**      - the action is invalid whatever the intent (spent holdout budget,
                   random folds on grouped data, a feature built from the target).
                   There is no override; the server simply refuses.
* **quarantine** - the action is *probably* wrong but could be legitimate (a flagged
                   leak column the agent insists is available at prediction time).
                   Refused until the agent passes `override_guards` together with a
                   written justification, which is then permanently attached to the
                   run and reproduced in the leaderboard and the final report.
* **taint**      - applied after the fact to results that are too good to believe.
                   The run is excluded from `best_run` until it is explained.

The point is not to make the agent obedient. It is to make silent self-deception
impossible: anything questionable either does not run, or runs with its reasoning
on the record.
"""
from __future__ import annotations

import copy
from typing import Any

from .schema import LOWER_IS_BETTER

# A score at or beyond this is implausible on a genuinely hard problem.
IMPLAUSIBLE_HIGH = {
    "roc_auc": 0.995, "pr_auc": 0.995, "accuracy": 0.995, "balanced_accuracy": 0.995,
    "f1": 0.995, "macro_f1": 0.995, "r2": 0.999,
}
IMPLAUSIBLE_LOW = {"logloss": 0.01, "mape": 0.005}


class GuardBlock(Exception):
    """Refusal carrying machine-readable detail, so the agent can act on it."""

    def __init__(self, code: str, problem: str, hint: str, overridable: bool = False,
                 fix: dict | None = None):
        super().__init__(problem)
        self.code, self.problem, self.hint, self.overridable = code, problem, hint, overridable
        # A ready-to-send corrected recipe. Telling a small model what is wrong is
        # not enough -- observed live, a 1.7B took ten consecutive blocks on the same
        # recipe while the hint spelled out the fix in words. Handing it the repaired
        # object converts a dead end into one more call.
        self.fix = fix

    def payload(self) -> dict[str, Any]:
        return {
            "status": "blocked",
            "guard": self.code,
            "problem": self.problem,
            "hint": self.hint,
            "overridable": self.overridable,
            **({"corrected_recipe": self.fix,
                "next_step": "Send run_experiment again with corrected_recipe exactly as given."}
               if self.fix else {}),
            **({"how_to_override":
                f'Pass override_guards=["{self.code}"] together with a justification '
                f'explaining why this is correct. It is recorded permanently on the run.'}
               if self.overridable else
               {"how_to_override": "This guard cannot be overridden."}),
        }


def _critical_leaks(dataset) -> list[str]:
    return [i["column"] for i in dataset.issues
            if i["kind"] == "target_leakage" and i["severity"] == "critical" and i["column"]]


def check_experiment(dataset, recipe: dict, overrides: list[str], justification: str) -> list[str]:
    """Run every pre-flight gate. Returns the acknowledgements to record on the run."""
    acknowledged: list[str] = []
    dropped = set(recipe.get("drop_columns") or [])

    # --- block: a feature built from the target ---------------------------
    for f in recipe.get("features") or []:
        for arg in ("a", "b"):
            if f.get(arg) == dataset.target:
                raise GuardBlock(
                    "TARGET_IN_FEATURE",
                    f"feature '{f.get('name')}' is built from the target column '{dataset.target}'",
                    "A feature derived from the target is pure leakage. Remove it.",
                    overridable=False,
                )

    # --- block: the declared grouping is being ignored ---------------------
    if dataset.group_column and recipe["validation"]["scheme"] not in ("group_kfold",):
        fixed = copy.deepcopy(recipe)
        fixed["validation"] = {**fixed.get("validation", {}), "scheme": "group_kfold",
                               "group_column": dataset.group_column}
        raise GuardBlock(
            "GROUP_LEAKAGE",
            f"this dataset declares group_column='{dataset.group_column}', but validation.scheme "
            f"is '{recipe['validation']['scheme']}'",
            f"Rows sharing a group must not be split across folds, or the score is inflated. "
            f'Set validation = {{"scheme": "group_kfold", "group_column": "{dataset.group_column}"}}.',
            overridable=False,
        )

    # --- block: the declared time order is being ignored -------------------
    if dataset.time_column and recipe["validation"]["scheme"] != "time_series":
        fixed = copy.deepcopy(recipe)
        fixed["validation"] = {**fixed.get("validation", {}), "scheme": "time_series"}
        raise GuardBlock(
            "TIME_LEAKAGE",
            f"this dataset declares time_column='{dataset.time_column}', but validation.scheme "
            f"is '{recipe['validation']['scheme']}'",
            "Random folds let the model train on the future to predict the past. "
            'Set validation = {"scheme": "time_series", "folds": 5}.',
            overridable=False,
        )

    # --- block: nothing left to learn from ---------------------------------
    remaining = [c for c in dataset.X.columns if c not in dropped]
    if not remaining:
        raise GuardBlock(
            "NO_FEATURES_LEFT", "drop_columns removes every feature",
            "Keep at least one predictor.", overridable=False,
        )

    # --- quarantine: a flagged leak is still in play -----------------------
    # The dummy baseline ignores features entirely, so a leak cannot reach it.
    # Blocking it would break the very first step of the prescribed workflow.
    is_dummy = recipe["model"]["type"] == "dummy"
    live_leaks = [] if is_dummy else [c for c in _critical_leaks(dataset) if c not in dropped]
    if live_leaks:
        code = "LEAK_NOT_DROPPED"
        trusted = set(getattr(dataset, "trusted_columns", []) or [])
        untrusted = [c for c in live_leaks if c not in trusted]

        # Whether a column is knowable at prediction time is a domain fact about the
        # business process, not something a model can establish from the data. So the
        # agent may not grant itself this exemption. Observed live: a 1.7B model, once
        # blocked, invented the justification "this column is known at prediction time"
        # purely to get through the gate. Only the operator, at load_dataset time, can
        # put a column on the trusted list.
        if untrusted:
            fixed = copy.deepcopy(recipe)
            fixed["drop_columns"] = sorted(set(fixed.get("drop_columns") or []) | set(untrusted))
            fixed["name"] = (fixed.get("name") or "recipe") + "_no_leak"
            fixed["notes"] = f"Dropped leaking column(s) {untrusted}."
            raise GuardBlock(
                code,
                f"column(s) {untrusted} were flagged as critical target leakage and are still "
                f"in the feature set",
                f"Add them to drop_columns. A model trained on a leak scores near-perfectly in "
                f"validation and fails completely in production. Whether such a column really is "
                f"available at prediction time is a question about the business process, which you "
                f"cannot answer from the data -- so you cannot waive this yourself. If it genuinely "
                f"is available, the operator must reload the dataset with "
                f"trusted_columns={untrusted}.",
                overridable=False,
                fix=fixed,
            )

        if code not in overrides:
            raise GuardBlock(
                code,
                f"column(s) {live_leaks} are flagged as leakage but the operator marked them trusted",
                "Confirm you intend to use them by overriding with a justification.",
                overridable=True,
            )
        _require_justification(code, justification)
        acknowledged.append(f"{code}: kept operator-trusted {live_leaks} -- {justification}")

    return acknowledged


def _require_justification(code: str, justification: str) -> None:
    if len((justification or "").strip()) < 20:
        raise GuardBlock(
            "JUSTIFICATION_REQUIRED",
            f"override of '{code}' requires a written justification of at least 20 characters",
            "Explain why this is correct -- e.g. why the flagged column really is known at "
            "prediction time. The text is attached to the run permanently.",
            overridable=False,
        )


def check_final_evaluate(dataset, run, overrides: list[str]) -> None:
    from .session import HOLDOUT_BUDGET

    if dataset.holdout_used >= HOLDOUT_BUDGET:
        raise GuardBlock(
            "HOLDOUT_EXHAUSTED",
            f"the holdout budget for '{dataset.id}' is spent ({dataset.holdout_used}/{HOLDOUT_BUDGET} calls used)",
            "The holdout gives one unbiased estimate, not a second validation set. "
            "Report the result you already have.",
            overridable=False,
        )
    if run.tainted and "TAINTED_RUN" not in overrides:
        raise GuardBlock(
            "TAINTED_RUN",
            f"run '{run.id}' is flagged as implausible ({run.taint_reason})",
            "Investigate the cause before spending holdout budget on it -- an implausible CV score "
            "almost always means leakage. Fix it, or override with a justification.",
            overridable=True,
        )


def taint_check(score: float, metric: str, task: str) -> str | None:
    """Flag results that are too good to be true. Returns a reason, or None."""
    if metric in LOWER_IS_BETTER:
        thr = IMPLAUSIBLE_LOW.get(metric)
        if thr is not None and score <= thr:
            return f"{metric}={score:.5f} is at the theoretical floor ({thr})"
        return None
    thr = IMPLAUSIBLE_HIGH.get(metric)
    if thr is not None and score >= thr:
        return (f"{metric}={score:.5f} exceeds the plausibility threshold ({thr}); "
                f"on a genuinely hard problem this nearly always indicates target leakage")
    return None
