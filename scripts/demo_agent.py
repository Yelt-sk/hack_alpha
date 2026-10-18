#!/usr/bin/env python
"""
A scripted agent run: exercises every tool in the order the system prompt
prescribes, with no LLM in the loop.

Its job is to prove the server works end to end and to give the live demo a
deterministic fallback. The real agent makes these same calls -- it just
chooses them itself.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import server as S  # noqa: E402


def show(title: str, payload) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(json.dumps(payload, indent=2, default=str)[:2600])


def main() -> None:
    show("1. load_dataset", S.load_dataset("synthetic_churn"))

    prof = S.profile_dataset("synthetic_churn")
    show("2. profile_dataset (issues only)", {"n_rows": prof["n_rows"], "issues": prof["issues"]})

    show("3. search_knowledge", S.search_knowledge("my model scores 1.0 roc auc what is wrong", top_k=1))

    show("4. dummy baseline", S.run_experiment("synthetic_churn",
         {"name": "baseline_dummy", "notes": "Floor.", "model": {"type": "dummy"}}))

    show("5. GUARD: lightgbm while keeping the flagged leak", S.run_experiment(
        "synthetic_churn",
        {"name": "lgbm_with_leak", "notes": "Naive first attempt.",
         "drop_columns": ["customer_id"], "model": {"type": "lightgbm"}}))

    # The agent cannot waive this itself. Only the operator can, by declaring the
    # column trusted at load time -- so the demonstration reloads the same data
    # under a second id with that declaration.
    show("6a. agent tries to self-authorise (refused)", S.run_experiment(
        "synthetic_churn",
        {"name": "lgbm_self_authorised", "drop_columns": ["customer_id"],
         "model": {"type": "lightgbm"}},
        override_guards=["LEAK_NOT_DROPPED"],
        justification="I believe this column is available at prediction time."))

    show("6b. operator declares the column trusted", S.load_dataset(
        "synthetic_churn", dataset_id="churn_operator_trusted",
        trusted_columns=["support_ticket_after_cancel"]))

    S.run_experiment("churn_operator_trusted", {"name": "dummy", "model": {"type": "dummy"}})
    show("6c. now the override is allowed, and recorded", S.run_experiment(
        "churn_operator_trusted",
        {"name": "lgbm_with_leak_forced", "notes": "Deliberately keeping the leak to show what happens.",
         "drop_columns": ["customer_id"], "model": {"type": "lightgbm"}},
        override_guards=["LEAK_NOT_DROPPED"],
        justification="Operator declared this column trusted; demonstrating the failure mode."))

    clean = {"name": "lgbm_clean", "notes": "Dropped the ID and the post-outcome column.",
             "drop_columns": ["customer_id", "support_ticket_after_cancel"],
             "model": {"type": "lightgbm"}}
    show("7. clean lightgbm", S.run_experiment("synthetic_churn", clean))

    eng = dict(clean, name="lgbm_engineered",
               notes="Added date parts, a charge-per-month ratio and a missingness counter.",
               features=[{"op": "datetime_parts", "name": "signup", "a": "signup_date"},
                         {"op": "ratio", "name": "charge_per_month", "a": "total_charges", "b": "tenure_months"},
                         {"op": "missing_count", "name": "n_missing"}])
    show("8. feature engineering", S.run_experiment("synthetic_churn", eng))

    reg = dict(eng, name="lgbm_regularised", notes="Previous run showed a large train/validation gap.",
               model={"type": "lightgbm",
                      "params": {"num_leaves": 15, "min_child_samples": 50, "subsample": 0.8,
                                 "colsample_bytree": 0.8, "reg_lambda": 5.0,
                                 "learning_rate": 0.03, "n_estimators": 400}})
    show("9. regularised", S.run_experiment("synthetic_churn", reg))

    show("10. tune_hyperparams", S.tune_hyperparams("synthetic_churn", reg, n_trials=12, time_budget_seconds=90))

    lb = S.leaderboard("synthetic_churn")
    show("11. leaderboard", lb)

    best = lb["best_run_id"]
    show("12. inspect_run (best)", S.inspect_run(best, top_k=8))
    show("13. final_evaluate", S.final_evaluate("synthetic_churn", best))
    show("14. export_solution", S.export_solution(best))

    # The guard must refuse a second holdout call on a different run.
    others = [r["run_id"] for r in lb["runs"] if r["run_id"] != best]
    if others:
        show("15. holdout budget still enforced", S.final_evaluate("synthetic_churn", others[0]))


if __name__ == "__main__":
    main()
