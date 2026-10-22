"""What our results look like once the folds match how the test set was built.

Every number this project reported before ``agent_split`` existed was measured on
random folds. On the regression task that meant validating interpolation on a
task that is extrapolation; on classification it meant scoring 43.8% of each
validation fold against rows the model had already memorised. This script runs
the same recipes under both schemes and prints the two columns side by side, so
the size of the correction is visible rather than argued about.

It is deliberately not an agent run: the point is a controlled comparison of
identical recipes, not a search.

    python scripts/remeasure.py                # both tasks
    python scripts/remeasure.py --task reg     # one of reg / cls
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import agent_split  # noqa: E402
from agent_metrics import score  # noqa: E402
from agent_recipe_pipeline import evaluate  # noqa: E402
from agent_recipe_schema import validate  # noqa: E402
from agent_state import DataStore  # noqa: E402
from agent_task import load_task  # noqa: E402

TASKS = {
    "reg": ROOT / "data" / "Регрессия" / "task.md",
    "cls": ROOT / "data" / "Многоклассовая классификация" / "task.md",
}

#: The recipes this project actually relied on, plus the ones the old scheme
#: dismissed. Both lists matter: the question is not only "how much worse are we"
#: but "did the ranking change", and it did.
RECIPES: dict[str, list[dict]] = {
    "regression": [
        {"name": "dummy", "model": {"type": "dummy"}},
        {"name": "lightgbm_small", "model": {"type": "lightgbm", "params": {"n_estimators": 300}}},
        {"name": "lightgbm_default", "model": {"type": "lightgbm"}},
        {"name": "lightgbm_log_target", "target_transform": "log1p",
         "model": {"type": "lightgbm"}},
        {"name": "lightgbm_log_small", "target_transform": "log1p",
         "model": {"type": "lightgbm", "params": {"n_estimators": 300, "num_leaves": 31}}},
        {"name": "catboost_default", "model": {"type": "catboost"}},
    ],
    "classification": [
        {"name": "dummy", "model": {"type": "dummy"}},
        {"name": "lightgbm_plain", "model": {"type": "lightgbm"}},
        {"name": "lightgbm_balanced", "class_weight": "balanced", "model": {"type": "lightgbm"}},
        {"name": "lightgbm_sqrt", "class_weight": "sqrt", "model": {"type": "lightgbm"}},
        {"name": "v2_floor", "class_weight": "sqrt",
         "postprocess": {"thresholds": "per_class"},
         "model": {"type": "lightgbm", "params": {"n_estimators": 300}}},
        {"name": "sqrt_thresholds", "class_weight": "sqrt",
         "postprocess": {"thresholds": "per_class"}, "model": {"type": "lightgbm"}},
    ],
}


def _quiet(*_: object) -> None:
    pass


def _plain_plan(store: DataStore) -> agent_split.FoldPlan:
    """The scheme this project used before: rows split at random.

    Rebuilt here rather than left behind, because "how wrong were we" is only
    answerable by running both.
    """
    splits = agent_split._plain(
        store.y, 5, store.spec.is_classification, store.seed
    )
    return agent_split._as_plan("plain(old)", splits, len(store.train),
                                "the scheme used before agent_split", {})


def _run(store: DataStore, recipes: list[dict]) -> dict[str, tuple[float, list[float], float]]:
    out: dict[str, tuple[float, list[float], float]] = {}
    for raw in recipes:
        started = time.perf_counter()
        recipe, errors, _ = validate(raw, store.spec.kind, list(store.X.columns))
        if errors:
            print(f"    {raw['name']}: invalid -- {errors}")
            continue
        try:
            result = evaluate(recipe, store)
        except Exception as err:  # noqa: BLE001
            print(f"    {raw['name']}: failed -- {type(err).__name__}: {err}")
            continue
        keep = store.evaluable
        primary = score(store.spec.metric, store.y[keep], result["oof"][keep],
                        result["oof_proba"][keep] if result["oof_proba"] is not None else None,
                        result["classes"])
        per_fold = []
        for fold in range(store.n_splits):
            mask = store.fold_id == fold
            try:
                per_fold.append(score(
                    store.spec.metric, store.y[mask], result["oof"][mask],
                    result["oof_proba"][mask] if result["oof_proba"] is not None else None,
                    result["classes"]))
            except Exception:  # noqa: BLE001
                pass
        out[recipe["name"]] = (primary, per_fold, time.perf_counter() - started)
        print(f"    {recipe['name']:24s} {primary:9.4f}   [{time.perf_counter()-started:5.0f}s]")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), action="append")
    args = parser.parse_args()

    for key in args.task or sorted(TASKS):
        spec = load_task(TASKS[key])
        cache = ROOT / ".remeasure_cache" / key
        print(f"\n{'=' * 78}\n{spec.kind.upper()}  --  {spec.metric} ({spec.direction})\n{'=' * 78}")

        store = DataStore(spec, cache, log=lambda m: print(f"  {m}"))
        honest_plan = store.plan
        print(f"\n  honest scheme: {honest_plan.name}, {honest_plan.n_splits} splits, "
              f"{int(honest_plan.evaluable.sum())} of {len(store.train)} rows validated")
        print("  --- honest folds ---")
        honest = _run(store, RECIPES[spec.kind if spec.kind == "regression" else "classification"])

        # Swap in the old scheme on the same loaded data, so nothing else differs.
        store.plan = _plain_plan(store)
        store.fold_id = store.plan.fold_id
        store.n_splits = store.plan.n_splits
        store.evaluable = store.plan.evaluable
        print("  --- old folds (random) ---")
        old = _run(store, RECIPES[spec.kind if spec.kind == "regression" else "classification"])

        print(f"\n  {'recipe':24s} {'old (random)':>14s} {'honest':>12s} {'difference':>12s}")
        print(f"  {'-' * 64}")
        for name in honest:
            if name not in old:
                continue
            o, h = old[name][0], honest[name][0]
            print(f"  {name:24s} {o:14.4f} {h:12.4f} {h - o:+12.4f}")
        best_old = min(old, key=lambda n: old[n][0] * (1 if spec.direction == "minimize" else -1))
        best_new = min(honest, key=lambda n: honest[n][0] * (1 if spec.direction == "minimize" else -1))
        print(f"\n  best under old folds   : {best_old} ({old[best_old][0]:.4f})")
        print(f"  best under honest folds: {best_new} ({honest[best_new][0]:.4f})")
        if best_old != best_new:
            print(f"  >>> the ranking changed: fixing the folds changed which recipe wins")


if __name__ == "__main__":
    main()
