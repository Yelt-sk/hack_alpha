"""Measure AutoGluon and LightAutoML against our own recipes, on honest folds.

Both engines are in the organisers' ``requirements.txt`` and neither had ever been
run. The reason to run them now rather than earlier is that until the folds matched
how the test set was built, a strong engine would simply have optimised the wrong
objective more thoroughly.

**The engines must be handed the right kind of split, and the two schemes are not
the same shape.** AutoGluon's ``groups=`` and LightAutoML's ``FoldsRole`` both mean
*K-fold bagging*: hold out one fold, train on all the others. That is exactly our
``grouped`` scheme, so for classification the fold column can be passed straight
through. It is exactly wrong for ``forward_chain``, where training on "all the
other folds" means training on later periods than the one being validated -- the
leak this project just spent a day removing. So a forecast is measured by fitting
the engine once per origin, on the earlier periods only.

Scored with our own metric on our own evaluable rows, so the numbers sit in the
same table as every recipe result.

    python scripts/bench_engines.py --task reg --engine autogluon --budget 600
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent_metrics import score  # noqa: E402
from agent_state import DataStore  # noqa: E402
from agent_task import load_task  # noqa: E402

TASKS = {
    "reg": ROOT / "data" / "Регрессия" / "task.md",
    "cls": ROOT / "data" / "Многоклассовая классификация" / "task.md",
}

#: Our metric names to each engine's. Nothing clever: the engines optimise what
#: they are told to, and telling them the wrong thing is the commonest way a
#: benchmark like this lies.
AG_METRIC = {"rmse": "root_mean_squared_error", "mae": "mean_absolute_error",
             "r2": "r2", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted",
             "accuracy": "accuracy", "roc_auc": "roc_auc", "logloss": "log_loss"}

WORK = Path(os.environ.get("BENCH_WORK", "/tmp/bench_engines"))
N_CPUS = int(os.environ.get("BENCH_CPUS", 4))


def _fresh(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# AutoGluon
# ---------------------------------------------------------------------------


#: Set by whichever engine ran last, so the reporting code can use the same class
#: order the predictions are in. A module global because this is a one-shot script.
_LAST_CLASSES: list | None = None


def autogluon_oof(store: DataStore, budget: int, preset: str,
                  log_target: bool = False) -> tuple[np.ndarray, np.ndarray | None, float]:
    from autogluon.tabular import TabularPredictor

    spec = store.spec
    target, idc = spec.target_column, spec.id_column
    frame = store.train.drop(columns=[c for c in (idc,) if c in store.train.columns])
    # The engine does not try a target transform of its own, and on this task that
    # is worth more than anything the engine does: log1p took a plain LightGBM from
    # 52.45 to 49.13 RMSE on honest folds. Whether the two gains stack is the point
    # of the comparison, so it has to be possible to ask for both.
    if log_target:
        frame = frame.copy()
        frame[target] = np.log1p(frame[target])
    metric = AG_METRIC.get(spec.metric, None)
    started = time.perf_counter()

    classification = spec.is_classification
    oof = np.full(len(frame), None, dtype=object) if classification else np.full(len(frame), np.nan)
    proba: np.ndarray | None = None

    if store.plan.name == "forward_chain":
        # One fit per origin, trained only on earlier periods. Budget is split
        # between them so the whole measurement still costs what was asked.
        per = max(60, budget // max(1, store.plan.n_splits))
        for index, (train_idx, val_idx) in enumerate(store.splits()):
            predictor = TabularPredictor(
                label=target, eval_metric=metric, problem_type=None,
                path=str(_fresh(WORK / f"ag_fc_{index}")), verbosity=1,
            ).fit(frame.iloc[train_idx], presets=preset, time_limit=per, num_cpus=N_CPUS)
            predicted = np.asarray(predictor.predict(frame.iloc[val_idx].drop(columns=[target])))
            oof[val_idx] = np.expm1(predicted) if log_target else predicted
            if classification:
                p = predictor.predict_proba(frame.iloc[val_idx].drop(columns=[target]))
                if proba is None:
                    proba = np.zeros((len(frame), p.shape[1]))
                proba[val_idx] = p.to_numpy()
            print(f"      origin {index + 1}/{store.plan.n_splits} done "
                  f"[{time.perf_counter() - started:.0f}s]")
        return oof, proba, time.perf_counter() - started

    # K-fold bagging on our own fold assignment: this is what `groups` is for.
    with_fold = frame.copy()
    with_fold["__fold"] = store.fold_id
    # `groups` belongs to the constructor, not to .fit().
    predictor = TabularPredictor(
        label=target, eval_metric=metric, problem_type=None, groups="__fold",
        path=str(_fresh(WORK / "ag_kfold")), verbosity=1,
    ).fit(with_fold, presets=preset, time_limit=budget, num_cpus=N_CPUS)

    if classification:
        frame_proba = predictor.predict_proba_oof()
        proba = frame_proba.to_numpy()
        classes = list(frame_proba.columns)
        oof = np.asarray(classes, dtype=object)[proba.argmax(1)]
    else:
        raw = np.asarray(predictor.predict_oof())
        oof = np.expm1(raw) if log_target else raw
    return oof, proba, time.perf_counter() - started


# ---------------------------------------------------------------------------
# LightAutoML
# ---------------------------------------------------------------------------


def _lama_classes(model, fallback: list) -> list:
    """The class order LightAutoML actually used, read from the model.

    Not ``sorted(unique(y))``. Its reader assigns indices in its own order --
    measured on a three-class toy problem it produced ``{'mid': 0, 'alpha': 1,
    'zebra': 2}`` where alphabetical order would be alpha, mid, zebra. Assuming
    the alphabetical order silently permutes every prediction: it turned a
    classification benchmark into macro F1 0.17, which looks like a weak engine
    and is in fact a scrambled label mapping.
    """
    mapping = getattr(getattr(model, "reader", None), "class_mapping", None)
    if not mapping:
        return list(fallback)
    return [label for label, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def lightautoml_oof(store: DataStore, budget: int, preset: str) -> tuple[np.ndarray, np.ndarray | None, float]:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    spec = store.spec
    target, idc = spec.target_column, spec.id_column
    classification = spec.is_classification
    kind = {"regression": "reg", "binary": "binary", "multiclass": "multiclass"}[spec.kind]
    started = time.perf_counter()

    frame = store.train.copy()
    oof = np.full(len(frame), None, dtype=object) if classification else np.full(len(frame), np.nan)
    proba: np.ndarray | None = None
    classes = sorted(pd.unique(store.y.astype(str))) if classification else None

    def automl() -> TabularAutoML:
        return TabularAutoML(task=Task(kind), timeout=budget, cpu_limit=N_CPUS,
                             reader_params={"n_jobs": N_CPUS, "random_state": 42})

    if store.plan.name == "forward_chain":
        per = max(60, budget // max(1, store.plan.n_splits))
        for index, (train_idx, val_idx) in enumerate(store.splits()):
            model = TabularAutoML(task=Task(kind), timeout=per, cpu_limit=N_CPUS,
                                  reader_params={"n_jobs": N_CPUS, "random_state": 42})
            model.fit_predict(frame.iloc[train_idx],
                              roles={"target": target, "drop": [idc]}, verbose=0)
            if classification:
                classes = _lama_classes(model, classes)
            predicted = model.predict(frame.iloc[val_idx]).data
            oof[val_idx] = predicted.reshape(-1) if not classification else \
                np.asarray(classes, dtype=object)[predicted.argmax(1)]
            print(f"      origin {index + 1}/{store.plan.n_splits} done "
                  f"[{time.perf_counter() - started:.0f}s]")
        return oof, proba, time.perf_counter() - started

    with_fold = frame.copy()
    with_fold["__fold"] = store.fold_id
    model = automl()
    out = model.fit_predict(
        with_fold, roles={"target": target, "folds": "__fold", "drop": [idc]}, verbose=0
    ).data
    if classification:
        classes = _lama_classes(model, classes)
    if classification:
        proba = out
        oof = np.asarray(classes, dtype=object)[out.argmax(1)]
    else:
        oof = out.reshape(-1)
    return oof, proba, time.perf_counter() - started


def _call(engine: str, store: DataStore, budget: int, preset: str, log_target: bool):
    if engine == "autogluon":
        return autogluon_oof(store, budget, preset, log_target)
    if log_target:
        raise SystemExit("--log-target is only wired for autogluon so far")
    return lightautoml_oof(store, budget, preset)


ENGINES = {"autogluon": autogluon_oof, "lightautoml": lightautoml_oof}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--engine", choices=sorted(ENGINES), required=True)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--preset", default="good_quality")
    parser.add_argument("--log-target", action="store_true",
                        help="fit the engine on log1p(target) and invert afterwards")
    args = parser.parse_args()
    if args.log_target and load_task(TASKS[args.task]).is_classification:
        raise SystemExit("--log-target makes no sense for a classification target")

    spec = load_task(TASKS[args.task])
    store = DataStore(spec, ROOT / ".bench_cache" / args.task, log=lambda m: print(f"  {m}"))
    print(f"\n{spec.kind} / {spec.metric} / scheme {store.plan.name} "
          f"({store.plan.n_splits} splits, {int(store.evaluable.sum())} rows validated)")
    print(f"running {args.engine}, preset {args.preset}, budget {args.budget}s, {N_CPUS} cpus\n")

    oof, proba, seconds = _call(args.engine, store, args.budget, args.preset, args.log_target)
    # The column order the engine actually used. Without it the threshold block
    # below is comparing probabilities to the wrong labels.
    engine_classes = globals().get("_LAST_CLASSES")

    keep = store.evaluable
    primary = score(spec.metric, store.y[keep], oof[keep],
                    proba[keep] if proba is not None else None, spec.classes)
    label = f"{args.engine} {args.preset}" + (" +log1p" if args.log_target else "")
    print(f"\n  {label}: {spec.metric} = {primary:.4f}  [{seconds:.0f}s]")

    per_fold = []
    for fold in range(store.n_splits):
        mask = store.fold_id == fold
        try:
            per_fold.append(score(spec.metric, store.y[mask], oof[mask],
                                  proba[mask] if proba is not None else None, spec.classes))
        except Exception:  # noqa: BLE001
            pass
    print(f"  per fold: {[round(v, 4) for v in per_fold]}")

    if spec.is_classification and proba is not None and engine_classes:
        import agent_thresholds

        # The engine's own column order, not sorted(unique(y)). Fixing this in the
        # fitting path and leaving it here produced a second wrong number from the
        # same cause: LightAutoML scored a real 0.627 on argmax and a nonsense
        # 0.184 with thresholds, because only half the ordering was corrected.
        classes = np.asarray([str(c) for c in engine_classes])
        order = np.argsort(classes)
        y_index = np.searchsorted(classes[order], store.y.astype(str).to_numpy())
        y_index = order[y_index]
        divisors, fitted = agent_thresholds.fit(proba[keep], y_index[keep])
        honest = agent_thresholds.honest_score(proba, y_index, store.fold_id)
        print(f"  + per-class thresholds: fitted {fitted:.4f}, nested {honest:.4f}")

    out = ROOT / "runs" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_{args.engine}" + ("_log" if args.log_target else "")
    np.save(out / f"{tag}_oof.npy", oof.astype(object), allow_pickle=True)
    if proba is not None:
        np.save(out / f"{tag}_proba.npy", proba)
    print(f"  predictions saved under {out}")


if __name__ == "__main__":
    main()
