"""Full AutoML engines as one tool call, on the run's own folds.

``autogluon.tabular``, ``lightautoml`` and ``flaml`` all sit in the organisers'
``requirements.txt``, and for three runs none of them was ever called. They are
worth having for the reason a recipe is not: they search model families, stack
them and build their own ensemble, which is work the agent would otherwise spend
its whole budget imitating by hand.

Three things make this safe to hand an agent, and all three matter:

**The engine is given our folds, not its own.** Otherwise its score is measured on
a different split from every recipe in the ledger and the two cannot be compared,
which is how a run ends up submitting the thing that merely looks best.

**The two fold schemes are not the same shape.** ``groups=`` in AutoGluon and
``FoldsRole`` in LightAutoML both mean K-fold bagging: hold out one fold, train on
all the others. That *is* our ``grouped`` scheme. It is the exact opposite of what
``forward_chain`` needs, where "all the other folds" includes later periods than
the one being validated -- the leak that cost this project two submissions. So a
forecast is fitted once per origin, on earlier periods only.

**The result goes through the ledger.** Same ``_register``/``_finish`` as a recipe,
so it gets a paired verdict, a fold spread, the plausibility guard, and the
submission lock. An engine is not a way around the honesty machinery.

The module imports with no engine present: each import happens inside the call, so
a missing engine is one refusal rather than a disabled tool.
"""

from __future__ import annotations

import shutil
import time
from typing import Any, Literal

import numpy as np
import pandas as pd

from agent_registry import tool
from agent_session import session

#: Seconds held back from the asked-for budget, on top of the run's own reserve.
#: An engine that is still fitting when the watchdog fires contributes nothing,
#: and the fit cannot be interrupted usefully.
MARGIN_S = 120

#: Our metric names to each engine's. An engine optimises what it is told to, and
#: telling it the wrong objective is the quietest way to waste an hour.
AUTOGLUON_METRIC = {
    "rmse": "root_mean_squared_error", "mae": "mean_absolute_error", "r2": "r2",
    "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "accuracy": "accuracy",
    "roc_auc": "roc_auc", "logloss": "log_loss",
}


def _available(engine: str) -> str:
    """Empty string when the engine can be imported, else why it cannot."""
    module = {"autogluon": "autogluon.tabular", "lightautoml": "lightautoml",
              "flaml": "flaml"}[engine]
    try:
        __import__(module)
    except Exception as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return ""


@tool(experimental=True)
def engine_capabilities() -> dict[str, Any]:
    """Which AutoML engines this container can actually run, and how folds will be fed to them.

    Call it before `run_engine` so a missing engine costs a cheap question rather
    than a long fit that fails at the end. The answer also says which fold scheme
    the run is on, because that decides how the engine has to be driven.
    """
    current = session()
    plan = current.store.plan
    engines = {name: (_available(name) or "ready") for name in ("autogluon", "lightautoml", "flaml")}
    return {
        "engines": engines,
        "validation_scheme": plan.name,
        "how_folds_are_passed": (
            "one fit per origin on earlier periods only -- the engine's own bagging "
            "would train on later periods than it validates"
            if plan.name == "forward_chain"
            else "the run's fold column is handed to the engine's own bagging"
        ),
        "note": (
            "ray is absent from requirements.txt, so AutoGluon fits bagged folds "
            "sequentially rather than in parallel. Budget accordingly: it is slower "
            "here than its documentation assumes."
        ),
    }


def _fit_autogluon(store, frame: pd.DataFrame, budget: int, preset: str,
                   log) -> dict[str, Any]:
    from autogluon.tabular import TabularPredictor

    spec = store.spec
    target = spec.target_column
    metric = AUTOGLUON_METRIC.get(spec.metric)
    classification = spec.is_classification
    path = store.cache_dir / "engine_autogluon"

    oof = np.full(len(frame), None, dtype=object) if classification else np.full(len(frame), np.nan)
    proba: np.ndarray | None = None
    classes: list | None = None

    test = store.test.drop(columns=[c for c in (spec.id_column,) if c in store.test.columns])

    if store.plan.name == "forward_chain":
        # One share per origin, plus one for the fit that predicts the test set.
        # That last fit is the only one allowed to see the most recent period, and
        # the test set is closer to that period than to anything else.
        per = max(60, budget // (store.plan.n_splits + 1))
        for index, (train_idx, val_idx) in enumerate(store.splits()):
            predictor = TabularPredictor(
                label=target, eval_metric=metric, path=str(path / f"origin_{index}"),
                verbosity=0,
            ).fit(frame.iloc[train_idx], presets=preset, time_limit=per)
            held = frame.iloc[val_idx].drop(columns=[target])
            oof[val_idx] = np.asarray(predictor.predict(held))
            if classification:
                block = predictor.predict_proba(held)
                classes = list(block.columns)
                if proba is None:
                    proba = np.zeros((len(frame), block.shape[1]))
                proba[val_idx] = block.to_numpy()
            log(f"[engine] origin {index + 1}/{store.plan.n_splits} fitted")

        final = TabularPredictor(
            label=target, eval_metric=metric, path=str(path / "full"), verbosity=0,
        ).fit(frame, presets=preset, time_limit=per)
        log("[engine] final fit on every period, for the test prediction")
        test_proba = None
        if classification:
            block = final.predict_proba(test)
            classes = list(block.columns)
            test_proba = block.to_numpy()
        return {"oof": oof, "proba": proba, "classes": classes,
                "test_pred": np.asarray(final.predict(test)), "test_proba": test_proba}

    with_fold = frame.copy()
    with_fold["__fold"] = store.fold_id
    # `groups` is a constructor argument, not a fit keyword. Passing it to .fit()
    # raises, and the mistake is invisible on a forecast task because that path
    # never uses it -- it only surfaced on the first grouped-fold run.
    predictor = TabularPredictor(
        label=target, eval_metric=metric, path=str(path), verbosity=0, groups="__fold",
    ).fit(with_fold, presets=preset, time_limit=budget)

    test_proba = None
    if classification:
        block = predictor.predict_proba_oof()
        classes = list(block.columns)
        proba = block.to_numpy()
        oof = np.asarray(classes, dtype=object)[proba.argmax(1)]
        test_block = predictor.predict_proba(test)
        test_proba = test_block[classes].to_numpy()
    else:
        oof = np.asarray(predictor.predict_oof(), dtype="float64")
    return {"oof": oof, "proba": proba, "classes": classes,
            "test_pred": np.asarray(predictor.predict(test)), "test_proba": test_proba}


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


def _fit_lightautoml(store, frame: pd.DataFrame, budget: int, preset: str,
                     log) -> dict[str, Any]:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    spec = store.spec
    target, idc = spec.target_column, spec.id_column
    classification = spec.is_classification
    kind = {"regression": "reg", "binary": "binary", "multiclass": "multiclass"}[spec.kind]
    # Filled in after the first fit, from the model itself -- see _lama_classes.
    classes = sorted(pd.unique(store.y.astype(str))) if classification else None
    roles: dict[str, Any] = {"target": target, "drop": [idc]}

    oof = np.full(len(frame), None, dtype=object) if classification else np.full(len(frame), np.nan)
    proba: np.ndarray | None = None

    def build(seconds: int) -> Any:
        return TabularAutoML(task=Task(kind), timeout=seconds,
                             reader_params={"random_state": 42})

    if store.plan.name == "forward_chain":
        per = max(60, budget // (store.plan.n_splits + 1))
        for index, (train_idx, val_idx) in enumerate(store.splits()):
            model = build(per)
            model.fit_predict(frame.iloc[train_idx], roles=roles, verbose=0)
            if classification:
                classes = _lama_classes(model, classes)
            block = model.predict(frame.iloc[val_idx]).data
            if classification:
                if proba is None:
                    proba = np.zeros((len(frame), block.shape[1]))
                proba[val_idx] = block
                oof[val_idx] = np.asarray(classes, dtype=object)[block.argmax(1)]
            else:
                oof[val_idx] = block.reshape(-1)
            log(f"[engine] origin {index + 1}/{store.plan.n_splits} fitted")

        final = build(per)
        final.fit_predict(frame, roles=roles, verbose=0)
        log("[engine] final fit on every period, for the test prediction")
        block = final.predict(store.test).data
        if classification:
            return {"oof": oof, "proba": proba, "classes": classes,
                    "test_pred": np.asarray(classes, dtype=object)[block.argmax(1)],
                    "test_proba": block}
        return {"oof": oof, "proba": proba, "classes": classes,
                "test_pred": block.reshape(-1), "test_proba": None}

    with_fold = frame.copy()
    with_fold["__fold"] = store.fold_id
    model = build(budget)
    block = model.fit_predict(with_fold, roles={**roles, "folds": "__fold"}, verbose=0).data
    if classification:
        classes = _lama_classes(model, classes)
    test_block = model.predict(store.test).data
    if classification:
        return {"oof": np.asarray(classes, dtype=object)[block.argmax(1)],
                "proba": block, "classes": classes,
                "test_pred": np.asarray(classes, dtype=object)[test_block.argmax(1)],
                "test_proba": test_block}
    return {"oof": block.reshape(-1), "proba": None, "classes": None,
            "test_pred": test_block.reshape(-1), "test_proba": None}


ENGINES = {"autogluon": _fit_autogluon, "lightautoml": _fit_lightautoml}


@tool(experimental=True)
def run_engine(engine: Literal["autogluon", "lightautoml"], time_budget_s: int,
               preset: str = "good_quality", target_transform: Literal["none", "log1p"] = "none",
               note: str = "") -> dict[str, Any]:
    """Run a full AutoML engine on this run's folds and enter the result in the ledger.

    The engine searches model families, tunes them and stacks them itself, which is
    the one kind of work a recipe cannot express and the agent should not imitate by
    hand. It is scored on the same folds as every recipe, so the leaderboard stays
    comparable, and it is subject to the same verdict, plausibility guard and
    submission lock.

    Call `engine_capabilities` first: it says which engines this container has and
    how long they are likely to take here.

    Budget honestly. The engine cannot be interrupted usefully, so a fit that
    overruns contributes nothing at all. On a forecast task the budget is divided
    between the validation origins **and** a final fit on every period, so ask for
    more than you would for a single model.

    `target_transform="log1p"` is worth trying on a skewed non-negative target: no
    engine searches over target transforms, and on this project's regression task
    log1p was worth more than everything the engine does on top of it.

    Args:
        engine: Which engine to run.
        time_budget_s: Total seconds for the whole thing, fits included.
        preset: Engine preset. For AutoGluon: medium_quality, good_quality,
            high_quality, best_quality -- later ones stack more and cost more.
        target_transform: Apply log1p to the target and invert afterwards. Regression only.
        note: What this run is testing, for the leaderboard.
    """
    import agent_tool_modeling as modeling

    current = session()
    spec, store, clock = current.spec, current.store, current.clock

    missing = _available(engine)
    if missing:
        return {"status": "refused", "problem": f"{engine} cannot be imported here: {missing}",
                "hint": "Call engine_capabilities to see what this container has."}

    if target_transform == "log1p":
        if spec.is_classification:
            return {"status": "refused", "problem": "log1p needs a numeric target",
                    "hint": 'Use target_transform="none" for a classification task.'}
        if float(np.nanmin(store.y)) < 0:
            return {"status": "refused",
                    "problem": "the target has negative values, so log1p is undefined",
                    "hint": 'Use target_transform="none".'}

    # The clock, before anything expensive. An engine that is still fitting when the
    # watchdog fires has spent the time and produced nothing, which is worse than
    # not having been called.
    affordable = int(clock.usable_remaining - MARGIN_S)
    if affordable < 120:
        return {"status": "refused",
                "problem": f"only {max(0, affordable)}s of usable time remain",
                "hint": "Finalise with what is already in the ledger."}
    budget = min(int(time_budget_s), affordable)
    if budget < time_budget_s:
        note = (note + " " if note else "") + f"budget cut from {time_budget_s}s to {budget}s"

    if spec.is_classification:
        counts = store.y.value_counts()
        if int(counts.min()) < store.n_splits:
            return {"status": "refused",
                    "problem": (f"class '{counts.idxmin()}' has {int(counts.min())} rows and the "
                                f"run uses {store.n_splits} folds, so some fold would train "
                                f"without it"),
                    "hint": "Engines bag internally and cannot be given a fold that lacks a class."}

    frame = store.train.drop(columns=[c for c in (spec.id_column,) if c in store.train.columns])
    if target_transform == "log1p":
        frame = frame.copy()
        frame[spec.target_column] = np.log1p(frame[spec.target_column])

    started = time.perf_counter()
    log = current.log if hasattr(current, "log") else print
    try:
        out = ENGINES[engine](store, frame, budget, preset, log)
    except Exception as err:  # noqa: BLE001 - an engine that falls over must not end the run
        return {"status": "error", "problem": f"{type(err).__name__}: {err}",
                "hint": ("The engine failed, the run is intact and nothing was saved. Try a "
                         "smaller preset or a plain recipe."),
                "seconds": round(time.perf_counter() - started, 1)}

    # The artifacts are large and no longer needed: AutoGluon wrote 470 MB per
    # origin here, and on the platform this directory sits next to the task's own
    # data. The predictions are in hand; the models are not reused.
    for folder in (store.cache_dir / "engine_autogluon", store.cache_dir / "engine_lightautoml"):
        shutil.rmtree(folder, ignore_errors=True)

    oof, proba, classes = out["oof"], out["proba"], out["classes"]
    test_pred, test_proba = out["test_pred"], out["test_proba"]
    if target_transform == "log1p":
        oof = np.where(store.evaluable, np.expm1(oof.astype("float64")), np.nan)
        test_pred = np.expm1(np.asarray(test_pred, dtype="float64"))

    keep = store.evaluable
    primary = modeling.score(spec.metric, store.y[keep], oof[keep],
                             proba[keep] if proba is not None else None, classes)

    detail: dict[str, Any] = {
        "engine": engine, "preset": preset, "target_transform": target_transform,
        "budget_s": budget, "scheme": store.plan.name,
        "all_metrics": modeling.report(spec.kind, store.y[keep], oof[keep],
                                       proba[keep] if proba is not None else None, classes),
    }

    # Per-class thresholds on top, for the same reason they sit on top of a recipe:
    # no engine tunes a decision rule for a macro-averaged metric, and the engine's
    # argmax answers to the class prior instead of to the metric.
    if spec.is_classification and proba is not None and spec.metric == "macro_f1":
        import agent_thresholds

        order = np.asarray([str(c) for c in classes])
        y_index = np.searchsorted(order, store.y.astype(str).to_numpy())
        divisors, fitted = agent_thresholds.fit(proba[keep], y_index[keep])
        honest = agent_thresholds.honest_score(proba, y_index, store.fold_id)
        if fitted > primary:
            oof = np.where(keep, order[agent_thresholds.apply(proba, divisors)], None)
            if test_proba is not None:
                test_pred = order[agent_thresholds.apply(np.asarray(test_proba), divisors)]
            primary = fitted
            detail["thresholds"] = {
                "divisors": {str(c): round(float(d), 3) for c, d in zip(classes, divisors)},
                "macro_f1_fitted": round(fitted, 6),
                "macro_f1_honest": round(honest, 6),
                "optimism": round(fitted - honest, 6),
                "note": ("The reported score uses the fitted rule; macro_f1_honest is what it "
                         "is worth on predictions the rule has not seen."),
            }

    detail["predictions"] = modeling._keep_predictions(
        current.ledger.next_id(),
        {"oof": oof, "oof_proba": proba, "test_pred": test_pred,
         "test_proba": test_proba, "classes": classes},
    )
    name = f"{engine}_{preset}" + ("_log1p" if target_transform == "log1p" else "")
    experiment = modeling._register(
        name, "custom", primary,
        modeling._fold_scores(oof, proba, classes),
        time.perf_counter() - started, detail, note,
    )
    return modeling._finish(experiment, {"test_pred": test_pred}, [])
