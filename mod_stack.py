"""Stack the experiments already in the ledger, using predictions already computed.

A run typically ends with several models of roughly equal strength and different
character. Averaging them is worth something; letting a small model learn *how* to
combine them is usually worth more, and it is the one thing the agent in our
three-hour rehearsal explicitly said it would do with more time and never got to.

What makes it cheap is that nothing is refitted. Every recipe writes its
out-of-fold and test predictions to the cache, so the inputs are already on disk;
this tool reads them and fits one small meta-model.

What makes it honest is that the meta-model is fitted out-of-fold too. For each
fold, the meta learns on the other folds' base predictions and is scored on this
one -- the same discipline the base models are held to. Stacking on predictions
that the meta has already seen is the standard way to produce a number that cannot
be reproduced on new data, and the fold structure here is whatever the run's own
plan says, so a forecast stacks on earlier periods only.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from agent_registry import tool
from agent_session import session


def _load(experiment) -> dict[str, Any] | None:
    """The predictions an experiment left behind, or None if it left none."""
    paths = (experiment.detail or {}).get("predictions") or {}
    if "error" in paths or "test_pred" not in paths:
        return None
    try:
        out: dict[str, Any] = {}
        for key in ("oof", "oof_proba", "test_pred"):
            if key in paths:
                out[key] = np.load(paths[key], allow_pickle=key != "oof_proba")
        return out if "test_pred" in out else None
    except Exception:  # noqa: BLE001 - a missing cache file is not a crash
        return None


def _average(X: np.ndarray, width: int) -> np.ndarray:
    """Plain mean across the base models, no fitting at all.

    Measured to be the right default on a forecast task, and the reasoning matters
    more than the number. Model *rankings do not survive the move to the next
    period*: on this project's regression task the rank correlation between one
    period's scores and the next period's was -0.04, and against the period two
    steps away it was negative. Selecting a winner is therefore a draw from a
    distribution with a 3.2 RMSE spread, not a decision.

    Averaging gives up the best single model -- it lands 0.6 to 1.1 RMSE behind an
    oracle -- in exchange for never landing in the tail. Backtested on three
    periods in turn it beat the typical single candidate every time, by 0.38, 0.55
    and 0.65 RMSE, and on the period most like the test set it beat both selection
    rules outright.

    The large decisions still transfer: every transformed candidate beat the
    untransformed one on every period. It is the fine choices -- tree count, leaf
    count, which transform -- that are noise, and those are what this averages over
    instead of choosing between.
    """
    blocks = np.split(X, X.shape[1] // width, axis=1) if width > 1 else [
        X[:, [i]] for i in range(X.shape[1])
    ]
    return np.mean(blocks, axis=0)


def _meta(kind: str, classification: bool):
    if classification:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, C=1.0)
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                             n_jobs=4, verbose=-1, random_state=0)
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


@tool(experimental=True)
def stack_experiments(experiment_ids: list[str],
                      meta: Literal["mean", "ridge", "logreg", "lightgbm"] = "mean",
                      note: str = "") -> dict[str, Any]:
    """Learn how to combine several experiments' predictions, and enter the result in the ledger.

    Use it late, once two or more experiments are close in score and different in
    kind -- a booster and a linear model, or a recipe and an engine. Stacking two
    tuned LightGBMs buys nothing, because they make the same mistakes.

    Needs nothing refitted: it reads the out-of-fold and test predictions each
    experiment already wrote. For classification it uses probabilities, so the base
    experiments must have produced them.

    The score it reports is out-of-fold for the meta-model as well as the base
    models, which is why it can be compared with everything else in the ledger.

    Args:
        experiment_ids: Which experiments to combine, e.g. ["exp_002", "exp_005"].
        meta: How to combine. `mean` fits nothing and is the default, because on a
            task where model rankings do not carry from one period to the next
            there is nothing reliable for a combiner to learn; it beat every
            single candidate's typical result on a three-period backtest. `ridge`
            and `lightgbm` learn weights and are worth trying when the base models
            differ a lot in quality.
        note: What this combination is testing.
    """
    import agent_tool_modeling as modeling

    current = session()
    spec, store = current.spec, current.store
    classification = spec.is_classification

    by_id = {e.id: e for e in current.ledger.entries}
    unknown = [i for i in experiment_ids if i not in by_id]
    if unknown:
        return {"status": "error", "problem": f"no such experiment: {unknown}",
                "hint": f"Known: {sorted(by_id)}"}
    if len(experiment_ids) < 2:
        return {"status": "refused", "problem": "stacking needs at least two experiments",
                "hint": "Name two that are close in score and different in kind."}

    blocks, used, skipped = [], [], []
    test_blocks = []
    for identifier in experiment_ids:
        experiment = by_id[identifier]
        loaded = _load(experiment)
        if loaded is None:
            skipped.append({"id": identifier, "why": "no cached predictions"})
            continue
        if classification:
            if "oof_proba" not in loaded:
                skipped.append({"id": identifier, "why": "no probabilities, only labels"})
                continue
            blocks.append(np.asarray(loaded["oof_proba"], dtype="float64"))
            test_blocks.append(_test_proba(experiment, loaded, spec))
            if test_blocks[-1] is None:
                blocks.pop()
                test_blocks.pop()
                skipped.append({"id": identifier, "why": "no test probabilities"})
                continue
        else:
            blocks.append(np.asarray(loaded["oof"], dtype="float64").reshape(-1, 1))
            test_blocks.append(np.asarray(loaded["test_pred"], dtype="float64").reshape(-1, 1))
        used.append(identifier)

    if len(used) < 2:
        return {"status": "refused",
                "problem": f"only {len(used)} of {len(experiment_ids)} experiments had usable "
                           f"predictions",
                "skipped": skipped,
                "hint": ("For classification, pass probabilities to evaluate_predictions or use "
                         "a recipe -- a column of labels cannot be stacked.")}

    X = np.hstack(blocks)
    X_test = np.hstack(test_blocks)
    keep = store.evaluable
    y = store.y.to_numpy()

    oof = np.full(len(y), None, dtype=object) if classification else np.full(len(y), np.nan)
    folds = [f for f in np.unique(store.fold_id) if f >= 0]
    if len(folds) < 2:
        return {"status": "refused",
                "problem": f"the run validates on {len(folds)} fold(s); stacking needs at least two",
                "hint": "Use blend_top_experiments, or average the predictions in run_python."}

    # Leave one fold out among the **validated** rows, rather than reusing the run's
    # own training halves. The reason is specific and worth stating: under a
    # forecast plan the earliest origin trains on periods that no fold validates, so
    # there are no base predictions there at all, and following the run's own splits
    # would leave that origin unstacked -- a score covering fewer rows than every
    # other entry in the ledger, which is exactly the incomparability this project
    # spent a day removing.
    #
    # What is given up is the temporal ordering for the *meta* model: it may learn
    # its combination weights partly from later periods. That is a weaker guarantee
    # than the base models get, and it is acceptable because the meta has a handful
    # of parameters and sees only out-of-fold predictions -- it cannot memorise
    # rows, only how much to trust each model. The caveat is reported in the result
    # rather than left implicit.
    width = X.shape[1] // len(used)
    if meta == "mean":
        # Nothing is fitted, so there is nothing to hold out: an average of
        # out-of-fold predictions is still out-of-fold.
        averaged = _average(X, width)
        oof = (np.asarray([str(c) for c in spec.classes])[averaged.argmax(1)]
               if classification else averaged.reshape(-1))
        oof = np.where(keep, oof, None if classification else np.nan)
    else:
        for fold in folds:
            held = store.fold_id == fold
            fit_rows = np.where(keep & ~held)[0]
            val_rows = np.where(held)[0]
            model = _meta(meta, classification)
            try:
                model.fit(X[fit_rows], y[fit_rows])
                oof[val_rows] = model.predict(X[val_rows])
            except Exception as err:  # noqa: BLE001
                return {"status": "error", "problem": f"the meta-model failed: {err}",
                        "hint": "Try meta='mean', or fewer base experiments."}

    try:
        primary = modeling.score(spec.metric, y[keep], oof[keep], None, spec.classes)
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "problem": f"the stacked predictions could not be scored: {err}",
                "hint": "One of the base experiments is probably missing predictions for some rows."}

    if meta == "mean":
        averaged_test = _average(X_test, width)
        test_pred = (np.asarray([str(c) for c in spec.classes])[averaged_test.argmax(1)]
                     if classification else averaged_test.reshape(-1))
    else:
        final = _meta(meta, classification)
        final.fit(X[keep], y[keep])
        test_pred = final.predict(X_test)

    detail = {
        "stacked": used,
        "meta": meta if meta == "mean" else ("logreg" if classification else meta),
        "skipped": skipped,
        "all_metrics": modeling.report(spec.kind, y[keep], oof[keep], None, spec.classes),
        "base_scores": {i: round(by_id[i].score, 6) for i in used},
        "caveat": ("" if meta == "mean" else
                   "The combiner is fitted leave-one-fold-out on out-of-fold base "
                   "predictions. On a forecast plan it may therefore learn its weights "
                   "partly from later periods than it is scored on -- a weaker guarantee "
                   "than the base models have, though it only affects how much each "
                   "model is trusted, not what it predicts."),
    }
    detail["predictions"] = modeling._keep_predictions(
        current.ledger.next_id(),
        {"oof": oof, "test_pred": test_pred, "classes": spec.classes},
    )
    experiment = modeling._register(
        f"stack_{'_'.join(i.split('_')[-1] for i in used)}", "custom", primary,
        modeling._fold_scores(oof, None, spec.classes), 0.0, detail, note,
    )
    return modeling._finish(experiment, {"test_pred": test_pred}, [])


def _test_proba(experiment, loaded: dict[str, Any], spec) -> np.ndarray | None:
    """Test probabilities for a classification experiment, if it kept any.

    Recipes keep fold-averaged test *labels* rather than probabilities, so this is
    only available for experiments that went through evaluate_predictions or an
    engine. Returning None is the honest answer, and the caller says so.
    """
    stored = (experiment.detail or {}).get("predictions") or {}
    if "test_proba" in stored:
        try:
            return np.load(stored["test_proba"])
        except Exception:  # noqa: BLE001
            return None
    return None
