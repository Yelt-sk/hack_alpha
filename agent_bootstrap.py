"""Guaranteeing an answer before the agent is allowed to be clever.

The platform grades whatever file exists at the three-hour mark. An agent that
spends two hours on a brilliant idea and crashes at the end scores exactly zero,
which is worse than a mediocre model written in the first five minutes.

So the run climbs a ladder, and each rung only replaces the previous one if it
validates and scores better:

1. a constant prediction, within seconds of start;
2. a default gradient boosting model, within the first minutes;
3. anything the agent earns after that.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from agent_metrics import score
from agent_recipe_pipeline import evaluate
from agent_recipe_schema import validate
from agent_session import Session
from agent_state import Experiment
from agent_submission import answer_column, frame_from_predictions


def floor(session: Session, log=print) -> dict[str, Any]:
    """Rung one: a valid file on disk immediately."""
    result = session.submissions.save_constant(session.store.y)
    session.note(f"constant submission written: {result['status']}")
    log(f"[bootstrap] constant submission: {result['status']}")
    return result


def baseline(session: Session, log=print) -> dict[str, Any]:
    """Rung two: an untuned booster, scored on the shared folds and saved."""
    spec, store = session.spec, session.store
    # The floor is weighted for an imbalanced classification metric, because the
    # metric is what it has to clear. Measured on the classification task: the
    # same booster scores 0.527 unweighted, 0.598 balanced and 0.608 with sqrt
    # weights. Leaving the floor unweighted means the run's guaranteed minimum
    # is a tenth of macro F1 below what the same model gives for free.
    def candidate(name: str, transform: str = "none", weighted: str = "none",
                  params: dict[str, Any] | None = None,
                  rows: str = "none") -> dict[str, Any]:
        return {
            "name": name,
            "notes": "Automatic floor, before the agent starts.",
            "drop_columns": [spec.id_column],
            "class_weight": weighted,
            # Thresholds go in the floor, not only within the agent's reach,
            # because the floor is the part of the run that is guaranteed. On the
            # classification task they are worth 0.6144 -> 0.6379 by the honest
            # nested estimate, which is more than the whole of the previous
            # submission earned.
            "postprocess": {"thresholds": "per_class" if weighted != "none" else "none"},
            "target_transform": transform,
            "sample_weight": rows,
            "model": {"type": "lightgbm", "params": {"n_estimators": 300, **(params or {})}},
        }

    wanted: list[dict[str, Any]] = []
    if spec.kind in ("binary", "multiclass"):
        # Which weighting wins depends on whether thresholds follow, and the two
        # answers disagree. Measured on honest folds: `sqrt` beats `balanced` on
        # argmax (0.6235 against 0.6073) and loses to it once per-class thresholds
        # are fitted (0.6322 against 0.6427, on four folds of five). The mechanism
        # is that `balanced` over-predicts the rare classes on its own but leaves
        # the thresholds more rare-class signal to work with.
        #
        # So the floor measures both rather than carrying a rule of thumb that is
        # right for the wrong decision rule.
        wanted = [candidate("bootstrap_lightgbm", weighted="sqrt"),
                  candidate("bootstrap_lightgbm_balanced", weighted="balanced")]
    else:
        wanted = [candidate("bootstrap_lightgbm")]
    # A skewed non-negative target is the textbook case for log1p, and on this
    # project's regression task it was worth 6 RMSE on honest folds -- more than
    # every other change to that task put together, and invisible under the random
    # folds that preceded agent_split. So the floor tries it rather than assuming
    # either way: two fits of an untuned booster cost seconds, and the guaranteed
    # rung of the ladder is the wrong place to leave that much on the table.
    if spec.kind == "regression":
        try:
            skew = float(pd.Series(store.y).skew())
            # A relative metric (RMSLE, MAPE, SMAPE) asks for the logarithm
            # whatever the skew: it is the space such a metric is computed in.
            relative = spec.metric in ("rmsle", "mape", "smape")
            if float(np.nanmin(store.y)) >= 0 and (skew > 1.0 or relative):
                wanted.append(candidate("bootstrap_lightgbm_log1p", transform="log1p"))
                # And a loss built for a skewed non-negative target, which is a
                # different answer to the same problem: log1p changes what the
                # model predicts, Tweedie changes what it is penalised for. The
                # distinction matters because the metric is RMSE -- an absolute
                # error -- while log1p optimises a relative one, so the logarithm
                # buys the bulk by giving up the tail, and on this task 3% of the
                # rows carry a third of the squared error.
                #
                # Measured on the honest forward-chaining folds, against the log1p
                # floor's 48.458: Tweedie on the raw target gives 46.48 and wins
                # on all three origins, at both 300 trees and 900. The variance
                # power barely matters (46.48 at 1.1, 46.80 at 1.5, 46.48 at 1.7),
                # which is the usual finding here -- the large decision transfers,
                # the fine tuning is noise -- so the floor spends one fit on the
                # middle of that range and leaves the sweep to the agent.
                wanted.append(candidate(
                    "bootstrap_lightgbm_tweedie",
                    params={"objective": "tweedie", "tweedie_variance_power": 1.5},
                ))
                # And the third answer to the same question: keep the logarithm,
                # which is what holds the bulk together, but weight each row by
                # the square root of its target, which is what tells an absolute
                # metric where its error is. Measured best of the three on the
                # honest folds -- 46.05 against the tweedie 46.82 and the plain
                # log1p 48.46, winning on all three origins at 300 trees and
                # again at 900, where it gives 45.82 while the unweighted log1p
                # gets worse. Proportional weighting was measured too and is a
                # disaster (64.49): it hands the whole fit to 3% of the rows.
                wanted.append(candidate("bootstrap_lightgbm_log1p_weighted",
                                        transform="log1p", rows="sqrt_target"))
                log(f"[bootstrap] target skew {skew:.2f} and non-negative; trying "
                    f"log1p, a tweedie loss and magnitude weights beside the plain floor")
        except Exception:  # noqa: BLE001 - an unmeasurable target just gets the plain floor
            pass

    # When the test is a later period of entities already seen in train, their own
    # past target is worth more than any loss or weighting: on the regression task
    # it takes the best floor from 46.05 to 41.38 RMSE, winning on all three
    # origins. Offered on top of the most refined candidate so far, and kept only
    # if it measures better -- an entity key found by accident would simply lose.
    entity = (getattr(store.plan, "diagnosis", None) or {}).get("entity")
    if spec.kind == "regression" and entity:
        richest = dict(wanted[-1])
        richest["name"] = f"{richest['name']}_history"
        richest["history"] = {"key": entity["columns"], "period": entity["period"]}
        wanted.append(richest)
        log(f"[bootstrap] entities recur across '{entity['period']}' "
            f"({entity['test_match']:.0%} of test seen before); trying their target history")

    started = time.perf_counter()
    attempts: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for index, raw in enumerate(wanted):
        if index and not session.clock.affordable(time.perf_counter() - started):
            # The first candidate timed itself; a second costs about the same. If
            # that no longer fits, the floor stops with what it has rather than
            # running past the reserve -- a measured comparison is worth having,
            # not worth the run.
            log(f"[bootstrap] no time for {raw['name']}; keeping what is measured")
            break
        recipe, errors, _ = validate(raw, spec.kind, list(store.X.columns))
        if errors:  # the vocabulary and the task disagree
            log(f"[bootstrap] {raw['name']} rejected: {errors}")
            continue
        try:
            outcome = evaluate(recipe, store)
        except Exception as err:  # noqa: BLE001
            log(f"[bootstrap] {raw['name']} failed: {type(err).__name__}: {err}")
            continue
        mask = store.evaluable
        value = score(spec.metric, store.y[mask], outcome["oof"][mask],
                      outcome["oof_proba"][mask] if outcome["oof_proba"] is not None else None,
                      outcome["classes"])
        attempts.append((recipe, outcome, value))
        if len(wanted) > 1:
            log(f"[bootstrap] {recipe['name']}: {spec.metric}={value:.5f}")

    if not attempts:
        # The constant file still stands, so the run is not lost.
        session.note("baseline produced nothing; the constant submission stands")
        return {"status": "failed", "problem": "no baseline recipe could be evaluated"}

    attempts.sort(key=lambda a: a[2] if spec.direction == "minimize" else -a[2])
    recipe, result, primary = attempts[0]
    if len(attempts) > 1:
        log(f"[bootstrap] floor is {recipe['name']} "
            f"({spec.metric}={primary:.5f} against {attempts[1][2]:.5f})")

    # Per-fold scores as well, not just the aggregate: without a spread this entry
    # would claim zero variance, and every later comparison against it would read
    # a coin flip as a real improvement.
    fold_scores: list[float] = []
    for fold in range(store.n_splits):
        mask = store.fold_id == fold
        try:
            fold_scores.append(score(
                spec.metric, store.y[mask], result["oof"][mask],
                result["oof_proba"][mask] if result["oof_proba"] is not None else None,
                result["classes"],
            ))
        except Exception:  # a fold that cannot be scored simply does not contribute
            continue

    # The floor's predictions are kept like any other experiment's. It is often
    # among the strongest models in a run -- on the regression task it *is* the best
    # for the first half hour -- and a model whose predictions were thrown away
    # cannot be stacked, blended or re-thresholded by anything that follows.
    from agent_tool_modeling import _keep_predictions

    predictions = _keep_predictions(session.ledger.next_id(), result)

    experiment = session.ledger.add(Experiment(
        id=session.ledger.next_id(), name=recipe["name"], source="baseline", score=primary,
        fold_scores=fold_scores, metric=spec.metric, direction=spec.direction,
        seconds=time.perf_counter() - started, note="automatic floor",
        detail={"recipe": recipe,
                "floor_candidates": {r["name"]: round(v, 6) for r, _, v in attempts},
                "predictions": predictions},
    ))
    answer = answer_column(spec, result)
    if answer is None:
        log(f"[bootstrap] {recipe['name']} has no test probabilities; the answer was not written")
        return {"status": "ok", "score": primary, "submission": None}
    frame = frame_from_predictions(spec, store.test[spec.id_column], answer)
    saved = session.submissions.save(frame, score=primary, source=experiment.id, note=recipe["name"])

    log(f"[bootstrap] {spec.metric}={primary:.5f} in {experiment.seconds:.0f}s -> {saved['status']}")
    session.note(f"baseline {spec.metric}={primary:.5f}")
    return {"status": "ok", "score": primary, "submission": saved}
