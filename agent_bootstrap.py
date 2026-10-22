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

from agent_metrics import score
from agent_recipe_pipeline import evaluate
from agent_recipe_schema import validate
from agent_session import Session
from agent_state import Experiment
from agent_submission import frame_from_predictions


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
    weighted = "sqrt" if spec.kind in ("binary", "multiclass") else "none"
    recipe, errors, _ = validate(
        {
            "name": "bootstrap_lightgbm",
            "notes": "Automatic floor, before the agent starts.",
            "drop_columns": [spec.id_column],
            "class_weight": weighted,
            # Thresholds go in the floor, not only within the agent's reach,
            # because the floor is the part of the run that is guaranteed. On the
            # classification task they are worth 0.6144 -> 0.6379 by the honest
            # nested estimate, which is more than the whole of the previous
            # submission earned.
            "postprocess": {"thresholds": "per_class" if weighted != "none" else "none"},
            "model": {"type": "lightgbm", "params": {"n_estimators": 300}},
        },
        spec.kind,
        list(store.X.columns),
    )
    if errors:  # the vocabulary and the task disagree; the constant file still stands
        log(f"[bootstrap] baseline recipe rejected: {errors}")
        return {"status": "skipped", "errors": errors}

    started = time.perf_counter()
    try:
        result = evaluate(recipe, store)
    except Exception as err:  # noqa: BLE001
        log(f"[bootstrap] baseline failed: {type(err).__name__}: {err}")
        session.note(f"baseline failed: {err}")
        return {"status": "failed", "problem": str(err)}

    primary = score(spec.metric, store.y, result["oof"], result["oof_proba"], result["classes"])
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

    experiment = session.ledger.add(Experiment(
        id=session.ledger.next_id(), name=recipe["name"], source="baseline", score=primary,
        fold_scores=fold_scores, metric=spec.metric, direction=spec.direction,
        seconds=time.perf_counter() - started, note="automatic floor",
        detail={"recipe": recipe},
    ))
    frame = frame_from_predictions(spec, store.test[spec.id_column], result["test_pred"])
    saved = session.submissions.save(frame, score=primary, source=experiment.id, note=recipe["name"])

    log(f"[bootstrap] {spec.metric}={primary:.5f} in {experiment.seconds:.0f}s -> {saved['status']}")
    session.note(f"baseline {spec.metric}={primary:.5f}")
    return {"status": "ok", "score": primary, "submission": saved}
