"""Experiment tools: the declarative fast path, tuning, and the bridge to free code."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

import agent_guards as guards
from agent_metrics import report, score
import agent_recipe_schema as recipe_schema
from agent_recipe_pipeline import CompileError, evaluate
from agent_session import session
from agent_state import Experiment
from agent_submission import frame_from_predictions
from agent_registry import tool


def _register(name: str, source: str, primary: float, fold_scores: list[float],
              seconds: float, detail: dict[str, Any], note: str) -> Experiment:
    current = session()
    spec = current.spec
    experiment = Experiment(
        id=current.ledger.next_id(), name=name, source=source, score=primary,
        fold_scores=fold_scores, metric=spec.metric, direction=spec.direction,
        seconds=seconds, detail=detail, note=note,
        tainted=guards.implausible(primary, spec.metric),
    )
    return current.ledger.add(experiment)


def _fold_scores(oof: np.ndarray, oof_proba: np.ndarray | None, classes: list | None) -> list[float]:
    """Score each fold separately, so the spread is real rather than assumed."""
    current = session()
    spec, store = current.spec, current.store
    out: list[float] = []
    for fold in range(store.n_splits):
        mask = store.fold_id == fold
        try:
            out.append(
                score(spec.metric, store.y[mask], oof[mask],
                      oof_proba[mask] if oof_proba is not None else None, classes)
            )
        except Exception:
            continue
    return out


def _finish(experiment: Experiment, result: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Score, record, and promote the submission if this is the best so far."""
    current = session()
    spec = current.spec
    payload: dict[str, Any] = {
        "status": "ok",
        "experiment_id": experiment.id,
        "metric": spec.metric,
        "score": round(experiment.score, 6),
        "fold_std": round(experiment.std, 6),
        "fold_scores": [round(v, 5) for v in experiment.fold_scores],
        "all_metrics": experiment.detail.get("all_metrics", {}),
        "seconds": round(experiment.seconds, 1),
        "verdict": current.ledger.verdict(experiment),
        "time": current.clock.summary(),
    }
    if experiment.detail.get("thresholds"):
        payload["thresholds"] = experiment.detail["thresholds"]
    if warnings:
        payload["warnings"] = warnings
    if experiment.tainted:
        payload["TAINTED"] = experiment.tainted
        payload["action_required"] = (
            "This result is excluded from the best until explained. Use inspect_columns on the "
            "features with the highest predictive power to find the leak."
        )
        return payload

    predictions = result.get("test_pred")
    if predictions is not None:
        # Keep the test predictions on the experiment so later tools -- blending,
        # inspection, a manual re-save -- can reach them without a refit.
        experiment.detail["test_pred"] = np.asarray(predictions)
        frame = frame_from_predictions(spec, current.store.test[spec.id_column], predictions)
        payload["submission"] = current.submissions.save(
            frame, score=experiment.score, source=experiment.id, note=experiment.name
        )
    return payload


def _untried(current) -> dict[str, Any]:
    """What has not been tried yet, as concrete next moves.

    A bare "you already ran this" is read by a weak model as "try the adjacent
    one", and it oscillates. Observed live: fourteen consecutive duplicates
    alternating between two recipes. Naming the unexplored axes turns the
    refusal into a direction.
    """
    spec = current.spec
    recipes = [e.detail["recipe"] for e in current.ledger.entries if e.detail.get("recipe")]

    tried_models = {r["model"]["type"] for r in recipes}
    tried_encoders = {r["categorical"]["encode"] for r in recipes}
    usable = [m for m, meta in recipe_schema.MODELS.items()
              if spec.kind in meta["tasks"] and m not in tried_models and m != "dummy"]

    moves: list[str] = []
    if usable:
        moves.append(f'a different model: "model": {{"type": "{usable[0]}"}} '
                     f"(untried: {sorted(usable)})")
    untried_encoders = [e for e in recipe_schema.CAT_ENCODE if e not in tried_encoders]
    if untried_encoders:
        moves.append(f'a different encoder: "categorical": {{"encode": "{untried_encoders[0]}"}}')
    if not any(r["features"] for r in recipes):
        moves.append('engineered features, e.g. "features": [{"op": "missing_count", "name": "n_missing"}]')
    if spec.kind == "regression" and not any(r["target_transform"] != "none" for r in recipes):
        moves.append('a target transform: "target_transform": "log1p"')
    if spec.is_classification and not any(r["class_weight"] != "none" for r in recipes):
        moves.append('"class_weight": "balanced"')
    if not any(e.name.endswith("_tuned") for e in current.ledger.entries):
        moves.append("tune_recipe on the best recipe so far")

    return {"tried_models": sorted(tried_models), "untried_moves": moves[:4]}


@tool
def get_recipe_schema() -> dict[str, Any]:
    """The full recipe vocabulary: every field, every allowed value, the model list, examples.

    Read it once before your first `run_recipe`. Note that the validation scheme
    is not part of a recipe -- folds are fixed for the whole run so that every
    result, including ones produced by your own code, is directly comparable.
    """
    return recipe_schema.describe()


@tool
def run_recipe(recipe: dict, note: str = "") -> dict[str, Any]:
    """Compile a recipe, cross-validate it honestly, score it and record the result.

    The central tool of the standard regime. Returns the metric, the spread across
    folds, a verdict saying whether the change beats the current best by more than
    noise, and -- when it is the best so far -- saves the submission for you.

    Change one thing per experiment, or a better score will not tell you which
    change caused it.

    Args:
        recipe: The experiment, as described by get_recipe_schema.
        note: Why this differs from the previous attempt.
    """
    current = session()
    spec, store, clock = current.spec, current.store, current.clock

    normalised, errors, warnings = recipe_schema.validate(
        recipe, spec.kind, list(store.X.columns)
    )
    if errors:
        payload: dict[str, Any] = {
            "status": "invalid_recipe", "errors": errors, "warnings": warnings,
            "hint": "Call get_recipe_schema if the shape is unclear.",
        }
        # Offer something the model can send back verbatim. Naming the problem is
        # not enough: observed live, a model given only the list of legal keys
        # resubmitted the same invented shape three times in a row.
        candidate = {k: v for k, v in normalised.items() if k in recipe_schema.DEFAULT_RECIPE}
        _, residual, _ = recipe_schema.validate(candidate, spec.kind, list(store.X.columns))
        if not residual:
            payload["corrected_recipe"] = candidate
            payload["next_step"] = "Call run_recipe again with corrected_recipe exactly as given."
        else:
            payload["minimal_example"] = {"name": "lgbm", "drop_columns": [spec.id_column],
                                          "model": {"type": "lightgbm"}}
        return payload

    fingerprint = json.dumps(
        {k: v for k, v in normalised.items() if k not in ("name", "notes")},
        sort_keys=True, default=str,
    )
    previous = next(
        (e for e in current.ledger.entries if e.detail.get("fingerprint") == fingerprint), None
    )
    if previous is not None:
        current.duplicates += 1
        payload = {
            "status": "duplicate",
            "problem": f"this is the same experiment as {previous.id} (only the name differs), "
                       f"already scored {spec.metric}={previous.score:.6f}",
            "existing": previous.row(),
            **_untried(current),
        }
        if current.duplicates >= 3:
            # Repeating the same refusal has not worked; be directive instead.
            payload["hint"] = (
                f"You have now resubmitted an already-scored experiment {current.duplicates} times. "
                f"Alternating between two recipes you have both already run is not a search. "
                f"Pick one untried move from the list and send it."
            )
            payload["next_step"] = "Choose one item from untried_moves and call run_recipe with it."
        else:
            # Deliberately not "change one thing": taken literally against a
            # two-recipe history, that instruction walks straight into the other
            # one. Point at what has not been tried instead.
            payload["hint"] = (
                "Re-running a scored experiment teaches you nothing. Take one of untried_moves."
            )
            payload["next_step"] = current.ledger.verdict(previous)
        return payload

    if spec.id_column not in (normalised["drop_columns"] or []) and not current.warned_about_id:
        # Once, not every run. Repeated on every call it became half of a
        # two-state loop: this warning pushed the id out, and "change exactly one
        # thing" pushed it back in.
        current.warned_about_id = True
        warnings.append(
            f"'{spec.id_column}' is the row identifier and is still a feature; "
            f"add it to drop_columns unless you have a reason to keep it"
        )
    if clock.must_finalize:
        return {"status": "refused", "problem": "the time reserve has been reached",
                "hint": "Finalise now: check leaderboard and make sure the best result is saved."}

    started = time.perf_counter()
    try:
        result = evaluate(normalised, store, time_budget_s=clock.usable_remaining * 0.5)
    except CompileError as err:
        return {"status": "error", "problem": err.problem, "hint": err.hint}
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "problem": f"{type(err).__name__}: {err}",
                "hint": "Simplify the recipe to isolate the cause, or try run_python."}

    primary = score(spec.metric, store.y, result["oof"], result["oof_proba"], result["classes"])
    detail = {
        "all_metrics": report(spec.kind, store.y, result["oof"], result["oof_proba"], result["classes"]),
        "recipe": normalised,
        "fingerprint": fingerprint,
        "thresholds": result.get("thresholds") or {},
    }
    experiment = _register(
        normalised["name"], "recipe", primary,
        _fold_scores(result["oof"], result["oof_proba"], result["classes"]),
        time.perf_counter() - started, detail, note or normalised.get("notes", ""),
    )
    return _finish(experiment, result, warnings + result.get("warnings", []))


@tool
def tune_recipe(recipe: dict, time_budget_s: int = 600, n_trials: int = 40) -> dict[str, Any]:
    """Search hyperparameters for a recipe with Optuna, on the same folds.

    Tune only once the preprocessing is settled: on tabular data a sensible
    pipeline change usually moves the metric far more than tuning does.

    The search space is the one registered for the model. The budget is honoured
    by shrinking the expensive bounds first -- Optuna's own timeout only stops new
    trials, so a single huge trial would otherwise overrun it.

    Args:
        recipe: The base recipe to search around.
        time_budget_s: Wall-clock seconds for the whole search.
        n_trials: Upper bound on trials; the budget usually binds first.
    """
    import copy

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    current = session()
    spec, store, clock = current.spec, current.store, current.clock

    normalised, errors, _ = recipe_schema.validate(recipe, spec.kind, list(store.X.columns))
    if errors:
        return {"status": "invalid_recipe", "errors": errors}

    model_type = normalised["model"]["type"]
    space = recipe_schema.MODELS[model_type]["space"]
    if not space:
        return {"status": "error", "problem": f"'{model_type}' has no tunable parameters",
                "hint": "Tune lightgbm, catboost, xgboost, hist_gbm, random_forest, ridge or logreg."}

    budget = min(time_budget_s, max(60.0, clock.usable_remaining * 0.6))
    if budget < 60:
        return {"status": "refused", "problem": "not enough time remains to tune",
                "hint": "Finalise with the best recipe you already have."}

    # What a single evaluation costs is already known: every experiment in the
    # ledger timed one. Checking it here, before the probe, is the difference
    # between saying "you will get four trials, not forty" and discovering it
    # after a quarter of an hour. Observed twice: a request for twenty-five
    # trials that managed two in 473 seconds and was then refused on the
    # reserve, and a request for forty that spent seventeen minutes to gain
    # 0.004.
    timed = sorted(e.seconds for e in current.ledger.entries if e.seconds > 0)
    if timed:
        per_trial = timed[len(timed) // 2]
        affordable = int(budget // per_trial)
        if affordable < 3:
            return {
                "status": "refused",
                "problem": (
                    f"one evaluation of this task costs about {per_trial:.0f}s, so "
                    f"{budget:.0f}s of budget buys {affordable} trials -- too few to "
                    f"search anything"
                ),
                "hint": (
                    "Tuning is the wrong move with this much time left. Change the "
                    "recipe itself, or finalise what you have."
                ),
                "seconds_per_trial": round(per_trial, 1),
            }
        if affordable < n_trials:
            notes_upfront = (
                f"asked for {n_trials} trials; at about {per_trial:.0f}s each only "
                f"{affordable} fit in the budget, so the search will stop there"
            )
        else:
            notes_upfront = ""
    else:
        per_trial, affordable, notes_upfront = None, None, ""

    space = copy.deepcopy(space)
    notes: list[str] = [notes_upfront] if notes_upfront else []
    probe_started = time.perf_counter()
    try:
        evaluate(normalised, store)
        base_seconds = max(time.perf_counter() - probe_started, 0.1)
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "problem": f"the base recipe failed: {err}",
                "hint": "Run it through run_recipe first and fix it there."}

    affordable = max(base_seconds, budget / max(1, n_trials))
    heavy = next((k for k in ("n_estimators", "iterations", "max_iter") if k in space), None)
    if heavy:
        reference = recipe_schema.MODELS[model_type]["defaults"].get(heavy, space[heavy]["high"])
        capped = int(max(space[heavy]["low"], min(space[heavy]["high"], reference * affordable / base_seconds)))
        if capped < space[heavy]["high"]:
            space[heavy] = {**space[heavy], "high": capped}
            notes.append(f"capped {heavy} at {capped}: one evaluation of the base recipe takes "
                         f"{base_seconds:.0f}s")
    n_trials = max(3, min(n_trials, int(budget / affordable) + 1))

    def objective(trial: Any) -> float:
        params = dict(normalised["model"].get("params") or {})
        for name, bounds in space.items():
            if bounds["type"] == "float":
                params[name] = trial.suggest_float(name, bounds["low"], bounds["high"],
                                                   log=bounds.get("log", False))
            elif bounds["type"] == "int":
                params[name] = trial.suggest_int(name, bounds["low"], bounds["high"])
            else:
                params[name] = trial.suggest_categorical(name, bounds["choices"])
        candidate = copy.deepcopy(normalised)
        candidate["model"]["params"] = params
        outcome = evaluate(candidate, store)
        return score(spec.metric, store.y, outcome["oof"], outcome["oof_proba"], outcome["classes"])

    started = time.perf_counter()
    study = optuna.create_study(direction="minimize" if spec.direction == "minimize" else "maximize")
    study.optimize(
        objective, n_trials=n_trials, timeout=budget,
        callbacks=[lambda s, _t: s.stop() if time.perf_counter() - started > budget else None],
        catch=(Exception,), show_progress_bar=False,
    )

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        return {"status": "error", "problem": "every trial failed",
                "hint": "Run the base recipe through run_recipe to see the error."}

    best_recipe = copy.deepcopy(normalised)
    best_recipe["model"]["params"] = {**(normalised["model"].get("params") or {}), **study.best_params}
    best_recipe["name"] = f"{normalised.get('name', 'recipe')}_tuned"

    outcome = run_recipe(best_recipe, note=f"Optuna, {len(completed)} trials")
    outcome["tuning"] = {
        "trials": len(completed),
        "search_seconds": round(time.perf_counter() - started, 1),
        "best_params": study.best_params,
        "budget_notes": notes,
    }
    outcome["tuned_recipe"] = best_recipe
    return outcome


@tool(regimes=["custom"])
def evaluate_predictions(oof_csv: str, test_csv: str, name: str, note: str = "") -> dict[str, Any]:
    """Score predictions your own code produced, on the run's shared folds.

    This is the bridge between writing code and the leaderboard. Without it a
    hand-written result is not comparable with a recipe, and you cannot tell
    which of the two is actually better.

    Both files need the id column plus one prediction column. The out-of-fold
    file must cover every training row exactly once, each row predicted by a model
    that did not see it -- use the fold assignment in folds_csv from get_context.

    Args:
        oof_csv: Path to out-of-fold predictions for the training rows.
        test_csv: Path to predictions for the test rows.
        name: Short name for this experiment, shown in the leaderboard.
        note: What this approach does differently.
    """
    current = session()
    spec, store = current.spec, current.store
    started = time.perf_counter()

    try:
        oof_frame = pd.read_csv(oof_csv)
        test_frame = pd.read_csv(test_csv)
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "problem": f"could not read the files: {err}",
                "hint": "Write them with pandas to_csv(index=False)."}

    for label, frame in (("oof", oof_frame), ("test", test_frame)):
        if spec.id_column not in frame.columns or frame.shape[1] < 2:
            return {"status": "error",
                    "problem": f"{label} file needs '{spec.id_column}' and one prediction column",
                    "hint": f"Columns found: {list(frame.columns)[:10]}"}

    prediction_column = [c for c in oof_frame.columns if c != spec.id_column][0]
    aligned = store.train[[spec.id_column]].merge(
        oof_frame[[spec.id_column, prediction_column]], on=spec.id_column, how="left"
    )
    if aligned[prediction_column].isna().any():
        missing = int(aligned[prediction_column].isna().sum())
        return {"status": "error",
                "problem": f"out-of-fold predictions cover only {len(aligned) - missing} of "
                           f"{len(aligned)} training rows",
                "hint": "Every training row needs a prediction from a model that did not train on it."}

    oof = aligned[prediction_column].to_numpy()
    try:
        primary = score(spec.metric, store.y, oof, None, spec.classes)
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "problem": f"the predictions could not be scored: {err}",
                "hint": f"For this task the column must hold {spec.kind} predictions."}

    test_column = [c for c in test_frame.columns if c != spec.id_column][0]
    test_aligned = store.test[[spec.id_column]].merge(
        test_frame[[spec.id_column, test_column]], on=spec.id_column, how="left"
    )

    detail = {"all_metrics": report(spec.kind, store.y, oof, None, spec.classes), "source_files":
              {"oof": oof_csv, "test": test_csv}}
    experiment = _register(name, "custom", primary, _fold_scores(oof, None, spec.classes),
                           time.perf_counter() - started, detail, note)
    return _finish(experiment, {"test_pred": test_aligned[test_column].to_numpy()}, [])
