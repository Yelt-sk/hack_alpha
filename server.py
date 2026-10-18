#!/usr/bin/env python
"""
MLForge MCP server.

Twelve tools, no more. Every extra tool is a way for a small model to get lost,
so the surface is kept narrow and each tool returns the smallest payload that
still lets the agent decide what to do next.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from mcp.server.mcpserver import MCPServer  # noqa: E402

from mlforge import data as D  # noqa: E402
from mlforge import guards as G  # noqa: E402
from mlforge import rag  # noqa: E402
from mlforge.pipeline import CompileError, compile_pipeline  # noqa: E402
from mlforge.schema import (LOWER_IS_BETTER, MODEL_REGISTRY, resolve_metric,  # noqa: E402
                            schema_doc, validate_recipe)
from mlforge.session import WORKSPACE as W  # noqa: E402
from mlforge.session import HOLDOUT_BUDGET, Dataset, Run, recipe_fingerprint  # noqa: E402
from mlforge.train import SCORERS, diagnose, make_cv, run_cv  # noqa: E402

INSTRUCTIONS = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8") \
    if (Path(__file__).parent / "prompts" / "system.md").exists() else ""

mcp = MCPServer(name="mlforge", version="0.1.0", instructions=INSTRUCTIONS)


def _fail(problem: str, hint: str, **extra) -> dict[str, Any]:
    return {"status": "error", "problem": problem, "hint": hint, **extra}


# ===========================================================================
# 1-2. Data
# ===========================================================================


@mcp.tool(description="List the datasets available to load, with their task type and a one-line description.")
def list_datasets() -> dict[str, Any]:
    return {
        "builtin": {k: {"task": v["task"], "needs_network": not v["offline"], "description": v["doc"]}
                    for k, v in D.CATALOG.items()},
        "custom": "Pass source='/path/to/file.csv' together with target='<column>' to load your own data.",
        "loaded": {k: {"task": v.task, "rows": len(v.train)} for k, v in W.datasets.items()},
    }


@mcp.tool(description=(
    "Load a dataset, split off a locked holdout, and return a profile plus a data-quality report. "
    "Always the first call. Declare group_column when rows repeat per entity, or time_column when "
    "predicting the future -- the server then enforces the matching validation scheme. "
    "trusted_columns is an operator-only escape hatch for columns flagged as leakage that really "
    "are available at prediction time; an agent must not set it on its own initiative."))
def load_dataset(
    source: str,
    target: str | None = None,
    dataset_id: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    trusted_columns: list[str] | None = None,
    holdout_fraction: float = 0.2,
    max_rows: int = 50000,
) -> dict[str, Any]:
    try:
        if source in D.CATALOG:
            df, tgt = D.load_builtin(source)
        elif Path(source).exists():
            if not target:
                return _fail(f"loading a CSV requires the target column", "Pass target='<column name>'.")
            df, tgt = D.load_csv(source, target)
        else:
            return _fail(f"unknown source '{source}'",
                         f"Use a built-in name {sorted(D.CATALOG)} or an existing CSV path.")
        if target and target in df.columns:
            tgt = target

        notes = []
        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=0).reset_index(drop=True)
            notes.append(f"Subsampled to {max_rows} rows to keep the iteration loop fast.")

        df = df[df[tgt].notna()].reset_index(drop=True)
        task = D.infer_task(df[tgt])
        train, holdout = D.split_holdout(df, tgt, task, frac=holdout_fraction)
        issues = D.detect_issues(train, tgt, task)

        did = dataset_id or (source if source in D.CATALOG else Path(source).stem)
        ds = Dataset(id=did, source=source, target=tgt, task=task, train=train, holdout=holdout,
                     issues=issues, group_column=group_column, time_column=time_column,
                     trusted_columns=list(trusted_columns or []))
        W.add_dataset(ds)
        W.save_meta()

        crit = [i for i in issues if i["severity"] == "critical"]
        return {
            "status": "ok",
            "dataset_id": did,
            "task": task,
            "target": tgt,
            "n_train": len(train),
            "n_holdout": len(holdout),
            "holdout_policy": f"locked; {HOLDOUT_BUDGET} final_evaluate calls available",
            "default_metric": resolve_metric({"metric": "auto"}, task),
            "enforced_validation": ("group_kfold" if group_column else
                                    "time_series" if time_column else "your choice"),
            "critical_issues": crit,
            "n_other_issues": len(issues) - len(crit),
            "notes": notes,
            "next_step": ("Resolve the critical issues above (usually drop_columns), then run a dummy "
                          "baseline." if crit else "Call profile_dataset, then run a dummy baseline."),
        }
    except Exception as e:
        return _fail(f"could not load '{source}': {e}", "Check list_datasets for valid sources.")


@mcp.tool(description="Column-level profile and the full data-quality report for a loaded dataset. Reads the training split only -- the holdout is never exposed.")
def profile_dataset(dataset_id: str, max_columns: int = 60) -> dict[str, Any]:
    try:
        ds = W.get_dataset(dataset_id)
    except KeyError as e:
        return _fail(str(e), "Call load_dataset first.")
    prof = D.profile_frame(ds.train, ds.target, ds.task, max_cols=max_columns)
    prof["issues"] = ds.issues
    prof["dataset_id"] = dataset_id
    return prof


# ===========================================================================
# 3-5. Knowledge and schema
# ===========================================================================


@mcp.tool(description=(
    "Search the ML playbook: decision rules for metrics, encoders, models, tuning and reading "
    "diagnostics. Consult this before guessing -- it is written for this server specifically."))
def search_knowledge(query: str, top_k: int = 3) -> dict[str, Any]:
    hits = rag.search(query, top_k=max(1, min(top_k, 8)))
    if not hits:
        return {"status": "no_match", "hint": "Try plainer wording, e.g. 'overfitting' or 'imbalanced classes'.",
                "available_topics": rag.corpus_stats()["topics"][:40]}
    return {"status": "ok", "query": query, "results": hits}


@mcp.tool(description="The full recipe schema: every field, every allowed value, the model list with their tunable parameters, and worked examples. Read this before your first run_experiment.")
def get_recipe_schema() -> dict[str, Any]:
    return schema_doc()


@mcp.tool(description="Check a recipe without training it. Fast. Returns every error at once with a fix for each. Use this when unsure rather than burning a training run.")
def validate_recipe_dry_run(recipe: dict, dataset_id: str | None = None) -> dict[str, Any]:
    task, cols = None, None
    if dataset_id and dataset_id in W.datasets:
        ds = W.get_dataset(dataset_id)
        task, cols = ds.task, list(ds.X.columns)
    norm, errors, warns = validate_recipe(recipe, task=task, columns=cols)
    if errors:
        return {"status": "invalid", "errors": errors, "warnings": warns}
    if dataset_id and dataset_id in W.datasets:
        try:
            _, cw = compile_pipeline(norm, W.get_dataset(dataset_id).X, task)
            warns = warns + cw
        except CompileError as e:
            return {"status": "invalid",
                    "errors": [{"field": "recipe", "problem": e.problem, "hint": e.hint}], "warnings": warns}
    return {"status": "valid", "warnings": warns, "normalised_recipe": norm}


# ===========================================================================
# 6. The core loop
# ===========================================================================


def _suggest_next(ds, run) -> str:
    """One concrete next action.

    A capable model infers this from the diagnosis. A 1.7B does not -- observed
    live, it re-submitted an identical recipe four times in nine steps. Naming the
    next move costs nothing and is what turns the loop from a repeat into a search.
    """
    runs = W.runs_for(ds.id)
    real = [r for r in runs if r.recipe["model"]["type"] != "dummy"]
    if not any(r.recipe["model"]["type"] == "dummy" for r in runs):
        return 'Run the dummy baseline first: {"name": "baseline", "model": {"type": "dummy"}}.'
    if run.tainted:
        return f"Call inspect_run with run_id='{run.id}' to find which feature produced this implausible score."
    if len(real) <= 1:
        leaks = [i["column"] for i in ds.issues
                 if i["kind"] == "target_leakage" and i["severity"] == "critical"]
        ids = [i["column"] for i in ds.issues if i["kind"] == "identifier"]
        drop = leaks + ids
        if drop and set(drop) - set(run.recipe.get("drop_columns") or []):
            return f'Drop the flagged columns and run again: "drop_columns": {drop}.'
        return ('Add a feature next, keeping everything else identical: '
                '"features": [{"op": "missing_count", "name": "n_missing"}].')
    if any(d.startswith("OVERFITTING") for d in run.diagnosis):
        return ('Regularise: keep the rest of the recipe and set "model": {"type": "lightgbm", '
                '"params": {"num_leaves": 15, "min_child_samples": 50, "reg_lambda": 5.0}}.')
    if len(real) >= 3:
        best = W.best_run(ds.id)
        if best:
            return (f"You have {len(real)} candidates and the best is {best.id}. Either call "
                    f"tune_hyperparams on it, or finish: final_evaluate(dataset_id='{ds.id}', "
                    f"run_id='{best.id}').")
    return ('Change exactly one thing and run again -- e.g. "categorical": {"encode": "target"}, '
            'or "model": {"type": "catboost"}.')


def _execute(ds, recipe: dict, overrides: list[str], justification: str,
             tuned_from: str | None = None) -> dict[str, Any]:
    acks = G.check_experiment(ds, recipe, overrides, justification)
    pipe, cw = compile_pipeline(recipe, ds.X, ds.task)
    groups = ds.train[recipe["validation"]["group_column"]] \
        if recipe["validation"].get("group_column") else None

    result = run_cv(pipe, ds.X, ds.y, ds.task, recipe, groups=groups)
    primary = resolve_metric(recipe, ds.task)
    score = result["metrics"][primary]["mean"]

    best = W.best_run(ds.id, primary)
    diagnosis = diagnose(result, primary, ds.task,
                         best.score if best else None, W.baseline_score(ds.id, primary))

    taint = G.taint_check(score, primary, ds.task)
    run = Run(id=W.new_run_id(), dataset_id=ds.id, recipe=recipe, primary_metric=primary,
              score=score, result=result, diagnosis=diagnosis, warnings=cw,
              fingerprint=recipe_fingerprint(recipe),
              tuned_from=tuned_from, tainted=bool(taint), taint_reason=taint or "",
              acknowledgements=acks)
    W.add_run(run)
    W.save_meta()

    out: dict[str, Any] = {
        "status": "ok",
        "run_id": run.id,
        "primary_metric": primary,
        "score": score,
        "std": result["metrics"][primary]["std"],
        "all_metrics": {k: v["mean"] for k, v in result["metrics"].items()},
        "train_score": result["metrics"][primary]["train_mean"],
        "cv": f"{result['cv_scheme']} x{result['folds']}",
        "fit_seconds": result["fit_seconds"],
        "diagnosis": diagnosis,
        "next_step": _suggest_next(ds, run),
    }
    if cw:
        out["warnings"] = cw
    if acks:
        out["acknowledged_guards"] = acks
    if taint:
        out["TAINTED"] = taint
        out["action_required"] = ("This run is excluded from the leaderboard's best until explained. "
                                  "Use inspect_run to find which feature is doing all the work.")
    return out


@mcp.tool(description=(
    "Train and honestly cross-validate one recipe, then score it. The central tool. "
    "Returns metrics, fold spread, train-vs-validation gap and a diagnosis telling you what to change next. "
    "Change one thing per experiment."))
def run_experiment(dataset_id: str, recipe: dict, override_guards: list[str] | None = None,
                   justification: str = "") -> dict[str, Any]:
    try:
        ds = W.get_dataset(dataset_id)
    except KeyError as e:
        return _fail(str(e), "Call load_dataset first.")

    norm, errors, warns = validate_recipe(recipe, task=ds.task, columns=list(ds.X.columns))
    if errors:
        return {"status": "invalid_recipe", "errors": errors, "warnings": warns,
                "hint": "Fix every error above and resubmit. Call get_recipe_schema if unsure."}
    dup = W.find_duplicate(dataset_id, recipe_fingerprint(norm))
    if dup is not None:
        return {"status": "duplicate",
                "problem": f"this is the same experiment as {dup.id} (only name/notes differ), "
                           f"already scored {dup.primary_metric}={dup.score}",
                "hint": "Re-running it cannot teach you anything. Change exactly one thing.",
                "existing_run_id": dup.id, "score": dup.score,
                "diagnosis": dup.diagnosis, "next_step": _suggest_next(ds, dup)}

    try:
        out = _execute(ds, norm, override_guards or [], justification)
        if warns:
            out["warnings"] = warns + out.get("warnings", [])
        return out
    except G.GuardBlock as e:
        return e.payload()
    except CompileError as e:
        return _fail(e.problem, e.hint)
    except Exception as e:
        return _fail(f"training failed: {type(e).__name__}: {e}",
                     "Try validate_recipe_dry_run, or simplify the recipe to isolate the cause.")


@mcp.tool(description=(
    "Search hyperparameters for a recipe with Optuna, using the same cross-validation as run_experiment "
    "so results are directly comparable. Tune only after the preprocessing is settled."))
def tune_hyperparams(dataset_id: str, recipe: dict, n_trials: int = 30,
                     time_budget_seconds: int = 120, override_guards: list[str] | None = None,
                     justification: str = "") -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        ds = W.get_dataset(dataset_id)
    except KeyError as e:
        return _fail(str(e), "Call load_dataset first.")

    norm, errors, _ = validate_recipe(recipe, task=ds.task, columns=list(ds.X.columns))
    if errors:
        return {"status": "invalid_recipe", "errors": errors}

    mtype = norm["model"]["type"]
    space = MODEL_REGISTRY[mtype]["space"]
    if not space:
        return _fail(f"model '{mtype}' has no tunable parameters",
                     "Tune lightgbm, catboost, hist_gbm, random_forest, extra_trees, logreg, ridge or knn.")

    primary = resolve_metric(norm, ds.task)
    lower = primary in LOWER_IS_BETTER
    try:
        G.check_experiment(ds, norm, override_guards or [], justification)
    except G.GuardBlock as e:
        return e.payload()

    import copy as _copy
    import time as _time

    # Optuna's `timeout` only stops *new* trials from starting -- a single trial
    # that samples n_estimators=1200 can overrun the whole budget on its own. So
    # time the base recipe once and shrink the expensive bounds to fit.
    space = _copy.deepcopy(space)
    budget_notes: list[str] = []
    try:
        _t0 = _time.perf_counter()
        _probe, _ = compile_pipeline(norm, ds.X, ds.task)
        _pg = ds.train[norm["validation"]["group_column"]] \
            if norm["validation"].get("group_column") else None
        run_cv(_probe, ds.X, ds.y, ds.task, norm, groups=_pg)
        base_seconds = max(_time.perf_counter() - _t0, 0.05)
    except Exception:
        base_seconds = None

    if base_seconds is not None:
        affordable = max(2.0, time_budget_seconds / max(1, n_trials))
        heavy = next((k for k in ("n_estimators", "iterations", "max_iter") if k in space), None)
        if heavy:
            ref = MODEL_REGISTRY[mtype]["defaults"].get(heavy, space[heavy]["high"])
            scale = affordable / base_seconds
            capped = int(max(space[heavy]["low"], min(space[heavy]["high"], ref * scale)))
            if capped < space[heavy]["high"]:
                budget_notes.append(
                    f"Capped {heavy} at {capped} (from {space[heavy]['high']}): one CV of the base "
                    f"recipe takes {base_seconds:.1f}s, leaving ~{affordable:.0f}s per trial.")
                space[heavy] = {**space[heavy], "high": capped}
        n_trials = max(3, min(n_trials, int(time_budget_seconds / max(affordable, 0.1)) + 2))

    def objective(trial):
        params = dict(norm["model"]["params"])
        for name, spec in space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            else:
                params[name] = trial.suggest_categorical(name, spec["choices"])
        cand = _copy.deepcopy(norm)
        cand["model"]["params"] = params
        pipe, _ = compile_pipeline(cand, ds.X, ds.task)
        groups = ds.train[cand["validation"]["group_column"]] \
            if cand["validation"].get("group_column") else None
        res = run_cv(pipe, ds.X, ds.y, ds.task, cand, groups=groups)
        return res["metrics"][primary]["mean"]

    study = optuna.create_study(direction="minimize" if lower else "maximize")
    _start = _time.perf_counter()

    def _stop_when_spent(st, _trial):
        if _time.perf_counter() - _start > time_budget_seconds:
            st.stop()

    try:
        study.optimize(objective, n_trials=n_trials, timeout=time_budget_seconds,
                       callbacks=[_stop_when_spent], catch=(Exception,), show_progress_bar=False)
    except Exception as e:
        return _fail(f"tuning failed: {e}", "Lower n_trials, or validate the base recipe first.")

    done = [t for t in study.trials if t.value is not None]
    if not done:
        return _fail("every tuning trial failed", "Run the base recipe through run_experiment to see the error.")

    best = _copy.deepcopy(norm)
    best["model"]["params"] = {**norm["model"]["params"], **study.best_params}
    best["name"] = f"{norm.get('name', 'recipe')}_tuned"
    best["notes"] = f"Optuna, {len(done)} trials. " + (norm.get("notes") or "")

    try:
        out = _execute(ds, best, override_guards or [], justification)
    except G.GuardBlock as e:
        return e.payload()

    out["tuning"] = {"trials_completed": len(done),
                     "search_seconds": round(_time.perf_counter() - _start, 1),
                     "budget_notes": budget_notes,
                     "best_params": study.best_params,
                     "best_trial_cv_score": round(float(study.best_value), 5),
                     "worst_trial_cv_score": round(float(min(t.value for t in done) if not lower
                                                        else max(t.value for t in done)), 5)}
    out["tuned_recipe"] = best
    return out


# ===========================================================================
# 7-9. Inspection
# ===========================================================================


@mcp.tool(description="Feature importances and per-fold detail for a run. Use it to find which column is driving a suspiciously good score, or which engineered features earned their place.")
def inspect_run(run_id: str, top_k: int = 15) -> dict[str, Any]:
    import numpy as np

    try:
        run = W.get_run(run_id)
    except KeyError as e:
        return _fail(str(e), "Call leaderboard to see the available run ids.")
    ds = W.get_dataset(run.dataset_id)

    fitted = W.fitted(run_id)
    if fitted is None:
        try:
            fitted, _ = compile_pipeline(run.recipe, ds.X, ds.task)
            fitted.fit(ds.X, ds.y)
            W._fitted[run_id] = fitted
        except Exception as e:
            return _fail(f"could not refit the run for inspection: {e}", "Re-run the recipe first.")

    # SanitizeNames is the last step before the model, so its output names are
    # exactly the model's input names. Walking get_feature_names_out() through the
    # whole pipeline is fragile -- the stateless FeatureEngineer has no such method.
    try:
        names = list(fitted.named_steps["names"].names_)
    except Exception:
        try:
            names = list(fitted[:-1].get_feature_names_out())
        except Exception:
            names = []
    model = fitted[-1]
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        c = np.asarray(model.coef_, dtype=float)
        imp = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)

    top: list[dict] = []
    if imp is not None and len(names) == len(imp):
        total = float(imp.sum()) or 1.0
        order = np.argsort(imp)[::-1][:top_k]
        top = [{"feature": names[i], "importance": round(float(imp[i]), 5),
                "share": round(float(imp[i]) / total, 4)} for i in order]

    out = {
        "run_id": run_id, "recipe": run.recipe, "primary_metric": run.primary_metric,
        "score": run.score, "fold_scores": run.result["metrics"][run.primary_metric]["folds"],
        "train_score": run.result["metrics"][run.primary_metric]["train_mean"],
        "all_metrics": {k: v["mean"] for k, v in run.result["metrics"].items()},
        "top_features": top, "diagnosis": run.diagnosis,
    }
    if run.tainted:
        out["TAINTED"] = run.taint_reason
    if top and top[0]["share"] > 0.5:
        out["concentration_warning"] = (
            f"'{top[0]['feature']}' accounts for {top[0]['share']:.0%} of total importance. "
            f"One column dominating this strongly is a classic leakage signature -- check that it is "
            f"available at prediction time.")
    if run.acknowledgements:
        out["acknowledged_guards"] = run.acknowledgements
    return out


@mcp.tool(description="All runs for a dataset, ranked, with the delta to the best. Tainted runs are listed but excluded from the top spot.")
def leaderboard(dataset_id: str) -> dict[str, Any]:
    try:
        ds = W.get_dataset(dataset_id)
    except KeyError as e:
        return _fail(str(e), "Call load_dataset first.")
    runs = W.runs_for(dataset_id)
    if not runs:
        return {"status": "empty", "hint": "No runs yet. Start with a dummy baseline."}

    primary = runs[0].primary_metric
    lower = primary in LOWER_IS_BETTER
    ranked = sorted([r for r in runs if r.primary_metric == primary],
                    key=lambda r: r.score, reverse=not lower)
    best = W.best_run(dataset_id, primary)

    rows = []
    for r in ranked:
        row = {"run_id": r.id, "name": r.recipe.get("name"), "model": r.recipe["model"]["type"],
               "encode": r.recipe["categorical"]["encode"], "score": r.score,
               "std": r.result["metrics"][primary]["std"], "notes": r.recipe.get("notes", "")[:120]}
        if best and r.id != best.id:
            row["delta_to_best"] = round((r.score - best.score) * (-1 if lower else 1), 5)
        if r.holdout_score is not None:
            row["holdout_score"] = r.holdout_score
        if r.tainted:
            row["TAINTED"] = r.taint_reason
        if r.acknowledgements:
            row["acknowledged_guards"] = r.acknowledgements
        rows.append(row)

    return {"dataset_id": dataset_id, "task": ds.task, "metric": primary,
            "higher_is_better": not lower, "n_runs": len(rows),
            "best_run_id": best.id if best else None,
            "baseline_score": W.baseline_score(dataset_id, primary),
            "holdout_budget_remaining": HOLDOUT_BUDGET - ds.holdout_used,
            "runs": rows}


# ===========================================================================
# 10-12. Finishing
# ===========================================================================


@mcp.tool(description=(
    "Score one chosen run on the locked holdout. Budget-limited and logged. This is a measurement, "
    "not a selection tool -- choose your model by cross-validation first, then confirm it here once."))
def final_evaluate(dataset_id: str, run_id: str, override_guards: list[str] | None = None,
                   justification: str = "") -> dict[str, Any]:
    import numpy as np

    try:
        ds, run = W.get_dataset(dataset_id), W.get_run(run_id)
    except KeyError as e:
        return _fail(str(e), "Check leaderboard for valid ids.")

    try:
        G.check_final_evaluate(ds, run, override_guards or [])
    except G.GuardBlock as e:
        return e.payload()

    n_real = len([r for r in W.runs_for(dataset_id) if r.recipe["model"]["type"] != "dummy"])
    if n_real < 2:
        return _fail("only one real model has been cross-validated so far",
                     "The holdout confirms a choice you already made. Run at least two candidate "
                     "recipes and compare them by cross-validation first.")

    try:
        pipe, _ = compile_pipeline(run.recipe, ds.X, ds.task)
        pipe.fit(ds.X, ds.y)
        Xh, yh = ds.holdout.drop(columns=[ds.target]), ds.holdout[ds.target]

        from sklearn import metrics as M
        scores: dict[str, float] = {}
        if ds.task == "regression":
            pred = pipe.predict(Xh)
            scores = {"rmse": float(np.sqrt(M.mean_squared_error(yh, pred))),
                      "mae": float(M.mean_absolute_error(yh, pred)),
                      "r2": float(M.r2_score(yh, pred))}
        else:
            pred = pipe.predict(Xh)
            scores["accuracy"] = float(M.accuracy_score(yh, pred))
            scores["balanced_accuracy"] = float(M.balanced_accuracy_score(yh, pred))
            if hasattr(pipe[-1], "predict_proba"):
                proba = pipe.predict_proba(Xh)
                if ds.task == "binary":
                    pos = list(pipe[-1].classes_)[-1]
                    scores["roc_auc"] = float(M.roc_auc_score((yh == pos).astype(int), proba[:, -1]))
                    scores["pr_auc"] = float(M.average_precision_score((yh == pos).astype(int), proba[:, -1]))
                    scores["f1"] = float(M.f1_score(yh, pred, pos_label=pos))
                else:
                    scores["macro_f1"] = float(M.f1_score(yh, pred, average="macro"))
                scores["logloss"] = float(M.log_loss(yh, proba, labels=list(pipe[-1].classes_)))

        scores = {k: round(v, 5) for k, v in scores.items()}
        primary = run.primary_metric
        hold = scores.get(primary)
        run.holdout_score = hold
        ds.holdout_used += 1
        ds.holdout_log.append({"run_id": run_id, "scores": scores, "justification": justification})
        W._fitted[run_id] = pipe
        W.save_meta()

        verdict = []
        if hold is not None:
            gap = abs(hold - run.score)
            tol = max(3 * run.result["metrics"][primary]["std"], 0.02 * max(abs(run.score), 1e-9))
            if gap <= tol:
                verdict.append(f"Holdout {primary}={hold:.4f} agrees with cross-validation ({run.score:.4f}). "
                               f"The estimate is trustworthy.")
            else:
                worse = (hold < run.score) if primary not in LOWER_IS_BETTER else (hold > run.score)
                verdict.append(
                    f"Holdout {primary}={hold:.4f} diverges from cross-validation ({run.score:.4f}) by {gap:.4f}. "
                    + ("The CV estimate was optimistic -- suspect leakage or selection on the validation set."
                       if worse else "The holdout is kinder than CV; most likely just a small-sample effect."))
        return {"status": "ok", "run_id": run_id, "holdout_scores": scores,
                "cv_score": run.score, "primary_metric": primary, "verdict": verdict,
                "holdout_budget_remaining": HOLDOUT_BUDGET - ds.holdout_used}
    except Exception as e:
        return _fail(f"holdout evaluation failed: {type(e).__name__}: {e}", "Re-run the recipe first.")


@mcp.tool(description="Export a run as a standalone Python script plus a serialised model and a markdown report. The script reproduces the pipeline without this server.")
def export_solution(run_id: str, output_dir: str | None = None) -> dict[str, Any]:
    import json

    import joblib

    from mlforge.session import ROOT

    try:
        run = W.get_run(run_id)
        ds = W.get_dataset(run.dataset_id)
    except KeyError as e:
        return _fail(str(e), "Check leaderboard for valid run ids.")

    # A relative output_dir from the agent is resolved under the workspace, not the
    # server's cwd -- the agent should not be scattering artifacts around the repo.
    if output_dir:
        candidate = Path(output_dir)
        out = candidate if candidate.is_absolute() else ROOT / candidate
    else:
        out = ROOT / f"solution_{run_id}"
    out.mkdir(parents=True, exist_ok=True)

    pipe = W.fitted(run_id)
    if pipe is None:
        pipe, _ = compile_pipeline(run.recipe, ds.X, ds.task)
        pipe.fit(ds.X, ds.y)
    joblib.dump(pipe, out / "model.joblib")
    (out / "recipe.json").write_text(json.dumps(run.recipe, indent=2))

    (out / "train.py").write_text(f'''"""Standalone reproduction of {run_id}. Generated by MLForge."""
import json, sys
from pathlib import Path

import joblib
import pandas as pd

# Locate the MLForge checkout relative to this file rather than hard-coding an
# absolute path -- an exported solution should be portable, and should not carry
# the directory layout of the machine that produced it.
for _candidate in (Path(__file__).resolve().parents[2], Path.cwd()):
    if (_candidate / "mlforge").is_dir():
        sys.path.insert(0, str(_candidate))
        break
from mlforge.pipeline import compile_pipeline

RECIPE = json.loads(Path(__file__).with_name("recipe.json").read_text())
TARGET = {ds.target!r}
TASK = {ds.task!r}


def train(csv_path: str):
    df = pd.read_csv(csv_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pipe, _ = compile_pipeline(RECIPE, X, TASK)
    pipe.fit(X, y)
    joblib.dump(pipe, "model.joblib")
    return pipe


def predict(csv_path: str):
    pipe = joblib.load(Path(__file__).with_name("model.joblib"))
    return pipe.predict(pd.read_csv(csv_path))


if __name__ == "__main__":
    print(train(sys.argv[1]) if len(sys.argv) > 1 else "usage: train.py <train.csv>")
''')

    m = run.result["metrics"]
    lines = [
        f"# Solution report -- {run_id}", "",
        f"- **Dataset**: `{ds.id}` ({ds.task}, target `{ds.target}`, {len(ds.train)} training rows)",
        f"- **Model**: {run.recipe['model']['type']}",
        f"- **Validation**: {run.result['cv_scheme']} x{run.result['folds']}, seed {run.result['seed']}",
        f"- **Primary metric**: {run.primary_metric} = **{run.score:.5f}** "
        f"(+/- {m[run.primary_metric]['std']:.5f} across folds)",
    ]
    if run.holdout_score is not None:
        lines.append(f"- **Held-out score**: {run.primary_metric} = **{run.holdout_score:.5f}** "
                     f"(never seen during development)")
    else:
        lines.append("- **Held-out score**: not evaluated")
    lines += ["", "## Cross-validated metrics", "",
              "| metric | mean | std | train |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean']:.5f} | {v['std']:.5f} | {v['train_mean']:.5f} |" for k, v in m.items()]
    lines += ["", "## Recipe", "", "```json", json.dumps(run.recipe, indent=2), "```", "",
              "## Diagnosis", ""] + [f"- {d}" for d in run.diagnosis]
    if run.acknowledgements:
        lines += ["", "## Overridden guards", ""] + [f"- {a}" for a in run.acknowledgements]
    if run.tainted:
        lines += ["", f"> **Flagged as implausible**: {run.taint_reason}"]
    if ds.issues:
        lines += ["", "## Data-quality findings at load time", ""]
        lines += [f"- **{i['severity']}** `{i['column']}`: {i['detail']}" for i in ds.issues[:15]]
    (out / "report.md").write_text("\n".join(lines))

    return {"status": "ok", "output_dir": str(out),
            "files": ["model.joblib", "recipe.json", "train.py", "report.md"],
            "cv_score": run.score, "holdout_score": run.holdout_score}


if __name__ == "__main__":
    mcp.run("stdio")
