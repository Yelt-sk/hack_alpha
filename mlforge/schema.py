"""
Recipe schema -- the narrowed, LLM-facing surface over sklearn / LightGBM / CatBoost.

Design rule: the agent never writes training code. It emits a JSON *recipe*.
Everything that is easy to get subtly wrong (fold-safe encoding, CV scheme,
metric choice, holdout hygiene) lives in this server, not in the agent's head.
That is what lets a small local model drive the loop.

Every rejection carries a `hint` -- a weak model recovers from "did you mean X?"
far better than from a stack trace.
"""
from __future__ import annotations

import copy
import os
from typing import Any

# n_jobs=-1 is actively harmful here: with cross-validation already iterating
# folds, letting every booster claim all cores oversubscribes them badly.
# Measured on an 8-core box: n_jobs=-1 took 47s for a 5-fold LightGBM run that
# n_jobs=4 finished in 2.2s, for identical scores. Cap it at half the cores.
DEFAULT_N_JOBS = max(1, min(4, (os.cpu_count() or 2) // 2))

# --------------------------------------------------------------------------
# Vocabularies. These are deliberately small: every extra option is a way for
# the agent to get lost. Anything not listed here is simply not expressible.
# --------------------------------------------------------------------------

NUM_IMPUTE = ["median", "mean", "constant", "none"]
NUM_SCALE = ["none", "standard", "robust", "minmax"]
CAT_IMPUTE = ["most_frequent", "constant", "none"]
CAT_ENCODE = ["native", "onehot", "ordinal", "target", "frequency"]
CV_SCHEMES = ["auto", "kfold", "stratified_kfold", "group_kfold", "time_series"]

TASKS = ["binary", "multiclass", "regression"]

METRICS = {
    "binary": ["roc_auc", "pr_auc", "f1", "accuracy", "balanced_accuracy", "logloss"],
    "multiclass": ["accuracy", "macro_f1", "balanced_accuracy", "logloss"],
    "regression": ["rmse", "mae", "r2", "mape"],
}
DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "macro_f1", "regression": "rmse"}

# Metrics where lower is better. Everything else: higher is better.
LOWER_IS_BETTER = {"logloss", "rmse", "mae", "mape"}

# --------------------------------------------------------------------------
# Feature engineering ops. All of them are *stateless* row-wise transforms --
# they cannot leak across folds by construction. That is the entire reason the
# list is this short; anything requiring fitted state must be an encoder.
# --------------------------------------------------------------------------

FEATURE_OPS = {
    "ratio":          {"args": ["a", "b"], "doc": "a / (b + eps). Good for rates/densities."},
    "product":        {"args": ["a", "b"], "doc": "a * b. Captures simple interactions."},
    "diff":           {"args": ["a", "b"], "doc": "a - b. Good for before/after pairs."},
    "sum":            {"args": ["a", "b"], "doc": "a + b. Totals from components."},
    "log1p":          {"args": ["a"],      "doc": "log(1+a) for non-negative skewed columns."},
    "datetime_parts": {"args": ["a"],      "doc": "Expands a datetime column into year/month/day/dayofweek/hour."},
    "missing_count":  {"args": [],         "doc": "Per-row count of missing values; missingness is often signal."},
}

# --------------------------------------------------------------------------
# Model registry. `tasks` gates which targets a model may be used on; `space`
# is the search space tune_hyperparams() walks -- only params that actually
# move the needle, so a small budget is spent where it matters.
# --------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "dummy": {
        "tasks": ["binary", "multiclass", "regression"],
        "defaults": {},
        "space": {},
        "handles_native_cat": False,
        "doc": "Constant/prior baseline. Always run this first: it is the floor any real model must clear.",
    },
    "logreg": {
        "tasks": ["binary", "multiclass"],
        "defaults": {"C": 1.0, "max_iter": 2000},
        "space": {"C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True}},
        "handles_native_cat": False,
        "needs_scaling": True,
        "doc": "Logistic regression. Strong, fast, well-calibrated linear baseline. Needs scaling + one-hot.",
    },
    "ridge": {
        "tasks": ["regression"],
        "defaults": {"alpha": 1.0},
        "space": {"alpha": {"type": "float", "low": 1e-3, "high": 1e3, "log": True}},
        "handles_native_cat": False,
        "needs_scaling": True,
        "doc": "L2 linear regression. Fast sanity check against tree models.",
    },
    "random_forest": {
        "tasks": ["binary", "multiclass", "regression"],
        "defaults": {"n_estimators": 300, "n_jobs": DEFAULT_N_JOBS},
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 800},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "max_features": {"type": "cat", "choices": ["sqrt", "log2", 0.5, 0.8]},
        },
        "handles_native_cat": False,
        "doc": "Bagged trees. Very hard to overfit, decent default, rarely the winner on tabular.",
    },
    "extra_trees": {
        "tasks": ["binary", "multiclass", "regression"],
        "defaults": {"n_estimators": 300, "n_jobs": DEFAULT_N_JOBS},
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 800},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        },
        "handles_native_cat": False,
        "doc": "Extremely randomised trees. Lower variance than RF, useful in ensembles.",
    },
    "hist_gbm": {
        "tasks": ["binary", "multiclass", "regression"],
        "defaults": {"max_iter": 300, "early_stopping": False},
        "space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 800},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 255},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
            "l2_regularization": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        "handles_native_cat": False,
        "doc": "sklearn's histogram gradient boosting. No extra dependency; close to LightGBM in quality.",
    },
    "lightgbm": {
        "tasks": ["binary", "multiclass", "regression"],
        # importance_type="gain": the default "split" counts how often a feature was
        # used, which badly understates a leak -- one split on it separates the
        # classes perfectly. Gain is what makes inspect_run able to name the culprit.
        "defaults": {"n_estimators": 400, "learning_rate": 0.05, "n_jobs": DEFAULT_N_JOBS,
                     "verbose": -1, "importance_type": "gain"},
        "space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "n_estimators": {"type": "int", "low": 100, "high": 1200},
            "num_leaves": {"type": "int", "low": 15, "high": 255},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 20.0, "log": True},
        },
        "handles_native_cat": True,
        "doc": "LightGBM. The default first choice for tabular data. Handles categoricals natively (encode='native').",
    },
    "catboost": {
        "tasks": ["binary", "multiclass", "regression"],
        "defaults": {"iterations": 500, "verbose": 0, "allow_writing_files": False},
        "space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "iterations": {"type": "int", "low": 200, "high": 1500},
            "depth": {"type": "int", "low": 4, "high": 10},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 30.0, "log": True},
        },
        "handles_native_cat": True,
        "doc": "CatBoost. Usually the strongest option when there are many high-cardinality categoricals.",
    },
    "knn": {
        "tasks": ["binary", "multiclass", "regression"],
        "defaults": {"n_neighbors": 15, "n_jobs": DEFAULT_N_JOBS},
        "space": {
            "n_neighbors": {"type": "int", "low": 3, "high": 60},
            "weights": {"type": "cat", "choices": ["uniform", "distance"]},
        },
        "handles_native_cat": False,
        "needs_scaling": True,
        "doc": "k-NN. Rarely wins, but a useful diagnostic: if it is competitive, the signal is local/geometric.",
    },
}

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

DEFAULT_RECIPE: dict[str, Any] = {
    "name": "unnamed",
    "notes": "",
    "drop_columns": [],
    "features": [],
    "numeric": {"impute": "median", "scale": "none", "winsorize": None},
    "categorical": {
        "impute": "most_frequent",
        "encode": "native",
        "max_onehot_cardinality": 25,
        "rare_threshold": 0.01,
    },
    "class_weight": "none",
    "model": {"type": "lightgbm", "params": {}},
    "validation": {"scheme": "auto", "folds": 5, "seed": 42, "group_column": None},
    "metric": "auto",
}


def _err(field: str, problem: str, hint: str) -> dict[str, str]:
    return {"field": field, "problem": problem, "hint": hint}


def _closest(value: str, options: list[str]) -> str | None:
    """Cheap did-you-mean. Keeps a confused model from guessing twice."""
    import difflib

    m = difflib.get_close_matches(str(value), options, n=1, cutoff=0.5)
    return m[0] if m else None


def _enum(value: Any, options: list[str], field: str, errors: list) -> None:
    if value not in options:
        hint = f"Allowed: {options}."
        near = _closest(value, options)
        if near:
            hint = f"Did you mean '{near}'? Allowed: {options}."
        errors.append(_err(field, f"'{value}' is not a valid value", hint))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# Params a small model habitually hoists to the top level instead of nesting
# them under model.params.
_MODEL_PARAM_HINTS = {
    "learning_rate", "n_estimators", "num_leaves", "max_depth", "min_child_samples",
    "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "iterations", "depth",
    "l2_leaf_reg", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
    "n_neighbors", "weights", "max_features", "C", "alpha",
}


def repair_shape(recipe: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fix the shape mistakes small models make, and say so out loud.

    A 1.7B model reliably emits {"type": "lightgbm", "learning_rate": 0.05} instead
    of {"model": {"type": "lightgbm", "params": {"learning_rate": 0.05}}}. The intent
    is unambiguous, so rejecting it just burns the model's budget in a retry loop --
    observed live: eighteen calls, all rejected, never recovered. Repair it and
    return a warning: repaired, never silent.
    """
    if not isinstance(recipe, dict):
        return recipe, []
    r = copy.deepcopy(recipe)
    repairs: list[str] = []

    if isinstance(r.get("model"), str):
        r["model"] = {"type": r["model"]}
        repairs.append('model was a string; read it as {"model": {"type": ...}}.')
    if not isinstance(r.get("model"), dict):
        r["model"] = {}

    if "type" in r and isinstance(r["type"], str) and r["type"] in MODEL_REGISTRY:
        r["model"].setdefault("type", r.pop("type"))
        repairs.append('moved top-level "type" into model.type.')

    # a whole params/hyperparameters dict parked at the top level
    for key in ("params", "model_params", "hyperparameters"):
        if isinstance(r.get(key), dict):
            merged = dict(r["model"].get("params") or {})
            merged.update(r.pop(key))
            r["model"]["params"] = merged
            repairs.append(f'moved the top-level "{key}" object into model.params.')

    hoisted = {k: r.pop(k) for k in list(r) if k in _MODEL_PARAM_HINTS}
    if hoisted:
        params = dict(r["model"].get("params") or {})
        params.update(hoisted)
        r["model"]["params"] = params
        repairs.append(f"moved {sorted(hoisted)} into model.params.")

    if repairs:
        repairs.append(
            'Correct shape: {"name": "...", "model": {"type": "<model>", "params": {...}}}. '
            "Use this shape from now on.")
    return r, repairs


def validate_recipe(
    recipe: dict[str, Any],
    task: str | None = None,
    columns: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict], list[str]]:
    """Normalise a recipe and report everything wrong with it at once.

    Returns (normalised_recipe, errors, warnings). Reporting *all* errors in one
    pass matters: a model that gets one error per round trip burns its budget on
    syntax instead of on modelling.
    """
    errors: list[dict] = []
    warnings: list[str] = []

    if not isinstance(recipe, dict):
        return DEFAULT_RECIPE, [_err("<root>", "recipe must be a JSON object", "Call get_recipe_schema for a valid example.")], []

    recipe, repairs = repair_shape(recipe)
    warnings.extend(f"Repaired your recipe: {m}" for m in repairs)

    unknown = set(recipe) - set(DEFAULT_RECIPE)
    for k in sorted(unknown):
        near = _closest(k, list(DEFAULT_RECIPE))
        hint = f"Did you mean '{near}'?" if near else f"Allowed keys: {sorted(DEFAULT_RECIPE)}"
        if k in ("params", "model_type", "estimator", "algorithm", "hyperparameters"):
            hint = ('Model settings are nested: {"model": {"type": "lightgbm", '
                    '"params": {"learning_rate": 0.05}}}.')
        errors.append(_err(k, "unknown top-level key", hint))

    r = _deep_merge(DEFAULT_RECIPE, recipe)

    # ---- model -----------------------------------------------------------
    mtype = r["model"].get("type")
    if mtype not in MODEL_REGISTRY:
        near = _closest(mtype, list(MODEL_REGISTRY))
        errors.append(_err("model.type", f"unknown model '{mtype}'",
                           f"Did you mean '{near}'? Available: {sorted(MODEL_REGISTRY)}" if near
                           else f"Available: {sorted(MODEL_REGISTRY)}"))
    elif task and task not in MODEL_REGISTRY[mtype]["tasks"]:
        ok = [m for m, s in MODEL_REGISTRY.items() if task in s["tasks"]]
        errors.append(_err("model.type", f"'{mtype}' does not support task '{task}'",
                           f"Use one of: {sorted(ok)}"))
    if not isinstance(r["model"].get("params", {}), dict):
        errors.append(_err("model.params", "must be an object", 'Example: {"learning_rate": 0.05}'))

    # ---- numeric / categorical ------------------------------------------
    _enum(r["numeric"]["impute"], NUM_IMPUTE, "numeric.impute", errors)
    _enum(r["numeric"]["scale"], NUM_SCALE, "numeric.scale", errors)
    w = r["numeric"].get("winsorize")
    if w is not None and not (isinstance(w, (int, float)) and 0 < w < 0.5):
        errors.append(_err("numeric.winsorize", f"'{w}' is out of range",
                           "Use null, or a fraction in (0, 0.5) such as 0.01 to clip the tails at 1%/99%."))

    _enum(r["categorical"]["impute"], CAT_IMPUTE, "categorical.impute", errors)
    _enum(r["categorical"]["encode"], CAT_ENCODE, "categorical.encode", errors)

    if r["categorical"]["encode"] == "native" and mtype in MODEL_REGISTRY \
            and not MODEL_REGISTRY[mtype]["handles_native_cat"]:
        warnings.append(
            f"categorical.encode='native' is only supported by lightgbm/catboost; "
            f"'{mtype}' will fall back to ordinal encoding. Use 'onehot' or 'target' for an explicit choice."
        )

    if mtype in MODEL_REGISTRY and MODEL_REGISTRY[mtype].get("needs_scaling") \
            and r["numeric"]["scale"] == "none":
        warnings.append(
            f"'{mtype}' is distance/penalty based and is usually much worse unscaled. "
            f"Consider numeric.scale='standard'."
        )

    # ---- validation ------------------------------------------------------
    v = r["validation"]
    _enum(v["scheme"], CV_SCHEMES, "validation.scheme", errors)
    if not isinstance(v["folds"], int) or not (2 <= v["folds"] <= 20):
        errors.append(_err("validation.folds", f"'{v['folds']}' is out of range",
                           "Use an integer in [2, 20]. 5 is the usual choice."))
    if v["scheme"] == "group_kfold" and not v.get("group_column"):
        errors.append(_err("validation.group_column", "group_kfold needs a group column",
                           "Set validation.group_column to the column that identifies the entity "
                           "(user/patient/device) that must not be split across folds."))
    if v.get("group_column") and columns and v["group_column"] not in columns:
        errors.append(_err("validation.group_column", f"column '{v['group_column']}' not in the dataset",
                           f"Available columns: {columns[:25]}{' ...' if len(columns) > 25 else ''}"))

    # ---- metric ----------------------------------------------------------
    if r["metric"] != "auto":
        if task:
            if r["metric"] not in METRICS[task]:
                errors.append(_err("metric", f"'{r['metric']}' is not valid for task '{task}'",
                                   f"Valid metrics for {task}: {METRICS[task]}"))
        else:
            allm = sorted({m for v2 in METRICS.values() for m in v2})
            _enum(r["metric"], allm, "metric", errors)

    # ---- class_weight ----------------------------------------------------
    _enum(r["class_weight"], ["none", "balanced"], "class_weight", errors)
    if r["class_weight"] == "balanced" and task == "regression":
        errors.append(_err("class_weight", "class_weight is meaningless for regression",
                           "Set class_weight='none'."))

    # ---- columns ---------------------------------------------------------
    if not isinstance(r["drop_columns"], list):
        errors.append(_err("drop_columns", "must be a list of column names", 'Example: ["id", "customer_uuid"]'))
    elif columns:
        for c in r["drop_columns"]:
            if c not in columns:
                near = _closest(c, columns)
                errors.append(_err("drop_columns", f"column '{c}' does not exist",
                                   f"Did you mean '{near}'?" if near else "Check profile_dataset for the column list."))

    # ---- features --------------------------------------------------------
    if not isinstance(r["features"], list):
        errors.append(_err("features", "must be a list of feature ops",
                           'Example: [{"op": "ratio", "name": "debt_ratio", "a": "debt", "b": "income"}]'))
    else:
        for i, f in enumerate(r["features"]):
            fld = f"features[{i}]"
            if not isinstance(f, dict):
                errors.append(_err(fld, "each feature must be an object", 'Example: {"op": "log1p", "name": "log_amount", "a": "amount"}'))
                continue
            op = f.get("op")
            if op not in FEATURE_OPS:
                near = _closest(op, list(FEATURE_OPS))
                errors.append(_err(f"{fld}.op", f"unknown op '{op}'",
                                   f"Did you mean '{near}'? Available: {sorted(FEATURE_OPS)}" if near
                                   else f"Available ops: {sorted(FEATURE_OPS)}"))
                continue
            if not f.get("name"):
                errors.append(_err(f"{fld}.name", "missing name for the new column",
                                   "Give the engineered column an explicit name so you can read it in inspect_run."))
            for arg in FEATURE_OPS[op]["args"]:
                col = f.get(arg)
                if not col:
                    errors.append(_err(f"{fld}.{arg}", f"op '{op}' requires '{arg}'",
                                       f"'{op}' takes {FEATURE_OPS[op]['args']}. {FEATURE_OPS[op]['doc']}"))
                elif columns and col not in columns:
                    near = _closest(col, columns)
                    errors.append(_err(f"{fld}.{arg}", f"column '{col}' does not exist",
                                       f"Did you mean '{near}'?" if near else "Check profile_dataset for the column list."))

    return r, errors, warnings


def resolve_metric(recipe: dict, task: str) -> str:
    m = recipe.get("metric", "auto")
    return DEFAULT_METRIC[task] if m == "auto" else m


def schema_doc() -> dict[str, Any]:
    """Machine-readable schema handed to the agent by get_recipe_schema."""
    return {
        "description": (
            "A recipe fully specifies one ML experiment. You never write training code -- "
            "you emit this object and the server compiles it into a leak-safe pipeline, "
            "cross-validates it honestly, and scores it."
        ),
        "fields": {
            "name": "Short identifier for this experiment, e.g. 'lgbm_target_enc'.",
            "notes": "Why you made this change vs the previous run. Shown in the leaderboard.",
            "drop_columns": "Columns to remove before anything else (IDs, leaks, constants).",
            "features": f"Stateless engineered columns. Ops: {{op: spec}} -> { 'see feature_ops' }",
            "numeric": {"impute": NUM_IMPUTE, "scale": NUM_SCALE,
                        "winsorize": "null or fraction in (0,0.5) to clip tails"},
            "categorical": {"impute": CAT_IMPUTE, "encode": CAT_ENCODE,
                            "max_onehot_cardinality": "int; above this, onehot is refused as too wide",
                            "rare_threshold": "float; categories rarer than this are grouped into __RARE__"},
            "class_weight": ["none", "balanced"],
            "model": {"type": sorted(MODEL_REGISTRY), "params": "model-specific overrides"},
            "validation": {"scheme": CV_SCHEMES, "folds": "2..20", "seed": "int",
                           "group_column": "required for group_kfold"},
            "metric": "auto, or one of the task's metrics",
        },
        "feature_ops": {k: v["doc"] for k, v in FEATURE_OPS.items()},
        "models": {k: {"tasks": v["tasks"], "doc": v["doc"], "tunable": sorted(v["space"])}
                   for k, v in MODEL_REGISTRY.items()},
        "metrics_by_task": METRICS,
        "default_metric_by_task": DEFAULT_METRIC,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "minimal_example": {
            "name": "baseline_lgbm",
            "notes": "First real model after the dummy baseline.",
            "model": {"type": "lightgbm", "params": {}},
        },
        "full_example": {
            "name": "lgbm_tuned_v2",
            "notes": "Dropped the ID column, added debt/income ratio, switched to target encoding.",
            "drop_columns": ["customer_id"],
            "features": [{"op": "ratio", "name": "debt_to_income", "a": "debt", "b": "income"}],
            "numeric": {"impute": "median", "scale": "none", "winsorize": 0.01},
            "categorical": {"impute": "most_frequent", "encode": "target", "rare_threshold": 0.02},
            "class_weight": "balanced",
            "model": {"type": "lightgbm", "params": {"learning_rate": 0.03, "num_leaves": 63}},
            "validation": {"scheme": "stratified_kfold", "folds": 5, "seed": 42},
            "metric": "roc_auc",
        },
        "defaults": DEFAULT_RECIPE,
    }
