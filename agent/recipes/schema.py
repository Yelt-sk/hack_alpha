"""The recipe vocabulary: the declarative fast path.

A recipe is a complete description of one experiment. The agent emits JSON, the
server compiles it into a leak-safe pipeline and scores it on the shared folds.

Note what is *absent*: the validation scheme. Folds belong to :class:`DataStore`
and are identical for every experiment in the run. The agent therefore cannot
choose a favourable split, and two results are always comparable -- which is
also what lets a hand-written result be scored against a recipe.

The model driving this is strong, so the vocabulary is wide rather than
protective. The guard rails live in the evaluator, not in a narrowed grammar.
"""

from __future__ import annotations

import copy
import difflib
import os
from typing import Any

#: Cap on worker threads. Cross-validation already iterates folds, so letting a
#: booster claim every core oversubscribes them. Measured on an 8-core machine,
#: n_jobs=-1 turned a 2-second fit into 47 seconds for an identical score.
#:
#: Overridable because the development machine also runs the inference server:
#: four training threads beside four generation threads on eight cores starves
#: both. The grading container has no such neighbour and can use the default.
N_JOBS: int = int(os.environ.get("AGENT_N_JOBS", 0)) or max(1, min(4, (os.cpu_count() or 4) // 2))

NUM_IMPUTE = ["median", "mean", "constant", "none"]
NUM_SCALE = ["none", "standard", "robust", "minmax"]
CAT_IMPUTE = ["most_frequent", "constant", "none"]
CAT_ENCODE = ["native", "onehot", "ordinal", "target", "frequency"]
TARGET_TRANSFORM = ["none", "log1p"]

FEATURE_OPS: dict[str, dict[str, Any]] = {
    "ratio": {"args": ["a", "b"], "doc": "a / (b + eps). Normalise a quantity by an exposure."},
    "product": {"args": ["a", "b"], "doc": "a * b. An explicit interaction."},
    "diff": {"args": ["a", "b"], "doc": "a - b."},
    "sum": {"args": ["a", "b"], "doc": "a + b."},
    "log1p": {"args": ["a"], "doc": "log(1+a) for a skewed non-negative column."},
    "row_stat": {
        "args": ["columns", "stat"],
        "doc": "Row-wise mean/std/min/max/sum over a list of columns. "
        "Useful when many columns form one group, such as monthly temperatures.",
    },
    "missing_count": {"args": [], "doc": "Per-row count of missing values; missingness is often signal."},
}

MODELS: dict[str, dict[str, Any]] = {
    "dummy": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {},
        "space": {},
        "native_cat": False,
        "doc": "Constant baseline. The floor any real model must clear.",
    },
    "ridge": {
        "tasks": ["regression"],
        "defaults": {"alpha": 1.0},
        "space": {"alpha": {"type": "float", "low": 1e-3, "high": 1e3, "log": True}},
        "native_cat": False,
        "needs_scaling": True,
        "doc": "L2 linear regression. Fast check of whether the problem is linear.",
    },
    "logreg": {
        "tasks": ["binary", "multiclass"],
        "defaults": {"C": 1.0, "max_iter": 1000, "n_jobs": N_JOBS},
        "space": {"C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True}},
        "native_cat": False,
        "needs_scaling": True,
        "doc": "Logistic regression. Needs scaling and one-hot encoding.",
    },
    "random_forest": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {"n_estimators": 300, "n_jobs": N_JOBS},
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 800},
            "max_depth": {"type": "int", "low": 4, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        },
        "native_cat": False,
        "doc": "Bagged trees. Hard to overfit, rarely the winner, good sanity check.",
    },
    "extra_trees": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {"n_estimators": 300, "n_jobs": N_JOBS},
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 800},
            "max_depth": {"type": "int", "low": 4, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        },
        "native_cat": False,
        "doc": "Extremely randomised trees. Lower variance than a forest; useful in a blend.",
    },
    "hist_gbm": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {"max_iter": 300},
        "space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 800},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 255},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
            "l2_regularization": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        "native_cat": False,
        "doc": "sklearn histogram boosting. No extra dependency, close to LightGBM.",
    },
    "lightgbm": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "n_jobs": N_JOBS,
            "verbose": -1,
            "importance_type": "gain",
        },
        "space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "n_estimators": {"type": "int", "low": 200, "high": 2000},
            "num_leaves": {"type": "int", "low": 15, "high": 255},
            "min_child_samples": {"type": "int", "low": 5, "high": 150},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 30.0, "log": True},
        },
        "native_cat": True,
        "doc": "The default first choice on tabular data. Handles categoricals natively.",
    },
    "catboost": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {"iterations": 600, "verbose": 0, "allow_writing_files": False,
                     "thread_count": N_JOBS},
        "space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "iterations": {"type": "int", "low": 300, "high": 2000},
            "depth": {"type": "int", "low": 4, "high": 10},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 30.0, "log": True},
        },
        "native_cat": True,
        "doc": "Usually strongest when there are many high-cardinality categoricals. Slower.",
    },
    "xgboost": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {"n_estimators": 500, "learning_rate": 0.05, "n_jobs": N_JOBS,
                     "tree_method": "hist", "enable_categorical": True},
        "space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "n_estimators": {"type": "int", "low": 200, "high": 2000},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "min_child_weight": {"type": "float", "low": 1.0, "high": 50.0, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 30.0, "log": True},
        },
        "native_cat": True,
        "doc": "A third independent booster. Useful to confirm a result is not a LightGBM quirk.",
    },
    "flaml": {
        "tasks": ["regression", "binary", "multiclass"],
        "defaults": {},
        "space": {},
        "native_cat": False,
        "needs_time_budget": True,
        "doc": "AutoML: searches estimators and hyperparameters itself within time_budget_s. "
        "Use it when you want a strong result without spending your own iterations on tuning.",
    },
}

DEFAULT_RECIPE: dict[str, Any] = {
    "name": "unnamed",
    "notes": "",
    "drop_columns": [],
    "features": [],
    "numeric": {"impute": "median", "scale": "none", "winsorize": None},
    "categorical": {"impute": "most_frequent", "encode": "native", "rare_threshold": 0.0},
    "target_transform": "none",
    "class_weight": "none",
    "model": {"type": "lightgbm", "params": {}},
    "time_budget_s": None,
}

#: Hyperparameters a model habitually hoists to the top level of the JSON.
_HOISTED = {
    "learning_rate", "n_estimators", "num_leaves", "max_depth", "min_child_samples",
    "subsample", "colsample_bytree", "reg_lambda", "iterations", "depth", "l2_leaf_reg",
    "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "alpha", "C",
    "min_child_weight", "n_jobs",
}


def repair(recipe: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fix unambiguous shape mistakes and say so.

    ``{"model": "lightgbm", "learning_rate": 0.05}`` has exactly one sensible
    reading. Rejecting it buys nothing but a wasted round trip, so it is repaired
    and reported -- repaired, never silently.
    """
    if not isinstance(recipe, dict):
        return dict(DEFAULT_RECIPE), ["recipe was not an object; using defaults"]

    out = copy.deepcopy(recipe)
    notes: list[str] = []

    if isinstance(out.get("model"), str):
        out["model"] = {"type": out["model"]}
        notes.append('model was a string; read as {"model": {"type": ...}}')
    if not isinstance(out.get("model"), dict):
        out["model"] = {}

    if isinstance(out.get("type"), str) and out["type"] in MODELS:
        out["model"].setdefault("type", out.pop("type"))
        notes.append('moved top-level "type" into model.type')

    for key in ("params", "model_params", "hyperparameters"):
        if isinstance(out.get(key), dict):
            merged = dict(out["model"].get("params") or {})
            merged.update(out.pop(key))
            out["model"]["params"] = merged
            notes.append(f'moved top-level "{key}" into model.params')

    if "note" in out and "notes" not in out:
        out["notes"] = out.pop("note")
        notes.append('renamed "note" to "notes"')

    # A list such as ["drop_columns", "scale_numeric", "encode_categorical"] names
    # stages rather than configuring them. There is nothing to salvage, but saying
    # where those settings actually live is worth more than listing legal keys.
    if isinstance(out.get("preprocessing"), list):
        out.pop("preprocessing")
        notes.append(
            'dropped "preprocessing": preprocessing is configured through the '
            '"numeric", "categorical", "drop_columns" and "features" fields, not as a list of stages'
        )

    hoisted = {k: out.pop(k) for k in list(out) if k in _HOISTED}
    if hoisted:
        params = dict(out["model"].get("params") or {})
        params.update(hoisted)
        out["model"]["params"] = params
        notes.append(f"moved {sorted(hoisted)} into model.params")

    return out, notes


def _closest(value: Any, options: list[str]) -> str | None:
    match = difflib.get_close_matches(str(value), options, n=1, cutoff=0.5)
    return match[0] if match else None


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def validate(
    recipe: dict[str, Any], kind: str, columns: list[str] | None = None
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    """Normalise a recipe and report every problem at once.

    Returning all errors together matters: one error per round trip turns the
    model's step budget into a spelling contest.
    """
    recipe, warnings = repair(recipe)
    errors: list[dict[str, str]] = []

    def fail(field: str, problem: str, hint: str) -> None:
        errors.append({"field": field, "problem": problem, "hint": hint})

    for key in sorted(set(recipe) - set(DEFAULT_RECIPE)):
        near = _closest(key, list(DEFAULT_RECIPE))
        fail(key, "unknown top-level key",
             f"Did you mean '{near}'?" if near else f"Allowed: {sorted(DEFAULT_RECIPE)}")

    merged = _merge(DEFAULT_RECIPE, recipe)
    model_type = merged["model"].get("type")

    if model_type not in MODELS:
        near = _closest(model_type, list(MODELS))
        fail("model.type", f"unknown model '{model_type}'",
             (f"Did you mean '{near}'? " if near else "") + f"Available: {sorted(MODELS)}")
    elif kind not in MODELS[model_type]["tasks"]:
        usable = sorted(m for m, s in MODELS.items() if kind in s["tasks"])
        fail("model.type", f"'{model_type}' does not support a {kind} task", f"Use one of: {usable}")
    elif MODELS[model_type].get("needs_time_budget") and not merged.get("time_budget_s"):
        fail("time_budget_s", f"'{model_type}' is an AutoML search and needs a time budget",
             'Set time_budget_s, e.g. 600 for ten minutes. Check get_context for time remaining.')

    for field_name, allowed in (
        ("numeric.impute", NUM_IMPUTE), ("numeric.scale", NUM_SCALE),
        ("categorical.impute", CAT_IMPUTE), ("categorical.encode", CAT_ENCODE),
    ):
        section, key = field_name.split(".")
        value = merged[section].get(key)
        if value not in allowed:
            near = _closest(value, allowed)
            fail(field_name, f"'{value}' is not valid",
                 (f"Did you mean '{near}'? " if near else "") + f"Allowed: {allowed}")

    if merged["target_transform"] not in TARGET_TRANSFORM:
        fail("target_transform", f"'{merged['target_transform']}' is not valid",
             f"Allowed: {TARGET_TRANSFORM}")
    elif merged["target_transform"] == "log1p" and kind != "regression":
        fail("target_transform", "log1p only applies to regression", 'Use "none".')

    if merged["class_weight"] not in ("none", "balanced"):
        fail("class_weight", f"'{merged['class_weight']}' is not valid", 'Use "none" or "balanced".')
    elif merged["class_weight"] == "balanced" and kind == "regression":
        fail("class_weight", "class_weight is meaningless for regression", 'Use "none".')

    if not isinstance(merged["drop_columns"], list):
        fail("drop_columns", "must be a list of column names", 'Example: ["id"]')
    elif columns:
        for name in merged["drop_columns"]:
            if name not in columns:
                near = _closest(name, columns)
                fail("drop_columns", f"column '{name}' does not exist",
                     f"Did you mean '{near}'?" if near else "See profile_data for the column list.")

    _validate_features(merged, columns, fail)

    if model_type in MODELS and MODELS[model_type].get("needs_scaling") and merged["numeric"]["scale"] == "none":
        warnings.append(
            f"'{model_type}' is penalty- or distance-based and is usually much worse unscaled; "
            f'consider numeric.scale = "standard"'
        )
    if merged["categorical"]["encode"] == "native" and model_type in MODELS and not MODELS[model_type]["native_cat"]:
        warnings.append(
            f"categorical.encode='native' is only supported by lightgbm/catboost/xgboost; "
            f"'{model_type}' will fall back to ordinal"
        )

    return merged, errors, warnings


def _validate_features(merged: dict, columns: list[str] | None, fail) -> None:
    if not isinstance(merged["features"], list):
        fail("features", "must be a list of feature operations",
             'Example: [{"op": "ratio", "name": "x", "a": "c1", "b": "c2"}]')
        return

    for index, feature in enumerate(merged["features"]):
        where = f"features[{index}]"
        if not isinstance(feature, dict):
            fail(where, "each feature must be an object", 'Example: {"op": "log1p", "name": "la", "a": "amount"}')
            continue
        op = feature.get("op")
        if op not in FEATURE_OPS:
            near = _closest(op, list(FEATURE_OPS))
            fail(f"{where}.op", f"unknown op '{op}'",
                 (f"Did you mean '{near}'? " if near else "") + f"Available: {sorted(FEATURE_OPS)}")
            continue
        if not feature.get("name"):
            fail(f"{where}.name", "the new column needs a name",
                 "Name it so you can recognise it in feature importances.")
        for arg in FEATURE_OPS[op]["args"]:
            value = feature.get(arg)
            if value is None:
                fail(f"{where}.{arg}", f"op '{op}' requires '{arg}'",
                     f"'{op}' takes {FEATURE_OPS[op]['args']}. {FEATURE_OPS[op]['doc']}")
            elif arg == "columns":
                if not isinstance(value, list) or not value:
                    fail(f"{where}.columns", "must be a non-empty list of column names",
                         'Example: {"op": "row_stat", "name": "temp_mean", "columns": [...], "stat": "mean"}')
                elif columns:
                    unknown = [c for c in value if c not in columns]
                    if unknown:
                        fail(f"{where}.columns", f"unknown columns {unknown[:5]}",
                             "See profile_data for the column list.")
            elif arg == "stat":
                if value not in ("mean", "std", "min", "max", "sum"):
                    fail(f"{where}.stat", f"'{value}' is not a supported statistic",
                         "Use one of: mean, std, min, max, sum.")
            elif columns and value not in columns:
                near = _closest(value, columns)
                fail(f"{where}.{arg}", f"column '{value}' does not exist",
                     f"Did you mean '{near}'?" if near else "See profile_data for the column list.")


def describe() -> dict[str, Any]:
    """The schema handed to the agent by ``get_recipe_schema``."""
    return {
        "description": (
            "A recipe fully specifies one experiment. The server compiles it into a leak-safe "
            "pipeline and scores it on the run's shared folds. The validation scheme is not part "
            "of a recipe: folds are fixed once per task so that every result is comparable."
        ),
        "fields": {
            "name": "Short identifier, e.g. 'lgbm_target_enc'.",
            "notes": "Why this differs from the previous run. Shown in the leaderboard.",
            "drop_columns": "Columns removed before anything else.",
            "features": "Stateless engineered columns; see feature_ops.",
            "numeric": {"impute": NUM_IMPUTE, "scale": NUM_SCALE,
                        "winsorize": "null, or a fraction in (0, 0.5) to clip the tails"},
            "categorical": {"impute": CAT_IMPUTE, "encode": CAT_ENCODE,
                            "rare_threshold": "fraction below which categories merge into __RARE__"},
            "target_transform": TARGET_TRANSFORM,
            "class_weight": ["none", "balanced"],
            "model": {"type": sorted(MODELS), "params": "model-specific overrides"},
            "time_budget_s": "Seconds granted to an AutoML model such as flaml.",
        },
        "feature_ops": {k: v["doc"] for k, v in FEATURE_OPS.items()},
        "models": {k: {"tasks": v["tasks"], "doc": v["doc"], "tunable": sorted(v["space"])}
                   for k, v in MODELS.items()},
        "minimal_example": {"name": "lgbm_default", "model": {"type": "lightgbm"}},
        "full_example": {
            "name": "lgbm_engineered",
            "notes": "Added a yearly temperature summary and switched to target encoding.",
            "drop_columns": [],
            "features": [
                {"op": "row_stat", "name": "temp_year_mean",
                 "columns": ["january_avg_temp_c", "february_avg_temp_c"], "stat": "mean"},
                {"op": "missing_count", "name": "n_missing"},
            ],
            "numeric": {"impute": "median", "scale": "none", "winsorize": 0.01},
            "categorical": {"impute": "most_frequent", "encode": "target", "rare_threshold": 0.01},
            "target_transform": "log1p",
            "model": {"type": "lightgbm", "params": {"learning_rate": 0.03, "num_leaves": 63}},
        },
    }
