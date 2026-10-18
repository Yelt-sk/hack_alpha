"""
Honest cross-validation, metrics and diagnostics.

The agent gets back more than a number: it gets the per-fold spread, the
train-vs-validation gap and a plain-language read of what those mean. A weak
model cannot infer "this is overfitting" from a bare 0.87 -- so we say it.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, cross_validate)

from .schema import LOWER_IS_BETTER

# scorer name -> (sklearn scoring string, needs_negation)
SCORERS: dict[str, dict[str, tuple[str, bool]]] = {
    "binary": {
        "roc_auc": ("roc_auc", False),
        "pr_auc": ("average_precision", False),
        "f1": ("f1", False),
        "accuracy": ("accuracy", False),
        "balanced_accuracy": ("balanced_accuracy", False),
        "logloss": ("neg_log_loss", True),
    },
    "multiclass": {
        "accuracy": ("accuracy", False),
        "macro_f1": ("f1_macro", False),
        "balanced_accuracy": ("balanced_accuracy", False),
        "logloss": ("neg_log_loss", True),
    },
    "regression": {
        "rmse": ("neg_root_mean_squared_error", True),
        "mae": ("neg_mean_absolute_error", True),
        "r2": ("r2", False),
        "mape": ("neg_mean_absolute_percentage_error", True),
    },
}


def make_cv(scheme: str, folds: int, seed: int, task: str):
    if scheme == "auto":
        scheme = "stratified_kfold" if task in ("binary", "multiclass") else "kfold"
    if scheme == "stratified_kfold":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), scheme
    if scheme == "kfold":
        return KFold(n_splits=folds, shuffle=True, random_state=seed), scheme
    if scheme == "group_kfold":
        return GroupKFold(n_splits=folds), scheme
    if scheme == "time_series":
        return TimeSeriesSplit(n_splits=folds), scheme
    raise ValueError(f"unknown cv scheme '{scheme}'")


def run_cv(pipe, X: pd.DataFrame, y: pd.Series, task: str, recipe: dict,
           groups: pd.Series | None = None) -> dict[str, Any]:
    v = recipe["validation"]
    cv, resolved = make_cv(v["scheme"], v["folds"], v["seed"], task)
    scoring = {k: s for k, (s, _) in SCORERS[task].items()}

    t0 = time.perf_counter()
    raw = cross_validate(
        pipe, X, y, cv=cv, groups=groups, scoring=scoring,
        return_train_score=True, error_score="raise", n_jobs=1,
    )
    elapsed = time.perf_counter() - t0

    metrics: dict[str, Any] = {}
    for name, (_, neg) in SCORERS[task].items():
        val = np.asarray(raw[f"test_{name}"], dtype=float)
        trn = np.asarray(raw[f"train_{name}"], dtype=float)
        if neg:
            val, trn = -val, -trn
        metrics[name] = {
            "mean": round(float(np.mean(val)), 5),
            "std": round(float(np.std(val)), 5),
            "folds": [round(float(x), 5) for x in val],
            "train_mean": round(float(np.mean(trn)), 5),
            "higher_is_better": name not in LOWER_IS_BETTER,
        }

    return {
        "metrics": metrics,
        "cv_scheme": resolved,
        "folds": v["folds"],
        "seed": v["seed"],
        "n_rows": int(len(X)),
        "fit_seconds": round(elapsed, 2),
    }


def diagnose(result: dict, primary: str, task: str, best_so_far: float | None,
             baseline: float | None) -> list[str]:
    """Turn raw numbers into the next action. This is the agent's feedback signal."""
    m = result["metrics"][primary]
    notes: list[str] = []
    hib = m["higher_is_better"]
    mean, std, train = m["mean"], m["std"], m["train_mean"]

    gap = (train - mean) if hib else (mean - train)
    scale = max(abs(mean), 1e-9)
    if gap / scale > 0.15:
        notes.append(
            f"OVERFITTING: train {primary}={train:.4f} vs validation {mean:.4f} "
            f"(gap {gap / scale:.0%}). Reduce capacity (fewer num_leaves / lower max_depth), "
            f"raise regularisation (reg_lambda, min_child_samples), or lower learning_rate with fewer estimators."
        )
    elif gap / scale < 0.01 and baseline is not None:
        lift = (mean - baseline) if hib else (baseline - mean)
        if lift / max(abs(baseline), 1e-9) < 0.05:
            notes.append(
                "UNDERFITTING: the model barely beats the dummy baseline and does not even fit the "
                "training data. Add capacity, add features, or check that the target is predictable at all."
            )

    if std / scale > 0.10:
        notes.append(
            f"HIGH VARIANCE across folds (std {std:.4f} on mean {mean:.4f}). Differences smaller than "
            f"~{2 * std:.4f} between runs are noise -- raise folds to 10 before trusting small gains."
        )

    if baseline is not None:
        lift = (mean - baseline) if hib else (baseline - mean)
        notes.append(f"Beats the dummy baseline by {lift:+.4f} ({primary} {baseline:.4f} -> {mean:.4f}).")

    if best_so_far is not None:
        delta = (mean - best_so_far) if hib else (best_so_far - mean)
        if delta > 2 * std:
            notes.append(f"NEW BEST: improves on the previous best by {abs(delta):.4f}, which exceeds fold noise.")
        elif delta > 0:
            notes.append(
                f"Nominally better by {abs(delta):.4f}, but that is inside fold noise (2*std={2 * std:.4f}). "
                f"Do not treat this as a real improvement."
            )
        else:
            notes.append(f"Worse than the current best by {abs(delta):.4f}. Revert this change and try another axis.")

    if result["fit_seconds"] > 120:
        notes.append(f"Slow: {result['fit_seconds']:.0f}s per CV run. Lower n_estimators while exploring.")

    return notes
