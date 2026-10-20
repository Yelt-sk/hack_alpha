"""Metric computation, with one entry point and no ambiguity about direction.

Every score in the system goes through :func:`score`, so a recipe result and a
hand-written result are produced by the same code path on the same folds. If the
two regimes computed their own metrics, their numbers would not be comparable
and the leaderboard would be meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from .task import METRICS

#: Metrics that need class probabilities rather than hard labels.
PROBA_METRICS: frozenset[str] = frozenset({"roc_auc", "logloss"})


def direction_of(metric: str) -> str:
    return METRICS[metric][0]


def is_better(metric: str, candidate: float, reference: float) -> bool:
    return (
        candidate > reference if direction_of(metric) == "maximize" else candidate < reference
    )


def score(
    metric: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    labels: list | None = None,
) -> float:
    """Compute `metric`. Raises on a combination that cannot be scored."""
    y_true = np.asarray(y_true)

    if metric in PROBA_METRICS:
        if y_proba is None:
            raise ValueError(f"metric '{metric}' needs probabilities, none were given")
        if metric == "logloss":
            return float(skm.log_loss(y_true, y_proba, labels=labels))
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            return float(skm.roc_auc_score((y_true == labels[-1]).astype(int), y_proba[:, 1]))
        if y_proba.ndim == 2:
            return float(
                skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
            )
        return float(skm.roc_auc_score(y_true, y_proba))

    if y_pred is None:
        raise ValueError(f"metric '{metric}' needs predictions, none were given")
    y_pred = np.asarray(y_pred)

    if metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if metric == "mape":
        return float(skm.mean_absolute_percentage_error(y_true, y_pred))
    if metric == "r2":
        return float(skm.r2_score(y_true, y_pred))
    if metric == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "weighted_f1":
        return float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0))
    if metric == "f1":
        return float(skm.f1_score(y_true, y_pred, average="binary", zero_division=0, pos_label=labels[-1] if labels else 1))

    raise ValueError(f"unknown metric '{metric}'")


def report(
    kind: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray | None,
    y_proba: np.ndarray | None = None,
    labels: list | None = None,
) -> dict[str, float]:
    """All metrics appropriate to the task kind.

    The primary metric decides, but the others are cheap and often explain *why*
    it moved -- accuracy rising while macro F1 falls means the rare classes were
    traded away.
    """
    wanted = (
        ["rmse", "mae", "r2"]
        if kind == "regression"
        else ["macro_f1", "weighted_f1", "accuracy"]
    )
    out: dict[str, float] = {}
    for name in wanted:
        try:
            out[name] = round(score(name, y_true, y_pred, y_proba, labels), 6)
        except Exception:  # a metric that cannot apply here is simply omitted
            continue
    if y_proba is not None and kind != "regression":
        for name in ("logloss", "roc_auc"):
            try:
                out[name] = round(score(name, y_true, y_pred, y_proba, labels), 6)
            except Exception:
                continue
    return out
