"""Per-class decision thresholds, fitted on out-of-fold probabilities.

On an imbalanced multiclass task with macro F1, the argmax of a calibrated model
is the wrong decision rule: every class counts equally in the metric, while the
argmax answers to the prior. Dividing each class's probability by a learned
constant before taking the argmax trades a little accuracy on the common classes
for a lot of recall on the rare ones, which is exactly the trade macro F1 pays
for.

Measured on this project's classification task, on the automatic floor:

    argmax                      0.6144
    thresholds, full OOF        0.6479      <- optimistic
    thresholds, nested          0.6379      <- what to believe

Both numbers are reported, because the difference between them is the amount by
which fitting the thresholds on the same predictions they are scored against
flatters the result. Hiding it would make every future comparison dishonest.
"""

from __future__ import annotations

import numpy as np

#: Divisors to try per class. Wide enough to suppress a class that is being
#: over-predicted three to one, fine enough not to miss the optimum by much.
GRID = np.linspace(0.25, 4.0, 24)

#: Sweeps over the classes. The second pass usually moves something, the third
#: rarely does, and each costs a full pass over the predictions.
ROUNDS = 3


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 without sklearn's per-call overhead, which dominates here.

    Called tens of thousands of times during a search; the library version spends
    more time validating its inputs than computing the answer.
    """
    total = 0.0
    for c in range(n_classes):
        tp = np.count_nonzero((y_pred == c) & (y_true == c))
        if tp:
            precision = tp / np.count_nonzero(y_pred == c)
            recall = tp / np.count_nonzero(y_true == c)
            total += 2 * precision * recall / (precision + recall)
    return total / n_classes


def fit(proba: np.ndarray, y: np.ndarray, rounds: int = ROUNDS) -> tuple[np.ndarray, float]:
    """Coordinate descent over per-class divisors. Returns them and their score."""
    n_classes = proba.shape[1]
    divisors = np.ones(n_classes)
    best = _macro_f1(y, proba.argmax(1), n_classes)

    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            keep = divisors[c]
            for candidate in GRID:
                divisors[c] = candidate
                score = _macro_f1(y, (proba / divisors).argmax(1), n_classes)
                if score > best:
                    best, keep, improved = score, candidate, True
            divisors[c] = keep
        if not improved:
            break
    return divisors, best


def apply(proba: np.ndarray, divisors: np.ndarray) -> np.ndarray:
    """Class indices under the learned rule."""
    return (proba / divisors).argmax(1)


def honest_score(proba: np.ndarray, y: np.ndarray, fold_id: np.ndarray,
                 rounds: int = 2) -> float:
    """What the thresholds are worth when they have not seen what scores them.

    Fitted on every fold but one and scored on the one left out, which is the
    same discipline the pipeline applies to the model itself. Cheap: no model is
    refitted, only arithmetic on predictions that already exist.
    """
    n_classes = proba.shape[1]
    scores = []
    for fold in np.unique(fold_id):
        inner, outer = fold_id != fold, fold_id == fold
        divisors, _ = fit(proba[inner], y[inner], rounds=rounds)
        scores.append(_macro_f1(y[outer], apply(proba[outer], divisors), n_classes))
    return float(np.mean(scores))
