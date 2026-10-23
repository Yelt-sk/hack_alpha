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

#: How far the learned rule is moved from argmax: divisors ** SHRINK.
#:
#: Fitted thresholds are fitted to the training distribution, and the platform
#: says they do not travel: +0.05 macro F1 by our folds, about +0.004 on the test.
#: On the 20% of training rows an adversarial model finds most test-like, full
#: thresholds lose to plain argmax (0.590 against 0.597). Shrinking them part of
#: the way back is the compromise that measured free on our own folds (0.6275
#: against 0.6288, inside the noise, 2 of 5) and better on the shifted rows
#: (0.595 against 0.590). Half-way gave more on the shift (0.598) but cost 0.005
#: on every one of our folds. scripts/measure_shrink.py, measure_testlike.py.
SHRINK = 0.75


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
              class_weight: np.ndarray | None = None) -> float:
    """Macro F1 without sklearn's per-call overhead, which dominates here.

    Called tens of thousands of times during a search; the library version spends
    more time validating its inputs than computing the answer.

    ``class_weight`` rescales each true class's rows, which is how a class mix
    other than the training one is scored without resampling: recall is
    unchanged, precision sees the false positives of a class in the proportion the
    target mix would produce them.
    """
    confusion = np.bincount(y_true * n_classes + y_pred,
                            minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    if class_weight is not None:
        confusion = confusion * np.asarray(class_weight, dtype="float64")[:, None]
    tp = np.diag(confusion).astype("float64")
    predicted = confusion.sum(axis=0)
    actual = confusion.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(tp > 0, 2 * tp / (predicted + actual), 0.0)
    return float(f1.sum() / n_classes)


def fit(proba: np.ndarray, y: np.ndarray, rounds: int = ROUNDS,
        class_weight: np.ndarray | None = None,
        shrink: float | None = None) -> tuple[np.ndarray, float]:
    """Coordinate descent over per-class divisors. Returns them and their score."""
    shrink = SHRINK if shrink is None else shrink
    n_classes = proba.shape[1]
    divisors = np.ones(n_classes)
    best = _macro_f1(y, proba.argmax(1), n_classes, class_weight)

    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            keep = divisors[c]
            for candidate in GRID:
                divisors[c] = candidate
                score = _macro_f1(y, (proba / divisors).argmax(1), n_classes, class_weight)
                if score > best:
                    best, keep, improved = score, candidate, True
            divisors[c] = keep
        if not improved:
            break
    if shrink != 1.0:
        divisors = divisors ** shrink
        best = _macro_f1(y, (proba / divisors).argmax(1), n_classes, class_weight)
    return divisors, best


def apply(proba: np.ndarray, divisors: np.ndarray) -> np.ndarray:
    """Class indices under the learned rule."""
    return (proba / divisors).argmax(1)


def honest_score(proba: np.ndarray, y: np.ndarray, fold_id: np.ndarray,
                 rounds: int = 2, shrink: float | None = None) -> float:
    """What the thresholds are worth when they have not seen what scores them.

    Fitted on every fold but one and scored on the one left out, which is the
    same discipline the pipeline applies to the model itself. Cheap: no model is
    refitted, only arithmetic on predictions that already exist.
    """
    n_classes = proba.shape[1]
    scores = []
    # Fold -1 marks rows that no fold validates -- the early periods a forecast
    # plan only ever trains on. Treating it as a fold would score predictions that
    # were never made.
    for fold in (f for f in np.unique(fold_id) if f >= 0):
        inner = (fold_id != fold) & (fold_id >= 0)
        outer = fold_id == fold
        divisors, _ = fit(proba[inner], y[inner], rounds=rounds, shrink=shrink)
        scores.append(_macro_f1(y[outer], apply(proba[outer], divisors), n_classes))
    return float(np.mean(scores))
