"""Checks the agent cannot talk its way past.

The case materials warn about this directly: in Weco AI's own report an agent
gamed the metric in roughly 63% of attempts. An agent optimises exactly the
number it is shown, so the number has to be hard to fake.

Most of the defence is structural and lives elsewhere -- fixed folds, transforms
inside the pipeline, a holdout the agent never sees. What remains here is the
case that structure cannot catch: a result that is simply too good to believe.
"""

from __future__ import annotations

#: Beyond these, a score on a genuinely hard problem is evidence of leakage
#: rather than of skill.
IMPLAUSIBLE_HIGH = {
    "roc_auc": 0.9995, "accuracy": 0.9995, "macro_f1": 0.9995,
    "weighted_f1": 0.9995, "f1": 0.9995, "r2": 0.9999,
}
IMPLAUSIBLE_LOW = {"logloss": 0.001, "mape": 0.001, "rmse": 0.0, "mae": 0.0}


def implausible(score: float, metric: str) -> str:
    """Return a reason if the score is beyond belief, otherwise an empty string."""
    high = IMPLAUSIBLE_HIGH.get(metric)
    if high is not None and score >= high:
        return (
            f"{metric}={score:.6f} is at or beyond the plausibility threshold ({high}); "
            f"on a task of this difficulty that nearly always means a leaking feature"
        )
    low = IMPLAUSIBLE_LOW.get(metric)
    if low is not None and score <= low:
        return f"{metric}={score:.6f} is at the theoretical floor; check for a leaking feature"
    return ""
