"""Example mod: average the strongest experiments into one submission.

Included as a worked example of the gateway, and because averaging two models
that are close in score but different in kind is the cheapest real gain
available late in a run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agent_session import session
from agent_registry import tool


@tool(experimental=True)
def blend_top_experiments(k: int = 3, note: str = "") -> dict[str, Any]:
    """Average the test predictions of the k best experiments and save the result.

    Only meaningful for regression, and only when the blended models differ --
    averaging two tuned LightGBMs buys nothing.

    Args:
        k: How many of the leading experiments to average.
        note: Why this blend is worth trying.
    """
    current = session()
    spec = current.spec
    if spec.is_classification:
        return {"status": "refused",
                "problem": "this mod averages numeric predictions and does not apply to classification",
                "hint": "For classification, average probabilities inside run_python instead."}

    ranked = [e for e in current.ledger.entries if not e.tainted and "test_pred" in e.detail]
    if len(ranked) < 2:
        return {"status": "refused", "problem": "fewer than two experiments carry test predictions",
                "hint": "Run more recipes first."}

    ranked.sort(key=lambda e: e.score if spec.direction == "maximize" else -e.score, reverse=True)
    chosen = ranked[:k]
    blended = np.mean([np.asarray(e.detail["test_pred"], dtype="float64") for e in chosen], axis=0)

    frame = pd.DataFrame({
        spec.id_column: current.store.test[spec.id_column],
        spec.target_column: blended,
    })
    saved = current.submissions.save(
        frame, score=None, source="blend",
        note=note or f"mean of {[e.id for e in chosen]}", force=False,
    )
    return {"status": "ok", "blended": [e.id for e in chosen], "submission": saved,
            "caveat": "A blend has no cross-validated score of its own; "
                      "verify it with evaluate_predictions before trusting it."}
