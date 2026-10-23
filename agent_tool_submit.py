"""Writing the answer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent_session import session
from agent_registry import tool


@tool
def save_submission(csv_path: str, note: str = "", force: bool = False) -> dict[str, Any]:
    """Validate a predictions file and, if it is an improvement, make it the answer.

    Normally you do not need this: a recipe that becomes the best result saves
    itself. Use it for predictions produced outside the recipe path, or to restore
    a particular file at the end.

    The file is checked against test.csv before anything is written: exact column
    names and order, one row per test id in the test order, no missing values, and
    for classification only labels seen in training. A rejected file leaves the
    existing answer untouched.

    Args:
        csv_path: Path to a CSV with the id column and the target column.
        note: What produced these predictions.
        force: Overwrite even when the saved answer scored better.
    """
    current = session()
    try:
        frame = pd.read_csv(csv_path)
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "problem": f"could not read '{csv_path}': {err}"}
    return current.submissions.save(frame, score=None, source="manual", note=note, force=force)
