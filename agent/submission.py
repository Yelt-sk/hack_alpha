"""Custody of ``submission.csv``.

This is the component that decides whether the run scores at all. The platform
kills the process at three hours and grades whatever file is on disk; an absent
or malformed file is zero regardless of how good the model was.

So the rule is an invariant, not a step: **a valid submission exists from the
first minutes of the run**, and it is only ever replaced by something that
validates and scores better. The agent can improve it and cannot break it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import is_better
from .task import TaskSpec


@dataclass
class Submission:
    """What is currently on disk, and what produced it."""

    score: float | None = None
    source: str = ""
    note: str = ""
    rows: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class SubmissionManager:
    def __init__(self, spec: TaskSpec, test_ids: pd.Series, known_labels: list | None = None):
        self.spec = spec
        self.test_ids = test_ids.reset_index(drop=True)
        self.known_labels = set(known_labels) if known_labels else None
        self.state = Submission()

    # -- validation --------------------------------------------------------

    def validate(self, frame: pd.DataFrame) -> list[str]:
        """Every way this file could be rejected, reported at once.

        The task statements are explicit and unforgiving: exactly two columns, no
        missing values, no duplicate ids, row order identical to ``test.csv``.
        Each of those is a silent zero, so each is checked here rather than hoped for.
        """
        spec = self.spec
        problems: list[str] = []
        expected = [spec.id_column, spec.target_column]

        if list(frame.columns) != expected:
            problems.append(
                f"columns must be exactly {expected} in that order, got {list(frame.columns)}"
            )
            if not set(expected).issubset(frame.columns):
                return problems  # nothing further can be checked meaningfully

        ids, target = frame[spec.id_column], frame[spec.target_column]

        if len(frame) != len(self.test_ids):
            problems.append(f"expected {len(self.test_ids)} rows, got {len(frame)}")
        if ids.duplicated().any():
            problems.append(f"{int(ids.duplicated().sum())} duplicated '{spec.id_column}' values")

        missing = set(self.test_ids) - set(ids)
        extra = set(ids) - set(self.test_ids)
        if missing:
            problems.append(f"{len(missing)} test ids are missing, e.g. {sorted(missing)[:5]}")
        if extra:
            problems.append(f"{len(extra)} ids are not in test.csv, e.g. {sorted(extra)[:5]}")

        if not problems and not ids.reset_index(drop=True).equals(self.test_ids):
            problems.append(
                "row order differs from test.csv; reindex to the test order before saving"
            )

        if target.isna().any():
            problems.append(f"{int(target.isna().sum())} missing values in '{spec.target_column}'")

        if spec.is_classification:
            if self.known_labels is not None:
                unseen = set(target.dropna().unique()) - self.known_labels
                if unseen:
                    problems.append(
                        f"labels not present in training data: {sorted(map(str, unseen))[:10]}"
                    )
        else:
            numeric = pd.to_numeric(target, errors="coerce")
            if numeric.isna().sum() > target.isna().sum():
                problems.append("target contains values that are not numeric")
            elif not np.isfinite(numeric.dropna()).all():
                problems.append("target contains inf or -inf")

        return problems

    # -- writing -----------------------------------------------------------

    def save(
        self,
        frame: pd.DataFrame,
        score: float | None = None,
        source: str = "",
        note: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        """Validate and, if it is an improvement, write it atomically."""
        problems = self.validate(frame)
        if problems:
            return {
                "status": "rejected",
                "problems": problems,
                "hint": (
                    f"The file must have exactly the columns "
                    f"['{self.spec.id_column}', '{self.spec.target_column}'], one row per test id, "
                    f"in the same order as test.csv, with no missing values."
                ),
            }

        if (
            not force
            and score is not None
            and self.state.score is not None
            and not is_better(self.spec.metric, score, self.state.score)
        ):
            return {
                "status": "kept_previous",
                "reason": (
                    f"validation score {score:.6f} does not beat the saved "
                    f"{self.state.score:.6f} ({self.spec.metric}, {self.spec.direction})"
                ),
                "current": self.describe(),
            }

        self._atomic_write(frame)
        # The recorded score must describe the file that is actually on disk. A
        # forced write with no score makes it unknown again -- inheriting the
        # previous number would let a worse file masquerade as the better one.
        self.state.score = score
        self.state.source, self.state.note, self.state.rows = source, note, len(frame)
        self.state.history.append(
            {"source": source, "score": score, "note": note[:120], "rows": len(frame)}
        )
        return {"status": "saved", **self.describe()}

    def _atomic_write(self, frame: pd.DataFrame) -> None:
        """Write via a temporary file in the same directory, then rename.

        A kill in the middle of a plain write leaves a truncated CSV, which grades
        as zero. Rename is atomic, so the file on disk is always a complete one.
        """
        target = self.spec.submission_path
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            frame.to_csv(tmp, index=False, encoding="utf-8")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    # -- fallbacks ---------------------------------------------------------

    def save_constant(self, y_train: pd.Series) -> dict[str, Any]:
        """The floor: one constant prediction for every row.

        Worth almost no points by itself, but it makes the invariant true within
        the first minute. Everything after this is an improvement on a file that
        already exists.
        """
        if self.spec.is_classification:
            value: Any = y_train.mode().iloc[0]
        else:
            value = float(pd.to_numeric(y_train, errors="coerce").median())

        frame = pd.DataFrame(
            {self.spec.id_column: self.test_ids, self.spec.target_column: value}
        )
        return self.save(frame, score=None, source="constant", note="bootstrap floor", force=True)

    def exists(self) -> bool:
        return self.spec.submission_path.is_file()

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.spec.submission_path),
            "exists": self.exists(),
            "rows": self.state.rows,
            "validation_score": self.state.score,
            "metric": self.spec.metric,
            "source": self.state.source,
            "note": self.state.note,
            "revisions": len(self.state.history),
        }


def frame_from_predictions(
    spec: TaskSpec, test_ids: pd.Series, predictions: np.ndarray | pd.Series
) -> pd.DataFrame:
    """Assemble a submission frame in the required shape and order."""
    return pd.DataFrame(
        {
            spec.id_column: test_ids.reset_index(drop=True),
            spec.target_column: pd.Series(np.asarray(predictions)).reset_index(drop=True),
        }
    )
