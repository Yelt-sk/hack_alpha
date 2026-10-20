"""Run state: the clock, the data, and the record of what has been tried.

Three objects, each owning one thing the agent cannot be trusted to track by
itself:

* :class:`Clock` -- the three-hour wall. The agent cannot see time passing, and
  a run killed mid-experiment scores zero.
* :class:`DataStore` -- the data, loaded once, plus **one** fold assignment
  reused by every experiment. Shared folds are what make a recipe score and a
  hand-written score comparable; without them the hybrid is two disconnected
  regimes.
* :class:`Ledger` -- what was tried and what it scored, including the fold
  spread, so "better" can be distinguished from "noise".

Nothing here persists between runs. The task statement forbids carrying results
from one task into the next, and a fresh container gives us that for free.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .task import TaskSpec

Phase = Literal["bootstrap", "explore", "exploit", "finalize"]

#: Fractions of the total budget. `finalize` is generous on purpose: refitting on
#: full data and writing the answer must never be the thing that runs out of time.
PHASE_BOUNDS: tuple[tuple[Phase, float], ...] = (
    ("bootstrap", 0.05),
    ("explore", 0.50),
    ("exploit", 0.85),
    ("finalize", 1.00),
)


class Clock:
    """Wall-clock budget for a single task run."""

    def __init__(self, total_seconds: float = 3 * 3600, reserve_seconds: float = 600):
        self.total = float(total_seconds)
        #: Hard stop before the platform's kill, kept for the final write.
        self.reserve = float(reserve_seconds)
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed)

    @property
    def usable_remaining(self) -> float:
        """Time left before the reserve, i.e. what experiments may still spend."""
        return max(0.0, self.remaining - self.reserve)

    @property
    def fraction_used(self) -> float:
        return min(1.0, self.elapsed / self.total) if self.total else 1.0

    @property
    def phase(self) -> Phase:
        used = self.fraction_used
        for name, bound in PHASE_BOUNDS:
            if used < bound:
                return name
        return "finalize"

    @property
    def must_finalize(self) -> bool:
        return self.usable_remaining <= 0

    def affordable(self, seconds: float) -> bool:
        """Whether an operation of this length still fits before the reserve."""
        return seconds <= self.usable_remaining

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "elapsed_min": round(self.elapsed / 60, 1),
            "remaining_min": round(self.remaining / 60, 1),
            "usable_min": round(self.usable_remaining / 60, 1),
            "must_finalize": self.must_finalize,
        }


@dataclass
class Experiment:
    """One scored attempt, whatever produced it."""

    id: str
    name: str
    source: Literal["recipe", "custom", "baseline"]
    score: float
    fold_scores: list[float]
    metric: str
    direction: str
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    tainted: str = ""

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores)) if len(self.fold_scores) > 1 else 0.0

    def row(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "score": round(self.score, 6),
            "std": round(self.std, 6),
            "seconds": round(self.seconds, 1),
        }
        if self.note:
            out["note"] = self.note[:160]
        if self.tainted:
            out["TAINTED"] = self.tainted
        return out


class Ledger:
    """The record of attempts, and the arbiter of what counts as an improvement."""

    def __init__(self, direction: str):
        self.direction = direction
        self.entries: list[Experiment] = []

    def __len__(self) -> int:
        return len(self.entries)

    def next_id(self) -> str:
        return f"exp_{len(self.entries) + 1:03d}"

    def add(self, experiment: Experiment) -> Experiment:
        self.entries.append(experiment)
        return experiment

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.direction == "maximize" else a < b

    def best(self, include_tainted: bool = False) -> Experiment | None:
        pool = [e for e in self.entries if include_tainted or not e.tainted]
        if not pool:
            return None
        return max(pool, key=lambda e: e.score if self.direction == "maximize" else -e.score)

    def verdict(self, experiment: Experiment) -> str:
        """Plain-language reading of a result against the current best.

        A gain inside the fold noise is not a gain. Saying so costs one line and
        stops the agent building its next three experiments on a coin flip.
        """
        previous = [e for e in self.entries if e is not experiment and not e.tainted]
        if not previous:
            return "First scored result; this is the reference point."
        best = max(previous, key=lambda e: e.score if self.direction == "maximize" else -e.score)
        delta = abs(experiment.score - best.score)
        noise = 2 * max(experiment.std, best.std)

        if not self._better(experiment.score, best.score):
            return (
                f"Worse than {best.id} by {delta:.5f}. Revert this change and try another axis."
            )
        if delta <= noise:
            return (
                f"Nominally better than {best.id} by {delta:.5f}, but that is inside fold noise "
                f"(2*std={noise:.5f}). Do not treat it as a real improvement."
            )
        return f"NEW BEST: beats {best.id} by {delta:.5f}, which exceeds fold noise."

    def rows(self) -> list[dict[str, Any]]:
        ordered = sorted(
            self.entries,
            key=lambda e: e.score if self.direction == "maximize" else -e.score,
            reverse=True,
        )
        return [e.row() for e in ordered]


class DataStore:
    """The task's data, loaded once and shared by every tool.

    Also owns the fold assignment. It is written to disk alongside a parquet copy
    of the data so that code running in a subprocess scores on exactly the same
    splits as the built-in evaluator -- the hybrid depends on that equality.
    """

    def __init__(self, spec: TaskSpec, cache_dir: Path, seed: int = 42):
        self.spec = spec
        self.seed = seed
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.train = pd.read_csv(spec.files["train"])
        self.test = pd.read_csv(spec.files["test"])
        self.dictionary = (
            pd.read_csv(spec.files["dictionary"]) if "dictionary" in spec.files else None
        )
        self.sample_submission = (
            pd.read_csv(spec.files["sample_submission"])
            if "sample_submission" in spec.files
            else None
        )

        self.y = self.train[spec.target_column]
        drop = [c for c in (spec.target_column,) if c in self.train.columns]
        self.X = self.train.drop(columns=drop)
        self.dtype_fixes = self._align_test_dtypes()

        self.fold_id = self._make_folds()
        self._write_cache()

    def _align_test_dtypes(self) -> dict[str, str]:
        """Make test columns match the dtypes train was profiled with.

        A column read as float in train and as object in test -- which happens
        when one split holds a stray non-numeric value, or when a column is
        entirely missing in one of them -- would be routed to the numeric branch
        at fit time and then fail at predict time, after the model had already
        been trained. Reconciling it once here turns a late crash into a note.
        """
        fixes: dict[str, str] = {}
        for column in self.X.columns:
            if column not in self.test.columns:
                continue
            wanted = self.X[column].dtype
            if self.test[column].dtype == wanted:
                continue
            try:
                if pd.api.types.is_numeric_dtype(wanted):
                    coerced = pd.to_numeric(self.test[column], errors="coerce")
                    lost = int(coerced.isna().sum() - self.test[column].isna().sum())
                    self.test[column] = coerced.astype(wanted, errors="ignore")
                    fixes[column] = (
                        f"test coerced to {wanted}"
                        + (f"; {lost} values became missing" if lost > 0 else "")
                    )
                else:
                    self.test[column] = self.test[column].astype(wanted)
                    fixes[column] = f"test cast to {wanted}"
            except Exception as err:  # noqa: BLE001
                fixes[column] = f"could not reconcile train {wanted} with test {self.test[column].dtype}: {err}"
        return fixes

    # -- folds -------------------------------------------------------------

    def _n_splits(self) -> int:
        """Fewer folds on large data: five refits of 90k rows is a real cost here."""
        n = len(self.train)
        if n > 200_000:
            return 3
        return 5

    def _make_folds(self) -> np.ndarray:
        from sklearn.model_selection import KFold, StratifiedKFold

        n_splits = self._n_splits()
        fold_id = np.empty(len(self.train), dtype=np.int8)

        if self.spec.is_classification:
            counts = self.y.value_counts()
            # Stratification needs at least one member of each class per fold.
            if counts.min() < n_splits:
                n_splits = max(2, int(counts.min()))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            split = splitter.split(self.X, self.y)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            split = splitter.split(self.X)

        for index, (_, val_idx) in enumerate(split):
            fold_id[val_idx] = index
        self.n_splits = n_splits
        return fold_id

    def splits(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (np.where(self.fold_id != k)[0], np.where(self.fold_id == k)[0])
            for k in range(self.n_splits)
        ]

    # -- cache for subprocesses -------------------------------------------

    def _write_cache(self) -> None:
        """Parquet copies plus the fold assignment, for code run in a subprocess.

        Re-reading a 43 MB CSV on every step is pure waste; parquet with the
        dtypes already settled removes both the parse cost and the risk that a
        subprocess infers a column differently than the evaluator did.
        """
        self.train_parquet = self.cache_dir / "train.parquet"
        self.test_parquet = self.cache_dir / "test.parquet"
        self.folds_csv = self.cache_dir / "folds.csv"

        self.train.to_parquet(self.train_parquet, index=False)
        self.test.to_parquet(self.test_parquet, index=False)
        pd.DataFrame(
            {self.spec.id_column: self.train[self.spec.id_column], "fold": self.fold_id}
        ).to_csv(self.folds_csv, index=False)

        (self.cache_dir / "task.json").write_text(
            json.dumps(self.spec.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def paths(self) -> dict[str, str]:
        """Cache locations, as paths the agent can paste straight into code."""
        return {
            "train_parquet": str(self.train_parquet),
            "test_parquet": str(self.test_parquet),
            "folds_csv": str(self.folds_csv),
            "cache_dir": str(self.cache_dir),
        }

    def memory_mb(self) -> float:
        return float(
            (self.train.memory_usage(deep=True).sum() + self.test.memory_usage(deep=True).sum())
            / 1e6
        )
