"""
Workspace state: datasets, runs, leaderboard, holdout budget.

Persisted to disk so the server can restart mid-session without losing the
agent's progress -- during a live demo that matters more than it sounds.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "workspace"
ROOT.mkdir(parents=True, exist_ok=True)

# Calls to final_evaluate per dataset. Small on purpose: an unlimited holdout is
# just a second validation set, and the agent will overfit to it.
HOLDOUT_BUDGET = 3


@dataclass
class Dataset:
    id: str
    source: str
    target: str
    task: str
    train: pd.DataFrame
    holdout: pd.DataFrame
    issues: list[dict] = field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None
    # Columns the *operator* declared safe despite a leak flag. The agent cannot
    # add to this list; see guards.LEAK_NOT_DROPPED.
    trusted_columns: list[str] = field(default_factory=list)
    holdout_used: int = 0
    holdout_log: list[dict] = field(default_factory=list)
    notes: str = ""

    @property
    def X(self) -> pd.DataFrame:
        return self.train.drop(columns=[self.target])

    @property
    def y(self) -> pd.Series:
        return self.train[self.target]


def recipe_fingerprint(recipe: dict) -> str:
    """Identity of an experiment, ignoring its label.

    name/notes are documentation, not configuration: two runs differing only in
    those are the same experiment and must not be computed, or scored, twice.
    """
    import hashlib

    body = {k: v for k, v in recipe.items() if k not in ("name", "notes")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Run:
    id: str
    dataset_id: str
    recipe: dict
    primary_metric: str
    score: float
    result: dict
    diagnosis: list[str]
    warnings: list[str]
    created: float = field(default_factory=time.time)
    holdout_score: float | None = None
    tuned_from: str | None = None
    fingerprint: str = ""
    tainted: bool = False
    taint_reason: str = ""
    acknowledgements: list[str] = field(default_factory=list)


class Workspace:
    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.runs: dict[str, Run] = {}
        self._fitted: dict[str, Any] = {}

    # -- datasets ---------------------------------------------------------
    def add_dataset(self, ds: Dataset) -> None:
        self.datasets[ds.id] = ds

    def get_dataset(self, dataset_id: str) -> Dataset:
        if dataset_id not in self.datasets:
            raise KeyError(
                f"No dataset '{dataset_id}'. Loaded: {sorted(self.datasets) or 'none'}. "
                f"Call load_dataset first."
            )
        return self.datasets[dataset_id]

    # -- runs -------------------------------------------------------------
    def new_run_id(self) -> str:
        return f"run_{len(self.runs) + 1:03d}_{uuid.uuid4().hex[:4]}"

    def add_run(self, run: Run, fitted=None) -> None:
        self.runs[run.id] = run
        if fitted is not None:
            self._fitted[run.id] = fitted

    def find_duplicate(self, dataset_id: str, fingerprint: str) -> Run | None:
        return next((r for r in self.runs_for(dataset_id) if r.fingerprint == fingerprint), None)

    def get_run(self, run_id: str) -> Run:
        if run_id not in self.runs:
            raise KeyError(f"No run '{run_id}'. Known runs: {sorted(self.runs) or 'none'}.")
        return self.runs[run_id]

    def fitted(self, run_id: str):
        return self._fitted.get(run_id)

    def runs_for(self, dataset_id: str) -> list[Run]:
        return [r for r in self.runs.values() if r.dataset_id == dataset_id]

    def best_run(self, dataset_id: str, metric: str | None = None) -> Run | None:
        from .schema import LOWER_IS_BETTER

        rs = [r for r in self.runs_for(dataset_id)
              if r.recipe["model"]["type"] != "dummy" and not r.tainted
              and (metric is None or r.primary_metric == metric)]
        if not rs:
            return None
        lower = rs[0].primary_metric in LOWER_IS_BETTER
        return sorted(rs, key=lambda r: r.score, reverse=not lower)[0]

    def baseline_score(self, dataset_id: str, metric: str) -> float | None:
        for r in self.runs_for(dataset_id):
            if r.recipe["model"]["type"] == "dummy" and r.primary_metric == metric:
                return r.score
        return None

    # -- persistence ------------------------------------------------------
    def save_meta(self) -> None:
        payload = {
            "datasets": {k: {"source": v.source, "target": v.target, "task": v.task,
                             "n_train": len(v.train), "n_holdout": len(v.holdout),
                             "holdout_used": v.holdout_used} for k, v in self.datasets.items()},
            "runs": {k: {"dataset_id": v.dataset_id, "recipe": v.recipe, "score": v.score,
                         "metric": v.primary_metric, "holdout_score": v.holdout_score}
                     for k, v in self.runs.items()},
        }
        (ROOT / "state.json").write_text(json.dumps(payload, indent=2, default=str))


WORKSPACE = Workspace()
