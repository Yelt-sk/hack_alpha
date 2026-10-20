"""The run's shared objects, assembled once and read by every tool.

Tools are plain functions, so they need somewhere to find the clock, the data
and the ledger. A module-level session is the simplest thing that works and
keeps tool signatures free of plumbing arguments -- the model sees only the
parameters that are genuinely its decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state import Clock, DataStore, Ledger
from .submission import SubmissionManager
from .task import TaskSpec, load_task


@dataclass
class Session:
    spec: TaskSpec
    store: DataStore
    clock: Clock
    ledger: Ledger
    submissions: SubmissionManager
    #: Declared by the agent early; filters which tools are offered.
    regime: str = "standard"
    regime_rationale: str = ""
    #: How many times an already-scored experiment has been resubmitted.
    duplicates: int = 0
    events: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.events.append(f"[{self.clock.elapsed / 60:5.1f}m] {message}")


_SESSION: Session | None = None


def start(task_path: str | Path, total_seconds: float = 3 * 3600,
          reserve_seconds: float = 600, cache_dir: Path | None = None) -> Session:
    """Build the session: read the task, load the data, fix the folds."""
    global _SESSION
    clock = Clock(total_seconds=total_seconds, reserve_seconds=reserve_seconds)
    spec = load_task(task_path)
    store = DataStore(spec, cache_dir or (spec.task_dir / ".agent_cache"))
    _SESSION = Session(
        spec=spec,
        store=store,
        clock=clock,
        ledger=Ledger(spec.direction),
        submissions=SubmissionManager(
            spec, store.test[spec.id_column], known_labels=spec.classes
        ),
    )
    return _SESSION


def session() -> Session:
    if _SESSION is None:
        raise RuntimeError("session has not been started")
    return _SESSION


def summary() -> dict[str, Any]:
    """What `get_context` reports. Kept here so the loop can log it too."""
    current = session()
    best = current.ledger.best()
    return {
        "task": current.spec.summary(),
        "time": current.clock.summary(),
        "regime": current.regime,
        "experiments": len(current.ledger),
        "best": best.row() if best else None,
        "submission": current.submissions.describe(),
        "data_paths": current.store.paths(),
        "folds": current.store.n_splits,
    }
