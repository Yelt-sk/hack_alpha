"""The escape hatch: arbitrary Python, in a subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from agent_session import session
from agent_registry import tool

#: Trimmed hard, because tool output is read by the model and every character is
#: a token. A traceback's last lines carry the information; its middle rarely does.
_MAX_STREAM = 6000


@tool(regimes=["standard", "custom"])
def run_python(code: str, timeout_s: int = 900) -> dict[str, Any]:
    """Execute Python in a fresh subprocess and return stdout, stderr and the exit code.

    **Each call is a new process: files persist, variables do not.** Anything you
    want in the next call must be written to `CACHE_DIR` first. The working
    directory is the task directory.

    Use `n_jobs=4`, not `-1`: the container has 8 cores and oversubscribing them
    across folds measured 21x slower for an identical score.

    A working start, with the paths and the run's own folds already wired up:

        import os, numpy as np, pandas as pd
        from sklearn.metrics import f1_score

        cache = os.environ["CACHE_DIR"]
        train = pd.read_parquet(os.environ["TRAIN_PARQUET"])
        test  = pd.read_parquet(os.environ["TEST_PARQUET"])
        folds = pd.read_csv(os.environ["FOLDS_CSV"])          # id,fold
        df    = train.merge(folds, on="id")

        oof = np.zeros((len(df), n_classes))
        for f in sorted(df["fold"].unique()):
            tr, va = df[df["fold"] != f], df[df["fold"] == f]
            model.fit(tr[features], tr[target])
            oof[va.index] = model.predict_proba(va[features])

        np.save(f"{cache}/oof_mine.npy", oof)       # or your result is lost
        np.save(f"{cache}/testP_mine.npy", test_probabilities)

    Always use `FOLDS_CSV`. A result scored on any other split is not comparable
    with anything else in the run, and cannot be judged against the leaderboard.

    To put a hand-written model on the leaderboard, save out-of-fold and test
    predictions as above and pass them to `evaluate_predictions`. A model that
    never reaches the ledger cannot be compared, cannot be blended, and cannot
    defend the submission it produced.

    Args:
        code: The program to run.
        timeout_s: Hard limit in seconds; capped by the time left in the run.
    """
    current = session()
    store, clock = current.store, current.clock

    budget = int(max(10, min(timeout_s, clock.usable_remaining)))
    if budget < 10:
        return {"status": "refused", "problem": "the time reserve has been reached",
                "hint": "Finalise now rather than starting new work."}

    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": os.environ.get("HOME", str(store.cache_dir)),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        # Threads are capped for the same reason the models cap them: oversubscribing
        # eight cores makes everything slower without making anything better.
        "OMP_NUM_THREADS": "4",
        "TRAIN_PARQUET": str(store.train_parquet),
        "TEST_PARQUET": str(store.test_parquet),
        "FOLDS_CSV": str(store.folds_csv),
        "CACHE_DIR": str(store.cache_dir),
    }

    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", code],
            capture_output=True, text=True, timeout=budget,
            cwd=str(current.spec.task_dir), env=environment,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "problem": f"execution exceeded {budget}s",
                "hint": "Work on a sample first, or lower the model size."}

    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-_MAX_STREAM:],
        "stderr": completed.stderr[-_MAX_STREAM:],
        "time": current.clock.summary(),
    }
