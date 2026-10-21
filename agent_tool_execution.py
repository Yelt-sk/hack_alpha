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

    Each call is a new process: files persist, variables do not. The working
    directory is the task directory.

    The data is already available as parquet, which is far faster to read than the
    CSVs and carries settled dtypes. These environment variables point at it:
    `TRAIN_PARQUET`, `TEST_PARQUET`, `FOLDS_CSV`, `CACHE_DIR`. `FOLDS_CSV` holds
    the run's fold assignment -- use it, and your result stays comparable with
    every other experiment.

    To get a hand-written model onto the leaderboard, write out-of-fold and test
    predictions and pass them to `evaluate_predictions`.

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
