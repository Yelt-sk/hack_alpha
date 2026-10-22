"""Experimental mods: drop a ``mod_*.py`` beside ``solution.py`` and it becomes a tool.

The gateway exists so the team can try ideas without touching the core. A mod is
an ordinary module that registers tools with ``@tool(experimental=True)``; it is
discovered at startup, and if it raises at import or at call time it is disabled
rather than allowed to end the run.

Discovery is by filename prefix in the directory holding this file, because that
directory is exactly what the platform copies next to the task. A mod in a
subfolder would be left behind, so there is nowhere else to look.
"""

from __future__ import annotations

import importlib
from pathlib import Path

#: What marks a module as a mod. Chosen to be unmistakable in a flat directory
#: that also holds the task's own CSVs and the core modules.
PREFIX = "mod_"


def discover() -> list[str]:
    """Module names of the mods present, without importing them."""
    here = Path(__file__).resolve().parent
    return sorted(
        path.stem for path in here.glob(f"{PREFIX}*.py")
        if not path.stem.startswith(f"{PREFIX}_")
    )


def load_all() -> dict[str, str]:
    """Import every mod. Returns ``{module: error}`` for the ones that failed."""
    failures: dict[str, str] = {}
    for name in discover():
        try:
            importlib.import_module(name)
        except Exception as err:  # noqa: BLE001 - a broken idea must not cost a submission
            failures[name] = f"{type(err).__name__}: {err}"
    return failures
