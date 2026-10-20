"""Experimental mods: drop a module here and it becomes a tool.

The gateway exists so that the team can try ideas without touching the core. A
mod is an ordinary module that registers tools with ``@tool(experimental=True)``;
it is discovered at startup, and if it raises at import or at call time it is
disabled rather than allowed to end the run.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def load_all() -> dict[str, str]:
    """Import every mod in this package. Returns {module: error} for the failures."""
    failures: dict[str, str] = {}
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{info.name}")
        except Exception as err:  # noqa: BLE001 - a broken idea must not cost a submission
            failures[info.name] = f"{type(err).__name__}: {err}"
    return failures
