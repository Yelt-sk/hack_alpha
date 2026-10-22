#!/usr/bin/env python
"""Drive the tools by hand, without a language model.

    python scripts/drive.py <task.md> <calls.json>

`calls.json` is a list of ``{"tool": name, "args": {...}}``. They run in one
session, against the same folds and the same submission custody the agent uses,
and each result is printed compactly.

This exists to separate two questions that are easy to confuse: *can this tool
surface support good work*, and *will a given model find it*. Driving the tools
deliberately answers the first on its own, and what it produces is the bar any
model has to clear.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import agent_bootstrap as bootstrap  # noqa: E402,F401
import agent_rag as _rag  # noqa: E402,F401
import agent_session as session_module  # noqa: E402,F401
import agent_tools as _core  # noqa: E402,F401
import agent_mods  # noqa: E402
from agent_registry import REGISTRY  # noqa: E402

# Mods too, same as a real run: a tool that only exists when solution.py loads it
# is a tool this script cannot be used to test.
_MOD_FAILURES = agent_mods.load_all()

#: Keys worth seeing at a glance; the rest is folded away.
HIGHLIGHT = ("status", "experiment_id", "score", "fold_std", "verdict", "problem",
             "hint", "next_step", "seconds", "untried_moves", "tuning")


def summarise(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:400]
    if not isinstance(payload, dict):
        return raw[:400]
    shown = {k: payload[k] for k in HIGHLIGHT if k in payload}
    rest = [k for k in payload if k not in shown]
    out = json.dumps(shown, ensure_ascii=False)[:900]
    return f"{out}   (+ {', '.join(rest[:8])})" if rest else out


def main() -> int:
    task_path = sys.argv[1]
    calls = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    total = float(sys.argv[3]) if len(sys.argv) > 3 else 1800

    session = session_module.start(task_path, total_seconds=total, reserve_seconds=60)
    print(f"task: {session.spec.kind}, metric {session.spec.metric}, "
          f"{session.spec.n_train} train rows")
    bootstrap.floor(session)
    bootstrap.baseline(session)
    print(f"floor to beat: {session.ledger.best().score:.5f}")
    if _MOD_FAILURES:
        print(f"mods that failed to import: {_MOD_FAILURES}")
    print()

    for index, call in enumerate(calls, start=1):
        name, args = call["tool"], call.get("args", {})
        print(f"--- {index}. {name} {json.dumps(args, ensure_ascii=False)[:160]}")
        print(f"    {summarise(REGISTRY.dispatch(name, args))}\n")

    print("=" * 70)
    print(json.dumps(REGISTRY.dispatch("leaderboard", {}) and
                     json.loads(REGISTRY.dispatch("leaderboard", {}))["runs"],
                     ensure_ascii=False, indent=1)[:2500])
    print("\nsubmission:", session.submissions.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
