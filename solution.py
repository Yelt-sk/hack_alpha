#!/usr/bin/env python
"""Entry point. The platform runs this and grades the file it leaves behind.

    python solution.py task.md

Everything it guarantees is here rather than in the agent, because the agent is
the part that might not finish:

* a valid ``submission.csv`` exists within seconds and only ever improves;
* a watchdog writes the best known answer if time runs out;
* the agent failing, the endpoint being unreachable, or no key being configured
  all fall back to a deterministic pipeline instead of to nothing.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from agent import autopilot, bootstrap, session as session_module  # noqa: E402
from agent import rag as _rag_seam  # noqa: E402,F401  (registers search_knowledge)
from agent.tools import core as _core_tools  # noqa: E402,F401  (registers the tools)
from agent.tools.mods import load_all as load_mods  # noqa: E402

#: The platform allows three hours; stop early enough to write the answer calmly.
TOTAL_SECONDS = float(os.environ.get("AGENT_TOTAL_SECONDS", 3 * 3600))
RESERVE_SECONDS = float(os.environ.get("AGENT_RESERVE_SECONDS", 600))
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", 120))


def _watchdog(session, log) -> None:
    """Make sure an answer exists, whatever else is happening."""
    while True:
        if session.clock.remaining <= 30:
            if not session.submissions.exists():
                log("[watchdog] no submission on disk; writing the constant floor")
                session.submissions.save_constant(session.store.y)
            log("[watchdog] time is up")
            return
        threading.Event().wait(min(20.0, max(1.0, session.clock.remaining - 25)))


def main() -> int:
    log = print
    task_path = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")

    session = session_module.start(
        task_path, total_seconds=TOTAL_SECONDS, reserve_seconds=RESERVE_SECONDS
    )
    log(f"[solution] {session.spec.kind} task, metric {session.spec.metric}, "
        f"{session.spec.n_train} train / {session.spec.n_test} test rows")
    for warning in session.spec.warnings:
        log(f"[solution] warning: {warning}")

    failures = load_mods()
    if failures:
        log(f"[solution] experimental mods that failed to load: {failures}")

    # Rung one, before anything that can fail.
    bootstrap.floor(session, log)

    def finalise(*_: object) -> None:
        if not session.submissions.exists():
            session.submissions.save_constant(session.store.y)
        log(f"[solution] final: {session.submissions.describe()}")

    signal.signal(signal.SIGTERM, lambda *_: (finalise(), sys.exit(0)))
    threading.Thread(target=_watchdog, args=(session, log), daemon=True).start()

    try:
        bootstrap.baseline(session, log)

        if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"):
            from agent import loop

            try:
                outcome = loop.run(session, max_steps=MAX_STEPS, log=log)
                log(f"[solution] agent stopped after {outcome['steps']} steps "
                    f"({outcome['stopped']})")
                if outcome["final_message"]:
                    log(f"[solution] agent report:\n{outcome['final_message'][:3000]}")
            except Exception as err:  # noqa: BLE001
                log(f"[solution] the agent loop failed: {type(err).__name__}: {err}")
                traceback.print_exc()

            # Whatever ended the agent -- finishing early, running out of steps,
            # a dead endpoint -- the remaining minutes are still worth spending.
            # The autopilot can only improve the answer: a worse result is refused
            # by the submission manager, so this is free insurance.
            if session.clock.usable_remaining > 120:
                log(f"[solution] {session.clock.usable_remaining / 60:.1f} min left; "
                    f"continuing with the deterministic pipeline")
                autopilot.run(session, log)
        else:
            log("[solution] no model configured; running the deterministic pipeline")
            autopilot.run(session, log)

    except Exception as err:  # noqa: BLE001
        log(f"[solution] unrecoverable error: {type(err).__name__}: {err}")
        traceback.print_exc()
    finally:
        finalise()
        best = session.ledger.best()
        if best:
            log(f"[solution] best experiment: {best.row()}")
        log(f"[solution] elapsed {session.clock.elapsed / 60:.1f} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
