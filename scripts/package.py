#!/usr/bin/env python
"""Build the submission ZIP, and prove it works the way the platform runs it.

    python scripts/package.py                 # build + verify on both tasks
    python scripts/package.py --seconds 300   # give each verification run longer

The platform copies **only the root ``.py`` files and ``requirements.txt``** into
the task directory and runs ``python3 solution.py task.md`` there. A module left
in a subfolder is simply absent at grading time, which costs the whole run.

So this script does not trust the layout: it copies exactly what the platform
copies into a scratch task directory and runs from there. If an import lives
somewhere that does not survive the trip, the verification fails here instead of
on the leaderboard.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

#: Everything the platform copies, and nothing else.
def payload() -> list[Path]:
    files = sorted(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]
    return [p for p in files if p.exists()]


def build(destination: Path) -> Path:
    """Write the ZIP. Flat, no folders, as the instructions require."""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in payload():
            archive.write(path, arcname=path.name)
    return destination


def verify(task_dir: Path, seconds: float, env_extra: dict[str, str] | None = None,
           log_dir: str | None = None) -> bool:
    """Run the solution the way the platform does, in a copy of the task directory."""
    with tempfile.TemporaryDirectory(prefix="platform_") as tmp:
        scratch = Path(tmp) / task_dir.name
        shutil.copytree(task_dir, scratch,
                        ignore=shutil.ignore_patterns(".agent_cache", "submission.csv"))
        for path in payload():                      # the platform's copy step
            shutil.copy2(path, scratch / path.name)

        import os
        environment = {**os.environ,
                       "AGENT_TOTAL_SECONDS": str(seconds),
                       "AGENT_RESERVE_SECONDS": str(max(20.0, seconds * 0.15))}
        # Stripped rather than inherited by accident: a verification run that
        # silently used the model would not be testing the deterministic path.
        environment.pop("OPENAI_API_KEY", None)
        environment.pop("OPENAI_MODEL", None)
        environment.update(env_extra or {})

        # Written to a file as it happens rather than captured, so a three-hour
        # run can be watched with `tail -f` instead of staying silent until the
        # end -- which is exactly when watching stops being useful.
        log_path = Path(log_dir or tempfile.gettempdir()) / f"run_{task_dir.name}.log"
        print(f"\n{'=' * 70}\n{task_dir.name}: python3 solution.py task.md "
              f"({seconds:.0f}s budget)\n  log: {log_path}\n{'=' * 70}", flush=True)
        with log_path.open("w", encoding="utf-8") as sink:
            completed = subprocess.run(
                # -u for the same reason baseline.py uses it: without it Python
                # buffers in blocks when stdout is a file, and a log that only
                # appears at the end cannot be watched.
                [sys.executable, "-u", "solution.py", "task.md"],
                cwd=scratch, env=environment, stdout=sink, stderr=subprocess.STDOUT,
                text=True, timeout=seconds + 180,
            )
        output = log_path.read_text(encoding="utf-8")
        print("\n".join(output.strip().splitlines()[-30:]))
        if completed.returncode != 0:
            print(f"\nexit code {completed.returncode}")

        produced = scratch / "submission.csv"
        if not produced.exists():
            print("\nFAIL: no submission.csv was written")
            return False

        sys.path.insert(0, str(ROOT))
        import pandas as pd
        from agent_submission import SubmissionManager
        from agent_task import load_task

        spec = load_task(scratch / "task.md")
        test = pd.read_csv(spec.files["test"], usecols=[spec.id_column])
        manager = SubmissionManager(spec, test[spec.id_column])
        problems = manager.validate(pd.read_csv(produced))
        if problems:
            print(f"\nFAIL: the submission would be rejected: {problems}")
            return False
        print(f"\nOK: {produced.name} validates "
              f"({len(pd.read_csv(produced))} rows, columns "
              f"{[spec.id_column, spec.target_column]})")
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--with-model", action="store_true",
                        help="let the agent loop run, using OPENAI_* from this shell")
    parser.add_argument("--only", default="",
                        help="verify just the task whose directory name contains this")
    parser.add_argument("--log-dir", default="",
                        help="where to write the live run logs")
    parser.add_argument("--out", default=str(ROOT / "submission.zip"))
    args = parser.parse_args()

    archive = build(Path(args.out))
    names = zipfile.ZipFile(archive).namelist()
    nested = [n for n in names if "/" in n]
    size_mb = archive.stat().st_size / 1e6
    print(f"{archive.name}: {len(names)} files, {size_mb:.2f} MB")
    print(f"  nested paths: {nested or 'none -- flat, as required'}")
    if nested or size_mb > 100:
        print("  FAIL: the archive breaks the packaging rules")
        return 1

    if args.skip_verify:
        return 0

    import os
    passthrough = {}
    if args.with_model:
        missing = [k for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
                   if not os.environ.get(k)]
        if missing:
            print(f"  FAIL: --with-model needs {missing} in the environment")
            return 1
        passthrough = {k: os.environ[k] for k in
                       ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")}
        print(f"  model: {os.environ['OPENAI_MODEL']} via {os.environ['OPENAI_BASE_URL']}")

    ok = True
    for task_dir in sorted(p.parent for p in DATA.glob("*/task.md")):
        if args.only and args.only.lower() not in task_dir.name.lower():
            continue
        ok &= verify(task_dir, args.seconds, passthrough, args.log_dir)
    print(f"\n{'=' * 70}\n{'READY TO SUBMIT' if ok else 'NOT READY'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
