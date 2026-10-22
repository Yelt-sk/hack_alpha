#!/usr/bin/env python
"""Verify the archive itself, by unzipping it and running from the result.

    python scripts/verify_zip.py [seconds]

``scripts/package.py`` copies the root sources into a scratch task directory,
which is what the archive is built from. This closes the last gap: what gets
graded is the archive, so the archive is what gets run here -- extracted into a
copy of each task directory, then ``python3 solution.py task.md`` from there.

The model is deliberately not configured, so this exercises the path that has to
work when the endpoint is unreachable: the guarantee ladder on its own.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARCHIVE = ROOT / "submission.zip"


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 240

    if not ARCHIVE.exists():
        print(f"{ARCHIVE.name} is missing -- run scripts/package.py first")
        return 1

    names = zipfile.ZipFile(ARCHIVE).namelist()
    print(f"{ARCHIVE.name}: {len(names)} files, {ARCHIVE.stat().st_size / 1e6:.2f} MB")
    nested = [n for n in names if "/" in n or "\\" in n]
    if nested:
        print(f"FAIL: the archive has nested paths: {nested[:5]}")
        return 1
    for required in ("solution.py", "requirements.txt"):
        if required not in names:
            print(f"FAIL: {required} is not in the archive root")
            return 1

    import pandas as pd
    from agent_submission import SubmissionManager
    from agent_task import load_task

    ok = True
    for task_dir in sorted(p.parent for p in (ROOT / "data").glob("*/task.md")):
        with tempfile.TemporaryDirectory(prefix="zip_") as tmp:
            scratch = Path(tmp) / task_dir.name
            shutil.copytree(task_dir, scratch,
                            ignore=shutil.ignore_patterns(".agent_cache", "submission.csv"))
            with zipfile.ZipFile(ARCHIVE) as archive:   # the artifact, not the sources
                archive.extractall(scratch)

            environment = {**os.environ,
                           "AGENT_TOTAL_SECONDS": str(seconds),
                           "AGENT_RESERVE_SECONDS": str(max(20.0, seconds * 0.15))}
            environment.pop("OPENAI_API_KEY", None)
            environment.pop("OPENAI_MODEL", None)

            print(f"\n{'=' * 66}\n{task_dir.name}: unzipped, python3 solution.py task.md"
                  f"\n{'=' * 66}", flush=True)
            done = subprocess.run([sys.executable, "-u", "solution.py", "task.md"],
                                  cwd=scratch, env=environment, capture_output=True,
                                  text=True, timeout=seconds + 300)
            print("\n".join(done.stdout.strip().splitlines()[-8:]))

            produced = scratch / "submission.csv"
            if not produced.exists():
                print("FAIL: no submission.csv was written")
                ok = False
                continue
            spec = load_task(scratch / "task.md")
            test = pd.read_csv(spec.files["test"], usecols=[spec.id_column])
            problems = SubmissionManager(spec, test[spec.id_column]).validate(
                pd.read_csv(produced))
            if problems:
                print(f"FAIL: the submission would be rejected: {problems}")
                ok = False
                continue
            print(f"OK: validates, columns {[spec.id_column, spec.target_column]}")

    print(f"\n{'=' * 66}\n{'ARCHIVE VERIFIED' if ok else 'ARCHIVE NOT READY'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
