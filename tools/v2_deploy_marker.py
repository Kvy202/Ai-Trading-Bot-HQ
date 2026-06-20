"""V2 deploy marker — record every deploy/config change for the audit trail.

Appends one row to logs/deploy_markers.csv (ts_utc, git_sha, branch, note) AND
one line to logs/DEPLOY_MARKERS.txt in the exact format tools/sim_exits.py
parses for --since lookup:

    2026-06-12 12:32:59 UTC fd8128b deployed: <note>

Run it on every deploy, flag change, or rollback (see OPERATIONS_RUNBOOK.md).

Usage:  python tools/v2_deploy_marker.py --note "v2 risk layer enabled (paper)"
Exit codes: 0 ok · 1 IO failure (git failures degrade to sha/branch="unknown").
Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_HEADER = ["ts_utc", "git_sha", "branch", "note"]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append a deploy marker (csv + txt).")
    ap.add_argument("--note", required=True, help="what was deployed/changed")
    ap.add_argument("--logs-dir", default=str(BASE_DIR / "logs"))
    return ap.parse_args(argv)


def git_info() -> tuple:
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=BASE_DIR, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or "unknown"
        except Exception:
            return "unknown"
    return run("rev-parse", "--short", "HEAD"), run("rev-parse", "--abbrev-ref", "HEAD")


def main(argv=None) -> int:
    args = parse_args(argv)
    logs_dir = Path(args.logs_dir)
    sha, branch = git_info()
    now = datetime.now(timezone.utc)
    ts_csv = now.strftime("%Y-%m-%d %H:%M:%S%z")
    ts_txt = now.strftime("%Y-%m-%d %H:%M:%S")
    note = args.note.strip()

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = logs_dir / "deploy_markers.csv"
        new_file = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(CSV_HEADER)
            w.writerow([ts_csv, sha, branch, note])

        # sim_exits.py --since compatibility: "<ts> UTC <sha> deployed: <note>"
        with (logs_dir / "DEPLOY_MARKERS.txt").open("a", encoding="utf-8") as f:
            f.write(f"{ts_txt} UTC {sha} deployed: {note}\n")
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[deploy_marker] {ts_txt} UTC {sha} ({branch}): {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
