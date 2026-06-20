"""V2 log archiver — gzip dated logs older than N days. DRY-RUN by default.

Moves files like trades_paper_20260530.csv / live_meta_log_20260530.csv into
logs/archive/YYYYMM/<name>.gz once they are older than --days (UTC, by the
date embedded in the FILENAME, strictly older than the cutoff day).

Safety properties:
  - DRY-RUN unless --apply is passed: prints the plan, touches nothing.
  - Only files matching  ^[A-Za-z_]+_YYYYMMDD.(csv|log)$  are candidates.
    Undated masters (trades_closed.csv, live_signals.csv, live_meta_log.csv,
    *.out/*.err, heartbeats, state files) can never match.
  - Today's file can never be selected (cutoff is in the past).
  - Files already under logs/archive/ are never re-scanned.
  - Apply = gzip-copy -> verify decompressed size equals the original ->
    only then delete the original. Verification failure keeps the original
    and exits 1. Restore with: gunzip logs/archive/YYYYMM/<name>.gz

Run AFTER v2_evidence_export.py (the exporter reads the same dated CSVs).

Usage: python tools/v2_log_archive.py [--days 14] [--logs-dir ...] [--apply]
Exit codes: 0 ok or nothing to do · 1 error / verification failure.
Stdlib only.
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATED_RE = re.compile(r"^[A-Za-z_]+_(\d{8})\.(csv|log)$")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Archive dated log files (dry-run by default).")
    ap.add_argument("--days", type=int, default=14,
                    help="archive files dated STRICTLY older than N days ago (UTC)")
    ap.add_argument("--logs-dir", default=str(BASE_DIR / "logs"))
    ap.add_argument("--apply", action="store_true", help="actually move files (default: dry-run)")
    return ap.parse_args(argv)


def find_candidates(logs_dir: Path, days: int) -> list:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    out = []
    if not logs_dir.is_dir():
        return out
    for p in sorted(logs_dir.iterdir()):
        if not p.is_file():
            continue
        m = DATED_RE.match(p.name)
        if not m:
            continue
        if m.group(1) < cutoff:  # strictly older than the cutoff day
            out.append(p)
    return out


def archive_one(path: Path, archive_root: Path) -> Path:
    """gzip-copy `path`, verify, delete original. Returns the .gz path."""
    day = DATED_RE.match(path.name).group(1)
    dest_dir = archive_root / day[:6]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (path.name + ".gz")

    original = path.read_bytes()
    with gzip.open(dest, "wb") as gz:
        gz.write(original)
    with gzip.open(dest, "rb") as gz:  # verify before deleting anything
        restored = gz.read()
    if restored != original:
        dest.unlink(missing_ok=True)
        raise IOError(f"verification failed for {path.name} - original kept")
    path.unlink()
    return dest


def main(argv=None) -> int:
    args = parse_args(argv)
    logs_dir = Path(args.logs_dir)
    archive_root = logs_dir / "archive"

    candidates = find_candidates(logs_dir, args.days)
    if not candidates:
        print(f"[archive] nothing older than {args.days} days in {logs_dir}")
        return 0

    total = sum(p.stat().st_size for p in candidates)
    mode = "APPLY" if args.apply else "DRY-RUN (pass --apply to act)"
    print(f"[archive] {mode}: {len(candidates)} file(s), {total/1024:.1f} KiB, "
          f"older than {args.days} days -> {archive_root}\\YYYYMM\\")
    for p in candidates:
        print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KiB)")

    if not args.apply:
        return 0

    failures = 0
    for p in candidates:
        try:
            dest = archive_one(p, archive_root)
            print(f"  archived {p.name} -> {dest.relative_to(logs_dir)}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR {p.name}: {exc}", file=sys.stderr)
    if failures:
        print(f"[archive] {failures} failure(s); originals kept where verification failed",
              file=sys.stderr)
        return 1
    print(f"[archive] done: {len(candidates)} file(s) archived")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
