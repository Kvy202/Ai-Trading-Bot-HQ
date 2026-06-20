#!/usr/bin/env python
"""
tools/watchdog.py
=================
Health monitor for the live trading stack.

Checks (in order):
  1. Process liveness  -- live_writer.py and live_executor.py
  2. Writer heartbeat  -- logs/live_writer_heartbeat.json must be fresh
  3. Meta-log growth   -- live_meta_log*.csv must have been written recently
  4. Executor heartbeat -- logs/heartbeat.json must not be ancient

Restart policy (--restart flag):
  - Only if ALL monitored processes are dead (safe to run run_all.ps1 without -Force)
  - If only SOME processes are dead, log CRITICAL but do NOT auto-restart
    (avoids duplicate processes and data corruption)

Usage:
    python tools/watchdog.py                    # single check; exit 0=ok, 1=problem
    python tools/watchdog.py --loop 60          # check every 60 s, log to logs/watchdog.log
    python tools/watchdog.py --loop 60 --restart  # also restart if all processes dead
    python tools/watchdog.py --quiet            # suppress stdout, only write log file
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LOGS = _ROOT / "logs"

UTC = timezone.utc

# Staleness limits
WRITER_HB_MAX_AGE   = 120    # live_writer_heartbeat.json: writer ticks every ~3s
META_LOG_MAX_AGE    = 300    # live_meta_log*.csv: new row per writer tick
EXECUTOR_HB_MAX_AGE = 600    # heartbeat.json: executor only writes on trades


# ---------------------------------------------------------------------------
# Process detection (Windows: CimInstance, fallback: tasklist)
# ---------------------------------------------------------------------------

def _find_pids(keyword: str) -> list[int]:
    """Return PIDs of python.exe processes whose CommandLine contains keyword."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"Get-CimInstance Win32_Process | "
             f"Where-Object {{ $_.Name -eq 'python.exe' -and $_.CommandLine -match '{keyword}' }} | "
             f"Select-Object -ExpandProperty ProcessId"],
            text=True, timeout=15, stderr=subprocess.DEVNULL
        )
        return [int(p) for p in out.strip().splitlines() if p.strip().isdigit()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_process(keyword: str) -> tuple[bool, str]:
    """keyword is matched against the CommandLine of python.exe processes."""
    pids = _find_pids(keyword)
    if pids:
        return True, f"running (PIDs: {pids})"
    return False, "NOT RUNNING"


def check_heartbeat(path: Path, max_age_sec: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "file missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts_str = data.get("ts", "")
        ts = datetime.fromisoformat(ts_str.replace("+0000", "+00:00"))
        age = (datetime.now(UTC) - ts).total_seconds()
        if age > max_age_sec:
            return False, f"STALE ({age:.0f}s old, limit {max_age_sec}s)"
        return True, f"fresh ({age:.0f}s old)"
    except Exception as exc:
        return False, f"read error: {exc}"


def check_meta_log_growth(max_age_sec: int) -> tuple[bool, str]:
    """Find the most recently modified live_meta_log*.csv and check its mtime."""
    candidates = list(_LOGS.glob("live_meta_log*.csv"))
    if not candidates:
        return False, "no live_meta_log*.csv found"
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    age = time.time() - newest.stat().st_mtime
    if age > max_age_sec:
        return False, f"{newest.name} not written in {age:.0f}s (limit {max_age_sec}s)"
    return True, f"{newest.name} written {age:.0f}s ago"


# ---------------------------------------------------------------------------
# Restart helper
# ---------------------------------------------------------------------------

def attempt_restart(quiet: bool) -> None:
    log("RESTART: all processes dead -- running run_all.ps1 -Paper", quiet)
    script = _ROOT / "tools" / "run_all.ps1"
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script), "-Paper"],
            cwd=str(_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log("RESTART: launched run_all.ps1 (async, check logs in ~5s)", quiet)
    except Exception as exc:
        log(f"RESTART FAILED: {exc}", quiet)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_file: Path | None = None


def log(msg: str, quiet: bool = False) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    if not quiet:
        print(line)
    if _log_file is not None:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Single check pass
# ---------------------------------------------------------------------------

def run_check(quiet: bool) -> bool:
    """Run all checks. Returns True if everything is healthy."""
    results: list[tuple[str, bool, str]] = []

    # 1. Processes (match on unique keyword in CommandLine)
    ok_w, msg_w = check_process("live_writer")
    ok_e, msg_e = check_process("live_executor")
    results.append(("writer process",   ok_w, msg_w))
    results.append(("executor process", ok_e, msg_e))

    # 2. Writer heartbeat
    ok_wh, msg_wh = check_heartbeat(
        _LOGS / "live_writer_heartbeat.json", WRITER_HB_MAX_AGE
    )
    results.append(("writer heartbeat", ok_wh, msg_wh))

    # 3. Meta-log growth
    ok_ml, msg_ml = check_meta_log_growth(META_LOG_MAX_AGE)
    results.append(("meta-log growth",  ok_ml, msg_ml))

    # 4. Executor heartbeat
    ok_eh, msg_eh = check_heartbeat(
        _LOGS / "heartbeat.json", EXECUTOR_HB_MAX_AGE
    )
    results.append(("executor heartbeat", ok_eh, msg_eh))

    all_ok = all(ok for _, ok, _ in results)

    # Print results
    for name, ok, msg in results:
        marker = "OK   " if ok else "WARN "
        log(f"  [{marker}] {name:22s} {msg}", quiet)

    if all_ok:
        log("STATUS: healthy", quiet)
    else:
        failed = [name for name, ok, _ in results if not ok]
        log(f"STATUS: DEGRADED -- {', '.join(failed)}", quiet)

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trading stack health monitor")
    p.add_argument("--loop", type=int, default=0, metavar="SEC",
                   help="Run continuously every SEC seconds (default: single check)")
    p.add_argument("--restart", action="store_true",
                   help="Restart all processes if ALL are dead (requires --loop)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress stdout; only write to logs/watchdog.log")
    return p.parse_args()


def main() -> None:
    global _log_file
    args = parse_args()

    _LOGS.mkdir(exist_ok=True)
    _log_file = _LOGS / "watchdog.log"

    if args.loop:
        log(f"watchdog started -- interval={args.loop}s  restart={args.restart}",
            args.quiet)
        while True:
            log("--- check ---", args.quiet)
            healthy = run_check(args.quiet)

            if not healthy and args.restart:
                ok_w2, _ = check_process("live_writer")
                ok_e2, _ = check_process("live_executor")
                if not ok_w2 and not ok_e2:
                    attempt_restart(args.quiet)
                else:
                    log("RESTART skipped: some processes still alive; "
                        "manual intervention required", args.quiet)

            time.sleep(args.loop)
    else:
        log("--- single check ---", args.quiet)
        healthy = run_check(args.quiet)
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
