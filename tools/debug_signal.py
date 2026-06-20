#!/usr/bin/env python
"""
tools/debug_signal.py
=====================
Quick diagnostic: shows the current raw vs calibrated per-model p distribution
from the last N rows of live_signals.csv and today's live_meta_log.

Purpose: after applying DL_BIAS_* calibration, confirms whether the offsets
are large enough to bring the ensemble signal toward 50/50 LONG/SHORT, or
whether the current regime has drifted beyond the calibration range.

Usage:
    python tools/debug_signal.py
    python tools/debug_signal.py --rows 200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
_LOGS = _ROOT / "logs"
_ENV  = _ROOT / ".env"

KNOWN_MODELS = ["lstm", "tcn", "tx"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live signal bias diagnostic")
    p.add_argument("--rows", type=int, default=100,
                   help="Number of recent rows to analyse (default 100)")
    return p.parse_args()


def _read_env_bias(model: str) -> float:
    key = f"DL_BIAS_{model.upper()}="
    try:
        with open(_ENV, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith(key):
                    return float(s[len(key):])
    except Exception:
        pass
    return 0.0


def _read_env_temp(model: str) -> float:
    key = f"DL_TEMP_{model.upper()}="
    try:
        with open(_ENV, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith(key):
                    return float(s[len(key):])
    except Exception:
        pass
    return 1.0


def tail_csv(path: Path, n: int) -> tuple[list[str], list[dict]]:
    """Return (header, last-n-rows) from a CSV file."""
    if not path.exists():
        return [], []
    buf: deque[list[str]] = deque(maxlen=n + 1)
    header: list[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, parts in enumerate(reader):
            if i == 0:
                header = parts
            else:
                buf.append(parts)
    rows = [dict(zip(header, p)) for p in buf if len(p) >= len(header)]
    return header, rows


def safe_float(v: str, default: float = float("nan")) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except (ValueError, TypeError):
        return default


def _pct(n: int, total: int) -> str:
    return f"{n/total:.1%}" if total else "n/a"


def main() -> None:
    args = parse_args()

    # Read current calibration params from .env
    biases = {m: _read_env_bias(m) for m in KNOWN_MODELS}
    temps  = {m: _read_env_temp(m) for m in KNOWN_MODELS}

    print(f"\n[debug_signal]  last {args.rows} rows")
    print(f"  Current .env calibration:")
    for m in KNOWN_MODELS:
        print(f"    DL_BIAS_{m.upper()}={biases[m]:+.4f}  DL_TEMP_{m.upper()}={temps[m]:.1f}")

    # -----------------------------------------------------------------------
    # live_signals.csv -- shows calibrated aggregate signal (p_meta = p_centered)
    # -----------------------------------------------------------------------
    print(f"\n--- live_signals.csv  (last {args.rows} rows) ---")
    _, sig_rows = tail_csv(_LOGS / "live_signals.csv", args.rows)
    if not sig_rows:
        print("  File not found or empty.")
    else:
        p_metas = [safe_float(r.get("p_meta", "")) for r in sig_rows]
        p_metas = [p for p in p_metas if not math.isnan(p)]
        allow1  = [r for r in sig_rows if r.get("allow", "0") == "1"]
        long_a1  = sum(1 for r in allow1 if r.get("side_hint", "") == "LONG")
        short_a1 = sum(1 for r in allow1 if r.get("side_hint", "") == "SHORT")
        print(f"  Rows: {len(sig_rows)}  allow=1: {len(allow1)}  "
              f"LONG: {long_a1} ({_pct(long_a1, len(allow1))})  "
              f"SHORT: {short_a1} ({_pct(short_a1, len(allow1))})")
        if p_metas:
            print(f"  p_centered (calibrated agg - 0.5):  "
                  f"min={min(p_metas):+.4f}  max={max(p_metas):+.4f}  "
                  f"mean={sum(p_metas)/len(p_metas):+.4f}")
        if allow1:
            a1_p = [safe_float(r.get("p_meta", "")) for r in allow1]
            a1_p = [p for p in a1_p if not math.isnan(p)]
            if a1_p:
                print(f"  allow=1 p_centered:  "
                      f"min={min(a1_p):+.4f}  max={max(a1_p):+.4f}  "
                      f"mean={sum(a1_p)/len(a1_p):+.4f}")

    # -----------------------------------------------------------------------
    # Today's meta log -- shows per-model calibrated p
    # -----------------------------------------------------------------------
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    meta_path = _LOGS / f"live_meta_log_{today_str}.csv"
    print(f"\n--- {meta_path.name}  (last {args.rows} rows) ---")
    _, meta_rows = tail_csv(meta_path, args.rows)
    if not meta_rows:
        print("  File not found or empty (writer not yet running today?).")
    else:
        print(f"  Rows: {len(meta_rows)}")
        for m in KNOWN_MODELS:
            col = f"{m}_p"
            vals_cal = [safe_float(r.get(col, "")) for r in meta_rows]
            vals_cal = [v for v in vals_cal if not math.isnan(v)]
            if not vals_cal:
                print(f"  {m.upper():6s}: column '{col}' not found -- "
                      f"meta log may still have 7-col header from old run")
                continue
            # Reconstruct raw p: raw = calibrated + bias
            bias = biases[m]
            vals_raw = [min(1.0 - 1e-6, max(1e-6, v + bias)) for v in vals_cal]
            cal_long = sum(1 for v in vals_cal if v >= 0.5)
            raw_long = sum(1 for v in vals_raw if v >= 0.5)
            n = len(vals_cal)
            cal_mean = sum(vals_cal) / n
            raw_mean = sum(vals_raw) / n
            print(f"  {m.upper():6s}  n={n}")
            print(f"    calibrated p:  mean={cal_mean:.4f}  LONG={cal_long}/{n} "
                  f"({_pct(cal_long, n)})")
            print(f"    raw p (+ bias {bias:+.4f}):  mean={raw_mean:.4f}  "
                  f"LONG={raw_long}/{n} ({_pct(raw_long, n)})")
            # Suggest new bias needed
            sorted_raw = sorted(vals_raw)
            raw_median = sorted_raw[n // 2]
            new_bias = raw_median - 0.5
            print(f"    raw median={raw_median:.4f}  => new DL_BIAS_{m.upper()} "
                  f"needed: {new_bias:+.4f}  "
                  f"(current: {bias:+.4f}  delta: {new_bias - bias:+.4f})")

    # -----------------------------------------------------------------------
    # heartbeat summary
    # -----------------------------------------------------------------------
    print(f"\n--- heartbeat.json ---")
    hb_path = _LOGS / "heartbeat.json"
    if hb_path.exists():
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            print(f"  event={hb.get('event')}  bias_locked={hb.get('bias_locked')}  "
                  f"ts={hb.get('ts')}")
            if "recent_long" in hb:
                rl, rs = hb["recent_long"], hb["recent_short"]
                total = rl + rs
                print(f"  recent allowed: {rl}L / {rs}S / {total} total  "
                      f"({_pct(rl, total)} LONG)")
        except Exception as exc:
            print(f"  read error: {exc}")
    else:
        print("  not found")
    print()


if __name__ == "__main__":
    main()
