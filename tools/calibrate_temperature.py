#!/usr/bin/env python
"""
tools/calibrate_temperature.py
================================
Compute per-model calibration parameters (bias offset + temperature) that
bring each model's LONG rate into the target range using logged meta data.

Two calibration mechanisms are applied in order:
  1. Bias offset (DL_BIAS_<MODEL>): subtracts the model's median deviation from
     0.5 in probability space.  This is the primary fix when a model
     consistently predicts above or below 0.5 without being overconfident.
     Median is used (not mean) so the offset exactly centres the distribution:
     after applying it, exactly 50% of predictions land each side of 0.5
     regardless of distribution skew.  A mean-based offset undershoots
     for right-skewed distributions (the common LONG-bias case).
       p_debiased = clamp(p - bias, eps, 1-eps)
       bias = median(p_long) - 0.5

  2. Temperature (DL_TEMP_<MODEL>): logit-space sharpness scaling.
     Only useful when predictions are EXTREME (e.g. 0.9 or 0.1) and you
     want to soften them.  Has NO effect if the distribution is clustered
     near 0.5, because logit compression never crosses zero.
       p_t = sigmoid(logit(p_debiased) / t)

For the current LONG bias (TCN/TX clustered at 0.54-0.59), bias offset is
the correct tool.  Temperature is kept as a secondary adjustment.

Usage:
    python tools/calibrate_temperature.py                              # last 7 days
    python tools/calibrate_temperature.py --since 2026-05-08
    python tools/calibrate_temperature.py --since-datetime "2026-05-10 19:19:29"
    python tools/calibrate_temperature.py --write-env                  # patch .env
    python tools/calibrate_temperature.py --allow-only                 # allow=1 rows only
    python tools/calibrate_temperature.py --source rolling             # rolling log only

Safety gates (checked before --write-env):
  [1] Stability: recommended bias must not change sign across --stability-windows
      and must not spread more than --spread-limit across those windows.
  [2] Max step: bias change from current .env must not exceed --max-step.
  Use --force to bypass all safety gates and write the raw recommended values.

Output: recommended DL_BIAS_* and DL_TEMP_* env vars.
With --write-env: backs up .env -> .env.bak, then patches those lines.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

UTC = timezone.utc

_ROOT = Path(__file__).resolve().parents[1]
_LOGS = _ROOT / "logs"
_ENV  = _ROOT / ".env"

KNOWN_MODELS = ["lstm", "tcn", "tx"]
TEMP_GRID    = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]
_EPS         = 1e-6


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-model bias + temperature calibration")
    p.add_argument("--since", default=None,
                   help="Start date YYYY-MM-DD (default: 7 days ago); "
                        "ignored when --since-datetime is given")
    p.add_argument("--since-datetime", default=None,
                   help="Intraday UTC cutoff 'YYYY-MM-DD HH:MM:SS' -- only rows with "
                        "ts >= this value are used.  Supersedes --since.  Use when the "
                        "calibration fix was applied mid-day so pre-fix rows in the same "
                        "UTC day are excluded.")
    p.add_argument("--source", choices=["rolling", "daily", "auto"], default="auto",
                   help="Which meta log files to read: 'rolling' = only "
                        "live_meta_log.csv (full per-model header, no dup risk); "
                        "'daily' = only live_meta_log_2*.csv; "
                        "'auto' = rolling first, dated files as supplement (deduped by ts)")
    p.add_argument("--target", type=float, default=0.50,
                   help="Target LONG fraction (default 0.50)")
    p.add_argument("--write-env", action="store_true",
                   help="Patch DL_BIAS_* and DL_TEMP_* lines in .env (backs up .env.bak)")
    p.add_argument("--allow-only", action="store_true",
                   help="Use only allow=1 rows (default: use all rows)")
    p.add_argument("--undo-env-bias", action="store_true",
                   help="Add current DL_BIAS_* values back to logged p before computing "
                        "new offsets.  Use this when the window points to data logged AFTER "
                        "a previous calibration run (meta log stores calibrated p, so the "
                        "raw distribution must be reconstructed before re-calibrating).")
    p.add_argument("--stability-windows", type=int, nargs="+", default=[100, 250, 500],
                   metavar="N",
                   help="Row counts for stability check (default: 100 250 500). "
                        "Recommended bias must be consistent across all windows.")
    p.add_argument("--spread-limit", type=float, default=0.10,
                   help="Max allowed spread of recommended biases across stability "
                        "windows (default 0.10).  Larger spread => UNSTABLE.")
    p.add_argument("--max-step", type=float, default=0.05,
                   help="Max bias change from current .env per run (default 0.05). "
                        "Larger changes are capped unless --force is given.")
    p.add_argument("--force", action="store_true",
                   help="Bypass stability check and max-step cap; write raw recommended "
                        "values.  Use only when you have manually verified the distribution.")
    return p.parse_args()


def _read_env_bias(model: str) -> float:
    """Read DL_BIAS_<MODEL> directly from .env file (not from os.environ).

    Reading from the file rather than os.environ avoids picking up stale
    shell-level overrides that may not match what the writer is actually using.
    Returns 0.0 if the key is absent or the file is unreadable.
    """
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


# ---------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------

def apply_bias(p: float, bias: float) -> float:
    return max(_EPS, min(1.0 - _EPS, p - bias))


def apply_temp(p: float, t: float) -> float:
    if t == 1.0 or not (_EPS < p < 1.0 - _EPS):
        return p
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-logit / t))


def apply_calibration(p: float, bias: float, t: float) -> float:
    return apply_temp(apply_bias(p, bias), t)


def long_rate(vals: List[float], bias: float = 0.0, t: float = 1.0) -> float:
    if not vals:
        return float("nan")
    return sum(1 for p in vals if apply_calibration(p, bias, t) >= 0.5) / len(vals)


def compute_bias(vals: List[float]) -> float:
    """Median deviation of p_long from 0.5.

    Using the median (not mean) ensures exactly 50% of values land above
    0.5 after the offset is applied, regardless of distribution skew.
    A mean-based offset leaves residual bias whenever the distribution is
    asymmetric (right-skewed -> mean > median -> mean offset undershoots).
    """
    if not vals:
        return 0.0
    return float(np.median(vals)) - 0.5


def best_temp_after_bias(vals: List[float], bias: float, target: float) -> Tuple[float, float]:
    """Given a bias-corrected distribution, find temperature closest to target."""
    best_t, best_rate, best_dist = 1.0, float("nan"), float("inf")
    for t in TEMP_GRID:
        rate = long_rate(vals, bias, t)
        if math.isnan(rate):
            continue
        dist = abs(rate - target)
        if dist < best_dist:
            best_dist, best_t, best_rate = dist, t, rate
    return best_t, best_rate


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _parse_ts(s: str) -> Optional[datetime]:
    """Parse UTC timestamp strings like '2026-05-10 09:42:34+0000'."""
    try:
        return datetime.fromisoformat(s.strip().replace("+0000", "+00:00"))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_vals(since: date,
                    allow_only: bool,
                    undo_bias: bool = False,
                    since_dt: Optional[datetime] = None,
                    source: str = "auto") -> Dict[str, List[float]]:
    """Load per-model p_long values from meta log files.

    Sources (controlled by the `source` parameter):
      rolling  -- only logs/live_meta_log.csv (always has full per-model header)
      daily    -- only logs/live_meta_log_2*.csv (dated daily snapshots)
      auto     -- rolling first, then dated files as supplement for older dates.
                  Rows are deduplicated by ts so overlapping entries are counted once.

    since_dt: when given (from --since-datetime), filter rows by ts >= since_dt.
    undo_bias: add current DL_BIAS_* back to reconstruct raw p from calibrated logs.
    """
    current_biases = {m: (_read_env_bias(m) if undo_bias else 0.0)
                      for m in KNOWN_MODELS}
    if undo_bias:
        print(f"  undo-env-bias: adding back {current_biases}")

    result: Dict[str, List[float]] = {m: [] for m in KNOWN_MODELS}
    seen_ts: Set[str] = set()
    files_used = 0

    # Build ordered file list based on --source
    files_to_read: List[Tuple[Path, Optional[date]]] = []

    rolling = _LOGS / "live_meta_log.csv"
    if source in ("rolling", "auto") and rolling.exists():
        files_to_read.append((rolling, None))  # None = must filter rows by ts

    if source in ("daily", "auto"):
        for path in sorted(_LOGS.glob("live_meta_log_2*.csv")):
            stem_date_str = path.stem.replace("live_meta_log_", "")
            try:
                file_date = datetime.strptime(stem_date_str, "%Y%m%d").date()
            except ValueError:
                continue
            if file_date < since:
                continue
            files_to_read.append((path, file_date))

    for path, file_date in files_to_read:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header: Optional[List[str]] = None
            rows_this_file = 0

            for parts in reader:
                if not parts:
                    continue
                if header is None:
                    header = parts
                    continue
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))

                # Always parse ts -- needed for dedup and rolling-log date filter
                ts_str = row.get("ts", "").strip()
                row_ts = _parse_ts(ts_str)

                # Deduplication (guards against rolling + daily overlap)
                if ts_str and ts_str in seen_ts:
                    continue
                if ts_str:
                    seen_ts.add(ts_str)

                # Date / datetime filter
                if since_dt is not None:
                    if row_ts is None or row_ts < since_dt:
                        continue
                elif file_date is None:
                    # Rolling log with day-level filter only
                    if row_ts is None or row_ts.date() < since:
                        continue

                try:
                    allow = int(row.get("allow", 0))
                except (ValueError, TypeError):
                    allow = 0
                if allow_only and allow != 1:
                    continue

                for m in KNOWN_MODELS:
                    col = f"{m}_p"
                    if col not in row or not row[col]:
                        continue
                    try:
                        pval = float(row[col])
                    except ValueError:
                        continue
                    pval = min(1.0 - _EPS, max(_EPS, pval + current_biases[m]))
                    if math.isnan(pval) or math.isinf(pval) or not (_EPS < pval < 1.0 - _EPS):
                        continue
                    result[m].append(pval)
                    rows_this_file += 1

            if rows_this_file > 0:
                files_used += 1

    total = sum(len(v) for v in result.values())
    print(f"  Loaded {total} per-model values from {files_used} file(s)  "
          f"(source={source}, deduped={len(seen_ts)})")
    return result


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def stability_check(model_vals: Dict[str, List[float]],
                    windows: List[int],
                    spread_limit: float) -> Tuple[bool, List[str]]:
    """
    Compare recommended bias across tail sub-windows.
    Fails if any model's bias changes sign or spreads beyond spread_limit.
    Returns (all_stable, report_lines).
    """
    lines: List[str] = []
    all_stable = True

    for m in KNOWN_MODELS:
        vals = model_vals[m]
        if not vals:
            lines.append(f"  [skip  ]  {m.upper():6s}: no data")
            continue

        sub_biases: List[float] = []
        cols: List[str] = []
        for n in windows:
            sub = vals[-n:]
            b = compute_bias(sub)
            label = f"last-{len(sub)}" if len(sub) < n else f"last-{n}"
            sub_biases.append(b)
            cols.append(f"{label}={b:+.4f}")

        spread = max(sub_biases) - min(sub_biases)
        signs = set(1 if b >= 0 else -1 for b in sub_biases)
        sign_flip = len(signs) > 1
        stable = not sign_flip and spread <= spread_limit

        if not stable:
            all_stable = False
            reasons: List[str] = []
            if sign_flip:
                reasons.append("sign flip")
            if spread > spread_limit:
                reasons.append(f"spread={spread:.4f} > {spread_limit}")
            flag = "UNSTABLE"
        else:
            reasons = [f"spread={spread:.4f}"]
            flag = "ok"

        lines.append(f"  [{flag:8s}]  {m.upper():6s}:  {'  '.join(cols)}  "
                     f"({', '.join(reasons)})")

    return all_stable, lines


def maxstep_check(rec_bias: Dict[str, float],
                  max_step: float) -> Tuple[Dict[str, float], List[str], bool]:
    """
    Cap each recommended bias to at most max_step away from current .env value.
    Returns (capped_biases, report_lines, any_capped).
    """
    capped: Dict[str, float] = {}
    lines: List[str] = []
    any_capped = False

    for m in KNOWN_MODELS:
        current = _read_env_bias(m)
        rec = rec_bias.get(m, current)
        delta = rec - current
        if abs(delta) > max_step:
            direction = 1 if delta > 0 else -1
            capped_val = round(current + direction * max_step, 6)
            capped[m] = capped_val
            any_capped = True
            lines.append(
                f"  {m.upper():6s}:  current={current:+.4f}  rec={rec:+.4f}  "
                f"delta={delta:+.4f}  CAPPED -> {capped_val:+.4f}"
            )
        else:
            capped[m] = rec
            lines.append(
                f"  {m.upper():6s}:  current={current:+.4f}  rec={rec:+.4f}  "
                f"delta={delta:+.4f}  ok"
            )

    return capped, lines, any_capped


# ---------------------------------------------------------------------------
# .env patching
# ---------------------------------------------------------------------------

def patch_env(biases: Dict[str, float], temps: Dict[str, float]) -> None:
    if not _ENV.exists():
        print(f"  .env not found at {_ENV}, cannot patch")
        return

    bak = _ENV.with_suffix(".env.bak")
    shutil.copy2(_ENV, bak)
    print(f"  backed up .env -> {bak}")

    lines = _ENV.read_text(encoding="utf-8").splitlines()
    new_lines: List[str] = []
    seen: Dict[str, bool] = {}
    for m in KNOWN_MODELS:
        seen[f"DL_BIAS_{m.upper()}"] = False
        seen[f"DL_TEMP_{m.upper()}"] = False

    for line in lines:
        stripped = line.strip()
        matched = False
        for m in KNOWN_MODELS:
            bias_key = f"DL_BIAS_{m.upper()}"
            temp_key = f"DL_TEMP_{m.upper()}"
            if stripped.startswith(bias_key + "=") or stripped.startswith(f"# {bias_key}"):
                new_lines.append(f"{bias_key}={biases[m]:.4f}")
                seen[bias_key] = True
                matched = True
                break
            if stripped.startswith(temp_key + "=") or stripped.startswith(f"# {temp_key}"):
                new_lines.append(f"{temp_key}={temps[m]:.1f}")
                seen[temp_key] = True
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for m in KNOWN_MODELS:
        bias_key = f"DL_BIAS_{m.upper()}"
        temp_key = f"DL_TEMP_{m.upper()}"
        if not seen[bias_key]:
            new_lines.append(f"{bias_key}={biases[m]:.4f}")
            print(f"  added {bias_key}={biases[m]:.4f}")
        if not seen[temp_key]:
            new_lines.append(f"{temp_key}={temps[m]:.1f}")
            print(f"  added {temp_key}={temps[m]:.1f}")

    _ENV.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print("  .env patched")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve since / since_dt -- --since-datetime wins over --since
    since_dt: Optional[datetime] = None
    if args.since_datetime:
        since_dt = datetime.strptime(args.since_datetime,
                                     "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        since = since_dt.date()
        since_label = args.since_datetime + " UTC"
    elif args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").date()
        since_label = str(since)
    else:
        since = date.today() - timedelta(days=6)
        since_label = str(since)

    print(f"\n[calibrate] since={since_label}  target={args.target:.0%}  "
          f"allow_only={args.allow_only}  source={args.source}")

    model_vals = load_model_vals(since,
                                 allow_only=args.allow_only,
                                 undo_bias=args.undo_env_bias,
                                 since_dt=since_dt,
                                 source=args.source)

    # -----------------------------------------------------------------------
    # Full-window calibration analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  CALIBRATION ANALYSIS  (target LONG rate = {args.target:.0%})")
    print(f"{'='*64}")

    rec_bias: Dict[str, float] = {}
    rec_temp: Dict[str, float] = {}

    for m in KNOWN_MODELS:
        vals = model_vals[m]
        if not vals:
            print(f"\n  {m.upper():6s}: no data -- meta_log lacks '{m}_p' column.")
            print(f"           Restart writer to populate it.")
            rec_bias[m] = _read_env_bias(m)  # keep current to avoid zeroing
            rec_temp[m] = 1.0
            continue

        arr = np.array(vals)
        raw_mean = float(np.mean(arr))
        raw_std  = float(np.std(arr))
        raw_rate = long_rate(vals)

        print(f"\n  {m.upper()}  n={len(vals)}")
        print(f"    raw: mean={raw_mean:.4f}  std={raw_std:.5f}  LONG={raw_rate:.1%}")

        bias = compute_bias(vals)
        rate_after_bias = long_rate(vals, bias)
        print(f"    bias offset = {bias:+.4f}  ->  LONG after debias = {rate_after_bias:.1%}")

        if abs(rate_after_bias - args.target) < 0.03:
            best_t = 1.0
            rate_after_temp = rate_after_bias
            print(f"    debiased rate within 3% of target -- temp=1.0 (no change)")
        else:
            best_t, rate_after_temp = best_temp_after_bias(vals, bias, args.target)
            print(f"    temp={best_t}  ->  LONG after temp = {rate_after_temp:.1%}")

        print(f"    PIPELINE: {raw_rate:.1%} --[bias {bias:+.4f}]--> "
              f"{rate_after_bias:.1%} --[temp {best_t}]--> {rate_after_temp:.1%}")

        if raw_std > 0.05:
            print(f"    temp grid (after debias):")
            for t in TEMP_GRID:
                r = long_rate(vals, bias, t)
                marker = " <--" if t == best_t else ""
                print(f"      t={t:5.1f}  LONG={r:.1%}{marker}")

        rec_bias[m] = bias
        rec_temp[m] = best_t

    # -----------------------------------------------------------------------
    # Stability check
    # -----------------------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  STABILITY CHECK  (windows: {' / '.join(str(w) for w in args.stability_windows)}  "
          f"spread_limit={args.spread_limit})")
    print(f"{'='*64}")
    stable, stab_lines = stability_check(model_vals, args.stability_windows,
                                         args.spread_limit)
    for line in stab_lines:
        print(line)
    if stable:
        print("  => distribution is stable across windows")
    else:
        print("  => UNSTABLE: bias recommendation varies too much across windows")
        print("     Collect more data or tighten --since-datetime before writing.")

    # -----------------------------------------------------------------------
    # Max-step check (always shown, caps if --write-env used without --force)
    # -----------------------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  MAX-STEP CHECK  (max_step={args.max_step})")
    print(f"{'='*64}")
    capped_biases, step_lines, any_capped = maxstep_check(rec_bias, args.max_step)
    for line in step_lines:
        print(line)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*64}")
    print("  RECOMMENDED ENV VARS  (full window)")
    print(f"{'='*64}")
    for m in KNOWN_MODELS:
        vals = model_vals[m]
        if vals:
            final_rate = long_rate(vals, rec_bias[m], rec_temp[m])
            print(f"  DL_BIAS_{m.upper()}={rec_bias[m]:+.4f}   "
                  f"# {long_rate(vals):.1%} -> {final_rate:.1%} LONG")
            if rec_temp[m] != 1.0:
                print(f"  DL_TEMP_{m.upper()}={rec_temp[m]:.1f}")
        else:
            print(f"  DL_BIAS_{m.upper()}={rec_bias[m]:+.4f}   # no data -- kept current")

    if any_capped and not args.force:
        print(f"\n  Capped values (what would be written with --write-env):")
        for m in KNOWN_MODELS:
            if capped_biases[m] != rec_bias.get(m):
                print(f"    DL_BIAS_{m.upper()}={capped_biases[m]:+.4f}  "
                      f"(rec={rec_bias[m]:+.4f})")

    print()
    print("  Apply: add/update in .env, then restart the writer.")
    print("  Auto-patch: run with --write-env")
    if not stable or any_capped:
        if args.force:
            print("  --force is set: safety checks bypassed, raw recommended values used")
        else:
            print("  Use --force to bypass safety gates and write raw recommended values")
    print()

    # -----------------------------------------------------------------------
    # Write .env (if requested, after safety gates)
    # -----------------------------------------------------------------------
    if args.write_env:
        if not stable and not args.force:
            print(f"  --write-env BLOCKED: distribution is UNSTABLE across windows.")
            print(f"  Collect more stable data or use --force to override.")
            return

        write_biases = rec_bias if args.force else capped_biases
        patch_env(write_biases, rec_temp)

        if args.force and not stable:
            print("  WARNING: wrote despite UNSTABLE distribution (--force used)")
        if args.force and any_capped:
            print("  WARNING: wrote uncapped values (--force used)")


if __name__ == "__main__":
    main()
