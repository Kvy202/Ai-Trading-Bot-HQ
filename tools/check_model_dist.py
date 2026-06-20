"""
Offline model prediction distribution check.

Loads the current ensemble, pulls recent candles, and slides a seq_len window
over the full history. Reports two views:

  RAW    - predict_next called directly; no degenerate-detection exclusion.
           Shows the true p_long output distribution.
  LIVE   - predict_ensemble called exactly as the live writer does.
           Shows effective predictions and how often each model is excluded.

Quick offline mode (no exchange needed):
  --offline-live-meta   Read per-model columns already logged in
                        logs/live_meta_log_YYYYMMDD.csv instead of fetching
                        candles and running inference. Useful when the exchange
                        is unreachable or you just want a fast bias check.

Run from the project root:
    python tools/check_model_dist.py                          # 5000 windows, live fetch
    python tools/check_model_dist.py --quick                  # 1000 windows, step=20
    python tools/check_model_dist.py --candles 10000
    python tools/check_model_dist.py --offline-live-meta      # no exchange, reads logs
    python tools/check_model_dist.py --offline-live-meta --since 2026-05-08
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import sys
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
_LOGS = _ROOT / "logs"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass

import numpy as np

# Defer heavy imports so --offline-live-meta works without torch/ccxt.
_ENSEMBLE_AVAILABLE = True
try:
    from ml_dl.dl_ensemble import (
        load_ensemble, predict_ensemble,
        _align_feat_dim, _DEGEN_COUNTS, _DEGEN_HISTORY,
    )
    from ml_dl.dl_infer import predict_next
    from data import load_prices_and_features
except ImportError as e:
    _ENSEMBLE_AVAILABLE = False
    _IMPORT_ERROR = str(e)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline model p_long distribution check")
    p.add_argument("--candles", type=int, default=None,
                   help="Number of windows to slide (default 5000, or 1000 in --quick)")
    p.add_argument("--symbols", default=os.getenv("DL_SYMBOLS", "BTCUSDT,ETHUSDT"))
    p.add_argument("--timeframe", default=os.getenv("DL_TIMEFRAME", "1m"))
    p.add_argument("--seq-len", type=int, default=int(os.getenv("DL_SEQ_LEN", "64")))
    p.add_argument("--var-thr", type=float, default=0.002,
                   help="Variance gate threshold used in live ensemble (default 0.002)")
    p.add_argument("--var-window", type=int, default=30,
                   help="Variance gate rolling window (default 30)")
    p.add_argument("--step", type=int, default=None,
                   help="Slide step (default 1; use 5-20 for faster runs)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: candles=1000, step=20 unless explicitly overridden")
    p.add_argument("--timeout", type=int, default=90,
                   help="Max seconds to wait for exchange data fetch (default 90)")
    p.add_argument("--offline-live-meta", action="store_true",
                   help="Skip inference; read per-model data from logs/live_meta_log_*.csv "
                        "(no exchange or torch needed)")
    p.add_argument("--since", default=None,
                   help="For --offline-live-meta: start date YYYY-MM-DD "
                        "(default: 7 days ago)")
    args = p.parse_args()
    # Apply quick-mode defaults for unset args
    if args.quick:
        if args.candles is None:
            args.candles = 1000
        if args.step is None:
            args.step = 20
    else:
        if args.candles is None:
            args.candles = 5000
        if args.step is None:
            args.step = 1
    return args


def _temp(kind: str) -> float:
    try:
        return float(os.getenv(f"DL_TEMP_{kind.upper()}", "1.0"))
    except Exception:
        return 1.0


def _apply_temp(p: float, t: float) -> float:
    if t == 1.0 or not (0.0 < p < 1.0):
        return p
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-logit / t))


def _print_section(title: str) -> None:
    print("\n" + "=" * 64)  # noqa: E501 (intentional width)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# Offline mode: read per-model data from live_meta_log_*.csv
# ---------------------------------------------------------------------------

def _offline_live_meta_report(since: date) -> None:
    """
    Parse live_meta_log_YYYYMMDD.csv files and print per-model LONG-bias stats.
    Uses only the already-logged data -- no exchange, no torch, no inference.
    """
    # Known per-model column suffixes written by live_writer.py
    known_models = ["lstm", "tcn", "tx"]
    # {model: {day: [p_long_values]}}
    per_model_day: Dict[str, Dict[str, List[float]]] = {m: {} for m in known_models}
    rows_total = 0

    log_files = sorted(_LOGS.glob("live_meta_log_2*.csv"))
    if not log_files:
        print("[offline-live-meta] No live_meta_log_YYYYMMDD.csv files found in logs/")
        return

    for path in log_files:
        # Extract date from filename
        stem_date = path.stem.replace("live_meta_log_", "")
        try:
            file_date = datetime.strptime(stem_date, "%Y%m%d").date()
        except ValueError:
            continue
        if file_date < since:
            continue
        day_str = str(file_date)

        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header: Optional[List[str]] = None
            for parts in reader:
                if not parts:
                    continue
                if header is None:
                    header = parts
                    continue
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))
                try:
                    allow = int(row.get("allow", 0))
                except (ValueError, TypeError):
                    allow = 0

                for m in known_models:
                    p_col = f"{m}_p"
                    if p_col not in row or not row[p_col]:
                        continue
                    try:
                        p_val = float(row[p_col])
                    except ValueError:
                        continue
                    if math.isnan(p_val) or math.isinf(p_val):
                        continue
                    per_model_day[m].setdefault(day_str, []).append((p_val, allow))
                rows_total += 1

    if rows_total == 0:
        print("[offline-live-meta] No rows found (files may only have BASE_COLS header "
              "-- restart writer after live_writer.py fix to populate per-model columns)")
        return

    _print_section(f"OFFLINE  per-model LONG bias  (live_meta_log, since {since})")
    print(f"  Files scanned: {len(log_files)}  rows processed: {rows_total}\n")

    all_days = sorted({d for m in per_model_day for d in per_model_day[m]})

    for m in known_models:
        day_data = per_model_day[m]
        if not day_data:
            print(f"  {m.upper():6s}: no data (column '{m}_p' missing from all files)")
            continue

        print(f"  {m.upper()}")
        all_vals_allow = []
        for day in all_days:
            pairs = day_data.get(day, [])
            if not pairs:
                continue
            all_vals = [p for p, _ in pairs]
            allow1 = [p for p, a in pairs if a == 1]
            long_all = sum(1 for p in all_vals if p >= 0.5)
            long_a1  = sum(1 for p in allow1 if p >= 0.5) if allow1 else 0
            all_vals_allow.extend(allow1)
            pct_all = long_all / len(all_vals) if all_vals else 0.0
            pct_a1  = long_a1  / len(allow1)  if allow1  else float("nan")
            flag = "  *** BIAS" if pct_all > 0.85 or (allow1 and pct_a1 > 0.85) else ""
            print(f"    {day}  all={long_all}/{len(all_vals)} ({pct_all:.1%} LONG)"
                  f"  allow1={long_a1}/{len(allow1)} ({pct_a1:.1%}){flag}")

        # Overall for this model
        if all_vals_allow:
            arr = np.array(all_vals_allow)
            long_n = int(np.sum(arr >= 0.5))
            print(f"    TOTAL  allow=1 rows: {len(arr)}  "
                  f"LONG(>=0.5): {long_n} ({long_n/len(arr):.1%})  "
                  f"mean_p={float(np.mean(arr)):.4f}  std={float(np.std(arr)):.5f}")
        print()

    # Per-day cross-model summary
    _print_section("CROSS-MODEL  per-day LONG fraction  (allow=1 only)")
    print(f"  {'DATE':12s} {'LSTM':>8} {'TCN':>8} {'TX':>8}  NOTE")
    for day in all_days:
        fracs = []
        for m in known_models:
            pairs = per_model_day[m].get(day, [])
            allow1 = [p for p, a in pairs if a == 1]
            if allow1:
                fracs.append(sum(1 for p in allow1 if p >= 0.5) / len(allow1))
            else:
                fracs.append(float("nan"))
        notes = []
        for m, f in zip(known_models, fracs):
            if not math.isnan(f) and f > 0.85:
                notes.append(f"{m.upper()} biased")
        frac_strs = [f"{f:.1%}" if not math.isnan(f) else "n/a" for f in fracs]
        print(f"  {day:12s} {frac_strs[0]:>8} {frac_strs[1]:>8} {frac_strs[2]:>8}  "
              f"{'  '.join(notes)}")
    print()


def main() -> None:
    args = parse_args()

    # -- Offline meta-log mode -- no exchange, no torch, no inference ----------
    if args.offline_live_meta:
        if args.since:
            since = datetime.strptime(args.since, "%Y-%m-%d").date()
        else:
            since = date.today() - timedelta(days=6)
        _offline_live_meta_report(since)
        return

    # -- Live-inference mode -- requires ensemble + exchange ------------------
    if not _ENSEMBLE_AVAILABLE:
        print(f"[check_model_dist] import error: {_IMPORT_ERROR}")
        print("Run from the project root with the venv active.")
        print("For a no-exchange check, use: --offline-live-meta")
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    seq_len = args.seq_len
    need = args.candles * args.step + seq_len + 200

    mode_tag = " [QUICK]" if args.quick else ""
    print(f"\n[check_model_dist]{mode_tag} symbols={symbols} tf={args.timeframe} "
          f"seq_len={seq_len} target_windows={args.candles} step={args.step} "
          f"fetch_timeout={args.timeout}s")
    print(f"  variance gate: std < {args.var_thr} over {args.var_window} ticks")

    # --- Fetch features (add_symbol_id=True to match the live writer exactly) ---
    print(f"\n[1/3] Fetching candles + features for {symbols} "
          f"(lookback={need}, timeout={args.timeout}s)...")

    def _fetch() -> object:
        try:
            return load_prices_and_features(
                symbols=symbols,
                timeframe=args.timeframe,
                lookback=need,
                add_symbol_id=True,
                return_dfs=True,
            )
        except TypeError:
            return load_prices_and_features(
                symbols=symbols,
                timeframe=args.timeframe,
                lookback=need,
                add_symbol_id=True,
            )

    X_full = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_fetch)
        print(f"  waiting (up to {args.timeout}s) ...", flush=True)
        try:
            result = fut.result(timeout=args.timeout)
            X_full = result[0] if isinstance(result, (tuple, list)) else result
            print("  fetch complete.")
        except concurrent.futures.TimeoutError:
            print(f"[check_model_dist] exchange fetch timed out after {args.timeout}s")
            print("  Try: --timeout 180  or  --offline-live-meta  for a no-exchange check")
            sys.exit(1)
        except Exception as e:
            print(f"[check_model_dist] feature load failed: {type(e).__name__}: {e}")
            print("  Check that the exchange is reachable, SYMBOL_WHITELIST/DL_SYMBOLS are valid,")
            print("  and the venv has ccxt + all feature dependencies installed.")
            print("  For a no-exchange check, use: --offline-live-meta")
            sys.exit(1)

    if X_full is None:
        print("[check_model_dist] load_prices_and_features returned None")
        sys.exit(1)
    if not hasattr(X_full, "shape"):
        print(f"[check_model_dist] unexpected return type: {type(X_full)}")
        sys.exit(1)

    total_rows = X_full.shape[0]
    n_windows = min(args.candles, max(0, (total_rows - seq_len) // args.step))
    if n_windows <= 0:
        print(f"[check_model_dist] not enough rows: {total_rows} rows, need {seq_len + 1}")
        sys.exit(1)
    print(f"  got {total_rows} rows, {X_full.shape[1]} features -> {n_windows} windows")

    # --- Load models ---
    print("\n[2/3] Loading ensemble...")
    try:
        models, device = load_ensemble(X_dim=X_full.shape[1], device=None)
    except Exception as e:
        print(f"[check_model_dist] load_ensemble failed: {e}")
        sys.exit(1)
    print(f"  loaded: {sorted(models.keys())} on {device}")

    # --- RAW PASS: predict_next directly, no degenerate exclusion ---
    print(f"\n[3/3] RAW pass ({n_windows} windows, no degenerate exclusion)...")
    raw_preds: Dict[str, List[float]] = {k: [] for k in models}
    raw_errors: Dict[str, int] = {k: 0 for k in models}
    first_error_printed: Dict[str, bool] = {k: False for k in models}

    # Rolling variance trackers for our own var-gate accounting
    raw_var_hist: Dict[str, deque] = {k: deque(maxlen=args.var_window) for k in models}
    raw_var_excl: Dict[str, int] = {k: 0 for k in models}

    start_idx = total_rows - n_windows * args.step - seq_len
    for i in range(n_windows):
        idx = start_idx + i * args.step
        xw = X_full[idx: idx + seq_len]

        for kind, m_info in models.items():
            scaler = m_info["scaler"]
            model_obj = m_info["model"]
            target_dim = int(getattr(scaler, "n_features_in_", xw.shape[1]))
            try:
                xw_a = _align_feat_dim(xw, target_dim, kind=kind)
                # Correct arg order: predict_next(window, scaler, model, device)
                ret_hat, rv_hat, p_raw = predict_next(xw_a, scaler, model_obj, device)
                if not math.isfinite(p_raw):
                    raw_errors[kind] += 1
                    continue
                p_scaled = _apply_temp(p_raw, _temp(kind))

                # Track variance gate separately
                vh = raw_var_hist[kind]
                vh.append(p_scaled)
                if len(vh) >= args.var_window and float(np.std(list(vh))) < args.var_thr:
                    raw_var_excl[kind] += 1
                    # still record it  - raw pass doesn't actually exclude
                raw_preds[kind].append(p_scaled)

            except Exception as exc:
                raw_errors[kind] += 1
                if not first_error_printed[kind]:
                    first_error_printed[kind] = True
                    print(f"  [{kind}] first predict_next error: {exc}")

    # --- LIVE PASS: predict_ensemble (matches bot behavior exactly) ---
    print(f"\n     LIVE pass ({n_windows} windows, includes degenerate detection)...")
    _DEGEN_COUNTS.clear()
    _DEGEN_HISTORY.clear()
    live_preds: Dict[str, List[float]] = {k: [] for k in models}
    live_excl: Dict[str, int] = {k: 0 for k in models}
    agg_preds: List[float] = []

    for i in range(n_windows):
        idx = start_idx + i * args.step
        xw = X_full[idx: idx + seq_len]
        try:
            per_model, agg = predict_ensemble(xw, models, device, None)
        except Exception:
            continue

        for kind in models:
            if kind in per_model:
                live_preds[kind].append(float(per_model[kind][2]))
            else:
                live_excl[kind] += 1

        try:
            agg_p = float(agg[2]) if isinstance(agg, (tuple, list)) else float(agg)
            if math.isfinite(agg_p):
                agg_preds.append(agg_p)
        except Exception:
            pass

    # --- Report ---
    _print_section("RAW p_long DISTRIBUTION  (predict_next, no exclusion)")
    for kind in sorted(raw_preds):
        vals = raw_preds[kind]
        errs = raw_errors[kind]
        v_excl = raw_var_excl[kind]
        if not vals:
            print(f"\n  {kind.upper():6s}: 0 predictions  ({errs} errors)")
            continue
        arr = np.array(vals)
        long_n = int(np.sum(arr >= 0.5))
        short_n = len(arr) - long_n
        crossings = int(np.sum((arr[:-1] < 0.5) & (arr[1:] >= 0.5)) +
                        np.sum((arr[:-1] >= 0.5) & (arr[1:] < 0.5)))
        print(f"\n  {kind.upper():6s}  n={len(arr)}  errors={errs}  "
              f"would-be-var-excluded={v_excl} ({v_excl/len(arr):.1%})")
        print(f"    min={arr.min():.4f}  mean={float(np.mean(arr)):.4f}  "
              f"median={float(np.median(arr)):.4f}  max={arr.max():.4f}  "
              f"std={float(np.std(arr)):.5f}")
        print(f"    LONG(>=0.5): {long_n} ({long_n/len(arr):.1%})   "
              f"SHORT(<0.5): {short_n} ({short_n/len(arr):.1%})")
        if crossings == 0:
            print(f"    *** NEVER crosses 0.5  - stuck on one side entirely ***")
        else:
            print(f"    0.5 crossings: {crossings}  "
                  f"({'active' if crossings > n_windows * 0.02 else 'rarely crosses'})")
        if arr.max() < 0.5:
            print(f"    *** max={arr.max():.4f}: model NEVER predicts LONG ***")
        elif arr.min() >= 0.5:
            print(f"    *** min={arr.min():.4f}: model NEVER predicts SHORT ***")

    _print_section("LIVE p_long DISTRIBUTION  (predict_ensemble, degenerate detection active)")
    for kind in sorted(live_preds):
        vals = live_preds[kind]
        excl = live_excl[kind]
        total = len(vals) + excl
        if not vals:
            print(f"\n  {kind.upper():6s}: 0 predictions  - all {excl} windows excluded")
            continue
        arr = np.array(vals)
        long_n = int(np.sum(arr >= 0.5))
        short_n = len(arr) - long_n
        print(f"\n  {kind.upper():6s}  survived={len(arr)} ({len(arr)/total:.1%})  "
              f"excluded={excl} ({excl/total:.1%})")
        print(f"    min={arr.min():.4f}  mean={float(np.mean(arr)):.4f}  "
              f"median={float(np.median(arr)):.4f}  max={arr.max():.4f}  "
              f"std={float(np.std(arr)):.5f}")
        print(f"    LONG(>=0.5): {long_n} ({long_n/len(arr):.1%})   "
              f"SHORT(<0.5): {short_n} ({short_n/len(arr):.1%})")
        if arr.max() < 0.5:
            print(f"    *** max={arr.max():.4f}: surviving predictions NEVER go LONG ***")

    _print_section("ENSEMBLE AGGREGATE  (live pass)")
    if agg_preds:
        agg = np.array(agg_preds)
        long_n = int(np.sum(agg >= 0.5))
        short_n = len(agg) - long_n
        print(f"\n  n={len(agg)}")
        print(f"  min={agg.min():.4f}  mean={float(np.mean(agg)):.4f}  "
              f"median={float(np.median(agg)):.4f}  max={agg.max():.4f}  "
              f"std={float(np.std(agg)):.5f}")
        print(f"  LONG(>=0.5): {long_n} ({long_n/len(agg):.1%})   "
              f"SHORT(<0.5): {short_n} ({short_n/len(agg):.1%})")
        if long_n == 0:
            print("  *** ENSEMBLE NEVER goes LONG over this window ***")
        elif short_n == 0:
            print("  *** ENSEMBLE NEVER goes SHORT over this window ***")
    else:
        print("  no aggregate predictions")

    _print_section("VARIANCE GATE SUMMARY  (raw pass  - how often std < threshold)")
    for kind in sorted(raw_var_excl):
        total = len(raw_preds.get(kind, [])) + raw_errors.get(kind, 0)
        excl = raw_var_excl[kind]
        pct = excl / total if total else 0.0
        flag = "  *** HIGH  - gate firing frequently" if pct > 0.20 else ""
        print(f"  {kind.upper():6s}: {excl}/{total} windows ({pct:.1%}) below std={args.var_thr}{flag}")

    print()


if __name__ == "__main__":
    main()
