"""
Refined live writer for the AI trading bot.

Loads an ensemble of deep-learning models, refreshes real-time features,
runs predictions, and writes:

  * logs/live_signals.csv          - one row per (tick, symbol). Read by
                                     live_executor.py. Columns and order
                                     match what the executor expects.
  * logs/live_meta_log.csv         - one aggregate row per tick (master log).
  * logs/live_meta_log_YYYYMMDD.csv- same shape, rotated daily.
  * logs/live_writer_heartbeat.json- liveness file written every tick.
  * logs/live_writer.out / .err    - human-readable logs.

Changelog vs the previous version
---------------------------------
- CRITICAL FIX: the per-symbol price fallback no longer substitutes realised
  volatility for missing prices. Missing/invalid prices are now written as
  "0", which the executor already treats as "skip this signal" (`price <= 0`).
  The old behaviour could size live orders against a 0.0001-style number and
  produce catastrophically large positions.
- Per-model prediction extraction now correctly handles BOTH possible shapes
  returned by `predict_ensemble`:
    * tuple/list -> aggregate (ret, rv, p_long)
    * dict       -> per-symbol mapping {symbol: (ret, rv, p_long)}
  The previous `sym_preds` was unreachable code (always fell through to the
  aggregate fallback), so per-symbol signals were silently identical.
- SIGTERM/SIGINT handlers release the lock cleanly. The previous version
  only caught KeyboardInterrupt, so service-managed kills could leak the lock.
- The master CSV column schema is locked once at startup. Previously the
  column list was rebuilt every tick from the current row's keys, which meant
  rows could drift out of alignment with the header if the model set changed.
- Heartbeat is written atomically (write tmp, then rename) so dashboards
  reading concurrently never see a half-written JSON file.
- NaN / inf are sanitised out of every numeric field before being written.
- log_err also writes to stderr now, matching live_executor.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal as signal_module
import sys
import time
import traceback
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths and file constants
# ---------------------------------------------------------------------------
TOOLS_DIR: Path = Path(__file__).resolve().parent
BASE_DIR: Path = TOOLS_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load run.json defaults first, then .env (override=True) so .env always wins.
# Priority: shell env > .env > config/run.json
try:
    from runtime.loader import apply_run_config as _apply_run_config
    _apply_run_config(BASE_DIR)
except Exception:
    pass

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    pass

LOGS = BASE_DIR / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

WR_OUT: Path = LOGS / "live_writer.out"
WR_ERR: Path = LOGS / "live_writer.err"
LOCK: Path = LOGS / "live_writer.lock"
HB_JSON: Path = LOGS / "live_writer_heartbeat.json"
SIGNALS: Path = LOGS / "live_signals.csv"
MASTER_LOG: Path = LOGS / "live_meta_log.csv"
DAILY_PREFIX = "live_meta_log_"
# Per-symbol-per-model probabilities. live_meta_log only stores per-model
# values AVERAGED across symbols, which makes per-symbol ensemble-variant
# analysis (tools/sim_ensemble.py) impossible. This file records each model's
# p_long for each symbol on every tick. Diagnostic only — nothing reads it
# on the trading path.
MODELS_BY_SYMBOL: Path = LOGS / "live_models_by_symbol.csv"

# Stable column schemas. The executor reads the first 8 columns of SIGNALS,
# in this exact order, so don't reorder these without updating the executor.
SIGNAL_COLS: List[str] = [
    "ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode",
    "kinds_used", "side_hint",
]
BASE_COLS: List[str] = ["ts", "p_meta", "thr", "mode", "rv_mean", "allow", "kinds_used"]


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def _append(fp: Path, text: str) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(text)


def log(msg: str) -> None:
    _append(WR_OUT, f"[{_ts()}] {msg}\n")


def log_err(msg: str) -> None:
    line = f"[{_ts()}] {msg}\n"
    _append(WR_ERR, line)
    # Echo errors to stderr so they appear in service logs / terminal too.
    print(line, end="", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Heartbeat handling (atomic write so concurrent readers never see a partial file)
# ---------------------------------------------------------------------------

def write_heartbeat(ok: bool, detail: str, symbols: List[str], allow: int,
                    p_meta: float, thr: float, mode: str, sleep_s: int,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    hb: Dict[str, Any] = {
        "ts": _ts(),
        "ok": bool(ok),
        "detail": str(detail),
        "symbols": list(symbols),
        "p_meta": _safe_float(p_meta, 0.0),
        "thr": _safe_float(thr, 0.0),
        "mode": str(mode),
        "allow": int(allow),
        "sleep_sec": int(sleep_s),
    }
    if extra:
        hb.update(extra)
    try:
        tmp = HB_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(hb, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, HB_JSON)
    except Exception:
        # Heartbeat is informational - never let it crash the loop.
        pass


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_header(path: Path, cols: Iterable[str]) -> None:
    """Write the header if the file is missing or empty."""
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(cols))


def append_aligned_row(path: Path, cols: List[str], row: Dict[str, Any]) -> None:
    """Append `row` aligned to `cols`. Missing keys become empty strings."""
    ensure_header(path, cols)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in cols])


def read_existing_header(path: Path) -> Optional[List[str]]:
    """Return the existing header from a CSV, or None if file is empty/missing."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            first_line = f.readline()
        if not first_line.strip():
            return None
        return next(csv.reader([first_line]))
    except Exception:
        return None


def today_daily(prefix: str) -> Path:
    return LOGS / f"{prefix}{datetime.now(timezone.utc):%Y%m%d}.csv"


# ---------------------------------------------------------------------------
# Locking (cross-platform: works on POSIX and Windows)
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if a process with `pid` is running on this host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        # No such process, or permission denied. Treat as not alive.
        return False
    except Exception:
        # Anything weirder - assume alive to avoid stomping on a real writer.
        return True
    return True


def writer_lock(stale_sec: int) -> None:
    """Acquire a single-writer lock. Exit if another live writer holds it."""
    if LOCK.exists():
        try:
            pid_str, when_str = LOCK.read_text(encoding="utf-8").strip().split(",", 1)
            pid = int(pid_str)
            started = float(when_str)
            if pid != os.getpid():
                alive = _pid_alive(pid)
                fresh = (time.time() - started) < stale_sec
                if alive and fresh:
                    log_err(f"another writer alive (PID={pid}); exiting")
                    sys.exit(0)
                else:
                    log("writer lock: stale or dead lock found; replacing")
        except Exception:
            log("writer lock: unparsable lock; replacing")
    try:
        LOCK.write_text(f"{os.getpid()},{time.time()}", encoding="utf-8")
    except Exception as e:
        log_err(f"writer lock: cannot create lock file: {e}")


def writer_unlock() -> None:
    """Release the lock if we still own it."""
    try:
        if LOCK.exists():
            owner = LOCK.read_text(encoding="utf-8").split(",", 1)[0]
            if int(owner) == os.getpid():
                LOCK.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Safe import of model utilities
# ---------------------------------------------------------------------------
# Imported here (not lazily) because the writer is useless without them.
# Wrapped so the failure shows up in the log file rather than just stderr.
try:
    from ml_dl.dl_ensemble import (  # noqa: F401
        load_ensemble,
        refresh_live_features,
        refresh_live_features_per_symbol,
        predict_ensemble,
    )
except Exception as _e:
    log_err(f"FATAL import: {_e}\n{traceback.format_exc()}")
    raise


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _env_first(names: Iterable[str], default: str) -> str:
    """Return the first non-empty env value in `names`, otherwise `default`."""
    for name in names:
        val = os.getenv(name)
        if val is not None and str(val).strip():
            return str(val).strip()
    return default


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Improved live writer")
    p.add_argument(
        "--symbols",
        default=_env_first(("DL_SYMBOLS", "SYMBOL_WHITELIST"), "BTCUSDT,ETHUSDT"),
    )
    p.add_argument(
        "--timeframe",
        default=_env_first(("DL_TIMEFRAME", "FEATURE_TIMEFRAME", "TIMEFRAME"), "1m"),
    )
    p.add_argument("--seq", type=int, default=int(os.getenv("DL_SEQ_LEN", "128")))
    p.add_argument("--sleep", type=int, default=max(1, int(os.getenv("DL_WRITER_SLEEP", "3"))))
    p.add_argument("--signals", default=str(SIGNALS))
    p.add_argument("--master-log", default=str(MASTER_LOG))
    p.add_argument("--daily-prefix", default=DAILY_PREFIX)
    p.add_argument("--allow-only", default=os.getenv("DL_ALLOW_ONLY", "1"))
    p.add_argument("--thr", type=float, default=float(os.getenv("DL_P_LONG", "0.45")))
    p.add_argument("--mode", default=(os.getenv("DL_P_LONG_MODE", "abs") or "abs").lower(),
                   choices=["abs", "raw"])
    p.add_argument("--lookback-pad", type=int, default=int(os.getenv("DL_MAX_LOOKBACK_PAD", "6000")))
    p.add_argument("--stale-sec", type=int, default=int(os.getenv("DL_WRITER_STALE_SEC", "600")))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Numeric / safety helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert to float, returning `default` for None / non-finite / parse failure."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _is_valid_price(x: Any) -> bool:
    """A price is only valid if it's a finite, strictly positive number."""
    try:
        v = float(x)
        return math.isfinite(v) and v > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Allow gate
# ---------------------------------------------------------------------------

def allow_gate(p_long: float, thr: float, mode: str, allow_only: str) -> int:
    """Return 1 if this prediction passes the allow gate, else 0."""
    if str(allow_only) == "0":
        return 1
    val = abs(p_long) if mode == "abs" else p_long
    return 1 if val >= thr else 0


def side_hint(p: float) -> str:
    if p > 0:
        return "LONG"
    if p < 0:
        return "SHORT"
    return "FLAT"


# ---------------------------------------------------------------------------
# Meta extraction
# ---------------------------------------------------------------------------

def extract_last_prices(meta: Any) -> Dict[str, float]:
    """Pull a {symbol: price} dict out of the model's `meta` blob.

    Tries common key names. Only finite, strictly positive values are kept.
    """
    out: Dict[str, float] = {}
    if not isinstance(meta, dict):
        return out
    for k in ("last_px", "last_price", "px"):
        sub = meta.get(k)
        if isinstance(sub, dict):
            for s, v in sub.items():
                if _is_valid_price(v):
                    out[str(s)] = float(v)
    return out


def extract_symbol_list(meta: Any, fallback: List[str]) -> List[str]:
    if isinstance(meta, dict):
        syms = meta.get("symbols")
        if isinstance(syms, (list, tuple)) and syms:
            return [str(s) for s in syms]
    return list(fallback)


# DL_TEMP_* and DL_BIAS_* calibration is applied inside predict_ensemble()
# in ml_dl/dl_ensemble.py -- do NOT re-apply it here.  All per_model values
# returned by predict_ensemble already have both adjustments baked in.


# ---------------------------------------------------------------------------
# Per-model output handling
# ---------------------------------------------------------------------------
# `predict_ensemble` returns (per_model, aggregate). The shape of each
# per_model entry depends on the model:
#   - aggregate-only models return a 3-tuple (ret_hat, rv_hat, p_long)
#   - per-symbol models return a dict {symbol: (ret_hat, rv_hat, p_long)}
# We support both, and silently skip entries we don't understand.

def _unpack_triple(v: Any) -> Optional[Tuple[float, float, float]]:
    """Try to unpack a value as (ret, rv, p_long). Returns None on failure."""
    if isinstance(v, (tuple, list)) and len(v) >= 3:
        try:
            ret, rv, p = float(v[0]), float(v[1]), float(v[2])
            if not (math.isfinite(ret) and math.isfinite(rv) and math.isfinite(p)):
                return None
            return ret, rv, p
        except Exception:
            return None
    return None


def model_aggregate(vals: Any) -> Optional[Tuple[float, float, float]]:
    """Get aggregate (ret, rv, p) for one model regardless of its output shape."""
    triple = _unpack_triple(vals)
    if triple is not None:
        return triple
    if isinstance(vals, dict):
        # Average across symbols.
        rets: List[float] = []
        rvs: List[float] = []
        ps: List[float] = []
        for sym_v in vals.values():
            t = _unpack_triple(sym_v)
            if t is None:
                continue
            rets.append(t[0])
            rvs.append(t[1])
            ps.append(t[2])
        if ps:
            return (sum(rets) / len(rets), sum(rvs) / len(rvs), sum(ps) / len(ps))
    return None


def model_for_symbol(vals: Any, sym: str) -> Optional[Tuple[float, float, float]]:
    """Get per-symbol (ret, rv, p) for one model.

    If the model returned per-symbol dicts, look up `sym`.
    If the model returned an aggregate tuple, that aggregate is the only
    information available and applies equally to every symbol.
    """
    if isinstance(vals, dict):
        if sym in vals:
            return _unpack_triple(vals[sym])
        return None
    return _unpack_triple(vals)


def aggregate_per_symbol(per_model: Dict[str, Any], sym: str,
                         fallback_p: float, fallback_rv: float) -> Tuple[float, float]:
    """Average per-symbol (p, rv) across all models for one symbol.

    DL_BIAS_* and DL_TEMP_* calibration is already applied inside
    predict_ensemble() before this is called, so per_model values are
    calibrated -- do not re-apply here.

    Falls back to the ensemble aggregate if no model produced a per-symbol
    value for `sym`.
    """
    ps: List[float] = []
    rvs: List[float] = []
    for vals in per_model.values():
        t = model_for_symbol(vals, sym)
        if t is None:
            continue
        _ret, rv, p = t
        ps.append(p)
        rvs.append(rv)
    if ps:
        return sum(ps) / len(ps), sum(rvs) / len(rvs)
    return fallback_p, fallback_rv


# ---------------------------------------------------------------------------
# Master-log column locking
# ---------------------------------------------------------------------------

def lock_master_cols(path: Path, base_cols: List[str], extra_cols: List[str]) -> List[str]:
    """Decide the canonical column list for the master log.

    Priority:
      1. If the file already exists and has a header, use it as-is. This keeps
         old runs and downstream tools compatible.
      2. Otherwise, return base_cols + sorted(extra_cols).

    Once chosen, the column list is locked for the entire run. Any keys not in
    the list are dropped at write time, so rows always align with the header.
    """
    existing = read_existing_header(path)
    if existing:
        return existing
    return list(base_cols) + sorted(c for c in extra_cols if c not in base_cols)


# ---------------------------------------------------------------------------
# Main writer loop
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = build_args()
    writer_lock(args.stale_sec)

    # Install signal handlers AFTER the lock so a fast SIGTERM still releases it.
    def _stop(signum: int, frame: Any) -> None:
        log(f"writer stopped by signal={signum}")
        writer_unlock()
        sys.exit(0)

    signal_module.signal(signal_module.SIGINT, _stop)
    try:
        signal_module.signal(signal_module.SIGTERM, _stop)
    except (AttributeError, ValueError):
        # SIGTERM may not exist on some Windows builds; ignore gracefully.
        pass

    master_log = Path(args.master_log)
    signals_path = Path(args.signals)
    symlist = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

    # NOTE: we do NOT pre-create the master OR daily log here. Calling
    # ensure_header(..., BASE_COLS) writes a 7-column header, and then
    # lock_master_cols() on the first tick reads that header back and locks the
    # schema to those 7 columns — silently dropping EVERY per-model diagnostic
    # column (adv_p, lstm_p, tcn_p, tx_p, ...), which is what tools/diagnose_bias.py
    # and tools/calibrate_temperature.py read. Both files are created with the
    # full locked schema on their first append_aligned_row call instead.
    # (The signals file has a fixed schema, so pre-creating it is fine.)
    ensure_header(signals_path, SIGNAL_COLS)

    # Load the model ensemble.
    try:
        models, dev = load_ensemble(X_dim=30, device=None)
        log("ensemble loaded")
    except Exception as e:
        log_err(f"FATAL load_ensemble: {e}\n{traceback.format_exc()}")
        write_heartbeat(False, f"load_ensemble_failed: {e}", [], 0,
                        0.0, float(args.thr), args.mode, args.sleep)
        writer_unlock()
        sys.exit(1)

    # Resolve whether to append the symbol_id channel from DL_ADD_SYMBOL_ID,
    # reconciled against the scalers' ACTUAL width, and fail loud on a mismatch
    # (_align_feat_dim no longer silently pads/truncates). This supports both the
    # deployed 27-feature artifacts (DL_ADD_SYMBOL_ID=0) and a future 28-feature
    # retrain (DL_ADD_SYMBOL_ID=1) with no code change.
    try:
        from features import FEATURE_COLS as _FEATURE_COLS
        from ml_dl.dl_ensemble import common_scaler_dim, resolve_add_symbol_id
        _scaler_dim = common_scaler_dim(models)
        add_sid = resolve_add_symbol_id(_scaler_dim)
        log(f"scaler dim check OK: all models expect {_scaler_dim} features "
            f"(FEATURE_COLS={len(_FEATURE_COLS)}, add_symbol_id={add_sid})")
    except Exception as e:
        log_err(f"FATAL scaler dim check: {e}\n{traceback.format_exc()}")
        write_heartbeat(False, f"scaler_dim_mismatch: {e}", [], 0,
                        0.0, float(args.thr), args.mode, args.sleep)
        writer_unlock()
        sys.exit(1)

    # Per-symbol-per-model log schema is fixed by the loaded model set. If an
    # existing file has a different header (model set changed between runs),
    # rotate it aside so rows never misalign with the header.
    models_by_sym_cols = ["ts", "symbol", "px"] + [
        f"{k}_p" for k in sorted(models.keys())
    ]
    _existing_mbs = read_existing_header(MODELS_BY_SYMBOL)
    if _existing_mbs is not None and _existing_mbs != models_by_sym_cols:
        rotated = MODELS_BY_SYMBOL.with_name(
            f"live_models_by_symbol_{datetime.now(timezone.utc):%Y%m%d%H%M%S}.csv"
        )
        MODELS_BY_SYMBOL.rename(rotated)
        log(f"models_by_symbol schema changed; rotated old file to {rotated.name}")

    log(f"writer started symbols={symlist} tf={args.timeframe} "
        f"thr={args.thr} mode={args.mode}")

    # Filled on the first successful tick; fixed for the rest of the run.
    locked_master_cols: Optional[List[str]] = None
    feature_guard_block: Optional[bool] = None  # set once; True = refuse to trade
    tick_count = 0

    try:
        while True:
            ts_now = _ts()
            tick_count += 1
            try:
                # 1) Refresh a SEPARATE feature window per symbol and score each
                #    one independently. Previously a single window (the last
                #    symbol's) was scored and its signal copied to every symbol;
                #    now ETH and SOL get genuinely distinct predictions.
                meta, windows = refresh_live_features_per_symbol(
                    seq_len=args.seq,
                    add_symbol_id=add_sid,
                    lookback_pad=args.lookback_pad,
                    symbols=symlist,
                    timeframe=args.timeframe,
                )

                syms = extract_symbol_list(meta, symlist)
                last_px = extract_last_prices(meta)

                # Feature-distribution guard (computed once). Dimension-match is
                # NOT feature-match: a train/serve feature-SET mismatch passes every
                # dim check but feeds the models out-of-distribution inputs and
                # silently saturates them (one-sided collapse). If the live scaled
                # features are wildly off-distribution, refuse to emit tradeable
                # signals (force allow=0) and log loudly. See tools/diagnose_features.py.
                if feature_guard_block is None:
                    from ml_dl.dl_ensemble import feature_ood
                    _n_ood, _max_z = 0, 0.0
                    for _w in windows.values():
                        for _pk in models.values():
                            try:
                                _n, _mz = feature_ood(_w, _pk["scaler"])
                            except Exception:
                                continue
                            _n_ood = max(_n_ood, _n)
                            _max_z = max(_max_z, _mz)
                    feature_guard_block = (
                        _n_ood >= int(os.getenv("DL_OOD_MAX_FEATURES", "3"))
                        or _max_z > float(os.getenv("DL_OOD_MAX_Z", "20"))
                    )
                    if feature_guard_block:
                        log_err(
                            f"FATAL feature-distribution guard: {_n_ood} features "
                            f"off-distribution, max|mean z|={_max_z:.0f}. Likely a "
                            f"train/serve feature mismatch (run tools/diagnose_features.py). "
                            f"Forcing allow=0 — NO TRADES until fixed."
                        )
                    else:
                        log(f"feature guard OK: OOD={_n_ood} max|mean z|={_max_z:.1f}")

                # Score every symbol on its OWN window. predict_ensemble scales
                # internally with each model's paired scaler.
                per_symbol_pred: Dict[str, Tuple[Dict[str, Any], Any]] = {}
                for _sym, _xw in windows.items():
                    per_symbol_pred[_sym] = predict_ensemble(
                        _xw, models, dev, None, symbol=_sym
                    )

                # 2) Master / daily aggregate row = mean across symbols. This is
                #    DIAGNOSTIC ONLY; the executor reads the per-symbol rows below.
                _aggs = [t for t in (_unpack_triple(a)
                                     for (_pm, a) in per_symbol_pred.values())
                         if t is not None]
                if _aggs:
                    ret_hat = sum(t[0] for t in _aggs) / len(_aggs)
                    rv_hat = sum(t[1] for t in _aggs) / len(_aggs)
                    p_long = sum(t[2] for t in _aggs) / len(_aggs)
                else:
                    ret_hat = rv_hat = p_long = 0.0

                _kinds: set = set()
                for _pm, _a in per_symbol_pred.values():
                    _kinds.update(_pm.keys())
                kinds_used = ",".join(sorted(_kinds))

                allow = allow_gate(p_long, args.thr, args.mode, args.allow_only)
                agg_row: Dict[str, Any] = {
                    "ts": ts_now,
                    "p_meta": _safe_float(p_long),
                    "thr": float(args.thr),
                    "mode": args.mode,
                    "rv_mean": _safe_float(rv_hat),
                    "allow": int(allow),
                    "kinds_used": kinds_used,
                }

                # 3) Per-model diagnostics = mean of each model's per-symbol output.
                _model_acc: Dict[str, List[Tuple[float, float, float]]] = {}
                for _pm, _a in per_symbol_pred.values():
                    for _name, _vals in _pm.items():
                        _t = model_aggregate(_vals)
                        if _t is not None:
                            _model_acc.setdefault(_name, []).append(_t)
                for _name, _lst in _model_acc.items():
                    agg_row[f"{_name}_ret"] = _safe_float(sum(x[0] for x in _lst) / len(_lst))
                    agg_row[f"{_name}_rv"] = _safe_float(sum(x[1] for x in _lst) / len(_lst))
                    agg_row[f"{_name}_p"] = _safe_float(sum(x[2] for x in _lst) / len(_lst))

                # 4) Lock the master-log column schema on first tick.
                if locked_master_cols is None:
                    extras = [c for c in agg_row if c not in BASE_COLS]
                    locked_master_cols = lock_master_cols(master_log, BASE_COLS, extras)
                    log(f"master log columns locked: {len(locked_master_cols)} cols")

                append_aligned_row(master_log, locked_master_cols, agg_row)
                append_aligned_row(today_daily(args.daily_prefix),
                                   locked_master_cols, agg_row)

                # 5) Write per-symbol signal rows — each from its OWN prediction.
                missing_price_syms: List[str] = []
                for sym in syms:
                    pred = per_symbol_pred.get(sym)
                    if pred is None:
                        # No window/score for this symbol this tick -> safe no-trade.
                        # p=0.5 centers to 0.0 (FLAT); allow is forced 0 below.
                        per_model_sym: Dict[str, Any] = {}
                        agg_sym: Any = (0.0, 0.0, 0.5)
                    else:
                        per_model_sym, agg_sym = pred

                    _t = _unpack_triple(agg_sym)
                    # Neutral fallback (centers to 0.0/FLAT), never 0.0 (fake SHORT).
                    p_blended = _t[2] if _t is not None else 0.5
                    rv_s = _t[1] if _t is not None else 0.0

                    px = last_px.get(sym)
                    if _is_valid_price(px):
                        px_final = f"{float(px):.8f}"
                    else:
                        # SAFETY: do NOT substitute volatility or any other
                        # number here. Write "0" so the executor's
                        # `price <= 0` guard skips this row cleanly.
                        px_final = "0"
                        missing_price_syms.append(sym)

                    # Center around 0.5 so the signal is signed:
                    #   p_centered > 0 -> bullish -> LONG;  < 0 -> bearish -> SHORT.
                    # Agree-gate suppression arrives as p=0.5 -> centered 0.0 ->
                    # FLAT/allow=0 (it used to arrive as 0.0 -> a fake SHORT).
                    p_centered = _safe_float(p_blended - 0.5)
                    # SAFETY: if no models survived for this symbol, force allow=0
                    # regardless of the aggregate value.
                    if not per_model_sym:
                        sig_allow = 0
                    else:
                        sig_allow = allow_gate(p_centered, args.thr, args.mode, args.allow_only)
                    # Feature-distribution guard: refuse to trade on OOD inputs.
                    if feature_guard_block:
                        sig_allow = 0
                    row = {
                        "ts": ts_now,
                        "symbol": sym,
                        "px": px_final,
                        "p_meta": p_centered,
                        "rv_mean": _safe_float(rv_s),
                        "allow": sig_allow,
                        "thr": float(args.thr),
                        "mode": args.mode,
                        "kinds_used": ",".join(sorted(per_model_sym.keys())),
                        "side_hint": side_hint(p_centered),
                    }
                    append_aligned_row(signals_path, SIGNAL_COLS, row)

                    # Per-symbol-per-model diagnostics (missing models -> blank).
                    mbs_row: Dict[str, Any] = {"ts": ts_now, "symbol": sym, "px": px_final}
                    for _name, _vals in per_model_sym.items():
                        _t3 = model_aggregate(_vals)
                        if _t3 is not None:
                            mbs_row[f"{_name}_p"] = _safe_float(_t3[2])
                    append_aligned_row(MODELS_BY_SYMBOL, models_by_sym_cols, mbs_row)

                    # Phase 2A: shadow influence — log only, never affects signal
                    try:
                        from tier2.influence import shadow_evaluate
                        shadow_evaluate(
                            symbol=sym,
                            side="long" if p_centered > 0 else "short",
                            p_centered=p_centered,
                            base_thr=float(args.thr),
                        )
                    except Exception:
                        pass

                # 6) Tick log + heartbeat.
                if missing_price_syms:
                    log(f"WARN missing price for: {','.join(missing_price_syms)}")
                log(f"tick allow={agg_row['allow']} p_meta={agg_row['p_meta']:.4f} "
                    f"syms={','.join(syms)}")
                write_heartbeat(
                    ok=True,
                    detail="tick_ok" if not missing_price_syms else "tick_missing_prices",
                    symbols=syms,
                    allow=int(agg_row["allow"]),
                    p_meta=float(agg_row["p_meta"]),
                    thr=float(args.thr),
                    mode=args.mode,
                    sleep_s=args.sleep,
                    extra={
                        "tick": tick_count,
                        "missing_price_syms": missing_price_syms,
                    },
                )

            except Exception as e:
                detail = f"ERROR: {e}"
                log_err(f"{detail}\n{traceback.format_exc()}")
                write_heartbeat(
                    ok=False, detail=detail, symbols=[], allow=0,
                    p_meta=0.0, thr=float(args.thr), mode=args.mode,
                    sleep_s=args.sleep, extra={"tick": tick_count},
                )

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        log("writer stopped (ctrl-c)")
    except SystemExit:
        # Re-raise so the signal handler's exit code propagates correctly.
        raise
    except Exception as e:
        log_err(f"FATAL writer crash: {e}\n{traceback.format_exc()}")
    finally:
        writer_unlock()


if __name__ == "__main__":
    main()
