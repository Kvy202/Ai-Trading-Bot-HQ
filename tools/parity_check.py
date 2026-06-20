#!/usr/bin/env python3
r"""
Parity / live-inference check for fixes #1-#2 (NEEDS network + model artifacts).

Run:
    ~/bot/.venv/bin/python ~/bot/tools/parity_check.py          # Linux/AWS
    .\.venv\Scripts\python.exe tools\parity_check.py            # Windows

It adapts to DL_ADD_SYMBOL_ID, reconciled against the scaler's actual width:
  - DL_ADD_SYMBOL_ID=0  -> 27-feature artifacts, no symbol_id channel (current AWS).
  - DL_ADD_SYMBOL_ID=1  -> 28-feature artifacts with symbol_id (after a retrain).
A contradiction between the env var and the scaler width fails loud.

What it proves:
  A. Each symbol is scored on ITS OWN window - per-symbol windows AND
     predictions are distinct (fix #1: no signal is copied between symbols).
  B. When add_symbol_id is on, every window carries the TRAINED symbol_id
     (fix #2). When off, symbol_id checks are skipped.
  C. Windows are (seq_len, expected_n) where expected_n == the scaler width, and
     scaled features are finite for every model.

Exit code 0 only if all hard checks pass. The loader-parity section is
informational (a forming live candle legitimately differs between two fetches);
exact feature parity is proven deterministically in tools/test_fixes_123.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Load run.json + .env so DL_SYMBOLS / DL_TIMEFRAME / DL_SEQ_LEN match the bot.
try:
    from runtime.loader import apply_run_config
    apply_run_config(BASE)
except Exception:
    pass
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env", override=True)
except Exception:
    pass

_FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


def main() -> int:
    from features import canonical_feature_columns, FEATURE_COLS
    from ml_dl.dl_ensemble import (
        load_ensemble, refresh_live_features_per_symbol, predict_ensemble,
        _load_symbol_id_map, common_scaler_dim, resolve_add_symbol_id,
    )
    from data import load_prices_and_features, executor_symbol

    syms = [s.strip() for s in os.getenv(
        "DL_SYMBOLS", os.getenv("SYMBOL_WHITELIST", "ETHUSDT,SOLUSDT")).split(",") if s.strip()]
    tf = os.getenv("DL_TIMEFRAME", "5m")
    seq = int(os.getenv("DL_SEQ_LEN", "64"))

    print("=" * 62 + "\n  PARITY CHECK - per-symbol inference (DL_ADD_SYMBOL_ID-aware)\n" + "=" * 62)

    models, dev = load_ensemble(X_dim=len(FEATURE_COLS) + 1, device=None)
    print(f"  loaded models: {sorted(models)} on {dev}")

    # Resolve add_symbol_id from DL_ADD_SYMBOL_ID, reconciled with the scaler
    # width. Fails loud if env and artifacts disagree (e.g. env says 28, scaler 27).
    try:
        scaler_dim = common_scaler_dim(models)
        add_sid = resolve_add_symbol_id(scaler_dim)
    except Exception as e:
        print(f"  [FAIL] add_symbol_id/scaler reconciliation - {e}")
        print("\n" + "=" * 62 + "\n  RESULT: FAIL - DL_ADD_SYMBOL_ID does not match the artifacts\n" + "=" * 62)
        return 1

    cols = canonical_feature_columns(add_sid)
    expected_n = len(cols)
    sid_idx = cols.index("symbol_id") if add_sid else None

    print(f"  symbols={syms}  tf={tf}  seq_len={seq}")
    print(f"  DL_ADD_SYMBOL_ID -> add_symbol_id={add_sid}  expected_n={expected_n}  "
          f"scaler_n_features={scaler_dim}  sid_col_idx={sid_idx}")
    check("expected feature count == scaler width", expected_n == scaler_dim,
          f"{expected_n} vs {scaler_dim}")

    id_map = _load_symbol_id_map() if add_sid else {}
    if add_sid:
        print(f"  trained symbol_id map: {id_map}")
        check("symbol_id map is non-empty", bool(id_map))

    # --- Live per-symbol windows ------------------------------------------
    meta, windows = refresh_live_features_per_symbol(
        seq_len=seq, add_symbol_id=add_sid,
        lookback_pad=int(os.getenv("DL_MAX_LOOKBACK_PAD", "6000")),
        symbols=syms, timeframe=tf,
    )
    check("got a window for every requested symbol",
          set(windows) == {executor_symbol(s) for s in syms},
          f"{sorted(windows)}")

    preds = {}
    for sym, win in windows.items():
        print(f"\n  --- {sym} ---")
        check(f"{sym}: window shape == (seq,{expected_n})",
              win.shape == (seq, expected_n), f"{win.shape}")

        # symbol_id checks ONLY when the channel is present.
        if add_sid:
            sid_vals = np.unique(win[:, sid_idx])
            want_id = id_map.get(executor_symbol(sym))
            check(f"{sym}: symbol_id constant across window", sid_vals.size == 1, f"{sid_vals}")
            check(f"{sym}: symbol_id == trained id ({want_id})",
                  want_id is not None and abs(float(sid_vals[0]) - float(want_id)) < 1e-6,
                  f"got {float(sid_vals[0])}")

        # scaled features finite for each model's scaler.
        finite_ok = True
        for _k, pack in models.items():
            scaled = pack["scaler"].transform(win.astype(np.float32))
            finite_ok = finite_ok and bool(np.isfinite(scaled).all())
        check(f"{sym}: scaled features finite (all models)", finite_ok)

        # Score this symbol on its OWN window.
        per_model, agg = predict_ensemble(win, models, dev, None, symbol=sym)
        p_long = float(agg[2]) if agg and len(agg) >= 3 else float("nan")
        preds[sym] = p_long
        print(f"  {sym}: models_used={sorted(per_model)}  p_long={p_long:.6f}")

    # --- distinct predictions (no copy) -----------------------------------
    if len(windows) >= 2:
        keys = list(windows)
        a, b = keys[0], keys[1]
        win_a, win_b = windows[a], windows[b]
        # Exclude the symbol_id channel only when it exists.
        feat_cols = [i for i in range(expected_n) if not (add_sid and i == sid_idx)]
        windows_differ = not np.array_equal(win_a[:, feat_cols], win_b[:, feat_cols])
        check(f"{a} and {b} windows are NOT identical (own data each)", windows_differ)
        pa, pb = preds.get(a, float("nan")), preds.get(b, float("nan"))
        check(f"{a} and {b} predictions are NOT identical (no copy)",
              not (np.isfinite(pa) and np.isfinite(pb) and abs(pa - pb) < 1e-12),
              f"p[{a}]={pa:.6f} p[{b}]={pb:.6f}")
    else:
        print("  (only one symbol configured - skipping distinctness check)")

    # --- loader parity (INFORMATIONAL ONLY) -------------------------------
    # Exact loader-vs-live parity is proven deterministically in
    # tools/test_fixes_123.py. Two independent LIVE fetches can't match on the
    # last (forming) candle, so this section only prints diffs and never fails.
    print("\n  --- loader parity (informational; exact parity is in test_fixes_123.py) ---")
    for sym in list(windows):
        try:
            X_np, _p = load_prices_and_features(
                symbols=[sym], timeframe=tf,
                lookback=seq + int(os.getenv("DL_MAX_LOOKBACK_PAD", "6000")),
                add_symbol_id=add_sid, symbol_id_map=(id_map or None), return_dfs=False,
            )
            direct = np.asarray(X_np[-seq:, :], dtype=np.float32)
        except Exception as e:
            print(f"  [info] {sym}: direct loader failed ({type(e).__name__}: {e})")
            continue
        live = np.asarray(windows[sym], dtype=np.float32)
        if direct.shape != live.shape:
            print(f"  [info] {sym}: shape {direct.shape} vs {live.shape} (timing)")
            continue
        closed_diff = float(np.max(np.abs(direct[:-1] - live[:-1]))) if seq > 1 else float("nan")
        last_diff = float(np.max(np.abs(direct[-1] - live[-1])))
        print(f"  [info] {sym}: max|diff| closed-bars={closed_diff:.4g}  forming-bar={last_diff:.4g}")

    print("\n" + "=" * 62)
    if _FAILS:
        print(f"  RESULT: FAIL - {len(_FAILS)} check(s): {_FAILS}")
        print("=" * 62)
        return 1
    sid_note = "with symbol_id" if add_sid else "no symbol_id (27-feature)"
    print(f"  RESULT: PASS - per-symbol scoring verified ({sid_note})")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"\n[ERROR] parity_check crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)
