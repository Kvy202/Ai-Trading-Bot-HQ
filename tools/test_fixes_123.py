#!/usr/bin/env python3
r"""
Offline checks for fixes #1-#3 (no network / no model forward passes).

Run:
    ~/bot/.venv/bin/python ~/bot/tools/test_fixes_123.py        # Linux/AWS
    .\.venv\Scripts\python.exe tools\test_fixes_123.py          # Windows

Covers:
  1. Feature column order is EXPLICIT and STABLE (canonical_feature_columns).
  2. The persisted symbol_id map loads from model metadata and matches the
     trained `symbols` order (stable symbol semantics).
  3. A feature-dimension mismatch FAILS LOUDLY (no silent pad/truncate).
  4. Paper P&L includes taker fees + slippage (and zero-cost reproduces mid).
  5. The new per-symbol inference API surface exists with the right signatures.

Each check prints PASS/FAIL. Exit code is 0 only if every check passes.
The network-dependent parity test lives in tools/parity_check.py.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

_FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  - {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


def section(title: str) -> None:
    print("\n" + "=" * 62 + f"\n  {title}\n" + "-" * 62)


def main() -> int:
    print("=" * 62 + "\n  OFFLINE CHECKS - fixes #1-#3\n" + "=" * 62)

    # ----- 1. Explicit, stable column order --------------------------------
    section("1. Feature column order is explicit and stable")
    from features import FEATURE_COLS, SYMBOL_ID_COL, canonical_feature_columns

    cols_a = canonical_feature_columns(True)
    cols_b = canonical_feature_columns(True)
    expected = sorted(list(FEATURE_COLS) + [SYMBOL_ID_COL])
    check("canonical order is deterministic", cols_a == cols_b)
    check("canonical order == sorted(FEATURE_COLS + symbol_id)", cols_a == expected,
          f"{len(cols_a)} cols")
    check("includes every FEATURE_COLS entry", set(FEATURE_COLS).issubset(cols_a))
    check("symbol_id present when add_symbol_id=True", SYMBOL_ID_COL in cols_a)
    check("symbol_id absent when add_symbol_id=False",
          SYMBOL_ID_COL not in canonical_feature_columns(False))
    check("dimension is len(FEATURE_COLS)+1", len(cols_a) == len(FEATURE_COLS) + 1,
          f"{len(cols_a)}")

    # ----- 2. Stable symbol_id map -----------------------------------------
    section("2. Persisted symbol_id map matches trained order")
    from ml_dl.dl_ensemble import _load_symbol_id_map, _norm_symbol
    import json

    model_dir = os.getenv("DL_MODEL_DIR", str(BASE / "model_artifacts"))
    # Find a metadata file to derive the expected map independently.
    expected_map = None
    for kind in ("adv", "tx", "lstm", "tcn"):
        p = Path(model_dir) / f"dl_{kind}_metadata.json"
        if p.is_file():
            meta = json.loads(p.read_text())
            syms = meta.get("symbols") or []
            if syms:
                expected_map = {_norm_symbol(s): i for i, s in enumerate(syms)}
                break
    got = _load_symbol_id_map()
    check("metadata with symbols found", expected_map is not None,
          f"{model_dir}")
    if expected_map is not None:
        check("loaded map matches trained symbol order", got == expected_map,
              f"{got}")
        # Sanity: ids are a contiguous 0..N-1 range.
        ids = sorted(got.values())
        check("ids are contiguous 0..N-1", ids == list(range(len(ids))), f"{ids}")

    # ----- 3. Dimension mismatch fails loudly ------------------------------
    section("3. Feature-dim mismatch fails loudly (no silent pad/truncate)")
    import numpy as np
    from ml_dl.dl_ensemble import _align_feat_dim

    ok_dim = _align_feat_dim(np.zeros((8, 28), dtype=np.float32), 28, "tcn")
    check("correct dim passes through unchanged", ok_dim.shape == (8, 28))
    raised = False
    try:
        _align_feat_dim(np.zeros((8, 27), dtype=np.float32), 28, "tcn")
    except RuntimeError:
        raised = True
    check("wrong dim raises RuntimeError (not pad)", raised)
    raised2 = False
    try:
        _align_feat_dim(np.zeros((8, 30), dtype=np.float32), 28, "tcn")
    except RuntimeError:
        raised2 = True
    check("oversized dim raises RuntimeError (not truncate)", raised2)

    # ----- 4. Paper P&L includes fees + slippage ---------------------------
    section("4. Paper P&L nets fees + slippage")
    from tools.live_executor import (
        Position, pnl_on_close, apply_slippage, fee_cost, net_pnl_on_close,
    )

    check("slippage worsens a buy (fills higher)",
          abs(apply_slippage(100.0, "BUY", 2.0) - 100.02) < 1e-9)
    check("slippage worsens a sell (fills lower)",
          abs(apply_slippage(100.0, "SELL", 2.0) - 99.98) < 1e-9)
    check("fee_cost is bps of notional",
          abs(fee_cost(100.0, 5.0) - 0.05) < 1e-12)

    pos = Position(side="long", qty=1.0, avg=100.0)
    gross_mid = pnl_on_close(pos, 101.0)            # 1.0, cost-free
    net, exit_fill = net_pnl_on_close(pos, 101.0, "SELL", fee_bps=5.0, slippage_bps=2.0)
    check("exit fill is adverse vs mid", exit_fill < 101.0, f"{exit_fill:.6f}")
    check("net < gross mid (costs subtracted)", net < gross_mid, f"net={net:.6f} gross={gross_mid:.6f}")
    check("net matches hand-computed value", abs(net - 0.87931010) < 1e-6, f"{net:.8f}")

    net0, fill0 = net_pnl_on_close(pos, 101.0, "SELL", fee_bps=0.0, slippage_bps=0.0)
    check("zero fee+slip reproduces mid pnl", abs(net0 - gross_mid) < 1e-12 and abs(fill0 - 101.0) < 1e-12)

    # short side
    spos = Position(side="short", qty=2.0, avg=50.0)
    snet, sfill = net_pnl_on_close(spos, 49.0, "BUY_TO_COVER", fee_bps=5.0, slippage_bps=2.0)
    check("short close fill is adverse (buys higher)", sfill > 49.0, f"{sfill:.6f}")
    check("short net < gross mid", snet < pnl_on_close(spos, 49.0))

    # ----- 6. Deterministic feature parity (frozen OHLCV) ------------------
    # Prove the LIVE per-symbol path produces the SAME scaled features as the
    # training/offline loader for IDENTICAL input bars. We freeze fetch_ohlcv to
    # synthetic, deterministic data so there is no forming-candle / timing race
    # (which is why a live double-fetch can never match — see parity_check.py).
    section("6. Feature parity: live per-symbol window == loader window (frozen data)")
    import pandas as pd
    import data as _data
    from ml_dl.dl_ensemble import refresh_live_features_per_symbol
    from sklearn.preprocessing import StandardScaler

    def _synth_ohlcv(symbol, timeframe=None, limit=None, n=600):
        rng = np.random.default_rng(abs(hash(symbol)) % (2 ** 32))
        base = 50.0 + (abs(hash(symbol)) % 100)
        close = base * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        openp = np.concatenate([[close[0]], close[:-1]])
        hi = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
        lo = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
        vol = rng.uniform(100, 1000, n)
        idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
        return pd.DataFrame({"open": openp, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)

    _data.fetch_ohlcv = _synth_ohlcv          # freeze the data source
    smap = {"AAAUSDT": 0, "BBBUSDT": 1}
    seq = 32

    # offline / training-style loader window for AAAUSDT
    Xa, _pa = _data.load_prices_and_features(
        symbols=["AAAUSDT"], timeframe="1m", lookback=seq + 400,
        add_symbol_id=True, symbol_id_map=smap, return_dfs=False)
    direct_a = np.asarray(Xa[-seq:, :], dtype=np.float32)

    # live per-symbol path (multi-symbol fetch, then sliced per symbol)
    _meta, wins = refresh_live_features_per_symbol(
        seq_len=seq, add_symbol_id=True, lookback_pad=400,
        symbols=["AAAUSDT", "BBBUSDT"], timeframe="1m", symbol_id_map=smap)
    live_a = np.asarray(wins["AAAUSDT"], dtype=np.float32)

    check("live window shape == loader window shape", direct_a.shape == live_a.shape,
          f"{direct_a.shape} vs {live_a.shape}")
    raw_match = direct_a.shape == live_a.shape and np.allclose(direct_a, live_a, atol=1e-6)
    check("RAW features match exactly (live == loader)", raw_match)

    sid_idx = canonical_feature_columns(True).index(SYMBOL_ID_COL)
    check("AAAUSDT symbol_id == 0 in live window", float(np.unique(live_a[:, sid_idx])[0]) == 0.0)
    check("BBBUSDT symbol_id == 1 in live window",
          float(np.unique(np.asarray(wins["BBBUSDT"])[:, sid_idx])[0]) == 1.0)
    feat_idx = [i for i in range(direct_a.shape[1]) if i != sid_idx]
    check("AAAUSDT and BBBUSDT windows differ (own data each)",
          not np.array_equal(live_a[:, feat_idx], np.asarray(wins["BBBUSDT"])[:, feat_idx]))

    scaler = StandardScaler().fit(direct_a)
    scaled_match = raw_match and np.allclose(scaler.transform(direct_a), scaler.transform(live_a), atol=1e-6)
    check("SCALED features match exactly (live == loader)", scaled_match)

    # ----- 5. Per-symbol API surface ---------------------------------------
    section("5. Per-symbol inference API surface")
    from ml_dl.dl_ensemble import refresh_live_features_per_symbol, predict_ensemble
    from ml_dl.dl_dataset import load_prices_and_features as lpf_wrap
    from data import load_prices_and_features as lpf_data

    check("predict_ensemble accepts symbol kwarg",
          "symbol" in inspect.signature(predict_ensemble).parameters)
    check("refresh_live_features_per_symbol exists", callable(refresh_live_features_per_symbol))
    check("data.load_prices_and_features accepts symbol_id_map",
          "symbol_id_map" in inspect.signature(lpf_data).parameters)
    check("dl_dataset wrapper forwards symbol_id_map",
          "symbol_id_map" in inspect.signature(lpf_wrap).parameters)

    # ----- 7. DL_ADD_SYMBOL_ID reconciliation + 27-feature window path ------
    section("7. DL_ADD_SYMBOL_ID reconciliation (27- and 28-feature artifacts)")
    from ml_dl.dl_ensemble import resolve_add_symbol_id
    base = len(FEATURE_COLS)  # 27

    os.environ["DL_ADD_SYMBOL_ID"] = "1"
    check("env=1 + 28-feature scaler -> add_sid True", resolve_add_symbol_id(base + 1) is True)
    os.environ["DL_ADD_SYMBOL_ID"] = "0"
    check("env=0 + 27-feature scaler -> add_sid False", resolve_add_symbol_id(base) is False)

    os.environ["DL_ADD_SYMBOL_ID"] = "1"          # env wants 28 but scaler is 27
    raised = False
    try:
        resolve_add_symbol_id(base)
    except RuntimeError as e:
        raised = "mismatch" in str(e).lower()
    check("env=1 vs 27-feature scaler raises clear mismatch", raised)

    os.environ["DL_ADD_SYMBOL_ID"] = "0"          # env wants 27 but scaler is 28
    raised = False
    try:
        resolve_add_symbol_id(base + 1)
    except RuntimeError:
        raised = True
    check("env=0 vs 28-feature scaler raises", raised)

    os.environ.pop("DL_ADD_SYMBOL_ID", None)      # unset -> auto-derive from scaler
    check("unset -> auto False for 27-feature scaler", resolve_add_symbol_id(base) is False)
    check("unset -> auto True for 28-feature scaler", resolve_add_symbol_id(base + 1) is True)
    raised = False
    try:
        resolve_add_symbol_id(99)
    except RuntimeError:
        raised = True
    check("nonsense scaler width raises", raised)

    # 27-feature live window path (AWS case), frozen data — reuses the section-6
    # monkeypatched fetch_ohlcv. Proves add_symbol_id=False yields (seq, 27)
    # windows that still differ per symbol — no real 27-feature models needed.
    _m27, wins27 = refresh_live_features_per_symbol(
        seq_len=seq, add_symbol_id=False, lookback_pad=400,
        symbols=["AAAUSDT", "BBBUSDT"], timeframe="1m")
    w27a = np.asarray(wins27["AAAUSDT"])
    check("add_symbol_id=False -> window width == len(FEATURE_COLS)",
          w27a.shape == (seq, base), f"{w27a.shape}")
    check("27-feature windows differ across symbols (no copy)",
          not np.array_equal(w27a, np.asarray(wins27["BBBUSDT"])))

    # ----- summary ---------------------------------------------------------
    print("\n" + "=" * 62)
    if _FAILS:
        print(f"  RESULT: FAIL - {len(_FAILS)} check(s) failed: {_FAILS}")
        print("=" * 62)
        return 1
    print("  RESULT: PASS - all offline checks passed")
    print("  Next: run tools/parity_check.py (needs network + models).")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
