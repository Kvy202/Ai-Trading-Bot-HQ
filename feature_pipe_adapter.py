# feature_pipe_adapter.py — L2 pluggable (placeholder | bitget_rest | csv)
from __future__ import annotations
import os, time
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from atb.data.sources.klines_ccxt import fetch_klines
from atb.features.pipeline import build_leakage_safe_features

# ---- env / knobs ----
EXCHANGE = (os.getenv("EXCHANGE_ID", "bitget") or "bitget").lower()
LIMIT    = int(os.getenv("FEATURE_KLINE_LIMIT", "1000"))

# L2 source: 'placeholder' | 'bitget_rest' | 'csv'
L2_SOURCE   = (os.getenv("L2_SOURCE", "placeholder") or "placeholder").lower()
L2_CSV_DIR  = os.getenv("L2_CSV_DIR", "data/l2/bitget_rest")
L2_LEVELS   = int(os.getenv("L2_REST_LEVELS", "5"))  # depth for REST snapshot

# gates
DEFAULT_SPREAD_GUARD = float(os.getenv("SPREAD_GUARD_BPS", "10"))
VOLZ_MIN = float(os.getenv("VOLZ_MIN", "-0.5"))
RSI_MIN  = float(os.getenv("RSI_MIN", "20.0"))
RSI_MAX  = float(os.getenv("RSI_MAX", "80.0"))

# Optional per-symbol spread overrides
SYMBOL_SPREAD_OVERRIDES: Dict[str, float] = {
    # "DOGE/USDT:USDT": 6.0,
    # "XRP/USDT:USDT": 6.0,
    # "PEPE/USDT:USDT": 20.0,
    # "1000BONK/USDT:USDT": 22.0,
}

# ---- helpers ----
_OB_COLS = [
    "ts","symbol","bid_px","bid_sz","ask_px","ask_sz",
    "mid_px","spread","spread_bps","bid_sz_sum_k","ask_sz_sum_k"
]

def _empty_ob() -> pd.DataFrame:
    return pd.DataFrame(columns=_OB_COLS)

def _as_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _placeholder_ob(kl: pd.DataFrame) -> pd.DataFrame:
    """Synthetic 10 bps top-of-book from 1m klines for the entire history window."""
    if kl is None or kl.empty:
        return _empty_ob()
    one_m = kl[kl["timeframe"] == "1m"].copy()
    if one_m.empty:
        return _empty_ob()

    ob = one_m[["ts","symbol","close","volume"]].copy()
    ob["bid_px"] = ob["close"] * 0.9995
    ob["ask_px"] = ob["close"] * 1.0005
    ob["bid_sz"] = ob["volume"].fillna(0).clip(lower=1)
    ob["ask_sz"] = ob["volume"].fillna(0).clip(lower=1)
    ob = ob.rename(columns={"close": "mid_px"})
    ob["spread"] = ob["ask_px"] - ob["bid_px"]
    denom = ((ob["ask_px"] + ob["bid_px"]) / 2.0).replace(0, np.nan)
    ob["spread_bps"] = (ob["spread"] / denom) * 1e4
    ob["bid_sz_sum_k"] = ob["bid_sz"]
    ob["ask_sz_sum_k"] = ob["ask_sz"]
    return ob[_OB_COLS]

def _l2_from_rest_now(symbols: List[str], ts_align_map: Dict[str, int]) -> pd.DataFrame:
    """Fetch live top-of-book via ccxt; align each row's ts to the latest 1m kline ts per symbol."""
    try:
        import ccxt
    except Exception:
        return _empty_ob()

    ex = getattr(ccxt, EXCHANGE)({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType":"swap","defaultSubType":"linear","defaultSettle":"USDT"},
    })
    try:
        ex.load_markets()
    except Exception:
        pass

    rows = []
    for s in symbols:
        try:
            ob = ex.fetch_order_book(s, limit=L2_LEVELS)
            bids = ob.get("bids", [])[:L2_LEVELS]
            asks = ob.get("asks", [])[:L2_LEVELS]
            if not bids or not asks:
                continue
            bid_px, bid_sz = float(bids[0][0]), float(bids[0][1])
            ask_px, ask_sz = float(asks[0][0]), float(asks[0][1])
            bid_sum = sum(float(x[1]) for x in bids)
            ask_sum = sum(float(x[1]) for x in asks)
            mid = (bid_px + ask_px)/2.0
            spread = ask_px - bid_px
            spread_bps = (spread / mid) * 1e4 if mid > 0 else np.nan
            ts_last = ts_align_map.get(s)
            if ts_last is None:
                # align to current minute if we can't find kline ts
                ts_last = int(pd.Timestamp.utcnow().floor("min").timestamp() * 1000)
            rows.append([ts_last, s, bid_px, bid_sz, ask_px, ask_sz, mid, spread, spread_bps, bid_sum, ask_sum])
        except Exception:
            continue
    try:
        ex.close()
    except Exception:
        pass

    ob = pd.DataFrame(rows, columns=_OB_COLS)
    return ob

def _l2_from_csv(symbols: List[str], kl: pd.DataFrame) -> pd.DataFrame:
    """Load captured L2 CSVs and resample to 1m last; clip to klines time range."""
    chunks = []
    for s in symbols:
        path = os.path.join(L2_CSV_DIR, f"{s.replace('/','_').replace(':','_')}.csv")
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_csv(path)
            if raw.empty: 
                continue
            # expected columns: ts,symbol,bid_px,bid_sz,ask_px,ask_sz,mid_px,spread,spread_bps,bid_sz_sum_k,ask_sz_sum_k
            raw["ts_dt"] = pd.to_datetime(raw["ts"], unit="ms", utc=True)
            one = (raw.set_index("ts_dt")
                       .resample("1min")
                       .last()
                       .dropna()
                       .reset_index())
            one["ts"] = (one["ts_dt"].astype("int64") // 10**6).astype("int64")
            one["symbol"] = s
            # clip to kl time range for that symbol
            k = kl[kl["symbol"] == s]
            if k.empty:
                continue
            tmin, tmax = int(k["ts"].min()), int(k["ts"].max())
            one = one[(one["ts"] >= tmin) & (one["ts"] <= tmax)]
            # ensure required columns exist
            for c in _OB_COLS:
                if c not in one.columns:
                    one[c] = np.nan
            chunks.append(one[_OB_COLS])
        except Exception:
            continue
    if not chunks:
        return _empty_ob()
    return pd.concat(chunks, ignore_index=True)

def _merge_ob(base_ob: pd.DataFrame, overlay_ob: pd.DataFrame) -> pd.DataFrame:
    """Update base_ob rows with overlay_ob values on (symbol, ts)."""
    if base_ob.empty:
        return overlay_ob[_OB_COLS].copy()
    if overlay_ob.empty:
        return base_ob[_OB_COLS].copy()
    left = base_ob.set_index(["symbol","ts"])
    right = overlay_ob.set_index(["symbol","ts"])
    left.update(right)
    return left.reset_index()[_OB_COLS]

def get_latest_features(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Returns {symbol: {ts, close, ema_20/50/100, rsi, rv, vol_z, spread_bps, imbalance,
                      pass_spread_guard, pass_vol_gate, pass_rsi_gate}}
    """
    # Always fetch 1m klines (pipeline is 1m)
    kl = fetch_klines(EXCHANGE, symbols, timeframe="1m", limit=LIMIT, market_type="swap")
    if kl is None or kl.empty:
        return {}

    # Build placeholder OB across history
    ob = _placeholder_ob(kl)

    # Optionally overlay with real L2
    if L2_SOURCE == "csv":
        ob_csv = _l2_from_csv(symbols, kl)
        if not ob_csv.empty:
            ob = _merge_ob(ob, ob_csv)
    elif L2_SOURCE == "bitget_rest":
        # Align live snapshot to the last 1m bar per symbol
        ts_align = {}
        for s in symbols:
            ks = kl[(kl["timeframe"]=="1m") & (kl["symbol"]==s)]
            if not ks.empty:
                ts_align[s] = int(ks["ts"].max())
        ob_now = _l2_from_rest_now(symbols, ts_align)
        if not ob_now.empty:
            ob = _merge_ob(ob, ob_now)

    # Build features
    if ob is None or ob.empty:
        return {}
    feat = build_leakage_safe_features(kl, ob)
    if feat is None or feat.empty:
        return {}

    # pick latest per symbol
    latest = (
        feat.sort_values(["symbol","ts"])
            .groupby("symbol", as_index=False)
            .tail(1)
            .set_index("symbol")
    )

    out: Dict[str, Dict[str, Any]] = {}
    for sym, row in latest.iterrows():
        close = _as_float(row.get("close"))
        rsi   = _as_float(row.get("rsi"))
        rv    = _as_float(row.get("rv"))
        vol_z = _as_float(row.get("vol_z"))
        spbps = _as_float(row.get("spread_bps"))
        imb   = _as_float(row.get("imbalance"))

        # guards
        sp_th = SYMBOL_SPREAD_OVERRIDES.get(sym, DEFAULT_SPREAD_GUARD)
        pass_spread = (not np.isnan(spbps)) and (spbps <= sp_th)
        pass_vol    = (not np.isnan(vol_z)) and (vol_z >= VOLZ_MIN)
        pass_rsi    = (not np.isnan(rsi))   and (RSI_MIN <= rsi <= RSI_MAX)

        try:
            ts_val = int(_as_float(row.get("ts"), np.nan))
        except Exception:
            ts_val = int(pd.Timestamp.utcnow().value // 10**6)

        out[sym] = {
            "ts": ts_val,
            "close": close,
            "ema_20": _as_float(row.get("ema_20")),
            "ema_50": _as_float(row.get("ema_50")),
            "ema_100": _as_float(row.get("ema_100")),
            "rsi": rsi,
            "rv": rv,
            "vol_z": vol_z,
            "spread_bps": spbps,
            "imbalance": imb,
            "pass_spread_guard": bool(pass_spread),
            "pass_vol_gate": bool(pass_vol),
            "pass_rsi_gate": bool(pass_rsi),
        }
    return out
