# data.py
"""
Data loading and feature-building helpers for the AI trading bot.

This module is used by:
  - model/backtest code
  - ml_dl.dl_dataset.load_prices_and_features
  - ml_dl.dl_ensemble.refresh_live_features
  - tools/live_writer.py

Important live-writer contract:
  load_prices_and_features(..., return_dfs=True) must return:

      X_np, dfs_by_symbol

  where:
      X_np           = numpy feature matrix
      dfs_by_symbol  = {symbol: DataFrame}, and each DataFrame contains a
                       "close" column.

The previous version returned (X_cat, p_cat) when return_dfs=True. That broke
dl_ensemble.py price extraction, caused live_writer.py to write px=0, and made
live_executor.py skip every row as bad_price.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ccxt
import numpy as np
import pandas as pd

from config import EXCHANGE_ID, TIMEFRAME, LOOKBACK_CANDLES

_LOG = logging.getLogger("data")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_str(name, str(default))))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    val = _env_str(name, "1" if default else "0").lower()
    return val in {"1", "true", "yes", "y", "on"}


def _quiet_data() -> bool:
    return _env_bool("QUIET_DATA", False)


def _log_data(msg: str) -> None:
    if not _quiet_data():
        print(msg)
    _LOG.debug(msg)


def _derivs_enabled() -> bool:
    return _env_bool("DERIVS", True)


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: str, kind: str = "swap") -> str:
    """Normalize common user formats into CCXT unified symbols.

    Examples for swap:
      BTCUSDT       -> BTC/USDT:USDT
      BTCUSDT:USDT  -> BTC/USDT:USDT
      BTC/USDT      -> BTC/USDT:USDT
      BTC/USDT:USDT -> BTC/USDT:USDT

    Examples for spot:
      BTCUSDT       -> BTC/USDT
      BTC/USDT:USDT -> BTC/USDT
    """
    s = str(symbol or "").strip()
    if not s:
        return s

    if "/" in s and ":" in s:
        return s if kind == "swap" else s.split(":", 1)[0]

    if "/" in s:
        return f"{s}:USDT" if kind == "swap" and not s.endswith(":USDT") else s

    if ":" in s:
        s = s.split(":", 1)[0]

    if s.upper().endswith("USDT") and len(s) > 4:
        base = s[:-4]
        quote = "USDT"
        return f"{base}/{quote}:USDT" if kind == "swap" else f"{base}/{quote}"

    return s


def executor_symbol(symbol: str) -> str:
    """Convert a CCXT symbol back to executor/writer compact format."""
    s = str(symbol or "").strip()
    if ":" in s:
        s = s.split(":", 1)[0]
    return s.replace("/", "")


def _symbol_candidates(symbol: str, kind: str) -> List[str]:
    """Candidate symbols to try against loaded ccxt markets."""
    norm = normalize_symbol(symbol, kind)
    compact = executor_symbol(norm)
    spot = norm.split(":", 1)[0] if ":" in norm else norm

    candidates = [norm]
    if kind == "swap" and spot != norm:
        candidates.append(spot)
    if compact and compact not in candidates:
        candidates.append(compact)

    out: List[str] = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _resolve_market_symbol(ex: ccxt.Exchange, symbol: str, kind: str) -> str:
    """Resolve user symbol to an actual loaded ccxt market symbol."""
    markets = getattr(ex, "markets", None) or {}
    symbols = getattr(ex, "symbols", None) or []

    for cand in _symbol_candidates(symbol, kind):
        if cand in markets or cand in symbols:
            return cand

    wanted = executor_symbol(symbol).upper()
    for market_symbol in symbols:
        if executor_symbol(market_symbol).upper() == wanted:
            return market_symbol

    return normalize_symbol(symbol, kind)


# ---------------------------------------------------------------------------
# Exchange helpers
# ---------------------------------------------------------------------------

def get_exchange(exchange_id: Optional[str] = None, kind: Optional[str] = None) -> ccxt.Exchange:
    """Create a configured CCXT exchange client."""
    ex_id = (exchange_id or EXCHANGE_ID or "bitget").lower()
    use_kind = kind or ("swap" if _derivs_enabled() else "spot")

    cls = getattr(ccxt, ex_id, None)
    if cls is None:
        raise ValueError(f"Unknown exchange id: {ex_id!r}")

    options: Dict[str, Any] = {"defaultType": use_kind}

    if ex_id in {"bitget", "bybit", "mexc"} and use_kind == "swap":
        options.update({
            "defaultSubType": "linear",
            "defaultSettle": "USDT",
        })

    ex = cls({
        "enableRateLimit": True,
        "timeout": _env_int("CCXT_TIMEOUT_MS", 120000),
        "options": options,
    })

    ex.verbose = _env_bool("CCXT_DEBUG", False)

    if _env_bool("BITGET_SANDBOX", False) or _env_bool("CCXT_SANDBOX", False):
        try:
            ex.set_sandbox_mode(True)
        except Exception as e:
            _LOG.debug("set_sandbox_mode failed for %s: %s", ex_id, e)

    return ex


def _market_params(ex: ccxt.Exchange, kind: str) -> Dict[str, Any]:
    if ex.id == "bitget":
        return {"productType": "USDT-FUTURES"} if kind == "swap" else {"type": "spot"}
    if ex.id == "mexc" and kind == "spot":
        return {"type": "spot"}
    return {}


def _ensure_markets(ex: ccxt.Exchange, kind: str) -> None:
    params = _market_params(ex, kind)
    try:
        ex.load_markets(False, params)
    except Exception:
        ex.load_markets()


def _close_exchange(ex: ccxt.Exchange) -> None:
    try:
        ex.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    timeframe: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch paginated OHLCV and return a clean DataFrame."""
    tf = (timeframe or TIMEFRAME or "1m").lower()
    want = int(limit or LOOKBACK_CANDLES or 1000)
    if want <= 0:
        raise ValueError(f"limit/lookback must be > 0, got {want}")

    raw_symbol = str(symbol or "").strip()
    looks_contract = (":USDT" in raw_symbol) or ("-SWAP" in raw_symbol)
    kind = "swap" if (_derivs_enabled() or looks_contract) else "spot"

    ex = get_exchange(kind=kind)
    try:
        _ensure_markets(ex, kind)
        market_symbol = _resolve_market_symbol(ex, raw_symbol, kind)

        per = 1000
        if ex.id == "mexc" and kind == "spot":
            per = 500

        ms_per_bar = int(ex.parse_timeframe(tf) * 1000)
        end_ms = ex.milliseconds()
        since = end_ms - (want + 200) * ms_per_bar

        params = _market_params(ex, kind)
        rows: List[List[float]] = []
        retries = 0
        pages = 0
        max_pages = max(5, min(2000, int(np.ceil((want + 200) / per)) + 5))
        last_seen_ts: Optional[int] = None

        while len(rows) < want and pages < max_pages:
            try:
                part = ex.fetch_ohlcv(
                    market_symbol,
                    timeframe=tf,
                    since=since,
                    limit=per,
                    params=params,
                )
            except Exception:
                retries += 1
                if retries > _env_int("CCXT_RETRIES", 5):
                    raise
                time.sleep(min(2 ** retries, 8))
                continue

            retries = 0

            if not part:
                break

            if last_seen_ts is not None and part[-1][0] <= last_seen_ts:
                break

            rows.extend(part)
            pages += 1
            last_seen_ts = int(part[-1][0])
            since = last_seen_ts + ms_per_bar

            if since >= end_ms - ms_per_bar:
                break

            time.sleep((getattr(ex, "rateLimit", None) or 250) / 1000.0)

        if len(rows) < min(want, 64):
            try:
                fallback = ex.fetch_ohlcv(
                    market_symbol,
                    timeframe=tf,
                    limit=min(1000, want),
                    params=params,
                )
                if fallback:
                    rows = fallback if not rows else rows + fallback
            except Exception as e:
                _LOG.debug("fallback OHLCV failed for %s: %s", market_symbol, e)

        if not rows:
            raise RuntimeError(f"No OHLCV returned for {raw_symbol} -> {market_symbol} {tf} on {ex.id} ({kind})")

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            raise RuntimeError(f"OHLCV became empty after cleaning for {raw_symbol} {tf}")

        return df.tail(want)

    finally:
        _close_exchange(ex)


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def _build_one_symbol(
    sym: str,
    tf: str,
    lb: int,
    feature_cols: Optional[Sequence[str]],
) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """Build features and aligned close prices for one symbol.

    Uses build_features (no forward-looking labels) so the latest closed
    candle is included. Previously used make_dataset, which dropped the
    last ~max_h rows for triple-barrier label computation -- fine for
    training but wrong for live inference.
    """
    from features import build_features, FEATURE_COLS

    try:
        bars = fetch_ohlcv(sym, tf, lb)
    except Exception as e:
        _log_data(f"[load_prices_and_features] skipping {sym}: {type(e).__name__}: {e}")
        return None

    try:
        feats = build_features(bars)
    except Exception as e:
        _log_data(f"[load_prices_and_features] {sym}: build_features failed: {type(e).__name__}: {e}")
        return None

    if feats is None or len(feats) == 0:
        _log_data(f"[load_prices_and_features] {sym}: empty feature set")
        return None

    # Use the canonical training feature list unless the caller explicitly
    # asked for a different subset. This is the critical line: it guarantees
    # live X_df has the SAME columns in the SAME order as training.
    cols_wanted = list(feature_cols) if feature_cols else list(FEATURE_COLS)

    missing = [c for c in cols_wanted if c not in feats.columns]
    keep = [c for c in cols_wanted if c in feats.columns]
    if missing:
        _log_data(
            f"[load_prices_and_features] {sym}: missing features "
            f"{missing[:6]}{'...' if len(missing) > 6 else ''}"
        )
    if not keep:
        return None

    X_df = feats[keep].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan).dropna()
    if X_df.empty:
        _log_data(f"[load_prices_and_features] {sym}: features all NaN/inf after cleaning")
        return None

    common_idx = X_df.index.intersection(bars.index)
    X_df = X_df.loc[common_idx]
    prices = bars.loc[common_idx, "close"].astype("float32")

    if len(X_df) == 0 or len(prices) == 0:
        _log_data(f"[load_prices_and_features] {sym}: no overlap between features and prices")
        return None

    live_df = X_df.copy()
    live_df["close"] = prices

    return X_df, prices, live_df
def _parse_symbols(symbols: Optional[Sequence[str]]) -> List[str]:
    if symbols:
        return [str(s).strip() for s in symbols if str(s).strip()]

    wl = _env_str("SYMBOL_WHITELIST", "")
    if wl:
        return [s.strip() for s in wl.split(",") if s.strip()]

    sym_env = _env_str("SYMBOL", "BTCUSDT")
    return [sym_env]

def load_prices_and_features(
    symbols: Optional[Sequence[str]] = None,
    timeframe: Optional[str] = None,
    lookback: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    add_symbol_id: bool = False,
    return_dfs: bool = False,
    return_symbol_lengths: bool = False,
    symbol_id_map: Optional[Dict[str, int]] = None,
):
    """Build a feature matrix across one or many symbols.

    If return_dfs=False:
        returns (X_np, p_np)

    If return_dfs=True:
        returns (X_np, dfs_by_symbol)

        dfs_by_symbol is:
            {
                "BTCUSDT": DataFrame(..., "close"),
                "ETHUSDT": DataFrame(..., "close"),
            }

        This return shape is required by dl_ensemble.refresh_live_features().
    """
    tf = timeframe or TIMEFRAME or "1m"
    lb = int(lookback or LOOKBACK_CANDLES or 1000)
    syms = _parse_symbols(symbols)

    if lb <= 0:
        raise ValueError(f"lookback must be > 0, got {lb}")
    if not syms:
        raise RuntimeError("No symbols configured for load_prices_and_features")

    built: List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]] = []

    for si, sym in enumerate(syms):
        result = _build_one_symbol(sym, tf, lb, feature_cols)
        if result is None:
            continue

        X_df, prices, live_df = result

        if add_symbol_id:
            X_df = X_df.copy()
            live_df = live_df.copy()
            if symbol_id_map is not None:
                # SERVING (and any caller that supplies a persisted map): the
                # symbol_id MUST match what each model learned during training,
                # otherwise the scaler/model receive an out-of-distribution value
                # on this channel. Position-based ids (the else branch) are only
                # correct when the symbol list is identical to training.
                key = executor_symbol(sym)
                if key not in symbol_id_map:
                    raise RuntimeError(
                        f"symbol_id_map has no id for {key!r}; trained symbols="
                        f"{sorted(symbol_id_map)}. Refusing to serve a symbol the "
                        f"models never saw (would feed a wrong symbol_id)."
                    )
                sid = float(symbol_id_map[key])
            else:
                # TRAINING default: id = position in the requested symbol list.
                sid = float(si)
            X_df["symbol_id"] = sid
            live_df["symbol_id"] = sid

        out_sym = executor_symbol(sym)
        built.append((X_df, prices, live_df, out_sym))

    if not built:
        raise RuntimeError("No features could be built for any symbol.")

    # Explicit, stable column order (single source of truth in features.py).
    # The OLD code used `cols = sorted(set-intersection-of-columns)`, which
    # silently DROPPED a column for ALL symbols if any one symbol happened to
    # lack it — a quiet feature-drift / dim-mismatch vector. We now compute the
    # expected canonical order up front and FAIL LOUDLY if any symbol is missing
    # a column, instead of silently reshaping. The resulting order is identical
    # to the old sorted() order, so deployed artifacts stay valid.
    from features import canonical_feature_columns
    if feature_cols:
        cols = sorted(list(feature_cols) + (["symbol_id"] if add_symbol_id else []))
    else:
        cols = canonical_feature_columns(add_symbol_id)
    if not cols:
        raise RuntimeError("No feature columns to assemble.")
    for X_df, _, _, out_sym in built:
        missing_cols = [c for c in cols if c not in X_df.columns]
        if missing_cols:
            raise RuntimeError(
                f"load_prices_and_features: symbol {out_sym!r} is missing feature "
                f"columns {missing_cols}. Train and serve must expose an identical "
                f"feature set; refusing to silently drop/pad columns."
            )

    X_parts: List[pd.DataFrame] = []
    p_parts: List[pd.Series] = []
    sym_lengths_list: List[int] = []
    dfs_by_symbol: Dict[str, pd.DataFrame] = {}

    for X_df, prices, live_df, out_sym in built:
        Xc = X_df[cols].astype("float32")
        pc = prices.astype("float32")

        idx = Xc.index.intersection(pc.index)
        Xc = Xc.loc[idx]
        pc = pc.loc[idx]

        if Xc.empty:
            continue

        X_parts.append(Xc)
        p_parts.append(pc)
        sym_lengths_list.append(len(Xc))

        live_cols = [c for c in cols if c in live_df.columns]
        keep_df = live_df[live_cols + ["close"]].copy()
        keep_df = keep_df.loc[keep_df.index.intersection(idx)]
        dfs_by_symbol[out_sym] = keep_df

    if not X_parts:
        raise RuntimeError("No aligned feature rows after column/index alignment.")

    X_cat = pd.concat(X_parts, axis=0, ignore_index=True)
    p_cat = pd.concat(p_parts, axis=0, ignore_index=True)

    if len(X_cat) != len(p_cat):
        raise RuntimeError(f"X/prices misaligned after concat: X={len(X_cat)} prices={len(p_cat)}")

    X_np = X_cat.values.astype(np.float32, copy=False)
    p_np = p_cat.values.astype(np.float32, copy=False)

    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    p_np = np.nan_to_num(p_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if return_dfs:
        if return_symbol_lengths:
            return X_np, dfs_by_symbol, sym_lengths_list
        return X_np, dfs_by_symbol

    if return_symbol_lengths:
        return X_np, p_np, sym_lengths_list
    return X_np, p_np
