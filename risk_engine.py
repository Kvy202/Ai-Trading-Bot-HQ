"""
risk_engine.py

Pure-function risk and sizing helpers used by the trade execution layer.
No I/O, no order placement - this module only computes numbers and
boolean gates.

Public API
----------
This module exposes TWO call shapes for each function:

    NEW shape (used by live_executor.py and the cleaned-up code paths):
      stop_distance_abs(price, atr, rv) -> float
      size_from_risk(equity, risk_per_trade, price, stop_abs) -> float
      cap_notional(price, qty, leverage, wallet_usdt) -> float
      should_pause_spread(cur_spread_bps, roll_median_bps) -> bool
      portfolio_exposure_ok(open_positions, price_map) -> bool

    OLD shape (used by trade_multi_bitget.py):
      stop_distance_abs(entry=, atr=, rv=, stop_atr_mult=, rv_mult=, min_stop_frac=)
      size_from_risk(equity=, risk_frac=, entry=, stop=)
      cap_notional(..., per_symbol_cap=, max_margin_frac=)
      should_pause_spread(current_spread_bps=, median_bps=, pause_bps=, widen_mult=)
      portfolio_exposure_ok(current_total=, new_notional=, cap=)

Detection is done by checking which kwargs are present. The wrapper
dispatches to the appropriate implementation. trade_multi_bitget.py
works unchanged.

Also exports:
    TIME_STOP_BARS - int, read from env at import time
    has_data_gap(df, timeframe_ms=60000) -> bool

Bug fixes preserved from the earlier rewrite
--------------------------------------------
- portfolio_exposure_ok accepts both dict-shaped and dataclass-shaped
  position values when called via the NEW shape.
- has_data_gap handles datetime64 timestamps correctly.
- Negative env caps log a one-time warning and are treated as "off".
- size_from_risk floors stop_abs at price * RISK_MIN_STOP_FRAC.
- should_pause_spread treats roll_median_bps == 0.0 as a valid value.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_LOG = logging.getLogger("risk_engine")

_NEG_CAP_WARNED: set = set()


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


def _positive_cap_value(v: float, name: str) -> float:
    """Treat negative values as 'cap off' with a one-time warning."""
    if v < 0:
        if name not in _NEG_CAP_WARNED:
            _NEG_CAP_WARNED.add(name)
            _LOG.warning(
                "%s is negative (%s); treating as 'cap off'. "
                "Set to a positive number to enforce a cap, or 0/unset to disable.",
                name, v,
            )
        return 0.0
    return v


def _positive_cap_env(name: str, default: float = 0.0) -> float:
    return _positive_cap_value(_env_float(name, default), name)


# ---------------------------------------------------------------------------
# Module-level constants re-exported for backwards compatibility
# ---------------------------------------------------------------------------
# trade_multi_bitget.py does:
#     from risk_engine import ..., TIME_STOP_BARS
# so this MUST be importable. Read from env at import time. Setting
# TIME_STOP_BARS=0 (or unset) means "no time stop"; the caller falls
# back to MAX_HOLD_BARS.

TIME_STOP_BARS: int = _env_int("TIME_STOP_BARS", 0)


# ---------------------------------------------------------------------------
# Stop sizing
# ---------------------------------------------------------------------------

def _stop_distance_abs_core(price: float, atr: float, rv: float,
                             atr_mult: float, rv_mult: float,
                             min_stop_frac: float) -> float:
    """Pure computation. No env reads. Both shapes call this."""
    safe_price = max(float(price), 1e-12)
    atr_part = atr_mult * max(float(atr), 0.0)
    rv_part = rv_mult * max(float(rv or 0.0), 0.0) * safe_price
    min_part = min_stop_frac * safe_price
    return max(atr_part, rv_part, min_part)


def stop_distance_abs(*args, **kwargs) -> float:
    """Absolute stop distance in price units.

    Stop is the LARGEST of:
      - ATR_MULT * ATR
      - RV_MULT * RV * price
      - MIN_STOP_FRAC * price

    Two call shapes supported:
      NEW: stop_distance_abs(price, atr, rv)
      OLD: stop_distance_abs(entry=, atr=, rv=,
                              stop_atr_mult=, rv_mult=, min_stop_frac=)
    """
    is_old_shape = (
        "entry" in kwargs
        or "stop_atr_mult" in kwargs
        or "min_stop_frac" in kwargs
    )

    if is_old_shape:
        price = kwargs.pop("entry", kwargs.pop("price", None))
        atr = kwargs.pop("atr", 0.0)
        rv = kwargs.pop("rv", 0.0)
        atr_mult = kwargs.pop("stop_atr_mult", _env_float("RISK_ATR_MULT", 2.5))
        rv_mult = kwargs.pop("rv_mult", _env_float("RISK_RV_MULT", 2.0))
        min_stop_frac = kwargs.pop("min_stop_frac", _env_float("RISK_MIN_STOP_FRAC", 0.002))
        if price is None:
            raise TypeError("stop_distance_abs (old shape): 'entry' or 'price' is required")
        return _stop_distance_abs_core(price, atr, rv, atr_mult, rv_mult, min_stop_frac)

    if args:
        price = args[0] if len(args) > 0 else kwargs.pop("price", None)
        atr = args[1] if len(args) > 1 else kwargs.pop("atr", 0.0)
        rv = args[2] if len(args) > 2 else kwargs.pop("rv", 0.0)
    else:
        price = kwargs.pop("price", None)
        atr = kwargs.pop("atr", 0.0)
        rv = kwargs.pop("rv", 0.0)
    if price is None:
        raise TypeError("stop_distance_abs: 'price' is required")
    atr_mult = _env_float("RISK_ATR_MULT", 2.5)
    rv_mult = _env_float("RISK_RV_MULT", 2.0)
    min_stop_frac = _env_float("RISK_MIN_STOP_FRAC", 0.002)
    return _stop_distance_abs_core(price, atr, rv, atr_mult, rv_mult, min_stop_frac)


def _size_from_risk_core(equity: float, risk_per_trade: float,
                          price: float, stop_abs: float) -> float:
    """Pure computation. Floors stop_abs to avoid huge positions."""
    if price <= 0 or stop_abs <= 0 or equity <= 0 or risk_per_trade <= 0:
        return 0.0
    min_stop_frac = _env_float("RISK_MIN_STOP_FRAC", 0.002)
    floor = min_stop_frac * price
    effective_stop = max(stop_abs, floor)
    return (equity * risk_per_trade) / effective_stop


def size_from_risk(*args, **kwargs) -> float:
    """Position size such that hitting `stop` loses `risk * equity`.

    Two call shapes supported:
      NEW: size_from_risk(equity, risk_per_trade, price, stop_abs)
           where stop_abs is a DISTANCE (positive number).
      OLD: size_from_risk(equity=, risk_frac=, entry=, stop=)
           where `stop` is the absolute stop PRICE level.
           This shim converts: stop_abs = abs(entry - stop).
    """
    is_old_shape = (
        "risk_frac" in kwargs
        or "entry" in kwargs
        or "stop" in kwargs
    )

    if is_old_shape:
        equity = kwargs.pop("equity", None)
        risk = kwargs.pop("risk_frac", kwargs.pop("risk_per_trade", None))
        entry = kwargs.pop("entry", kwargs.pop("price", None))
        stop = kwargs.pop("stop", kwargs.pop("stop_abs", None))
        if equity is None or risk is None or entry is None or stop is None:
            raise TypeError("size_from_risk (old shape): "
                            "equity, risk_frac, entry, stop required")
        # OLD `stop` is the stop PRICE; NEW `stop_abs` is a DISTANCE.
        stop_abs = abs(float(entry) - float(stop))
        return _size_from_risk_core(float(equity), float(risk), float(entry), stop_abs)

    if args:
        equity = args[0] if len(args) > 0 else kwargs.pop("equity", None)
        risk = args[1] if len(args) > 1 else kwargs.pop("risk_per_trade", None)
        price = args[2] if len(args) > 2 else kwargs.pop("price", None)
        stop_abs = args[3] if len(args) > 3 else kwargs.pop("stop_abs", None)
    else:
        equity = kwargs.pop("equity", None)
        risk = kwargs.pop("risk_per_trade", None)
        price = kwargs.pop("price", None)
        stop_abs = kwargs.pop("stop_abs", None)
    if equity is None or risk is None or price is None or stop_abs is None:
        raise TypeError("size_from_risk: equity, risk_per_trade, price, stop_abs required")
    return _size_from_risk_core(float(equity), float(risk), float(price), float(stop_abs))


# ---------------------------------------------------------------------------
# Notional caps
# ---------------------------------------------------------------------------

def cap_notional(price: float, qty: float, leverage: int,
                 wallet_usdt: Optional[float] = None,
                 per_symbol_cap: Optional[float] = None,
                 max_margin_frac: Optional[float] = None) -> float:
    """Apply notional and margin caps to a desired qty.

    Caps applied (most restrictive wins):
      - MAX_NOTIONAL_USDT
      - PER_SYMBOL_NOTIONAL_USDT (or per_symbol_cap arg if provided)
      - MAX_MARGIN_FRACTION      (or max_margin_frac arg if provided)

    NEW callers use just (price, qty, leverage, wallet_usdt) and the env
    is consulted. OLD callers (trade_multi_bitget.py) pass per_symbol_cap
    and max_margin_frac directly, which override env. Negative cap
    values are treated as 'off' with a one-time warning.
    """
    if price <= 0:
        return 0.0
    q = max(float(qty), 0.0)

    max_not_global = _positive_cap_env("MAX_NOTIONAL_USDT", 0.0)
    if max_not_global > 0:
        q = min(q, max_not_global / price)

    if per_symbol_cap is not None:
        per_symbol = _positive_cap_value(float(per_symbol_cap), "per_symbol_cap")
    else:
        per_symbol = _positive_cap_env("PER_SYMBOL_NOTIONAL_USDT", 0.0)
    if per_symbol > 0:
        q = min(q, per_symbol / price)

    if max_margin_frac is not None:
        margin_frac = _positive_cap_value(float(max_margin_frac), "max_margin_frac")
    else:
        margin_frac = _positive_cap_env("MAX_MARGIN_FRACTION", 0.0)

    if (wallet_usdt is not None and wallet_usdt > 0
            and margin_frac > 0 and leverage and leverage > 0):
        max_margin = wallet_usdt * margin_frac
        max_notional = max_margin * leverage
        q = min(q, max_notional / price)

    return max(q, 0.0)


# ---------------------------------------------------------------------------
# Spread guard
# ---------------------------------------------------------------------------

def should_pause_spread(*args, **kwargs) -> bool:
    """Pause entries if spread is too wide or has widened abnormally.

    Two call shapes supported:
      NEW: should_pause_spread(cur_spread_bps, roll_median_bps)
      OLD: should_pause_spread(current_spread_bps=, median_bps=,
                                pause_bps=, widen_mult=)
    """
    is_old_shape = (
        "current_spread_bps" in kwargs
        or "median_bps" in kwargs
        or "pause_bps" in kwargs
        or "widen_mult" in kwargs
    )

    if is_old_shape:
        cur = kwargs.pop("current_spread_bps", kwargs.pop("cur_spread_bps", None))
        med = kwargs.pop("median_bps", kwargs.pop("roll_median_bps", None))
        pause_bps = kwargs.pop("pause_bps", _env_float("SPREAD_PAUSE_BPS", 15.0))
        widen_mult = kwargs.pop("widen_mult", _env_float("SPREAD_WIDEN_MULT", 2.5))
    else:
        if args:
            cur = args[0] if len(args) > 0 else kwargs.pop("cur_spread_bps", None)
            med = args[1] if len(args) > 1 else kwargs.pop("roll_median_bps", None)
        else:
            cur = kwargs.pop("cur_spread_bps", None)
            med = kwargs.pop("roll_median_bps", None)
        pause_bps = _env_float("SPREAD_PAUSE_BPS", 15.0)
        widen_mult = _env_float("SPREAD_WIDEN_MULT", 2.5)

    if cur is None:
        return False
    try:
        cur_f = float(cur)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(cur_f):
        return False

    if cur_f > pause_bps:
        return True

    if med is not None:
        try:
            med_f = float(med)
        except (TypeError, ValueError):
            return False
        if np.isfinite(med_f) and med_f >= 0:
            if cur_f > widen_mult * med_f:
                return True

    return False


# ---------------------------------------------------------------------------
# Data freshness
# ---------------------------------------------------------------------------

def _ts_to_ms(ts_value: Any) -> Optional[int]:
    if isinstance(ts_value, (pd.Timestamp,)) or hasattr(ts_value, "value"):
        try:
            return int(ts_value.value) // 1_000_000
        except Exception:
            pass
    if isinstance(ts_value, np.datetime64):
        try:
            return int(ts_value.astype("datetime64[ms]").astype("int64"))
        except Exception:
            pass
    try:
        import datetime as _dt
        if isinstance(ts_value, _dt.datetime):
            return int(ts_value.timestamp() * 1000)
    except Exception:
        pass
    try:
        v = float(ts_value)
        if not math.isfinite(v):
            return None
        if v > 1e14:
            return int(v // 1_000_000)
        if v > 1e11:
            return int(v)
        return int(v * 1000)
    except (TypeError, ValueError):
        return None


def has_data_gap(df: pd.DataFrame, timeframe_ms: int = 60000) -> bool:
    data_gap_max_ms = _env_int("DATA_GAP_MAX_MS", 180_000)
    if df is None or len(df) == 0 or "ts" not in df.columns:
        return False
    ts = df["ts"].dropna()
    if ts.size < 2:
        return False
    last_ms = _ts_to_ms(ts.iloc[-1])
    prev_ms = _ts_to_ms(ts.iloc[-2])
    if last_ms is None or prev_ms is None:
        _LOG.debug("has_data_gap: could not parse ts dtype %s; skipping", ts.dtype)
        return False
    gap_ms = last_ms - prev_ms
    threshold = max(data_gap_max_ms, int(timeframe_ms) * 3)
    return gap_ms > threshold


# ---------------------------------------------------------------------------
# Portfolio exposure
# ---------------------------------------------------------------------------

def _position_qty_and_avg(pos: Any) -> Optional[tuple]:
    """Extract (qty, avg) from a dict-shaped or attribute-shaped position."""
    if isinstance(pos, dict):
        if "in_pos" in pos and not pos.get("in_pos"):
            return None
        try:
            qty = float(pos.get("qty", 0))
            avg = float(pos.get("avg", pos.get("entry_avg", pos.get("entry", 0))))
            if qty <= 0 or avg <= 0:
                return None
            return qty, avg
        except (TypeError, ValueError):
            return None

    qty = getattr(pos, "qty", None)
    avg = getattr(pos, "avg", None)
    if qty is None or avg is None:
        return None
    try:
        qty_f = float(qty)
        avg_f = float(avg)
        if qty_f <= 0 or avg_f <= 0:
            return None
        return qty_f, avg_f
    except (TypeError, ValueError):
        return None


def portfolio_exposure_ok(*args, **kwargs) -> bool:
    """Check the running portfolio against an exposure cap.

    Two call shapes supported:
      NEW: portfolio_exposure_ok(open_positions, price_map=None)
      OLD: portfolio_exposure_ok(current_total=, new_notional=, cap=)
    """
    is_old_shape = (
        "current_total" in kwargs
        or "new_notional" in kwargs
        or "cap" in kwargs
    )

    if is_old_shape:
        try:
            current_total = float(kwargs.pop("current_total", 0.0))
            new_notional = float(kwargs.pop("new_notional", 0.0))
            cap_val = kwargs.pop("cap", None)
        except (TypeError, ValueError):
            return True
        if cap_val is None:
            return True
        try:
            cap_f = _positive_cap_value(float(cap_val), "cap (arg)")
        except (TypeError, ValueError):
            return True
        if cap_f <= 0:
            return True
        return (current_total + new_notional) <= cap_f

    if args:
        open_positions = args[0]
        price_map = args[1] if len(args) > 1 else kwargs.pop("price_map", None)
    else:
        open_positions = kwargs.pop("open_positions", {})
        price_map = kwargs.pop("price_map", None)

    cap = _positive_cap_env("MAX_PORTFOLIO_EXPOSURE_USDT", 0.0)
    if cap <= 0:
        return True

    pmap = price_map or {}
    total = 0.0
    for sym, pos in (open_positions or {}).items():
        qa = _position_qty_and_avg(pos)
        if qa is None:
            continue
        qty, avg = qa
        px = pmap.get(sym)
        try:
            mark = float(px) if px is not None else 0.0
        except (TypeError, ValueError):
            mark = 0.0
        if mark <= 0 or not math.isfinite(mark):
            mark = avg
        total += qty * mark

    return total <= cap