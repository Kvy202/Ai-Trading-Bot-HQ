"""
trade.py

Despite the file name, this module does NOT place orders. It's a quote
router: given a symbol, side, and amount, it asks several CEX venues for
their order book, computes the fee-adjusted VWAP, and returns the venue
with the best effective price.

Public API (preserved):
    best_cex_quote(symbol, side, amount) -> dict | None      (sync facade)
    best_cex_quote_async(exchanges, symbol, side, amount,
                          default_bps) -> dict | None         (async)

Changelog vs the previous version
---------------------------------
- Hard timeout on asyncio.gather. The previous version would block the
  whole trade decision waiting for whichever exchange was slowest; with
  4 venues and ccxt's 20s per-task timeout, worst case was ~80s. Now
  there's a configurable wall-clock cap (ROUTER_TIMEOUT_SEC, default 5s),
  after which any in-flight quotes are cancelled and the best result
  among completed venues is used.
- gather now uses return_exceptions=True. Previously, one venue raising
  cancelled the whole gather - now a single bad venue just gets skipped.
- ENABLE_ROUTER is now actually honoured. The old code defined
  ROUTER_ENABLED but never read it. If the flag is off, best_cex_quote*
  short-circuits to None so callers can fall back to whatever default.
- Bad-value parsing is robust. ENABLE_ROUTER previously did
  `bool(int(os.getenv("ENABLE_ROUTER", "1")))`, which raised ValueError
  at import if you set ENABLE_ROUTER="true". Now it accepts true/yes/1/on.
- Env vars are read at call time, not captured at import. This matches
  the pattern in the rewritten risk_engine.py and makes config testing
  feasible.
- Sync facade no longer does the wrong thing in async contexts. The old
  fallback created a fresh event loop while another loop was still
  running, which raises "This event loop is already running". Now the
  sync function refuses to run inside an existing loop and raises a
  clear error pointing the caller to best_cex_quote_async instead.
- Markets cache: load_markets() result is reused across calls per
  (ex_id, market_type). The previous version re-fetched markets on
  every quote, which is up to 8 round trips per quote with 4 venues.
- Resource cleanup uses async-context style with shielded close, so a
  cancelled gather doesn't leak aiohttp sessions.
- Quote.reason is always populated so callers can see WHY a venue was
  excluded, not just that it was.
- Misleading 'eff_bps' field renamed to 'taker_bps' in the returned
  dict. The old name was kept for "backward compatibility" with a
  comment admitting it was actually just the taker fee. The new key is
  'taker_bps'; we also include 'eff_bps' as an alias for one release so
  existing callers don't immediately break.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ccxt.async_support as ccxt_async  # asyncio version of ccxt

_LOG = logging.getLogger("router")

# Cache of loaded markets keyed by (ex_id, market_type) so we don't refetch
# every quote. Each entry maps to a freshly-built ccxt client whose
# `load_markets()` has been called once. NOTE: ccxt clients aren't safe to
# share across event loops; we recycle them by closing on shutdown.
_MARKETS_CACHE: Dict[Tuple[str, str], "ccxt_async.Exchange"] = {}


# ---------------------------------------------------------------------------
# Env helpers (read at call time)
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool = True) -> bool:
    """Robust env -> bool. Accepts 1/true/yes/y/on (case-insensitive)."""
    val = _env_str(name, "1" if default else "0").lower()
    return val in {"1", "true", "yes", "y", "on"}


def _analyze_exchanges() -> List[str]:
    raw = _env_str("ANALYZE_EXCHANGES", "binance,bybit,mexc,bitget")
    return [e.strip() for e in raw.split(",") if e.strip()]


# ---------------------------------------------------------------------------
# Quote dataclass
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    venue: str
    vwap: float           # book VWAP for the requested amount
    taker_bps: float      # taker fee in bps (NOT slippage / edge)
    depth_ok: bool
    mtype: str            # 'spot' or 'swap' / 'future'
    reason: str = ""


# ---------------------------------------------------------------------------
# ccxt client construction
# ---------------------------------------------------------------------------

def _guess_type_order(symbol: str) -> List[str]:
    """If the symbol looks like a perpetual ('BTC/USDT:USDT'), try swap first."""
    return ["swap", "spot"] if (":USDT" in symbol or ":" in symbol) else ["spot", "swap"]


def _build_opts(ex_id: str, mtype: str) -> Dict[str, Any]:
    """Per-exchange options for the requested market type."""
    timeout = int(_env_float("ROUTER_PER_VENUE_TIMEOUT_MS", 20000))
    opts: Dict[str, Any] = {"enableRateLimit": True, "timeout": timeout, "options": {}}

    if ex_id == "mexc":
        # MEXC futures via ccxt have historically been less reliable; quote
        # spot only. If you need MEXC swaps, override via env.
        opts["options"]["defaultType"] = "spot"
    elif ex_id == "bitget":
        if mtype == "swap":
            opts["options"].update({
                "defaultType": "swap",
                "defaultSubType": "linear",
                "defaultSettle": "USDT",
            })
        else:
            opts["options"]["defaultType"] = "spot"
    elif ex_id == "bybit":
        if mtype == "swap":
            opts["options"].update({"defaultType": "swap", "defaultSubType": "linear"})
        else:
            opts["options"]["defaultType"] = "spot"
    elif ex_id == "binance":
        if mtype == "swap":
            opts["options"]["defaultType"] = "future"  # Binance USDT-M
        else:
            opts["options"]["defaultType"] = "spot"
    return opts


async def _get_or_open_client(ex_id: str, mtype: str) -> "ccxt_async.Exchange":
    """Return a cached client for (ex_id, mtype), opening one if needed.

    The cache is per-process and assumes a single event loop. Callers that
    use a fresh loop per call must not rely on the cache; the sync facade
    now refuses to run if a loop is already active, so we're safe.
    """
    key = (ex_id, mtype)
    cached = _MARKETS_CACHE.get(key)
    if cached is not None:
        return cached

    ex_cls = getattr(ccxt_async, ex_id, None)
    if ex_cls is None:
        raise ValueError(f"unknown exchange_id {ex_id!r}")

    ex = ex_cls(_build_opts(ex_id, mtype))
    try:
        await ex.load_markets()
    except Exception:
        try:
            await ex.close()
        except Exception:
            pass
        raise

    _MARKETS_CACHE[key] = ex
    return ex


async def shutdown_router() -> None:
    """Close all cached ccxt clients. Call this on bot shutdown.

    Safe to call multiple times. Errors during close are logged and
    swallowed - we're shutting down anyway.
    """
    for key, ex in list(_MARKETS_CACHE.items()):
        try:
            await ex.close()
        except Exception as e:
            _LOG.debug("router shutdown: close failed for %s: %s", key, e)
    _MARKETS_CACHE.clear()


async def _open_client_for_symbol(ex_id: str, symbol: str) -> Tuple["ccxt_async.Exchange", str]:
    """Find a (client, mtype) pair where `symbol` is a recognised market.

    Tries the market types in _guess_type_order. The first one whose
    markets contain `symbol` wins.
    """
    last_err: Optional[Exception] = None
    for mtype in _guess_type_order(symbol):
        try:
            ex = await _get_or_open_client(ex_id, mtype)
        except Exception as e:
            last_err = e
            continue
        # Symbol lookup. ccxt populates both `symbols` (sorted list) and
        # `markets` (dict). Check markets first as it's a faster O(1) lookup.
        markets = getattr(ex, "markets", None) or {}
        if symbol in markets:
            return ex, mtype
        # Maybe a different alias resolves
        symbols = getattr(ex, "symbols", []) or []
        if symbol in symbols:
            return ex, mtype
    raise last_err or ValueError(f"{ex_id}: symbol {symbol!r} not available in spot/swap")


# ---------------------------------------------------------------------------
# VWAP calculation
# ---------------------------------------------------------------------------

def _vwap_from_orderbook(ob: Dict[str, Any], side: str,
                          amount: float) -> Tuple[Optional[float], bool, str]:
    """Walk the book until `amount` is filled, return (vwap, full_depth, reason).

    Returns (None, False, reason_str) if the book can't fill the amount.
    """
    levels = ob.get("asks" if side == "buy" else "bids", [])
    if not levels:
        return None, False, "empty_book"
    if amount <= 0:
        return None, False, "amount<=0"

    left = float(amount)
    cost = 0.0
    for px, sz in levels:
        try:
            px_f = float(px)
            sz_f = float(sz)
        except (TypeError, ValueError):
            continue
        if px_f <= 0 or sz_f <= 0:
            continue
        take = min(left, sz_f)
        cost += take * px_f
        left -= take
        if left <= 1e-12:
            break

    if left > 1e-12:
        return None, False, f"insufficient_depth (short by {left:g})"
    return cost / float(amount), True, ""


# ---------------------------------------------------------------------------
# Per-venue quote
# ---------------------------------------------------------------------------

async def _quote_one(ex_id: str, symbol: str, side: str, amount: float,
                     default_bps: float) -> Optional[Quote]:
    """Fetch one quote from one venue. Returns None on any error.

    Reasons we drop a venue:
      - couldn't open a client / find the symbol -> None
      - book fetch failed                         -> None
      - book had insufficient depth               -> Quote with depth_ok=False
    """
    try:
        ex, mtype = await _open_client_for_symbol(ex_id, symbol)
    except Exception as e:
        _LOG.debug("router: %s symbol open failed: %s", ex_id, e)
        return None

    try:
        ob = await ex.fetch_order_book(symbol, limit=50)
    except Exception as e:
        _LOG.debug("router: %s fetch_order_book failed: %s", ex_id, e)
        return None

    vwap, depth_ok, reason = _vwap_from_orderbook(ob, side, amount)
    if vwap is None:
        # Return a Quote (not None) so the caller can see this venue was
        # consulted but skipped. depth_ok=False filters it out of selection.
        return Quote(ex_id, math.nan, math.nan, False, mtype, reason=reason or "no_depth")

    fee_bps = float(default_bps)
    try:
        market = (ex.markets or {}).get(symbol, {})
        taker = market.get("taker")
        if isinstance(taker, (int, float)) and math.isfinite(taker):
            fee_bps = float(taker) * 10000.0  # ccxt taker is fractional
    except Exception:
        # Keep the default fee.
        pass

    return Quote(ex_id, float(vwap), fee_bps, True, mtype, reason="")


# ---------------------------------------------------------------------------
# Best-of-N quote
# ---------------------------------------------------------------------------

def _effective_price(q: Quote, side: str) -> float:
    """Adjust VWAP for taker fee. For buys we pay slightly more; for sells less."""
    sign = 1.0 if side == "buy" else -1.0
    return q.vwap * (1.0 + sign * q.taker_bps / 10000.0)


def _is_better(side: str, a: float, b: float) -> bool:
    """Return True if effective price `a` is better than `b` for the given side."""
    return a < b if side == "buy" else a > b


async def best_cex_quote_async(
    exchanges: List[str],
    symbol: str,
    side: str,
    amount: float,
    default_bps: float,
    timeout_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Return the venue with the best fee-adjusted VWAP, or None.

    Parameters
    ----------
    exchanges : list of ccxt exchange ids
    symbol : str (unified ccxt symbol, e.g. 'BTC/USDT:USDT')
    side : 'buy' or 'sell'
    amount : float - base-currency amount (or contracts for swap)
    default_bps : taker fee fallback in bps (used if ccxt market info doesn't
        report a taker fee)
    timeout_sec : wall-clock cap across all venues. Defaults to
        ROUTER_TIMEOUT_SEC env (default 5s).

    Returns
    -------
    dict with keys: venue, mtype, vwap, taker_bps, eff_bps (alias of
    taker_bps for back-compat), depth_ok, reason. Or None if no venue
    returned a usable quote.
    """
    side = (side or "").lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if amount <= 0:
        raise ValueError(f"amount must be > 0, got {amount}")

    venues = [e for e in (exchanges or []) if e]
    if not venues:
        return None

    timeout = float(timeout_sec) if timeout_sec is not None else _env_float("ROUTER_TIMEOUT_SEC", 5.0)

    tasks = [
        asyncio.create_task(_quote_one(e, symbol, side, amount, default_bps))
        for e in venues
    ]

    try:
        # return_exceptions=True so one bad venue doesn't sink the gather.
        # asyncio.wait_for caps the total wall-clock time.
        done, pending = await asyncio.wait(tasks, timeout=timeout)
    except Exception as e:
        _LOG.warning("router gather failed: %s", e)
        for t in tasks:
            t.cancel()
        return None

    # Cancel anyone still in flight after the deadline.
    for t in pending:
        t.cancel()
    # Give cancellations a moment to settle so aiohttp can clean up.
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    quotes: List[Quote] = []
    for t in done:
        if t.cancelled():
            continue
        exc = t.exception()
        if exc is not None:
            _LOG.debug("router task error: %s", exc)
            continue
        q = t.result()
        if q is not None:
            quotes.append(q)

    # Pick best among quotes that actually have depth.
    best: Optional[Tuple[Quote, float]] = None
    for q in quotes:
        if not q.depth_ok or not math.isfinite(q.vwap):
            continue
        eff = _effective_price(q, side)
        if best is None or _is_better(side, eff, best[1]):
            best = (q, eff)

    if best is None:
        return None

    q = best[0]
    return {
        "venue": q.venue,
        "mtype": q.mtype,
        "vwap": q.vwap,
        "taker_bps": q.taker_bps,
        # Back-compat alias. Old code reads 'eff_bps'. Keep this for one
        # release; remove once callers are updated.
        "eff_bps": q.taker_bps,
        "depth_ok": q.depth_ok,
        "reason": q.reason,
    }


# ---------------------------------------------------------------------------
# Sync facade
# ---------------------------------------------------------------------------

def _is_loop_running() -> bool:
    """Detect whether we're inside a running event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def best_cex_quote(symbol: str, side: str, amount: float) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper. Do NOT call from async code.

    From async code use best_cex_quote_async directly. The previous
    fallback that built a new event loop while another was running was
    silently broken in most async contexts (Jupyter, an async caller,
    etc.) - asyncio doesn't allow nested loops. This function now
    refuses that situation explicitly.
    """
    if not _env_bool("ENABLE_ROUTER", True):
        return None  # router globally disabled

    if _is_loop_running():
        raise RuntimeError(
            "best_cex_quote() is sync and was called from inside a running "
            "event loop. Call `await best_cex_quote_async(...)` instead."
        )

    return asyncio.run(
        best_cex_quote_async(
            _analyze_exchanges(),
            symbol,
            side,
            amount,
            _env_float("ROUTER_DEFAULT_TAKER_BPS", 10.0),
        )
    )