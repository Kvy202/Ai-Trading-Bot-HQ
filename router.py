"""
router.py

Sync facade over multi_exchange.best_cex_quote_async, plus an optional
factory for authenticated ccxt clients.

NOTE on parallel implementations
--------------------------------
There is ANOTHER quote-router function called best_cex_quote_async in
trade.py. router.py imports the one from multi_exchange.py, NOT the one
from trade.py. The two are parallel implementations and can diverge.
The cleanest fix is at the project level: pick one, delete the other.
This file only fixes its own bugs and doesn't try to merge the two.

Public API (preserved):
    best_cex_quote(symbol, side, base_amount, market_type='swap') -> dict
    client_for(exchange_id) -> ccxt instance

Changelog vs the previous version
---------------------------------
- ROUTER_ENABLED is now actually honoured. Previously defined but never
  read; the env var ENABLE_ROUTER did nothing. When disabled, the sync
  facade returns None instead of attempting a quote.
- best_cex_quote returns None on no-venue rather than raising. The old
  RuntimeError forced every caller to wrap a try/except just to handle
  the routine "no venue available" case. Returning None matches the
  trade.py rewrite and the cross_venue_last pattern in async_scanner.py.
- Sync facade detects an already-running event loop and raises a clear
  error pointing the caller at best_cex_quote_async. The old asyncio.run
  call would die with "asyncio.run() cannot be called from a running
  event loop" - same effect, but the error message didn't help.
- client_for() defaults to USDT-M futures (defaultType='swap'), matching
  the rest of the project. The old code defaulted to ccxt's built-in
  market type for everything except MEXC, which it explicitly set to
  'spot' - the OPPOSITE of what a futures bot wants.
- client_for() sets a configurable timeout (CCXT_TIMEOUT_MS) and honours
  the BITGET_SANDBOX/CCXT_SANDBOX env vars.
- EX_KEYS_JSON parse failures are now logged with the exception. The old
  silent fallback meant a typo in your keys file silently produced
  unauthenticated clients - and you only found out when an order was
  rejected with AuthenticationRequired.
- Env vars read at call time, not module load. Same fix pattern as the
  other modules.
- Public functions have docstrings now.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from multi_exchange import best_cex_quote_async

_LOG = logging.getLogger("router")


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


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_str(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    """Robust env -> bool. Accepts 1/true/yes/y/on (case-insensitive)."""
    val = _env_str(name, "1" if default else "0").lower()
    return val in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: str = "") -> List[str]:
    return [s.strip() for s in _env_str(name, default).split(",") if s.strip()]


# ---------------------------------------------------------------------------
# EX_KEYS - parse once at import, but log failures clearly
# ---------------------------------------------------------------------------
# This is the only thing read at import time, since EX_KEYS_JSON is the
# kind of thing you set once. If you change keys you'll restart anyway.

def _load_ex_keys() -> Dict[str, Dict[str, str]]:
    raw = _env_str("EX_KEYS_JSON", "{}")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            _LOG.warning("EX_KEYS_JSON did not parse to a dict (got %s); ignoring",
                         type(parsed).__name__)
            return {}
        return parsed
    except json.JSONDecodeError as e:
        # Loud failure here saves a confusing AuthenticationRequired later.
        _LOG.error("EX_KEYS_JSON failed to parse: %s. "
                   "Authenticated clients will run UNAUTHENTICATED.", e)
        return {}


EX_KEYS: Dict[str, Dict[str, str]] = _load_ex_keys()


# ---------------------------------------------------------------------------
# Sync facade for best_cex_quote_async
# ---------------------------------------------------------------------------

def _is_loop_running() -> bool:
    """True if we're inside a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def best_cex_quote(symbol: str, side: str, base_amount: float,
                   market_type: str = "swap") -> Optional[Dict[str, Any]]:
    """Synchronous best-quote across configured CEX venues.

    Parameters
    ----------
    symbol : str
        ccxt unified symbol (e.g. 'BTC/USDT:USDT').
    side : 'buy' or 'sell'.
    base_amount : float
        Base-currency amount (NOT quote/USDT).
    market_type : str, default 'swap'
        Passed through to the underlying router. Use 'spot' for spot quotes.

    Returns
    -------
    dict with keys: venue, vwap, eff_bps, mtype.
    None if router is disabled, no venue returned a valid quote, or all
    venues failed.

    Raises
    ------
    RuntimeError
        If called from inside a running event loop. Use
        best_cex_quote_async directly from async code.
    """
    if not _env_bool("ENABLE_ROUTER", False):
        # Old code defined ROUTER_ENABLED but never read it. Now it's a
        # real gate so callers can disable the router via .env without
        # changing code.
        return None

    if _is_loop_running():
        raise RuntimeError(
            "best_cex_quote() is sync and was called from inside a running "
            "event loop. Call `await best_cex_quote_async(...)` from "
            "multi_exchange directly instead."
        )

    venues = _env_list("ROUTER_EXCHANGES", "binance,bybit,bitget,mexc")
    default_taker_bps = _env_float("ROUTER_DEFAULT_TAKER_BPS", 10.0)
    analyze_dex = _env_bool("ANALYZE_DEX", False)

    try:
        q = asyncio.run(best_cex_quote_async(
            venues, symbol, side, base_amount, default_taker_bps,
            market_type=market_type, analyze_dex=analyze_dex,
        ))
    except Exception as e:
        # Don't crash the caller for transient routing problems. Log and
        # let them fall back to whatever default they have.
        _LOG.warning("best_cex_quote: routing failed (%s); returning None",
                     type(e).__name__)
        return None

    if q is None:
        # No venue returned a usable quote. The previous code raised here;
        # returning None matches trade.py and lets callers fall back gracefully.
        _LOG.info("best_cex_quote: no venue returned a quote for %s", symbol)
        return None

    return {
        "venue": q.venue,
        "vwap": q.vwap,
        "eff_bps": q.eff_bps,
        "mtype": q.mtype,
    }


# ---------------------------------------------------------------------------
# Authenticated client factory
# ---------------------------------------------------------------------------

def client_for(exchange_id: str) -> Any:
    """Build a configured ccxt client for `exchange_id`.

    Pulls credentials from EX_KEYS (parsed from EX_KEYS_JSON env var).
    If no credentials are set for that exchange, returns a public-only
    client - which is fine for read-only ops but will fail on any
    signed call.

    Configuration:
      - defaultType: 'swap' by default (USDT-M futures), overridable
        per-call by passing it via env. The old default was ccxt's built-
        in market type for most exchanges and 'spot' for MEXC, which
        was the wrong direction for a futures bot.
      - timeout: CCXT_TIMEOUT_MS (default 30000)
      - sandbox: BITGET_SANDBOX or CCXT_SANDBOX env vars
    """
    import ccxt  # local import: no top-level dep on ccxt for unit tests

    market_type = _env_str("EXCHANGE_MARKET_TYPE", "swap")
    timeout_ms = _env_int("CCXT_TIMEOUT_MS", 30000)

    opts: Dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": timeout_ms,
        "options": {"defaultType": market_type},
    }

    keys = EX_KEYS.get(exchange_id, {}) or {}
    if keys:
        api_key = keys.get("apiKey", "") or ""
        secret = keys.get("secret", "") or ""
        if api_key and secret:
            opts["apiKey"] = api_key
            opts["secret"] = secret
            if keys.get("password"):
                opts["password"] = keys["password"]
        else:
            _LOG.warning("client_for(%s): EX_KEYS entry exists but apiKey/secret "
                         "are empty; building unauthenticated client", exchange_id)

    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        available = ", ".join(sorted(ccxt.exchanges)[:8]) + ", ..."
        raise ValueError(
            f"Unknown exchange_id {exchange_id!r}. ccxt knows about "
            f"{len(ccxt.exchanges)} exchanges (e.g. {available})."
        )

    client = cls(opts)

    if _env_bool("BITGET_SANDBOX", False) or _env_bool("CCXT_SANDBOX", False):
        if hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
            _LOG.info("client_for(%s): sandbox enabled", exchange_id)
        else:
            _LOG.warning("client_for(%s): sandbox requested but not supported",
                         exchange_id)

    client.load_markets()
    return client