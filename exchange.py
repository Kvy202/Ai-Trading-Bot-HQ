"""
exchange.py

Thin ccxt wrapper used by the legacy trading paths.

NOTE: This module appears to be a SEPARATE code path from the new
live_executor.py, which builds its own ccxt client inside its Broker
class. Both files target the same exchange, but they exist in parallel.
Before relying on changes here, audit which trade*.py files import
this module - the fix you actually want depends on whether this is
the live path, the legacy path, or both.

Public API (preserved):
    live_client(exchange_id=None) -> ccxt exchange instance
    market_buy(client, symbol, amount) -> order dict
    market_sell(client, symbol, amount) -> order dict

Changelog vs the previous version
---------------------------------
- defaultType is now driven by EXCHANGE_MARKET_TYPE env (default 'swap'
  for Bitget USDT-M futures, which is what the rest of the project uses).
  The previous version hardcoded 'spot' for MEXC and {} for everything
  else, which silently meant Bitget got ccxt's default - usually wrong
  for a futures bot.
- Configurable timeout via CCXT_TIMEOUT_MS (default 30000ms). The
  previous version used ccxt's default ~10s, which is tight for futures
  orders during volatility.
- Sandbox support: live_client() honours BITGET_SANDBOX/CCXT_SANDBOX env
  by calling set_sandbox_mode(True) when supported.
- market_buy/market_sell accept reduce_only and extra params, so callers
  can close positions safely. The old API silently couldn't express this.
- Empty/None API_KEY now raises a clear error at live_client() time,
  instead of producing a half-built client that fails on first signed call.
- Added ALL the parameters you'd actually want as a single market_order()
  helper, with market_buy/market_sell as backwards-compatible shims.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import ccxt

from config import EXCHANGE_ID, API_KEY, API_SECRET, API_PASSWORD

_LOG = logging.getLogger("exchange")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = _env(name, "1" if default else "0").lower()
    return val in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def _default_market_type(exchange_id: str) -> str:
    """Pick the default ccxt market type for an exchange.

    Honours EXCHANGE_MARKET_TYPE env var first. If not set, defaults to
    'swap' (USDT-M perpetual futures) since that's what the rest of this
    project trades. This is a reversal of the old default - the previous
    version defaulted to 'spot' for MEXC only, leaving every other
    exchange to ccxt's built-in default (which is also usually 'spot').
    """
    explicit = _env("EXCHANGE_MARKET_TYPE")
    if explicit:
        return explicit.lower()
    # Project-level default: USDT-M futures.
    return "swap"


def live_client(exchange_id: Optional[str] = None) -> Any:
    """Build a configured ccxt client for the given exchange.

    Configuration sources (highest priority first):
      - exchange_id argument
      - EXCHANGE_ID from config.py
      - EXCHANGE_MARKET_TYPE env var (default 'swap')
      - CCXT_TIMEOUT_MS env var (default 30000)
      - BITGET_SANDBOX / CCXT_SANDBOX env var (default off)

    Raises
    ------
    ValueError
        If exchange_id resolves to something ccxt doesn't know about,
        or if API_KEY/API_SECRET are missing/empty.
    """
    ex_id = (exchange_id or EXCHANGE_ID or "").strip()
    if not ex_id:
        raise ValueError(
            "exchange_id is empty. Set EXCHANGE_ID in config / .env, "
            "or pass exchange_id explicitly."
        )

    cls = getattr(ccxt, ex_id, None)
    if cls is None:
        # Friendlier error than the bare AttributeError you used to get.
        available = ", ".join(sorted(ccxt.exchanges)[:8]) + ", ..."
        raise ValueError(
            f"Unknown exchange_id {ex_id!r}. ccxt knows about "
            f"{len(ccxt.exchanges)} exchanges (e.g. {available})."
        )

    if not API_KEY or not API_SECRET:
        raise ValueError(
            "API_KEY and API_SECRET are required to build a live client. "
            "Set them in your .env or config.py. (For paper trading, use "
            "live_executor.py's PAPER mode instead of building a real client.)"
        )

    market_type = _default_market_type(ex_id)
    timeout_ms = _env_int("CCXT_TIMEOUT_MS", 30000)

    params: Dict[str, Any] = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": timeout_ms,
        "options": {"defaultType": market_type},
    }
    if API_PASSWORD:
        params["password"] = API_PASSWORD

    client = cls(params)

    # Sandbox toggle. Bitget's sandbox env name is BITGET_SANDBOX in the
    # rest of the project; CCXT_SANDBOX is a generic fallback.
    sandbox = _env_bool("BITGET_SANDBOX", False) or _env_bool("CCXT_SANDBOX", False)
    if sandbox:
        if hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
            _LOG.info("exchange %s: sandbox mode enabled", ex_id)
        else:
            _LOG.warning("exchange %s: sandbox requested but ccxt class doesn't "
                         "support set_sandbox_mode; running against PRODUCTION", ex_id)

    _LOG.info("exchange %s: client built (market_type=%s, timeout=%dms)",
              ex_id, market_type, timeout_ms)
    return client


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------

def market_order(client: Any, symbol: str, side: str, amount: float,
                 reduce_only: bool = False,
                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Place a market order.

    Parameters
    ----------
    client : ccxt exchange instance
    symbol : str
        Use whatever symbol format your exchange expects. For Bitget USDT-M
        swaps that's the unified ccxt form like 'BTC/USDT:USDT'. The
        live_executor.py Broker also has symbol normalisation - if you're
        going through it, pass the shorthand 'BTCUSDT' instead.
    side : 'buy' or 'sell'
    amount : float
        Base-currency amount (NOT quote/USDT). For futures this is contracts
        in the base unit (e.g. 0.001 BTC).
    reduce_only : bool
        If True, the order can only reduce an existing position. Use this
        when closing/exiting to avoid accidentally opening a new position
        in the wrong direction.
    params : dict, optional
        Extra params passed through to ccxt. Use this for things like
        clientOrderId, time-in-force, etc.

    Raises
    ------
    ValueError
        For invalid side or non-positive amount.
    """
    side = (side or "").lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if amount <= 0:
        raise ValueError(f"amount must be > 0, got {amount}")

    extra: Dict[str, Any] = dict(params or {})
    if reduce_only:
        extra["reduceOnly"] = True

    return client.create_order(
        symbol=symbol,
        type="market",
        side=side,
        amount=amount,
        params=extra,
    )


def market_buy(client: Any, symbol: str, amount: float,
               reduce_only: bool = False,
               params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backwards-compatible market buy. Forwards to market_order(side='buy')."""
    return market_order(client, symbol, "buy", amount, reduce_only, params)


def market_sell(client: Any, symbol: str, amount: float,
                reduce_only: bool = False,
                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backwards-compatible market sell. Forwards to market_order(side='sell')."""
    return market_order(client, symbol, "sell", amount, reduce_only, params)