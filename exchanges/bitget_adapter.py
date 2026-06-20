"""BitgetAdapter — the legacy Bitget USDT-M futures execution path.

This is the original ``Broker`` class lifted out of ``tools/live_executor.py``
**verbatim** (logic unchanged), re-homed behind the :class:`ExchangeAdapter`
interface so the executor can select it via ``EXCHANGE=bitget``. Keeping it
intact preserves every Bitget-specific quirk the bot was tuned against:

* ccxt ``bitget`` client, USDT-M swap default type
* hedge-mode ``tradeSide`` / ``holdSide`` params on opens (avoids error 40774)
* the dedicated ``/api/v2/mix/order/close-positions`` REST endpoint for closes
  (the normal place-order endpoint rejects hedge-mode closes with error 22002)
* shorthand→unified symbol normalization (``BTCUSDT`` → ``BTC/USDT:USDT``)
* live position reconciliation via ``fetch_positions``

The only structural changes vs the old ``Broker``:
* it subclasses ``ExchangeAdapter`` and returns the shared ``Position`` type
* logging is injected (``log`` / ``log_err`` callables) instead of importing the
  executor's module-level loggers — adapters must not import from ``tools/``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, Optional

try:
    import ccxt  # type: ignore
except Exception:  # ccxt is optional in paper mode
    ccxt = None

from exchanges.base import ExchangeAdapter
from exchanges.types import Position


# -- local env helpers (kept identical to the executor's, so behaviour matches) --

def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_str(name, str(default))))
    except Exception:
        return int(default)


class BitgetAdapter(ExchangeAdapter):
    """Thin wrapper around ccxt.bitget that no-ops in paper mode.

    Includes a symbol normalization layer: signal files use exchange-internal
    shorthand like 'BTCUSDT', but ccxt's unified swap symbols look like
    'BTC/USDT:USDT'. We build a lookup table on init from the loaded markets
    so live orders go to the right place.
    """

    name = "bitget"

    def __init__(self, live: bool, sandbox: bool,
                 log: Optional[Callable[[str], None]] = None,
                 log_err: Optional[Callable[[str], None]] = None) -> None:
        self.live = live
        self.sandbox = sandbox
        self._log = log or (lambda msg: None)
        self._log_err = log_err or (lambda msg: None)
        self.exchange = None
        # Maps shorthand ('BTCUSDT', 'BTC/USDT', 'BTC/USDT:USDT') -> unified symbol.
        # In paper mode this stays empty and normalize_symbol is a no-op.
        self._symbol_map: Dict[str, str] = {}
        self._leverage_set: set = set()  # symbols where set_leverage has been called
        if live:
            if ccxt is None:
                raise RuntimeError("ccxt is not installed; cannot run live orders")
            api_key = _env_str("API_KEY")
            api_secret = _env_str("API_SECRET")
            api_password = _env_str("API_PASSWORD")
            if not api_key or not api_secret or not api_password:
                raise RuntimeError("LIVE_MODE=1 requires API_KEY, API_SECRET and API_PASSWORD")
            self.exchange = ccxt.bitget({
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
                "timeout": _env_int("CCXT_TIMEOUT_MS", 120000),
                "options": {"defaultType": "swap"},
            })
            if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
                self.exchange.set_sandbox_mode(True)
            self.exchange.load_markets()
            self._build_symbol_map()

    # -- Symbol normalization --------------------------------------------------

    def _build_symbol_map(self) -> None:
        """Index loaded markets so we can translate 'BTCUSDT' to a ccxt symbol.

        Only USDT-settled perpetual swaps are indexed; spot and other futures
        types are ignored on purpose - this bot only trades USDT-M perps.
        """
        if not self.exchange or not getattr(self.exchange, "markets", None):
            return
        count = 0
        for unified_symbol, market in self.exchange.markets.items():
            if not market.get("swap"):
                continue
            if market.get("settle") != "USDT":
                continue
            base = market.get("base") or ""
            mid = market.get("id") or ""

            # The ccxt unified symbol itself, e.g. 'BTC/USDT:USDT'.
            self._symbol_map[unified_symbol.upper()] = unified_symbol
            # The exchange-internal id, e.g. 'BTCUSDT'.
            if mid:
                self._symbol_map[mid.upper()] = unified_symbol
            # Common shorthand the writer might emit, e.g. 'BTCUSDT' or 'BTC/USDT'.
            if base:
                self._symbol_map[f"{base}USDT".upper()] = unified_symbol
                self._symbol_map[f"{base}/USDT".upper()] = unified_symbol
            count += 1
        self._log(f"broker: indexed {count} USDT-M swap markets for symbol normalization")

    def normalize_symbol(self, symbol: str) -> str:
        """Return the ccxt unified symbol for `symbol`, or the input unchanged."""
        if not self.live:
            return symbol  # paper: nothing to normalize
        if not symbol:
            return symbol
        unified = self._symbol_map.get(symbol.upper())
        if unified:
            return unified
        # Unknown shorthand - don't silently swallow. Log clearly so the
        # operator sees it; pass through and let ccxt raise if it's invalid.
        self._log_err(f"broker: symbol {symbol!r} not in market map - passing through unchanged")
        return symbol

    # -- Orders ----------------------------------------------------------------

    def _close_position_flash(self, symbol: str, hold_side: str) -> str:
        """Close an entire position via Bitget's /api/v2/mix/order/close-positions.

        The regular place-order endpoint returns error 22002 for USDT-FUTURES
        hedge-mode close orders (Bitget API quirk). The dedicated close-positions
        endpoint is reliable and works correctly.
        """
        import hmac as _hmac
        import hashlib as _hashlib
        import base64 as _b64
        import json as _json
        import requests as _req

        assert self.exchange is not None
        ts = str(int(time.time() * 1000))
        body = _json.dumps({
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "holdSide": hold_side,
        })
        path = "/api/v2/mix/order/close-positions"
        msg = ts + "POST" + path + body
        sig = _b64.b64encode(
            _hmac.new(self.exchange.secret.encode(), msg.encode(), _hashlib.sha256).digest()
        ).decode()
        headers = {
            "ACCESS-KEY": self.exchange.apiKey,
            "ACCESS-SIGN": sig,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.exchange.password,
            "Content-Type": "application/json",
        }
        resp = _req.post("https://api.bitget.com" + path, headers=headers, data=body, timeout=10)
        data = resp.json()
        if data.get("code") != "00000":
            raise Exception(f"flash_close {symbol} holdSide={hold_side}: {data}")
        success = (data.get("data") or {}).get("successList") or []
        if not success:
            raise Exception(f"flash_close {symbol}: empty successList: {data}")
        return str(success[0].get("orderId", "flash-close"))

    def create_market_order(self, symbol: str, action: str, qty: float, reduce_only: bool = False) -> str:
        """action: BUY, SELL, SELL_SHORT, BUY_TO_COVER. Returns an order id (or 'paper')."""
        if not self.live:
            return "paper"
        assert self.exchange is not None

        # Close orders use Bitget's dedicated close-positions endpoint.
        # The regular place-order endpoint rejects hedge-mode closes with error 22002.
        if action == "SELL":
            return self._close_position_flash(symbol, "long")
        if action == "BUY_TO_COVER":
            return self._close_position_flash(symbol, "short")

        unified = self.normalize_symbol(symbol)
        side = "buy"  # BUY and SELL_SHORT both open positions; SELL_SHORT uses holdSide

        # Set leverage once per symbol using the configured LEVERAGE env var.
        if symbol not in self._leverage_set:
            leverage = _env_int("LEVERAGE", 5)
            try:
                self.exchange.set_leverage(leverage, unified)
                self._leverage_set.add(symbol)
            except Exception as e:
                self._log_err(f"broker: set_leverage({leverage}x, {symbol}) failed: {e}")

        # Apply exchange amount precision and enforce minimum lot size.
        qty = float(self.exchange.amount_to_precision(unified, qty))
        market = self.exchange.market(unified)
        min_qty = float((market.get("limits") or {}).get("amount", {}).get("min") or 0)
        if min_qty > 0 and qty < min_qty:
            raise Exception(
                f"order rejected: {symbol} qty={qty} below exchange minimum={min_qty} "
                f"(increase PER_SYMBOL_NOTIONAL_USDT)"
            )

        # Bitget USDT-M hedge mode: tradeSide (open) and holdSide (long/short)
        # are required on every open order or Bitget returns error 40774.
        params: Dict[str, Any] = {}
        if action == "BUY":            # open long
            params["tradeSide"] = "open"
            params["holdSide"] = "long"
        elif action == "SELL_SHORT":   # open short
            side = "sell"
            params["tradeSide"] = "open"
            params["holdSide"] = "short"
        order = self.exchange.create_order(unified, "market", side, qty, None, params)
        return str(order.get("id") or order.get("clientOrderId") or "live")

    # -- Position sync (live only) ---------------------------------------------

    def fetch_open_positions(self) -> Dict[str, Position]:
        """Pull live positions from the exchange and return them keyed by shorthand symbol.

        Returns an empty dict in paper mode or if the exchange doesn't support
        fetchPositions. On any error, returns an empty dict and logs - never
        raises into the caller.
        """
        if not self.live or not self.exchange:
            return {}
        if not self.exchange.has.get("fetchPositions"):
            self._log("broker: exchange does not support fetchPositions; cannot sync")
            return {}
        try:
            raw = self.exchange.fetch_positions()
        except Exception as e:
            self._log_err(f"broker: fetch_positions failed: {e}")
            return {}

        # Reverse map: unified ccxt symbol -> shorthand the rest of our code uses.
        # We prefer the 'BTCUSDT' form because that's what the signal CSV uses.
        unified_to_short: Dict[str, str] = {}
        for short, unified in self._symbol_map.items():
            # Only keep mappings where `short` looks like 'BTCUSDT' (no slash, no colon).
            if "/" not in short and ":" not in short:
                unified_to_short.setdefault(unified, short)

        out: Dict[str, Position] = {}
        for entry in raw:
            try:
                contracts = float(entry.get("contracts") or entry.get("contractSize") or 0)
                if contracts <= 0:
                    continue
                side = str(entry.get("side") or "").lower()
                if side not in ("long", "short"):
                    continue
                avg = float(entry.get("entryPrice") or 0)
                if avg <= 0 or not math.isfinite(avg):
                    continue
                unified_sym = entry.get("symbol") or ""
                short_sym = unified_to_short.get(unified_sym, unified_sym)
                out[short_sym] = Position(side=side, qty=contracts, avg=avg)
            except Exception:
                continue
        return out

    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Return last trade price for symbol, or None on error. Live mode only."""
        if not self.live or not self.exchange:
            return None
        try:
            unified = self.normalize_symbol(symbol)
            ticker = self.exchange.fetch_ticker(unified)
            price = float(ticker.get("last") or ticker.get("close") or 0)
            return price if price > 0 else None
        except Exception as e:
            self._log_err(f"broker: fetch_current_price({symbol}) failed: {e}")
            return None

    def close(self) -> None:
        # ccxt sync clients hold no persistent connection that needs closing.
        return None
