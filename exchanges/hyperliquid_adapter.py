"""HyperliquidSDKAdapter — execution on Hyperliquid via the OFFICIAL SDK.

Uses ``hyperliquid-python-sdk`` (imported as ``hyperliquid``) — **not CCXT** — to
place market orders, read positions, and read mid prices. Implements the venue-
neutral :class:`ExchangeAdapter` so the existing executor loop runs unchanged.

Security model (matches the project requirements)
-------------------------------------------------
* ``HL_ACCOUNT_ADDRESS`` = the **main wallet PUBLIC address**. It is passed to the
  SDK as ``account_address`` — the account whose funds/positions are traded.
* ``HL_AGENT_PRIVATE_KEY`` = an **API/agent wallet** private key, used only to
  build the signer. An approved Hyperliquid agent can sign orders but CANNOT
  withdraw. The main wallet's private key is never used, requested, or stored.
* The agent wallet must be approved on Hyperliquid (UI → API) before any order.

Offline-safe: the SDK is imported lazily and only when ``live=True``. In paper
mode this adapter constructs nothing and every method no-ops, so the module
imports cleanly without the SDK installed (tests mock the SDK entirely).

The pure helpers near the bottom (``action_to_intent``, ``round_size``,
``parse_order_response``, ``coin_from_symbol``) contain all the fiddly venue
rules and are unit-tested without any network or SDK.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Tuple

from exchanges.base import ExchangeAdapter
from exchanges.types import Position

# Hyperliquid coins differ from the writer's shorthand. Known special cases where
# stripping "USDT" is not enough (Hyperliquid lists 1000x markets as k-prefixed).
_SHORTHAND_TO_COIN_ALIASES = {
    "1000PEPE": "kPEPE",
    "1000BONK": "kBONK",
    "1000SHIB": "kSHIB",
    "1000FLOKI": "kFLOKI",
}
_COIN_TO_SHORTHAND_ALIASES = {v: k for k, v in _SHORTHAND_TO_COIN_ALIASES.items()}

# Default market-order slippage tolerance (the SDK turns market orders into
# aggressive IOC limit orders priced this far through the mid).
_DEFAULT_SLIPPAGE = 0.05


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except Exception:
        return float(default)


class HyperliquidSDKAdapter(ExchangeAdapter):
    name = "hyperliquid"

    def __init__(self, live: bool, testnet: bool,
                 log: Optional[Callable[[str], None]] = None,
                 log_err: Optional[Callable[[str], None]] = None) -> None:
        self.live = live
        self.testnet = testnet
        self._log = log or (lambda msg: None)
        self._log_err = log_err or (lambda msg: None)
        self._account_address = ""
        self.info = None
        self.exchange = None
        self._sz_decimals: Dict[str, int] = {}     # coin -> szDecimals
        self._coin_by_upper: Dict[str, str] = {}    # "BTC" -> canonical "BTC"; "KPEPE" -> "kPEPE"
        self._leverage_set: set = set()
        self._slippage = _env_float("HL_SLIPPAGE", _DEFAULT_SLIPPAGE)

        if not live:
            return  # paper mode: construct nothing, every method no-ops

        # --- lazy, fail-loud SDK import (kept out of module top for offline use) ---
        try:
            import eth_account  # noqa: F401
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils import constants
        except Exception as exc:  # pragma: no cover - import-time only
            raise RuntimeError(
                "hyperliquid-python-sdk (and eth-account) are required for live "
                "Hyperliquid trading. Install with: pip install hyperliquid-python-sdk"
            ) from exc

        account_address = _env_str("HL_ACCOUNT_ADDRESS")
        agent_key = _env_str("HL_AGENT_PRIVATE_KEY")
        if not account_address or not agent_key:
            # Never echo the key; only state which var is missing.
            missing = "HL_ACCOUNT_ADDRESS" if not account_address else "HL_AGENT_PRIVATE_KEY"
            raise RuntimeError(f"live Hyperliquid trading requires {missing} to be set")

        self._account_address = account_address
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL

        # Build the signer from the AGENT key only; main key never touched.
        wallet = eth_account.Account.from_key(agent_key)

        self.info = Info(base_url, skip_ws=True)
        self.exchange = Exchange(wallet, base_url, account_address=account_address)
        self._load_meta()
        self._log(
            f"hyperliquid: connected net={'testnet' if testnet else 'MAINNET'} "
            f"account={account_address[:6]}…(redacted) "
            f"universe={len(self._sz_decimals)} coins slippage={self._slippage}"
        )

    # -- metadata ----------------------------------------------------------------

    def _load_meta(self) -> None:
        """Index szDecimals + coin names from the exchange universe."""
        try:
            meta = self.info.meta()
            for asset in meta.get("universe", []):
                name = str(asset.get("name", ""))
                if not name:
                    continue
                self._sz_decimals[name] = int(asset.get("szDecimals", 2))
                self._coin_by_upper[name.upper()] = name
        except Exception as exc:
            self._log_err(f"hyperliquid: meta load failed: {exc}")

    # -- symbol mapping ----------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        coin = coin_from_symbol(symbol, self._coin_by_upper)
        if coin is None:
            self._log_err(f"hyperliquid: symbol {symbol!r} not in universe — passing through")
            # strip USDT as a last resort so an obviously-correct coin still works
            return symbol.upper().removesuffix("USDT")
        return coin

    def _coin_to_shorthand(self, coin: str) -> str:
        if coin in _COIN_TO_SHORTHAND_ALIASES:
            return _COIN_TO_SHORTHAND_ALIASES[coin]
        return f"{coin.upper()}USDT"

    # -- orders ------------------------------------------------------------------

    def create_market_order(self, symbol: str, action: str, qty: float, reduce_only: bool = False) -> str:
        if not self.live:
            return "paper"
        assert self.exchange is not None

        coin = self.normalize_symbol(symbol)
        kind, is_buy = action_to_intent(action)
        sz = round_size(qty, self._sz_decimals.get(coin, 2))
        if sz <= 0:
            raise Exception(
                f"order rejected: {symbol} qty={qty} rounds to 0 at szDecimals="
                f"{self._sz_decimals.get(coin, 2)} (increase PER_SYMBOL_NOTIONAL_USDT)"
            )

        if kind == "open":
            # Set leverage once per coin (best-effort; never blocks the order).
            self._maybe_set_leverage(coin)
            resp = self.exchange.market_open(coin, is_buy, sz, None, self._slippage)
        else:  # close — reduce the existing position
            resp = self.exchange.market_close(coin, sz, None, self._slippage)

        return parse_order_response(resp)

    def _maybe_set_leverage(self, coin: str) -> None:
        if coin in self._leverage_set:
            return
        lev = 0
        try:
            lev = int(float(_env_str("LEVERAGE", "5")))
        except Exception:
            lev = 5
        try:
            self.exchange.update_leverage(lev, coin, True)  # is_cross=True
            self._leverage_set.add(coin)
        except Exception as exc:
            self._log_err(f"hyperliquid: update_leverage({lev}x, {coin}) failed: {exc}")

    def set_leverage(self, leverage: int, symbol: str) -> None:
        if not self.live or self.exchange is None:
            return
        coin = self.normalize_symbol(symbol)
        try:
            self.exchange.update_leverage(int(leverage), coin, True)
            self._leverage_set.add(coin)
        except Exception as exc:
            self._log_err(f"hyperliquid: set_leverage({leverage}x, {symbol}) failed: {exc}")

    # -- account state -----------------------------------------------------------

    def fetch_open_positions(self) -> Dict[str, Position]:
        if not self.live or self.info is None:
            return {}
        try:
            state = self.info.user_state(self._account_address)
        except Exception as exc:
            self._log_err(f"hyperliquid: user_state failed: {exc}")
            return {}
        return parse_positions(state, self._coin_to_shorthand)

    def fetch_current_price(self, symbol: str) -> Optional[float]:
        if not self.live or self.info is None:
            return None
        try:
            coin = self.normalize_symbol(symbol)
            mids = self.info.all_mids()
            px = mids.get(coin)
            if px is None:
                return None
            val = float(px)
            return val if val > 0 and math.isfinite(val) else None
        except Exception as exc:
            self._log_err(f"hyperliquid: all_mids({symbol}) failed: {exc}")
            return None

    def close(self) -> None:
        # Info was built with skip_ws=True, so there is no socket to tear down.
        return None


# ---------------------------------------------------------------------------
# Pure helpers (no SDK, no network) — unit-tested directly.
# ---------------------------------------------------------------------------

def action_to_intent(action: str) -> Tuple[str, bool]:
    """Map the executor's action vocabulary to (kind, is_buy).

    Returns ``("open"|"close", is_buy)``. ``is_buy`` is meaningful for opens
    (BUY=long, SELL_SHORT=short) and reflects the reducing direction for closes.
    Raises ValueError on an unrecognised action.
    """
    a = (action or "").upper()
    if a == "BUY":            # open long
        return "open", True
    if a == "SELL_SHORT":     # open short
        return "open", False
    if a == "SELL":           # close long -> sell to reduce
        return "close", False
    if a == "BUY_TO_COVER":   # close short -> buy to reduce
        return "close", True
    raise ValueError(f"unknown action {action!r}")


def round_size(qty: float, sz_decimals: int) -> float:
    """Floor a size to the venue's szDecimals (never round UP past available)."""
    try:
        q = float(qty)
    except Exception:
        return 0.0
    if not math.isfinite(q) or q <= 0:
        return 0.0
    d = max(0, int(sz_decimals))
    factor = 10 ** d
    return math.floor(q * factor) / factor


def coin_from_symbol(symbol: str, coin_by_upper: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Translate writer shorthand ('BTCUSDT') to a Hyperliquid coin ('BTC').

    Resolution order: known 1000x alias → universe lookup (USDT/USD/USDC/PERP
    suffix stripped) → ``None`` (caller decides the fallback).
    """
    if not symbol:
        return None
    coin_by_upper = coin_by_upper or {}
    s = symbol.strip().upper()

    # explicit 1000x alias (1000PEPEUSDT -> kPEPE)
    for shorthand, coin in _SHORTHAND_TO_COIN_ALIASES.items():
        if s == shorthand or s == f"{shorthand}USDT":
            return coin

    candidate = s
    for suffix in ("USDT", "USDC", "USD", "PERP"):
        if candidate.endswith(suffix) and len(candidate) > len(suffix):
            candidate = candidate[: -len(suffix)]
            break

    if candidate in coin_by_upper:
        return coin_by_upper[candidate]
    if s in coin_by_upper:        # already a bare coin like "BTC"
        return coin_by_upper[s]
    return None


def parse_order_response(resp: dict) -> str:
    """Extract an order id / status from a Hyperliquid order response.

    Raises Exception on a non-ok status or a per-order error, so the executor
    logs and skips (parity with the Bitget adapter's hard-failure behaviour).
    """
    if not isinstance(resp, dict):
        raise Exception(f"hyperliquid: unexpected order response: {resp!r}")
    if resp.get("status") != "ok":
        raise Exception(f"hyperliquid: order failed: {resp!r}")
    try:
        statuses = resp["response"]["data"]["statuses"]
    except Exception:
        raise Exception(f"hyperliquid: malformed order response: {resp!r}")
    ids: List[str] = []
    for st in statuses:
        if not isinstance(st, dict):
            continue
        if "error" in st:
            raise Exception(f"hyperliquid: order error: {st['error']}")
        if "filled" in st:
            ids.append(str(st["filled"].get("oid", "filled")))
        elif "resting" in st:
            ids.append(str(st["resting"].get("oid", "resting")))
    return ids[0] if ids else "live"


def parse_positions(state: dict, coin_to_shorthand: Callable[[str], str]) -> Dict[str, Position]:
    """Convert ``info.user_state`` output into ``{shorthand: Position}``."""
    out: Dict[str, Position] = {}
    if not isinstance(state, dict):
        return out
    for ap in state.get("assetPositions", []) or []:
        try:
            pos = ap.get("position", {})
            coin = str(pos.get("coin", ""))
            szi = float(pos.get("szi", 0) or 0)
            if not coin or szi == 0 or not math.isfinite(szi):
                continue
            avg = float(pos.get("entryPx", 0) or 0)
            if avg <= 0 or not math.isfinite(avg):
                continue
            side = "long" if szi > 0 else "short"
            out[coin_to_shorthand(coin)] = Position(side=side, qty=abs(szi), avg=avg)
        except Exception:
            continue
    return out
