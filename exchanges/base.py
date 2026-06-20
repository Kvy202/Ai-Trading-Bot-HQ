"""ExchangeAdapter — the venue-neutral interface the executor depends on.

The live executor (``tools/live_executor.py``) is intentionally written against
this small surface. As long as a concrete adapter implements these methods with
the same semantics, the 1600-line executor loop runs unchanged on any venue.

Method contract (mirrors the behaviour the executor already relied on):

* ``create_market_order(symbol, action, qty, reduce_only) -> str``
    Place a market order. ``action`` is one of ``BUY``, ``SELL``,
    ``SELL_SHORT``, ``BUY_TO_COVER`` (the executor's vocabulary). ``BUY`` /
    ``SELL_SHORT`` open; ``SELL`` / ``BUY_TO_COVER`` close. Returns an order-id
    string (``"paper"`` in paper mode). May raise on hard validation failures
    (e.g. below the venue minimum) so the caller can log and skip.

* ``fetch_open_positions() -> Dict[str, Position]``
    Pull authoritative open positions, keyed by the executor's shorthand symbol
    (e.g. ``"BTCUSDT"``). Returns ``{}`` in paper mode or on ANY error — never
    raises into the caller.

* ``fetch_current_price(symbol) -> Optional[float]``
    Last/mid price for restart-close checks. ``None`` on error/paper.

* ``normalize_symbol(symbol) -> str``  (optional helper)
    Translate the executor's shorthand to the venue's native symbol.

* ``set_leverage(leverage, symbol)``  (optional, best-effort)

* ``close()``  release any network resources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from exchanges.types import Position


class ExchangeAdapter(ABC):
    """Abstract base every venue adapter implements.

    Attributes
    ----------
    live : bool
        True when the adapter places real orders against the venue. False means
        the executor handles paper simulation and never calls the order path.
    name : str
        Short venue identifier for logs (e.g. ``"bitget"``, ``"hyperliquid"``).
    """

    live: bool = False
    name: str = "base"

    @abstractmethod
    def create_market_order(self, symbol: str, action: str, qty: float,
                            reduce_only: bool = False) -> str:
        ...

    @abstractmethod
    def fetch_open_positions(self) -> Dict[str, Position]:
        ...

    @abstractmethod
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        ...

    # -- optional helpers (safe defaults) -----------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        return symbol

    def set_leverage(self, leverage: int, symbol: str) -> None:
        return None

    def close(self) -> None:
        return None
