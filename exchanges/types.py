"""Shared, venue-neutral data types for the exchange adapter layer.

Stdlib-only on purpose: this module is imported by ``tools/live_executor.py`` at
module load, and must never pull in ccxt, the Hyperliquid SDK, or anything that
could fail offline.

``Position`` is the single source of truth for an open position across the whole
system. It was previously defined inside ``tools/live_executor.py``; it lives
here now so both the executor and every adapter agree on the exact shape. The
field order (``side``, ``qty``, ``avg``) is preserved so existing positional
construction — e.g. ``Position(want, qty, entry_fill)`` — and keyword
construction — e.g. ``Position(side="long", qty=1.0, avg=100.0)`` — both keep
working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    side: str  # "long" or "short"
    qty: float
    avg: float


@dataclass
class OrderResult:
    """Normalized result of an order placement across venues.

    The executor only needs an order-id string today (it stores nothing else),
    so adapters return a plain ``str`` from ``create_market_order``. This richer
    type is provided for adapters/tests that want to surface fill details without
    breaking the executor's existing string contract.
    """

    order_id: str
    status: str = ""          # e.g. "filled", "resting", "paper", "live"
    filled_qty: float = 0.0
    avg_price: float = 0.0
    raw: object = None        # venue-native response, for debugging only

    def __str__(self) -> str:  # so str(result) behaves like the legacy id return
        return self.order_id
