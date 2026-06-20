"""
features/collectors/open_interest.py

Collects open interest (OI) snapshots via ccxt (public endpoint, no auth required).

ccxt fetch_open_interest(symbol) returns:
    {
        'symbol':              'BTC/USDT:USDT',
        'openInterestAmount':  12345.678,   # base units (BTC)
        'openInterestValue':   370703400.0, # quote units (USDT)
        'timestamp':           1234567890000,
        ...
    }

Symbols are expected in compact form (BTCUSDT).  The exchange ID is read from
the ccxt exchange object (exchange.id) so no separate env lookup is needed.

Dedup: the store uses INSERT OR IGNORE on (ts, exchange, symbol) where ts is
minute-rounded.  Within-minute restarts are silently dropped.

Env gate: TIER2_OPEN_INTEREST=0 disables this collector.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from features.collectors.base import BaseCollector

_LOG = logging.getLogger("tier2.open_interest")


def _unified(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT:USDT (USDT-M perps only)."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT:USDT"
    if symbol.endswith("USDC"):
        return f"{symbol[:-4]}/USDC:USDC"
    return symbol


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class OpenInterestCollector(BaseCollector):
    """Collects current open interest snapshots for all tracked symbols."""

    def __init__(self) -> None:
        super().__init__("open_interest")

    def collect(self, exchange: Any, symbols: List[str], store: Any) -> int:
        exchange_id: str = getattr(exchange, "id", "unknown")
        written = 0

        for compact_sym in symbols:
            unified_sym = _unified(compact_sym)
            try:
                info = exchange.fetch_open_interest(unified_sym)
            except Exception as exc:
                _LOG.warning("[open_interest] %s: %s", compact_sym, exc)
                continue

            if not info:
                continue

            oi_base = _safe_float(
                info.get("openInterestAmount") or info.get("openInterest")
            )
            oi_usd = _safe_float(
                info.get("openInterestValue") or info.get("openInterestUsd")
            )
            mark_px = _safe_float(info.get("markPrice"))

            stored = store.append_open_interest(
                symbol=compact_sym,
                oi_usd=oi_usd,
                exchange=exchange_id,
                oi_base=oi_base,
                mark_px=mark_px,
            )
            if stored:
                written += 1

        return written
