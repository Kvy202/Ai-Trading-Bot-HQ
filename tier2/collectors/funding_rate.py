"""
features/collectors/funding_rate.py

Collects perpetual funding rates via ccxt (public endpoint, no auth required).

Strategy:
  1. Try exchange.fetch_funding_rates(symbols) — bulk, one call for all symbols.
     Bitget supports this; most other perp venues do too.
  2. Fall back to exchange.fetch_funding_rate(symbol) per symbol if the bulk
     call raises NotSupported or any other error.

Symbols are expected in compact form (BTCUSDT).  They are converted to ccxt
unified swap form (BTC/USDT:USDT) for the API call.  The exchange ID is read
from the ccxt exchange object (exchange.id) so no separate env lookup is needed.

Dedup: the store uses INSERT OR IGNORE on (ts, exchange, symbol) where ts is
minute-rounded.  Within-minute restarts are silently dropped.

Env gate: TIER2_FUNDING_RATE=0 disables this collector.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ccxt

from tier2.collectors.base import BaseCollector

_LOG = logging.getLogger("tier2.funding_rate")


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


class FundingRateCollector(BaseCollector):
    """Collects current funding rates for all tracked symbols."""

    def __init__(self) -> None:
        super().__init__("funding_rate")

    def collect(self, exchange: Any, symbols: List[str], store: Any) -> int:
        exchange_id: str = getattr(exchange, "id", "unknown")
        unified_syms = [_unified(s) for s in symbols]

        # --- attempt bulk fetch ---
        bulk_data: Dict[str, Any] = {}
        try:
            if hasattr(exchange, "fetch_funding_rates"):
                raw = exchange.fetch_funding_rates(unified_syms)
                if isinstance(raw, dict):
                    bulk_data = raw
        except (ccxt.NotSupported, ccxt.ExchangeError, ccxt.NetworkError) as exc:
            _LOG.debug("bulk fetch_funding_rates failed (%s), falling back per-symbol", exc)

        written = 0
        for compact_sym, unified_sym in zip(symbols, unified_syms):
            info: Optional[Dict[str, Any]] = bulk_data.get(unified_sym)

            # per-symbol fallback
            if info is None:
                try:
                    info = exchange.fetch_funding_rate(unified_sym)
                except Exception as exc:
                    _LOG.warning("[funding_rate] %s: %s", compact_sym, exc)
                    continue

            if not info:
                continue

            rate = _safe_float(info.get("fundingRate"))
            mark_px = _safe_float(info.get("markPrice"))

            interval_h: Optional[float] = None
            raw_interval = info.get("fundingInterval") or info.get("interval")
            if raw_interval is not None:
                try:
                    v = float(raw_interval)
                    interval_h = v / 3_600_000 if v > 24 else v
                except (TypeError, ValueError):
                    pass
            if interval_h is None:
                interval_h = 8.0  # Bitget default

            stored = store.append_funding_rate(
                symbol=compact_sym,
                rate=rate,
                exchange=exchange_id,
                interval_hours=interval_h,
                mark_px=mark_px,
            )
            if stored:
                written += 1

        return written
