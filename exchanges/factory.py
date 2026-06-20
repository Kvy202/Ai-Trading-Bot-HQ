"""exchanges/factory.py — choose the right ExchangeAdapter at runtime.

The executor calls :func:`make_adapter` once at startup. It imports the concrete
adapters lazily so that selecting (or paper-mode running) one venue never forces
the other venue's heavy/optional dependency (ccxt vs the Hyperliquid SDK) to be
installed.

Selection is by ``exchange`` name only; whether real orders are placed is decided
upstream by :func:`runtime.guardrails.resolve_trading_mode` and passed in as
``live`` (+ ``testnet`` / ``sandbox``). In paper mode (``live=False``) the adapter
constructs no network client and requires no credentials.
"""

from __future__ import annotations

from typing import Callable, Optional

from exchanges.base import ExchangeAdapter


def make_adapter(exchange: str,
                 live: bool,
                 testnet: bool = True,
                 sandbox: bool = False,
                 log: Optional[Callable[[str], None]] = None,
                 log_err: Optional[Callable[[str], None]] = None) -> ExchangeAdapter:
    """Build the adapter for ``exchange`` ('hyperliquid' | 'bitget').

    Raises ValueError for an unknown venue (the guardrail layer already forces
    paper for unknown venues, but the factory fails loud if called directly).
    """
    ex = (exchange or "").strip().lower()

    if ex == "hyperliquid":
        from exchanges.hyperliquid_adapter import HyperliquidSDKAdapter
        return HyperliquidSDKAdapter(live=live, testnet=testnet, log=log, log_err=log_err)

    if ex == "bitget":
        from exchanges.bitget_adapter import BitgetAdapter
        return BitgetAdapter(live=live, sandbox=sandbox, log=log, log_err=log_err)

    raise ValueError(
        f"unknown EXCHANGE={exchange!r}; expected 'hyperliquid' or 'bitget'"
    )
