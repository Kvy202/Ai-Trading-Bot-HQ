"""Exchange adapter layer.

A thin, stdlib-light package that decouples the trading executor from any single
venue. The executor talks only to the :class:`ExchangeAdapter` interface; the
concrete venue (Bitget via ccxt, or Hyperliquid via the official SDK) is chosen
at runtime by :func:`exchanges.factory.make_adapter`.

IMPORTANT: keep this ``__init__`` import-cheap and offline. Do NOT import the
concrete adapters here — ``bitget_adapter`` pulls in ccxt and
``hyperliquid_adapter`` pulls in the Hyperliquid SDK. They are imported lazily
inside :mod:`exchanges.factory` so that ``import exchanges.types`` (used by the
executor at module load) never drags a heavy/optional dependency into the
process. This mirrors the failure-isolation contract used by ``v2/``.
"""

from exchanges.types import Position, OrderResult  # noqa: F401  (cheap, stdlib-only)

__all__ = ["Position", "OrderResult"]
