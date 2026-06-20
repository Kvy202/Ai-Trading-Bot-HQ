"""Tests for exchanges/factory.py and paper-mode adapter behaviour.

Paper mode must construct NO network client and require NO credentials, for
either venue. We never import ccxt or the Hyperliquid SDK here.
"""

import pytest

from exchanges.base import ExchangeAdapter
from exchanges.factory import make_adapter


def test_factory_builds_bitget_paper():
    a = make_adapter("bitget", live=False)
    assert isinstance(a, ExchangeAdapter)
    assert a.name == "bitget"
    assert a.live is False
    # paper no-ops
    assert a.create_market_order("BTCUSDT", "BUY", 0.01) == "paper"
    assert a.fetch_open_positions() == {}
    assert a.fetch_current_price("BTCUSDT") is None
    assert a.normalize_symbol("BTCUSDT") == "BTCUSDT"


def test_factory_builds_hyperliquid_paper():
    a = make_adapter("hyperliquid", live=False, testnet=True)
    assert isinstance(a, ExchangeAdapter)
    assert a.name == "hyperliquid"
    assert a.live is False
    assert a.create_market_order("BTCUSDT", "BUY", 0.01) == "paper"
    assert a.fetch_open_positions() == {}
    assert a.fetch_current_price("BTCUSDT") is None


def test_factory_case_insensitive():
    assert make_adapter("HyperLiquid", live=False).name == "hyperliquid"
    assert make_adapter("BITGET", live=False).name == "bitget"


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        make_adapter("binance", live=False)
