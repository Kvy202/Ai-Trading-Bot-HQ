"""Tests for exchanges/hyperliquid_adapter.py.

Two layers:
  1. Pure helpers (action mapping, size rounding, symbol map, response/position
     parsing) — no SDK, no network.
  2. A live adapter exercised against a FAKE Hyperliquid SDK injected into
     sys.modules, so we validate the order payloads and call flow without
     installing the SDK or touching the network. No real keys are ever used.
"""

import sys
import types

import pytest

from exchanges.hyperliquid_adapter import (
    HyperliquidSDKAdapter,
    action_to_intent,
    coin_from_symbol,
    parse_order_response,
    parse_positions,
    round_size,
)

VALID_ADDR = "0x" + "a" * 40
VALID_KEY = "0x" + "b" * 64


# --------------------------------------------------------------------------- #
# 1. Pure helpers
# --------------------------------------------------------------------------- #

def test_action_to_intent():
    assert action_to_intent("BUY") == ("open", True)
    assert action_to_intent("SELL_SHORT") == ("open", False)
    assert action_to_intent("SELL") == ("close", False)
    assert action_to_intent("BUY_TO_COVER") == ("close", True)
    with pytest.raises(ValueError):
        action_to_intent("HODL")


def test_round_size_floors():
    assert round_size(0.123456, 3) == 0.123
    assert round_size(1.9999, 0) == 1.0
    assert round_size(0.0004, 3) == 0.0     # below precision -> 0 (caller rejects)
    assert round_size(-5, 2) == 0.0
    assert round_size(float("inf"), 2) == 0.0


def test_coin_from_symbol():
    coins = {"BTC": "BTC", "ETH": "ETH", "KPEPE": "kPEPE"}
    assert coin_from_symbol("BTCUSDT", coins) == "BTC"
    assert coin_from_symbol("ETH", coins) == "ETH"
    assert coin_from_symbol("1000PEPEUSDT", coins) == "kPEPE"   # 1000x alias
    assert coin_from_symbol("DOGEUSDT", coins) is None          # not in universe
    assert coin_from_symbol("", coins) is None


def test_parse_order_response_ok():
    filled = {"status": "ok", "response": {"data": {"statuses": [
        {"filled": {"oid": 42, "avgPx": "100", "totalSz": "0.1"}}]}}}
    assert parse_order_response(filled) == "42"
    resting = {"status": "ok", "response": {"data": {"statuses": [
        {"resting": {"oid": 7}}]}}}
    assert parse_order_response(resting) == "7"


def test_parse_order_response_errors():
    with pytest.raises(Exception):
        parse_order_response({"status": "err"})
    with pytest.raises(Exception):
        parse_order_response({"status": "ok", "response": {"data": {"statuses": [
            {"error": "Insufficient margin"}]}}})
    with pytest.raises(Exception):
        parse_order_response("not a dict")


def test_parse_positions():
    def short(coin: str) -> str:
        return f"{coin.upper()}USDT"

    state = {"assetPositions": [
        {"position": {"coin": "BTC", "szi": "0.5", "entryPx": "100"}},
        {"position": {"coin": "ETH", "szi": "-2", "entryPx": "50"}},
        {"position": {"coin": "DOGE", "szi": "0", "entryPx": "1"}},   # flat -> skip
    ]}
    out = parse_positions(state, short)
    assert out["BTCUSDT"].side == "long" and out["BTCUSDT"].qty == 0.5 and out["BTCUSDT"].avg == 100
    assert out["ETHUSDT"].side == "short" and out["ETHUSDT"].qty == 2 and out["ETHUSDT"].avg == 50
    assert "DOGEUSDT" not in out


# --------------------------------------------------------------------------- #
# 2. Paper mode (constructs nothing)
# --------------------------------------------------------------------------- #

def test_paper_adapter_is_inert():
    a = HyperliquidSDKAdapter(live=False, testnet=True)
    assert a.create_market_order("BTCUSDT", "BUY", 0.01) == "paper"
    assert a.fetch_open_positions() == {}
    assert a.fetch_current_price("BTCUSDT") is None


# --------------------------------------------------------------------------- #
# 3. Live mode against a FAKE injected SDK
# --------------------------------------------------------------------------- #

def _install_fake_sdk(monkeypatch):
    rec = {"orders": [], "leverage": []}

    eth = types.ModuleType("eth_account")

    class _Account:
        @staticmethod
        def from_key(_k):
            return types.SimpleNamespace(address="0xAGENT")

    eth.Account = _Account
    monkeypatch.setitem(sys.modules, "eth_account", eth)

    hl = types.ModuleType("hyperliquid")
    info_mod = types.ModuleType("hyperliquid.info")
    exch_mod = types.ModuleType("hyperliquid.exchange")
    utils_mod = types.ModuleType("hyperliquid.utils")

    class _Constants:
        TESTNET_API_URL = "https://testnet.example"
        MAINNET_API_URL = "https://mainnet.example"

    utils_mod.constants = _Constants

    class _Info:
        def __init__(self, base_url, skip_ws=True):
            self.base_url = base_url

        def meta(self):
            return {"universe": [
                {"name": "BTC", "szDecimals": 3},
                {"name": "ETH", "szDecimals": 2},
                {"name": "kPEPE", "szDecimals": 0},
            ]}

        def user_state(self, _addr):
            return {"assetPositions": [
                {"position": {"coin": "BTC", "szi": "0.25", "entryPx": "100"}}]}

        def all_mids(self):
            return {"BTC": "101.5", "ETH": "50"}

    info_mod.Info = _Info

    class _Exchange:
        def __init__(self, wallet, base_url, account_address=None):
            self.account_address = account_address

        def market_open(self, coin, is_buy, sz, px, slippage):
            rec["orders"].append(("open", coin, is_buy, sz, px, slippage))
            return {"status": "ok", "response": {"data": {"statuses": [
                {"filled": {"oid": 111, "avgPx": "101", "totalSz": str(sz)}}]}}}

        def market_close(self, coin, sz, px, slippage):
            rec["orders"].append(("close", coin, sz, px, slippage))
            return {"status": "ok", "response": {"data": {"statuses": [
                {"filled": {"oid": 222}}]}}}

        def update_leverage(self, lev, coin, is_cross):
            rec["leverage"].append((lev, coin, is_cross))

    exch_mod.Exchange = _Exchange

    monkeypatch.setitem(sys.modules, "hyperliquid", hl)
    monkeypatch.setitem(sys.modules, "hyperliquid.info", info_mod)
    monkeypatch.setitem(sys.modules, "hyperliquid.exchange", exch_mod)
    monkeypatch.setitem(sys.modules, "hyperliquid.utils", utils_mod)
    return rec


def test_live_requires_credentials(monkeypatch):
    _install_fake_sdk(monkeypatch)
    monkeypatch.delenv("HL_ACCOUNT_ADDRESS", raising=False)
    monkeypatch.delenv("HL_AGENT_PRIVATE_KEY", raising=False)
    with pytest.raises(RuntimeError):
        HyperliquidSDKAdapter(live=True, testnet=True)


def test_live_order_flow(monkeypatch):
    rec = _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", VALID_ADDR)
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", VALID_KEY)

    a = HyperliquidSDKAdapter(live=True, testnet=True)

    # open long: shorthand mapped to coin, size floored to szDecimals=3
    oid = a.create_market_order("BTCUSDT", "BUY", 0.123456)
    assert oid == "111"
    kind, coin, is_buy, sz, px, _slip = rec["orders"][-1]
    assert kind == "open" and coin == "BTC" and is_buy is True
    assert sz == 0.123
    assert rec["leverage"], "update_leverage should be set once on open"

    # close long uses market_close
    oid2 = a.create_market_order("BTCUSDT", "SELL", 0.123456)
    assert oid2 == "222"
    assert rec["orders"][-1][0] == "close"

    # positions + price come back parsed
    pos = a.fetch_open_positions()
    assert pos["BTCUSDT"].side == "long" and pos["BTCUSDT"].qty == 0.25
    assert a.fetch_current_price("BTCUSDT") == 101.5


def test_live_order_below_precision_rejected(monkeypatch):
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", VALID_ADDR)
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", VALID_KEY)

    a = HyperliquidSDKAdapter(live=True, testnet=True)
    # kPEPE has szDecimals=0; 0.4 floors to 0 -> hard reject before any order call
    with pytest.raises(Exception):
        a.create_market_order("1000PEPEUSDT", "BUY", 0.4)
