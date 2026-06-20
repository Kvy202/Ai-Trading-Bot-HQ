"""Tests for runtime/guardrails.py — the real-money safety gate.

Settings are constructed directly (not from env) so each scenario is explicit and
deterministic. The golden rule under test: anything short of a FULL confirmation
set resolves to PAPER.
"""

from runtime.settings import LIVE_CONFIRM_TOKEN, Settings
from runtime.guardrails import TradingMode, resolve_trading_mode

VALID_ADDR = "0x" + "a" * 40
VALID_KEY = "0x" + "b" * 64


def make_settings(**over) -> Settings:
    base = dict(
        exchange="hyperliquid",
        environment="development",
        live_trading=False,
        paper_trading=True,
        hl_testnet=True,
        confirm_live_trading="",
        hl_account_address="",
        hl_agent_private_key="",
        live_mode=False,
        exec_paper=True,
        bitget_sandbox=False,
    )
    base.update(over)
    return Settings(**base)


def test_default_is_paper():
    d = resolve_trading_mode(make_settings())
    assert d.mode is TradingMode.PAPER
    assert d.place_real_orders is False
    assert d.mode_name == "PAPER"


def test_cli_paper_forces_paper_even_if_everything_set():
    s = make_settings(
        exchange="hyperliquid", environment="production", hl_testnet=False,
        live_trading=True, paper_trading=False, confirm_live_trading=LIVE_CONFIRM_TOKEN,
        hl_account_address=VALID_ADDR, hl_agent_private_key=VALID_KEY,
    )
    d = resolve_trading_mode(s, cli_paper=True)
    assert d.mode is TradingMode.PAPER


def test_live_not_requested_stays_paper():
    # creds present but no live request
    s = make_settings(hl_account_address=VALID_ADDR, hl_agent_private_key=VALID_KEY)
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.PAPER


def test_hl_testnet_live_with_creds():
    s = make_settings(
        live_trading=True, paper_trading=False, hl_testnet=True,
        hl_account_address=VALID_ADDR, hl_agent_private_key=VALID_KEY,
    )
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.TESTNET_LIVE
    assert d.place_real_orders is True
    assert d.testnet is True
    assert d.mode_name == "LIVE"


def test_hl_testnet_live_without_creds_forced_paper():
    s = make_settings(live_trading=True, paper_trading=False, hl_testnet=True)
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.PAPER
    assert any("credentials" in r for r in d.reasons)


def test_cli_live_without_creds_forced_paper():
    s = make_settings(hl_testnet=True)  # paper defaults, but --live requests live
    d = resolve_trading_mode(s, cli_live=True)
    assert d.mode is TradingMode.PAPER


def test_hl_mainnet_missing_confirmations_forced_paper():
    # Everything but the typed confirmation token.
    s = make_settings(
        environment="production", hl_testnet=False,
        live_trading=True, paper_trading=False,
        hl_account_address=VALID_ADDR, hl_agent_private_key=VALID_KEY,
        confirm_live_trading="",
    )
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.PAPER
    assert any(LIVE_CONFIRM_TOKEN in r for r in d.reasons)


def test_hl_mainnet_all_confirmations_allows_real_money():
    s = make_settings(
        environment="production", hl_testnet=False,
        live_trading=True, paper_trading=False,
        hl_account_address=VALID_ADDR, hl_agent_private_key=VALID_KEY,
        confirm_live_trading=LIVE_CONFIRM_TOKEN,
    )
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.MAINNET_LIVE
    assert d.place_real_orders is True
    assert d.testnet is False


def test_bitget_sandbox_live_is_allowed_without_token():
    s = make_settings(exchange="bitget", bitget_sandbox=True,
                      live_trading=True, paper_trading=False)
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.TESTNET_LIVE
    assert d.sandbox is True
    assert d.place_real_orders is True


def test_bitget_mainnet_requires_full_confirmation():
    s = make_settings(exchange="bitget", bitget_sandbox=False,
                      live_mode=True, exec_paper=False)  # legacy live request only
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.PAPER  # forced paper: missing new confirmations

    s2 = make_settings(
        exchange="bitget", bitget_sandbox=False, environment="production",
        live_trading=True, paper_trading=False, confirm_live_trading=LIVE_CONFIRM_TOKEN,
    )
    d2 = resolve_trading_mode(s2)
    assert d2.mode is TradingMode.MAINNET_LIVE


def test_unknown_exchange_forced_paper():
    s = make_settings(exchange="ftx", live_trading=True, paper_trading=False)
    d = resolve_trading_mode(s)
    assert d.mode is TradingMode.PAPER
