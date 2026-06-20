"""Tests for runtime/settings.py — config loading, defaults, and redaction.

All env reads go through os.environ; we clear the relevant vars per test so the
developer's shell/.env cannot make results non-deterministic.
"""

import pytest

from runtime.settings import LIVE_CONFIRM_TOKEN, Settings, redact, scrub

_RELEVANT = [
    "EXCHANGE", "EXCHANGE_ID", "ENVIRONMENT", "LIVE_TRADING", "PAPER_TRADING",
    "HL_TESTNET", "CONFIRM_LIVE_TRADING", "HL_ACCOUNT_ADDRESS",
    "HL_AGENT_PRIVATE_KEY", "LIVE_MODE", "EXEC_PAPER", "BITGET_SANDBOX",
    "API_SECRET", "API_KEY", "API_PASSWORD",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _RELEVANT:
        monkeypatch.delenv(name, raising=False)
    yield


def test_defaults_are_safe():
    s = Settings.from_env()
    # Default venue falls back to legacy EXCHANGE_ID, which itself defaults bitget.
    assert s.exchange == "bitget"
    assert s.environment == "development"
    assert s.live_trading is False
    assert s.paper_trading is True
    assert s.hl_testnet is True
    assert s.confirmed_live is False
    assert s.has_hl_credentials is False


def test_exchange_prefers_EXCHANGE_then_EXCHANGE_ID(monkeypatch):
    monkeypatch.setenv("EXCHANGE_ID", "bitget")
    assert Settings.from_env().exchange == "bitget"
    monkeypatch.setenv("EXCHANGE", "Hyperliquid")  # case-insensitive
    assert Settings.from_env().exchange == "hyperliquid"


def test_bool_parsing(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER_TRADING", "0")
    monkeypatch.setenv("HL_TESTNET", "no")
    s = Settings.from_env()
    assert s.live_trading is True
    assert s.paper_trading is False
    assert s.hl_testnet is False


def test_confirmed_live_requires_exact_token(monkeypatch):
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "yes please")
    assert Settings.from_env().confirmed_live is False
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", LIVE_CONFIRM_TOKEN)
    assert Settings.from_env().confirmed_live is True


def test_has_hl_credentials_format(monkeypatch):
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x" + "a" * 40)
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", "0x" + "b" * 64)
    assert Settings.from_env().has_hl_credentials is True
    # placeholder-style values must NOT be treated as valid credentials
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xYourMainWalletPublicAddressHere")
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", "0xYourAgentWalletPrivateKeyHere")
    assert Settings.from_env().has_hl_credentials is False


def test_summary_never_leaks_key(monkeypatch):
    secret = "0x" + "c" * 64
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", secret)
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x" + "d" * 40)
    s = Settings.from_env()
    text = s.summary() + repr(s)
    assert secret not in text
    assert "redacted" in text


def test_redact_helper():
    assert redact("") == "(unset)"
    assert redact("0x") == "(redacted)"        # too short to show a prefix
    masked = redact("0x" + "e" * 64)
    assert masked.endswith("(redacted)") and "0x" in masked
    assert "e" * 64 not in masked


def test_scrub_masks_known_secret_and_hex(monkeypatch):
    secret = "0x" + "f" * 64
    monkeypatch.setenv("API_SECRET", "supersecretvalue123")
    text = f"boom key={secret} api=supersecretvalue123 ok"
    out = scrub(text)
    assert secret not in out
    assert "supersecretvalue123" not in out
    assert "redacted" in out
