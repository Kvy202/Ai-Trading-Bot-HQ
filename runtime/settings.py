"""runtime/settings.py — typed, redaction-safe view of the bot's configuration.

Why this lives in ``runtime/`` and not ``config/``: creating ``config/__init__.py``
would shadow the legacy top-level ``config.py`` module (``from config import
EXCHANGE_ID``), breaking ``data.py``/``exchange.py``. ``runtime/`` is already a
package (it holds ``loader.py``), so new config code belongs here.

This module does NOT load ``.env`` itself — callers (``live_executor``,
``live_writer``) already run ``runtime.loader.apply_run_config`` + ``load_dotenv``
before reading settings. :meth:`Settings.from_env` simply reads ``os.environ``.

SECURITY: ``HL_AGENT_PRIVATE_KEY`` is the only secret here. It is never returned
by ``__repr__``/``summary`` and ``redact`` / ``scrub`` scrub it (and other
key-like values) out of any string before it reaches a log, error, or report.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Explicit, typed confirmation required before ANY real-money (mainnet) order.
LIVE_CONFIRM_TOKEN = "I_UNDERSTAND_LIVE_TRADING"

_TRUE = {"1", "true", "yes", "y", "on"}

# Names of environment variables that must never appear in logs/reports.
SECRET_ENV_NAMES = (
    "HL_AGENT_PRIVATE_KEY", "API_SECRET", "API_PASSWORD", "API_KEY",
    "SUPERVISOR_HMAC_SECRET", "SUPERVISOR_JWT_SECRET", "TELEGRAM_BOT_TOKEN",
)


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE


def redact(value: str, keep: int = 4) -> str:
    """Return a masked form of a secret-ish value, safe to print.

    ``0xabcdef...123456`` -> ``0xab…(redacted)``. Empty/short values become
    ``"(unset)"`` / ``"(redacted)"`` so we never leak length-revealing detail.
    """
    if not value:
        return "(unset)"
    if len(value) <= keep:
        return "(redacted)"
    return f"{value[:keep]}…(redacted)"


def scrub(text: str) -> str:
    """Best-effort scrub of secrets out of an arbitrary string.

    Masks (a) any current secret env *values* and (b) anything that looks like a
    hex private key or long token. Use before logging exception text that might
    embed config. Defense-in-depth — adapters already avoid logging secrets.
    """
    if not text:
        return text
    out = text
    for name in SECRET_ENV_NAMES:
        val = os.getenv(name)
        if val and len(val) >= 6:
            out = out.replace(val, "(redacted)")
    # 0x-prefixed 64-hex private keys, and bare 64+ hex blobs.
    out = re.sub(r"0x[0-9a-fA-F]{40,}", "0x…(redacted)", out)
    out = re.sub(r"\b[0-9a-fA-F]{64,}\b", "(redacted)", out)
    return out


@dataclass
class Settings:
    # -- venue / environment --
    exchange: str               # "hyperliquid" | "bitget"
    environment: str            # "development" | "testnet" | "production"

    # -- mode switches (canonical) --
    live_trading: bool          # master real-money switch (default OFF)
    paper_trading: bool         # simulate fills (default ON)
    hl_testnet: bool            # Hyperliquid testnet (default ON)
    confirm_live_trading: str   # must equal LIVE_CONFIRM_TOKEN for mainnet

    # -- Hyperliquid credentials --
    hl_account_address: str     # main wallet PUBLIC address (account_address)
    hl_agent_private_key: str   # agent/API wallet signer — SECRET

    # -- legacy Bitget switches (kept for backward compatibility) --
    live_mode: bool             # LIVE_MODE
    exec_paper: bool            # EXEC_PAPER
    bitget_sandbox: bool        # BITGET_SANDBOX

    @classmethod
    def from_env(cls) -> "Settings":
        exchange = (_env_str("EXCHANGE") or _env_str("EXCHANGE_ID") or "bitget").lower()
        return cls(
            exchange=exchange,
            environment=(_env_str("ENVIRONMENT", "development") or "development").lower(),
            live_trading=_env_bool("LIVE_TRADING", False),
            paper_trading=_env_bool("PAPER_TRADING", True),
            hl_testnet=_env_bool("HL_TESTNET", True),
            confirm_live_trading=_env_str("CONFIRM_LIVE_TRADING"),
            hl_account_address=_env_str("HL_ACCOUNT_ADDRESS"),
            hl_agent_private_key=_env_str("HL_AGENT_PRIVATE_KEY"),
            live_mode=_env_bool("LIVE_MODE", False),
            exec_paper=_env_bool("EXEC_PAPER", True),
            bitget_sandbox=_env_bool("BITGET_SANDBOX", False),
        )

    # -- derived helpers --

    @property
    def has_hl_credentials(self) -> bool:
        """True when both an account address and an agent key are present and
        plausibly well-formed (we never validate against the network here)."""
        addr = self.hl_account_address
        key = self.hl_agent_private_key
        addr_ok = addr.startswith("0x") and len(addr) == 42
        key_ok = key.startswith("0x") and len(key) == 66
        return addr_ok and key_ok

    @property
    def confirmed_live(self) -> bool:
        return self.confirm_live_trading == LIVE_CONFIRM_TOKEN

    def summary(self) -> str:
        """Single-line, secret-free description for startup logs."""
        return (
            f"exchange={self.exchange} env={self.environment} "
            f"live_trading={self.live_trading} paper_trading={self.paper_trading} "
            f"hl_testnet={self.hl_testnet} "
            f"hl_account_address={redact(self.hl_account_address, keep=6)} "
            f"hl_agent_private_key={redact(self.hl_agent_private_key)} "
            f"confirm_live_trading={'set' if self.confirmed_live else 'unset/invalid'}"
        )

    # never leak the key via repr
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Settings({self.summary()})"
