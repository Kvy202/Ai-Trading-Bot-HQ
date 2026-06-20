"""runtime/guardrails.py — the single decision point for real-money trading.

The executor must NEVER place real orders unless every safety condition is met.
:func:`resolve_trading_mode` is the one function that decides, and it is
**safe-by-default**: anything ambiguous or incomplete collapses to PAPER.

Resolution rules
----------------
* Default (no config) → ``PAPER``. Live trading is opt-in.
* ``--paper`` (CLI) always wins → ``PAPER``.
* "Live requested" means any of: canonical ``LIVE_TRADING=true`` +
  ``PAPER_TRADING=false``; legacy ``LIVE_MODE=1`` + ``EXEC_PAPER=0``; or ``--live``.
* If live is requested but it is a *testnet/sandbox* run (Hyperliquid
  ``HL_TESTNET=true`` or Bitget ``BITGET_SANDBOX=1``), we allow live calls
  (no real money) — Hyperliquid additionally requires credentials to be present.
* If live is requested against **mainnet (real money)**, ALL of these must hold,
  or the run is forced to PAPER with a loud, itemised reason:
    1. ``LIVE_TRADING=true`` and ``PAPER_TRADING=false``
    2. ``ENVIRONMENT=production``
    3. mainnet selected (``HL_TESTNET=false`` / not sandbox)
    4. ``CONFIRM_LIVE_TRADING=I_UNDERSTAND_LIVE_TRADING``
    5. venue credentials present (Hyperliquid: account address + agent key)

No secrets are logged — only presence/validity flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from runtime.settings import LIVE_CONFIRM_TOKEN, Settings


class TradingMode(Enum):
    PAPER = "PAPER"
    TESTNET_LIVE = "TESTNET_LIVE"
    MAINNET_LIVE = "MAINNET_LIVE"


@dataclass
class TradingDecision:
    mode: TradingMode
    exchange: str
    place_real_orders: bool          # the executor's `live` flag
    testnet: bool                    # Hyperliquid base_url selector
    sandbox: bool                    # Bitget sandbox selector
    reasons: List[str] = field(default_factory=list)

    @property
    def mode_name(self) -> str:
        """Vocabulary the executor already logs / writes to CSV: LIVE | PAPER."""
        return "LIVE" if self.place_real_orders else "PAPER"

    def describe(self) -> str:
        why = "; ".join(self.reasons) if self.reasons else "default"
        return (f"trading_mode={self.mode.value} exchange={self.exchange} "
                f"real_orders={self.place_real_orders} testnet={self.testnet} "
                f"sandbox={self.sandbox} :: {why}")


def _paper(exchange: str, reasons: List[str]) -> TradingDecision:
    return TradingDecision(
        mode=TradingMode.PAPER, exchange=exchange,
        place_real_orders=False, testnet=False, sandbox=False, reasons=reasons,
    )


def _mainnet_blockers(s: Settings) -> List[str]:
    """List the unmet conditions for real-money trading (empty == all met)."""
    missing: List[str] = []
    if not s.live_trading:
        missing.append("LIVE_TRADING must be true")
    if s.paper_trading:
        missing.append("PAPER_TRADING must be false")
    if s.environment != "production":
        missing.append("ENVIRONMENT must be 'production'")
    if not s.confirmed_live:
        missing.append(f"CONFIRM_LIVE_TRADING must equal '{LIVE_CONFIRM_TOKEN}'")
    return missing


def resolve_trading_mode(settings: Settings,
                         cli_live: bool = False,
                         cli_paper: bool = False,
                         log: Optional[Callable[[str], None]] = None) -> TradingDecision:
    """Resolve the effective, safe trading mode. Never raises."""
    emit = log or (lambda msg: None)
    s = settings
    exch = s.exchange

    # 0) Hard paper override.
    if cli_paper:
        d = _paper(exch, ["--paper flag forces paper mode"])
        emit(f"GUARDRAIL {d.describe()}")
        return d

    # 1) Is live even requested?
    canonical_live = s.live_trading and not s.paper_trading
    legacy_live = s.live_mode and not s.exec_paper
    requested_live = cli_live or canonical_live or legacy_live
    if not requested_live:
        d = _paper(exch, ["live trading not requested — default safe (paper) mode"])
        emit(f"GUARDRAIL {d.describe()}")
        return d

    # 2) Live requested — branch by venue and testnet/mainnet.
    if exch == "hyperliquid":
        if s.hl_testnet:
            if not s.has_hl_credentials:
                d = _paper(exch, [
                    "live requested on Hyperliquid TESTNET but credentials missing/invalid "
                    "(need HL_ACCOUNT_ADDRESS 0x…42 + HL_AGENT_PRIVATE_KEY 0x…66) — forcing paper",
                ])
                emit(f"GUARDRAIL {d.describe()}")
                return d
            d = TradingDecision(TradingMode.TESTNET_LIVE, exch, True, True, False,
                                ["Hyperliquid testnet live (no real money)"])
            emit(f"GUARDRAIL {d.describe()}")
            return d
        # mainnet (real money) — require the full confirmation set
        blockers = _mainnet_blockers(s)
        if not s.has_hl_credentials:
            blockers.append("HL_ACCOUNT_ADDRESS + HL_AGENT_PRIVATE_KEY must be present/valid")
        if blockers:
            d = _paper(exch, ["MAINNET real-money blocked — forcing paper:"] + blockers)
            emit(f"GUARDRAIL {d.describe()}")
            return d
        d = TradingDecision(TradingMode.MAINNET_LIVE, exch, True, False, False,
                            ["ALL live confirmations satisfied — REAL MONEY"])
        emit(f"GUARDRAIL {d.describe()}")
        return d

    if exch == "bitget":
        if s.bitget_sandbox:
            d = TradingDecision(TradingMode.TESTNET_LIVE, exch, True, False, True,
                                ["Bitget sandbox live (no real money)"])
            emit(f"GUARDRAIL {d.describe()}")
            return d
        blockers = _mainnet_blockers(s)
        if blockers:
            d = _paper(exch, ["MAINNET real-money blocked — forcing paper:"] + blockers)
            emit(f"GUARDRAIL {d.describe()}")
            return d
        d = TradingDecision(TradingMode.MAINNET_LIVE, exch, True, False, False,
                            ["ALL live confirmations satisfied — REAL MONEY"])
        emit(f"GUARDRAIL {d.describe()}")
        return d

    # Unknown venue — never trade.
    d = _paper(exch, [f"unknown EXCHANGE={exch!r} — forcing paper"])
    emit(f"GUARDRAIL {d.describe()}")
    return d
