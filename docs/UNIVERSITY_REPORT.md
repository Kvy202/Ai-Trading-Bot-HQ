# Design and Implementation of a Risk-Managed AI Trading Bot on the Hyperliquid Decentralized Exchange

*Final-year project report*

> **Educational / research use only.** This system does not guarantee profit.
> Trading derivatives carries substantial risk of loss. See the Ethical
> Disclaimer (§13).

---

## 1. Abstract

This project remodels an existing machine-learning crypto-trading bot — originally
built for the Bitget centralized exchange — into a risk-managed automated trader
for **Hyperliquid**, a high-performance decentralized perpetual-futures exchange.
The execution layer is rebuilt on the **official Hyperliquid Python SDK** behind a
clean, venue-neutral *adapter* architecture, while the existing deep-learning
signal pipeline, backtesting, and multi-layer risk controls are preserved. A
central focus is **safety**: real-money trading is disabled by default and can
only be enabled through an explicit, multi-condition guardrail. The system is
structured for reproducible deployment on a dedicated AWS EC2 instance using
testnet and paper-trading modes for demonstration.

## 2. Problem statement

Centralized exchanges introduce custodial risk, opaque execution, and account
restrictions. Decentralized perpetual exchanges such as Hyperliquid offer
non-custodial trading with on-chain settlement and an **agent-wallet** model that
separates *signing* authority from *withdrawal* authority. However, automating
trading on a DEX safely requires: (a) correct use of the venue's native SDK and
order/sizing rules; (b) strict separation of secrets; and (c) guardrails that make
accidental real-money trading effectively impossible during development. This
project addresses how to migrate an existing AI trading system to Hyperliquid
**without** discarding its proven strategy, risk, and operational tooling, and
**without** weakening safety.

## 3. Objectives

1. Replace the Bitget execution layer with a Hyperliquid SDK implementation
   (official SDK, **not** CCXT).
2. Introduce a clean exchange-adapter interface (`ExchangeAdapter`) with
   interchangeable `BitgetAdapter` and `HyperliquidSDKAdapter` implementations.
3. Use the Hyperliquid **agent/API wallet** for signing; never require or store
   the main wallet's private key.
4. Make live trading **off by default** and gated behind explicit confirmations.
5. Preserve backtesting, the DL signal pipeline, risk controls, logging, and
   reporting.
6. Prepare the project for clean deployment on a **new** AWS EC2 instance,
   leaving the existing Bitget production bot untouched.

## 4. System architecture

```
            ┌──────────────────────────────────────────────────────────┐
            │                    Signal generation                       │
 market ───▶│  data.py (public OHLCV, ccxt)  →  features.py / ml_dl/*    │
 data       │  tools/live_writer.py  →  logs/live_signals.csv            │
            └──────────────────────────────────────────────────────────┘
                                   │  (CSV: ts,symbol,px,p_meta,…)
                                   ▼
            ┌──────────────────────────────────────────────────────────┐
            │                   tools/live_executor.py                   │
            │  risk gates (whitelist, threshold, RV, concurrency,        │
            │  cooldown, bias-lock, portfolio cap, TP/SL)                │
            │  + v2/risk_controls.py (time-stop, daily loss/DD, pause)   │
            └──────────────────────────────────────────────────────────┘
                                   │  create_market_order / fetch_positions
                                   ▼
   runtime/guardrails.resolve_trading_mode()  ──▶  PAPER | TESTNET_LIVE | MAINNET_LIVE
                                   │
                                   ▼
            exchanges/factory.make_adapter(EXCHANGE)
                       ┌───────────────┴────────────────┐
                       ▼                                 ▼
        exchanges/bitget_adapter.py        exchanges/hyperliquid_adapter.py
            (ccxt, legacy)                   (official Hyperliquid SDK)
```

The executor depends only on the small `ExchangeAdapter` interface, so the same
1,600-line trading loop runs unchanged on either venue. The **factory** picks the
adapter by the `EXCHANGE` env var; the **guardrail** decides whether real orders
are placed.

## 5. Module descriptions

| Module | Responsibility |
|---|---|
| `exchanges/types.py` | Shared `Position` / `OrderResult` data types (stdlib only). |
| `exchanges/base.py` | `ExchangeAdapter` abstract interface. |
| `exchanges/bitget_adapter.py` | Legacy Bitget execution (ccxt), extracted verbatim from the old `Broker`. |
| `exchanges/hyperliquid_adapter.py` | Hyperliquid execution via the official SDK; pure helpers for sizing/symbol/response parsing. |
| `exchanges/factory.py` | Selects the adapter by venue; lazy imports keep optional deps isolated. |
| `runtime/settings.py` | Typed, redaction-safe view of configuration. |
| `runtime/guardrails.py` | Single authority for the real-money decision; safe-by-default. |
| `tools/live_writer.py` | DL-ensemble signal generation → `logs/live_signals.csv` (unchanged). |
| `tools/live_executor.py` | Risk gates + order routing via the adapter (refactored). |
| `v2/risk_controls.py` | Time-stop, daily SL/loss/drawdown limits, pause-file kill switch (unchanged). |
| `ml_dl/*`, `features.py`, `data.py` | Feature pipeline + models (unchanged; public OHLCV). |

## 6. Algorithm flow (per executor tick)

1. Poll supervisor command channel (signed); apply pause/resume/risk-mode.
2. Roll daily files; update adaptive threshold; write state snapshot.
3. Read newest signal per symbol from `live_signals.csv`.
4. For each signal: reject bad price → check **TP/SL** exit → **V2 time-stop**
   exit → honor pauses / **V2 entry blocks** → side filter → threshold gate →
   volatility guard → concurrency caps → scale-in / flip / fresh entry, each
   subject to cooldown, duplicate-fill guard, bias-lock, and the portfolio cap.
5. Sizing converts a USDT notional to base quantity; the adapter rounds to the
   venue's precision and rejects sub-minimum orders.
6. Orders, fills, and closes are written to per-day CSVs; closes feed the V2
   daily counters.

## 7. API integration explanation

Market data (features) is currently fetched as **public OHLCV** via ccxt
(`data.py`), which is venue-neutral and requires no keys. Order **execution**,
**position**, and **price** queries for Hyperliquid go through the official SDK
(§8). The two concerns are deliberately decoupled: Phase 1 changes only execution,
so the trained models keep receiving the exact feature distribution they were
trained on (avoiding retraining/breakage).

## 8. Hyperliquid SDK explanation

The adapter uses `hyperliquid-python-sdk`:

- `Info(base_url, skip_ws=True)` — read `meta()` (asset universe + `szDecimals`),
  `user_state(address)` (positions/margin), and `all_mids()` (prices).
- `Exchange(wallet, base_url, account_address=...)` — `market_open` / `market_close`
  / `update_leverage` for execution.
- `eth_account.Account.from_key(agent_key)` builds the signer.

**Wallet model (key safety):**
- `HL_ACCOUNT_ADDRESS` — the **main wallet's public address**, passed as
  `account_address`; it is the account whose funds/positions are traded.
- `HL_AGENT_PRIVATE_KEY` — an **agent/API wallet** key used only to sign. An
  approved Hyperliquid agent **cannot withdraw**. The main wallet's private key is
  never used, requested, or stored.

**Venue specifics handled in the adapter:** shorthand→coin mapping
(`BTCUSDT`→`BTC`, `1000PEPEUSDT`→`kPEPE`); size flooring to `szDecimals`;
rejection of sub-precision/min-notional orders; and parsing of the SDK's
`response.data.statuses` (filled / resting / error).

## 9. Cloud deployment explanation

The project deploys to a **new, separate** AWS EC2 instance (the old Bitget
instance is untouched). Configuration is environment-driven (`.env`, optionally
SSM Parameter Store / Secrets Manager). systemd units (`hl-writer`,
`hl-executor`, `hl-trading-bot.target`) run the stack with `Restart=always`,
default to **paper/testnet**, and never pass `--live`. Full steps:
[deploy/aws/AWS_EC2_SETUP.md](../deploy/aws/AWS_EC2_SETUP.md).

## 10. Risk management explanation

Three layers, all preserved from the original system and unaffected by the venue
swap (full inventory: [docs/SAFETY_CONTROLS.md](SAFETY_CONTROLS.md)):

- **Signal-side:** OOD feature guard, model-agreement gate, calibration,
  confidence threshold.
- **Executor-side (V1):** symbol whitelist, threshold + volatility guards,
  concurrency caps, cooldown, duplicate-fill guard, side-bias lock, portfolio
  exposure cap, TP/SL with realistic paper fees + slippage.
- **V2 controls:** wall-clock time-stop, daily stop-loss budget, daily loss and
  drawdown limits, and a file-based **kill switch** (`run/V2_PAUSE`) that blocks
  new entries within one poll.

On top of these sits the new **live-trading guardrail** (§ below), which governs
whether *any* real order can be placed.

### Live-trading guardrail
`runtime/guardrails.resolve_trading_mode()` resolves to `PAPER` unless **every**
condition holds for mainnet: `LIVE_TRADING=true`, `PAPER_TRADING=false`,
`ENVIRONMENT=production`, mainnet selected (`HL_TESTNET=false`),
`CONFIRM_LIVE_TRADING=I_UNDERSTAND_LIVE_TRADING`, and valid credentials. Any
missing item forces paper mode with an explicit, secret-free reason in the log.

## 11. Limitations

- **TP/SL are executor-simulated**, not native venue trigger orders — they depend
  on executor uptime and the writer continuing to emit a symbol.
- **Market data is still ccxt public OHLCV** (Phase 1 scope), not yet sourced from
  Hyperliquid (see §12).
- Symbol mapping covers common assets and known 1000x aliases; exotic listings may
  need manual mapping.
- The daily loss/drawdown limits are day-scoped approximations, not account-equity
  drawdown enforcement.
- Hyperliquid mainnet trading requires a pre-approved agent wallet (manual UI
  step) and respects a ~$10 minimum order value.

## 12. Future scope

- **Phase 2 — market data on Hyperliquid:** migrate the feature pipeline to the
  Hyperliquid **Info API** (`candles_snapshot`, `all_mids`, L2 book) so data and
  execution share one venue. Deferred deliberately in Phase 1 to avoid retraining
  and to protect the existing model's input distribution.
- Native Hyperliquid trigger (TP/SL) and TWAP orders.
- WebSocket-driven fills and position events (replace polling).
- Account-equity drawdown enforcement; volatility-scaled position sizing.
- A Bitget-vs-Hyperliquid execution-quality comparison study (the adapter layer
  makes this a configuration change).

## 13. Ethical disclaimer

This software is provided for **educational and research purposes only**. It is
**not** financial advice and makes **no guarantee of profit**. Trading
cryptocurrency derivatives can result in the **total loss** of capital. The
authors accept no liability for any losses. Use only funds you can afford to lose,
comply with all applicable laws and exchange terms, and prefer **testnet and paper
trading** for any demonstration. Private keys and API secrets must never be shared,
committed to version control, or exposed in logs, reports, or screenshots.

---

## Appendix A — Exchange-adapter strategy & legacy Bitget inventory

**Hyperliquid is the primary execution adapter for this project.** Order
execution, position queries, and live prices flow through
`exchanges/hyperliquid_adapter.py` (official Hyperliquid SDK) selected by
`EXCHANGE=hyperliquid`. **Bitget support is retained only as a legacy adapter and
comparison layer** (`EXCHANGE=bitget`); it is *not* the primary path. Keeping
`BitgetAdapter` behind the same `ExchangeAdapter` interface is deliberate: it
demonstrates **modular, swappable exchange-adapter design** and enables a
centralized-vs-decentralized (Bitget vs Hyperliquid) architecture comparison
without touching the strategy, risk, or executor code.

The Bitget-named / Bitget-specific files are **kept on purpose** and fall into
these categories (none are the primary Hyperliquid execution path):

| File(s) | Category | Why kept |
|---|---|---|
| `exchanges/bitget_adapter.py` | **Legacy adapter + backward compatibility** | Selectable via `EXCHANGE=bitget`; the reference implementation the Hyperliquid adapter is modelled against. |
| `trade_multi_bitget.py`, `trade.py`, `trade_multi.py`, `exchange.py`, `multi_exchange.py` | **Comparison with centralized-exchange architecture** | Original Bitget/ccxt execution & routing scripts; preserved for history and CEX-vs-DEX comparison. |
| `risk_engine.py`, `router.py`, `async_scanner.py` | Legacy standalone modules | Earlier CEX-oriented risk/routing experiments; not on the live path. |
| `record_l2_bitget_rest.py`, `make_tier1_demo.py`, `backtest_day3.py`, `feature_pipe_adapter.py`, `universe.py`, `model.py` | Legacy research/data tooling | Historical analysis utilities. |
| `data.py`, `ml_dl/dl_infer.py`, `tier2/*`, `features/*` | **Market-data layer (Phase 1 retained, exchange-agnostic)** | Public OHLCV via ccxt feeds the feature pipeline; migrating this to the Hyperliquid Info API is Phase 2 (see §12). |

These files are **not deleted or renamed** — they document the migration path and
support the modular-design and comparison narrative of the dissertation.

## Appendix B — Monitoring & control plane (optional)

The project ships optional monitoring and control-plane tooling. **Validation
status is stated explicitly so nothing is over-claimed.**

### B.1 Dashboards
- **Static HTML dashboard — `tools/v2_dashboard.py`** (stdlib-only, read-only on
  `logs/`). Generates a single self-contained `reports/dashboard.html` with:
  executor heartbeat age + mode, open positions, supervisor pause state, V2 risk
  snapshot + block reason, daily realized-PnL table, a cumulative PnL SVG curve,
  exit-reason breakdown, and the last N closed trades.
  - Run: `python tools/v2_dashboard.py --days 14 --last-trades 30`
  - Reads: `logs/heartbeat.json`, `logs/executor_state.json`,
    `logs/v2_risk_state.json`, `logs/trades_closed_*.csv`.
  - **Covered by the pytest suite** (`tests/test_v2_dashboard.py`).
- **Live Flask dashboard — `tools/dashboard.py`** (served on a local port; start
  via `tools/start_dashboard.ps1 -Port 8787`). Useful for interactive monitoring.
  *Not covered by the automated test suite* — treat as a convenience tool.

**Screenshots to collect for submission:** (1) the generated
`reports/dashboard.html` (status cards + PnL curve), (2) a paper-mode
`logs/trades_paper_YYYYMMDD.csv` excerpt, (3) the guardrail startup line showing
`trading_mode=PAPER`, (4) the kill-switch in action (a `reason=v2_pause_file`
SKIP line). Never include `.env`, private keys, or addresses in screenshots.

### B.2 Telegram (optional)
- `tools/telegram_notifier.py` — read-only alerts (heartbeat-stale, process-down,
  recovery, daily PnL summary).
- `tools/telegram_controller.py` — relays `/status /pause /resume /reduce_risk
  /conservative /logs /health` to the Supervisor API over authenticated HTTP.
- **Validation status:** implemented with documented commands; **not part of the
  automated pytest run** and not validated here. Treat as optional.

### B.3 Supervisor control plane (optional)
- `supervisor/` package: an HMAC-signed command channel (`pause`, `resume`,
  `reduce_risk`, `conservative_mode`, `emergency_stop`) with two-step human
  approval for risky actions, an append-only audit trail, and JWT/rate-limited
  API auth. The executor verifies each command's HMAC before applying it
  (`poll_supervisor_cmd` in `tools/live_executor.py`).
- **Audit/safety value:** signed, auditable, least-privilege operator control;
  combined with the file-based kill switch it forms the kill-switch ladder in
  `docs/SAFETY_CONTROLS.md` §6.
- **Validation status:** the **pause-file kill switch and V2 daily risk limits are
  covered by pytest** (`tests/test_v2_risk_controls.py`). The supervisor HTTP API
  has a standalone script (`tools/test_supervisor.py`) but is **not** part of the
  default `pytest` run — present it as an implemented, optional control plane, not
  a fully test-validated one.

### B.4 Pause / resume / emergency stop
- Fastest kill switch: `touch run/V2_PAUSE` blocks new entries within one poll
  (~3 s); delete the file to resume. **Test-covered.**
- Signed supervisor `pause` / `emergency_stop` (optional, audited).
- Full stop on EC2: `sudo systemctl stop hl-trading-bot.target`.
