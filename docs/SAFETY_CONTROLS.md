# Safety Controls Inventory

Every guard in the system — V1 (existing) and V2 (new) — with its default, its knob, and
how to disable it. "Blocks" column: **E** = blocks new entries, **X** = forces/affects exits,
**S** = blocks the whole signal (writer side), **P** = process-level.

---

## 1. Writer-side (signal generation) — V1

| Guard | Default | Knob | Code | Blocks | Disable |
|---|---|---|---|---|---|
| OOD feature guard (off-distribution inputs → `allow=0`, FATAL log) | active | `DL_OOD_MAX_FEATURES`, `DL_OOD_MAX_Z` | `ml_dl/dl_ensemble.feature_ood` + writer first-tick check | S | unset thresholds (don't) |
| Scaler-dim assert (writer refuses to start on feature-width mismatch) | active | — (structural) | `tools/live_writer.py` startup | P | none — fail-loud by design |
| Agree-gate, disagreement-as-FLAT (≥2 models must agree; disagreement = neutral, never a fake short) | `DL_MIN_AGREE=2` | `DL_MIN_AGREE` | `ml_dl/dl_ensemble.predict_ensemble` | S | set to 1 (don't) |
| Calibration (per-model temperature + bias, logit space) | EC2: temps 2.5/2.0/2.0/1.0 | `DL_TEMP_*`, `DL_BIAS_*` | `predict_ensemble` | — | set to 1.0/0.0 |
| Confidence threshold | `DL_P_LONG` (EC2 value) | `DL_P_LONG` | writer | S | lower (don't blindly) |
| Side restriction | unset | `DL_ALLOW_ONLY` | writer | S | unset |

## 2. Executor-side entry gates — V1

| Guard | Default | Knob | Code | Blocks | Disable |
|---|---|---|---|---|---|
| Symbol whitelist | from env | `EXEC_SYMBOL_WHITELIST` / `SYMBOL_WHITELIST` | `symbol_allowed()` | E | empty list = allow all |
| Bad-price skip (`px<=0`/non-finite) | active | — | main loop | E | none |
| Supervisor pause | off | signed `pause` command | `poll_supervisor_cmd` → `sv_state.paused` | E (exits still run) | signed `resume` |
| Side filter | `both` | `--sides` | `side_allowed()` | E | `both` |
| Threshold gate (+ optional adaptive) | `--plong` / `EXEC_PLONG` | `--plong --pmode --adaptive --respect-writer-thr` | `threshold_pass()` | E | lower threshold |
| Volatility guard | `EXEC_RV_MAX` | `EXEC_RV_MAX` / `--rv-max` | main loop | E | raise (EC2 currently 100 = off) |
| Concurrency caps | `MAX_CONCURRENT=1` | `--one-position`, `--max-symbols` | main loop | E | raise |
| Cooldown between fills | `EXEC_COOLDOWN_SEC=300` | env / `--cooldown` | main loop | E | 0 |
| Flip-confirm ticks (N consecutive valid opposite signals before flip) | `EXEC_FLIP_CONFIRM_TICKS=20` | env / `--flip-confirm-ticks` | main loop `_flip_pending` | E/X | 0 |
| Duplicate-fill guard (same price within 2×cooldown) | active | — | `prices_close()` | E | none |
| Side-bias lock (≥95 % one-sided recent signals → suspend entries) | `EXEC_BIAS_GUARD=1` | env / `--bias-guard` | `check_side_bias()` | E (exits alive) | set 0 (don't) |
| Portfolio exposure cap (fresh entries AND scale-ins) | `MAX_PORTFOLIO_EXPOSURE_USDT=31` | env / `--max-portfolio-usdt` | `portfolio_exposure()` | E | raise |
| Per-trade notional sizing + minimums | `MAX_NOTIONAL_USDT=15`, `EXEC_MIN_NOTIONAL=1` | env / args | `qty_for()` | E | — |
| Supervisor risk-mode multiplier (reduced ×0.5 / conservative ×0.25 notional) | normal | signed commands | `_RISK_NOTIONAL_MULT` | E (sizing) | signed `resume` |

## 3. Executor-side exit & process guards — V1

| Guard | Default | Knob | Code | Blocks | Disable |
|---|---|---|---|---|---|
| TP / SL (trigger on mid, adverse fill) | `EXEC_TP_PCT=0.01`, `EXEC_SL_PCT` per EC2 | env / `--tp-pct --sl-pct` | `check_tp_sl()` | X | — |
| Paper fees + slippage at all five fill sites | `EXEC_FEE_BPS=5`, `EXEC_SLIPPAGE_BPS=2` | env / args | `net_pnl_on_close()`, `apply_slippage()` | — | 0 (don't — fake PF) |
| Restart position recovery + reconcile (live: exchange is truth) | `--restore-state` on | flag | `load_positions_from_state`, `reconcile_live_positions` | P | drop flag (don't) |
| Restart-close (restored position already past TP/SL → close immediately) | active (live) | — | main() restore block | X | none |
| Single-instance lock (stale after 900 s) | active | — | `single_instance_lock()` | P | none |
| Atomic state snapshot every tick | active | — | `write_state_snapshot()` | P | none |

## 4. Supervisor layer — V1

| Guard | Default | Knob | Code | Blocks | Disable |
|---|---|---|---|---|---|
| Signed command channel (HMAC-SHA256; pause/resume/reduce_risk/conservative_mode/emergency_stop) | armed | `SUPERVISOR_HMAC_SECRET` | `supervisor/` + executor `poll_supervisor_cmd` | E/P | — |
| Two-step human approval for risky commands (`resume_live`, `switch_to_live`, `increase_leverage`, `flatten_all`, `disable_safety`, …; TTL 300 s/120 s, single-use) | armed | API flow | `supervisor/approvals.py` | P | none — by design |
| Append-only audit trail | active | — | `logs/supervisor_audit.jsonl` | — | none |
| API auth (JWT + nonce/timestamp replay protection + rate limits) | armed | `SUPERVISOR_JWT_SECRET`, `SUPERVISOR_PORT` | `supervisor/auth.py`, `rate_limit.py` | P | — |

## 5. V2 risk controls — NEW (all **off by default**)

| Guard | Default | Knob | Code | Blocks | Disable |
|---|---|---|---|---|---|
| Master off-switch (executor runs pure V1) | `0` (V2 armed but inert) | `V2_RISK_DISABLED=1` | `v2/risk_controls.init_risk_controls` | — | n/a |
| **Time-stop** — close any position held ≥ N minutes (`EXIT_TIME`; TP/SL has priority; wall-clock based; scale-in does not refresh the clock) | `0` = off | `V2_TIME_STOP_MIN` | `v2/risk_controls.time_stop_due` + executor hook E4 | X | set 0 |
| **Daily SL-count limit** — after N `EXIT_SL*` closes in a UTC day, block new entries until midnight UTC | `0` = off | `V2_MAX_SL_PER_DAY` | `entry_block_reason()` + hook E5 | E (exits alive) | set 0 |
| **Daily loss limit** — block entries while realized day PnL ≤ −X USDT (profits can lift the block) | `0` = off | `V2_DAILY_LOSS_LIMIT_USDT` | same | E (exits alive) | set 0 |
| **Daily drawdown pause** — block entries while day PnL sits ≥ X USDT below its intraday peak | `0` = off | `V2_DAILY_DD_LIMIT_USDT` | same | E (exits alive) | set 0 |
| **Pause file** — file exists ⇒ entries blocked (works even with all numeric flags at 0) | armed, file absent | `V2_PAUSE_FILE` (default `run/V2_PAUSE`) | same | E (exits alive) | delete the file |

Shared semantics: V2 entry blocks sit at the same point as supervisor pause — TP/SL and
time-stop exits keep running; scale-ins, flips, and fresh entries are blocked (like
supervisor pause, a blocked flip means the position waits for TP/SL/time-stop).
Counters live in `logs/v2_risk_state.json` and are **rebuilt from
`logs/trades_closed_YYYYMMDD.csv` at startup** — restarting cannot reset today's SL count.
Every executor hook is try/except-guarded; a v2 failure logs and degrades to V1 behavior.

Known limitation: time-stop is signal-driven — it fires on the next signal for that
symbol. If the writer stops emitting a symbol, that position only exits via TP/SL or a
restart. (A price-polling sweep is post-submission.)

## 6. Kill-switch ladder (fastest → slowest)

1. `touch ~/bot/run/V2_PAUSE` — entries blocked within one poll (~3 s), no restart, no .env edit. Undo: `rm`.
2. Supervisor `pause` (signed) — same effect, audited; `emergency_stop` for hard stop.
3. `V2_RISK_DISABLED=1` or comment out `V2_*` in `.env` + `systemctl restart bot-executor` — removes V2 behavior entirely.
4. `git revert <wiring-commit>` + restart — removes V2 hooks from the code path.
5. `sudo systemctl stop trading-bot.target` — stops everything (writer + executor + proxy).

## 7. Configuration mapping note

`config/run.json` defines `MAX_HOLD_BARS=48` and `MAX_DD=0.05` but **nothing enforces
them** in the V1 executor. The V2 time-stop is the first enforcement of holding-time:
48 bars × 5 m = **240 minutes**, so `V2_TIME_STOP_MIN=240` matches the documented intent.
Account-level `MAX_DD` enforcement remains post-submission (the V2 daily loss/DD limits
are day-scoped approximations, not account-equity drawdown).

## 8. Live-trading guardrail — NEW (Hyperliquid remodel, **paper by default**)

| Guard | Default | Knob | Code | Blocks | Disable |
|---|---|---|---|---|---|
| Master live-trading decision (resolves to PAPER unless every confirmation is present) | `PAPER` | `LIVE_TRADING`, `PAPER_TRADING`, `ENVIRONMENT`, `HL_TESTNET`, `CONFIRM_LIVE_TRADING` | `runtime/guardrails.resolve_trading_mode` | P (no real orders) | set the full confirmation set |
| Typed real-money confirmation token | unset | `CONFIRM_LIVE_TRADING=I_UNDERSTAND_LIVE_TRADING` | same | P | by design |
| Credential presence/format check (Hyperliquid) | enforced | `HL_ACCOUNT_ADDRESS`, `HL_AGENT_PRIVATE_KEY` | `runtime/settings.Settings.has_hl_credentials` | P | provide valid creds |
| Secret redaction in logs/errors | active | — | `runtime/settings.redact` / `scrub` | — | none — by design |

Mainnet **real money** requires ALL of: `LIVE_TRADING=true`, `PAPER_TRADING=false`,
`ENVIRONMENT=production`, `HL_TESTNET=false`,
`CONFIRM_LIVE_TRADING=I_UNDERSTAND_LIVE_TRADING`, and valid credentials. Any missing
item forces PAPER with an itemised, secret-free reason. `--paper` always wins;
`--live` cannot bypass a failed guardrail. Testnet/sandbox live (no real money) is
allowed with credentials but without the confirmation token. The agent/API wallet
signs only and cannot withdraw; the main wallet private key is never used.
