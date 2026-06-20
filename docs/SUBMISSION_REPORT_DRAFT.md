# An AI Trading Bot with Defense-in-Depth: from a Live Incident History to a Safety-First V2 Architecture

**DRAFT — sections 1–4 and 7 are substantially complete; sections 5, 6 and 8 contain
`[EVIDENCE: …]` placeholders that are filled from `reports/evidence/` during the
13–24 June freeze window. Each placeholder names the exact artifact that produces it.**

Submission date: 25 June 2026. Repository branch: `bot-v2-architecture` (deployed at `bbdc82f`).

---

## 1. Problem statement & system overview

The project goal is an autonomous trading system for Bitget USDT-margined perpetual
futures that is safe to operate unattended on a small account, with machine-learning
signal generation and — more importantly — engineering controls that make every
decision auditable and every failure mode either impossible or loud.

The contribution of this work is **not** a claim of profitability. It is the
engineering process: a deployed system that survived a real production incident
history, the diagnostic tooling that found the root causes, and a V2 architecture
designed so the worst bug class cannot recur. All trading evidence in this report is
from **paper mode** with realistic fees and slippage; live trading is explicitly out
of scope (§7).

**System at a glance.** Two cooperating services on AWS EC2 under systemd:

- a **signal writer** (`tools/live_writer.py`): fetches 1-minute candles for six
  symbols (BTC, ETH, DOGE, XRP, SOL, BNB), computes 26 stationary price-action and
  volume features plus a symbol-id channel, scores them through a four-model deep
  ensemble (TCN, LSTM, Transformer, AdvancedTransformer) with per-model temperature
  calibration and an agreement gate, and writes one signal row per symbol per tick;
- a **trade executor** (`tools/live_executor.py`): polls the signal file and manages
  positions through ~20 layered guards (thresholds, volatility caps, cooldowns,
  flip-confirmation, exposure caps, side-bias lock, supervisor pause), with paper
  fills modeled adversely (taker fees both sides + slippage).

Monitoring (heartbeats, an 8-section health check, daily evidence bundles, a static
dashboard), validation tools (parity checks, counterfactual exit simulation,
ensemble-variant replay) and a signed-command supervisor API complete the system.
The architecture diagram is reproduced in §3.

## 2. The V1 journey: what production taught us

This section is the core of the report. Each incident below happened on the deployed
system; each fix is a commit in the repository; each diagnosis used tooling built for
that purpose. The pattern that emerges — *silent coincidences must be replaced by
explicit contracts* — is what V2 (§3) institutionalizes.

### 2.1 The flagship incident: feature-set skew (dimension-match ≠ feature-match)

**Symptom.** All four ensemble models saturated to ~100 % LONG. Every shape check
passed. Profit factor < 1 looked like a strategy failure.

**Diagnosis.** A purpose-built tool (`tools/diagnose_features.py`) compared the live
scaled feature distribution against the deployed scaler's training statistics. The
result: bounded features were arriving **hundreds of standard deviations**
out-of-distribution.

**Root cause.** The deployed scaler was trained on a 26-feature set *including* a
`gap` feature, with a symbol-id channel (26 + 1 = 27 inputs). The feature code had
since drifted to 27 named features (dropping `gap`, adding two trend anchors) with no
symbol-id (27 + 0 = 27 inputs). **Both pipelines were 27 wide.** Every dimension
check passed while the columns held *different features* — and because the canonical
order was alphabetical, removing one feature shifted every later column onto the
wrong scaler slot. The models were fed garbage that was numerically valid.

**Fixes** (commits `1870ecb`, `4bef365`, `8a831e8`): revert to the exact trained
feature set; make the symbol-id flag drive feature width with a fail-loud resolver
reconciled against the scaler's recorded width; add a startup scaler-dimension
assert; add a runtime **OOD guard** that forces `allow=0` and logs FATAL rather than
trade on off-distribution inputs.

**Lesson.** Dimension-match is not feature-match. The V2 answer is a hash-pinned
feature contract validated at model-load time (§3, post-submission).

### 2.2 The same incident's hidden accomplices

The investigation surfaced three more defects, each with its own fix:

- **Per-symbol inference bug** — the writer scored only the last symbol's feature
  window and copied that signal to every symbol (ETH literally traded on SOL's
  signal). Fixed with per-symbol windows and a persisted symbol-id map (`1870ecb`).
- **Zero-cost paper fills** — the paper executor charged no fees or slippage, so
  early profit factors were fiction. Fixed by applying taker fees on both sides and
  adverse slippage at all five fill sites (`1870ecb`). Every PnL number after this
  point is net of modeled costs.
- **Schema-locked logging dropped per-model columns**, blinding the diagnostics that
  would have caught the saturation earlier. Fixed in `afaac1b`, then extended with
  per-symbol-per-model probability logging.

### 2.3 Flip-churn: 131 of 132 exits were direction flips

Over the first four live-paper days, only **one** exit was a take-profit; 131 were
`FLIP_CLOSE` — the bot bleeding ~0.4 ¢ per flip in costs while signals oscillated.
Root cause was upstream (below), but the executor gained a flip-confirmation
requirement (N consecutive valid opposite signals before a flip executes) as a
structural damper. The V2 time-stop (§4) adds a holding-time bound; the counterfactual
exit simulator (`tools/sim_exits.py`) exists to choose its value from data.

### 2.4 The oscillating side-bias regime — and why daily ratios hid it

Daily long/short close ratios looked healthy (50–55 %). Signal-level monitoring told
the truth: the ensemble swung to **95–100 % one-sided for hours, then flipped
wholesale**, averaging out by day-end. Defenses shipped: a side-bias lock that
suspends entries (not exits) at ≥95 % one-sidedness, with a minimum-sample rule to
clear; and the agree-gate fix below. The deeper diagnosis — a confirmed
**train-on-5m / serve-on-1m timeframe mismatch** compressing every feature by ~√5
against the scaler — is exactly the class of skew the V2 model registry refuses at
load time.

### 2.5 The agree-gate emitted fake shorts

When models disagreed, the gate's neutral output was centered to −0.5 — which the
executor read as an *allowed SHORT at 50 % confidence*. Disagreement was being traded
as conviction. Fixed in `fd8128b`: disagreement now emits FLAT (`allow=0`), and
zero-weighted models lose their gate vote.

### 2.6 The dead model gaming the vote

Production monitoring showed model TCN's output collapsed to a constant
(std ≈ 0.009, ~0.506 for every input) — useless, but still casting a perpetual
mild-bull vote in the agreement gate. It was demoted to zero weight (where, per
`fd8128b`, it no longer votes). The V2 model-health layer (§3, post-submission)
automates this: rolling std/saturation metrics that down-weight degenerate models
without human noticing first.

### 2.7 Defined-but-unenforced risk limits

`config/run.json` declared `MAX_HOLD_BARS=48` and `MAX_DD=0.05`; an audit found
**nothing enforced either**. This motivated the V2 risk controls (§4): the first
actual enforcement of holding-time bounds plus daily SL-count/loss/drawdown pauses —
shipped flag-gated and off by default so the evidence sample stays clean.

### 2.8 Configuration drift

Local `.env`, EC2 `.env` and systemd CLI flags diverged (the CLI flag silently
overrides `.env` for live/paper mode). Process answer now: a documented runbook with
`systemctl cat` checks and deploy markers recording every change. Code answer
post-submission: one effective-config snapshot hashed and logged at startup.

## 3. V2 architecture

*[Reproduce here: the 8-layer summary table and Mermaid diagram from
`docs/V2_ARCHITECTURE.md` §§2–3, and the migration plan from §5.]*

Design principles, proven by construction in this submission window:

1. **V1 untouched by default** — V2 risk hooks total six call sites in one file, each
   `None`-guarded and exception-isolated; with flags off, the trading path is
   V1-identical (verified in production, §6).
2. **Everything reversible** — flag → restart; pause-file → instant; `git revert` of
   one commit removes the hooks entirely.
3. **Contracts over coincidence** — the post-submission registry refuses to load a
   model whose feature-contract hash or timeframe disagrees with the serving config,
   converting §2.1 and §2.4 from silent failures into refused starts.
4. **Evidence first** — daily machine-generated bundles, not hand-collected numbers.

## 4. Safety controls

*[Reproduce here: the full inventory table from `docs/SAFETY_CONTROLS.md` — ~20 V1
guards + 6 V2 controls — and the 5-rung kill-switch ladder.]*

The V2 additions in brief: a wall-clock **time-stop** (`EXIT_TIME`) subordinate to
TP/SL; a **daily stop-loss budget**, **daily loss limit** and **daily drawdown
pause** that block entries (never exits) until UTC midnight; and a **pause file**
that halts entries within one poll cycle with no restart. Counters rebuild from the
append-only closed-trade CSV at startup, so a restart cannot reset the day's budget.
All default off; all shipped with unit tests (§6).

## 5. Results & evidence  `[FILLED FROM FREEZE-WINDOW DATA]`

The clean evidence session began 2026-06-10 after the §2.1 fixes (deploy-marked) and
was **frozen** 2026-06-13 → submission: no flag, threshold, weight, model or code
changes (freeze proof: `reports/evidence/freeze/`).

- Headline sample: `[EVIDENCE: reports/evidence/index.json — final trades, win rate,
  net PnL, PF, max DD; state n explicitly and treat PF as descriptive, not
  significant, if n < 100]`
- Cumulative PnL figure: `[EVIDENCE: dashboard.html screenshot, 24 June]`
- Exit-reason mix: `[EVIDENCE: final summary.json exit_reasons — contrast with the
  131/132 FLIP_CLOSE pathology of §2.3]`
- Model-health findings (the monitoring layer working): per-model side bias and
  dispersion over the window — at freeze start: ADV 92 % LONG (mean 0.776), LSTM
  ~55 %, TX 25 % (bearish), TCN dead/zero-weight; ensemble output split LONG 49 % /
  SHORT 26 % / **suppressed 25 %** (disagreement-as-flat working). Per-symbol
  asymmetry: ETH 63 % LONG vs SOL 53 % suppressed.
  `[EVIDENCE: diagnostics/bias_*.txt, ensemble_*.txt — quote first and last snapshot]`
- Directional proxy after costs: ETH 49.9 % hit / −3.96 bps, SOL 54.4 % / −0.64 bps
  at freeze start — reported as measured; the point is the system measures honestly
  post §2.2. `[EVIDENCE: final diagnostics snapshot]`
- **Counterfactual exit analysis (n = 38, 2026-06-13).** Read-only replay of the
  recorded trades through `tools/sim_exits.py --sessions current`:
  - Baseline (as-traded exits): **PF 0.339, net −1.6969 USDT, max DD −1.7215 USDT**.
  - Best cell in the entire TP/SL/time-stop grid (TP 0.30 %, SL 0.30 %, no time-stop):
    **PF 0.388, net −1.1940 USDT**.

  At n = 38, exit replay showed that even the best counterfactual TP/SL cell remained
  below PF 1.0. This ruled out simple exit geometry as the primary fix and justified
  freezing the strategy instead of overfitting a small sample. The improvement from
  tuning (PF 0.339 → 0.388) is real but immaterial — it reduces the loss without
  crossing into profit — so the dominant cause lies upstream in signal quality / entry
  direction, not in how positions are closed. This is a deliberate negative result
  produced by the system's own validation tooling. Full reasoning: Appendix F,
  `reports/evidence/freeze/decision_20260613_n38.txt`.
  `[EVIDENCE: final diagnostics/exits_*.txt if n grows materially before 24 June]`
- Exit-policy decision (reached early at n = 38, satisfying the planned 19–20 June
  point): no exit or signal-side parameter changed; window kept frozen.
  `[EVIDENCE: reports/evidence/freeze/decision_20260613_n38.txt]`

## 6. Validation & testing  `[TRANSCRIPTS ATTACHED]`

- **42 unit tests** (risk-control state machine incl. UTC rollover and crash-restart
  counter rebuild; evidence exporter metric math; archiver dry-run safety; dashboard
  rendering/escaping): `pytest tests/ -q` → 42 passed, on the dev box and on EC2.
- **Five system validators green on the deployed EC2 box** (2026-06-12):
  `test_fixes_123.py` (offline regression of the §2.1/§2.2 fixes), `parity_check.py`
  (live per-symbol inference parity), `test_gate_fix.py`, `test_sim_exits.py`, plus
  the 8-section `bot_health_check.py`. `[EVIDENCE: transcripts/*.txt]`
- **Shadow-deploy parity**: V2 deployed with flags off; expected and observed: zero
  `reason=v2_*` skips, zero `EXIT_TIME` exits, one inert `v2_risk` status line —
  i.e., a deployed change with a measured behavioral delta of zero until explicitly
  enabled. `[EVIDENCE: freeze proof grep counters]`

## 7. Limitations — stated plainly

1. **No profitability claim; alpha is unproven.** The clean paper sample is small
   (n = 16 at freeze start, n = 38 at the 2026-06-13 exit review; final n in §5) and
   the strategy is loss-making over it. Crucially, the counterfactual analysis (§5)
   shows the loss is **not** an exit-geometry problem: the best TP/SL/time-stop cell
   in the grid still sits below PF 1.0, so the deficit is most likely signal-side
   (entry direction / model calibration), which exit tuning cannot repair. We
   therefore make no claim of edge, and — following the system's own n ≥ 30 standard
   for ranking exit policies — we do not tune parameters on this small sample.
2. **Paper mode only**, single exchange, six symbols. Live trading requires the
   post-submission gates (reconciliation harness + human sign-off) and is not
   requested here.
3. **Known model pathologies are managed, not solved**: ADV remains LONG-skewed, TCN
   is dead (zero-weighted), and the 5m/1m training-serving mismatch is fixed by
   checklist, not yet by code contract. The pending retrain is deliberately
   **post-submission** — retraining mid-window would reset the evidence sample.
4. **Time-stop is signal-driven**: a symbol the writer stops emitting only exits via
   TP/SL or restart. Documented; price-polling sweep is roadmap.
5. **Costs are modeled** (5 bps fee + 2 bps slippage per side), not measured from
   live fills.

## 8. Roadmap  `[CONDENSE FROM docs/V2_ROADMAP.md]`

The n = 38 finding (§5/§7) sets the post-submission priority: because the loss is
signal-side rather than exit-side, the next work targets the **signal pipeline**, not
the executor.

- **Phase 1 — feature/signal contracts.** Hash-pinned feature sets and a versioned
  signal schema so the §2.1 skew class becomes a refused load, not a silent collapse.
- **Phase 2 — model registry, calibration, and a health-gated retrain.** An explicit
  registry validating feature-hash + timeframe at load (closes §2.4); calibration
  monitoring and per-model health metrics that auto-down-weight degenerate models
  (closes §2.6); then the deliberately deferred **retrain** on a coherent
  train/serve timeframe — the direct attempt at the signal-quality deficit measured
  here. No retrain happens mid-window precisely so this evidence sample stays clean.
- **Phase 3 — risk-engine consolidation** (account-level max-DD, vol-aware sizing).
- **Phase 4 — purged/embargoed walk-forward CV + live-vs-backtest reconciliation.**
  This is where **per-symbol validation and threshold/regime work** is admitted:
  the ETH/SOL asymmetry in §5 is a hypothesis to be tested in the CV harness, not a
  change to ship on a 38-trade sample.
- **Phase 5 — supervisor read-only (Trady) integration, then the signed write path.**

Live trading is gated behind Phase 4 reconciliation plus human approval — by design,
not by omission. Until a retrained, contract-pinned, walk-forward-validated model
demonstrates positive expectancy out-of-sample, the correct posture is the one taken
here: stay frozen and keep measuring.

## Appendices

- A. Operations runbook (`docs/OPERATIONS_RUNBOOK.md`)
- B. Full safety-controls inventory (`docs/SAFETY_CONTROLS.md`)
- C. Annotated git history (`transcripts/git_log_*.txt` + §2 commit references)
- D. One complete daily evidence bundle (`reports/evidence/<best day>/summary.{json,md}`)
- E. Test and validator transcripts (`reports/evidence/transcripts/`)
- F. Freeze proof and decision record (`reports/evidence/freeze/`)
