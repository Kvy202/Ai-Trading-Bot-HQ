# V2 Roadmap

Companion to [V2_ARCHITECTURE.md](V2_ARCHITECTURE.md). Dates are 2026.

---

## Phase 0 — pre-submission (now → 25 June)

Goal: ship the safe incremental layer, accumulate auditable paper evidence, write the report.

| Date | Milestone | Acceptance criteria |
|---|---|---|
| Jun 12–13 | Docs + `.env.v2.example` committed | 5 docs render on GitHub (incl. Mermaid); example file lists every `V2_*` flag with default 0 |
| Jun 13–14 | `v2/risk_controls.py` + unit tests | `pytest tests/test_v2_risk_controls.py` green; module imports with stdlib only |
| Jun 14–15 | Executor wiring (6 hook sites) | `tools/test_fixes_123.py` → `RESULT: PASS` and `sanity_test.py` clean **after** the edit; with no `V2_*` flags the only new log line is the v2-disabled notice |
| Jun 15–17 | Monitoring tools + their tests | Evidence bundle generated from real local logs; dashboard.html opens offline; archive dry-run lists only dated files older than cutoff |
| Jun 18–19 | Deploy to EC2 **flags off** (shadow) | Service restarts clean; `v2_risk: disabled` in `live_executor.out`; trading log diff shows V1-identical behavior for ≥24 h |
| Jun 20 | Enable flags on paper + deploy marker | `EXIT_TIME` / `SKIP reason=v2_*` lines appear only per configuration; `logs/v2_risk_state.json` updating |
| Jun 20–24 | Evidence accumulation + report writing | Daily `reports/evidence/YYYYMMDD/` bundles; dashboard screenshots; report drafted from SUBMISSION_REPORT_OUTLINE.md |
| Jun 24 | Freeze | No code changes after this point; final evidence export |
| Jun 25 | **Submission** | — |

Slack: ~2 days built in. If slipping, cut in this order: `v2_log_archive.py` (+tests) → dashboard extras (keep tables, drop SVG) → DD-pause flag (keep loss limit).

### Explicit non-goals before 25 June
No live trading. No model/feature/artifact changes. No new symbols. No threshold retuning.
No writer changes. No supervisor/tier2 changes. No executor rewrite. Nothing that resets
the paper evidence sample.

---

## Phase 1 — contracts (post-submission, ~1 week)

- `v2/contracts/feature_contract.py`: versioned FeatureSet (id, ordered names, formula
  hashes, timeframe, warm-up); training stamps the hash into metadata; serving validates.
- `v2/contracts/signal_schema.py`: named-column, versioned signal records; writer stamps
  schema version; executor reads by name. Retires the "first 8 columns in this exact
  order" coupling.
- Effective-config snapshot + hash logged at startup (writer + executor) — kills .env drift.

Gate to exit phase: parity test proving contract-pinned features == `features.py` output
bit-for-bit on fixed input.

## Phase 2 — model registry & calibration (~2 weeks)

- `v2/models/registry.py`: immutable versioned bundles; load-time validation of feature
  hash + timeframe + scaler dims (refuse, don't warn).
- The pending retrain (drop `gap`, re-add `trend_50/trend_200`, train timeframe ==
  serve timeframe) lands **through** the registry as bundle v2.
- `v2/models/health.py`: rolling per-model std / saturation / agreement; dead models
  auto-excluded from the agree-gate vote.
- `v2/models/calibration.py`: reliability monitoring; shadow-model parallel scoring with
  promotion gate (N days of healthier metrics + reconciliation pass).

## Phase 3 — risk engine consolidation (~1 week)

- `v2/risk/engine.py` absorbs V1 guards + `risk_controls.py` behind one policy interface
  with unit tests; adds account-level max-DD (finally enforcing run.json `MAX_DD`) and
  volatility-aware sizing.
- Executor main loop decomposed into intake → policy → execution stages along the V2 hook
  seam; behavior-diff harness proves equivalence before cutover.

## Phase 4 — backtest & reconciliation harness (~2 weeks)

- `v2/backtest/walkforward.py`: purged/embargoed walk-forward CV for model selection.
- `v2/backtest/reconcile.py`: replay each live day from recorded signals; assert
  trade-count/PnL agreement within tolerance; wire into the promotion gate.
- Per-symbol thresholds + regime filters enter **only** through this harness.

## Phase 5 — supervisor & alerting (~1 week)

- Trady read-only: dashboard/evidence/health served via the existing Flask API.
- Alerting: telegram_bot.py wired to heartbeat-age, OOD, bias-lock, V2-block events.
- Then (human-approved) Trady write path over the existing signed command channel.

## Dependency graph

```
Phase 0 (safety+evidence)
   └─► Phase 1 (contracts) ─► Phase 2 (registry+retrain) ─► Phase 4 (backtest gates)
                                   │                              │
                                   └────────► Phase 3 (risk engine) ◄┘
Phase 5 (supervisor/alerting) — parallel to 2–4, after 1
Live-trading decision: only after Phases 1–4 complete + reconciliation green + human sign-off
```
