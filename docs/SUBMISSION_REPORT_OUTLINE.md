# Submission Report — Outline (due 25 June 2026)

Working title: *An AI Trading Bot with Defense-in-Depth: from a Live Incident History to a
Safety-First V2 Architecture.* Target length 12–20 pages + appendices. Every claim cites
an artifact in the repo (commit sha, log excerpt, evidence bundle, test output).

---

## 1. Problem statement & system overview (1–2 pages)
- Goal: small-account crypto perp trading with an ML/DL signal stack, run safely on AWS.
- Constraints: paper-first, tiny notional, defense-in-depth, full auditability.
- One-paragraph system tour + the architecture diagram from `docs/V2_ARCHITECTURE.md` §3.
- Honest framing up front: the contribution is the **engineering process and safety
  architecture**, evaluated in paper mode; profitability is not claimed.

## 2. V1 journey — what production taught us (3–4 pages)
The strongest section: real incidents, root causes, and fixes, in time order.
- The 8-pathology register (`V2_ARCHITECTURE.md` §6), each with: symptom → diagnosis
  tooling → root cause → fix commit → verification.
- Deep-dive the flagship incident: feature-set skew where every dimension check passed
  while columns were different features → OOD inputs → all four models saturated LONG
  (cite commits `1870ecb`, `4bef365`, `8a831e8`,
  `tools/diagnose_features.py` output). Lesson: *dimension-match ≠ feature-match*.
- Flip-churn evidence (131/132 exits FLIP_CLOSE) and the countermeasures
  (flip-confirm ticks; fees/slippage realism; later: time-stop).
- The oscillating side-bias regime and why daily side ratios hid it.

## 3. V2 architecture (2–3 pages)
- The 8 layers, each "V1 has / new now / post-submission" (condense §2 of the architecture doc).
- Design principles: V1 untouched by default, flag-gated reversibility, failure isolation,
  contracts over coincidence, evidence first.
- Migration plan summary + the post-submission file tree.

## 4. Safety controls (1–2 pages)
- The full inventory table from `docs/SAFETY_CONTROLS.md` (V1's ~20 guards + V2's 6).
- The kill-switch ladder; the supervisor's signed-command + human-approval flow.
- V2 risk-control semantics: time-stop, daily SL-count/loss/DD pauses, pause file;
  counter rebuild from append-only CSVs (restart-proof).

## 5. Results & evidence (2–3 pages)
- Paper-trading record: daily evidence bundles (`reports/evidence/YYYYMMDD/summary.md`),
  cumulative PnL chart from `reports/dashboard.html` (screenshot), exit-reason mix
  before/after the fee-realism fix and after time-stop enablement.
- Counterfactual analysis: `tools/sim_exits.py` TP/SL/time-stop grid → how the chosen
  `V2_TIME_STOP_MIN` value was selected from data, not vibes.
- Ensemble diagnostics: `tools/sim_ensemble.py` variants, per-model health
  (incl. the dead-TCN finding), `diagnose_bias.py` output.
- Be explicit about sample sizes and what is/isn't statistically meaningful.

## 6. Validation & testing (1 page)
- pytest suite (risk controls, evidence exporter, archive dry-run, dashboard): case list + output.
- Existing validators kept green: `tools/test_fixes_123.py` (`RESULT: PASS`), `sanity_test.py`.
- Shadow-mode deploy procedure: flags-off parity day on EC2 before enabling anything.

## 7. Limitations — honest list (1 page)
- Paper PF over a small sample; no live-trading evidence (by design).
- Models currently on the 26-feature stopgap set; retrain pending (post-submission).
- Time-stop is signal-driven (no price-polling sweep yet).
- Train-timeframe/serve-timeframe coherence enforced by checklist, not yet by code contract.
- Single-exchange, six symbols, no portfolio optimization.

## 8. Roadmap (½ page)
- Phases 1–5 from `docs/V2_ROADMAP.md` with the dependency graph and the live-trading
  gate (reconciliation green + human sign-off).

## Appendices
- A. Operations runbook (`docs/OPERATIONS_RUNBOOK.md`).
- B. Full safety-controls table.
- C. Selected git history (annotated shas → fixes).
- D. Sample evidence bundle (one full `summary.json` + `summary.md`).
- E. Test transcript (`pytest -q`, validator outputs).

---

## Evidence checklist (gather before 24 June)
- [ ] ≥5 daily evidence bundles in `reports/evidence/`
- [ ] `dashboard.html` screenshot with ≥10 days of cumulative PnL
- [ ] `sim_exits.py` grid output used to justify the time-stop value
- [ ] Flags-off vs flags-on log excerpts (`SKIP reason=v2_*`, `EXIT_TIME` lines)
- [ ] `pytest -q` + `test_fixes_123.py` + `sanity_test.py` transcripts
- [ ] Deploy-marker history (`logs/deploy_markers.csv`)
- [ ] `git log --oneline` of the branch showing the incremental commits
