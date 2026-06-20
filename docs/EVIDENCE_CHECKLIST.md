# Evidence Checklist — freeze period 13 → 25 June 2026

Bot frozen at branch `bot-v2-architecture`, commit `bbdc82f`, PAPER mode, all V2 flags off.
Companion to SUBMISSION_REPORT_OUTLINE.md. Everything below is **collection only** —
no behavior changes. All commands run on the EC2 box from `~/bot` unless marked [laptop].

---

## Phase A — freeze proof (once, today)

- [ ] Capture the freeze snapshot:

```bash
mkdir -p reports/evidence/freeze reports/evidence/transcripts reports/evidence/diagnostics
d=$(date -u +%Y%m%d)
{
  echo "=== freeze proof $d ==="
  date -u
  git rev-parse --short HEAD          # expect bbdc82f
  git branch --show-current           # expect bot-v2-architecture
  git status --short                  # expect clean (or note any local diffs)
  echo "--- systemd ---"
  systemctl cat bot-executor
  systemctl cat bot-writer
  echo "--- v2_risk line ---"
  grep "v2_risk" logs/live_executor.out | tail -2
  echo "--- shadow parity counters ---"
  echo "v2 skips:     $(grep -c 'reason=v2_' logs/live_executor.out)"   # expect 0
  echo "EXIT_TIME:    $(grep -c 'EXIT_TIME'  logs/live_executor.out)"   # expect 0
} > reports/evidence/freeze/freeze_proof_$d.txt
```

- [ ] Capture the **non-secret** config. ⚠️ Never copy the raw `.env` into evidence —
  it contains API keys. Filter:

```bash
grep -E '^(V2_|EXEC_|DL_|MAX_|PER_SYMBOL|LIVE_MODE|EXEC_PAPER|LEVERAGE|TIER2_)' .env \
  | grep -viE 'key|secret|pass|token' \
  > reports/evidence/freeze/env_nonsecret_$d.txt
```

- [ ] Deploy marker for the freeze itself:

```bash
.venv/bin/python tools/v2_deploy_marker.py --note "FREEZE start: evidence-only until submission"
```

## Phase B — validator transcripts (once, today)

- [ ] Re-run each validator with `tee` so Appendix E has dated transcripts:

```bash
.venv/bin/python -m pytest tests/ -q              | tee reports/evidence/transcripts/pytest_$d.txt
.venv/bin/python tools/test_fixes_123.py          | tee reports/evidence/transcripts/test_fixes_123_$d.txt
.venv/bin/python tools/test_gate_fix.py           | tee reports/evidence/transcripts/test_gate_fix_$d.txt
.venv/bin/python tools/test_sim_exits.py          | tee reports/evidence/transcripts/test_sim_exits_$d.txt
.venv/bin/python tools/parity_check.py            | tee reports/evidence/transcripts/parity_check_$d.txt
git log --oneline -25                             > reports/evidence/transcripts/git_log_$d.txt
```

## Phase C — daily (cron, set up once today)

- [ ] `crontab -e` and add:

```cron
10 0 * * * cd ~/bot && .venv/bin/python tools/v2_evidence_export.py --quiet && .venv/bin/python tools/v2_dashboard.py >/dev/null 2>&1
```

- [ ] Verify tomorrow that `reports/evidence/index.json` gained a day and
  `reports/dashboard.html` has a fresh "generated" timestamp.
- [ ] Do **NOT** run `v2_log_archive.py --apply` until after submission
  (dry-run is fine any time).

## Phase D — every 2–3 days (diagnostics snapshots)

- [ ] Dated snapshots (these numbers drift; a figure only exists if it was saved):

```bash
d=$(date -u +%Y%m%d)
.venv/bin/python tools/diagnose_bias.py                 > reports/evidence/diagnostics/bias_$d.txt
.venv/bin/python tools/sim_ensemble.py                  > reports/evidence/diagnostics/ensemble_$d.txt
.venv/bin/python tools/sim_exits.py --sessions current  > reports/evidence/diagnostics/exits_$d.txt
.venv/bin/python tools/bot_health_check.py              > reports/evidence/diagnostics/health_$d.txt
```

- [ ] [laptop] Off-box backup (EC2 is a single point of failure for the results section):

```bash
scp -r ubuntu@<EC2-IP>:~/bot/reports ./submission_backup/reports_$(date +%Y%m%d)
scp ubuntu@<EC2-IP>:"~/bot/logs/trades_closed_*.csv" ./submission_backup/
```

Suggested snapshot days: **13, 15, 17, 19, 21, 23 June** + final capture on the 24th.

## Phase E — decision point, 19–20 June (the ONLY planned decision)

Criteria agreed in advance — do not improvise:

- [ ] Check sample size: `wc -l logs/trades_closed_*.csv` for the clean session
      (or `sim_exits.py --sessions current` header).
- If **n < 30**: stay frozen. The report justifies the time-stop value via the
  `sim_exits.py` counterfactual grid instead of live evidence.
- If **n ≥ 30** and you choose to enable `V2_TIME_STOP_MIN`:
  - [ ] Diagnostics snapshot FIRST (closes the pre-change sample cleanly)
  - [ ] `tools/v2_deploy_marker.py --note "enable V2_TIME_STOP_MIN=<value> (paper)"`
  - [ ] Append flag to `.env`, `sudo systemctl restart bot-executor`
  - [ ] Verify: `grep "v2_risk" logs/live_executor.out | tail -1` shows `(active)` and the value
- Either choice is fine; **record which and why** in a one-line note in
  `reports/evidence/freeze/decision_20260619.txt`.

## Phase F — final capture (24 June, before writing stops)

- [ ] Last evidence export + dashboard run; screenshot `dashboard.html` (full page)
- [ ] Final diagnostics snapshot set (Phase D commands)
- [ ] Final shadow-parity counters (Phase A grep block) — still 0/0 if never enabled
- [ ] Uptime evidence: `systemctl status bot-writer bot-executor --no-pager | head -20`
      and journal restart count: `journalctl -u bot-executor --since "2026-06-13" | grep -c Started`
- [ ] [laptop] Final full backup (Phase D scp)
- [ ] Freeze-end marker: `tools/v2_deploy_marker.py --note "FREEZE end: final evidence captured"`

## Standing rules during the freeze

Do not: enable any V2 flag (except the Phase E decision), change weights/thresholds/
symbols/models/features/executor code, restart services without cause, `git pull` on the
box (the branch may receive docs-only commits; the box stays at `bbdc82f`), or run
`v2_log_archive.py --apply`. Every exception gets a deploy marker before and after.

## Report mapping (where each artifact lands)

| Artifact | Report section |
|---|---|
| freeze_proof + env_nonsecret + markers | §6 Validation / Appendix |
| transcripts/* | §6 + Appendix E |
| evidence/index.json + daily summary.md | §5 Results |
| dashboard.html screenshot | §5 Results |
| diagnostics/bias_*, ensemble_* | §5 model-health findings |
| diagnostics/exits_* (counterfactual grid) | §5 time-stop justification |
| decision_20260619.txt | §7 Limitations / §8 Roadmap |
