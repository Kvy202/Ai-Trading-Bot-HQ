# Operations Runbook — AWS (paper mode)

Audience: the operator. Everything here assumes **PAPER mode**.
Live trading is out of scope until after 25 June and requires the approval flow in
SAFETY_CONTROLS.md.

---

## 1. Box facts

| Item | Value |
|---|---|
| Host | AWS EC2, Ubuntu, project at `~/bot` |
| Venv | `~/bot/.venv` (use `.venv/bin/python`, never system python) |
| Services | `bot-writer.service` (signals), `bot-executor.service` (trades), `bot-proxy.service`, grouped under `trading-bot.target` |
| Service config | `WorkingDirectory=~/bot`, `EnvironmentFile=~/bot/.env`, `Environment=PYTHONPATH=~/bot` |
| **Mode switch** | The executor's mode is set by the **systemd ExecStart CLI flag** (`--paper` / `--live`), which **overrides `.env`**. Editing `.env` alone does NOT change mode. Check with: `systemctl cat bot-executor \| grep ExecStart` — expect `--paper`. |
| Key logs | `logs/live_executor.out|.err`, `logs/live_writer.out|.err`, `logs/trades_closed_*.csv`, `logs/live_signals.csv`, heartbeats `logs/heartbeat.json` + `logs/live_writer_heartbeat.json` |
| V2 state | `logs/v2_risk_state.json`, pause file `run/V2_PAUSE`, markers `logs/deploy_markers.csv` + `logs/DEPLOY_MARKERS.txt` |

---

## 2. Standard deploy sequence (pull → test → env → restart)

```bash
ssh ubuntu@<EC2-IP>
cd ~/bot

# 0. Snapshot current state (rollback anchor)
git rev-parse --short HEAD            # note this sha
cp .env /tmp/env.backup.$(date +%Y%m%d_%H%M%S)

# 1. Stop the executor only (writer keeps producing signals; they drain on restart)
sudo systemctl stop bot-executor

# 2. Pull the branch
git fetch origin
git checkout bot-v2-architecture
git pull --ff-only origin bot-v2-architecture

# 3. Test BEFORE starting (one-time: .venv/bin/python -m pip install pytest)
.venv/bin/python -m pytest tests/ -q          # V2 suite — must be all green
.venv/bin/python tools/test_fixes_123.py      # must end: RESULT: PASS
.venv/bin/python sanity_test.py               # offline model load + forward pass

# 4. Set env — APPEND V2 lines to .env, never overwrite the file.
#    First deploy: leave all V2_* unset (or =0) → shadow mode, V1-identical.
nano .env        # see .env.v2.example for the documented flags

# 5. Record the deploy
.venv/bin/python tools/v2_deploy_marker.py --note "deploy bot-v2-architecture (flags off)"

# 6. Start + verify
sudo systemctl start bot-executor
systemctl status bot-executor --no-pager
tail -n 30 logs/live_executor.out     # expect START mode=PAPER and a v2_risk status line
```

**Writer note:** these deploys do not change the writer; do not restart `bot-writer`
unless the deploy touched writer code (this branch does not).

---

## 3. Enabling / disabling V2 risk flags

Edit `~/bot/.env` (values documented in `.env.v2.example`), then:

```bash
sudo systemctl restart bot-executor
tail -n 20 logs/live_executor.out     # expect: v2_risk: enabled time_stop_min=240 ...
```

Suggested first enablement (paper), one flag at a time:

```bash
# Day 1: time-stop only (240 min mirrors run.json MAX_HOLD_BARS=48 × 5m bars)
V2_TIME_STOP_MIN=240
# Day 2+: add daily guards
V2_MAX_SL_PER_DAY=6
V2_DAILY_LOSS_LIMIT_USDT=1.5
V2_DAILY_DD_LIMIT_USDT=2.0
```

Disable: set the flag to `0` (or comment it out) and restart. Master off-switch without
removing lines: `V2_RISK_DISABLED=1` + restart.

**Instant pause (no restart, no .env change):**
```bash
touch ~/bot/run/V2_PAUSE      # blocks NEW entries within one poll (~3 s); exits keep working
rm ~/bot/run/V2_PAUSE         # resume
```

---

## 4. Health checks

```bash
cd ~/bot
systemctl status bot-writer bot-executor --no-pager
.venv/bin/python tools/bot_health_check.py          # 8-section diagnostic
tail -n 50 logs/live_executor.out                   # TRADE/SKIP lines; v2_risk status
tail -n 20 logs/live_executor.err                   # should be quiet
cat logs/heartbeat.json                             # executor heartbeat (age < ~2 min)
cat logs/live_writer_heartbeat.json                 # writer heartbeat (age < ~60 s)
cat logs/v2_risk_state.json 2>/dev/null             # V2 counters (exists once enabled)
grep -c "reason=v2_" logs/live_executor.out         # how often V2 gates fired
grep -c "EXIT_TIME" logs/live_executor.out          # time-stop exits
```

What healthy looks like: both services `active (running)`; heartbeats fresh; `.err` quiet;
signals flowing (`idle: no new signals` is fine off-hours); v2 state day = today (UTC).

---

## 5. Daily evidence routine (order matters)

Run export **before** archive — archiving moves dated CSVs the exporter reads.

```bash
cd ~/bot
.venv/bin/python tools/v2_evidence_export.py            # all days missing from the index
.venv/bin/python tools/v2_dashboard.py                  # → reports/dashboard.html
.venv/bin/python tools/v2_log_archive.py                # DRY-RUN: review the plan
.venv/bin/python tools/v2_log_archive.py --apply        # only if the plan looked right

# Pull artifacts to the laptop for the report:
#   (run from the laptop)
scp -r ubuntu@<EC2-IP>:~/bot/reports/evidence ./reports/
scp ubuntu@<EC2-IP>:~/bot/reports/dashboard.html ./reports/
```

---

## 6. Rollback ladder (fastest first)

| Level | Action | Scope | Commands |
|---|---|---|---|
| L0 | Pause file | Blocks new entries in seconds; positions exit normally | `touch ~/bot/run/V2_PAUSE` |
| L1 | Flags off | Disables all V2 risk behavior; code stays | `sed -i 's/^V2_/#V2_/' ~/bot/.env && sudo systemctl restart bot-executor` |
| L2 | Revert the wiring commit | Removes V2 hooks from the executor; new files remain (inert) | `cd ~/bot && git revert <wiring-commit-sha> && sudo systemctl restart bot-executor` |
| L3 | Checkout last-good sha | Full code rollback to the pre-deploy snapshot from step 0 | `cd ~/bot && git checkout <last-good-sha> && sudo systemctl restart bot-executor` |

After any rollback: run section 4 health checks and add a deploy marker noting the rollback.

---

## 7. Incident playbook

| Symptom | First moves |
|---|---|
| `LOOP_ERROR` repeating in `.err` | Read the traceback. If it mentions `v2`, apply L1 (flags off) — but note every v2 call is try/except-guarded, so a v2 traceback here would itself be a bug to file. Otherwise treat as V1 incident (usually exchange/network). |
| `FATAL` / OOD alarm from writer | The OOD guard forced `allow=0` — the bot is refusing to trade on bad inputs, which is correct. Run `.venv/bin/python tools/diagnose_features.py`. Do NOT restart-loop; fix the feature/scaler mismatch first. |
| `SIDE_BIAS` / `BIAS_LOCK` lines | Working as designed: entries suspended while signals are ≥95 % one-sided. Run `tools/diagnose_bias.py`. Do not disable the guard to "make it trade". |
| Stale heartbeat (> 5 min) | `systemctl status` the service; check disk space `df -h`; check OOM `dmesg \| tail`. Restart the affected service only. |
| Stuck position (no exits firing) | Check writer is emitting that symbol (`tail logs/live_signals.csv`). Time-stop/TP/SL are signal-driven — no signals for a symbol means no exits for it (known limitation). If urgent on paper: `sudo systemctl restart bot-executor` (restart-close handles past-TP/SL positions). |
| Executor won't start after deploy | `tail -n 50 logs/live_executor.err`; check the scaler-dim assert / artifact mismatch messages; if related to the deploy, go to L2/L3. |
| V2 counters look wrong | `cat logs/v2_risk_state.json`; counters rebuild from `logs/trades_closed_$(date -u +%Y%m%d).csv` on restart — restarting the executor re-derives them from the authoritative CSV. |

---

## 8. Live/paper switch warning

Changing `.env` `LIVE_MODE`/`EXEC_PAPER` does **not** switch modes while the systemd unit
passes `--paper` or `--live` (CLI overrides env). To actually switch you must edit the
unit's `ExecStart` (`sudo systemctl edit --full bot-executor`), `sudo systemctl
daemon-reload`, and restart. **Do not do this before 25 June.** Going live additionally
requires the supervisor approval flow and the checklist in SAFETY_CONTROLS.md.
