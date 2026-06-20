#!/usr/bin/env python
"""
tools/telegram_notifier.py
===========================
Notifier Bot — read-only.  Polls health files and sends alerts to Telegram.

Alerts fired:
  - Startup banner
  - Writer heartbeat stale (> 120 s)
  - Executor heartbeat stale (> 600 s)
  - Writer process not found
  - Executor process not found
  - Recovery (process came back / heartbeat fresh again)
  - Daily PnL summary at 00:05 UTC

Token env var: TELEGRAM_NOTIFIER_TOKEN  (falls back to TELEGRAM_BOT_TOKEN)
Chat env var:  TELEGRAM_CHAT_ID

Usage:
    python tools/telegram_notifier.py
    python tools/telegram_notifier.py --interval 60
    python tools/telegram_notifier.py --no-daily
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests as _req
except ImportError:
    print("ERROR: 'requests' not installed -- run: pip install requests")
    sys.exit(1)

_ROOT = Path(__file__).resolve().parents[1]
_ENV  = _ROOT / ".env"
_LOGS = _ROOT / "logs"

UTC = timezone.utc

# Staleness limits (seconds) — same as watchdog.py
WRITER_HB_MAX   = 120
EXECUTOR_HB_MAX = 600
META_LOG_MAX    = 300

# How long a channel must recover before we re-arm it (avoid spam)
RECOVERY_COOLDOWN = 120


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env() -> None:
    if not _ENV.exists():
        return
    for line in _ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Heartbeat / process helpers (mirrors watchdog.py)
# ---------------------------------------------------------------------------

def _hb_age(path: Path) -> float | None:
    """Return seconds since heartbeat was written, or None if missing/unreadable."""
    if not path.exists():
        return None
    try:
        data   = json.loads(path.read_text(encoding="utf-8"))
        ts_str = data.get("ts", "")
        ts     = datetime.fromisoformat(ts_str.replace("+0000", "+00:00"))
        return (datetime.now(UTC) - ts).total_seconds()
    except Exception:
        return None


def _process_alive(keyword: str) -> bool:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"Get-CimInstance Win32_Process | "
             f"Where-Object {{ $_.Name -eq 'python.exe' -and $_.CommandLine -match '{keyword}' }} | "
             f"Select-Object -ExpandProperty ProcessId"],
            text=True, timeout=15, stderr=subprocess.DEVNULL,
        )
        return bool([p for p in out.strip().splitlines() if p.strip().isdigit()])
    except Exception:
        return False


def _meta_log_age() -> float | None:
    candidates = list(_LOGS.glob("live_meta_log*.csv"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return time.time() - newest.stat().st_mtime


# ---------------------------------------------------------------------------
# Daily PnL summary
# ---------------------------------------------------------------------------

def _daily_summary() -> str:
    """Read today's realized PnL from the executor heartbeat / log files."""
    hb_path = _LOGS / "heartbeat.json"
    try:
        if hb_path.exists():
            data = json.loads(hb_path.read_text(encoding="utf-8"))
            pnl  = data.get("realized_today", data.get("pnl_today", None))
            trades = data.get("trades_today", "?")
            if pnl is not None:
                return (f"Daily PnL report\n"
                        f"  realized: {pnl:+.4f} USDT\n"
                        f"  trades:   {trades}")
    except Exception:
        pass

    # fallback: just report heartbeat age
    age = _hb_age(hb_path)
    if age is not None:
        return f"Daily report: executor heartbeat is {age:.0f}s old."
    return "Daily report: no executor heartbeat file found."


# ---------------------------------------------------------------------------
# Alert state machine (suppresses duplicates)
# ---------------------------------------------------------------------------

class AlertState:
    """Track fired/cleared state per channel so we don't spam."""

    def __init__(self) -> None:
        self._fired: dict[str, float] = {}   # channel -> time first fired
        self._clear: dict[str, float] = {}   # channel -> time cleared

    def should_alert(self, channel: str) -> bool:
        """True if alert should be sent now (first time or re-arm after cooldown)."""
        if channel not in self._fired:
            return True
        # already firing; don't repeat
        return False

    def mark_fired(self, channel: str) -> None:
        self._fired[channel] = time.time()
        self._clear.pop(channel, None)

    def should_recover(self, channel: str) -> bool:
        """True if a recovery message should be sent."""
        if channel not in self._fired:
            return False
        cleared_at = self._clear.get(channel)
        if cleared_at is None:
            self._clear[channel] = time.time()
            return False
        # fire recovery only after RECOVERY_COOLDOWN sustained clear
        return (time.time() - cleared_at) >= RECOVERY_COOLDOWN

    def mark_recovered(self, channel: str) -> None:
        self._fired.pop(channel, None)
        self._clear.pop(channel, None)

    def mark_clear_tick(self, channel: str) -> None:
        """Call each tick when condition is healthy to start cooldown timer."""
        if channel in self._fired and channel not in self._clear:
            self._clear[channel] = time.time()


# ---------------------------------------------------------------------------
# Notifier bot
# ---------------------------------------------------------------------------

class NotifierBot:
    def __init__(self, tg_token: str, chat_id: int,
                 interval: int, daily_report: bool) -> None:
        self.base      = f"https://api.telegram.org/bot{tg_token.strip()}"
        self.chat_id   = chat_id
        self.interval  = interval
        self.daily     = daily_report
        self._stop     = threading.Event()
        self._state    = AlertState()
        self._last_day: object = None

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    def send(self, text: str) -> None:
        try:
            _req.post(self.base + "/sendMessage",
                      json={"chat_id": self.chat_id, "text": text},
                      timeout=15)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def _check(self) -> None:
        now = datetime.now(UTC)

        # ---- writer heartbeat ----
        w_age = _hb_age(_LOGS / "live_writer_heartbeat.json")
        ch = "writer_hb"
        if w_age is None or w_age > WRITER_HB_MAX:
            detail = f"missing" if w_age is None else f"{w_age:.0f}s old"
            if self._state.should_alert(ch):
                self.send(f"ALERT: writer heartbeat stale ({detail}, limit {WRITER_HB_MAX}s)")
                self._state.mark_fired(ch)
        else:
            self._state.mark_clear_tick(ch)
            if self._state.should_recover(ch):
                self.send(f"RECOVERY: writer heartbeat fresh ({w_age:.0f}s old)")
                self._state.mark_recovered(ch)

        # ---- executor heartbeat ----
        e_age = _hb_age(_LOGS / "heartbeat.json")
        ch = "executor_hb"
        if e_age is None or e_age > EXECUTOR_HB_MAX:
            detail = "missing" if e_age is None else f"{e_age:.0f}s old"
            if self._state.should_alert(ch):
                self.send(f"ALERT: executor heartbeat stale ({detail}, limit {EXECUTOR_HB_MAX}s)")
                self._state.mark_fired(ch)
        else:
            self._state.mark_clear_tick(ch)
            if self._state.should_recover(ch):
                self.send(f"RECOVERY: executor heartbeat fresh ({e_age:.0f}s old)")
                self._state.mark_recovered(ch)

        # ---- writer process ----
        ch = "writer_proc"
        if not _process_alive("live_writer"):
            if self._state.should_alert(ch):
                self.send("ALERT: live_writer.py process not found")
                self._state.mark_fired(ch)
        else:
            self._state.mark_clear_tick(ch)
            if self._state.should_recover(ch):
                self.send("RECOVERY: live_writer.py is running again")
                self._state.mark_recovered(ch)

        # ---- executor process ----
        ch = "executor_proc"
        if not _process_alive("live_executor"):
            if self._state.should_alert(ch):
                self.send("ALERT: live_executor.py process not found")
                self._state.mark_fired(ch)
        else:
            self._state.mark_clear_tick(ch)
            if self._state.should_recover(ch):
                self.send("RECOVERY: live_executor.py is running again")
                self._state.mark_recovered(ch)

        # ---- meta-log growth ----
        ml_age = _meta_log_age()
        ch = "meta_log"
        if ml_age is None or ml_age > META_LOG_MAX:
            detail = "no file" if ml_age is None else f"{ml_age:.0f}s old"
            if self._state.should_alert(ch):
                self.send(f"ALERT: meta-log not growing ({detail}, limit {META_LOG_MAX}s)")
                self._state.mark_fired(ch)
        else:
            self._state.mark_clear_tick(ch)
            if self._state.should_recover(ch):
                self.send(f"RECOVERY: meta-log writing again ({ml_age:.0f}s ago)")
                self._state.mark_recovered(ch)

        # ---- daily report ----
        if self.daily:
            today = now.date()
            if now.hour == 0 and now.minute >= 5 and self._last_day != today:
                self.send(_daily_summary())
                self._last_day = today

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"[notifier] started (chat_id={self.chat_id}, interval={self.interval}s)")
        self.send(f"Notifier Bot started.  Monitoring every {self.interval}s.")
        while not self._stop.is_set():
            try:
                self._check()
            except Exception as exc:
                print(f"[notifier] check error: {exc}")
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    p = argparse.ArgumentParser(description="Telegram Notifier Bot")
    p.add_argument("--interval", type=int, default=60,
                   help="Health-check interval in seconds (default: 60)")
    p.add_argument("--no-daily", action="store_true",
                   help="Disable the daily PnL report")
    args = p.parse_args()

    tg_token = os.getenv("TELEGRAM_NOTIFIER_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id_raw = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not tg_token:
        print("ERROR: set TELEGRAM_NOTIFIER_TOKEN (or TELEGRAM_BOT_TOKEN) in .env")
        sys.exit(1)
    if not chat_id_raw.lstrip("-").isdigit():
        print("ERROR: TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    _LOGS.mkdir(exist_ok=True)
    bot = NotifierBot(
        tg_token  = tg_token,
        chat_id   = int(chat_id_raw),
        interval  = args.interval,
        daily_report = not args.no_daily,
    )
    bot.run()


if __name__ == "__main__":
    main()
