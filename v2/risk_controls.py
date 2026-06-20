"""V2 optional risk controls — time-stop, daily SL budget, daily loss/DD pause.

Design contract (see docs/SAFETY_CONTROLS.md §5):

* ALL controls are OFF by default. With no V2_* env vars set, every method is a
  no-op that returns the safe value, so the executor behaves exactly like V1.
* This module is stdlib-only and must never import from tools/ (the executor
  imports us lazily inside main(); a circular or heavy import would defeat the
  failure-isolation guarantee).
* Every public method swallows its own exceptions and returns the safe default.
  The executor additionally wraps each call site in try/except. A bug here may
  cost a V2 feature, never a V1 trade loop.
* Daily counters (SL count, realized PnL, intraday PnL peak) are REBUILT from
  logs/trades_closed_YYYYMMDD.csv at construction time. That CSV is append-only
  and written by the executor before this module ever sees the close, so
  restarts, stale state files, or enabling V2 mid-day cannot under-count.
  The state file only contributes position entry-times (which the CSV lacks).
* Time-stop measures wall-clock holding time (now - entry_epoch). Signal
  timestamps are never used — stale rows must not trigger exits.

Env flags (all read at init): V2_RISK_DISABLED, V2_TIME_STOP_MIN,
V2_MAX_SL_PER_DAY, V2_DAILY_LOSS_LIMIT_USDT, V2_DAILY_DD_LIMIT_USDT,
V2_PAUSE_FILE, V2_RISK_STATE.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

STATE_VERSION = 1


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    return _env_str(name, "1" if default else "0").lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_str(name, str(default))))
    except Exception:
        return int(default)


def _utc_day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class RiskConfig:
    time_stop_min: float
    max_sl_per_day: int
    daily_loss_limit_usdt: float
    daily_dd_limit_usdt: float
    pause_file: Path
    state_path: Path
    logs_dir: Path

    @classmethod
    def from_env(cls, base_dir: Path, logs_dir: Path) -> "RiskConfig":
        def _path(name: str, default: str) -> Path:
            p = Path(_env_str(name, default))
            return p if p.is_absolute() else (base_dir / p)

        return cls(
            time_stop_min=max(0.0, _env_float("V2_TIME_STOP_MIN", 0.0)),
            max_sl_per_day=max(0, _env_int("V2_MAX_SL_PER_DAY", 0)),
            daily_loss_limit_usdt=max(0.0, _env_float("V2_DAILY_LOSS_LIMIT_USDT", 0.0)),
            daily_dd_limit_usdt=max(0.0, _env_float("V2_DAILY_DD_LIMIT_USDT", 0.0)),
            pause_file=_path("V2_PAUSE_FILE", "run/V2_PAUSE"),
            state_path=_path("V2_RISK_STATE", "logs/v2_risk_state.json"),
            logs_dir=logs_dir,
        )

    def summary(self) -> str:
        return (f"time_stop_min={self.time_stop_min:g} max_sl_per_day={self.max_sl_per_day} "
                f"daily_loss_limit_usdt={self.daily_loss_limit_usdt:g} "
                f"daily_dd_limit_usdt={self.daily_dd_limit_usdt:g} "
                f"pause_file={self.pause_file}")


class RiskControls:
    """Daily risk state machine. now_fn is injectable so tests can fake the clock."""

    def __init__(self, cfg: RiskConfig,
                 now_fn: Callable[[], float] = time.time,
                 log_fn: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg
        self._now = now_fn
        self._log = log_fn or (lambda msg: None)
        self.entry_times: Dict[str, float] = {}
        self.utc_day: str = _utc_day(self._now())
        self.sl_count_today: int = 0
        self.realized_pnl_today: float = 0.0
        self.pnl_peak_today: float = 0.0
        self._load_state()
        self._rebuild_counters_from_csv()

    # -- persistence --------------------------------------------------------

    def _load_state(self) -> None:
        """Entry-times come from the state file; counters are overwritten by the
        CSV rebuild right after, so stale counter values here are harmless."""
        try:
            data = json.loads(self.cfg.state_path.read_text(encoding="utf-8"))
            times = data.get("entry_times", {})
            self.entry_times = {str(k): float(v) for k, v in times.items()
                                if isinstance(v, (int, float))}
        except Exception:
            self.entry_times = {}

    def _save_state(self) -> None:
        try:
            data = {
                "v": STATE_VERSION,
                "utc_day": self.utc_day,
                "entry_times": self.entry_times,
                "sl_count_today": self.sl_count_today,
                "realized_pnl_today": round(self.realized_pnl_today, 8),
                "pnl_peak_today": round(self.pnl_peak_today, 8),
                "updated_utc": datetime.fromtimestamp(
                    self._now(), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
            }
            self.cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.cfg.state_path)
        except Exception:
            pass  # state is advisory; in-memory values stay correct

    def _rebuild_counters_from_csv(self) -> None:
        """Derive today's counters from the executor's append-only daily close
        log. Runs at init only; closes after init arrive via on_close()."""
        self.sl_count_today = 0
        self.realized_pnl_today = 0.0
        self.pnl_peak_today = 0.0
        day_compact = self.utc_day.replace("-", "")
        path = self.cfg.logs_dir / f"trades_closed_{day_compact}.csv"
        try:
            if not path.exists():
                self._save_state()
                return
            with path.open("r", encoding="utf-8", newline="") as f:
                running = 0.0
                peak = 0.0
                for row in csv.DictReader(f):
                    try:
                        pnl = float(row.get("realized_pnl", "") or 0.0)
                    except Exception:
                        continue
                    reason = (row.get("reason") or "").strip()
                    if reason.startswith("EXIT_SL"):
                        self.sl_count_today += 1
                    running += pnl
                    peak = max(peak, running)
                self.realized_pnl_today = running
                self.pnl_peak_today = peak
        except Exception:
            pass
        self._save_state()

    # -- day rollover --------------------------------------------------------

    def _maybe_rollover(self) -> None:
        today = _utc_day(self._now())
        if today != self.utc_day:
            self._log(f"v2_risk: UTC day rollover {self.utc_day} -> {today}; "
                      f"counters reset (entry times preserved)")
            self.utc_day = today
            self.sl_count_today = 0
            self.realized_pnl_today = 0.0
            self.pnl_peak_today = 0.0
            self._save_state()

    # -- executor hooks ------------------------------------------------------

    def sync_open_positions(self, symbols: Iterable[str]) -> None:
        """Reconcile entry-time tracking with the executor's restored positions.
        Unknown open symbol -> first-seen baseline (clock starts now: conservative,
        a restart can extend a position's life by at most one restart gap).
        Tracked symbol no longer open -> pruned."""
        try:
            live = {str(s) for s in symbols}
            now = self._now()
            for sym in live:
                self.entry_times.setdefault(sym, now)
            for sym in list(self.entry_times):
                if sym not in live:
                    self.entry_times.pop(sym, None)
            self._save_state()
        except Exception:
            pass

    def on_entry(self, symbol: str) -> None:
        try:
            self._maybe_rollover()
            self.entry_times[str(symbol)] = self._now()
            self._save_state()
        except Exception:
            pass

    def on_close(self, symbol: str, reason: str, realized_pnl: float) -> None:
        try:
            self._maybe_rollover()
            self.entry_times.pop(str(symbol), None)
            if (reason or "").strip().startswith("EXIT_SL"):  # EXIT_SL + EXIT_SL_RESTART
                self.sl_count_today += 1
            try:
                self.realized_pnl_today += float(realized_pnl)
            except Exception:
                pass
            self.pnl_peak_today = max(self.pnl_peak_today, self.realized_pnl_today)
            self._save_state()
        except Exception:
            pass

    def time_stop_due(self, symbol: str) -> Optional[float]:
        """Return held minutes if the position is past the time-stop, else None.
        None when the feature is off (0) or the symbol isn't tracked."""
        try:
            if self.cfg.time_stop_min <= 0:
                return None
            entry = self.entry_times.get(str(symbol))
            if entry is None:
                return None
            held_min = (self._now() - entry) / 60.0
            return held_min if held_min >= self.cfg.time_stop_min else None
        except Exception:
            return None

    def entry_block_reason(self) -> Optional[str]:
        """Reason string when new entries must be blocked, else None.
        Precedence: pause file > SL budget > daily loss > daily drawdown."""
        try:
            self._maybe_rollover()
            if self.cfg.pause_file.exists():
                return "v2_pause_file"
            if 0 < self.cfg.max_sl_per_day <= self.sl_count_today:
                return f"v2_max_sl_per_day({self.sl_count_today}/{self.cfg.max_sl_per_day})"
            limit = self.cfg.daily_loss_limit_usdt
            if limit > 0 and self.realized_pnl_today <= -limit:
                return f"v2_daily_loss_limit({self.realized_pnl_today:.4f}<=-{limit:g})"
            dd_limit = self.cfg.daily_dd_limit_usdt
            if dd_limit > 0:
                dd = self.pnl_peak_today - self.realized_pnl_today
                if dd >= dd_limit:
                    return f"v2_daily_dd_limit(dd={dd:.4f}>={dd_limit:g})"
            return None
        except Exception:
            return None  # fail-open: a v2 bug must not freeze trading

    def snapshot(self) -> dict:
        try:
            return {
                "utc_day": self.utc_day,
                "sl_count_today": self.sl_count_today,
                "realized_pnl_today": round(self.realized_pnl_today, 8),
                "pnl_peak_today": round(self.pnl_peak_today, 8),
                "open_tracked": sorted(self.entry_times),
                "entry_block": self.entry_block_reason(),
                "config": self.cfg.summary(),
            }
        except Exception:
            return {}


def init_risk_controls(base_dir: Path, logs_dir: Path,
                       open_position_symbols: Iterable[str],
                       log_fn: Optional[Callable[[str], None]] = None,
                       err_fn: Optional[Callable[[str], None]] = None,
                       now_fn: Callable[[], float] = time.time,
                       ) -> Optional[RiskControls]:
    """Build the risk controls for the executor. Returns None when disabled or
    on ANY failure — the executor treats None as "run pure V1"."""
    log = log_fn or (lambda msg: None)
    err = err_fn or log
    try:
        if _env_bool("V2_RISK_DISABLED", False):
            log("v2_risk: disabled via V2_RISK_DISABLED=1 — executor runs as V1")
            return None
        cfg = RiskConfig.from_env(Path(base_dir), Path(logs_dir))
        rc = RiskControls(cfg, now_fn=now_fn, log_fn=log)
        rc.sync_open_positions(open_position_symbols)
        active = (cfg.time_stop_min > 0 or cfg.max_sl_per_day > 0
                  or cfg.daily_loss_limit_usdt > 0 or cfg.daily_dd_limit_usdt > 0)
        log(f"v2_risk: enabled ({'active' if active else 'inert — all flags 0, pause file armed'}) "
            f"{cfg.summary()} | today: sl_count={rc.sl_count_today} "
            f"pnl={rc.realized_pnl_today:.4f}")
        return rc
    except Exception as exc:
        err(f"v2_risk init failed — running without V2 risk controls: {exc}")
        return None
