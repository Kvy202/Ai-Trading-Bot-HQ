"""Unit tests for v2/risk_controls.py — no executor import, fake clock, tmp dirs."""

import csv
import json

import pytest

from v2.risk_controls import RiskConfig, RiskControls, init_risk_controls

DAY1_NOON = 1781179200.0  # mid-day UTC, far from midnight; fixed for determinism


class FakeClock:
    def __init__(self, start: float = DAY1_NOON):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_cfg(tmp_path, **over) -> RiskConfig:
    defaults = dict(
        time_stop_min=0.0,
        max_sl_per_day=0,
        daily_loss_limit_usdt=0.0,
        daily_dd_limit_usdt=0.0,
        pause_file=tmp_path / "run" / "V2_PAUSE",
        state_path=tmp_path / "logs" / "v2_risk_state.json",
        logs_dir=tmp_path / "logs",
    )
    defaults.update(over)
    (tmp_path / "logs").mkdir(exist_ok=True)
    return RiskConfig(**defaults)


def make_rc(tmp_path, clock=None, **over) -> RiskControls:
    return RiskControls(make_cfg(tmp_path, **over), now_fn=clock or FakeClock())


def write_closed_csv(logs_dir, day_compact, rows):
    path = logs_dir / f"trades_closed_{day_compact}.csv"
    header = ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price",
              "realized_pnl", "reason"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def day_compact(clock) -> str:
    from v2.risk_controls import _utc_day
    return _utc_day(clock()).replace("-", "")


# ---------------------------------------------------------------------------
# Disabled / no-op behavior
# ---------------------------------------------------------------------------

def test_all_flags_off_is_noop(tmp_path):
    rc = make_rc(tmp_path)
    assert rc.entry_block_reason() is None
    assert rc.time_stop_due("BTCUSDT") is None
    rc.on_entry("BTCUSDT")
    rc.on_close("BTCUSDT", "EXIT_SL p=0.1 pnl=-1", -1.0)
    assert rc.entry_block_reason() is None  # limits are 0 = off
    assert rc.time_stop_due("BTCUSDT") is None


def test_init_disabled_env_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("V2_RISK_DISABLED", "1")
    assert init_risk_controls(tmp_path, tmp_path / "logs", []) is None


def test_init_returns_instance_and_survives_bad_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("V2_RISK_DISABLED", raising=False)
    monkeypatch.setenv("V2_RISK_STATE", str(tmp_path / "logs" / "state.json"))
    monkeypatch.setenv("V2_PAUSE_FILE", str(tmp_path / "run" / "V2_PAUSE"))
    rc = init_risk_controls(tmp_path, tmp_path / "logs", ["ETHUSDT"])
    assert rc is not None
    assert "ETHUSDT" in rc.entry_times


# ---------------------------------------------------------------------------
# Time-stop
# ---------------------------------------------------------------------------

def test_time_stop_boundary(tmp_path):
    clock = FakeClock()
    rc = make_rc(tmp_path, clock=clock, time_stop_min=30.0)
    rc.on_entry("BTCUSDT")
    clock.advance(30 * 60 - 1)          # one second short
    assert rc.time_stop_due("BTCUSDT") is None
    clock.advance(1)                     # exactly 30 min -> due (>= semantics)
    held = rc.time_stop_due("BTCUSDT")
    assert held == pytest.approx(30.0)


def test_time_stop_zero_means_off(tmp_path):
    clock = FakeClock()
    rc = make_rc(tmp_path, clock=clock, time_stop_min=0.0)
    rc.on_entry("BTCUSDT")
    clock.advance(86400 * 3)
    assert rc.time_stop_due("BTCUSDT") is None


def test_time_stop_unknown_symbol(tmp_path):
    rc = make_rc(tmp_path, time_stop_min=1.0)
    assert rc.time_stop_due("NEVERSEEN") is None


def test_close_clears_entry_time(tmp_path):
    clock = FakeClock()
    rc = make_rc(tmp_path, clock=clock, time_stop_min=1.0)
    rc.on_entry("BTCUSDT")
    rc.on_close("BTCUSDT", "EXIT_TP pnl=0.5", 0.5)
    clock.advance(3600)
    assert rc.time_stop_due("BTCUSDT") is None


def test_sync_first_seen_and_prune(tmp_path):
    clock = FakeClock()
    rc = make_rc(tmp_path, clock=clock, time_stop_min=10.0)
    rc.on_entry("OLDPOS")
    clock.advance(5 * 60)
    rc.sync_open_positions(["OLDPOS", "RESTORED"])
    # OLDPOS keeps its original entry time; RESTORED baselines at now
    assert rc.entry_times["OLDPOS"] == clock.t - 5 * 60
    assert rc.entry_times["RESTORED"] == clock.t
    rc.sync_open_positions(["RESTORED"])  # OLDPOS no longer open -> pruned
    assert "OLDPOS" not in rc.entry_times


# ---------------------------------------------------------------------------
# Daily SL-count limit
# ---------------------------------------------------------------------------

def test_sl_count_blocks_at_limit(tmp_path):
    rc = make_rc(tmp_path, max_sl_per_day=3)
    for _ in range(2):
        rc.on_close("X", "EXIT_SL p=0.2 pnl=-0.1", -0.1)
    assert rc.entry_block_reason() is None
    rc.on_close("X", "EXIT_SL_RESTART pnl=-0.2", -0.2)  # restart-SL counts too
    blk = rc.entry_block_reason()
    assert blk is not None and blk.startswith("v2_max_sl_per_day(3/3)")


def test_non_sl_reasons_do_not_count(tmp_path):
    rc = make_rc(tmp_path, max_sl_per_day=1)
    rc.on_close("X", "EXIT_TP pnl=0.3", 0.3)
    rc.on_close("X", "FLIP_CLOSE p=-0.1 pnl=-0.05", -0.05)
    rc.on_close("X", "EXIT_TIME held_min=240 pnl=0.01", 0.01)
    assert rc.entry_block_reason() is None


# ---------------------------------------------------------------------------
# Daily loss limit and drawdown pause
# ---------------------------------------------------------------------------

def test_daily_loss_limit_blocks_and_lifts(tmp_path):
    rc = make_rc(tmp_path, daily_loss_limit_usdt=1.0)
    rc.on_close("X", "FLIP_CLOSE pnl=-0.6", -0.6)
    assert rc.entry_block_reason() is None
    rc.on_close("X", "FLIP_CLOSE pnl=-0.4", -0.4)      # net -1.0 -> blocked
    assert "v2_daily_loss_limit" in (rc.entry_block_reason() or "")
    rc.on_close("X", "EXIT_TP pnl=0.5", 0.5)            # net -0.5 -> lifted
    assert rc.entry_block_reason() is None


def test_daily_dd_pause(tmp_path):
    rc = make_rc(tmp_path, daily_dd_limit_usdt=1.0)
    rc.on_close("X", "EXIT_TP pnl=1.5", 1.5)            # peak 1.5
    rc.on_close("X", "FLIP_CLOSE pnl=-0.5", -0.5)       # dd 0.5 -> fine
    assert rc.entry_block_reason() is None
    rc.on_close("X", "FLIP_CLOSE pnl=-0.6", -0.6)       # net 0.4, dd 1.1 -> blocked
    blk = rc.entry_block_reason()
    assert blk is not None and blk.startswith("v2_daily_dd_limit")
    # NOTE: still net-positive on the day — only the DD flag catches this


# ---------------------------------------------------------------------------
# Pause file
# ---------------------------------------------------------------------------

def test_pause_file_blocks_even_with_flags_off(tmp_path):
    rc = make_rc(tmp_path)
    pf = rc.cfg.pause_file
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.touch()
    assert rc.entry_block_reason() == "v2_pause_file"
    pf.unlink()
    assert rc.entry_block_reason() is None


def test_pause_file_takes_precedence(tmp_path):
    rc = make_rc(tmp_path, max_sl_per_day=1)
    rc.on_close("X", "EXIT_SL pnl=-1", -1.0)
    pf = rc.cfg.pause_file
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.touch()
    assert rc.entry_block_reason() == "v2_pause_file"


# ---------------------------------------------------------------------------
# Persistence, CSV rebuild, rollover, robustness
# ---------------------------------------------------------------------------

def test_persistence_roundtrip_entry_times(tmp_path):
    clock = FakeClock()
    rc = make_rc(tmp_path, clock=clock, time_stop_min=60.0)
    rc.on_entry("BTCUSDT")
    entry_t = rc.entry_times["BTCUSDT"]
    rc2 = make_rc(tmp_path, clock=clock, time_stop_min=60.0)  # same state path
    assert rc2.entry_times["BTCUSDT"] == entry_t


def test_counters_rebuilt_from_csv_not_state(tmp_path, monkeypatch):
    clock = FakeClock()
    cfg = make_cfg(tmp_path, max_sl_per_day=2)
    # Stale state file claims a clean day...
    cfg.state_path.write_text(json.dumps({
        "v": 1, "utc_day": "2000-01-01", "entry_times": {},
        "sl_count_today": 0, "realized_pnl_today": 0.0, "pnl_peak_today": 0.0,
    }), encoding="utf-8")
    # ...but today's authoritative CSV says 2 SLs and a net loss.
    write_closed_csv(cfg.logs_dir, day_compact(clock), [
        ["t1", "BTCUSDT", "SELL", 1, 100, 99, -1.0, "EXIT_SL p=0.1 pnl=-1.0"],
        ["t2", "ETHUSDT", "SELL", 1, 100, 101, 1.0, "EXIT_TP pnl=1.0"],
        ["t3", "BTCUSDT", "SELL", 1, 100, 99, -1.0, "EXIT_SL_RESTART pnl=-1.0"],
    ])
    rc = RiskControls(cfg, now_fn=clock)
    assert rc.sl_count_today == 2
    assert rc.realized_pnl_today == pytest.approx(-1.0)
    assert rc.pnl_peak_today == pytest.approx(0.0)  # running: -1.0, 0.0, -1.0 -> peak 0
    assert rc.entry_block_reason().startswith("v2_max_sl_per_day(2/2)")


def test_csv_with_malformed_rows(tmp_path):
    clock = FakeClock()
    cfg = make_cfg(tmp_path)
    write_closed_csv(cfg.logs_dir, day_compact(clock), [
        ["t1", "BTCUSDT", "SELL", 1, 100, 99, "garbage", "EXIT_SL pnl=?"],
        ["t2", "BTCUSDT", "SELL", 1, 100, 101, 0.5, "EXIT_TP pnl=0.5"],
    ])
    rc = RiskControls(cfg, now_fn=clock)
    assert rc.realized_pnl_today == pytest.approx(0.5)
    assert rc.sl_count_today == 0  # malformed pnl row skipped entirely


def test_corrupt_state_file_recovers(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.state_path.write_text("{{{ not json", encoding="utf-8")
    rc = RiskControls(cfg, now_fn=FakeClock())
    assert rc.entry_times == {}
    assert rc.entry_block_reason() is None


def test_utc_rollover_resets_counters_keeps_entries(tmp_path):
    clock = FakeClock()
    rc = make_rc(tmp_path, clock=clock, max_sl_per_day=1,
                 daily_loss_limit_usdt=0.5, time_stop_min=10_000.0)
    rc.on_entry("BTCUSDT")
    rc.on_close("ETHUSDT", "EXIT_SL pnl=-1", -1.0)
    assert rc.entry_block_reason() is not None
    clock.advance(86400)  # cross UTC midnight
    assert rc.entry_block_reason() is None
    assert rc.sl_count_today == 0
    assert rc.realized_pnl_today == 0.0
    assert "BTCUSDT" in rc.entry_times  # held position survives rollover


def test_save_failure_does_not_break_logic(tmp_path, monkeypatch):
    rc = make_rc(tmp_path, max_sl_per_day=1)
    monkeypatch.setattr(rc, "_save_state",
                        lambda: (_ for _ in ()).throw(OSError("disk full")))
    rc.on_close("X", "EXIT_SL pnl=-1", -1.0)  # must not raise
    assert rc.entry_block_reason().startswith("v2_max_sl_per_day")


def test_snapshot_shape(tmp_path):
    rc = make_rc(tmp_path, time_stop_min=5.0)
    rc.on_entry("BTCUSDT")
    snap = rc.snapshot()
    assert snap["open_tracked"] == ["BTCUSDT"]
    assert snap["sl_count_today"] == 0
    assert "config" in snap and snap["entry_block"] is None
