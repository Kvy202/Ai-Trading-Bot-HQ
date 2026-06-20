"""Tests for tools/v2_dashboard.py — sections render, missing inputs OK, escaping."""

import csv
import json
from datetime import datetime, timedelta, timezone

from tools import v2_dashboard as db

HEADER = ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price",
          "realized_pnl", "reason"]


def day_str(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d")


def write_closed(logs, day, rows):
    path = logs / f"trades_closed_{day}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r)


def make_full_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_closed(logs, day_str(1), [
        ["2026-06-10 01:00:00+0000", "BTCUSDT", "SELL", 1, 100, 101, 1.0, "EXIT_TP pnl=1.0"],
        ["2026-06-10 02:00:00+0000", "ETHUSDT", "SELL", 1, 100, 99, -0.4, "EXIT_SL pnl=-0.4"],
    ])
    write_closed(logs, day_str(0), [
        ["2026-06-11 01:00:00+0000", "BTCUSDT", "SELL", 1, 100, 100.2, 0.2, "FLIP_CLOSE pnl=0.2"],
    ])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    (logs / "heartbeat.json").write_text(
        json.dumps({"ts": now, "event": "idle", "mode": "PAPER"}), encoding="utf-8")
    (logs / "executor_state.json").write_text(
        json.dumps({"mode": "PAPER", "paused": False,
                    "positions": {"BTCUSDT": {"side": "long", "qty": 1, "avg": 100}}}),
        encoding="utf-8")
    (logs / "v2_risk_state.json").write_text(
        json.dumps({"utc_day": "2026-06-11", "sl_count_today": 1,
                    "realized_pnl_today": 0.8, "pnl_peak_today": 1.0,
                    "entry_times": {"BTCUSDT": 0}}), encoding="utf-8")
    with (logs / "deploy_markers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_utc", "git_sha", "branch", "note"])
        w.writerow(["2026-06-10 00:00:00+0000", "abc1234", "bot-v2-architecture", "test deploy"])
    return logs


def run(tmp_path, logs):
    out = tmp_path / "reports" / "dashboard.html"
    code = db.main(["--logs-dir", str(logs), "--out", str(out)])
    return code, out


def test_sections_present(tmp_path):
    logs = make_full_logs(tmp_path)
    code, out = run(tmp_path, logs)
    assert code == 0
    page = out.read_text(encoding="utf-8")
    for marker in ("Cumulative realized PnL", "Daily PnL", "Exit reasons",
                   "Last trades", "Deploy markers", "Executor heartbeat",
                   "V2 risk", "PAPER", "BTCUSDT", "EXIT_TP", "abc1234"):
        assert marker in page, f"missing section/marker: {marker}"


def test_svg_polyline_with_two_days(tmp_path):
    logs = make_full_logs(tmp_path)
    _, out = run(tmp_path, logs)
    page = out.read_text(encoding="utf-8")
    assert "<svg" in page and "<polyline" in page


def test_missing_inputs_render_placeholders(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()  # completely empty
    code, out = run(tmp_path, logs)
    assert code == 0
    page = out.read_text(encoding="utf-8")
    assert "n/a" in page
    assert "Daily PnL" in page


def test_reason_strings_are_escaped(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_closed(logs, day_str(0), [
        ["2026-06-11 01:00:00+0000", "BTCUSDT", "SELL", 1, 100, 101, 1.0,
         "<script>alert(1)</script>"],
    ])
    code, out = run(tmp_path, logs)
    assert code == 0
    page = out.read_text(encoding="utf-8")
    assert "<script>" not in page                 # the page ships zero JS
    assert "&lt;script&gt;" in page               # reason rendered escaped
