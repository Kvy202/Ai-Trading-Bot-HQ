"""Tests for tools/v2_evidence_export.py — synthetic CSVs, hand-computed metrics."""

import csv
import json

import pytest

from tools import v2_evidence_export as ev

HEADER = ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price",
          "realized_pnl", "reason"]


def write_day(logs_dir, day, rows):
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"trades_closed_{day}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r)
    return path


def run(tmp_path, *extra):
    logs = tmp_path / "logs"
    out = tmp_path / "reports" / "evidence"
    return ev.main(["--logs-dir", str(logs), "--out-dir", str(out), "--quiet", *extra]), out


def test_known_day_metrics(tmp_path):
    day = "20260601"
    write_day(tmp_path / "logs", day, [
        ["2026-06-01 01:00:00+0000", "BTCUSDT", "SELL", 1, 100, 101, 1.0, "EXIT_TP pnl=1.0"],
        ["2026-06-01 02:00:00+0000", "BTCUSDT", "SELL", 1, 100, 99.5, -0.5, "EXIT_SL pnl=-0.5"],
        ["2026-06-01 03:00:00+0000", "ETHUSDT", "BUY_TO_COVER", 1, 100, 100.5, -0.5, "FLIP_CLOSE pnl=-0.5"],
        ["2026-06-01 04:00:00+0000", "ETHUSDT", "SELL", 1, 100, 102, 2.0, "EXIT_TP pnl=2.0"],
    ])
    code, out = run(tmp_path, "--date", day)
    assert code == 0
    s = json.loads((out / day / "summary.json").read_text(encoding="utf-8"))
    assert s["trades"] == 4
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["net_pnl"] == pytest.approx(2.0)
    assert s["profit_factor"] == pytest.approx(3.0)       # 3.0 / |-1.0|
    assert s["max_intraday_dd"] == pytest.approx(1.0)     # cum 1.0,0.5,0.0,2.0 -> peak 1, trough 0
    assert s["exit_reasons"]["EXIT_TP"] == {"count": 2, "net_pnl": 3.0}
    assert s["exit_reasons"]["EXIT_SL"]["count"] == 1
    assert s["per_symbol"]["BTCUSDT"] == {"trades": 2, "net_pnl": 0.5, "win_rate": 0.5}
    assert s["per_symbol"]["ETHUSDT"]["net_pnl"] == pytest.approx(1.5)
    assert s["skipped_rows"] == 0


def test_all_winners_pf_inf(tmp_path):
    day = "20260602"
    write_day(tmp_path / "logs", day, [
        ["t1", "BTCUSDT", "SELL", 1, 100, 101, 1.0, "EXIT_TP"],
        ["t2", "BTCUSDT", "SELL", 1, 100, 102, 2.0, "EXIT_TP"],
    ])
    code, out = run(tmp_path, "--date", day)
    assert code == 0
    s = json.loads((out / day / "summary.json").read_text(encoding="utf-8"))
    assert s["profit_factor"] is None
    assert "inf" in s["profit_factor_note"]
    assert s["max_intraday_dd"] == 0.0
    assert s["win_rate"] == 1.0


def test_all_losers(tmp_path):
    day = "20260603"
    write_day(tmp_path / "logs", day, [
        ["t1", "BTCUSDT", "SELL", 1, 100, 99, -1.0, "EXIT_SL"],
        ["t2", "BTCUSDT", "SELL", 1, 100, 99, -1.0, "EXIT_SL"],
    ])
    code, out = run(tmp_path, "--date", day)
    assert code == 0
    s = json.loads((out / day / "summary.json").read_text(encoding="utf-8"))
    assert s["win_rate"] == 0.0
    assert s["profit_factor"] == 0.0
    assert s["max_intraday_dd"] == pytest.approx(2.0)


def test_header_only_day(tmp_path):
    day = "20260604"
    write_day(tmp_path / "logs", day, [])
    code, out = run(tmp_path, "--date", day)
    assert code == 0
    s = json.loads((out / day / "summary.json").read_text(encoding="utf-8"))
    assert s["trades"] == 0 and s["win_rate"] is None and s["net_pnl"] == 0.0


def test_out_of_order_ts_sorted_for_dd(tmp_path):
    # File order: t3, t1, t2. Sorted cum: +2.0 -> 1.5 -> 0.5 => max DD 1.5.
    # Unsorted DD would be 1.0 — asserting 1.5 proves rows get ts-sorted.
    day = "20260605"
    write_day(tmp_path / "logs", day, [
        ["2026-06-05 03:00:00+0000", "X", "SELL", 1, 1, 1, -1.0, "EXIT_SL"],
        ["2026-06-05 01:00:00+0000", "X", "SELL", 1, 1, 1, 2.0, "EXIT_TP"],
        ["2026-06-05 02:00:00+0000", "X", "SELL", 1, 1, 1, -0.5, "FLIP_CLOSE"],
    ])
    code, out = run(tmp_path, "--date", day)
    assert code == 0
    s = json.loads((out / day / "summary.json").read_text(encoding="utf-8"))
    assert s["max_intraday_dd"] == pytest.approx(1.5)


def test_malformed_row_skipped(tmp_path):
    day = "20260606"
    write_day(tmp_path / "logs", day, [
        ["t1", "X", "SELL", 1, 1, 1, "not-a-number", "EXIT_SL"],
        ["t2", "X", "SELL", 1, 1, 1, 0.5, "EXIT_TP"],
    ])
    code, out = run(tmp_path, "--date", day)
    assert code == 0
    s = json.loads((out / day / "summary.json").read_text(encoding="utf-8"))
    assert s["trades"] == 1 and s["skipped_rows"] == 1
    assert s["net_pnl"] == pytest.approx(0.5)


def test_missing_date_exits_2(tmp_path):
    (tmp_path / "logs").mkdir()
    code, _ = run(tmp_path, "--date", "19990101")
    assert code == 2


def test_bad_date_format_exits_1(tmp_path):
    code, _ = run(tmp_path, "--date", "2026-06-01")
    assert code == 1


def test_index_accumulates_days(tmp_path):
    write_day(tmp_path / "logs", "20260601",
              [["t1", "X", "SELL", 1, 1, 1, 1.0, "EXIT_TP"]])
    write_day(tmp_path / "logs", "20260602",
              [["t1", "X", "SELL", 1, 1, 1, -1.0, "EXIT_SL"]])
    code1, out = run(tmp_path, "--date", "20260601")
    code2, _ = run(tmp_path, "--date", "20260602")
    assert code1 == code2 == 0
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert set(index) == {"20260601", "20260602"}
    assert index["20260601"]["net_pnl"] == pytest.approx(1.0)


def test_all_flag_exports_every_day(tmp_path):
    for d in ("20260601", "20260602", "20260603"):
        write_day(tmp_path / "logs", d, [["t1", "X", "SELL", 1, 1, 1, 0.1, "EXIT_TP"]])
    code, out = run(tmp_path, "--all")
    assert code == 0
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 3


def test_summary_md_headlines(tmp_path):
    day = "20260607"
    write_day(tmp_path / "logs", day, [
        ["t1", "BTCUSDT", "SELL", 1, 100, 101, 1.0, "EXIT_TP pnl=1.0"],
        ["t2", "ETHUSDT", "SELL", 1, 100, 99, -0.4, "EXIT_SL pnl=-0.4"],
    ])
    code, out = run(tmp_path, "--date", day)
    assert code == 0
    md = (out / day / "summary.md").read_text(encoding="utf-8")
    assert f"# Evidence — {day}" in md
    assert "EXIT_TP" in md and "EXIT_SL" in md
    assert "BTCUSDT" in md and "ETHUSDT" in md
    assert "| 2 |" in md  # trades headline
