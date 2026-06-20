#!/usr/bin/env python3
r"""test_sim_exits.py - offline tests for the counterfactual exit replay.

Builds a synthetic session (fill log + signals + closed book) in a temp dir
and asserts hand-computed outcomes: TP hit, SL hit, time-stop, flip fallback,
scale-in average math, and the fee/slippage cost model.

Run:  python tools/test_sim_exits.py   ->  must end RESULT: PASS
"""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import timezone as _tz  # noqa: E402

from tools.sim_exits import (  # noqa: E402
    classify_session,
    load_session,
    net_pnl,
    parse_since,
    replay,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def ts(s):
    return f"2026-06-12 {s}+0000"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    print("=" * 70 + "\n  SIM_EXITS REPLAY TESTS\n" + "=" * 70)
    tmp = Path(tempfile.mkdtemp(prefix="sim_exits_test_"))
    try:
        # --- synthetic session ------------------------------------------------
        write_csv(tmp / "trades_paper_20260612.csv",
                  ["ts", "symbol", "side", "price", "qty", "reason", "mode", "order_id"],
                  [
                      # long: entry 100, flips out at 100.1 ten minutes later
                      [ts("12:00:00"), "ETHUSDT", "BUY", 100.0, 1.0, "ENTRY", "PAPER", "p1"],
                      [ts("12:10:00"), "ETHUSDT", "SELL", 100.1, 1.0, "FLIP_CLOSE", "PAPER", "p2"],
                      # short: entry 200 qty .5, flips out at 201 (a loser)
                      [ts("13:00:00"), "SOLUSDT", "SELL_SHORT", 200.0, 0.5, "ENTRY", "PAPER", "p3"],
                      [ts("13:05:00"), "SOLUSDT", "BUY_TO_COVER", 201.0, 0.5, "FLIP_CLOSE", "PAPER", "p4"],
                      # scale-in: 1@100 + 1@102 -> avg 101 qty 2, close 103
                      [ts("14:00:00"), "ETHUSDT", "BUY", 100.0, 1.0, "ENTRY", "PAPER", "p5"],
                      [ts("14:01:00"), "ETHUSDT", "BUY", 102.0, 1.0, "SCALE_IN", "PAPER", "p6"],
                      [ts("14:30:00"), "ETHUSDT", "SELL", 103.0, 2.0, "FLIP_CLOSE", "PAPER", "p7"],
                  ])
        sig_rows = [[ts("12:01:00"), "ETHUSDT", 100.2],
                    [ts("12:02:00"), "ETHUSDT", 100.55],
                    [ts("12:03:00"), "ETHUSDT", 100.8],
                    [ts("12:09:00"), "ETHUSDT", 100.1],
                    [ts("13:01:00"), "SOLUSDT", 200.4],
                    [ts("13:02:00"), "SOLUSDT", 201.2],
                    [ts("13:04:00"), "SOLUSDT", 201.0],
                    [ts("14:05:00"), "ETHUSDT", 101.5],
                    [ts("14:29:00"), "ETHUSDT", 103.0]]
        write_csv(tmp / "live_signals.csv", ["ts", "symbol", "px"], sig_rows)
        write_csv(tmp / "trades_closed_20260612.csv",
                  ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price",
                   "realized_pnl", "reason"],
                  [[ts("12:10:00"), "ETHUSDT", "SELL", 1.0, 100.0, 100.1,
                    net_pnl("long", 100.0, 1.0, 100.1, 5), "FLIP_CLOSE"],
                   [ts("13:05:00"), "SOLUSDT", "BUY_TO_COVER", 0.5, 200.0, 201.0,
                    net_pnl("short", 200.0, 0.5, 201.0, 5), "FLIP_CLOSE"],
                   [ts("14:30:00"), "ETHUSDT", "SELL", 2.0, 101.0, 103.0,
                    net_pnl("long", 101.0, 2.0, 103.0, 5), "FLIP_CLOSE"]])

        # --- reconstruction ---------------------------------------------------
        sess = load_session(tmp)
        check("3 round-trips reconstructed", len(sess.trades) == 3,
              f"got {len(sess.trades)}")
        t_long = next(t for t in sess.trades if t.symbol == "ETHUSDT"
                      and t.entry_dt.hour == 12)
        t_short = next(t for t in sess.trades if t.symbol == "SOLUSDT")
        t_scale = next(t for t in sess.trades if t.symbol == "ETHUSDT"
                       and t.entry_dt.hour == 14)
        check("scale-in avg = 101, qty = 2",
              abs(t_scale.avg - 101.0) < 1e-9 and abs(t_scale.qty - 2.0) < 1e-9,
              f"avg={t_scale.avg} qty={t_scale.qty}")
        recomputed = sum(net_pnl(t.side, t.avg, t.qty, t.actual_exit_fill, 5)
                         for t in sess.trades)
        check("reconciles with trades_closed book",
              abs(recomputed - sess.recorded_net) < 1e-9,
              f"recomputed={recomputed:.6f} recorded={sess.recorded_net:.6f}")

        px_eth = sess.px["ETHUSDT"]
        px_sol = sess.px["SOLUSDT"]

        # --- replay outcomes (fee=0, slip=0 for hand math) --------------------
        pnl, reason, _ = replay(t_long, px_eth, 0.005, 0.005, 0, 0, 0)
        check("long TP 0.5%: exits at first tick >= 100.5",
              reason == "TP" and abs(pnl - 0.55) < 1e-9, f"{reason} pnl={pnl:.4f}")

        pnl, reason, _ = replay(t_long, px_eth, 0.015, 0.005, 0, 0, 0)
        check("long TP 1.5% unreachable -> falls back to actual flip",
              reason == "FLIP" and abs(pnl - 0.1) < 1e-9, f"{reason} pnl={pnl:.4f}")

        pnl, reason, exit_dt = replay(t_long, px_eth, 0.005, 0.005, 1, 0, 0)
        check("time-stop 1m fires before TP",
              reason == "TIME" and abs(pnl - 0.2) < 1e-9 and exit_dt.minute == 1,
              f"{reason} pnl={pnl:.4f}")

        pnl, reason, _ = replay(t_short, px_sol, 0.015, 0.005, 0, 0, 0)
        check("short SL 0.5%: stops at 201.2",
              reason == "SL" and abs(pnl - (-0.6)) < 1e-9, f"{reason} pnl={pnl:.4f}")

        pnl, reason, _ = replay(t_short, px_sol, 0.015, 0.010, 0, 0, 0)
        check("short SL 1.0% never hit -> actual flip loss",
              reason == "FLIP" and abs(pnl - (-0.5)) < 1e-9, f"{reason} pnl={pnl:.4f}")

        # --- cost model: fee 5 bps/side, slippage 2 bps on exit ---------------
        pnl, reason, _ = replay(t_long, px_eth, 0.005, 0.005, 0, 5, 2)
        fill = 100.55 * (1 - 0.0002)
        expected = (fill - 100.0) - (100.0 * 0.0005 + fill * 0.0005)
        check("fees+slippage match executor cost model",
              reason == "TP" and abs(pnl - expected) < 1e-9,
              f"pnl={pnl:.6f} expected={expected:.6f}")

        # --- parse_since: timestamps and deploy-marker tokens -----------------
        dt = parse_since("2026-06-12 12:32")
        check("--since parses 'YYYY-MM-DD HH:MM' as UTC",
              dt is not None and dt.tzinfo is not None
              and dt.astimezone(_tz.utc).hour == 12 and dt.minute == 32)
        check("--since parses bare date", parse_since("2026-06-12") is not None)
        markers = tmp / "DEPLOY_MARKERS.txt"
        markers.write_text(
            "2026-06-12 12:32:59 UTC fd8128b deployed: gate neutral fix\n"
            "2026-06-13 09:00:00 UTC ff1a455 deployed: sim_exits\n",
            encoding="utf-8")
        dt = parse_since("fd8128b", markers_path=markers)
        check("--since resolves a deploy-marker token",
              dt is not None and (dt.hour, dt.minute, dt.second) == (12, 32, 59),
              f"got {dt}")
        check("--since unknown token -> None",
              parse_since("deadbeef", markers_path=markers) is None)

        # --- session classification --------------------------------------------
        check("current session always CURRENT",
              classify_session(-12.6, -1.1, 363, True) == "CURRENT")
        check("consistent books -> ARCHIVE",
              classify_session(-3.23, -3.14, 82, False) == "ARCHIVE")
        check("pre-fee era books -> INVALID",
              classify_session(-12.62, -1.07, 363, False) == "INVALID")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 70)
    print(f"RESULT: {'FAIL (' + ', '.join(FAILS) + ')' if FAILS else 'PASS'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
