#!/usr/bin/env python3
r"""sim_exits.py - READ-ONLY counterfactual exit-grid replay.

Replays CLOSED paper round-trips against the tick-level price history in
live_signals.csv under a grid of TP / SL / time-stop settings and reports what
the P&L WOULD have been. Writes nothing, trades nothing, changes no settings,
never touches the exchange.

    ~/bot/.venv/bin/python ~/bot/tools/sim_exits.py
    ~/bot/.venv/bin/python ~/bot/tools/sim_exits.py --current-only
    ~/bot/.venv/bin/python ~/bot/tools/sim_exits.py --since fd8128b --symbols ETHUSDT,SOLUSDT
    ~/bot/.venv/bin/python ~/bot/tools/sim_exits.py --sessions _arch_fd8128b_20260612_1232
    ~/bot/.venv/bin/python ~/bot/tools/sim_exits.py --tp 0.003,0.005 --sl 0.005 --time-stops 0,60

Inputs (scanned in logs/ and every logs/_arch*/ session directory):
  trades_paper_*.csv   fill log (entries carry the entry timestamp + fill)
  trades_closed_*.csv  closed trades (used to RECONCILE the reconstruction)
  live_signals.csv     per-symbol tick prices (the same px the executor saw)

Session quality (computed, not assumed):
  Each session is reconciled against its own trades_closed book using the
  executor's exact cost model. Sessions whose recorded book disagrees badly
  with the recomputed one (e.g. trades recorded BEFORE fees/slippage existed)
  are classified INVALID and EXCLUDED from the grid by default - their trades
  were booked under different rules and would poison the comparison.
  Use --include-invalid to force them in.

Filtering:
  --current-only        only the live logs/ session; ignore all archives
  --sessions a,b        only the named session dirs ("current" = logs/)
  --since X             only trades entered at/after X. X is a UTC timestamp
                        ("2026-06-12 12:32", "2026-06-12") or a token looked
                        up in logs/DEPLOY_MARKERS.txt (e.g. "fd8128b")
  --symbols A,B         only these symbols
  --include-invalid     include sessions that fail cost-model reconciliation

How the replay works:
  1. Round-trips are rebuilt per symbol from the fill log (BUY/SELL_SHORT open
     or scale-in; SELL/BUY_TO_COVER close), reproducing the executor's
     avg-price math.
  2. Each trade is replayed tick-by-tick from its entry under every grid cell.
     Exits considered, in priority order on each tick:
       SL hit (conservative: checked before TP on the same tick)
       TP hit
       time-stop expiry (exit at that tick's price)
       the trade's ACTUAL historical exit time/fill (flip/restart/etc.) -
       entries and flips are HELD FIXED; only TP/SL/time-stop vary.

HONESTY CAVEATS (also printed in the report):
  - COUNTERFACTUAL: entries and flip exits are frozen history. A different
    TP/SL would have changed position state, cooldowns, flip confirmation and
    therefore SUBSEQUENT entries. This is per-trade replay, not a full resim.
  - Prices are ~3s ticks of the signal px, not OHLC: moves between ticks are
    invisible, and barrier fills happen at the NEXT observed tick price
    (same close-based behavior as the live executor, which can overshoot).
  - PF on small samples is noise. Below MIN_JUDGE_N trades the tool refuses
    to rank the grid as a recommendation.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "logs"
DEPLOY_MARKERS = LOGS / "DEPLOY_MARKERS.txt"

OPEN_ACTIONS = {"BUY": "long", "SELL_SHORT": "short"}
CLOSE_ACTIONS = {"SELL", "BUY_TO_COVER"}

MIN_JUDGE_N = 30          # below this many trades the grid is not a recommendation
RECON_OK_RATIO = 0.05     # |recomputed-recorded|/scale below this = clean books
RECON_INVALID_RATIO = 0.25  # above this = different booking rules -> exclude


# ---------------------------------------------------------------------------
# Cost model - mirrors tools/live_executor.py exactly
# ---------------------------------------------------------------------------

def apply_slippage(price: float, action: str, slippage_bps: float) -> float:
    if price <= 0 or not slippage_bps:
        return price
    s = float(slippage_bps) / 1e4
    a = (action or "").upper()
    if a in ("BUY", "BUY_TO_COVER"):
        return price * (1.0 + s)
    if a in ("SELL", "SELL_SHORT"):
        return price * (1.0 - s)
    return price


def fee_cost(notional: float, fee_bps: float) -> float:
    return abs(notional) * (float(fee_bps) / 1e4)


def net_pnl(side: str, avg: float, qty: float, exit_fill: float, fee_bps: float) -> float:
    """Net P&L given an already-slipped exit fill (executor's net_pnl_on_close)."""
    gross = (exit_fill - avg) * qty if side == "long" else (avg - exit_fill) * qty
    return gross - fee_cost(avg * qty, fee_bps) - fee_cost(exit_fill * qty, fee_bps)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_ts(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_since(spec: str, markers_path: Path = DEPLOY_MARKERS) -> Optional[datetime]:
    """Parse --since: a UTC timestamp, or a token found in DEPLOY_MARKERS.txt.

    Marker lines look like:  2026-06-12 12:32:59 UTC fd8128b deployed: ...
    The FIRST line containing the token wins.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            dt = datetime.strptime(spec, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    if markers_path.exists():
        for line in markers_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if spec in line:
                # leading "YYYY-MM-DD HH:MM:SS" (a trailing " UTC" is implied)
                head = " ".join(line.strip().split(" ")[:2])
                try:
                    return datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=timezone.utc)
                except ValueError:
                    continue
    return None


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def read_csv_rows(path: Path) -> List[dict]:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


@dataclass
class Trade:
    symbol: str
    side: str            # long | short
    entry_dt: datetime
    avg: float           # slipped average entry fill (executor math)
    qty: float
    close_dt: datetime
    actual_exit_fill: float  # recorded (already slipped) exit fill
    actual_reason: str
    session: str


@dataclass
class Session:
    name: str
    px: Dict[str, Tuple[List[datetime], List[float]]] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    recorded_net: float = 0.0
    recorded_n: int = 0
    orphan_closes: int = 0
    recomputed_net: float = 0.0
    klass: str = ""      # CURRENT | ARCHIVE | INVALID


def load_session(d: Path) -> Optional[Session]:
    fills: List[dict] = []
    for f in sorted(d.glob("trades_paper_*.csv")):
        fills.extend(read_csv_rows(f))
    sig_rows = read_csv_rows(d / "live_signals.csv")
    if not fills:
        return None
    sess = Session(name=d.name if d != LOGS else "current")

    # tick price series per symbol
    tmp: Dict[str, List[Tuple[datetime, float]]] = {}
    for r in sig_rows:
        t = parse_ts(r.get("ts", ""))
        px = _f(r.get("px"))
        if t and px and px > 0:
            tmp.setdefault(r.get("symbol", "?"), []).append((t, px))
    for sym, seq in tmp.items():
        seq.sort(key=lambda x: x[0])
        sess.px[sym] = ([t for t, _ in seq], [p for _, p in seq])

    # rebuild round-trips from the fill log (executor avg-price math)
    fills_p = []
    for r in fills:
        t = parse_ts(r.get("ts", ""))
        px = _f(r.get("price"))
        q = _f(r.get("qty"))
        if t and px and q:
            fills_p.append((t, r.get("symbol", "?"), (r.get("side") or "").upper(),
                            px, q, r.get("reason", "")))
    fills_p.sort(key=lambda x: x[0])

    pos: Dict[str, dict] = {}
    for t, sym, action, px, q, reason in fills_p:
        if action in OPEN_ACTIONS:
            want = OPEN_ACTIONS[action]
            p = pos.get(sym)
            if p is None or p["side"] != want:
                pos[sym] = {"side": want, "qty": q, "avg": px, "entry_dt": t}
            else:  # scale-in keeps the original entry time
                nq = p["qty"] + q
                p["avg"] = (p["avg"] * p["qty"] + px * q) / nq
                p["qty"] = nq
        elif action in CLOSE_ACTIONS:
            p = pos.pop(sym, None)
            if p is None:
                sess.orphan_closes += 1   # e.g. restart-close of a pre-log position
                continue
            sess.trades.append(Trade(sym, p["side"], p["entry_dt"], p["avg"],
                                     p["qty"], t, px, reason, sess.name))

    for f in sorted(d.glob("trades_closed_*.csv")):
        for r in read_csv_rows(f):
            v = _f(r.get("realized_pnl"))
            if v is not None:
                sess.recorded_net += v
                sess.recorded_n += 1
    return sess


def classify_session(recomputed: float, recorded: float, recorded_n: int,
                     is_current: bool) -> str:
    """CURRENT / ARCHIVE (books consistent) / INVALID (different booking rules).

    The ratio compares the session's recorded P&L book against the same trades
    re-priced under the CURRENT cost model. A big gap means the session was
    booked under different rules (e.g. the pre-fee era) and must not be mixed
    into a grid that assumes today's costs.
    """
    if is_current:
        return "CURRENT"
    if recorded_n == 0:
        return "ARCHIVE"
    scale = max(abs(recomputed), abs(recorded), 1.0)
    ratio = abs(recomputed - recorded) / scale
    if ratio > RECON_INVALID_RATIO:
        return "INVALID"
    return "ARCHIVE"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(tr: Trade, px: Tuple[List[datetime], List[float]],
           tp: float, sl: float, tstop_min: float,
           fee_bps: float, slip_bps: float) -> Tuple[float, str, datetime]:
    """Return (net_pnl, exit_reason, exit_dt) for one trade under one grid cell."""
    times, prices = px
    i = bisect_right(times, tr.entry_dt)
    deadline = tr.entry_dt + timedelta(minutes=tstop_min) if tstop_min > 0 else None
    close_action = "SELL" if tr.side == "long" else "BUY_TO_COVER"

    while i < len(times):
        t, mid = times[i], prices[i]
        if t >= tr.close_dt:
            break  # historical exit fires first -> actual fill below
        if tr.side == "long":
            hit_sl = sl > 0 and mid <= tr.avg * (1 - sl)
            hit_tp = tp > 0 and mid >= tr.avg * (1 + tp)
        else:
            hit_sl = sl > 0 and mid >= tr.avg * (1 + sl)
            hit_tp = tp > 0 and mid <= tr.avg * (1 - tp)
        if hit_sl:  # conservative: SL wins a same-tick tie
            fill = apply_slippage(mid, close_action, slip_bps)
            return net_pnl(tr.side, tr.avg, tr.qty, fill, fee_bps), "SL", t
        if hit_tp:
            fill = apply_slippage(mid, close_action, slip_bps)
            return net_pnl(tr.side, tr.avg, tr.qty, fill, fee_bps), "TP", t
        if deadline is not None and t >= deadline:
            fill = apply_slippage(mid, close_action, slip_bps)
            return net_pnl(tr.side, tr.avg, tr.qty, fill, fee_bps), "TIME", t
        i += 1

    # no barrier hit -> the trade exits exactly as it did historically
    return (net_pnl(tr.side, tr.avg, tr.qty, tr.actual_exit_fill, fee_bps),
            "FLIP", tr.close_dt)


def grid_stats(results: List[Tuple[float, str, datetime]]):
    pnls = [(dt, p, r) for p, r, dt in results]
    pnls.sort(key=lambda x: x[0])
    wins = [p for _, p, _ in pnls if p > 0]
    losses = [p for _, p, _ in pnls if p < 0]
    cum = peak = 0.0
    maxdd = 0.0
    for _, p, _ in pnls:
        cum += p
        peak = max(peak, cum)
        maxdd = min(maxdd, cum - peak)
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    n = len(pnls)
    return {
        "n": n,
        "tp": sum(1 for _, _, r in pnls if r == "TP"),
        "sl": sum(1 for _, _, r in pnls if r == "SL"),
        "time": sum(1 for _, _, r in pnls if r == "TIME"),
        "flip": sum(1 for _, _, r in pnls if r == "FLIP"),
        "win_rate": 100.0 * len(wins) / n if n else 0.0,
        "avg_win": st.fmean(wins) if wins else 0.0,
        "avg_loss": st.fmean(losses) if losses else 0.0,
        "net": sum(p for _, p, _ in pnls),
        "pf": pf,
        "maxdd": maxdd,
    }


def fmt_row(label: str, s: dict, mark: str = "") -> str:
    pf = f"{s['pf']:6.3f}" if math.isfinite(s["pf"]) else "   inf"
    return (f"  {label:<22} n={s['n']:<3} TP={s['tp']:<3} SL={s['sl']:<3} "
            f"TIME={s['time']:<3} FLIP={s['flip']:<3} win={s['win_rate']:5.1f}% "
            f"avgW={s['avg_win']:+8.4f} avgL={s['avg_loss']:+8.4f} "
            f"net={s['net']:+9.4f} PF={pf} maxDD={s['maxdd']:+8.4f}{mark}")


def parse_list(s: str) -> List[float]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="READ-ONLY counterfactual TP/SL/time-stop grid replay")
    ap.add_argument("--tp", default="0.003,0.005,0.0075,0.01,0.015",
                    help="comma list of TP fractions (0.005 = 0.5%%); 0 disables TP")
    ap.add_argument("--sl", default="0.003,0.005,0.0075,0.01",
                    help="comma list of SL fractions; 0 disables SL")
    ap.add_argument("--time-stops", default="0,60,180",
                    help="comma list of time-stops in MINUTES; 0 = no time-stop")
    ap.add_argument("--current-only", action="store_true",
                    help="only the live logs/ session; ignore all archives")
    ap.add_argument("--sessions", default="",
                    help="comma list of session dir names to include ('current' = logs/)")
    ap.add_argument("--since", default="",
                    help="UTC timestamp ('2026-06-12 12:32') or DEPLOY_MARKERS.txt token "
                         "(e.g. 'fd8128b'); only trades ENTERED at/after this moment")
    ap.add_argument("--symbols", default="", help="comma list of symbols to include")
    ap.add_argument("--include-invalid", action="store_true",
                    help="also replay sessions that FAIL cost-model reconciliation")
    ap.add_argument("--dirs", default="", help="extra session dirs (comma list)")
    ap.add_argument("--fee-bps", type=float,
                    default=float(os.getenv("EXEC_FEE_BPS", "5")))
    ap.add_argument("--slip-bps", type=float,
                    default=float(os.getenv("EXEC_SLIPPAGE_BPS", "2")))
    ap.add_argument("--top", type=int, default=20, help="show top N grid rows by PF")
    args = ap.parse_args()

    print("=" * 100)
    print("  COUNTERFACTUAL EXIT-GRID REPLAY (read-only; changes nothing; trades nothing)")
    print("=" * 100)
    print("  WARNING: counterfactual. Entries and flip exits are frozen history; a different")
    print("  TP/SL would have changed later entries. Tick px (no OHLC): barrier fills happen at")
    print("  the next observed tick, like the live close-based executor. Small n => PF is noisy.")

    # --- discover sessions ----------------------------------------------------
    dirs = [LOGS]
    if not args.current_only:
        dirs += sorted(p for p in LOGS.glob("_arch*") if p.is_dir())
        for extra in str(args.dirs).split(","):
            extra = extra.strip()
            if extra:
                p = Path(extra)
                dirs.append(p if p.is_absolute() else BASE / p)

    wanted = {s.strip() for s in str(args.sessions).split(",") if s.strip()}
    sessions: List[Session] = []
    for d in dirs:
        if not d.is_dir():
            continue
        s = load_session(d)
        if not (s and s.trades):
            continue
        if wanted and s.name not in wanted:
            continue
        sessions.append(s)
    if not sessions:
        print("\n  No round-trips found (after session filters). Check --sessions/--current-only.")
        return 1

    fee, slip = args.fee_bps, args.slip_bps
    since = parse_since(args.since)
    if args.since and since is None:
        print(f"\n  ERROR: --since {args.since!r} is neither a timestamp nor a token "
              f"found in {DEPLOY_MARKERS}")
        return 1
    symbols = {s.strip().upper() for s in str(args.symbols).split(",") if s.strip()}

    print(f"\n  cost model: fee={fee:g} bps/side  slippage={slip:g} bps/side "
          f"(override with --fee-bps/--slip-bps)")
    filt = []
    if args.current_only:
        filt.append("current-only")
    if wanted:
        filt.append(f"sessions={','.join(sorted(wanted))}")
    if since:
        filt.append(f"since={since:%Y-%m-%d %H:%M:%S} UTC")
    if symbols:
        filt.append(f"symbols={','.join(sorted(symbols))}")
    if args.include_invalid:
        filt.append("include-invalid")
    print(f"  filters: {'; '.join(filt) if filt else '(none)'}")
    if DEPLOY_MARKERS.exists():
        print("  deploy markers:")
        for line in DEPLOY_MARKERS.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                print(f"    | {line.strip()}")

    # --- reconciliation + classification --------------------------------------
    print("\n" + "-" * 100)
    print("  SESSIONS (reconstructed vs recorded books; INVALID = booked under different")
    print("  rules, e.g. the pre-fee era - EXCLUDED from the grid unless --include-invalid)")
    print("-" * 100)
    for s in sessions:
        s.recomputed_net = sum(net_pnl(t.side, t.avg, t.qty, t.actual_exit_fill, fee)
                               for t in s.trades)
        s.klass = classify_session(s.recomputed_net, s.recorded_net, s.recorded_n,
                                   s.name == "current")
    order = {"CURRENT": 0, "ARCHIVE": 1, "INVALID": 2}
    for s in sorted(sessions, key=lambda x: (order.get(x.klass, 9), x.name)):
        excl = "  [EXCLUDED]" if (s.klass == "INVALID" and not args.include_invalid) else ""
        print(f"  [{s.klass:<7}] {s.name:<40} trades={len(s.trades):<4} "
              f"recorded={s.recorded_n:<4} net(recomputed)={s.recomputed_net:+9.4f} "
              f"net(recorded)={s.recorded_net:+9.4f} orphans={s.orphan_closes}{excl}")

    # --- trade filters ----------------------------------------------------------
    all_trades: List[Trade] = []
    by_name = {s.name: s for s in sessions}
    for s in sessions:
        if s.klass == "INVALID" and not args.include_invalid:
            continue
        all_trades.extend(s.trades)

    n_before = len(all_trades)
    if since:
        all_trades = [t for t in all_trades if t.entry_dt >= since]
    if symbols:
        all_trades = [t for t in all_trades if t.symbol.upper() in symbols]

    replayable: List[Tuple[Trade, Tuple[List[datetime], List[float]]]] = []
    skipped = 0
    for t in all_trades:
        px = by_name[t.session].px.get(t.symbol)
        if px and px[0] and px[0][0] <= t.close_dt and t.entry_dt <= px[0][-1]:
            replayable.append((t, px))
        else:
            skipped += 1
    print(f"\n  trades: {n_before} eligible -> {len(all_trades)} after --since/--symbols "
          f"-> {len(replayable)} replayable ({skipped} skipped: no px coverage)")
    if not replayable:
        print("  Nothing to replay under these filters.")
        return 1

    n = len(replayable)
    too_small = n < MIN_JUDGE_N

    # --- baseline ---------------------------------------------------------------
    base = grid_stats([(net_pnl(t.side, t.avg, t.qty, t.actual_exit_fill, fee),
                        "FLIP" if not t.actual_reason.startswith("EXIT_")
                        else t.actual_reason.replace("EXIT_", ""), t.close_dt)
                       for t, _ in replayable])
    print("\n" + "-" * 100 + "\n  BASELINE (what actually happened, same trades)\n" + "-" * 100)
    print(fmt_row("ACTUAL HISTORY", base))

    # --- the grid -----------------------------------------------------------------
    tps, sls, tss = parse_list(args.tp), parse_list(args.sl), parse_list(args.time_stops)
    cur_tp = float(os.getenv("EXEC_TP_PCT", "0.005"))
    cur_sl = float(os.getenv("EXEC_SL_PCT", "0.005"))
    if cur_tp not in tps:
        tps.append(cur_tp)
    if cur_sl not in sls:
        sls.append(cur_sl)

    rows = []
    for tp in sorted(set(tps)):
        for sl in sorted(set(sls)):
            for ts_min in sorted(set(tss)):
                res = [replay(t, px, tp, sl, ts_min, fee, slip)
                       for t, px in replayable]
                rows.append(((tp, sl, ts_min), grid_stats(res)))

    def label(tp, sl, ts_min):
        t = f"{ts_min:g}m" if ts_min > 0 else "-"
        return f"TP {100*tp:.2f}% SL {100*sl:.2f}% ts {t}"

    def is_cur(tp, sl, ts_min):
        return abs(tp - cur_tp) < 1e-12 and abs(sl - cur_sl) < 1e-12 and ts_min == 0

    def is_old(tp, sl, ts_min):
        return abs(tp - 0.015) < 1e-12 and abs(sl - 0.010) < 1e-12 and ts_min == 0

    if too_small:
        # No ranking: cell order, no "best" verdict. The grid is data, not advice.
        print("\n" + "-" * 100)
        print(f"  GRID - TOO SMALL TO JUDGE: n={n} < {MIN_JUDGE_N}. Shown in parameter order,")
        print("  NOT ranked, NO recommendation. Collect more trades before acting on this.")
        print("-" * 100)
        for (tp, sl, ts_min), stats in rows:
            mark = "   <- CURRENT" if is_cur(tp, sl, ts_min) else (
                   "   <- OLD (1.5/1.0)" if is_old(tp, sl, ts_min) else "")
            print(fmt_row(label(tp, sl, ts_min), stats, mark))
        print("\n  REMINDER: counterfactual AND under-sampled. No conclusion is supported yet.")
        print("=" * 100)
        return 0

    rows.sort(key=lambda r: (-(r[1]["pf"] if math.isfinite(r[1]["pf"]) else 1e9),
                             -r[1]["net"]))
    print("\n" + "-" * 100 + f"\n  GRID (sorted by PF; top {args.top} of {len(rows)}; "
          "current and old settings always shown)\n" + "-" * 100)
    shown = 0
    for (tp, sl, ts_min), stats in rows:
        mark = "   <- CURRENT" if is_cur(tp, sl, ts_min) else (
               "   <- OLD (1.5/1.0)" if is_old(tp, sl, ts_min) else "")
        if shown < args.top or mark:
            print(fmt_row(label(tp, sl, ts_min), stats, mark))
            shown += 1

    best_cell, best = rows[0]
    print("\n" + "-" * 100)
    print(f"  BEST BY PF: {label(*best_cell)}  PF={best['pf'] if math.isfinite(best['pf']) else float('inf'):.3f} "
          f"net={best['net']:+.4f}  (baseline PF={base['pf']:.3f} net={base['net']:+.4f})")
    print(f"  Per-symbol net at best cell:")
    by_sym: Dict[str, List[Tuple[float, str, datetime]]] = {}
    for t, px in replayable:
        by_sym.setdefault(t.symbol, []).append(
            replay(t, px, best_cell[0], best_cell[1], best_cell[2], fee, slip))
    for sym in sorted(by_sym):
        s = grid_stats(by_sym[sym])
        print(fmt_row(f"  {sym}", s))
    print("\n  REMINDER: counterfactual + small sample. Use this to pick ONE candidate setting")
    print("  and validate it forward in paper - do not chase the single best grid cell.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
