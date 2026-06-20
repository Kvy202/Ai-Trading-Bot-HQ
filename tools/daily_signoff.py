#!/usr/bin/env python
"""
tools/daily_signoff.py
======================
Daily paper-trade signoff report with Tier 1.5 promotion gate.

Reads from logs/trades_closed_YYYYMMDD.csv and logs/live_signals.csv.

Usage:
    python tools/daily_signoff.py                                    # last 7 days
    python tools/daily_signoff.py --days 14
    python tools/daily_signoff.py --since 2026-05-05
    python tools/daily_signoff.py --since 2026-05-05 --fix-date 2026-05-10
    python tools/daily_signoff.py --fix-datetime "2026-05-10 14:30:00"  # intraday

Promotion gate checks (all must PASS before live consideration):
  [1] No day has an allowed-signal side ratio above 85/15
  [2] Executor is not currently bias-locked
  [3] >= 100 closed trades (post-fix-datetime if given)
  [4] Net realized PnL >= 0 (post-fix or overall)
  [5] Profit factor >= 1.0 (post-fix or overall)
  [WARN] Recent-100 allowed signals not skewed >85% either side (advisory)

Exit code 0 = all gates pass, 1 = one or more fail.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_LOGS = _ROOT / "logs"

import sys as _sys
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

UTC = timezone.utc

GATE_MAX_SIDE_RATIO = 0.85
GATE_MIN_TRADES     = 100
GATE_MIN_NET_PNL    = 0.0
GATE_MIN_PF         = 1.0

WINDOW_SIZES = [100, 250, 500]


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily paper signoff + promotion gate")
    p.add_argument("--days", type=int, default=7,
                   help="Days to include (default 7, ignored when --since given)")
    p.add_argument("--since", default=None,
                   help="Include trades/signals from this date YYYY-MM-DD")
    p.add_argument("--fix-date", default=None,
                   help="Gate trade-count / PnL / PF against post-fix trades only "
                        "(YYYY-MM-DD); all trades from that day onward count")
    p.add_argument("--fix-datetime", default=None,
                   help="Intraday gate cutoff (UTC) 'YYYY-MM-DD HH:MM:SS' -- more precise "
                        "than --fix-date for mid-day fixes; supersedes --fix-date")
    p.add_argument("--quiet", action="store_true",
                   help="Print only the gate result block, not the daily table")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _parse_ts(s: str) -> datetime | None:
    """Parse UTC timestamp strings like '2026-05-10 09:42:34+0000'."""
    try:
        return datetime.fromisoformat(s.strip().replace("+0000", "+00:00"))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _date_from_stem(stem: str, prefix: str) -> date | None:
    day_str = stem[len(prefix):]
    try:
        return datetime.strptime(day_str, "%Y%m%d").date()
    except ValueError:
        return None


def load_trades(since: date) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(_LOGS.glob("trades_closed_2*.csv")):
        day = _date_from_stem(path.stem, "trades_closed_")
        if day is None or day < since:
            continue
        with open(path, encoding="utf-8") as f:
            header: list[str] | None = None
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split(",")
                if header is None:
                    header = parts
                    continue
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))
                row["_date"] = day
                row["_ts"]   = _parse_ts(row.get("ts", ""))
                rows.append(row)
    return rows


def load_signals(since: date) -> list[dict]:
    """
    Load rows from logs/live_signals.csv for the given date range.
    Uses the side_hint column (LONG/SHORT) -- the executor's computed direction --
    not re-derived from raw p_meta probability.
    Uses csv.reader so quoted fields like "lstm,tcn,tx" parse correctly.
    """
    path = _LOGS / "live_signals.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header: list[str] | None = None
        for parts in reader:
            if not parts:
                continue
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            try:
                row_date = datetime.strptime(row["ts"][:10], "%Y-%m-%d").date()
            except (KeyError, ValueError):
                continue
            if row_date < since:
                continue
            row["_date"] = row_date
            row["_ts"]   = _parse_ts(row.get("ts", ""))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def exit_type(reason: str) -> str:
    if "EXIT_TP" in reason:
        return "TP"
    if "EXIT_SL" in reason:
        return "SL"
    if "FLIP_CLOSE" in reason:
        return "FLIP"
    return "OTHER"


def profit_factor(pnls: list[float]) -> float:
    gains  = sum(p for p in pnls if p > 0)
    losses = sum(-p for p in pnls if p < 0)
    return gains / losses if losses > 0 else math.inf


def max_drawdown_abs(pnls: list[float]) -> float:
    """Largest peak-to-trough drop in cumulative PnL (returned as positive number)."""
    if not pnls:
        return 0.0
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return max_dd


def fmt(v: float, decimals: int = 4) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}"


def _safe_allow(r: dict) -> int:
    try:
        return int(r.get("allow", 0))
    except (ValueError, TypeError):
        return 0


def side_ratio(signal_rows: list[dict]) -> tuple[int, int, float]:
    """Return (long_n, short_n, long_frac) among allow=1 rows using side_hint."""
    long_n = short_n = 0
    for r in signal_rows:
        if _safe_allow(r) != 1:
            continue
        side = r.get("side_hint", "").strip().upper()
        if side == "LONG":
            long_n += 1
        elif side == "SHORT":
            short_n += 1
    total = long_n + short_n
    frac = long_n / total if total > 0 else 0.5
    return long_n, short_n, frac


def influence_shadow_summary() -> dict:
    """
    Read logs/tier2_influence_shadow.csv and compute shadow influence stats.
    Also calls backtest_shadow_impact() if bt_tier2_join.csv exists.
    Informational only - no gate impact.
    """
    result: dict = {
        "mode": "unknown",
        "total_rows": 0,
        "rows_with_delta": 0,
        "avg_delta_when_triggered": 0.0,
        "top_reasons": {},
        "backtest": None,
        "error": "",
    }

    # Read mode from run.json (daily_signoff doesn't load runtime config)
    import os
    try:
        import json as _json
        run_cfg = _json.loads((_ROOT / "config" / "run.json").read_text(encoding="utf-8"))
        result["mode"] = run_cfg.get("tier2_influence", {}).get("TIER2_INFLUENCE_MODE", "off")
    except Exception:
        result["mode"] = os.getenv("TIER2_INFLUENCE_MODE", "off")

    shadow_csv = _LOGS / "tier2_influence_shadow.csv"
    if not shadow_csv.exists():
        result["error"] = "shadow log not found (mode may be off or writer not restarted)"
        return result

    try:
        with open(shadow_csv, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        result["total_rows"] = len(rows)

        triggered = [r for r in rows if float(r.get("delta", 0) or 0) > 0]
        result["rows_with_delta"] = len(triggered)

        if triggered:
            deltas = [float(r["delta"]) for r in triggered]
            result["avg_delta_when_triggered"] = sum(deltas) / len(deltas)
            reasons: dict = {}
            for r in triggered:
                key = r.get("reason", "unknown")
                reasons[key] = reasons.get(key, 0) + 1
            result["top_reasons"] = dict(sorted(reasons.items(), key=lambda x: -x[1])[:3])

    except Exception as exc:
        result["error"] = f"shadow log read error: {exc}"
        return result

    # Backtest impact (uses bt_tier2_join.csv produced by backtest_join.py)
    try:
        from tier2.influence import backtest_shadow_impact
        bt_path = _LOGS / "bt_tier2_join.csv"
        result["backtest"] = backtest_shadow_impact(bt_path)
    except Exception as exc:
        result["backtest"] = {"error": str(exc)}

    return result


def tier2_summary() -> dict:
    """Return Tier 2 shadow runner status for informational display."""
    result: dict = {
        "runner_hb_age_s": None,
        "runner_ok": None,
        "runner_cycle": None,
        "funding_rate_rows": None,
        "open_interest_rows": None,
        "enabled": False,
        "error": "",
    }
    hb_path = _LOGS / "tier2_runner_heartbeat.json"
    if hb_path.exists():
        try:
            with open(hb_path, encoding="utf-8") as f:
                hb = json.load(f)
            ts_str = hb.get("ts", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age = (datetime.now(UTC) - ts).total_seconds()
                result["runner_hb_age_s"] = int(age)
            result["runner_ok"] = hb.get("ok")
            result["runner_cycle"] = hb.get("cycle")
            db_counts = hb.get("db_row_counts", {})
            result["funding_rate_rows"] = db_counts.get("funding_rate")
            result["open_interest_rows"] = db_counts.get("open_interest")
            result["enabled"] = True
        except Exception as exc:
            result["error"] = str(exc)
    # Also try reading DB directly if heartbeat missing or stale
    if result["funding_rate_rows"] is None:
        try:
            from tier2.feature_store import FeatureStore
            store = FeatureStore()
            counts = store.row_counts()
            result["funding_rate_rows"] = counts.get("funding_rate", 0)
            result["open_interest_rows"] = counts.get("open_interest", 0)
        except Exception:
            pass
    return result


def is_bias_locked() -> bool:
    hb = _LOGS / "heartbeat.json"
    if not hb.exists():
        return False
    try:
        with open(hb, encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("bias_locked", False))
    except Exception:
        return False


def bias_summary() -> str:
    hb = _LOGS / "heartbeat.json"
    if not hb.exists():
        return "heartbeat.json missing"
    try:
        with open(hb, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("bias_locked"):
            side = data.get("bias_side", "?")
            rl   = data.get("recent_long", "?")
            rs   = data.get("recent_short", "?")
            ts   = data.get("ts", "")
            return f"LOCKED ({side})  {rl}L/{rs}S  as of {ts}"
        event = data.get("event", "?")
        return f"clear  (last event={event})"
    except Exception as exc:
        return f"read error: {exc}"


# ---------------------------------------------------------------------------
# Recent-window side ratio
# ---------------------------------------------------------------------------

def recent_window_stats(signals: list[dict],
                        sizes: list[int]) -> list[tuple[int, int, int, float]]:
    """
    For each window size N, compute side ratio of the last N allowed-signal rows.
    Returns list of (n_requested, long_n, short_n, long_frac).
    Rows are assumed to be in chronological order (file append order).
    """
    allowed = [r for r in signals
               if _safe_allow(r) == 1
               and r.get("side_hint", "").strip().upper() in ("LONG", "SHORT")]
    results = []
    for n in sizes:
        window  = allowed[-n:]
        long_n  = sum(1 for r in window
                      if r.get("side_hint", "").strip().upper() == "LONG")
        short_n = len(window) - long_n
        total   = long_n + short_n
        frac    = long_n / total if total > 0 else 0.5
        results.append((n, long_n, short_n, frac))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    today = date.today()

    since: date
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").date()
    else:
        since = today - timedelta(days=args.days - 1)

    # Resolve the fix cutoff -- --fix-datetime takes priority over --fix-date
    fix_dt: datetime | None = None
    fix_date: date | None = None
    fix_label: str = ""
    if args.fix_datetime:
        fix_dt    = datetime.strptime(args.fix_datetime, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        fix_date  = fix_dt.date()
        fix_label = args.fix_datetime + " UTC"
    elif args.fix_date:
        fix_date  = datetime.strptime(args.fix_date, "%Y-%m-%d").date()
        fix_dt    = datetime(fix_date.year, fix_date.month, fix_date.day, tzinfo=UTC)
        fix_label = str(fix_date)

    trades  = load_trades(since)
    signals = load_signals(since)

    # Collect all days in either dataset
    all_days = sorted(
        set(r["_date"] for r in trades) | set(r["_date"] for r in signals)
    )

    # -----------------------------------------------------------------------
    # Per-day table
    # -----------------------------------------------------------------------
    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"  DAILY SIGNOFF REPORT   {since} -- {today}")
        print(f"{'='*80}")
        hdr = (f"{'DATE':12s} {'N':>5} {'NET PNL':>12} {'WIN%':>6} "
               f"{'PF':>6} {'TP_PNL':>12} {'SL_PNL':>12} "
               f"{'FLIP_PNL':>12} {'L%':>5} {'BIAS?':>5}")
        print(f"\n{hdr}")
        print("-" * 80)

    day_side_info: list[tuple[date, float, int]] = []  # (day, long_frac, total_allowed)

    for day in all_days:
        day_trades  = [r for r in trades  if r["_date"] == day]
        day_signals = [r for r in signals if r["_date"] == day]

        pnls: list[float] = []
        tp_pnl = sl_pnl = flip_pnl = 0.0
        wins = 0
        for r in day_trades:
            try:
                pnl = float(r["realized_pnl"])
            except (ValueError, KeyError):
                continue
            pnls.append(pnl)
            if pnl > 0:
                wins += 1
            et = exit_type(r.get("reason", ""))
            if et == "TP":
                tp_pnl += pnl
            elif et == "SL":
                sl_pnl += pnl
            elif et == "FLIP":
                flip_pnl += pnl

        long_n, short_n, long_frac = side_ratio(day_signals)
        total_allowed = long_n + short_n
        day_side_info.append((day, long_frac, total_allowed))

        if not args.quiet:
            n        = len(pnls)
            net      = sum(pnls)
            win_str  = f"{wins/n:.0%}" if n > 0 else "n/a"
            pf_val   = profit_factor(pnls)
            pf_str   = f"{pf_val:.2f}" if math.isfinite(pf_val) else "inf"
            long_str = f"{long_frac:.0%}" if total_allowed > 0 else "n/a"
            bias_flag = "BIAS" if max(long_frac, 1 - long_frac) > GATE_MAX_SIDE_RATIO else "ok"
            print(
                f"{day!s:12s} {n:>5d} {fmt(net):>12s} {win_str:>6s} "
                f"{pf_str:>6s} {fmt(tp_pnl):>12s} {fmt(sl_pnl):>12s} "
                f"{fmt(flip_pnl):>12s} {long_str:>5s} {bias_flag:>5s}"
            )

    # -----------------------------------------------------------------------
    # Overall totals
    # -----------------------------------------------------------------------
    all_pnls: list[float] = []
    all_tp = all_sl = all_flip = 0.0
    all_wins = 0
    for r in trades:
        try:
            pnl = float(r["realized_pnl"])
        except (ValueError, KeyError):
            continue
        all_pnls.append(pnl)
        if pnl > 0:
            all_wins += 1
        et = exit_type(r.get("reason", ""))
        if et == "TP":
            all_tp += pnl
        elif et == "SL":
            all_sl += pnl
        elif et == "FLIP":
            all_flip += pnl

    n_total   = len(all_pnls)
    net_total = sum(all_pnls)
    pf_total  = profit_factor(all_pnls)
    dd        = max_drawdown_abs(all_pnls)
    win_pct   = all_wins / n_total if n_total > 0 else 0.0

    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"  OVERALL  ({since} to {today})")
        print(f"{'='*80}")
        print(f"  Closed trades  : {n_total}")
        print(f"  Net realized   : {fmt(net_total)} USDT")
        print(f"  Win rate       : {win_pct:.1%}")
        pf_str = f"{pf_total:.3f}" if math.isfinite(pf_total) else "inf"
        print(f"  Profit factor  : {pf_str}")
        print(f"  Max drawdown   : {fmt(-dd)} USDT")
        print(f"  TP exits PnL   : {fmt(all_tp)} USDT")
        print(f"  SL exits PnL   : {fmt(all_sl)} USDT")
        print(f"  FLIP exits PnL : {fmt(all_flip)} USDT")
        print(f"  Bias state     : {bias_summary()}")

    # Post-fix subset -- uses intraday timestamp when --fix-datetime was given
    post_fix_pnls: list[float] = []
    if fix_dt is not None:
        for r in trades:
            ts = r.get("_ts")
            if ts is not None:
                if ts < fix_dt:
                    continue
            else:
                # Fall back to day-level if timestamp couldn't be parsed
                if r["_date"] < fix_date:  # type: ignore[operator]
                    continue
            try:
                post_fix_pnls.append(float(r["realized_pnl"]))
            except (ValueError, KeyError):
                pass

        if not args.quiet:
            n_pf      = len(post_fix_pnls)
            net_pf    = sum(post_fix_pnls)
            pf_pf     = profit_factor(post_fix_pnls)
            pf_pf_str = f"{pf_pf:.3f}" if math.isfinite(pf_pf) else "inf"
            print(f"\n  Post-fix trades ({fix_label}+):")
            print(f"    Closed trades  : {n_pf}")
            print(f"    Net PnL        : {fmt(net_pf)} USDT")
            print(f"    Profit factor  : {pf_pf_str}")

    # -----------------------------------------------------------------------
    # Recent signal windows
    # -----------------------------------------------------------------------
    window_stats = recent_window_stats(signals, WINDOW_SIZES)

    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"  RECENT SIGNAL WINDOWS  (allowed rows only, newest at tail)")
        print(f"{'='*80}")
        for n_req, long_n, short_n, frac in window_stats:
            total = long_n + short_n
            if total == 0:
                print(f"  Last {n_req:>4d}: no data")
                continue
            avail     = f"{total}/{n_req}" if total < n_req else str(total)
            short_frac = 1.0 - frac
            if frac > GATE_MAX_SIDE_RATIO:
                flag = "  !! LONG BIAS"
            elif short_frac > GATE_MAX_SIDE_RATIO:
                flag = "  !! SHORT BIAS"
            else:
                flag = ""
            print(f"  Last {n_req:>4d}: LONG={long_n} ({frac:.0%})  "
                  f"SHORT={short_n} ({short_frac:.0%})  n={avail}{flag}")

    # -----------------------------------------------------------------------
    # Promotion gate
    # -----------------------------------------------------------------------
    gate_pnls   = post_fix_pnls if fix_dt is not None else all_pnls
    gate_trades = len(post_fix_pnls) if fix_dt is not None else n_total
    gate_net    = sum(gate_pnls)
    gate_pf     = profit_factor(gate_pnls)
    locked      = is_bias_locked()

    gates: list[tuple[bool, str]] = []

    # Gate 1: per-day side ratio
    bad_days = [
        (d, f) for d, f, tot in day_side_info
        if tot > 0 and max(f, 1 - f) > GATE_MAX_SIDE_RATIO
    ]
    if bad_days:
        for d, f in bad_days:
            gates.append((False, f"side ratio {max(f, 1-f):.0%} on {d} (limit {GATE_MAX_SIDE_RATIO:.0%})"))
    else:
        gates.append((True, f"side ratio <= {GATE_MAX_SIDE_RATIO:.0%} on all days"))

    # Gate 2: executor bias lock
    gates.append((not locked,
                  f"executor bias {'LOCKED' if locked else 'clear'}  ({bias_summary()})"))

    # Gate 3: trade count
    label = f"post-fix ({fix_label}+)" if fix_dt is not None else "total"
    gates.append((
        gate_trades >= GATE_MIN_TRADES,
        f"{gate_trades} {label} closed trades (need {GATE_MIN_TRADES})"
    ))

    # Gate 4: net PnL
    gates.append((
        gate_net >= GATE_MIN_NET_PNL,
        f"{label} net PnL {fmt(gate_net)} USDT (need >= {GATE_MIN_NET_PNL})"
    ))

    # Gate 5: profit factor
    if gate_pnls:
        pf_str = f"{gate_pf:.3f}" if math.isfinite(gate_pf) else "inf"
        gates.append((
            not math.isfinite(gate_pf) or gate_pf >= GATE_MIN_PF,
            f"{label} profit factor {pf_str} (need >= {GATE_MIN_PF})"
        ))
    else:
        gates.append((False, "no trades to evaluate profit factor"))

    # Advisory warnings (non-blocking)
    advisories: list[str] = []
    if window_stats:
        _, long_n100, short_n100, frac100 = window_stats[0]
        total100 = long_n100 + short_n100
        if total100 > 0:
            dominant  = max(frac100, 1.0 - frac100)
            side_name = "LONG" if frac100 > 0.5 else "SHORT"
            if dominant > GATE_MAX_SIDE_RATIO:
                advisories.append(
                    f"recent-100 signals {dominant:.0%} {side_name} -- "
                    f"watch for {'LONG bias' if side_name == 'LONG' else 'over-correction'}"
                )

    all_pass = all(ok for ok, _ in gates)

    print(f"\n{'='*80}")
    print(f"  PROMOTION GATE   [{'PASS' if all_pass else 'FAIL'}]")
    print(f"{'='*80}")
    for ok, msg in gates:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {msg}")
    for msg in advisories:
        print(f"  [WARN]  {msg}")

    verdict = "READY FOR LIVE PROMOTION" if all_pass else "NOT READY -- keep paper trading"
    print(f"\n  >> {verdict}")
    print(f"{'='*80}\n")

    # -----------------------------------------------------------------------
    # Tier 2 shadow status  (informational only - no gate impact)
    # -----------------------------------------------------------------------
    t2 = tier2_summary()
    print(f"{'='*80}")
    print(f"  TIER 2 SHADOW DATA  (informational only - no gate impact)")
    print(f"{'='*80}")
    if not t2["enabled"] and t2["funding_rate_rows"] is None:
        print("  Runner not started  (run with -Tier2 to begin shadow collection)")
    else:
        hb_age = t2["runner_hb_age_s"]
        if hb_age is None:
            hb_line = "heartbeat missing"
        elif hb_age < 120:
            hb_line = f"{hb_age}s ago  (ok)"
        elif hb_age < 600:
            hb_line = f"{hb_age}s ago  (STALE)"
        else:
            hb_line = f"{hb_age}s ago  (DEAD)"
        ok_str = str(t2["runner_ok"]) if t2["runner_ok"] is not None else "n/a"
        cycle_str = str(t2["runner_cycle"]) if t2["runner_cycle"] is not None else "n/a"
        print(f"  Runner heartbeat : {hb_line}  ok={ok_str}  cycle={cycle_str}")
        fr = t2["funding_rate_rows"]
        oi = t2["open_interest_rows"]
        print(f"  funding_rate rows: {fr if fr is not None else 'n/a'}")
        print(f"  open_interest rows: {oi if oi is not None else 'n/a'}")
        if t2["error"]:
            print(f"  Error: {t2['error']}")
    print(f"{'='*80}\n")

    # -----------------------------------------------------------------------
    # Phase 2A shadow influence  (informational only - no gate impact)
    # -----------------------------------------------------------------------
    inf = influence_shadow_summary()
    print(f"{'='*80}")
    print(f"  PHASE 2A SHADOW INFLUENCE  (informational only - no gate impact)")
    print(f"{'='*80}")
    print(f"  Mode             : {inf['mode']}")
    if inf["error"] and inf["total_rows"] == 0:
        print(f"  Status           : {inf['error']}")
    else:
        total = inf["total_rows"]
        triggered = inf["rows_with_delta"]
        trigger_pct = triggered / total * 100 if total > 0 else 0.0
        avg_d = inf["avg_delta_when_triggered"]
        print(f"  Shadow log rows  : {total}")
        print(f"  Triggered (d>0)  : {triggered}  ({trigger_pct:.1f}% of ticks)")
        if triggered > 0:
            print(f"  Avg delta        : +{avg_d:.4f} threshold tightening")
            if inf["top_reasons"]:
                reasons_str = "  |  ".join(f"{r}: {n}" for r, n in inf["top_reasons"].items())
                print(f"  Top reasons      : {reasons_str}")
        if inf["error"]:
            print(f"  Warning          : {inf['error']}")

        bt = inf.get("backtest")
        if bt:
            if "error" in bt:
                print(f"  Backtest impact  : error - {bt['error']}")
            elif "note" in bt:
                n_bt = bt.get("n_matched", 0)
                sufficient = bt.get("sufficient", False)
                print(f"  Backtest matched : {n_bt} trades  "
                      f"{'(sufficient)' if sufficient else '(need >= 200 for paper gate)'}")
                print(f"  Impact analysis  : needs signal-level p_meta - run backtest_join.py separately")
            else:
                n_bt = bt.get("n_matched", 0)
                sufficient = bt.get("sufficient", False)
                would_skip = bt.get("would_skip", 0)
                skip_pct   = bt.get("skip_rate_pct", 0.0)
                pnl_kept   = bt.get("pnl_kept", 0.0)
                pnl_skipped = bt.get("pnl_skipped", 0.0)
                print(f"  Backtest matched : {n_bt} trades  "
                      f"{'(sufficient for paper gate)' if sufficient else '(need >= 200 for paper gate)'}")
                print(f"  Would skip       : {would_skip} trades  ({skip_pct:.1f}%)")
                sign_k = '+' if pnl_kept >= 0 else ''
                sign_s = '+' if pnl_skipped >= 0 else ''
                print(f"  PnL if active    : kept={sign_k}{pnl_kept:.4f}  "
                      f"skipped={sign_s}{pnl_skipped:.4f} USDT")
                if sufficient:
                    net_d = bt.get("pnl_net_delta", 0.0)
                    impact = "POSITIVE" if net_d >= 0 else "NEGATIVE"
                    print(f"  Net PnL delta    : {'+' if net_d >= 0 else ''}{net_d:.4f} USDT  [{impact}]")
                    if net_d >= 0:
                        print(f"  >> Paper gate hint: impact positive - verify with full backtest_join.py")
    print(f"{'='*80}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
