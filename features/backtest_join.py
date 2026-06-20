"""
features/backtest_join.py

Joins closed trades with Tier 2 features (funding rate, OI) at entry time.

For each closed trade the script finds the nearest Tier 2 snapshot that was
collected at or before the trade's entry timestamp, then measures whether
the funding rate and OI level correlate with trade outcomes (PnL).

This is a read-only analysis tool.  It does NOT write to live_signals.csv
or influence any trading decision.

Usage:
    python features/backtest_join.py
    python features/backtest_join.py --since 2026-05-01
    python features/backtest_join.py --since 2026-05-01 --out bt_tier2.csv

Output:
  - Joined CSV (--out, default logs/bt_tier2_join.csv)
  - Console correlation analysis

Columns in output CSV:
    ts, symbol, side, entry_price, exit_price, pnl,
    funding_rate, funding_age_s, oi_usd, oi_age_s, exchange
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- path setup ---
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from runtime.loader import apply_run_config as _apply_run_config
    _apply_run_config(_ROOT)
except Exception:
    pass

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass

from features.feature_store import FeatureStore

_LOGS = _ROOT / "logs"
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Trade loading
# ---------------------------------------------------------------------------

def _parse_ts(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.strip().replace("+0000", "+00:00"))
    except (ValueError, AttributeError):
        return None


def load_closed_trades(since: Optional[date] = None) -> List[Dict[str, Any]]:
    """Load all closed-trade rows from logs/trades_closed_*.csv."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(_LOGS.glob("trades_closed_2*.csv")):
        if since is not None:
            # Extract date from filename: trades_closed_YYYYMMDD.csv
            try:
                stem_date = datetime.strptime(path.stem[-8:], "%Y%m%d").date()
                if stem_date < since:
                    continue
            except ValueError:
                pass
        try:
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ts = _parse_ts(r.get("ts", ""))
                    if ts is None:
                        continue
                    if since is not None and ts.date() < since:
                        continue
                    rows.append({
                        "ts": ts,
                        "symbol": r.get("symbol", "").strip(),
                        "side": r.get("closed_side", r.get("side", "")).strip(),
                        "entry_price": _safe_float(r.get("entry_avg") or r.get("entry_price")),
                        "exit_price": _safe_float(r.get("exit_price")),
                        "pnl": _safe_float(r.get("realized_pnl")),
                        "reason": r.get("reason", "").strip(),
                    })
        except Exception as exc:
            print(f"[warn] could not read {path.name}: {exc}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Nearest-row lookup from store
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _ts_to_minute_key(dt: datetime) -> str:
    """Convert datetime to the minute-bucket key used by the store."""
    return dt.replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%MZ")


def find_nearest_before(
    store: FeatureStore,
    table: str,
    symbol: str,
    entry_ts: datetime,
    lookback_minutes: int = 60,
) -> Optional[Dict[str, Any]]:
    """Return the most recent row at or before entry_ts, within lookback window."""
    since_ts = _ts_to_minute_key(
        entry_ts - timedelta(minutes=lookback_minutes)
    )
    entry_key = _ts_to_minute_key(entry_ts)

    rows = store.get_since(table, symbol, since_ts)
    # rows are oldest-first; filter to <= entry_ts and take last
    eligible = [r for r in rows if r["ts"] <= entry_key]
    return eligible[-1] if eligible else None


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson correlation coefficient; returns None if not computable."""
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    sy = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def _quartile_split(
    pairs: List[Tuple[float, float]],
) -> List[Tuple[str, float, int]]:
    """Split (x, pnl) pairs into quartiles by x; return (label, mean_pnl, count)."""
    if len(pairs) < 4:
        return []
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    n = len(sorted_pairs)
    q = n // 4
    result = []
    boundaries = [(0, q, "Q1 (low)"), (q, 2 * q, "Q2"), (2 * q, 3 * q, "Q3"), (3 * q, n, "Q4 (high)")]
    for lo, hi, label in boundaries:
        chunk = sorted_pairs[lo:hi]
        if not chunk:
            continue
        mean_pnl = sum(p[1] for p in chunk) / len(chunk)
        result.append((label, mean_pnl, len(chunk)))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Join closed trades with Tier 2 features and analyze correlations"
    )
    p.add_argument("--since", default=None, help="Only include trades from YYYY-MM-DD onwards")
    p.add_argument(
        "--out",
        default=str(_LOGS / "bt_tier2_join.csv"),
        help="Output CSV path (default: logs/bt_tier2_join.csv)",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Max minutes before entry to search for Tier 2 data (default 60)",
    )
    p.add_argument("--no-csv", action="store_true", help="Skip CSV output, print analysis only")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    since: Optional[date] = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").date()

    # Load trades
    trades = load_closed_trades(since)
    if not trades:
        print("No closed trades found. Run with --since YYYY-MM-DD or check logs/.")
        return

    print(f"Loaded {len(trades)} closed trades.")

    # Open store
    try:
        store = FeatureStore()
        counts = store.row_counts()
        print(f"Feature store: {store.db_path()}")
        print(f"  funding_rate rows : {counts.get('funding_rate', 0)}")
        print(f"  open_interest rows: {counts.get('open_interest', 0)}")
        if all(v == 0 for v in counts.values()):
            print(
                "\n[warn] Feature store is empty. Start the shadow runner first:\n"
                "  .\\tools\\run_all.ps1 -Tier2\n"
                "Then wait 10+ minutes for data to accumulate before re-running."
            )
            return
    except Exception as exc:
        print(f"[error] could not open feature store: {exc}")
        return

    # Join each trade with nearest Tier 2 snapshot
    joined: List[Dict[str, Any]] = []
    missing_fr = missing_oi = 0

    for trade in trades:
        sym = trade["symbol"]
        ts = trade["ts"]

        fr_row = find_nearest_before(store, "funding_rate", sym, ts, args.lookback)
        oi_row = find_nearest_before(store, "open_interest", sym, ts, args.lookback)

        if fr_row is None:
            missing_fr += 1
        if oi_row is None:
            missing_oi += 1

        def _age(row: Optional[Dict], entry: datetime) -> Optional[int]:
            if row is None:
                return None
            try:
                rt = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
                return int((entry - rt).total_seconds())
            except Exception:
                return None

        joined.append({
            "ts": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
            "symbol": sym,
            "side": trade["side"],
            "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"],
            "pnl": trade["pnl"],
            "reason": trade["reason"],
            "funding_rate": fr_row.get("rate") if fr_row else None,
            "funding_age_s": _age(fr_row, ts),
            "oi_usd": oi_row.get("oi_usd") if oi_row else None,
            "oi_age_s": _age(oi_row, ts),
            "exchange": (fr_row or oi_row or {}).get("exchange", ""),
        })

    matched = len([r for r in joined if r["funding_rate"] is not None])
    print(
        f"\nJoin result: {matched}/{len(trades)} trades matched to funding data "
        f"({missing_fr} missing FR, {missing_oi} missing OI)"
    )

    # Write CSV
    if not args.no_csv and joined:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "ts", "symbol", "side", "entry_price", "exit_price", "pnl", "reason",
            "funding_rate", "funding_age_s", "oi_usd", "oi_age_s", "exchange",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(joined)
        print(f"Joined CSV: {out_path}")

    # ---------------------------------------------------------------------------
    # Correlation analysis (only on rows with Tier 2 data)
    # ---------------------------------------------------------------------------
    matched_rows = [r for r in joined if r["pnl"] is not None and r["funding_rate"] is not None]
    if not matched_rows:
        print("\nNot enough matched rows for correlation analysis.")
        return

    pnls = [r["pnl"] for r in matched_rows]
    frs = [r["funding_rate"] for r in matched_rows]

    print(f"\n{'='*60}")
    print(f"  CORRELATION ANALYSIS  ({len(matched_rows)} trades with FR data)")
    print(f"{'='*60}")

    # --- Funding rate sign split ---
    pos_fr = [r for r in matched_rows if r["funding_rate"] > 0]
    neg_fr = [r for r in matched_rows if r["funding_rate"] <= 0]
    print(f"\n  Funding Rate Sign Split:")
    for label, subset in [("positive FR (longs pay shorts)", pos_fr),
                           ("zero/negative FR (shorts pay longs)", neg_fr)]:
        if not subset:
            print(f"    {label}: no data")
            continue
        mean_pnl = sum(r["pnl"] for r in subset) / len(subset)
        wins = sum(1 for r in subset if r["pnl"] > 0)
        print(
            f"    {label}: n={len(subset)}  mean_pnl={mean_pnl:+.4f}"
            f"  win%={wins/len(subset):.0%}"
        )

    # --- Pearson correlation: funding_rate vs pnl ---
    r_fr_pnl = _pearson(frs, pnls)
    print(f"\n  Pearson corr (funding_rate vs pnl) : "
          f"{r_fr_pnl:.4f}" if r_fr_pnl is not None else "  Pearson corr: n/a (< 3 points)")

    # --- OI quartile analysis ---
    oi_matched = [r for r in matched_rows if r["oi_usd"] is not None]
    if oi_matched:
        oi_pnl_pairs = [(r["oi_usd"], r["pnl"]) for r in oi_matched]
        oi_pnls = [r["pnl"] for r in oi_matched]
        oi_vals = [r["oi_usd"] for r in oi_matched]
        r_oi_pnl = _pearson(oi_vals, oi_pnls)
        print(f"\n  Pearson corr (oi_usd vs pnl)       : "
              f"{r_oi_pnl:.4f}" if r_oi_pnl is not None else "  Pearson corr OI: n/a")

        quartiles = _quartile_split(oi_pnl_pairs)
        if quartiles:
            print(f"\n  OI Quartile Mean PnL:")
            for label, mean_pnl, count in quartiles:
                print(f"    {label}: mean_pnl={mean_pnl:+.4f}  n={count}")

    # --- Side breakdown ---
    for side in ("LONG", "SHORT"):
        subset = [r for r in matched_rows
                  if r["side"].upper() == side and r["funding_rate"] is not None]
        if not subset:
            continue
        mean_pnl = sum(r["pnl"] for r in subset) / len(subset)
        mean_fr = sum(r["funding_rate"] for r in subset) / len(subset)
        print(
            f"\n  {side} trades: n={len(subset)}"
            f"  mean_pnl={mean_pnl:+.4f}"
            f"  mean_funding_rate={mean_fr:+.6f}"
        )

    print(f"\n{'='*60}")
    print("  Note: shadow data accumulates from runner start date only.")
    print("  Historical joins will improve as more data is collected.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
