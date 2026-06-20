"""V2 daily evidence exporter — read-only on logs/, writes reports/evidence/.

Builds one auditable bundle per UTC day from the executor's append-only
trades_closed_YYYYMMDD.csv:

  reports/evidence/YYYYMMDD/summary.json   machine-readable metrics
  reports/evidence/YYYYMMDD/summary.md     human-readable one-pager
  reports/evidence/index.json              one line of headline metrics per day

Metric definitions (kept deliberately simple and reproducible):
  trades         = data rows successfully parsed
  win_rate       = wins / trades            (win: realized_pnl > 0)
  net_pnl        = sum(realized_pnl)
  profit_factor  = gross_wins / |gross_losses|; None when no losses (noted "inf")
  max_intraday_dd= max peak-to-trough of cumulative realized PnL, rows sorted
                   by ts (string sort works: ts is "YYYY-MM-DD HH:MM:SS+0000")
  exit_reasons   = count + net pnl keyed on reason.split()[0]
                   (EXIT_TP / EXIT_SL / EXIT_SL_RESTART / FLIP_CLOSE / EXIT_TIME ...)
  per_symbol     = trades / net / win_rate per symbol

Usage:
  python tools/v2_evidence_export.py                 # all days missing from index + today
  python tools/v2_evidence_export.py --date 20260612 # one specific day (exit 2 if no CSV)
  python tools/v2_evidence_export.py --all           # re-export every day found

Exit codes: 0 ok · 1 bad args/IO · 2 --date requested but CSV absent.
Stdlib only. Never modifies anything under logs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CLOSED_RE = re.compile(r"^trades_closed_(\d{8})\.csv$")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export daily trading evidence bundles.")
    ap.add_argument("--date", default="", help="single UTC day YYYYMMDD")
    ap.add_argument("--all", action="store_true", help="re-export every day found in logs")
    ap.add_argument("--logs-dir", default=str(BASE_DIR / "logs"))
    ap.add_argument("--out-dir", default=str(BASE_DIR / "reports" / "evidence"))
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def discover_days(logs_dir: Path) -> list:
    days = []
    if logs_dir.is_dir():
        for p in logs_dir.iterdir():
            m = CLOSED_RE.match(p.name)
            if m:
                days.append(m.group(1))
    return sorted(days)


def compute_summary(csv_path: Path, day: str) -> dict:
    rows = []
    skipped = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            try:
                rows.append({
                    "ts": (raw.get("ts") or "").strip(),
                    "symbol": (raw.get("symbol") or "?").strip(),
                    "pnl": float(raw.get("realized_pnl", "")),
                    "reason": (raw.get("reason") or "").strip(),
                })
            except (TypeError, ValueError):
                skipped += 1
    rows.sort(key=lambda r: r["ts"])

    wins = [r["pnl"] for r in rows if r["pnl"] > 0]
    losses = [r["pnl"] for r in rows if r["pnl"] < 0]
    net = sum(r["pnl"] for r in rows)
    gross_w, gross_l = sum(wins), sum(losses)
    pf = (gross_w / abs(gross_l)) if losses else None

    cum = peak = max_dd = 0.0
    for r in rows:
        cum += r["pnl"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    reasons: dict = {}
    per_symbol: dict = {}
    for r in rows:
        key = r["reason"].split()[0] if r["reason"] else "UNKNOWN"
        rs = reasons.setdefault(key, {"count": 0, "net_pnl": 0.0})
        rs["count"] += 1
        rs["net_pnl"] = round(rs["net_pnl"] + r["pnl"], 8)
        ps = per_symbol.setdefault(r["symbol"], {"trades": 0, "net_pnl": 0.0, "wins": 0})
        ps["trades"] += 1
        ps["net_pnl"] = round(ps["net_pnl"] + r["pnl"], 8)
        ps["wins"] += 1 if r["pnl"] > 0 else 0
    for ps in per_symbol.values():
        ps["win_rate"] = round(ps["wins"] / ps["trades"], 4) if ps["trades"] else None
        del ps["wins"]

    return {
        "day": day,
        "source": csv_path.name,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
        "trades": len(rows),
        "skipped_rows": skipped,
        "win_rate": round(len(wins) / len(rows), 4) if rows else None,
        "net_pnl": round(net, 8),
        "gross_wins": round(gross_w, 8),
        "gross_losses": round(gross_l, 8),
        "profit_factor": round(pf, 4) if pf is not None else None,
        "profit_factor_note": "inf (no losing trades)" if (rows and not losses) else None,
        "max_intraday_dd": round(max_dd, 8),
        "exit_reasons": reasons,
        "per_symbol": per_symbol,
    }


def render_md(s: dict) -> str:
    pf = s["profit_factor"]
    pf_str = f"{pf:.3f}" if pf is not None else (s["profit_factor_note"] or "n/a")
    wr = f"{s['win_rate']:.1%}" if s["win_rate"] is not None else "n/a"
    lines = [
        f"# Evidence — {s['day']} (UTC)",
        "",
        f"Generated {s['generated_utc']} from `{s['source']}`.",
        "",
        f"| Trades | Win rate | Net PnL (USDT) | Profit factor | Max intraday DD |",
        f"|---|---|---|---|---|",
        f"| {s['trades']} | {wr} | {s['net_pnl']:.4f} | {pf_str} | {s['max_intraday_dd']:.4f} |",
        "",
        "## Exit reasons",
        "",
        "| Reason | Count | Net PnL |",
        "|---|---|---|",
    ]
    for k in sorted(s["exit_reasons"]):
        v = s["exit_reasons"][k]
        lines.append(f"| {k} | {v['count']} | {v['net_pnl']:.4f} |")
    lines += ["", "## Per symbol", "", "| Symbol | Trades | Net PnL | Win rate |", "|---|---|---|---|"]
    for sym in sorted(s["per_symbol"]):
        v = s["per_symbol"][sym]
        wr_s = f"{v['win_rate']:.1%}" if v["win_rate"] is not None else "n/a"
        lines.append(f"| {sym} | {v['trades']} | {v['net_pnl']:.4f} | {wr_s} |")
    if s["skipped_rows"]:
        lines += ["", f"_{s['skipped_rows']} malformed row(s) skipped._"]
    return "\n".join(lines) + "\n"


def export_day(day: str, logs_dir: Path, out_dir: Path, quiet: bool) -> dict:
    csv_path = logs_dir / f"trades_closed_{day}.csv"
    summary = compute_summary(csv_path, day) if csv_path.exists() else {
        "day": day, "source": csv_path.name, "trades": 0, "skipped_rows": 0,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
        "win_rate": None, "net_pnl": 0.0, "gross_wins": 0.0, "gross_losses": 0.0,
        "profit_factor": None, "profit_factor_note": None, "max_intraday_dd": 0.0,
        "exit_reasons": {}, "per_symbol": {},
    }
    day_dir = out_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (day_dir / "summary.md").write_text(render_md(summary), encoding="utf-8")
    if not quiet:
        pf = summary["profit_factor"]
        print(f"[evidence] {day}: trades={summary['trades']} net={summary['net_pnl']:.4f} "
              f"pf={pf if pf is not None else 'n/a'} -> {day_dir}")
    return summary


def update_index(out_dir: Path, summaries: list) -> None:
    index_path = out_dir / "index.json"
    index = {}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        index = {}
    for s in summaries:
        index[s["day"]] = {
            "trades": s["trades"], "win_rate": s["win_rate"], "net_pnl": s["net_pnl"],
            "profit_factor": s["profit_factor"], "max_intraday_dd": s["max_intraday_dd"],
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=2), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    logs_dir, out_dir = Path(args.logs_dir), Path(args.out_dir)

    if args.date:
        if not re.fullmatch(r"\d{8}", args.date):
            print(f"ERROR: --date must be YYYYMMDD, got {args.date!r}", file=sys.stderr)
            return 1
        if not (logs_dir / f"trades_closed_{args.date}.csv").exists():
            print(f"ERROR: no trades_closed_{args.date}.csv in {logs_dir}", file=sys.stderr)
            return 2
        days = [args.date]
    else:
        days = discover_days(logs_dir)
        if not args.all:
            # Default: only days not yet in the index, plus today (refreshed).
            try:
                done = set(json.loads((out_dir / "index.json").read_text(encoding="utf-8")))
            except Exception:
                done = set()
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            days = [d for d in days if d not in done or d == today]
        if not days:
            if not args.quiet:
                print("[evidence] nothing to export")
            return 0

    try:
        summaries = [export_day(d, logs_dir, out_dir, args.quiet) for d in days]
        update_index(out_dir, summaries)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
