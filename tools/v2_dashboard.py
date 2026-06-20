"""V2 static dashboard generator — read-only on logs/, writes one HTML file.

Produces a single self-contained page (inline CSS, one inline SVG, zero
JavaScript, zero external requests) so it can be scp'd anywhere or attached
to the submission report:

  - status cards: executor heartbeat age (stale > 120 s highlighted), mode,
    open positions, supervisor pause state, V2 risk snapshot incl. block reason
  - daily PnL table for the last N days (from trades_closed_*.csv)
  - cumulative realized PnL curve (inline SVG polyline)
  - exit-reason breakdown over the window
  - last N closed trades
  - recent deploy markers

Usage:  python tools/v2_dashboard.py [--days 14] [--last-trades 30]
                                     [--logs-dir ...] [--out reports/dashboard.html]
Exit codes: 0 ok (missing inputs render as "n/a") · 1 cannot write output.
Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CLOSED_RE = re.compile(r"^trades_closed_(\d{8})\.csv$")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate static HTML dashboard from logs.")
    ap.add_argument("--logs-dir", default=str(BASE_DIR / "logs"))
    ap.add_argument("--out", default=str(BASE_DIR / "reports" / "dashboard.html"))
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--last-trades", type=int, default=30)
    return ap.parse_args(argv)


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_ts_epoch(ts: str):
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts.strip(), fmt).timestamp()
        except Exception:
            continue
    return None


def load_closed_rows(logs_dir: Path, days: int) -> dict:
    """{day: [row, ...]} for the most recent `days` daily files."""
    found = sorted(p.name for p in logs_dir.iterdir() if CLOSED_RE.match(p.name)) \
        if logs_dir.is_dir() else []
    out: dict = {}
    for name in found[-days:]:
        day = CLOSED_RE.match(name).group(1)
        rows = []
        try:
            with (logs_dir / name).open("r", encoding="utf-8", newline="") as f:
                for raw in csv.DictReader(f):
                    try:
                        rows.append({
                            "ts": (raw.get("ts") or "").strip(),
                            "symbol": (raw.get("symbol") or "?").strip(),
                            "side": (raw.get("closed_side") or "").strip(),
                            "pnl": float(raw.get("realized_pnl", "")),
                            "reason": (raw.get("reason") or "").strip(),
                        })
                    except (TypeError, ValueError):
                        continue
        except OSError:
            continue
        out[day] = rows
    return out


def svg_cumulative(daily: dict, width=640, height=180) -> str:
    """Inline SVG polyline of cumulative net PnL across the day window."""
    days = sorted(daily)
    points = []
    cum = 0.0
    for d in days:
        cum += sum(r["pnl"] for r in daily[d])
        points.append((d, cum))
    if len(points) < 2:
        return "<p>n/a (need ≥2 days of closed trades)</p>"
    vals = [v for _, v in points]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    span = (hi - lo) or 1.0
    pad = 8
    n = len(points)
    xs = lambda i: pad + i * (width - 2 * pad) / (n - 1)
    ys = lambda v: pad + (hi - v) * (height - 2 * pad) / span
    pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, (_, v) in enumerate(points))
    zero_y = ys(0.0)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'style="width:100%;max-width:{width}px;background:#0d1117;border:1px solid #30363d;border-radius:6px">'
        f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width-pad}" y2="{zero_y:.1f}" '
        f'stroke="#484f58" stroke-dasharray="4 4"/>'
        f'<polyline points="{pts}" fill="none" stroke="#58a6ff" stroke-width="2"/>'
        f'<text x="{pad}" y="{pad+10}" fill="#8b949e" font-size="11">'
        f'cum PnL {points[-1][1]:+.4f} USDT ({esc(days[0])} → {esc(days[-1])})</text>'
        f"</svg>"
    )


def card(label: str, value: str, sub: str = "", warn: bool = False) -> str:
    border = "#f85149" if warn else "#30363d"
    return (f'<div style="border:1px solid {border};border-radius:6px;padding:10px 14px;'
            f'min-width:150px;background:#161b22">'
            f'<div style="color:#8b949e;font-size:12px">{esc(label)}</div>'
            f'<div style="font-size:18px;margin-top:2px">{value}</div>'
            f'<div style="color:#8b949e;font-size:11px;margin-top:2px">{esc(sub)}</div></div>')


def table(headers, rows) -> str:
    th = "".join(f'<th style="text-align:left;padding:4px 10px;border-bottom:1px solid #30363d;'
                 f'color:#8b949e;font-size:12px">{esc(h)}</th>' for h in headers)
    body = ""
    for r in rows:
        tds = "".join(f'<td style="padding:4px 10px;border-bottom:1px solid #21262d;'
                      f'font-size:13px">{c}</td>' for c in r)
        body += f"<tr>{tds}</tr>"
    if not rows:
        body = (f'<tr><td colspan="{len(headers)}" style="padding:8px 10px;'
                f'color:#8b949e">n/a</td></tr>')
    return f'<table style="border-collapse:collapse;width:100%">{th and f"<tr>{th}</tr>"}{body}</table>'


def pnl_cell(v: float) -> str:
    color = "#3fb950" if v > 0 else ("#f85149" if v < 0 else "#8b949e")
    return f'<span style="color:{color}">{v:+.4f}</span>'


def build_html(args) -> str:
    logs_dir = Path(args.logs_dir)
    now = datetime.now(timezone.utc)

    # --- status cards ---
    hb = read_json(logs_dir / "heartbeat.json") or {}
    hb_age = None
    if hb.get("ts"):
        ep = parse_ts_epoch(hb["ts"])
        hb_age = (now.timestamp() - ep) if ep else None
    hb_warn = hb_age is None or hb_age > 120
    hb_val = f"{hb_age:.0f}s ago" if hb_age is not None else "n/a"

    state = read_json(logs_dir / "executor_state.json") or {}
    mode = state.get("mode", hb.get("mode", "n/a"))
    open_pos = state.get("positions", {})
    paused = state.get("paused", "n/a")

    v2 = read_json(logs_dir / "v2_risk_state.json")
    if v2:
        v2_block = ""
        v2_sub = (f"day {v2.get('utc_day','?')} · SL {v2.get('sl_count_today','?')} · "
                  f"pnl {v2.get('realized_pnl_today','?')}")
        v2_val = "active"
    else:
        v2_sub, v2_val = "state file absent (flags off or V1 code)", "off / n/a"
    pause_file_present = (BASE_DIR / "run" / "V2_PAUSE").exists()

    cards = "".join([
        card("Executor heartbeat", esc(hb_val), esc(hb.get("event", "")), warn=hb_warn),
        card("Mode", esc(mode), "live_executor"),
        card("Open positions", esc(len(open_pos)),
             ", ".join(sorted(open_pos)) if open_pos else "flat"),
        card("Supervisor paused", esc(paused)),
        card("V2 risk", esc(v2_val), v2_sub, warn=pause_file_present),
        card("V2 pause file", "PRESENT" if pause_file_present else "absent",
             "run/V2_PAUSE", warn=pause_file_present),
    ])

    # --- trade data ---
    daily = load_closed_rows(logs_dir, args.days)
    day_rows = []
    reason_agg: dict = {}
    all_rows = []
    for day in sorted(daily, reverse=True):
        rows = daily[day]
        net = sum(r["pnl"] for r in rows)
        wins = sum(1 for r in rows if r["pnl"] > 0)
        losses = [r["pnl"] for r in rows if r["pnl"] < 0]
        gw = sum(r["pnl"] for r in rows if r["pnl"] > 0)
        pf = f"{gw/abs(sum(losses)):.3f}" if losses else ("inf" if rows else "n/a")
        wr = f"{wins/len(rows):.0%}" if rows else "n/a"
        day_rows.append([esc(day), esc(len(rows)), wr, pnl_cell(net), esc(pf)])
        for r in rows:
            key = r["reason"].split()[0] if r["reason"] else "UNKNOWN"
            agg = reason_agg.setdefault(key, [0, 0.0])
            agg[0] += 1
            agg[1] += r["pnl"]
            all_rows.append((day, r))

    reason_rows = [[esc(k), esc(v[0]), pnl_cell(v[1])]
                   for k, v in sorted(reason_agg.items(), key=lambda kv: -kv[1][0])]

    last_trades = [[esc(r["ts"]), esc(r["symbol"]), esc(r["side"]), pnl_cell(r["pnl"]),
                    esc(r["reason"][:80])]
                   for _, r in sorted(all_rows, key=lambda t: t[1]["ts"])[-args.last_trades:]][::-1]

    # --- deploy markers ---
    markers = []
    mpath = logs_dir / "deploy_markers.csv"
    if mpath.exists():
        try:
            with mpath.open("r", encoding="utf-8", newline="") as f:
                markers = [[esc(m.get("ts_utc", "")), esc(m.get("git_sha", "")),
                            esc(m.get("branch", "")), esc(m.get("note", ""))]
                           for m in csv.DictReader(f)][-10:][::-1]
        except OSError:
            pass

    def section(title: str, body: str) -> str:
        return (f'<h2 style="font-size:16px;border-bottom:1px solid #30363d;'
                f'padding-bottom:4px;margin:24px 0 10px">{esc(title)}</h2>{body}')

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>AI Trading Bot — dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="background:#0d1117;color:#c9d1d9;font-family:Segoe UI,system-ui,sans-serif;
             max-width:900px;margin:0 auto;padding:20px">
<h1 style="font-size:20px">AI Trading Bot — paper dashboard</h1>
<p style="color:#8b949e;font-size:12px">generated {esc(now.strftime('%Y-%m-%d %H:%M:%S'))} UTC ·
window {esc(args.days)} days · read-only snapshot</p>
<div style="display:flex;flex-wrap:wrap;gap:10px">{cards}</div>
{section("Cumulative realized PnL", svg_cumulative(daily))}
{section("Daily PnL", table(["Day", "Trades", "Win rate", "Net PnL", "PF"], day_rows))}
{section("Exit reasons", table(["Reason", "Count", "Net PnL"], reason_rows))}
{section(f"Last trades", table(["Closed (ts)", "Symbol", "Side", "PnL", "Reason"], last_trades))}
{section("Deploy markers", table(["UTC", "SHA", "Branch", "Note"], markers))}
</body></html>
"""


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_html(args), encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot write {out}: {exc}")
        return 1
    print(f"[dashboard] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
