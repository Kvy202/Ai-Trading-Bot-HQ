# -*- coding: utf-8 -*-
import os, csv, json, sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

from flask import Flask, jsonify, send_from_directory

TOOLS_DIR = Path(__file__).resolve().parent
BASE_DIR  = TOOLS_DIR.parent
LOGS      = BASE_DIR / "logs"
STATE_JSON = LOGS / "executor_state.json"

# Ensure project root is on path for feature_store imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__)

# ---------- helpers ----------
def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def paper_path() -> Path:         return LOGS / f"trades_paper_{today_str()}.csv"
def closed_daily_path() -> Path:  return LOGS / f"trades_closed_{today_str()}.csv"
def closed_master_path() -> Path: return LOGS / "trades_closed.csv"
def closed_path() -> Path:
    p = closed_daily_path()
    return p if p.exists() else closed_master_path()
def signals_path() -> Path:       return LOGS / "live_signals.csv"

def parse_float(x: str, d: float = 0.0) -> float:
    try: return float(x)
    except Exception: return d

def tail_lines(path: Path, n: int) -> List[str]:
    """Read last n non-empty lines efficiently."""
    if not path.exists(): return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= n + 1:
            take = min(block, size)
            size -= take
            f.seek(size)
            data = f.read(take) + data
        text = data.decode("utf-8", errors="ignore")
    return [ln for ln in text.splitlines() if ln.strip()]

# ---- read executor state (adaptive thr if present) ----
def read_executor_state() -> Dict[str, Any]:
    out = {
        "exec_thr": parse_float(os.getenv("DL_P_LONG","0.45"), 0.45),
        "exec_mode": (os.getenv("DL_P_LONG_MODE","abs") or "abs").lower(),
        "adaptive": False,
    }
    try:
        if STATE_JSON.exists():
            j = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            out["exec_thr"] = float(j.get("exec_thr", out["exec_thr"]))
            out["exec_mode"] = str(j.get("exec_mode", out["exec_mode"])).lower()
            out["adaptive"] = bool(j.get("adaptive", False))
    except Exception:
        pass
    return out

# ---------- core builders ----------
def build_positions() -> Dict[str, Dict[str, Any]]:
    pos: Dict[str, Dict[str, Any]] = {}
    pp = paper_path()
    if pp.exists():
        with open(pp, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                sym   = r.get("symbol", "")
                side  = r.get("side", "")
                price = parse_float(r.get("price", "0"))
                qty   = parse_float(r.get("qty", "0"))
                if not sym: continue

                p = pos.setdefault(sym, {"side":"", "qty":0.0, "avg":0.0, "last_price":price, "unrealized":0.0})
                p["last_price"] = price

                if side in ("BUY", "SELL_SHORT"):
                    want = "long" if side == "BUY" else "short"
                    if p["qty"] <= 0 or p["side"] != want:
                        p["side"], p["qty"], p["avg"] = want, qty, price
                    else:
                        new_qty = p["qty"] + qty
                        if new_qty > 0:
                            p["avg"] = (p["avg"] * p["qty"] + price * qty) / new_qty
                        p["qty"] = new_qty
                elif side in ("SELL", "BUY_TO_COVER"):
                    if p["qty"] > 0:
                        p["qty"] = max(0.0, p["qty"] - qty)
                        if p["qty"] == 0.0:
                            p["side"] = ""
                            p["avg"]  = 0.0

    # latest mark per symbol from signals
    marks: Dict[str, float] = {}
    sp = signals_path()
    if sp.exists():
        for row in tail_lines(sp, 1000):
            try:
                cols = next(csv.reader([row]))
                if len(cols) < 9: continue
                _, sym, px, *_ = cols
                if px:
                    marks[sym] = parse_float(px)
            except Exception:
                pass

    for s, p in pos.items():
        if p["qty"] <= 0 or not p["side"]:
            p["unrealized"] = 0.0
            continue
        mark = marks.get(s, p["last_price"])
        p["last_price"] = mark
        p["unrealized"] = (mark - p["avg"]) * p["qty"] if p["side"] == "long" else (p["avg"] - mark) * p["qty"]
    return pos

def parse_closed_tail(n=20) -> List[Dict[str, Any]]:
    cp = closed_path()
    if not cp.exists(): return []
    out: List[Dict[str, Any]] = []
    try:
        with open(cp, "r", encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        for r in rd[-n:]:
            out.append({
                "ts": r.get("ts", ""),
                "symbol": r.get("symbol", ""),
                "side": r.get("closed_side", ""),
                "qty": parse_float(r.get("qty", "0")),
                "entry_avg": parse_float(r.get("entry_avg", "0")),
                "exit_price": parse_float(r.get("exit_price", "0")),
                "realized_pnl": parse_float(r.get("realized_pnl", "0")),
                "reason": r.get("reason", ""),
            })
    except Exception:
        for ln in tail_lines(cp, n + 1):
            try:
                ts, sym, side, qty, entry, exitp, pnl, reason = next(csv.reader([ln]))
                out.append({
                    "ts": ts, "symbol": sym, "side": side,
                    "qty": parse_float(qty), "entry_avg": parse_float(entry),
                    "exit_price": parse_float(exitp), "realized_pnl": parse_float(pnl),
                    "reason": reason
                })
            except Exception:
                pass
    return out

def realized_sum_today() -> float:
    cp = closed_path()
    if not cp.exists(): return 0.0
    yyyy_mm_dd = today_iso()
    s = 0.0
    try:
        with open(cp, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ts = (r.get("ts") or "")
                if cp.name.startswith("trades_closed_"):
                    s += parse_float(r.get("realized_pnl","0"))
                else:
                    if ts.startswith(yyyy_mm_dd):
                        s += parse_float(r.get("realized_pnl","0"))
    except Exception:
        for ln in tail_lines(cp, 2000):
            try:
                cols = next(csv.reader([ln]))
                if cols and cols[0].startswith(yyyy_mm_dd):
                    s += parse_float(cols[6], 0.0)
            except Exception:
                pass
    return round(s, 6)

def parse_paper_tail(n=20) -> List[Dict[str, Any]]:
    pp = paper_path()
    if not pp.exists(): return []
    out: List[Dict[str, Any]] = []
    try:
        with open(pp, "r", encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        for r in rd[-n:]:
            out.append({
                "ts": r.get("ts",""),
                "symbol": r.get("symbol",""),
                "side": r.get("side",""),
                "price": parse_float(r.get("price","0")),
                "qty": parse_float(r.get("qty","0")),
                "reason": r.get("reason",""),
            })
    except Exception:
        pass
    return out

def parse_signals_tail(n=8, state: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    sp = signals_path()
    if not sp.exists(): return []
    if state is None:
        state = read_executor_state()
    exec_thr = state["exec_thr"]; exec_mode = state["exec_mode"]

    lines = tail_lines(sp, n + 4)
    out: List[Dict[str, Any]] = []
    for ln in lines[-n:]:
        try:
            ts, symbol, px, p_meta, rv_mean, allow, thr, mode, kinds, *_ = next(csv.reader([ln]))
            p = parse_float(p_meta, 0.0)
            writer_thr = parse_float(thr, exec_thr)
            eff_thr = max(writer_thr, exec_thr)
            mode_eff = (mode or exec_mode).lower()
            val = abs(p) if mode_eff == "abs" else p
            passed = 1 if val >= eff_thr else 0
            out.append({
                "ts": ts, "symbol": symbol, "px": px,
                "p_meta": p_meta, "rv_mean": rv_mean,
                "allow": allow, "thr": thr, "mode": mode, "kinds": kinds,
                "eff_thr": eff_thr, "pass_exec": passed
            })
        except Exception:
            pass
    return out

def latest_signals_by_symbol(state: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    sp = signals_path()
    out: Dict[str, Dict[str, Any]] = {}
    if not sp.exists(): return out
    if state is None:
        state = read_executor_state()
    exec_thr = state["exec_thr"]; exec_mode = state["exec_mode"]

    for ln in reversed(tail_lines(sp, 400)):
        try:
            ts, symbol, px, p_meta, rv_mean, allow, thr, mode, kinds, *_ = next(csv.reader([ln]))
            if symbol in out: continue
            p = parse_float(p_meta, 0.0)
            writer_thr = parse_float(thr, exec_thr)
            eff_thr = max(writer_thr, exec_thr)
            mode_eff = (mode or exec_mode).lower()
            val = abs(p) if mode_eff == "abs" else p
            passed = 1 if val >= eff_thr else 0
            out[symbol] = {
                "ts": ts,
                "px": px,
                "p_meta": p,
                "rv_mean": parse_float(rv_mean, 0.0),
                "allow": int(parse_float(allow, 0)),
                "writer_thr": writer_thr,
                "eff_thr": eff_thr,
                "mode": mode_eff,
                "kinds": kinds,
                "pass_exec": passed,
            }
        except Exception:
            continue
    return out

# ---- /api/heartbeat ----
@app.get("/api/heartbeat")
def api_heartbeat():
    state = read_executor_state()
    pos = build_positions()
    open_positions = {
        s: {
            "side": p["side"],
            "qty": round(float(p["qty"]), 6),
            "avg": round(float(p["avg"]), 6),
            "last": round(float(p["last_price"]), 6),
            "unrl": round(float(p["unrealized"]), 6),
        }
        for s, p in pos.items() if p.get("qty", 0) > 0
    }
    hb = {
        "status": "ok",
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
        "totals": {
            "open_symbols": len(open_positions),
            "unrealized_sum": round(sum(p["unrl"] for p in open_positions.values()), 6),
            "realized_sum_today": realized_sum_today(),
            "exec_thr": state["exec_thr"],
            "exec_mode": state["exec_mode"],
            "adaptive": state["adaptive"],
        },
        "positions": open_positions,
        "latest_signals": latest_signals_by_symbol(state),
    }
    return jsonify(hb)

# ---------- API ----------
@app.get("/api/state")
def api_state():
    state = read_executor_state()
    pos = build_positions()
    closed = parse_closed_tail(30)
    recent = parse_paper_tail(30)
    sigs = parse_signals_tail(8, state)

    totals = {
        "open_symbols": sum(1 for p in pos.values() if p["qty"] > 0),
        "unrealized_sum": round(sum(p["unrealized"] for p in pos.values()), 6),
        "realized_sum_today": realized_sum_today(),
        "closed_count_today": len(closed),
        "exec_thr": state["exec_thr"],
        "exec_mode": state["exec_mode"],
        "adaptive": state["adaptive"],
    }
    return jsonify({
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
        "positions": pos,
        "closed_trades": closed,
        "recent_trades": recent,
        "signals": sigs,
        "totals": totals
    })

# ---------- /api/tier2 ----------
@app.get("/api/tier2")
def api_tier2():
    """Tier 2 shadow data: latest funding/OI per symbol, runner health, quality."""
    result: Dict[str, Any] = {
        "enabled": False,
        "runner_hb_age_s": None,
        "runner_ok": None,
        "runner_cycle": None,
        "quality_ok": None,
        "quality_issues": [],
        "db_row_counts": {},
        "funding": {},
        "oi": {},
    }

    # Runner heartbeat
    hb_path = LOGS / "tier2_runner_heartbeat.json"
    if hb_path.exists():
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            ts_str = hb.get("ts", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                result["runner_hb_age_s"] = int(
                    (datetime.now(timezone.utc) - ts).total_seconds()
                )
            result["runner_ok"] = hb.get("ok")
            result["runner_cycle"] = hb.get("cycle")
            result["db_row_counts"] = hb.get("db_row_counts", {})
            result["enabled"] = True
        except Exception:
            pass

    # Quality heartbeat
    qual_path = LOGS / "tier2_quality_heartbeat.json"
    if qual_path.exists():
        try:
            qual = json.loads(qual_path.read_text(encoding="utf-8"))
            result["quality_ok"] = qual.get("ok")
            result["quality_issues"] = qual.get("issues", [])
        except Exception:
            pass

    # Latest feature rows from store (best-effort; skip if store unavailable)
    try:
        from tier2.feature_store import FeatureStore
        store = FeatureStore()
        now_utc = datetime.now(timezone.utc)
        symbols = [
            s.strip()
            for s in os.getenv("DL_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
            if s.strip()
        ]

        def _age(ts_str: str) -> int | None:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                return int((now_utc - dt).total_seconds())
            except Exception:
                return None

        for sym in symbols:
            rows = store.get_latest("funding_rate", sym, n=1)
            if rows:
                r = rows[0]
                result["funding"][sym] = {
                    "rate": r.get("rate"),
                    "interval_hours": r.get("interval_hours"),
                    "mark_px": r.get("mark_px"),
                    "ts": r.get("ts"),
                    "age_s": _age(r.get("ts", "")),
                }

            rows = store.get_latest("open_interest", sym, n=1)
            if rows:
                r = rows[0]
                result["oi"][sym] = {
                    "oi_usd": r.get("oi_usd"),
                    "oi_base": r.get("oi_base"),
                    "mark_px": r.get("mark_px"),
                    "ts": r.get("ts"),
                    "age_s": _age(r.get("ts", "")),
                }
    except Exception:
        pass

    return jsonify(result)


# ---------- Single-page UI ----------
INDEX_HTML = (BASE_DIR / "tools" / "dashboard_index.html")

@app.get("/")
def index_html():
    return send_from_directory(INDEX_HTML.parent, INDEX_HTML.name)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("DASH_PORT","8787")), debug=False)
