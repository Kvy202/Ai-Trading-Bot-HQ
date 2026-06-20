import os, sys, atexit, json, subprocess, glob
import time, joblib, logging, warnings, itertools, threading, statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== Day 3: Risk Engine hooks ======
from risk_engine import (
    stop_distance_abs,
    size_from_risk,
    cap_notional,
    should_pause_spread,
    portfolio_exposure_ok,
    TIME_STOP_BARS,
)

# ---- optional pretty logging (safe fallback if missing) ----
try:
    from fg_print import fg_emit, fg_setup  # type: ignore
except Exception:  # pragma: no cover
    def fg_emit(*a, **k): ...
    def fg_setup(*a, **k): ...

# ---- Day-1: leakage-safe, microstructure-aware features ----
from feature_pipe_adapter import get_latest_features

import pandas as pd
import numpy as np
import ccxt

from data import fetch_ohlcv, get_exchange  # existing helpers
from features import build_features          # your 5m feature builder
from utils import bps_to_frac
from router import ROUTER_ENABLED, best_cex_quote
from telegram_bot import TelegramBot, TraderState

# Day-4 (DL) inference
try:
    from ml_dl.dl_infer import infer_scores
except Exception:
    infer_scores = None  # keeps Day-3 running even if DL pocket not present

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===================== single-instance guard (Windows) =====================
AUTO_KILL_OLD = bool(int(os.getenv("AUTO_KILL_OLD", "1")))
LOCK_PATH = os.path.join("logs", "trader.lock")
os.makedirs("logs", exist_ok=True)

def _taskkill(pid: int):
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def _owner_pid() -> Optional[int]:
    try:
        if not os.path.exists(LOCK_PATH):
            return None
        with open(LOCK_PATH, "r", encoding="utf-8") as f:
            s = (f.read() or "").strip()
        return int(s) if s.isdigit() else None
    except Exception:
        return None

def _owner_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        out = subprocess.check_output(
            ["powershell","-NoProfile","-Command",f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").ProcessId"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        return str(pid) in (out or "")
    except Exception:
        return False

def _try_create_lock(my_pid: int) -> bool:
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(my_pid).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

def ensure_single_instance() -> bool:
    me = os.getpid()
    print("[boot] entering __main__"); print(f"[lock] AUTO_KILL_OLD={AUTO_KILL_OLD}")
    if _try_create_lock(me):
        print("[lock] acquired=True")
        def _cleanup_lock(me_pid: int):
            try:
                cur = _owner_pid()
                if cur == me_pid and os.path.exists(LOCK_PATH):
                    os.remove(LOCK_PATH)
            except Exception:
                pass
        atexit.register(_cleanup_lock, me)
        print("[boot] ensure_single_instance -> True"); return True

    owner = _owner_pid()
    if not _owner_alive(owner):
        try: os.remove(LOCK_PATH)
        except Exception: pass
        if _try_create_lock(me):
            print("[lock] acquired=True (stale lock cleared)")
            atexit.register(lambda pid=me: (os.path.exists(LOCK_PATH) and _owner_pid() == pid and os.remove(LOCK_PATH)))
            print("[boot] ensure_single_instance -> True"); return True

    if not AUTO_KILL_OLD:
        print("Another trader appears to be running. If stale, delete logs\\trader.lock."); return False

    if owner and _owner_alive(owner):
        print(f"[lock] takeover: killing owner pid={owner}")
        _taskkill(owner)
        deadline = time.time() + 20.0
        while time.time() < deadline and _owner_alive(owner):
            time.sleep(0.25)

    try: os.remove(LOCK_PATH)
    except Exception: pass

    for _ in range(40):
        if _try_create_lock(me):
            print("[lock] acquired=True (takeover)")
            atexit.register(lambda pid=me: (os.path.exists(LOCK_PATH) and _owner_pid() == pid and os.remove(LOCK_PATH)))
            print("[boot] ensure_single_instance -> True"); return True
        time.sleep(0.25)

    print("[lock] could not acquire after takeover; exiting."); return False

# ===================== Env / config =====================
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")  # Day-3 bundle
LOG_PATH   = os.getenv("LOG_PATH",   "logs/trades.csv")

TIMEFRAME = os.getenv("TIMEFRAME", "5m")
LOOKBACK  = int(os.getenv("LOOKBACK_CANDLES", "3000"))

# Strategy (Day-3)
PRED_THRESHOLD  = float(os.getenv("PRED_THRESHOLD", "0.70"))
STOP_ATR_MULT   = float(os.getenv("STOP_ATR_MULT", "2.5"))
TP_R_MULT       = float(os.getenv("TP_R_MULT", "3.0"))
MAX_HOLD_BARS   = int(os.getenv("MAX_HOLD_BARS", "36"))  # used only if TIME_STOP_BARS<=0
RISK_PER_TRADE  = float(os.getenv("RISK_PER_TRADE", "0.003"))
FEES_BPS        = float(os.getenv("FEES_BPS", "6"))

# Day-3 extra risk knobs
RISK_MIN_STOP_FRAC     = float(os.getenv("RISK_MIN_STOP_FRAC", "0.002"))
MAX_PORTFOLIO_EXPOSURE = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_USDT", "0"))
PER_SYMBOL_NOTIONAL    = float(os.getenv("PER_SYMBOL_NOTIONAL_USDT", os.getenv("MAX_NOTIONAL_USDT", "0")))
SPREAD_PAUSE_BPS       = float(os.getenv("SPREAD_PAUSE_BPS", "15"))
SPREAD_WIDEN_MULT      = float(os.getenv("SPREAD_WIDEN_MULT", "2.5"))
SPREAD_LOOKBACK        = int(os.getenv("SPREAD_LOOKBACK", "20"))
SLIPPAGE_MAX_BPS       = float(os.getenv("SLIPPAGE_MAX_BPS", "40"))

START_EQUITY = float(os.getenv("START_EQUITY", "10000"))

# Universe
SCAN_TOPN          = int(os.getenv("SCAN_TOPN", "4"))
SCAN_MIN_NOTIONAL  = float(os.getenv("SCAN_MIN_NOTIONAL", "100000"))
SYMBOL_WHITELIST   = [s.strip() for s in os.getenv("SYMBOL_WHITELIST", "").split(",") if s.strip()]

# Live / venue
LIVE_MODE     = bool(int(os.getenv("LIVE_MODE", "0")))  # default PAPER; set LIVE_MODE=1 in .env to go live
DERIVS        = bool(int(os.getenv("DERIVS", "1")))
EXCHANGE_ID   = os.getenv("EXCHANGE_ID", "bitget").lower()
LEVERAGE      = int(os.getenv("LEVERAGE", "2"))
MARGIN_MODE   = os.getenv("MARGIN_MODE", "cross")
POSITION_MODE = os.getenv("POSITION_MODE", "oneway").lower()

# Global risk controls
MAX_DD          = float(os.getenv("MAX_DD", "0.05"))
COOLDOWN_BARS   = int(os.getenv("COOLDOWN_BARS", "3"))
MAX_CONCURRENT  = int(os.getenv("MAX_CONCURRENT", "2"))

# Direction gates
LONG_ONLY  = bool(int(os.getenv("LONG_ONLY", "0")))
SHORT_ONLY = bool(int(os.getenv("SHORT_ONLY", "0")))

# Size caps (tiny accounts)
MAX_NOTIONAL_USDT   = float(os.getenv("MAX_NOTIONAL_USDT", "0"))
MAX_MARGIN_FRACTION = float(os.getenv("MAX_MARGIN_FRACTION", "0.0"))

# Output / UX
RUNTIME_LOG         = os.getenv("RUNTIME_LOG", "logs/runtime.log")
HEARTBEAT_PATH      = os.getenv("HEARTBEAT_PATH", "logs/heartbeat.json")
QUIET_MODE          = bool(int(os.getenv("QUIET_MODE", "1")))
RUNTIME_THROTTLE_MS = int(os.getenv("RUNTIME_THROTTLE_MS", "1000"))

SPINNER = bool(int(os.getenv("SPINNER", "1")))
COLOR   = bool(int(os.getenv("COLOR", "1")))

# Pre-sanitize
SANITIZE_UNIVERSE = bool(int(os.getenv("SANITIZE_UNIVERSE", "1")))
SANITIZE_WORKERS  = int(os.getenv("SANITIZE_WORKERS", "4"))

# Live verification / duplicate protection
POS_SYNC_SECS = int(os.getenv("POS_SYNC_SECS", "60"))
CHECK_OPEN_ORDERS = bool(int(os.getenv("CHECK_OPEN_ORDERS", "1")))
RECENT_ORDER_COOLDOWN_S = int(os.getenv("RECENT_ORDER_COOLDOWN_S", "12"))

# Telegram
TELEGRAM_ENABLED = bool(int(os.getenv("TELEGRAM_ENABLED", "1")))
TELEGRAM_POLL    = bool(int(os.getenv("TELEGRAM_POLL", "1")))
TELEGRAM_DAILY   = os.getenv("TELEGRAM_DAILY_HHMM", "00:05")

# ---- Day-1 feature toggles ----
FEATURE_ENABLE       = bool(int(os.getenv("FEATURE_ENABLE", "1")))
FEATURE_TIMEFRAME    = os.getenv("FEATURE_TIMEFRAME", "1m")
FEATURE_KLINE_LIMIT  = int(os.getenv("FEATURE_KLINE_LIMIT", "1000"))
SPREAD_GUARD_BPS     = float(os.getenv("SPREAD_GUARD_BPS", "10"))
IMB_ABS_MIN          = float(os.getenv("IMB_ABS_MIN", "0.00"))
RV_STOP_MULT         = float(os.getenv("RV_STOP_MULT", "2.0"))

# CCXT hardening (optional)
CCXT_TIMEOUT_MS = int(os.getenv("CCXT_TIMEOUT_MS", "20000"))
CCXT_RETRIES    = int(os.getenv("CCXT_RETRIES", "3"))

# ==== Day-4 (DL) toggles & knobs ====
USE_DL_SIGNALS = os.getenv("USE_DL_SIGNALS", "0") in ("1","true","True")
DL_MODEL_KIND  = os.getenv("DL_MODEL_KIND", "tcn")      # tcn | tx
DL_SEQ_LEN     = int(os.getenv("DL_SEQ_LEN", "64"))
DL_P_LONG      = float(os.getenv("DL_P_LONG", "0.55"))  # prob threshold
DL_MAX_RV      = float(os.getenv("DL_MAX_RV", "0.02"))  # vol gate
DL_SCALER_PATH = os.getenv("DL_SCALER_PATH", "")        # optional override
DL_MODEL_PATH  = os.getenv("DL_MODEL_PATH", "")         # optional override

# Runtime status for Telegram /model_status
DL_STATUS = {
    "ready": False,
    "kind": os.getenv("DL_MODEL_KIND", "tcn"),
    "seq_len": int(os.getenv("DL_SEQ_LEN", "64")),
    "p_long": float(os.getenv("DL_P_LONG", "0.55")),
    "max_rv": float(os.getenv("DL_MAX_RV", "0.02")),
    "model_path": "",
    "scaler_path": "",
    "last": {}  # symbol -> {"ts": iso, "p_long": float, "ret_hat": float, "rv_hat": float}
}

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ===== Shared state =====
STATE = TraderState(
    paused=False,
    running=True,
    pred_threshold=PRED_THRESHOLD,
    max_concurrent=MAX_CONCURRENT,
    scan_topn=SCAN_TOPN,
    timeframe=TIMEFRAME,
)

STOP_EVENT = threading.Event()
FORCE_EXIT_ALL = threading.Event()
FORCE_EXIT_SYMBOLS: Set[str] = set()

# ===================== helpers =====================
def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)

def _ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

_last_runtime_write = 0.0
def write_current(symbol: str, idx: int, total: int, phase: str = "scan"):
    global _last_runtime_write
    now_ms = time.time() * 1000.0
    if now_ms - _last_runtime_write < RUNTIME_THROTTLE_MS: return
    _last_runtime_write = now_ms
    _ensure_dir(RUNTIME_LOG)
    line = f"{_ts_utc()} | {phase} {idx}/{total} | {symbol}\n"
    with open(RUNTIME_LOG, "a", encoding="utf-8") as f: f.write(line)

def write_event(event: Dict[str, Any]):
    _ensure_dir(HEARTBEAT_PATH)
    payload = {"ts": _ts_utc(), **event}
    with open(HEARTBEAT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    try: fg_emit(payload)
    except Exception: pass

def _c(txt: str, code: str) -> str:
    if not COLOR: return txt
    return f"\u001b[{code}m{txt}\u001b[0m"

_SPIN = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
def spinner_line(symbol: str, idx: int, total: int, phase: str = "scan"):
    if not SPINNER: return
    spin = next(_SPIN)
    print(f"\r\u001b[2K{_c(spin, '90')} {_c(phase, '36')} {idx}/{total} {_c(symbol, '33')}", end="", flush=True)

def spinner_flash(msg: str, ok: bool = True):
    if not SPINNER: return
    icon = "✔" if ok else "⚠"; color = "32" if ok else "33"
    print(f"\r\u001b[2K{_c(icon, color)} {msg}")

def ensure_log():
    _ensure_dir(LOG_PATH)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["ts", "symbol", "side", "price", "qty", "reason"]).to_csv(LOG_PATH, index=False)

def log_trade(**k):
    df = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame(
        columns=["ts", "symbol", "side", "price", "qty", "reason"]
    )
    df.loc[len(df)] = [k.get("ts"), k.get("symbol"), k.get("side"), k.get("price"), k.get("qty"), k.get("reason")]
    df.to_csv(LOG_PATH, index=False)

# ===================== Bitget client =====================
def live_client():
    api_key   = os.getenv("API_KEY", "")
    api_secret= os.getenv("API_SECRET", "")
    api_pass  = os.getenv("API_PASSWORD", "")
    opts = {
        "enableRateLimit": True,
        "timeout": CCXT_TIMEOUT_MS,
        "options": {"defaultType": "swap", "defaultSubType": "linear", "defaultSettle": "USDT"} if DERIVS else {"defaultType": "spot"},
        "apiKey": api_key, "secret": api_secret, "password": api_pass,
    }
    ex = getattr(ccxt, EXCHANGE_ID)(opts)
    try: ex.set_sandbox_mode(bool(int(os.getenv("BITGET_SANDBOX", "0"))))
    except Exception: pass

    for _ in range(max(1, CCXT_RETRIES)):
        try: ex.load_markets(); break
        except Exception: time.sleep(1)

    if DERIVS:
        try:
            hedge = POSITION_MODE == "hedge"
            ex.set_position_mode(hedge)
        except Exception: pass
    return ex

def normalize_amount_for_derivs(ex, symbol, base_qty):
    m = ex.market(symbol)
    contract_size = float(m.get("contractSize") or 1.0)
    amt = base_qty / contract_size if m.get("contract", False) else base_qty
    return ex.amount_to_precision(symbol, amt)

def fetch_wallet_usdt(client) -> Optional[float]:
    try:
        bal = client.fetch_balance(params={"type": "swap"})
        usdt = bal.get("USDT") or {}
        if isinstance(usdt, dict):
            return float(usdt.get("total") or usdt.get("free") or 0.0)
        total = bal.get("total") or {}
        if "USDT" in total: return float(total["USDT"])
    except Exception: pass
    try:
        bal = client.fetch_balance(params={"productType": "USDT-FUTURES"})
        usdt = bal.get("USDT") or {}
        if isinstance(usdt, dict):
            return float(usdt.get("total") or usdt.get("free") or 0.0)
        total = bal.get("total") or {}
        if "USDT" in total: return float(total["USDT"])
    except Exception: pass
    return None

# ===================== universe =====================
def scan_bitget_topn(topn=SCAN_TOPN, min_notional=SCAN_MIN_NOTIONAL) -> List[str]:
    ex = getattr(ccxt, "bitget")({
        "enableRateLimit": True,
        "timeout": CCXT_TIMEOUT_MS,
        "options": {"defaultType": "swap", "defaultSubType": "linear", "defaultSettle": "USDT"},
    })
    try:
        ex.load_markets(); tickers = ex.fetch_tickers()
        cands: List[Tuple[str, float]] = []
        for sym, m in ex.markets.items():
            if not m.get("contract", False): continue
            if m.get("settle", "").upper() != "USDT": continue
            if m.get("linear") is False: continue
            t = tickers.get(sym, {})
            last = t.get("last"); qvol = t.get("quoteVolume")
            if qvol is None:
                base_vol = t.get("baseVolume")
                if base_vol is not None and last:
                    qvol = base_vol * last
                else:
                    info = t.get("info", {}) or {}
                    qvol = float(info.get("usdtVol") or 0.0)
            if last and qvol and float(qvol) >= min_notional:
                cands.append((sym, float(qvol)))
        cands.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in cands[:topn]]
    finally:
        try: ex.close()
        except Exception: pass

def _probe_symbol(sym: str, tf: str) -> bool:
    try:
        fetch_ohlcv(sym, tf, 1); return True
    except Exception:
        return False

def sanitize_universe(symbols: List[str], tf: str = "5m", workers: int = SANITIZE_WORKERS) -> List[str]:
    if not symbols: return []
    if workers <= 1: return [s for s in symbols if _probe_symbol(s, tf)]
    ok: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_probe_symbol, s, tf): s for s in symbols}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                if fut.result(): ok.append(s)
            except Exception: pass
    kept = [s for s in symbols if s in set(ok)]
    return kept

def build_universe() -> List[str]:
    st = STATE.snapshot()
    topn = int(st.get("scan_topn", SCAN_TOPN))
    scanned = scan_bitget_topn(topn=topn, min_notional=SCAN_MIN_NOTIONAL)
    merged = scanned[:]
    for s in SYMBOL_WHITELIST:
        s = s.strip()
        if s and s not in merged: merged.append(s)
    uni = sanitize_universe(merged, TIMEFRAME) if SANITIZE_UNIVERSE else merged
    try:
        write_event({
            "event":"universe_built","requested_topN":topn,"scan_count":len(scanned),
            "whitelist_count":len(SYMBOL_WHITELIST),"final_count":len(uni),
            "whitelist_added":[s for s in SYMBOL_WHITELIST if s and s not in scanned],
        })
        print(f"Universe source=scan+whitelist | requested_topN={topn} | scanned={len(scanned)} | whitelist_added={len([s for s in SYMBOL_WHITELIST if s and s not in scanned])} | final={len(uni)}")
    except Exception:
        pass
    return uni

# ===================== misc helpers =====================
def timeframe_to_seconds(tf: str) -> int:
    tf = str(tf).strip().lower()
    if tf.endswith("m"): return int(tf[:-1]) * 60
    if tf.endswith("h"): return int(tf[:-1]) * 3600
    if tf.endswith("d"): return int(tf[:-1]) * 86400
    return 300

# ===================== live position/order checks =====================
def _parse_ccxt_position(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sym = p.get("symbol"); side = p.get("side")
    cont = p.get("contracts"); size = None
    if isinstance(cont, (int, float)): size = float(cont)
    if (size is None or size == 0) and isinstance(p.get("info"), dict):
        info = p["info"]
        for k in ("total","holdVol","positions","size"):
            v = info.get(k)
            try:
                if v is not None and float(v) != 0:
                    size = abs(float(v)); break
            except Exception: pass
    if not size or size <= 0: return None
    entry = p.get("entryPrice")
    try: entry = float(entry) if entry is not None else None
    except Exception: entry = None
    if not side: side = "long"
    return {"symbol": sym, "side": side, "contracts": size, "entryPrice": entry}

def load_live_positions_map(client, symbols: Optional[List[str]] = None) -> Optional[Dict[str, Dict[str, Any]]]:
    try:
        pos = client.fetch_positions(symbols or [])
    except Exception:
        try: pos = client.fetch_positions()
        except Exception: return None
    res: Dict[str, Dict[str, Any]] = {}
    longs: Dict[str, Dict[str, Any]] = {}; shorts: Dict[str, Dict[str, Any]] = {}
    for p in pos or []:
        norm = _parse_ccxt_position(p)
        if not norm: continue
        sym = norm["symbol"]; sd = norm["side"]
        if symbols and sym not in symbols: continue
        if sd == "long":
            longs[sym] = {"qty": norm["contracts"], "entry": norm["entryPrice"]}
        elif sd == "short":
            shorts[sym] = {"qty": norm["contracts"], "entry": norm["entryPrice"]}
    for sym in set(list(longs.keys()) + list(shorts.keys())):
        if sym in longs and sym in shorts and longs[sym]["qty"] > 0 and shorts[sym]["qty"] > 0:
            res[sym] = {"dir": "hedged", "qty": max(longs[sym]["qty"], shorts[sym]["qty"]), "entry": None}
        elif sym in longs and longs[sym]["qty"] > 0:
            res[sym] = {"dir": "long", "qty": longs[sym]["qty"], "entry": longs[sym]["entry"]}
        elif sym in shorts and shorts[sym]["qty"] > 0:
            res[sym] = {"dir": "short", "qty": shorts[sym]["qty"], "entry": shorts[sym]["entry"]}
    return res

def has_open_orders(client, symbol: str) -> bool:
    try:
        oo = client.fetch_open_orders(symbol); return bool(oo)
    except Exception:
        return False

# ===================== PnL report =====================
def make_yesterday_report() -> str:
    if not os.path.exists(LOG_PATH): return "No trades yet."
    df = pd.read_csv(LOG_PATH)
    if df.empty: return "No trades yet."
    try:
        dts = pd.to_datetime(df["ts"], errors="coerce", utc=True).tz_convert(None)
    except Exception:
        dts = pd.to_datetime(df["ts"], errors="coerce")
    df["_dt"] = dts; df = df.dropna(subset=["_dt"]).sort_values("_dt")
    if df.empty: return "No trades yet."
    today = datetime.now().date(); yday = today - timedelta(days=1)
    df_day = df[(df["_dt"].dt.date == yday)]
    if df_day.empty: return f"No trades on {yday}."
    pnl = 0.0; lines = []; by_sym: Dict[str, Dict[str, Any]] = {}
    for _, r in df_day.iterrows():
        sym, side = r["symbol"], str(r["side"])
        px = float(r.get("price", 0.0) or 0.0); qty = float(r.get("qty", 0.0) or 0.0)
        if side in ("BUY","PAPER_BUY"):
            by_sym[sym] = {"dir":"long","px":px,"qty":qty}
        elif side in ("SELL","PAPER_SELL"):
            st = by_sym.pop(sym, None)
            if st and st["dir"] == "long":
                pnl += (px - st["px"]) / max(st["px"], 1e-12)
                lines.append(f"{sym} LONG: {st['px']} -> {px}")
        elif side in ("SELL_SHORT","PAPER_SELL_SHORT"):
            by_sym[sym] = {"dir":"short","px":px,"qty":qty}
        elif side in ("BUY_CLOSE","PAPER_BUY_CLOSE"):
            st = by_sym.pop(sym, None)
            if st and st["dir"] == "short":
                pnl += (st["px"] - px) / max(st["px"], 1e-12)
                lines.append(f"{sym} SHORT: {st['px']} -> {px}")
    txt = f"Report for {yday}:\n" + ("\n".join(lines) if lines else "No round trips") + f"\n\nApprox PnL (fractional sum): {pnl:.4f}"
    return txt

# ===================== Day-3 helpers (exposure, spreads, gaps) =====================
def total_open_notional(positions: Dict[str, Dict[str, Any]], price_map: Dict[str, float]) -> float:
    total = 0.0
    for sym, st in positions.items():
        if st.get("in_pos") and st.get("qty") and sym in price_map:
            total += float(st["qty"]) * float(price_map[sym])
    return total

spread_hist: Dict[str, deque] = {}  # symbol -> recent spread_bps deque
def update_spread_history(sym: str, spread_bps: Optional[float]):
    if spread_bps is None or spread_bps != spread_bps: return
    dq = spread_hist.setdefault(sym, deque(maxlen=max(5, SPREAD_LOOKBACK)))
    dq.append(float(spread_bps))

def rolling_median_spread(sym: str) -> Optional[float]:
    dq = spread_hist.get(sym)
    if not dq: return None
    try: return float(statistics.median(dq))
    except Exception: return None

# ===================== core loop =====================
def _latest(path_glob: str) -> Optional[str]:
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None

def run_loop(poll_s=15):
    # ----- Day-3 model (classical) -----
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    fcols = bundle["features"]           # feature order for Day-3 *and* reused for DL buffer
    ensure_log()

    # ----- DL artifacts (optional) -----
    dl_ready = False
    dl_model_path = DL_MODEL_PATH or _latest(f"models/dl_{DL_MODEL_KIND}_*.pt")
    dl_scaler_path = DL_SCALER_PATH or _latest("models/scaler_*.joblib")
    if USE_DL_SIGNALS and infer_scores is None:
        write_event({"event":"dl_unavailable","reason":"ml_dl.dl_infer not importable"})
    if USE_DL_SIGNALS and infer_scores and dl_model_path and dl_scaler_path:
        dl_ready = True
        write_event({"event":"dl_artifacts","model":dl_model_path,"scaler":dl_scaler_path,"kind":DL_MODEL_KIND})
    elif USE_DL_SIGNALS:
        write_event({"event":"dl_artifacts_missing","model":bool(dl_model_path),"scaler":bool(dl_scaler_path)})

    # ---- update runtime DL status for /model_status
    DL_STATUS.update({
        "ready": bool(USE_DL_SIGNALS and infer_scores and dl_model_path and dl_scaler_path),
        "kind": DL_MODEL_KIND,
        "seq_len": DL_SEQ_LEN,
        "p_long": DL_P_LONG,
        "max_rv": DL_MAX_RV,
        "model_path": dl_model_path or "",
        "scaler_path": dl_scaler_path or "",
    })

    client = live_client() if LIVE_MODE else None
    universe = build_universe()
    print(f"Multi-symbol Bitget futures | LIVE={LIVE_MODE} | topN={len(universe)} | th={STATE.get('pred_threshold', PRED_THRESHOLD)}")
    if SPINNER: print()

    paper_equity = START_EQUITY
    day_start_eq = START_EQUITY
    day_key = datetime.now().strftime("%Y-%m-%d")

    positions: Dict[str, Dict[str, Any]] = {}
    cooldowns: Dict[str, int] = {}
    last_order_at: Dict[str, float] = {}
    last_pos_sync = 0.0

    # --- Day-1 feature cache (pull once per loop, reused across symbols)
    tier1_cache: Dict[str, Dict[str, Any]] = {}
    last_tier1_pull = 0.0

    # --- Day-4: per-symbol DL buffers (L x F)
    dl_bufs: Dict[str, deque] = { }  # sym -> deque of np.ndarray shape [F], maxlen=DL_SEQ_LEN

    # initial sync (avoid duplicate re-entry on reconnect)
    if LIVE_MODE:
        try:
            ex_pos = load_live_positions_map(client, universe)
            if ex_pos:
                for sym, info in ex_pos.items():
                    st = positions.setdefault(sym, {"in_pos": False, "dir": None, "hold": 0})
                    st.update({"in_pos": True, "dir": info["dir"], "entry": info.get("entry"), "qty": info.get("qty", 0.0)})
        except Exception:
            pass

    tf_secs = timeframe_to_seconds(TIMEFRAME)

    while not STOP_EVENT.is_set():
        if int(time.time()) % 60 < poll_s:
            write_event({"event": "heartbeat", "note": "run_loop alive"})
        try:
            paused = bool(STATE.get("paused", False))
            pred_th = float(STATE.get("pred_threshold", PRED_THRESHOLD))
            maxc = int(STATE.get("max_concurrent", MAX_CONCURRENT))

            # daily reset
            now_key = datetime.now().strftime("%Y-%m-%d")
            if now_key != day_key:
                day_key = now_key; day_start_eq = paper_equity

            # periodic universe refresh (~20m)
            if int(time.time()) % 1200 < poll_s:
                try:
                    universe = build_universe()
                    if SPINNER: print()
                except Exception: pass

            # --- refresh Day-1 features (about every 45s)
            if FEATURE_ENABLE and (time.time() - last_tier1_pull > 45.0):
                try:
                    tier1_cache = get_latest_features(universe)
                    last_tier1_pull = time.time()
                    for sym, f in tier1_cache.items():
                        update_spread_history(sym, f.get("spread_bps"))
                    write_event({"event": "tier1_features_pull", "symbols": len(tier1_cache)})
                except Exception as e:
                    write_event({"event": "tier1_features_pull", "ok": False, "error": str(e)[:180]})

            # live position sync
            if LIVE_MODE and (time.time() - last_pos_sync >= POS_SYNC_SECS):
                ok = False
                try:
                    ex_pos = load_live_positions_map(client, universe)
                    if ex_pos is not None:
                        for sym in universe:
                            st = positions.setdefault(sym, {"in_pos": False, "dir": None, "hold": 0})
                            st.update({"in_pos": False, "dir": None})
                        for sym, info in ex_pos.items():
                            st = positions.setdefault(sym, {"in_pos": False, "dir": None, "hold": 0})
                            st.update({"in_pos": True, "dir": info["dir"], "entry": info.get("entry"), "qty": info.get("qty", 0.0)})
                        last_pos_sync = time.time(); ok = True
                        write_event({"event": "sync_positions", "mode": "live", "count": len(ex_pos), "ok": True})
                except Exception as e:
                    write_event({"event":"sync_positions","mode":"live","ok":False,"error":str(e)[:180]})
                if not ok: write_event({"event":"sync_positions","mode":"live","ok":False})

            # DD gate
            dd = (paper_equity - day_start_eq) / max(day_start_eq, 1e-9)
            if dd <= -MAX_DD:
                time.sleep(60); continue

            # force exits (from Telegram)
            if FORCE_EXIT_ALL.is_set() and LIVE_MODE:
                try:
                    for sym, st in list(positions.items()):
                        if not st.get("in_pos"): continue
                        amt = normalize_amount_for_derivs(client, sym, st["qty"])
                        side = "sell" if st["dir"] == "long" else "buy"
                        client.create_order(sym, "market", side, amt, None, {"reduceOnly": True})
                        write_event({"event":"order","mode":"live","symbol":sym,"side":"CLOSE_ALL"})
                        st.update({"in_pos": False, "dir": None})
                    FORCE_EXIT_ALL.clear()
                except Exception: pass

            # loop symbols
            open_count = sum(1 for s in positions.values() if s.get("in_pos"))
            can_open = open_count < maxc

            price_map: Dict[str, float] = {}

            for idx, sym in enumerate(list(universe), 1):
                spinner_line(sym, idx, len(universe), phase="scan"); write_current(sym, idx, len(universe), phase="scan")

                # per-symbol force exit
                if sym in FORCE_EXIT_SYMBOLS and LIVE_MODE:
                    try:
                        st = positions.get(sym, {})
                        if st.get("in_pos"):
                            amt = normalize_amount_for_derivs(client, sym, st["qty"])
                            side = "sell" if st["dir"] == "long" else "buy"
                            client.create_order(sym, "market", side, amt, None, {"reduceOnly": True})
                            write_event({"event":"order","mode":"live","symbol":sym,"side":"CLOSE"})
                            st.update({"in_pos": False, "dir": None})
                    except Exception: pass
                    finally:
                        FORCE_EXIT_SYMBOLS.discard(sym)
                    continue

                # cooldown
                if cooldowns.get(sym, 0) > 0:
                    cooldowns[sym] -= 1; continue

                st = positions.setdefault(sym, {"in_pos": False, "dir": None, "hold": 0})

                # throttles — only block *new entries*
                if LIVE_MODE and CHECK_OPEN_ORDERS and (not st["in_pos"]) and has_open_orders(client, sym):
                    continue
                if (not st["in_pos"]) and (time.time() - last_order_at.get(sym, 0) < RECENT_ORDER_COOLDOWN_S):
                    continue

                # fetch bars (your 5m)
                try:
                    df = fetch_ohlcv(sym, TIMEFRAME, LOOKBACK)
                except Exception:
                    continue

                # data gap circuit-breaker (last bar too old)
                try:
                    last_ts = int(pd.to_datetime(df.index[-1]).timestamp())
                    if time.time() - last_ts > 3 * tf_secs:
                        write_event({"event":"data_gap_skip","symbol":sym,"age_s":int(time.time() - last_ts)})
                        continue
                except Exception:
                    pass

                spinner_line(sym, idx, len(universe), phase="features")
                feats = build_features(df)
                need = [c for c in ("ema_20","ema_50","atr_14","close") if c not in feats.columns]
                if need:
                    write_event({"event":"missing_features_min","symbol":sym,"missing":need}); continue
                if "trend_ok" not in feats.columns:    feats["trend_ok"] = (feats["ema_12"] > feats["ema_26"]).astype(int)
                if "momentum_ok" not in feats.columns: feats["momentum_ok"] = ((feats["close"] > feats["ema_20"]) & (feats["ema_20"] > feats["ema_50"])).astype(int)
                if "vol_ok" not in feats.columns:      feats["vol_ok"] = (feats["atr_14"]/feats["close"] > 0.0015).astype(int)

                missing = [c for c in fcols if c not in feats.columns]
                if missing:
                    write_event({"event":"missing_features","symbol":sym,"missing":missing}); continue

                spinner_line(sym, idx, len(universe), phase="score")

                # --- Day-3 probability (classical) ---
                X = feats[fcols].iloc[-1:].copy()
                proba = float(model.predict_proba(X)[:, 1][0])
                price = float(feats["close"].iloc[-1]); atr = float(feats["atr_14"].iloc[-1])
                price_map[sym] = price

                trend_up = int(feats["trend_ok"].iloc[-1]) == 1
                vol_ok   = int(feats["vol_ok"].iloc[-1]) == 1
                mom_up   = int(feats["momentum_ok"].iloc[-1]) == 1

                rule_long  = 0.8 if mom_up else 0.2
                p_long_d3  = 0.7 * proba + 0.3 * rule_long
                rule_short = 0.8 if not mom_up else 0.2
                p_short_d3 = 0.7 * (1.0 - proba) + 0.3 * rule_short

                if LONG_ONLY:  p_short_d3 = -1.0
                if SHORT_ONLY: p_long_d3  = -1.0

                # ---- Day-1 microstructure gate ----
                rv_val = None
                f = tier1_cache.get(sym, {}) if FEATURE_ENABLE else {}
                if f:
                    if not st["in_pos"]:
                        if not f.get("pass_spread_guard", False):
                            write_event({"event":"spread_guard_skip","symbol":sym,"spread_bps":f.get("spread_bps")}); continue
                        if not f.get("pass_vol_gate", True):
                            write_event({"event":"vol_gate_skip","symbol":sym,"vol_z":f.get("vol_z")}); continue
                        if not f.get("pass_rsi_gate", True):
                            write_event({"event":"rsi_gate_skip","symbol":sym,"rsi":f.get("rsi")}); continue
                        if abs(float(f.get("imbalance") or 0.0)) < IMB_ABS_MIN:
                            write_event({"event":"imbalance_skip","symbol":sym,"imb":f.get("imbalance")}); continue
                        med = rolling_median_spread(sym)
                        sp = float(f.get("spread_bps") or np.nan)
                        if med is not None and sp == sp:
                            if should_pause_spread(current_spread_bps=sp, median_bps=med, pause_bps=SPREAD_PAUSE_BPS, widen_mult=SPREAD_WIDEN_MULT):
                                write_event({"event":"spread_widen_pause","symbol":sym,"spread_bps":sp,"median":med}); continue
                    rv_val = f.get("rv")

                # ======== Day-4 DL inference (optional) ========
                use_dl = USE_DL_SIGNALS and infer_scores and dl_ready
                dl_scores = None
                if use_dl:
                    # maintain per-symbol buffer
                    dq = dl_bufs.get(sym)
                    if dq is None:
                        dq = deque(maxlen=DL_SEQ_LEN); dl_bufs[sym] = dq
                    vec = feats[fcols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
                    dq.append(vec)

                    if len(dq) >= DL_SEQ_LEN:
                        xwin = np.stack(list(dq), axis=0)
                        try:
                            dl_scores = infer_scores(
                                X_windowed=xwin,
                                scaler_path=dl_scaler_path,
                                model_path=dl_model_path,
                                kind=DL_MODEL_KIND,
                                device="cpu"
                            )
                            # remember latest per-symbol DL scores for /model_status
                            try:
                                DL_STATUS["last"][sym] = {
                                    "ts": _ts_utc(),
                                    "p_long": float(dl_scores.get("p_long", float("nan"))),
                                    "ret_hat": float(dl_scores.get("ret_hat", float("nan"))),
                                    "rv_hat": float(dl_scores.get("rv_hat", float("nan"))),
                                }
                            except Exception:
                                pass

                            # override gates if DL active
                            p_long = float(dl_scores["p_long"])
                            p_short = 1.0 - p_long
                            rv_hat = float(dl_scores.get("rv_hat", np.nan))
                            if rv_hat == rv_hat:  # not NaN
                                rv_val = rv_hat  # use model RV as stop/risk input
                        except Exception as e:
                            write_event({"event":"dl_infer_error","symbol":sym,"err":str(e)[:160]})
                            # fallback to Day-3
                            p_long, p_short = p_long_d3, p_short_d3
                    else:
                        # buffer not ready yet -> fallback to Day-3
                        p_long, p_short = p_long_d3, p_short_d3
                else:
                    p_long, p_short = p_long_d3, p_short_d3

                # entries disabled when paused
                if paused: continue

                # ===== Entry gating =====
                if use_dl and dl_scores is not None:
                    # DL-driven entry rule
                    if (not st["in_pos"]):
                        if p_long < DL_P_LONG or (dl_scores["rv_hat"] >= DL_MAX_RV if dl_scores.get("rv_hat") is not None else False):
                            write_event({"event":"dl_skip_entry","symbol":sym,"p_long":p_long,"rv_hat":dl_scores.get("rv_hat")})
                            continue
                else:
                    # Day-3 gating
                    if (not st["in_pos"]) and (p_long < pred_th) and (p_short < pred_th):
                        write_event({"event":"skip_entry","symbol":sym,"p_long":p_long,"p_short":p_short,"pred_th":pred_th})
                        continue

                # ===== ENTRY LONG =====
                if (not st["in_pos"]) and can_open and trend_up and vol_ok and p_long >= (DL_P_LONG if use_dl else pred_th):
                    risk_abs = stop_distance_abs(entry=price, atr=atr, rv=rv_val,
                                                 stop_atr_mult=STOP_ATR_MULT, rv_mult=RV_STOP_MULT, min_stop_frac=RISK_MIN_STOP_FRAC)
                    stop = price - risk_abs; take = price + TP_R_MULT * risk_abs
                    base_qty = size_from_risk(equity=paper_equity, risk_frac=RISK_PER_TRADE, entry=price, stop=stop)

                    wallet_usdt = fetch_wallet_usdt(client) if LIVE_MODE else None
                    base_qty = cap_notional(price=price, qty=base_qty, leverage=LEVERAGE, wallet_usdt=wallet_usdt,
                                            per_symbol_cap=PER_SYMBOL_NOTIONAL or MAX_NOTIONAL_USDT, max_margin_frac=MAX_MARGIN_FRACTION)
                    if base_qty <= 0: continue

                    if MAX_PORTFOLIO_EXPOSURE > 0:
                        tot = total_open_notional(positions, price_map)
                        if not portfolio_exposure_ok(current_total=tot, new_notional=base_qty * price, cap=MAX_PORTFOLIO_EXPOSURE):
                            write_event({"event":"portfolio_cap_skip","symbol":sym,"current":tot,"new":base_qty * price}); continue

                    router_info = None
                    if ROUTER_ENABLED:
                        try:
                            q = best_cex_quote(sym, "buy", base_qty)
                            if q:
                                router_info = {"venue": q["venue"], "vwap": float(q["vwap"]), "eff_bps": float(q["eff_bps"])}
                                if float(q.get("eff_bps", 0)) > SLIPPAGE_MAX_BPS:
                                    write_event({"event":"slippage_anomaly_skip","symbol":sym,"eff_bps":float(q.get("eff_bps"))}); continue
                        except Exception: pass

                    if LIVE_MODE:
                        mp = load_live_positions_map(client, [sym])
                        if mp is None:
                            write_event({"event":"entry_abort","symbol":sym,"reason":"pos_sync_failed"}); continue
                        if sym in mp:
                            write_event({"event":"entry_abort","symbol":sym,"reason":"already_live_pos"}); continue
                        try:
                            mk = client.market(sym)
                            try: client.set_leverage(LEVERAGE, sym, params={"marginMode": MARGIN_MODE})
                            except Exception: pass
                            amt_to_send = normalize_amount_for_derivs(client, sym, base_qty)
                            min_amt = (mk.get("limits", {}).get("amount", {}).get("min") or 0.0)
                            if min_amt and float(amt_to_send) < float(min_amt):
                                amt_to_send = client.amount_to_precision(sym, min_amt)
                            client.create_order(sym, "market", "buy", amt_to_send)
                            last_order_at[sym] = time.time()
                            evt = {"event":"order","mode":"live","symbol":sym,"side":"buy","qty":float(amt_to_send),
                                   "stop":float(stop),"take":float(take),"p_src":"DL" if use_dl else "D3","p_long":float(p_long)}
                            if dl_scores: evt.update({"ret_hat":float(dl_scores["ret_hat"]), "rv_hat":float(dl_scores["rv_hat"])})
                            write_event(evt); spinner_flash(f"BUY {sym} qty≈{amt_to_send}", ok=True)
                        except Exception:
                            spinner_flash(f"BUY fail {sym}", ok=False); continue
                    else:
                        last_order_at[sym] = time.time()
                        evt = {"event":"order","mode":"paper","symbol":sym,"side":"buy","price":price,"qty":base_qty,
                               "stop":float(stop),"take":float(take),"p_src":"DL" if use_dl else "D3","p_long":float(p_long)}
                        if dl_scores: evt.update({"ret_hat":float(dl_scores["ret_hat"]), "rv_hat":float(dl_scores["rv_hat"])})
                        write_event(evt); spinner_flash(f"BUY {sym} qty≈{round(base_qty,6)}", ok=True)

                    st.update({"in_pos": True, "dir": "long", "entry": price, "stop": float(stop), "take": float(take), "qty": base_qty, "hold": 0})
                    log_trade(ts=str(df.index[-1]), symbol=sym, side="BUY", price=price, qty=base_qty, reason=f"pL={p_long:.3f}")
                    open_count += 1; can_open = open_count < maxc; continue

                # ===== ENTRY SHORT =====
                if (not st["in_pos"]) and can_open and (not trend_up) and vol_ok and ( (1-p_long) >= (DL_P_LONG if use_dl else pred_th) ) and (not LONG_ONLY):
                    risk_abs = stop_distance_abs(entry=price, atr=atr, rv=rv_val,
                                                 stop_atr_mult=STOP_ATR_MULT, rv_mult=RV_STOP_MULT, min_stop_frac=RISK_MIN_STOP_FRAC)
                    stop = price + risk_abs; take = price - TP_R_MULT * risk_abs
                    base_qty = size_from_risk(equity=paper_equity, risk_frac=RISK_PER_TRADE, entry=price, stop=stop)

                    wallet_usdt = fetch_wallet_usdt(client) if LIVE_MODE else None
                    base_qty = cap_notional(price=price, qty=base_qty, leverage=LEVERAGE, wallet_usdt=wallet_usdt,
                                            per_symbol_cap=PER_SYMBOL_NOTIONAL or MAX_NOTIONAL_USDT, max_margin_frac=MAX_MARGIN_FRACTION)
                    if base_qty <= 0: continue

                    if MAX_PORTFOLIO_EXPOSURE > 0:
                        tot = total_open_notional(positions, price_map)
                        if not portfolio_exposure_ok(current_total=tot, new_notional=base_qty * price, cap=MAX_PORTFOLIO_EXPOSURE):
                            write_event({"event":"portfolio_cap_skip","symbol":sym,"current":tot,"new":base_qty * price}); continue

                    router_info = None
                    if ROUTER_ENABLED:
                        try:
                            q = best_cex_quote(sym, "sell", base_qty)
                            if q:
                                router_info = {"venue": q["venue"], "vwap": float(q["vwap"]), "eff_bps": float(q["eff_bps"])}
                                if float(q.get("eff_bps", 0)) > SLIPPAGE_MAX_BPS:
                                    write_event({"event":"slippage_anomaly_skip","symbol":sym,"eff_bps":float(q.get("eff_bps"))}); continue
                        except Exception: pass

                    if LIVE_MODE:
                        mp = load_live_positions_map(client, [sym])
                        if mp is None:
                            write_event({"event":"entry_abort","symbol":sym,"reason":"pos_sync_failed"}); continue
                        if sym in mp:
                            write_event({"event":"entry_abort","symbol":sym,"reason":"already_live_pos"}); continue
                        try:
                            mk = client.market(sym)
                            try: client.set_leverage(LEVERAGE, sym, params={"marginMode": MARGIN_MODE})
                            except Exception: pass
                            amt_to_send = normalize_amount_for_derivs(client, sym, base_qty)
                            min_amt = (mk.get("limits", {}).get("amount", {}).get("min") or 0.0)
                            if min_amt and float(amt_to_send) < float(min_amt):
                                amt_to_send = client.amount_to_precision(sym, min_amt)
                            client.create_order(sym, "market", "sell", amt_to_send)
                            last_order_at[sym] = time.time()
                            evt = {"event":"order","mode":"live","symbol":sym,"side":"sell","qty":float(amt_to_send),
                                   "stop":float(stop),"take":float(take),"p_src":"DL" if use_dl else "D3","p_short":float(1-p_long)}
                            if dl_scores: evt.update({"ret_hat":float(dl_scores["ret_hat"]), "rv_hat":float(dl_scores["rv_hat"])})
                            write_event(evt); spinner_flash(f"SHORT {sym} qty≈{amt_to_send}", ok=True)
                        except Exception:
                            spinner_flash(f"SHORT fail {sym}", ok=False); continue
                    else:
                        last_order_at[sym] = time.time()
                        evt = {"event":"order","mode":"paper","symbol":sym,"side":"sell","price":price,"qty":base_qty,
                               "stop":float(stop),"take":float(take),"p_src":"DL" if use_dl else "D3","p_short":float(1-p_long)}
                        if dl_scores: evt.update({"ret_hat":float(dl_scores["ret_hat"]), "rv_hat":float(dl_scores["rv_hat"])})
                        write_event(evt); spinner_flash(f"SHORT {sym} qty≈{round(base_qty,6)}", ok=True)

                    st.update({"in_pos": True, "dir": "short", "entry": price, "stop": float(stop), "take": float(take), "qty": base_qty, "hold": 0})
                    log_trade(ts=str(df.index[-1]), symbol=sym, side="SELL_SHORT", price=price, qty=base_qty, reason=f"pS={(1-p_long):.3f}")
                    open_count += 1; can_open = open_count < maxc; continue

                # ===== MANAGE / EXIT =====
                if st["in_pos"]:
                    trail_abs = stop_distance_abs(entry=price, atr=atr, rv=rv_val,
                                                  stop_atr_mult=STOP_ATR_MULT, rv_mult=RV_STOP_MULT, min_stop_frac=RISK_MIN_STOP_FRAC)
                    if st["dir"] == "long":
                        st["stop"] = max(st["stop"], price - trail_abs)
                        hit = (price <= st["stop"]) or (price >= st["take"])
                    else:
                        st["stop"] = min(st["stop"], price + trail_abs)
                        hit = (price >= st["stop"]) or (price <= st["take"])

                    st["hold"] += 1
                    tstop = TIME_STOP_BARS if (isinstance(TIME_STOP_BARS, int) and TIME_STOP_BARS > 0) else MAX_HOLD_BARS
                    time_exit = (tstop > 0 and st["hold"] >= tstop)

                    if hit or time_exit:
                        reason = "exit_time" if time_exit else ("exit_sl" if ((st["dir"] == "long" and price <= st["stop"]) or (st["dir"] == "short" and price >= st["stop"])) else "exit_tp")
                        fees = bps_to_frac(FEES_BPS)
                        pnl = (((price - st["entry"]) if st["dir"] == "long" else (st["entry"] - price)) / max(st["entry"], 1e-12) - fees)
                        paper_equity *= (1 + pnl)
                        if LIVE_MODE:
                            try:
                                amt = normalize_amount_for_derivs(client, sym, st["qty"])
                                if st["dir"] == "long":
                                    client.create_order(sym, "market", "sell", amt, None, {"reduceOnly": True}); side = "SELL"
                                else:
                                    client.create_order(sym, "market", "buy", amt, None, {"reduceOnly": True}); side = "BUY_CLOSE"
                                write_event({"event":"order","mode":"live","symbol":sym,"side":side,"reason":reason})
                                spinner_flash(f"EXIT {sym} {reason}", ok=True)
                            except Exception:
                                spinner_flash(f"EXIT fail {sym}", ok=False)
                        else:
                            side = "PAPER_SELL" if st["dir"] == "long" else "PAPER_BUY_CLOSE"
                            write_event({"event":"order","mode":"paper","symbol":sym,"side":side,"reason":reason,"pnl":pnl})
                        log_trade(ts=str(df.index[-1]), symbol=sym, side=side, price=price, qty=st["qty"], reason=reason)
                        if pnl < 0: cooldowns[sym] = COOLDOWN_BARS
                        st.update({"in_pos": False, "dir": None, "hold": 0})

            time.sleep(poll_s)

        except Exception:
            import traceback
            tb = traceback.format_exc()
            logging.error("run_loop iteration failed:\n%s", tb)
            try:
                write_event({"event": "loop_error", "tb": tb[:800]})
            except Exception:
                pass
            time.sleep(poll_s)

# ===================== telegram wiring =====================
def _wire_telegram() -> Optional[TelegramBot]:
    if not TELEGRAM_ENABLED: return None

    def _spawn_relauncher():
        code = r"""
import os, time, sys, subprocess
lock = os.path.join('logs','trader.lock')
for _ in range(240):
    if not os.path.exists(lock):
        subprocess.Popen([sys.executable, '-u', 'trade_multi_bitget.py'], cwd=os.getcwd())
        sys.exit(0)
    time.sleep(0.5)
try: os.remove(lock)
except Exception: ...
subprocess.Popen([sys.executable, '-u', 'trade_multi_bitget.py'], cwd=os.getcwd())
"""
        subprocess.Popen([sys.executable, "-c", code], cwd=os.getcwd())

    def on_status() -> str:
        st = STATE.snapshot()
        dl = "ON" if USE_DL_SIGNALS else "OFF"
        return (f"Status:\npaused={st.get('paused')} running={st.get('running')}\n"
                f"th={st.get('pred_threshold'):.3f} maxc={st.get('max_concurrent')} scan={st.get('scan_topn')}\n"
                f"timeframe={st.get('timeframe')} | DL={dl}/{DL_MODEL_KIND} L={DL_SEQ_LEN} p*={DL_P_LONG}")

    def on_pause() -> str:
        STATE.update(paused=True); return "Paused new entries."

    def on_resume() -> str:
        STATE.update(paused=False); return "Resumed."

    def on_stop() -> str:
        STATE.update(running=False); STOP_EVENT.set(); return "Stopping loop…"

    def on_restart() -> str:
        try:
            _spawn_relauncher(); STATE.update(running=False); STOP_EVENT.set(); return "Restarting…"
        except Exception as e:
            return f"Restart failed: {e}"

    def on_set_th(v: float) -> str:
        STATE.update(pred_threshold=float(v)); return f"Set threshold to {float(v):.3f}"

    def on_set_maxc(v: int) -> str:
        STATE.update(max_concurrent=int(v)); return f"Set max_concurrent to {int(v)}"

    def on_set_scan(v: int) -> str:
        STATE.update(scan_topn=int(v)); return f"Set scan_topn to {int(v)} (applies on next refresh)"

    def on_close_symbol(sym: str) -> str:
        FORCE_EXIT_SYMBOLS.add(sym.strip()); return f"Requested close for {sym}"

    def on_close_all() -> str:
        FORCE_EXIT_ALL.set(); return "Requested close ALL."

    def on_positions() -> str:
        try:
            ex = live_client() if LIVE_MODE else None
            mp = load_live_positions_map(ex, None) if ex else {}
            if not mp: return "No live positions."
            lines = [f"{k} {v.get('dir')} qty={v.get('qty')} entry={v.get('entry')}" for k, v in mp.items()]
            return "Open positions:\n" + "\n".join(lines)
        except Exception as e:
            return f"Positions error: {e}"

    def on_report() -> str:
        return make_yesterday_report()

    # ---- NEW: /model_status command ----
    def on_model_status() -> str:
        st = DL_STATUS
        if not st.get("ready", False):
            return ("DL status:\n"
                    "  ready=FALSE (artifacts missing or USE_DL_SIGNALS=0)\n"
                    f"  kind={st.get('kind')} L={st.get('seq_len')} "
                    f"p*={st.get('p_long')} max_rv={st.get('max_rv')}\n"
                    f"  model={os.path.basename(st.get('model_path') or 'N/A')}\n"
                    f"  scaler={os.path.basename(st.get('scaler_path') or 'N/A')}")
        last = st.get("last", {})
        # show up to 5 most recent symbols
        items = list(last.items())
        items = items[-5:] if len(items) > 5 else items
        lines = []
        for sym, rec in items:
            try:
                lines.append(f"{sym}  p_long={rec['p_long']:.3f}  ret̂={rec['ret_hat']:.5f}  rv̂={rec['rv_hat']:.5f}  @ {rec['ts']}")
            except Exception:
                pass
        body = "\n".join(lines) if lines else "No DL inferences yet."
        return ( "DL status:\n"
                 f"  ready=TRUE  kind={st.get('kind')} L={st.get('seq_len')} "
                 f"p*={st.get('p_long')} max_rv={st.get('max_rv')}\n"
                 f"  model={os.path.basename(st.get('model_path') or '')}\n"
                 f"  scaler={os.path.basename(st.get('scaler_path') or '')}\n"
                 f"{body}" )

    token = (os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip()
    if not token:
        print("Telegram disabled: TELEGRAM_BOT_TOKEN missing"); return None

    bot = TelegramBot(
        token=token,
        state=STATE,
        callbacks={
            "on_status": on_status,
            "on_pause": on_pause,
            "on_resume": on_resume,
            "on_stop": on_stop,
            "on_restart": on_restart,
            "on_set_th": on_set_th,
            "on_set_maxc": on_set_maxc,
            "on_set_scan": on_set_scan,
            "on_close_symbol": on_close_symbol,
            "on_close_all": on_close_all,
            "on_positions": on_positions,
            "on_report": on_report,
            # NEW
            "on_model_status": on_model_status,
        },
        storage_dir="logs",
        daily_report_hhmm=TELEGRAM_DAILY,
        enable_poll=TELEGRAM_POLL,
    )
    bot.start()
    try: bot.send("🤖 Trader is online.")
    except Exception: pass
    return bot

# ===================== boot/shutdown breadcrumbs =====================
@atexit.register
def _on_exit():
    try: write_event({"event":"shutdown"})
    except Exception: pass

# ===================== main =====================
if __name__ == "__main__":
    ok = ensure_single_instance()
    if not ok: sys.exit(0)

    try:
        write_event({"event":"boot","pid":os.getpid()})
        print(f"Multi-symbol Bitget futures | LIVE={LIVE_MODE} | topN={SCAN_TOPN} | th={PRED_THRESHOLD}\n")
    except Exception: pass

    BOT = _wire_telegram()
    try:
        run_loop()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        print("[fatal] unhandled exception:"); traceback.print_exc()
    finally:
        if BOT:
            try: BOT.stop()
            except Exception: pass
