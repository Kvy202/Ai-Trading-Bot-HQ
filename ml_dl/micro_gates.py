# ml_dl/micro_gates.py
import os
from typing import Dict, Tuple

# ---- env knobs (with sane defaults) ----
def _f(env, default):  # float
    try: return float(os.getenv(env, str(default)))
    except Exception: return default

def _b(env, default):  # bool
    return os.getenv(env, "1" if default else "0").lower() in ("1","true","yes","y")

MAX_SPREAD      = _f("MAX_SPREAD", 0.50)     # absolute spread in quote units (e.g., USDT)
MAX_SPREAD_BPS  = _f("MAX_SPREAD_BPS", 0.0)  # OR as bps of mid; if >0 this wins
MIN_LIQ         = _f("MIN_LIQ", 50_000)      # quote-volume threshold over the lookback window
LIQ_LOOKBACK_S  = int(_f("LIQ_LOOKBACK_SECS", 300))  # e.g., last 5m rolling sum
USE_ORDER_BOOK  = _b("USE_ORDER_BOOK", True) # if False, use ticker.best_bid/ask when available

def _spread_ok(best_bid: float, best_ask: float) -> bool:
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return False
    spread_abs = best_ask - best_bid
    if MAX_SPREAD_BPS > 0:
        mid = 0.5 * (best_ask + best_bid)
        spread_bps = (spread_abs / mid) * 10_000.0
        return spread_bps <= MAX_SPREAD_BPS
    return spread_abs <= MAX_SPREAD

# ------------------------- CCXT backend -------------------------
def _ccxt_exchange():
    import ccxt
    exid = os.getenv("CCXT_EXCHANGE", "binance")
    kwargs = {}
    if os.getenv("CCXT_API_KEY"): kwargs["apiKey"] = os.getenv("CCXT_API_KEY")
    if os.getenv("CCXT_SECRET"):  kwargs["secret"]  = os.getenv("CCXT_SECRET")
    if os.getenv("CCXT_PASSWORD"):kwargs["password"]= os.getenv("CCXT_PASSWORD")
    if os.getenv("CCXT_OPTIONS"):
        # optional JSON like {"defaultType":"future"}
        import json
        try: kwargs["options"] = json.loads(os.getenv("CCXT_OPTIONS"))
        except Exception: pass
    ex = getattr(ccxt, exid) (kwargs)
    ex.load_markets()
    return ex

def _liq_from_trades_ccxt(ex, symbol: str) -> float:
    # sum quote volume of recent trades within LIQ_LOOKBACK_S
    import time
    try:
        now = int(time.time() * 1000)
        since = now - LIQ_LOOKBACK_S * 1000
        trades = ex.fetch_trades(symbol, since=since, limit=1000)
        qv = 0.0
        for t in trades:
            # prefer quoteVolume if available; else price*amount
            if "cost" in t and t["cost"] is not None:
                qv += float(t["cost"])
            else:
                qv += float(t["price"]) * float(t["amount"])
        return float(qv)
    except Exception:
        # fallback to ticker quoteVolume (exchange dependent)
        try:
            tick = ex.fetch_ticker(symbol)
            return float(tick.get("quoteVolume", 0.0))
        except Exception:
            return 0.0

def _best_bid_ask_ccxt(ex, symbol: str) -> Tuple[float, float]:
    if USE_ORDER_BOOK:
        ob = ex.fetch_order_book(symbol, limit=5)
        best_bid = float(ob["bids"][0][0]) if ob["bids"] else 0.0
        best_ask = float(ob["asks"][0][0]) if ob["asks"] else 0.0
        return best_bid, best_ask
    tick = ex.fetch_ticker(symbol)
    best_bid = float(tick.get("bid", 0.0))
    best_ask = float(tick.get("ask", 0.0))
    # sometimes only last price is present; treat as no book = reject
    return best_bid, best_ask

def micro_gates_ccxt(symbols) -> Dict[str, Dict[str, float]]:
    """
    Returns {symbol: { 'ok':0/1, 'spread':x, 'liq':y, 'best_bid':b, 'best_ask':a }}
    OK when (spread <= threshold) and (liq >= MIN_LIQ).
    """
    ex = _ccxt_exchange()
    out = {}
    for s in symbols:
        try:
            bb, ba = _best_bid_ask_ccxt(ex, s)
            liq = _liq_from_trades_ccxt(ex, s)
            ok = int(_spread_ok(bb, ba) and (liq >= MIN_LIQ))
            out[s] = {"ok": ok, "spread": max(0.0, ba - bb), "liq": liq, "best_bid": bb, "best_ask": ba}
        except Exception:
            out[s] = {"ok": 0, "spread": float("inf"), "liq": 0.0, "best_bid": 0.0, "best_ask": 0.0}
    return out

# ------------------------- Public entry -------------------------
def micro_gates(symbols, source: str = None):
    """
    Dispatch by data source. Currently 'ccxt' supported.
    """
    source = (source or os.getenv("DATA_SOURCE", "ccxt")).lower()
    if source == "ccxt":
        return micro_gates_ccxt(symbols)
    raise NotImplementedError(f"micro_gates: unsupported DATA_SOURCE={source}")
