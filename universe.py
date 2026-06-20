import os, ccxt, math
from typing import List, Dict

def _bitget(load_type="swap"):
    ex = ccxt.bitget({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": load_type, "defaultSubType": "linear", "defaultSettle": "USDT"},
    })
    ex.load_markets()
    return ex

def bitget_universe(quote="USDT", topn=25, min_notional=30,
                    include_spot=True, include_swap=True) -> List[Dict]:
    out = []

    if include_spot:
        spot = _bitget("spot")
        for m in spot.markets.values():
            if not m.get("spot"): continue
            if m.get("quote") != quote: continue
            t = (m.get("info") or {}).get("baseCoin") or m.get("base")
            if t and any(x in t.upper() for x in ["UP","DOWN","BULL","BEAR","3L","3S","5L","5S"]):  # avoid leveraged tokens
                continue
            vol_usd = float(m.get("info",{}).get("quoteVolume", 0)) or 0.0
            min_cost = float((m.get("limits",{}).get("cost",{}) or {}).get("min") or 0.0)
            if min_cost and min_cost > min_notional:   # some illiquid pairs
                continue
            out.append({"symbol": m["symbol"], "type":"spot", "vol_usd":vol_usd})

    if include_swap:
        swap = _bitget("swap")
        for m in swap.markets.values():
            if not m.get("swap"): continue
            if m.get("settle","USDT") != quote: continue
            vol_usd = float(m.get("info",{}).get("quoteVolume", 0)) or 0.0
            out.append({"symbol": m["symbol"], "type":"swap", "vol_usd":vol_usd})

    out = sorted(out, key=lambda x: x["vol_usd"], reverse=True)
    if topn > 0:
        out = out[:topn]

    # enforce min notional via Bitget limits (best-effort)
    return out
