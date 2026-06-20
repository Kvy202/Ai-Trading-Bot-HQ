import os, time, joblib
from datetime import datetime
import pandas as pd
from router import best_cex_quote, ROUTER_ENABLED

from universe import bitget_universe
from data import fetch_ohlcv, get_exchange
from exchange import live_client, market_buy, market_sell
from features import build_features
from utils import position_size, bps_to_frac
from multi_exchange import best_cex_quote

MODEL_PATH = os.getenv("MODEL_PATH","models/model.pkl")
LOG_PATH   = os.getenv("LOG_PATH","logs/trades.csv")
TIMEFRAME  = os.getenv("TIMEFRAME","5m")
PRED_TH    = float(os.getenv("PRED_THRESHOLD","0.7"))
STOPK      = float(os.getenv("STOP_ATR_MULT","2"))
TPK        = float(os.getenv("TP_R_MULT","2.5"))
RISK_R     = float(os.getenv("RISK_PER_TRADE","0.003"))
FEES_BPS   = float(os.getenv("FEES_BPS","6"))
LIVE_MODE  = bool(int(os.getenv("LIVE_MODE","0")))
USE_SWAP   = bool(int(os.getenv("SCAN_INCLUDE_SWAP","1")))
USE_SPOT   = bool(int(os.getenv("SCAN_INCLUDE_SPOT","1")))
DERIVS     = bool(int(os.getenv("DERIVS","1")))

def ensure_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["ts","symbol","mtype","side","price","qty","reason"]).to_csv(LOG_PATH, index=False)

def log_trade(**k):
    df = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame(columns=["ts","symbol","mtype","side","price","qty","reason"])
    df.loc[len(df)] = [k.get('ts'), k.get('symbol'), k.get('mtype'), k.get('side'), k.get('price'), k.get('qty'), k.get('reason')]
    df.to_csv(LOG_PATH, index=False)

def load_model():
    b = joblib.load(MODEL_PATH)
    return b["model"], b["features"]

def main_loop():
    ensure_log()
    model, fcols = load_model()

    # build universe
    uni = bitget_universe(
        quote=os.getenv("SCAN_QUOTE","USDT"),
        topn=int(os.getenv("SCAN_TOPN","25")),
        min_notional=float(os.getenv("SCAN_MIN_NOTIONAL","30")),
        include_spot=USE_SPOT, include_swap=USE_SWAP,
    )
    print(f"[Universe] {len(uni)} markets:", [u["symbol"]+("/S" if u["type"]=="swap" else "/spot") for u in uni])

    # Bitget public
    ex_pub = get_exchange()

    # Bitget auth for orders
    client = live_client() if LIVE_MODE else ex_pub

    # per-symbol state
    state = {u["symbol"]: {"in_pos": False, "entry":0.0, "stop":0.0, "take":0.0, "qty":0.0, "mtype":u["type"]} for u in uni}
    equity = float(os.getenv("START_EQUITY","10000"))

    while True:
        for u in uni:
            sym, mtype = u["symbol"], u["type"]
            try:
                df = fetch_ohlcv(sym, TIMEFRAME, 1000)    # public client respects defaultType from .env
            except Exception as e:
                print("[OHLCV]", sym, e); time.sleep(1); continue

            feats = build_features(df)
            if not set(fcols).issubset(feats.columns):
                print("[SKIP] missing features", sym); continue

            X = feats[fcols].iloc[-1:]
            proba = float(model.predict_proba(X)[:,1][0])
            price = float(feats["close"].iloc[-1])
            atr   = float(feats["atr_14"].iloc[-1])

            trend_ok   = int(feats.get("trend_ok", pd.Series([1], index=X.index)).iloc[-1]) == 1
            momentum_ok= int(feats.get("momentum_ok", pd.Series([1], index=X.index)).iloc[-1]) == 1
            vol_ok     = int(feats.get("vol_ok", pd.Series([1], index=X.index)).iloc[-1]) == 1

            # simple ensemble
            rule_prob = 0.8 if momentum_ok else 0.2
            p_ens = 0.7*proba + 0.3*rule_prob

            S = state[sym]

            # ----- Entry -----
            if (not S["in_pos"]) and trend_ok and vol_ok and p_ens >= PRED_TH:
                risk = STOPK * max(atr, 1e-8)
                S["stop"] = price - risk
                S["take"] = price + TPK * risk
                base_qty  = position_size(equity, RISK_R, price, S["stop"])
                if base_qty <= 0:
                    continue

                # convert to contracts for swap
                order_qty = base_qty
                try:
                    mk = ex_pub.market(sym)
                    if mtype == "swap":
                        csz = float(mk.get("contractSize") or 1.0)
                        order_qty = base_qty / max(csz,1e-12)
                    order_qty = ex_pub.amount_to_precision(sym, order_qty)
                except Exception:
                    pass

                # cross-venue quote (analysis only)
                try:
                    q = best_cex_quote(sym, "buy", float(order_qty) if mtype=="swap" else float(base_qty))
                    if q: print(f"[ANALYZE] {sym} best={q['venue']}/{q['mtype']} vwap={q['vwap']:.6f} eff_bps={q['eff_bps']:.1f}")
                except Exception: pass

                # place order (Bitget only)
                if LIVE_MODE:
                    try:
                        market_buy(client, sym, order_qty)
                    except Exception as e:
                        print("[BUY FAIL]", sym, e); continue

                S.update({"in_pos": True, "entry": price, "qty": order_qty})
                log_trade(ts=str(df.index[-1]), symbol=sym, mtype=mtype,
                          side=("BUY" if LIVE_MODE else "PAPER_BUY"), price=price, qty=order_qty,
                          reason=f"p_ens={p_ens:.3f}")

            # ----- Manage / Exit -----
            elif S["in_pos"]:
                trail = STOPK * atr
                S["stop"] = max(S["stop"], price - trail)
                exit_now = (price <= S["stop"]) or (price >= S["take"])

                if exit_now:
                    fees = bps_to_frac(FEES_BPS)
                    pnl = (price - S["entry"]) / max(S["entry"],1e-12) - fees
                    equity *= (1 + pnl)
                    if LIVE_MODE:
                        try:
                            market_sell(client, sym, S["qty"])
                        except Exception as e:
                            print("[SELL FAIL]", sym, e)
                    log_trade(ts=str(df.index[-1]), symbol=sym, mtype=mtype,
                              side=("SELL" if LIVE_MODE else "PAPER_SELL"),
                              price=price, qty=S["qty"], reason=("exit_tp" if price>=S["take"] else "exit_sl"))
                    S.update({"in_pos": False, "entry":0.0, "stop":0.0, "take":0.0, "qty":0.0})

            time.sleep(0.2)   # gentle rate limit

        # small idle between full passes
        time.sleep(3)

if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    main_loop()
