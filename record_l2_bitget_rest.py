# record_l2_bitget_rest.py — sample Bitget order book via REST and save to CSV (1-row/sec per symbol)
from __future__ import annotations
import os, time, argparse
import pandas as pd
import ccxt

def top_of_book(ob: dict, k: int = 5):
    bids = ob.get("bids", [])[:k]
    asks = ob.get("asks", [])[:k]
    if not bids or not asks: return None
    bid_px, bid_sz = float(bids[0][0]), float(bids[0][1])
    ask_px, ask_sz = float(asks[0][0]), float(asks[0][1])
    bid_sum = sum(float(x[1]) for x in bids)
    ask_sum = sum(float(x[1]) for x in asks)
    mid = (bid_px + ask_px) / 2.0
    spread = ask_px - bid_px
    spread_bps = (spread / mid) * 1e4 if mid > 0 else None
    return bid_px, bid_sz, ask_px, ask_sz, mid, spread, spread_bps, bid_sum, ask_sum

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--hz", type=float, default=1.0, help="samples per second (e.g., 1.0)")
    ap.add_argument("--outdir", default="data/l2/bitget_rest")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType":"swap","defaultSubType":"linear","defaultSettle":"USDT"}})
    ex.load_markets()

    try:
        print(f"Sampling {symbols} at {args.hz} Hz…  Ctrl+C to stop.")
        period = 1.0/max(1e-6,args.hz)
        writers = {s: open(os.path.join(args.outdir, f"{s.replace('/','_').replace(':','_')}.csv"), "a", encoding="utf-8") for s in symbols}
        for s in symbols:
            if os.path.getsize(writers[s].name) == 0:
                writers[s].write("ts,symbol,bid_px,bid_sz,ask_px,ask_sz,mid_px,spread,spread_bps,bid_sz_sum_k,ask_sz_sum_k\n")
                writers[s].flush()

        while True:
            t_ms = int(time.time()*1000)
            for s in symbols:
                try:
                    ob = ex.fetch_order_book(s, limit=5)
                    tup = top_of_book(ob, k=5)
                    if not tup: continue
                    bid_px,bid_sz,ask_px,ask_sz,mid,spread,spread_bps,bid_sum,ask_sum = tup
                    line = f"{t_ms},{s},{bid_px},{bid_sz},{ask_px},{ask_sz},{mid},{spread},{spread_bps},{bid_sum},{ask_sum}\n"
                    writers[s].write(line)
                except Exception:
                    pass
            for s in symbols:
                try: writers[s].flush()
                except Exception: pass
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        for s in symbols:
            try: writers[s].close()
            except Exception: pass
        try: ex.close()
        except Exception: pass

if __name__ == "__main__":
    main()
