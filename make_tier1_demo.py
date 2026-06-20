# make_tier1_demo.py — plots + trade markers + equity; optional GIF
from __future__ import annotations
import os, argparse
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering on Windows
import matplotlib.pyplot as plt

from atb.data.sources.klines_ccxt import fetch_klines
from atb.features.pipeline import build_leakage_safe_features
from backtest_day2 import baseline_signals_5m, run_sim_tp_sl   # reuse logic

def ensure_dir(d: str): os.makedirs(d, exist_ok=True)

def placeholder_ob(kl: pd.DataFrame) -> pd.DataFrame:
    one = kl[kl["timeframe"]=="1m"][["ts","symbol","close","volume"]].copy()
    one["bid_px"] = one["close"]*0.9995; one["ask_px"] = one["close"]*1.0005
    one["bid_sz"] = one["volume"].fillna(0).clip(lower=1); one["ask_sz"] = one["volume"].fillna(0).clip(lower=1)
    one = one.rename(columns={"close":"mid_px"})
    one["spread"] = one["ask_px"] - one["bid_px"]
    one["spread_bps"] = (one["spread"] / ((one["ask_px"]+one["bid_px"])/2))*1e4
    one["bid_sz_sum_k"] = one["bid_sz"]; one["ask_sz_sum_k"] = one["ask_sz"]
    return one[["ts","symbol","bid_px","bid_sz","ask_px","ask_sz","mid_px","spread","spread_bps","bid_sz_sum_k","ask_sz_sum_k"]]

def plot_lines(ts, series_dict, title, path):
    plt.figure()
    for label, s in series_dict.items():
        plt.plot(ts, s, label=label)
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

def plot_price_with_trades(five: pd.DataFrame, trades: pd.DataFrame, outpng: str, title: str):
    plt.figure()
    plt.plot(five["ts_dt"], five["close"], label="close")
    if "ema_fast" in five and "ema_slow" in five:
        plt.plot(five["ts_dt"], five["ema_fast"], label="ema_fast")
        plt.plot(five["ts_dt"], five["ema_slow"], label="ema_slow")
    # markers
    if not trades.empty:
        opens  = trades[~trades["ts_open"].isna()]
        closes = trades[~trades["ts_close"].isna()]
        if len(opens):
            plt.scatter(opens["ts_open"], opens["px_open"], marker="^", s=30, label="entry")
        if len(closes):
            # color exits by pnl sign
            good = closes["pnl"] > 0
            bad  = ~good
            if good.any():
                plt.scatter(closes.loc[good,"ts_close"], closes.loc[good,"px_close"], marker="v", s=30, label="exit +")
            if bad.any():
                plt.scatter(closes.loc[bad,"ts_close"], closes.loc[bad,"px_close"], marker="v", s=30, label="exit -")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(outpng, dpi=140); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default=os.getenv("EXCHANGE_ID","bitget"))
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--outdir", default="reports/tier1_demo")
    ap.add_argument("--gif", type=int, default=0, help="1 to also build an animated GIF of the equity curve (requires imageio)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    kl = fetch_klines(args.exchange, symbols, timeframe="1m", limit=args.limit, market_type="swap")
    if kl is None or kl.empty:
        print("No klines."); return
    ob = placeholder_ob(kl)

    all_stats = []
    for sym in sorted(kl["symbol"].unique()):
        feat = build_leakage_safe_features(kl[kl["symbol"]==sym], ob[ob["symbol"]==sym])
        if feat is None or feat.empty:
            print(f"[skip] no features for {sym}"); continue

        five = baseline_signals_5m(feat)
        if five.empty:
            print(f"[skip] no 5m signals for {sym}"); continue

        # simulate with TP/SL & latency costs
        res = run_sim_tp_sl(feat, equity0=float(os.getenv("BT_INIT_EQUITY","1000")))
        curve, trades, stats = res["curve"].reset_index(), res["trades"], res["stats"]
        all_stats.append({"symbol": sym, **stats})

        # Price+EMAs (clean)
        plot_lines(
            five["ts_dt"],
            {"close": five["close"], "ema_fast": five["ema_fast"], "ema_slow": five["ema_slow"]},
            f"{sym} — Price & EMAs (5m)",
            os.path.join(args.outdir, f"{sym.replace('/','_').replace(':','_')}_price_emas.png"),
        )
        # Price with trades overlay
        plot_price_with_trades(
            five, trades,
            os.path.join(args.outdir, f"{sym.replace('/','_').replace(':','_')}_price_trades.png"),
            f"{sym} — Entries/Exits (TP/SL + latency costs)"
        )
        # RSI
        if "rsi" in five.columns:
            plot_lines(five["ts_dt"], {"rsi": five["rsi"]}, f"{sym} — RSI", os.path.join(args.outdir, f"{sym.replace('/','_').replace(':','_')}_rsi.png"))
        # Spread
        if "spread_bps" in five.columns:
            plot_lines(five["ts_dt"], {"spread_bps": five["spread_bps"]}, f"{sym} — Spread (bps)", os.path.join(args.outdir, f"{sym.replace('/','_').replace(':','_')}_spread_bps.png"))
        # VolZ
        if "vol_z" in five.columns:
            plot_lines(five["ts_dt"], {"vol_z": five["vol_z"]}, f"{sym} — Volume z-score", os.path.join(args.outdir, f"{sym.replace('/','_').replace(':','_')}_volz.png"))
        # Equity curve
        if not curve.empty:
            plot_lines(curve["ts_dt"], {"equity": curve["equity"]}, f"{sym} — Equity (TP/SL)", os.path.join(args.outdir, f"{sym.replace('/','_').replace(':','_')}_equity.png"))

    # Export combined stats
    if all_stats:
        pd.DataFrame(all_stats).to_csv(os.path.join(args.outdir, "stats_day2.csv"), index=False)

    # Optional one-frame GIF of equity (just for easy share)
    if args.gif == 1 and len(symbols)==1:
        try:
            import imageio.v2 as imageio
            p = os.path.join(args.outdir, f"{symbols[0].replace('/','_').replace(':','_')}_equity.png")
            imageio.mimsave(os.path.join(args.outdir, f"{symbols[0].replace('/','_').replace(':','_')}_equity.gif"), [imageio.imread(p)], duration=0.6)
        except Exception as e:
            print(f"[gif skip] {e}")

if __name__ == "__main__":
    main()
