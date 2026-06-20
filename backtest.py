import math
import numpy as np
import pandas as pd
import joblib

from data import fetch_ohlcv
from features import build_features
from config import (
    SYMBOL, TIMEFRAME, LOOKBACK_CANDLES,
    START_EQUITY, FEES_BPS, PRED_THRESHOLD,
    STOP_ATR_MULT as CFG_STOP_ATR_MULT,
    TP_R_MULT as CFG_TP_R_MULT,
)
from utils import bps_to_frac


def _ensure_helpers(feats: pd.DataFrame) -> pd.DataFrame:
    """Ensure helper gates exist; add reasonable fallbacks if missing."""
    f = feats.copy()

    need = []
    for col in ["ema_12", "ema_26", "ema_20", "ema_50", "atr_14", "close"]:
        if col not in f.columns:
            need.append(col)
    if need:
        raise RuntimeError(f"features.py is missing columns: {need}")

    if "trend_ok" not in f.columns:
        f["trend_ok"] = (f["ema_12"] > f["ema_26"]).astype(int)
    if "momentum_ok" not in f.columns:
        f["momentum_ok"] = ((f["close"] > f["ema_20"]) & (f["ema_20"] > f["ema_50"])).astype(int)
    if "vol_ok" not in f.columns:
        # fallback: ~0.2% ATR, adjust in features.py for prod
        f["vol_ok"] = (f["atr_14"] / f["close"] > 0.002).astype(int)
    return f


def _simulate(
    feats: pd.DataFrame,
    prob: np.ndarray,
    th: float,
    stop_mult: float,
    tp_mult: float,
    max_hold_bars: int,
    trailing: bool = True,
) -> np.ndarray:
    """
    Simple long-only strategy:
      - Enter when prob >= th AND trend_ok & vol_ok.
      - Initial SL/TP based on ATR * multipliers.
      - Optional trailing stop (ratchet by ATR).
      - Time-based exit after max_hold_bars if still open.
    Returns per-trade returns (net of fees).
    """
    f = feats.copy()
    f["prob_up"] = prob
    f["signal"] = ((f["prob_up"] >= th) & (f["trend_ok"] == 1) & (f["vol_ok"] == 1)).astype(int)

    position = 0
    entry = stop = take = 0.0
    held = 0
    rets = []
    fee_frac = bps_to_frac(FEES_BPS)

    for _, row in f.iterrows():
        price = float(row["close"])
        atr = float(row["atr_14"])

        if position != 0:
            # trailing stop ratchet
            if trailing:
                trail = stop_mult * atr
                stop = max(stop, price - trail)

            # exits
            exit_reason = None
            if price <= stop:
                exit_reason = "SL"
            elif price >= take:
                exit_reason = "TP"
            elif max_hold_bars > 0 and held >= max_hold_bars:
                exit_reason = "TIME"

            if exit_reason is not None:
                pnl = (price - entry) / max(entry, 1e-12) - fee_frac
                rets.append(pnl)
                position = 0
                entry = stop = take = 0.0
                held = 0
            else:
                held += 1

        # entries
        if position == 0 and row["signal"] == 1:
            risk = stop_mult * max(atr, 1e-8)
            entry = price
            stop = entry - risk
            take = entry + tp_mult * risk
            position = 1
            held = 0

    return np.array(rets, dtype=float)


def _metrics_from_rets(rets: np.ndarray, start_equity: float):
    if len(rets) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "winrate": 0.0,
            "ending_equity": start_equity,
            "avg_trade": 0.0,
            "sharpe_naive": 0.0,
        }
    wins = int((rets > 0).sum())
    end_eq = float(start_equity * np.prod(1.0 + rets))
    avg = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (avg / std) * math.sqrt(len(rets)) if std > 0 else 0.0
    return {
        "trades": len(rets),
        "wins": wins,
        "winrate": wins / len(rets),
        "ending_equity": end_eq,
        "avg_trade": avg,
        "sharpe_naive": sharpe,
    }


def grid_scan(
    th_list,
    stop_list,
    take_list,
    hold_list,
    trailing=True,
    print_top_k=10,
):
    # Load data & features
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, LOOKBACK_CANDLES)
    feats = build_features(df).copy()
    feats = _ensure_helpers(feats)

    # Load model + probs aligned to features
    bundle = joblib.load("models/model.pkl")
    model = bundle["model"]
    fcols = bundle["features"]

    X = feats[fcols].copy().dropna()
    feats = feats.loc[X.index]
    prob = model.predict_proba(X)[:, 1]

    # diagnostics
    bars_all = len(feats)
    bars_trend = int((feats["trend_ok"] == 1).sum())
    bars_vol = int((feats["vol_ok"] == 1).sum())
    print(f"[Diag] bars={bars_all} | trend_ok={bars_trend} | vol_ok={bars_vol}")

    results = []
    for th in th_list:
        bars_prob = int((prob >= th).sum())
        if bars_prob == 0:
            results.append((th, None, None, None, 0, 0.0, START_EQUITY, 0.0, 0.0))
            continue

        for s in stop_list:
            for t in take_list:
                for h in hold_list:
                    rets = _simulate(feats, prob, th, s, t, h, trailing=trailing)
                    m = _metrics_from_rets(rets, START_EQUITY)
                    results.append((
                        th, s, t, h,
                        m["trades"], m["winrate"], m["ending_equity"],
                        m["avg_trade"], m["sharpe_naive"]
                    ))

    # sort by ending equity desc
    results = sorted(results, key=lambda r: r[6], reverse=True)

    # print table
    print("\nth\tstop\tTP\tmax_hold\ttrades\twinrate\tending_eq\tavg_trade\tsharpe")
    for row in results[:print_top_k]:
        th, s, t, h, ntr, wr, eq, avg, shp = row
        print(f"{th:.2f}\t{s}\t{t}\t{h}\t\t{ntr}\t{wr:.2%}\t{eq:.2f}\t{avg:.4f}\t{shp:.2f}")

    if results:
        best = results[0]
        print("\nBest by equity → "
              f"th={best[0]:.2f} | stop={best[1]} | TP={best[2]} | max_hold={best[3]} | "
              f"trades={best[4]} | win-rate={best[5]:.2%} | ending_equity={best[6]:.2f} | "
              f"avg_trade={best[7]:.4f} | sharpe={best[8]:.2f}")


if __name__ == "__main__":
    # Build small grids around your current settings.
    # Thresholds: around PRED_THRESHOLD, clipped to [0.30, 0.75]
    base_th = float(PRED_THRESHOLD)
    ths = sorted(set([
        max(0.30, min(0.75, round(base_th + d, 2)))
        for d in (-0.10, -0.05, 0.00, 0.05, 0.10, 0.15)
    ]))

    # Stop/TP grids (ATR multiples)
    stops = sorted(set([max(0.8, round(x, 2)) for x in (CFG_STOP_ATR_MULT, 1.0, 1.25, 1.5, 2.0)]))
    tps   = sorted(set([max(1.2, round(x, 2)) for x in (CFG_TP_R_MULT, 1.5, 2.0, 2.5, 3.0)]))

    # Time-based exit in bars (0 disables)
    holds = [0, 12, 24, 36]  # tune as you like

    print(f"[Grid] thresholds={ths}")
    print(f"[Grid] stops={stops}  tps={tps}  holds={holds}  (trailing stop: ON)")
    grid_scan(ths, stops, tps, holds, trailing=True, print_top_k=12)
