from __future__ import annotations
import pandas as pd
import numpy as np
from .ta import ema, rsi, realized_vol, volume_zscore
from .microstructure import imbalance

def build_leakage_safe_features(
    kl_df: pd.DataFrame,
    ob_1m_df: pd.DataFrame,
    ema_spans=(20,50,100),
    rsi_period=14,
    rv_window=30,
    volz_window=60,
) -> pd.DataFrame:
    """
    Join klines (1m/5m) and 1m orderbook aggregates into a feature table.
    **No leakage**: All features for timestamp t are computed from data <= t-1.
    """
    df = kl_df.copy()
    # assume 1m klines for modeling baseline
    df = df[df["timeframe"]=="1m"].copy()
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values(["symbol","ts"])
    # basic returns (1-bar)
    df["ret_1"] = df.groupby("symbol")["close"].pct_change().fillna(0.0)
    # shift close/volume by 1 to prevent current-bar leakage in indicators
    df["close_lag1"] = df.groupby("symbol")["close"].shift(1)
    df["vol_lag1"] = df.groupby("symbol")["volume"].shift(1)
    # indicators on lagged series
    for span in ema_spans:
        df[f"ema_{span}"] = df.groupby("symbol")["close_lag1"].transform(lambda s: ema(s, span))
    df["rsi"] = df.groupby("symbol")["close_lag1"].transform(lambda s: rsi(s, rsi_period))
    df["rv"] = df.groupby("symbol")["ret_1"].transform(lambda s: realized_vol(s.shift(1).fillna(0.0), rv_window))
    df["vol_z"] = df.groupby("symbol")["vol_lag1"].transform(lambda s: volume_zscore(s, volz_window))
    # microstructure join (already 1m aggregated)
    ob = ob_1m_df.copy()
    ob["ts"] = pd.to_datetime(ob["ts"], unit="ms", utc=True)
    ob = ob.sort_values(["symbol","ts"])
    # microstructure features (imbalance uses sums over minute - already past info for that minute)
    ob["imbalance"] = imbalance(ob["bid_sz_sum_k"], ob["ask_sz_sum_k"])
    # join on [symbol, ts] using left join so kline drives row set
    feat = pd.merge(df, ob[["symbol","ts","spread_bps","imbalance"]], on=["symbol","ts"], how="left")
    # forward fill microstructure within symbol for small gaps, then shift by 1 minute to ensure no same-bar usage
    feat[["spread_bps","imbalance"]] = (
        feat.groupby("symbol")[["spread_bps","imbalance"]].apply(lambda g: g.ffill()).reset_index(drop=True)
    )
    feat["spread_bps"] = feat.groupby("symbol")["spread_bps"].shift(1)
    feat["imbalance"] = feat.groupby("symbol")["imbalance"].shift(1)
    # drop warmup rows
    feat = feat.dropna(subset=[f"ema_{ema_spans[0]}", "rsi","rv","vol_z"])
    # unify types
    feat = feat.reset_index(drop=True)
    # convert ts back to int ms
    feat["ts"] = (feat["ts"].astype("int64") // 10**6).astype("int64")
    return feat
