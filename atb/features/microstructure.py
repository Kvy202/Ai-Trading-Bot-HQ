from __future__ import annotations
import pandas as pd
import numpy as np

def imbalance(bid_sum_k: pd.Series, ask_sum_k: pd.Series) -> pd.Series:
    denom = (bid_sum_k + ask_sum_k).replace(0, np.nan)
    return (bid_sum_k - ask_sum_k) / denom

def spread_bps_from_mid(ask_px: pd.Series, bid_px: pd.Series) -> pd.Series:
    mid = (ask_px + bid_px) / 2.0
    return ((ask_px - bid_px) / mid) * 1e4
