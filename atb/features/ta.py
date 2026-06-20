from __future__ import annotations
import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def realized_vol(returns: pd.Series, window: int = 30) -> pd.Series:
    # annualization not applied here; window of bars
    return (returns.pow(2).rolling(window).sum()).pow(0.5)

def volume_zscore(volume: pd.Series, window: int = 60) -> pd.Series:
    m = volume.rolling(window).mean()
    s = volume.rolling(window).std(ddof=0)
    return (volume - m) / s.replace(0, np.nan)
