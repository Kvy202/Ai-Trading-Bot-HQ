from __future__ import annotations
import numpy as np
import pandas as pd

def bps_to_frac(bps: float) -> float:
    return bps / 10000.0

def position_size(equity: float, risk_per_trade: float, entry: float, stop: float) -> float:
    risk_amt = equity * risk_per_trade
    rr = abs(entry - stop)
    if rr <= 0:
        return 0.0
    qty = risk_amt / rr
    return max(qty, 0.0)

def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0)

def sharpe_ratio(returns: pd.Series, periods_per_year=365*24) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(periods_per_year)

def equity_curve(trade_rets: pd.Series, start_equity: float) -> pd.Series:
    return (1.0 + trade_rets).cumprod() * start_equity
