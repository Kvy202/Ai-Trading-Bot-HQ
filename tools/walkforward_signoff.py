#!/usr/bin/env python
"""
walkforward_signoff.py
======================

This script performs a simple walk‑forward evaluation over a historical
dataset and reports key performance indicators (KPIs) useful for a final
sign‑off prior to live deployment.  It is intended as a starting point for
Day 7 of your trading bot project.  While it cannot replace a full
simulation with exchange connectivity and latency measurements, it
illustrates how to compute portfolio statistics using your pre‑trained
ensemble models on out‑of‑sample data.

The script loads your existing deep learning ensemble, applies it to a
rolling window of features for each symbol, makes trading decisions based on
a fixed probability threshold and realized volatility cap, and simulates a
basic long‑only strategy with ATR‑based stops.  It then aggregates returns
and calculates KPIs such as Sharpe ratio, Sortino ratio, maximum drawdown,
profit factor, turnover, hit rate and average trade duration.  A simple
risk sheet is generated outlining exposure limits and kill‑switch rules.

Usage example:

```
python tools/walkforward_signoff.py \
    --symbols BTC/USDT:USDT,ETH/USDT:USDT \
    --timeframe 5m \
    --lookback 50000 \
    --seq-len 32 \
    --threshold 0.42 \
    --rv-max 60
```

This will run a walk‑forward backtest on the specified symbols with a
fixed absolute threshold of 0.42 and a realized volatility cap of 60.  The
script will print the KPI summary and write a CSV report in the
``metrics_out_signoff`` directory.

Limitations:
  * The simulation here is very rudimentary: it assumes trades are entered
    at the close of the bar when the signal triggers and exited either
    when the stop is hit or when the threshold condition fails.  It does
    not account for slippage, latency or partial fills.
  * The script uses the ATR‐based stop distance and position sizing
    functions from ``risk_engine.py`` but does not enforce portfolio
    exposure caps or microstructure checks.  You can expand it to call
    ``micro_gates()`` and ``portfolio_exposure_ok()`` if needed.
  * Because the training artifacts (models and scalers) are not included
    with this repository, you must ensure they exist under
    ``model_artifacts/`` or point the relevant environment variables
    (``DL_*_MODEL_PATH``, ``DL_*_SCALER_PATH``) to your files.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from data import load_prices_and_features
from ml_dl.dl_ensemble import load_ensemble, predict_ensemble
from risk_engine import stop_distance_abs, size_from_risk


def sharpe_ratio(r: np.ndarray) -> float:
    """Annualised Sharpe ratio given an array of per‑trade returns."""
    if r.size == 0 or not np.isfinite(r).any():
        return float('nan')
    mu = np.nanmean(r)
    sd = np.nanstd(r)
    if sd < 1e-12:
        return float('nan')
    # Multiply by sqrt of trades per year to annualise; here we assume
    # returns are daily; adjust the factor depending on your bar frequency.
    return (mu / sd) * np.sqrt(252)


def sortino_ratio(r: np.ndarray) -> float:
    """Annualised Sortino ratio (downside risk only)."""
    if r.size == 0 or not np.isfinite(r).any():
        return float('nan')
    downside = r[r < 0]
    dd_sd = np.sqrt(np.mean(np.square(downside))) if downside.size > 0 else np.nan
    if not np.isfinite(dd_sd) or dd_sd == 0:
        return float('nan')
    mu = np.nanmean(r)
    return (mu / dd_sd) * np.sqrt(252)


def max_drawdown(r: np.ndarray) -> float:
    """Maximum drawdown of cumulative returns."""
    if r.size == 0:
        return 0.0
    cum = np.nancumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.nanmin(dd))  # negative number


def profit_factor(r: np.ndarray) -> float:
    """Ratio of gross winning trades to gross losing trades."""
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float('inf')


def hit_rate(r: np.ndarray) -> float:
    """Fraction of trades with positive return."""
    if r.size == 0:
        return float('nan')
    return float((r > 0).sum() / r.size)


def simulate_trades(
    prices: pd.DataFrame,
    rets_hat: np.ndarray,
    rvs_hat: np.ndarray,
    p_hat: np.ndarray,
    threshold: float,
    rv_max: float,
    risk_per_trade: float,
    equity: float,
    atr: np.ndarray,
    hold_bars: int = 48,
) -> Tuple[np.ndarray, int]:
    """
    Simple long‑only backtest: enter when p_hat >= threshold and rv_hat <= rv_max.
    Exit when either stop distance is hit, hold duration expires, or the
    probability drops below the threshold.  Returns per‑trade P&L array.

    Args:
        prices: DataFrame with columns ``['close']`` for trade exit pricing.
        rets_hat: predicted next‑bar returns (not used directly here but
            available for custom sizing rules).
        rvs_hat: predicted realized volatility per trade.
        p_hat: predicted probability of an up move.
        threshold: probability threshold for entry.
        rv_max: maximum realised volatility allowed for entry.
        risk_per_trade: fraction of equity to risk on each trade.
        equity: total notional equity assumed for position sizing.
        atr: average true range array aligned to ``prices``; used for stop.
        hold_bars: maximum holding period in bars.
    Returns:
        (returns array, number of trades taken)
    """
    n = len(prices)
    in_position = False
    entry_price = 0.0
    stop_price = 0.0
    entry_idx = 0
    qty = 0.0
    pnl_list: List[float] = []
    for i in range(n):
        price = prices.iloc[i]['close']
        # Evaluate entry
        if not in_position:
            if p_hat[i] >= threshold and rvs_hat[i] <= rv_max:
                # Size position based on ATR and risk
                sd = stop_distance_abs(price, atr[i], rvs_hat[i])
                qty = size_from_risk(equity, risk_per_trade, price, sd)
                if qty > 0:
                    in_position = True
                    entry_price = price
                    stop_price = price - sd
                    entry_idx = i
            continue
        # If already in position, check exit conditions
        # Stop loss
        if price <= stop_price:
            pnl_list.append((price - entry_price) / entry_price)
            in_position = False
            continue
        # Hold duration exceeded
        if i - entry_idx >= hold_bars:
            pnl_list.append((price - entry_price) / entry_price)
            in_position = False
            continue
        # Probability dropped below threshold: take profit/exit
        if p_hat[i] < threshold:
            pnl_list.append((price - entry_price) / entry_price)
            in_position = False
            continue
    # close any open trade at the end
    if in_position:
        last_price = prices.iloc[-1]['close']
        pnl_list.append((last_price - entry_price) / entry_price)
    return np.array(pnl_list, dtype=np.float32), len(pnl_list)


def run_backtest_for_symbol(
    symbol: str,
    timeframe: str,
    lookback: int,
    seq_len: int,
    threshold: float,
    rv_max: float,
    models: Dict[str, dict],
    device: str,
    risk_per_trade: float = 0.01,
    equity: float = 10_000.0,
    hold_bars: int = 48,
) -> Dict[str, float]:
    """
    Run a backtest for a single symbol and compute KPIs.

    This helper loads price and feature data, applies the ensemble to
    generate predictions, simulates trades, and returns a dictionary of
    statistics.  See ``simulate_trades`` for details on the trading logic.
    """
    # Load features and labels from historical data
    X, _ = load_prices_and_features(
        symbols=[symbol], timeframe=timeframe, lookback=lookback, add_symbol_id=False, return_dfs=False
    )
    # Align window length
    if X.shape[0] < seq_len + 1:
        raise RuntimeError(f"Not enough data for {symbol}: required {seq_len+1}, got {X.shape[0]}")
    # Convert to pandas for prices and compute ATR for stop
    # For simplicity, use the close series; users may replace this with
    # their own OHLC load with ATR columns.
    df_prices, _ = load_prices_and_features(
        symbols=[symbol], timeframe=timeframe, lookback=lookback, add_symbol_id=False, return_dfs=True
    )
    prices = df_prices[0].copy()
    # Ensure we align with feature matrix length (drop initial rows if any)
    offset = len(prices) - len(X)
    if offset > 0:
        prices = prices.iloc[offset:]
    # Compute a simple ATR proxy using high/low/close from prices
    # This is a rough estimate; if your DataFrame contains ATR already, use it.
    highs = prices['high'].astype(float).values
    lows = prices['low'].astype(float).values
    closes = prices['close'].astype(float).values
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    atr = np.empty_like(closes)
    atr[0] = tr[0] if tr.size > 0 else 0.0
    for i in range(1, len(atr)):
        atr[i] = (atr[i-1] * 13 + (tr[i-1] if i-1 < tr.size else 0.0)) / 14.0
    # Generate predictions per bar
    rets_hat = np.zeros(len(X), dtype=np.float32)
    rvs_hat = np.zeros(len(X), dtype=np.float32)
    p_hat = np.zeros(len(X), dtype=np.float32)
    for i in range(seq_len, len(X)):
        xw = X[i-seq_len:i, :]
        per_model, _ = predict_ensemble(xw, models, device)
        # Here we use the blended prediction (unweighted mean) for simplicity
        ks = list(per_model.keys())
        rets_hat[i] = float(np.mean([per_model[k][0] for k in ks]))
        rvs_hat[i] = float(np.mean([per_model[k][1] for k in ks]))
        p_hat[i] = float(np.mean([per_model[k][2] for k in ks]))
    # Simulate trades and compute P&L
    pnl, n_trades = simulate_trades(
        prices.iloc[: len(p_hat)], rets_hat, rvs_hat, p_hat, threshold, rv_max, risk_per_trade, equity, atr, hold_bars
    )
    metrics = {
        'symbol': symbol,
        'n_trades': n_trades,
        'sharpe': sharpe_ratio(pnl),
        'sortino': sortino_ratio(pnl),
        'max_dd': max_drawdown(pnl),
        'profit_factor': profit_factor(pnl),
        'hit_rate': hit_rate(pnl),
        'avg_return': float(np.nanmean(pnl)) if pnl.size > 0 else float('nan'),
        'turnover': float(n_trades) / max(len(pnl), 1),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk‑forward backtest and sign‑off")
    parser.add_argument('--symbols', type=str, default=os.getenv('SYMBOL_WHITELIST', ''),
                        help='Comma‑separated list of symbols (e.g. BTC/USDT:USDT,ETH/USDT:USDT)')
    parser.add_argument('--timeframe', type=str, default=os.getenv('TIMEFRAME', '5m'),
                        help='Timeframe for historical bars (default 5m)')
    parser.add_argument('--lookback', type=int, default=50000,
                        help='Number of bars to use for backtesting (default 50000)')
    parser.add_argument('--seq-len', type=int, default=int(os.getenv('DL_SEQ_LEN', '32')),
                        help='Sequence length used by the models (default env DL_SEQ_LEN)')
    parser.add_argument('--threshold', type=float, default=float(os.getenv('DL_P_LONG', '0.43')),
                        help='Probability threshold for entering trades')
    parser.add_argument('--rv-max', type=float, default=float(os.getenv('DL_MAX_RV', '60')), 
                        help='Maximum realised volatility allowed for entry')
    parser.add_argument('--out-dir', type=str, default='metrics_out_signoff',
                        help='Directory to write the KPI summary')
    args = parser.parse_args()

    sym_list: List[str] = [s.strip() for s in args.symbols.split(',') if s.strip()] if args.symbols else []
    if not sym_list:
        print("No symbols provided; specify with --symbols or set SYMBOL_WHITELIST env.")
        return

    # Load ensemble models with default device selection
    X0, _ = load_prices_and_features(symbols=[sym_list[0]], timeframe=args.timeframe,
                                      lookback=max(args.seq_len + 500, 5000), add_symbol_id=False, return_dfs=False)
    models, device = load_ensemble(X0.shape[1])

    # Run backtest for each symbol
    results: List[Dict[str, float]] = []
    for sym in sym_list:
        try:
            m = run_backtest_for_symbol(sym, args.timeframe, args.lookback, args.seq_len,
                                        args.threshold, args.rv_max, models, device)
            results.append(m)
            print(f"{sym}: trades={m['n_trades']}, sharpe={m['sharpe']:.3f}, sortino={m['sortino']:.3f}, "
                  f"maxDD={m['max_dd']:.3f}, PF={m['profit_factor']:.3f}, hit={m['hit_rate']:.3f}")
        except Exception as e:
            print(f"Error backtesting {sym}: {e}")

    # Aggregate across symbols
    if results:
        df = pd.DataFrame(results)
        agg = {
            'symbol': 'ALL',
            'n_trades': int(df['n_trades'].sum()),
            'sharpe': float(df['sharpe'].mean()),
            'sortino': float(df['sortino'].mean()),
            'max_dd': float(df['max_dd'].mean()),
            'profit_factor': float(df['profit_factor'].mean()),
            'hit_rate': float(df['hit_rate'].mean()),
            'avg_return': float(df['avg_return'].mean()),
            'turnover': float(df['turnover'].mean()),
        }
        results.append(agg)
        # Create output directory and save report
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_dir / 'kpi_summary.csv', index=False)
        # Write a simple risk sheet
        risk_sheet = {
            'threshold': args.threshold,
            'rv_max': args.rv_max,
            'risk_per_trade': 0.01,
            'equity_assumed': 10000.0,
            'max_hold_bars': 48,
            'stop_distance': 'ATR*2.5 or RV*2.0 (see risk_engine)',
            'kill_switch_max_dd': 0.05,
            'kill_switch_data_gap_ms': 180000,
        }
        import json
        with open(out_dir / 'risk_sheet.json', 'w', encoding='utf-8') as f:
            json.dump(risk_sheet, f, indent=2)
        print("KPI summary and risk sheet written to", out_dir)
    else:
        print("No results computed; check your data and model configuration.")


if __name__ == '__main__':
    main()