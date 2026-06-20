# backtest_day3.py — Day 3 risk backtester:
# fee/spread/slip/latency + ATR/RV TP/SL + position sizing + caps + spread-widen pause
# slippage anomaly + data-gap guard + time-stop. Optional real L2 CSV. Exports stats_day3.csv.
from __future__ import annotations
import os, argparse, warnings
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from atb.data.sources.klines_ccxt import fetch_klines
from atb.features.pipeline import build_leakage_safe_features

# ---------------- ENV / CONFIG ----------------
SEED = int(os.getenv("SEED", "42"))
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

# Trade frictions
FEE_BPS         = float(os.getenv("BT_FEE_BPS", "6"))          # taker fee (per side)
IMPACT_COEFF    = float(os.getenv("BT_IMPACT_COEFF", "3"))     # slippage bps per |vol_z|
LATENCY_BARS    = max(0, int(os.getenv("BT_LATENCY_BARS", "1")))

# Equity & sizing
INIT_EQUITY     = float(os.getenv("BT_INIT_EQUITY", "1000"))
RISK_PER_TRADE  = float(os.getenv("BT_RISK_PER_TRADE", "0.003"))  # fraction of equity to risk to stop
LEVERAGE        = float(os.getenv("LEVERAGE", "1"))               # soft cap helper (not PnL leverage)
MAX_MARGIN_FRAC = float(os.getenv("BT_MAX_MARGIN_FRACTION", "1.0"))  # 0..1 of equity usable as margin
MAX_NOTIONAL    = float(os.getenv("BT_MAX_NOTIONAL_USDT", "0"))      # per-trade cap ($); 0=ignore
# (Portfolio cap is moot in single-symbol runs, but kept for parity)
MAX_PORTFOLIO_EXPO = float(os.getenv("BT_MAX_PORTFOLIO_EXPOSURE_USDT", "0"))  # 0=ignore

# Stops / targets (reuse live knobs)
STOP_ATR_MULT   = float(os.getenv("STOP_ATR_MULT", "2.5"))
TP_R_MULT       = float(os.getenv("TP_R_MULT", "3.0"))
RV_STOP_MULT    = float(os.getenv("RV_STOP_MULT", "2.0"))      # RV * price multiplier
RISK_MIN_STOP_FRAC = float(os.getenv("RISK_MIN_STOP_FRAC", "0.002"))  # floor on stop distance as % of price
TIME_STOP_BARS  = int(os.getenv("BT_TIME_STOP_BARS", "0"))     # 0 = disabled

# Baseline rule (5m EMA cross + RSI band)
EMA_FAST = 20
EMA_SLOW = 50
RSI_MIN  = float(os.getenv("BT_RSI_MIN", "30"))
RSI_MAX  = float(os.getenv("BT_RSI_MAX", "70"))

# Optional entry gates (live-like)
USE_GATES        = bool(int(os.getenv("BT_USE_GATES", "1")))
SPREAD_GUARD_BPS = float(os.getenv("SPREAD_GUARD_BPS", "10"))
VOLZ_MIN         = float(os.getenv("BT_VOLZ_MIN", "-0.5"))     # skip entries if vol_z < VOLZ_MIN

# Day-3 microstructure guards
SPREAD_PAUSE_BPS   = float(os.getenv("BT_SPREAD_PAUSE_BPS", "15"))   # absolute guard
SPREAD_WIDEN_MULT  = float(os.getenv("BT_SPREAD_WIDEN_MULT", "2.5")) # × rolling median
SPREAD_LOOKBACK    = int(os.getenv("BT_SPREAD_LOOKBACK", "20"))
SLIP_MAX_BPS       = float(os.getenv("BT_SLIPPAGE_MAX_BPS", "60"))   # eff_bps=halfspread+impact*|vz|
GAP_MAX_MINUTES    = int(os.getenv("BT_GAP_MAX_MINUTES", "15"))
DAILY_MAX_DD       = float(os.getenv("BT_DAILY_MAX_DD", "0"))        # 0=off, use e.g. 0.1 for 10%

# ---------------- small helpers ----------------
def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _last_in_5m(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols: return pd.DataFrame()
    return df.set_index("dt")[cols].resample("5min").last()

def _sim_fill_price(px: float, spread_bps: float, vol_z: float, side_mkt: str) -> float:
    # pay half-spread + slippage
    half_sp = (spread_bps / 2.0) / 1e4
    slip    = (IMPACT_COEFF * abs(vol_z)) / 1e4
    adj = half_sp + slip
    return px * (1 + adj if side_mkt == "buy" else 1 - adj)

def _eff_bps(spread_bps: float, vol_z: float) -> float:
    return (spread_bps / 2.0) + (IMPACT_COEFF * abs(vol_z))

def _stop_abs(entry_px: float, atr: float, rv: float) -> float:
    atr_risk = STOP_ATR_MULT * max(atr, 0.0)
    rv_risk  = RV_STOP_MULT * max(rv, 0.0) * max(entry_px, 1e-12)
    frac_floor = max(entry_px, 1e-12) * RISK_MIN_STOP_FRAC
    return max(atr_risk, rv_risk, frac_floor)

def _qty_from_risk(equity: float, entry_px: float, stop_abs: float) -> float:
    if stop_abs <= 0: return 0.0
    # risk dollars = qty * stop_abs  -> qty = (equity * RISK_PER_TRADE) / stop_abs
    return (equity * RISK_PER_TRADE) / stop_abs

def _cap_qty(entry_px: float, qty: float, equity: float) -> float:
    if entry_px <= 0 or qty <= 0:
        return 0.0
    notional = qty * entry_px

    # 1) per-trade hard cap
    if MAX_NOTIONAL > 0:
        notional = min(notional, MAX_NOTIONAL)

    # 2) margin-style cap (equity * leverage * margin_frac)
    max_notional_margin = equity * max(LEVERAGE, 1.0) * max(min(MAX_MARGIN_FRAC, 1.0), 0.0)
    if max_notional_margin > 0:
        notional = min(notional, max_notional_margin)

    # 3) (optional) portfolio exposure cap — single-symbol run, so same as per-trade
    if MAX_PORTFOLIO_EXPO > 0:
        notional = min(notional, MAX_PORTFOLIO_EXPO)

    return max(notional / entry_px, 0.0)

def _pnl_usd(entry_px: float, exit_px: float, side: str, qty: float) -> float:
    gross = (exit_px - entry_px) * qty if side == "long" \
            else (entry_px - exit_px) * qty
    # fees on notional both sides
    fee_rate = FEE_BPS / 1e4
    fees = fee_rate * (entry_px * qty) + fee_rate * (exit_px * qty)
    return gross - fees

# ---------------- 5m signals from 1m features ----------------
def baseline_signals_5m(feat1m: pd.DataFrame) -> pd.DataFrame:
    if feat1m is None or feat1m.empty:
        return pd.DataFrame()
    df = feat1m.copy()
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    base = df.set_index("dt")[["close"]].resample("5min").last().dropna()
    base["ema_fast"] = _ema(base["close"], EMA_FAST)
    base["ema_slow"] = _ema(base["close"], EMA_SLOW)

    carry = _last_in_5m(df, ["rsi","vol_z","spread_bps","atr_14","rv"])
    if not carry.empty:
        base = base.join(carry, how="left")

    need = [c for c in ["ema_fast","ema_slow","rsi"] if c in base.columns]
    if not need or base[need].dropna().empty:
        return pd.DataFrame()
    base = base.dropna(subset=need)

    base["long_sig"]  = ((base["ema_fast"] > base["ema_slow"]) & (base["rsi"].between(RSI_MIN, RSI_MAX))).astype(int)
    base["short_sig"] = ((base["ema_fast"] < base["ema_slow"]) & (base["rsi"].between(RSI_MIN, RSI_MAX))).astype(int)
    base["ts_dt"]     = base.index

    # microstructure “pause on widening spread”
    base["spread_med"] = base["spread_bps"].rolling(SPREAD_LOOKBACK, min_periods=5).median()
    base["pause_widen"] = (base["spread_bps"] > np.maximum(SPREAD_PAUSE_BPS, SPREAD_WIDEN_MULT * base["spread_med"]))

    # data gap flag (vs. prior 5m bar)
    base["gap_min"] = base["ts_dt"].diff().dt.total_seconds().div(60.0)
    base["gap_flag"] = base["gap_min"].fillna(0) > GAP_MAX_MINUTES

    # symbol label
    sym = "UNKNOWN"
    if "symbol" in feat1m and not feat1m["symbol"].dropna().empty:
        sym = str(feat1m["symbol"].dropna().iloc[-1])
    base["symbol"] = sym

    return base.reset_index(drop=True)

# ---------------- Simulator (TP/SL + trailing + Day3 risk) ----------------
def run_sim_day3(feat1m: pd.DataFrame, equity0: float = INIT_EQUITY) -> Dict[str, Any]:
    five = baseline_signals_5m(feat1m)
    if five is None or five.empty:
        return {
            "curve": pd.DataFrame({"ts_dt": [], "equity": []}),
            "trades": pd.DataFrame(),
            "stats": {"cum_return": 0.0, "max_drawdown": 0.0, "n_trades": 0, "win_rate": 0.0, "final_equity": equity0}
        }

    eq = equity0
    curve: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    pos = {"open": False, "side": None, "entry": None, "stop": None, "take": None, "qty": 0.0, "i_open": -1}

    # per-bar pending fills: (i_fill, action, side, meta)
    pending: List[Tuple[int,str,str,Dict[str,Any]]] = []

    # Daily DD pause
    paused_today = False
    day_key = pd.Timestamp(five["ts_dt"].iloc[0]).tz_convert(None).date()
    day_start_eq = eq

    for i in range(len(five)):
        row = five.iloc[i]
        ts  = pd.Timestamp(row["ts_dt"]).tz_convert(None)
        px  = _f(row["close"])
        sp  = _f(row.get("spread_bps"), 10.0)
        vz  = _f(row.get("vol_z"), 0.0)

        # new day reset DD pause
        if ts.date() != day_key:
            day_key = ts.date()
            day_start_eq = eq
            paused_today = False

        # 0) daily DD breaker
        if DAILY_MAX_DD > 0:
            dd = (eq - day_start_eq) / max(day_start_eq, 1e-12)
            if dd <= -DAILY_MAX_DD:
                paused_today = True

        # 1) process due fills
        done = []
        for k,(t_fill, action, side, meta) in enumerate(pending):
            if i >= t_fill:
                if action == "open":
                    # slippage & fill price at entry
                    fill_px = _sim_fill_price(px, sp, vz, "buy" if side=="long" else "sell")

                    atr = _f(row.get("atr_14"), 0.0)
                    rv  = _f(row.get("rv"), 0.0)
                    risk_abs = _stop_abs(fill_px, atr, rv)

                    # position sizing -> qty (coins), with caps
                    qty_raw = _qty_from_risk(eq, fill_px, risk_abs)
                    qty     = _cap_qty(fill_px, qty_raw, eq)

                    if qty <= 0:
                        # abort open if caps/risk result in 0 size
                        done.append(k)
                        continue

                    # initialize TP/SL
                    if side == "long":
                        stop = fill_px - risk_abs
                        take = fill_px + TP_R_MULT*risk_abs
                    else:
                        stop = fill_px + risk_abs
                        take = fill_px - TP_R_MULT*risk_abs

                    pos = {"open": True, "side": side, "entry": fill_px, "stop": stop, "take": take, "qty": qty, "i_open": i}
                    trades.append({
                        "ts_open": ts, "px_open": fill_px, "side": side, "qty": qty,
                        "notional_usd": qty * fill_px, "reason_open": meta.get("reason","signal")
                    })
                else:
                    # close
                    side_open = pos["side"]
                    mkt_side  = "sell" if side_open=="long" else "buy"
                    fill_px   = _sim_fill_price(px, sp, vz, mkt_side)

                    pnl_usd = _pnl_usd(pos["entry"], fill_px, side_open, pos["qty"])
                    eq += pnl_usd

                    trades[-1].update({
                        "ts_close": ts, "px_close": fill_px, "pnl_usd": pnl_usd,
                        "pnl_frac": pnl_usd / max(trades[-1]["notional_usd"], 1e-12),
                        "reason_close": meta.get("reason","exit")
                    })
                    pos = {"open": False, "side": None, "entry": None, "stop": None, "take": None, "qty": 0.0, "i_open": -1}
                done.append(k)
        for k in reversed(done): pending.pop(k)

        # 2) manage open position (trailing + time stop + signal flip)
        if pos["open"]:
            atr = _f(row.get("atr_14"), 0.0)
            rv  = _f(row.get("rv"), 0.0)
            trail = max(_stop_abs(px, atr, rv), 0.0)  # dynamic trail

            if pos["side"]=="long":
                pos["stop"] = max(pos["stop"], px - trail)
                hit = (px <= pos["stop"]) or (px >= pos["take"])
            else:
                pos["stop"] = min(pos["stop"], px + trail)
                hit = (px >= pos["stop"]) or (px <= pos["take"])

            # time stop
            if TIME_STOP_BARS > 0 and (i - pos["i_open"]) >= TIME_STOP_BARS:
                pending.append((i + LATENCY_BARS, "close", pos["side"], {"reason":"time"}))
            elif hit:
                pending.append((i + LATENCY_BARS, "close", pos["side"],
                                {"reason":"tp" if ((pos['side']=='long' and px>=pos['take']) or (pos['side']=='short' and px<=pos['take'])) else "sl"}))
            else:
                # flip-to-exit only if not already queued
                want_long  = int(row.get("long_sig",0)) == 1
                want_short = int(row.get("short_sig",0)) == 1
                if (pos["side"]=="long" and want_short) or (pos["side"]=="short" and want_long):
                    pending.append((i + LATENCY_BARS, "close", pos["side"], {"reason":"flip"}))

        # 3) entries (with Day-3 guards)
        if (not pos["open"]) and (not paused_today):
            want_long  = int(row.get("long_sig",0)) == 1
            want_short = int(row.get("short_sig",0)) == 1
            if want_long or want_short:
                # basic live-like gates
                if USE_GATES:
                    if row.get("spread_bps", np.nan) == row.get("spread_bps", np.nan):
                        if row["spread_bps"] > SPREAD_GUARD_BPS:
                            curve.append({"ts_dt": ts, "equity": eq}); 
                            continue
                    if row.get("vol_z", np.nan) == row.get("vol_z", np.nan):
                        if row["vol_z"] < VOLZ_MIN:
                            curve.append({"ts_dt": ts, "equity": eq}); 
                            continue

                # Day-3: spread-widen pause & data gap
                if bool(row.get("pause_widen", False)) or bool(row.get("gap_flag", False)):
                    curve.append({"ts_dt": ts, "equity": eq}); 
                    continue

                # Day-3: slippage anomaly
                eff = _eff_bps(_f(row.get("spread_bps"), 10.0), _f(row.get("vol_z"), 0.0))
                if eff > SLIP_MAX_BPS:
                    curve.append({"ts_dt": ts, "equity": eq}); 
                    continue

                side = "long" if want_long else "short"
                pending.append((i + LATENCY_BARS, "open", side, {}))

        curve.append({"ts_dt": ts, "equity": eq})

    # final close
    if pos["open"] and len(five) > 0:
        last = five.iloc[-1]
        ts   = pd.Timestamp(last["ts_dt"]).tz_convert(None)
        px   = _f(last["close"]); sp = _f(last.get("spread_bps"), 10.0); vz = _f(last.get("vol_z"), 0.0)
        mkt_side = "sell" if pos["side"]=="long" else "buy"
        fill_px  = _sim_fill_price(px, sp, vz, mkt_side)
        pnl_usd  = _pnl_usd(pos["entry"], fill_px, pos["side"], pos["qty"])
        eq += pnl_usd
        trades[-1].update({"ts_close": ts, "px_close": fill_px, "pnl_usd": pnl_usd,
                           "pnl_frac": pnl_usd / max(trades[-1]["notional_usd"], 1e-12),
                           "reason_close":"eod"})
        curve.append({"ts_dt": ts, "equity": eq})

    curve_df  = pd.DataFrame(curve).set_index("ts_dt")
    trades_df = pd.DataFrame(trades)

    ret = curve_df["equity"].pct_change().fillna(0.0)
    cumret = float((1 + ret).prod() - 1)
    dd = float((curve_df["equity"] / curve_df["equity"].cummax() - 1).min())
    n = int(len(trades_df))
    win_rate = float((trades_df["pnl_usd"] > 0).mean()) if n else 0.0

    return {
        "curve": curve_df,
        "trades": trades_df,
        "stats": {
            "cum_return": cumret,
            "max_drawdown": dd,
            "n_trades": n,
            "win_rate": win_rate,
            "final_equity": float(curve_df["equity"].iloc[-1]) if not curve_df.empty else equity0,
            "avg_notional": float(trades_df["notional_usd"].mean()) if n else 0.0,
            "median_effbps_cap": SLIP_MAX_BPS,
        }
    }

# ---------------- L2 loading (optional real CSV) ----------------
def _load_l2_csv(path: str, symbol: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    ob = pd.read_csv(path)
    if ob.empty: return pd.DataFrame()
    ob["ts_dt"] = pd.to_datetime(ob["ts"], unit="ms", utc=True)
    one = (ob.set_index("ts_dt")
             .resample("1min")
             .last()
             .dropna()
             .reset_index())
    one["symbol"] = symbol
    cols = ["ts","symbol","bid_px","bid_sz","ask_px","ask_sz","mid_px","spread","spread_bps","bid_sz_sum_k","ask_sz_sum_k"]
    return one[cols] if all(c in one.columns for c in cols) else pd.DataFrame()

# ---------------- symbol resolver ----------------
def _resolve(req_symbols: List[str], kl: pd.DataFrame) -> List[str]:
    actuals = sorted(map(str, kl["symbol"].unique()))
    out: List[str] = []
    for s in req_symbols:
        if s in actuals:
            out.append(s); continue
        base = s.replace(":USDT","")
        found = None
        for a in actuals:
            if a == s or a.replace(":USDT","") == base:
                found = a; break
        if found: out.append(found)
    return out if out else actuals

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default=os.getenv("EXCHANGE_ID","bitget"))
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--limit", type=int, default=3000)      # 1m bars
    ap.add_argument("--outdir", default="reports/tier1_demo")
    ap.add_argument("--l2dir", default="", help="Optional folder with captured L2 CSVs (from record_l2_bitget_rest.py)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    req_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # 1m klines
    kl = fetch_klines(args.exchange, req_symbols, timeframe="1m", limit=args.limit, market_type="swap")
    if kl is None or kl.empty:
        print("No klines returned."); return

    # resolve requested→actual names
    use_symbols = _resolve(req_symbols, kl)

    # Build order book frame
    if args.l2dir:
        ob_chunks = []
        for sym in use_symbols:
            f = os.path.join(args.l2dir, f"{sym.replace('/','_').replace(':','_')}.csv")
            one = _load_l2_csv(f, sym)
            if not one.empty: ob_chunks.append(one)
        if ob_chunks:
            ob = pd.concat(ob_chunks, ignore_index=True)
        else:
            print("[l2] no matching CSVs found; falling back to placeholder OB.")
            args.l2dir = ""  # fall through to placeholder
    if not args.l2dir:
        # placeholder OB (synthetic 10 bps)
        ob = kl[kl["timeframe"]=="1m"][["ts","symbol","close","volume"]].copy()
        ob["bid_px"] = ob["close"]*0.9995; ob["ask_px"] = ob["close"]*1.0005
        ob["bid_sz"] = ob["volume"].clip(lower=1); ob["ask_sz"] = ob["volume"].clip(lower=1)
        ob = ob.rename(columns={"close":"mid_px"})
        ob["spread"] = ob["ask_px"] - ob["bid_px"]
        ob["spread_bps"] = (ob["spread"] / ((ob["ask_px"]+ob["bid_px"])/2))*1e4
        ob["bid_sz_sum_k"] = ob["bid_sz"]; ob["ask_sz_sum_k"] = ob["ask_sz"]

    rows = []
    for sym in use_symbols:
        kl_sym = kl[kl["symbol"]==sym]
        ob_sym = ob[ob["symbol"]==sym]
        if kl_sym.empty or ob_sym.empty:
            print(f"[skip] empty data for {sym}"); continue

        feat = build_leakage_safe_features(kl_sym, ob_sym)
        if feat is None or feat.empty:
            print(f"[skip] no features for {sym}"); continue

        res = run_sim_day3(feat, equity0=INIT_EQUITY)
        print(f"{sym} → {res['stats']}")
        rows.append({"symbol": sym, **res["stats"]})
        # per-symbol trade log
        res["trades"].to_csv(os.path.join(args.outdir, f"{sym.replace('/','_').replace(':','_')}_trades.csv"), index=False)

    if rows:
        outcsv = os.path.join(args.outdir, "stats_day3.csv")
        pd.DataFrame(rows).to_csv(outcsv, index=False)
        print(f"Wrote {outcsv}")

if __name__ == "__main__":
    main()
