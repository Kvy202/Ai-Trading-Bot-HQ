import os, math, joblib, warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import ccxt

from features import make_dataset  # uses triple-barrier and returns (X,y,feats)

warnings.filterwarnings("ignore", category=UserWarning)

# -------- env / defaults --------
TIMEFRAME      = os.getenv("TIMEFRAME", "5m")
LOOKBACK       = int(os.getenv("LOOKBACK_CANDLES", "3000"))
SCAN_TOPN      = int(os.getenv("SCAN_TOPN", "25"))
SCAN_MIN_NOT   = float(os.getenv("SCAN_MIN_NOTIONAL", "100000"))  # USDT 24h vol floor
SEED           = int(os.getenv("SEED", "42"))

MODEL_PATH     = os.getenv("MODEL_PATH", "models/model.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# -------- ccxt helpers (Bitget futures) --------
def bitget_swap():
    ex = ccxt.bitget({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType":"swap","defaultSubType":"linear","defaultSettle":"USDT"},
    })
    ex.load_markets()
    return ex

def scan_bitget_topn(ex: ccxt.Exchange, topn: int, min_notional: float) -> List[str]:
    tickers = ex.fetch_tickers()
    cands: List[Tuple[str, float]] = []
    for sym, m in ex.markets.items():
        if not m.get("contract", False):            continue
        if m.get("settle", "").upper() != "USDT":   continue
        if m.get("linear") is False:                continue
        t = tickers.get(sym, {})
        last  = t.get("last")
        qvol  = t.get("quoteVolume")
        if qvol is None:
            base_vol = t.get("baseVolume")
            if base_vol is not None and last:
                qvol = base_vol * last
            else:
                info = t.get("info", {}) or {}
                qvol = float(info.get("usdtVol") or 0.0)
        if last and qvol and float(qvol) >= min_notional:
            cands.append((sym, float(qvol)))
    cands.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in cands[:topn]]

def fetch_ohlcv_swap(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Paginated OHLCV for swap (USDT-M)."""
    per = min(limit, 1000)
    ms_per_bar = ex.parse_timeframe(timeframe) * 1000
    since = None
    rows = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=per)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + ms_per_bar
        if len(batch) < per:
            break
    if not rows:
        raise RuntimeError(f"No OHLCV for {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().tail(limit)

# -------- dataset assembly --------
def build_multi_symbol_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str,int]]:
    ex = bitget_swap()
    symbols = scan_bitget_topn(ex, SCAN_TOPN, SCAN_MIN_NOT)
    print(f"[Universe] Bitget USDT-M top{len(symbols)} (by 24h USDT vol): {', '.join(symbols)}")

    per_X: Dict[str, pd.DataFrame] = {}
    per_y: Dict[str, pd.Series] = {}
    dropped: Dict[str, str] = {}

    for sym in symbols:
        try:
            df = fetch_ohlcv_swap(ex, sym, TIMEFRAME, LOOKBACK)
            X, y, _ = make_dataset(df)  # uses your features + triple barrier
            # require both classes and minimum size
            if len(X) < 200 or y.nunique() < 2:
                dropped[sym] = f"too_small_or_oneclass(n={len(X)}, uniq={y.nunique()})"
                continue
            per_X[sym] = X.copy()
            per_y[sym] = y.copy()
            print(f"[{sym}] rows={len(X)}  pos={int(y.sum())} ({y.mean():.3f})")
        except Exception as e:
            dropped[sym] = f"error:{e}"

    # common feature columns across all kept symbols
    if not per_X:
        raise RuntimeError("No symbols produced a usable dataset.")
    col_sets = [set(X.columns) for X in per_X.values()]
    common_cols = sorted(list(set.intersection(*col_sets)))
    if len(common_cols) == 0:
        raise RuntimeError("No common feature columns across symbols.")
    print(f"[Dataset] common feature count = {len(common_cols)}")

    # concat into MultiIndex (symbol, timestamp-like order)
    X_all_list, y_all_list = [], []
    for sym in list(per_X.keys()):
        Xi = per_X[sym][common_cols].copy()
        yi = per_y[sym].loc[Xi.index]
        # multiindex index to keep symbol identity
        Xi.index = pd.MultiIndex.from_product([[sym], Xi.index], names=["symbol","ts"])
        yi.index = Xi.index
        X_all_list.append(Xi)
        y_all_list.append(yi)

    X_all = pd.concat(X_all_list).sort_index(level="ts")
    y_all = pd.concat(y_all_list).loc[X_all.index]

    kept = {s: len(per_X[s]) for s in per_X}
    return X_all, y_all, common_cols, {**kept, **{f"DROP:{k}": v for k,v in dropped.items()}}

# -------- modeling --------
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

def cv_scores(X: pd.DataFrame, y: pd.Series, n_splits=5, test_size=100, gap= int(LOOKBACK*0.01)) -> Dict[str,float]:
    """
    Simple rolling CV over the *global time order* (across all symbols).
    """
    n = len(X)
    if n < (n_splits*test_size + gap + 10):
        n_splits = max(2, n // max(test_size, 50))
    print(f"[CV] n={n}  n_splits={n_splits}  test_size={test_size}  gap={gap}")

    idx = np.arange(n)
    auc_rf, auc_hgb = [], []

    # custom rolling splits from the tail
    for k in range(n_splits):
        end = n - k*test_size
        start = end - test_size
        if start - gap <= 100:  # ensure some train region
            break
        tr_idx = idx[:start-gap]
        te_idx = idx[start:end]

        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        rf  = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=3,
                random_state=SEED, n_jobs=-1
              )
        hgb = HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=400,
                random_state=SEED
              )
        rf.fit(Xtr, ytr)
        hgb.fit(Xtr, ytr)
        auc_rf.append(roc_auc_score(yte, rf.predict_proba(Xte)[:,1]))
        auc_hgb.append(roc_auc_score(yte, hgb.predict_proba(Xte)[:,1]))

    auc_rf_m  = float(np.mean(auc_rf))  if auc_rf else float("nan")
    auc_hgb_m = float(np.mean(auc_hgb)) if auc_hgb else float("nan")
    print(f"[CV] AUC_RF={auc_rf_m:.3f}  AUC_HGB={auc_hgb_m:.3f}")
    return {"RF": auc_rf_m, "HGB": auc_hgb_m}

def train():
    X, y, cols, stats = build_multi_symbol_dataset()
    print("[Summary] symbols & rows:", stats)

    # model selection
    scores = cv_scores(X, y, n_splits=5, test_size= max(100, len(X)//20), gap=50)
    # fit on all
    rf  = RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, random_state=SEED, n_jobs=-1
         )
    hgb = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=500, random_state=SEED
         )
    rf.fit(X, y)
    hgb.fit(X, y)

    # choose by full-data AUC (proxy)
    p_rf  = rf.predict_proba(X)[:,1]
    p_hgb = hgb.predict_proba(X)[:,1]
    auc_rf_all  = roc_auc_score(y, p_rf)
    auc_hgb_all = roc_auc_score(y, p_hgb)
    acc_rf_all  = accuracy_score(y, (p_rf  > 0.5).astype(int))
    acc_hgb_all = accuracy_score(y, (p_hgb > 0.5).astype(int))

    if (scores["HGB"] >= scores["RF"]) or (math.isnan(scores["RF"]) and not math.isnan(scores["HGB"])):
        chosen, auc_all, acc_all, name = hgb, auc_hgb_all, acc_hgb_all, "HGB"
    else:
        chosen, auc_all, acc_all, name = rf, auc_rf_all, acc_rf_all, "RF"

    joblib.dump({"model": chosen, "features": cols}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Chosen: {name} | AUC(all)={auc_all:.3f} | Acc(all)={acc_all:.3f}")

if __name__ == "__main__":
    train()
