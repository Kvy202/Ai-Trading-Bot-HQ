import os
import numpy as np
import pandas as pd
from labels import triple_barrier_labels

# 26 raw features — STOPGAP SET: includes 'gap', excludes trend_50/trend_200, so
# the LIVE pipeline matches the deployed b0f89fd scalers (26 market feats +
# symbol_id = 27, requires DL_ADD_SYMBOL_ID=1). The deployed models were trained
# on THIS set; the 27-feature "trend anchors" set drifted the columns and fed the
# models out-of-distribution inputs (see tools/diagnose_features.py).
# WHEN YOU RETRAIN on the improved pipeline: drop 'gap', re-add trend_50/trend_200,
# and set DL_ADD_SYMBOL_ID to match how you train.
# Every value is either a ratio, a return, or a normalised volume figure,
# so it is stationary and scale-invariant regardless of price level.
FEATURE_COLS = [
    # ── candle structure (what this candle looks like) ────────────────────────
    'ret',          # close-to-close % return
    'log_ret',      # log return — more stable for large moves
    'body',         # (close - open) / open  — direction & size of candle body
    'range_pct',    # (high - low) / close   — total candle width
    'upper_wick',   # (high  - max(open,close)) / close  — selling rejection above
    'lower_wick',   # (min(open,close) - low) / close    — buying rejection below
    'close_pos',    # (close - low) / (high - low)  — where did price close in range?
                    #   0 = closed at bottom (bearish), 1 = closed at top (bullish)
    'gap',          # (open - prev_close) / prev_close — STOPGAP: restored to match
                    #   the deployed b0f89fd scaler set. Remove again when retraining.
    # ── price levels — where is price relative to recent highs/lows ──────────
    'dist_high_20', # (20-bar high - close) / close — 0 = AT resistance (breakout or rejection)
    'dist_low_20',  # (close - 20-bar low)  / close — 0 = AT support   (bounce or breakdown)
    'hl_rank_20',   # (close - 20-bar low) / (20-bar high - 20-bar low)
                    #   0 = bottom of 20-bar range, 1 = top — raw unsmoothed position
    # (trend_50/trend_200 removed in the STOPGAP — they were NOT in the deployed
    #  scaler's feature set; re-add them when you retrain on current features.)
    # ── return history & momentum ─────────────────────────────────────────────
    'ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag4', 'ret_lag5',
    'ret_5',        # 5-bar cumulative return  (short-term momentum)
    'ret_20',       # 20-bar cumulative return (medium-term momentum)
    # ── raw volatility ────────────────────────────────────────────────────────
    'roll_std_10',  # 10-bar return std   (recent choppiness)
    'roll_std_30',  # 30-bar return std   (baseline choppiness)
    'range_ratio',  # (high-low) / 20-bar avg (high-low) — is this candle unusually large?
    # ── volume analysis ───────────────────────────────────────────────────────
    'vol_ratio_10', # volume / 10-bar avg volume  — short-term activity spike
    'vol_ratio_30', # volume / 30-bar avg volume  — medium-term activity spike
    'obv_delta',    # signed volume: +vol on up candles, -vol on down, normalised by avg
                    #   tells model whether volume is confirming or opposing the move
    'money_flow',   # close_pos × vol_ratio_10
                    #   high vol + bullish close = strong buying; high vol + bearish = selling
    'vol_surge_5',  # 5-bar rolling volume sum / (30-bar avg × 5)
                    #   > 1 means sustained volume above normal (not just one spike)
]

# Name of the optional multi-symbol id channel. Kept out of FEATURE_COLS because
# it is not a market feature — it is appended by the loader when add_symbol_id=True.
SYMBOL_ID_COL = "symbol_id"


def canonical_feature_columns(add_symbol_id: bool) -> list:
    """Explicit, stable feature-column order shared by training AND serving.

    This is the single source of truth for column ordering. Both
    data.load_prices_and_features (training + live) must order columns by this
    function so the scaler and model always see features in the same positions.

    IMPORTANT: the order is ``sorted(FEATURE_COLS [+ symbol_id])``. That exactly
    reproduces the order the *currently deployed* models were trained on (the old
    code used ``sorted(common_cols)``), so existing artifacts stay valid — the
    point of this helper is to make the order EXPLICIT and independent of any
    runtime set-intersection, not to change it. Do not "tidy" this into
    FEATURE_COLS order without retraining every model + scaler.
    """
    cols = list(FEATURE_COLS)
    if add_symbol_id:
        cols = cols + [SYMBOL_ID_COL]
    return sorted(cols)


def make_label(df: pd.DataFrame) -> pd.Series:
    """Simple next-candle label (unused when triple_barrier_labels is active)."""
    fut_ret = df['close'].pct_change().shift(-1)
    return (fut_ret > 0).astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out['close']
    o = out['open']
    h = out['high']
    l = out['low']
    v = out['volume']

    eps = 1e-10  # prevent division by zero without distorting values

    # ── candle structure ──────────────────────────────────────────────────────
    out['ret']       = c.pct_change()
    out['log_ret']   = np.log(c / c.shift(1))
    out['body']      = (c - o) / (o + eps)
    out['range_pct'] = (h - l) / (c + eps)

    candle_top    = np.maximum(o, c)
    candle_bottom = np.minimum(o, c)
    out['upper_wick'] = (h - candle_top)    / (c + eps)
    out['lower_wick'] = (candle_bottom - l) / (c + eps)

    # Split raw vs safe range so a doji (high == low) doesn't NaN-cascade into
    # range_ratio. hl_raw == 0 is a meaningful "zero-width candle"; only the
    # division features need the NaN-guarded denominator.
    hl_raw  = (h - l)
    hl_safe = hl_raw.replace(0, np.nan)
    # Doji-safe: a perfect doji makes (c-l)/hl undefined. In live mode that would
    # drop the latest bar and skip its signal, so fill with 0.5 (price mid-range).
    out['close_pos'] = ((c - l) / hl_safe).fillna(0.5)

    # STOPGAP: 'gap' restored to match the deployed b0f89fd scaler set. Crypto
    # trades 24/7 so this is ~0 most bars, but it occupies a sorted column slot
    # the scaler expects — dropping it shifted every later feature out of position.
    out['gap'] = (o - c.shift(1)) / (c.shift(1) + eps)

    # ── price levels — where is price relative to recent structure ────────────
    high_20 = h.rolling(20).max()
    low_20  = l.rolling(20).min()
    range_20 = (high_20 - low_20).replace(0, np.nan)

    out['dist_high_20'] = (high_20 - c) / (c + eps)   # 0 = AT the high (breakout zone)
    out['dist_low_20']  = (c - low_20)  / (c + eps)   # 0 = AT the low  (support zone)
    # Doji-safe same as close_pos: a flat 20-bar range makes hl_rank NaN.
    out['hl_rank_20']   = ((c - low_20) / range_20).fillna(0.5)   # 0.5 = neutral

    # (trend_50/trend_200 removed in the STOPGAP — not in the deployed scaler set.
    #  Re-add when retraining on the current feature pipeline.)

    # ── return history & momentum ─────────────────────────────────────────────
    for k in range(1, 6):
        out[f'ret_lag{k}'] = out['ret'].shift(k)

    out['ret_5']  = out['ret'].rolling(5).sum()
    out['ret_20'] = out['ret'].rolling(20).sum()

    # ── raw volatility ────────────────────────────────────────────────────────
    out['roll_std_10'] = out['ret'].rolling(10).std()
    out['roll_std_30'] = out['ret'].rolling(30).std()

    avg_range_20      = hl_raw.rolling(20).mean().replace(0, np.nan)
    out['range_ratio'] = (hl_raw / avg_range_20).fillna(0.0)   # 0 = zero-width (doji) candle

    # ── volume analysis ───────────────────────────────────────────────────────
    avg_vol_10 = v.rolling(10).mean().replace(0, np.nan)
    avg_vol_30 = v.rolling(30).mean().replace(0, np.nan)

    out['vol_ratio_10'] = v / avg_vol_10
    out['vol_ratio_30'] = v / avg_vol_30

    # OBV delta: volume signed by direction, normalised so values are comparable
    # across low-vol and high-vol periods.
    # No fillna on avg_vol_30: during the first 30 bars it's NaN, and filling
    # with 1 would divide raw crypto volume (often 1000+) by 1 and produce giant
    # warm-up spikes. Leaving it NaN lets the final dropna() drop those rows.
    direction        = np.sign(out['ret'].fillna(0))
    out['obv_delta'] = (direction * v) / avg_vol_30

    # Money flow: combines WHERE price closed (close_pos) with HOW MUCH volume
    # came through relative to the recent norm.
    #   ~1.0  → closed near high on normal volume  (mild buying)
    #   ~2.0+ → closed near high on a volume spike (strong buying)
    #   ~0    → closed near low  (selling pressure regardless of volume)
    # close_pos is already doji-safe (filled 0.5); vol_ratio_10 is NaN during the
    # first 10-bar warm-up and those rows are dropped by dropna() — no fillna spike.
    out['money_flow'] = out['close_pos'] * out['vol_ratio_10']

    # Volume surge over 5 bars: smooths out single-candle spikes.
    # > 1 means the last 5 candles collectively had above-average volume.
    out['vol_surge_5'] = v.rolling(5).sum() / (avg_vol_30 * 5).replace(0, np.nan)

    out = out.dropna().copy()
    return out


def make_dataset(df: pd.DataFrame):
    feats = build_features(df)

    tf = os.getenv("TIMEFRAME", "1h").lower()
    if tf in ("1m", "3m", "5m"):
        pt, sl, max_h = 0.003, 0.003, 36
    elif tf in ("15m", "30m"):
        pt, sl, max_h = 0.006, 0.006, 48
    else:
        pt, sl, max_h = 0.01, 0.01, 20

    y = triple_barrier_labels(feats, pt=pt, sl=sl, max_h=max_h).dropna()
    feats = feats.loc[y.index]

    X = feats[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').dropna()
    y = y.loc[X.index]

    if os.getenv("QUIET_LABELS", "0") != "1":
        print("[Label balance] n=", len(y),
              "  positives=", int((y == 1).sum()),
              "  frac=", float((y == 1).mean()))

    return X, y, feats.loc[X.index]
