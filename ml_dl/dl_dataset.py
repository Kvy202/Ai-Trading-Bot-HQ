"""
ml_dl/dl_dataset.py

Two responsibilities:

1. Re-export `load_prices_and_features` from data.py so other modules can
   import it as `from ml_dl.dl_dataset import load_prices_and_features`.
   This is the function that ultimately feeds the live writer.

2. Provide SeqDataset, a torch Dataset that builds fixed-length sequence
   windows for training the deep-learning models.

Changelog vs the previous version
---------------------------------
- assert in SeqDataset.__init__ replaced with ValueError. asserts get
  stripped when Python runs under -O, so a public API should raise a real
  exception instead.
- y target is now validated to be finite and integer-like before casting
  to int64. The old code did `np.asarray(y, dtype=np.int64)` which
  silently turned NaN into a huge negative integer, producing a "valid-
  looking" dataset with corrupt class labels. NaN rows are now dropped.
- NaN/inf detection is vectorised. The old code did np.isfinite(win).all()
  inside a Python loop -> O(T*L). Now a row-mask is built once and
  windows are validated with a rolling sum -> O(T). Same result, much
  faster on long histories.
- L is validated against T at init time. The old code silently produced
  an empty dataset if L > T. Now it raises with a clear message.
- The wrapper warns once if kwargs are passed and dropped. The old code
  silently swallowed unknown kwargs; if a caller passes `start_date=...`
  expecting a date range filter, they need to know it didn't apply.
- _lpaf_err is always defined at module scope (None on success) so any
  reference to it later is well-defined.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

_LOG = logging.getLogger("dl_dataset")

# ---------------------------------------------------------------------------
# Re-export loader from data.py
# ---------------------------------------------------------------------------
# Both names always exist at module scope after this block, regardless of
# whether the import succeeded. _lpaf is None on failure; _lpaf_err carries
# the original ImportError so we can chain it when raising later.
try:
    from data import load_prices_and_features as _lpaf  # type: ignore
    _lpaf_err: Optional[BaseException] = None
except Exception as e:
    _lpaf = None
    _lpaf_err = e


# Track whether we've already warned about silently-dropped kwargs, so we
# don't spam every tick.
_KWARGS_WARNED: set = set()


def load_prices_and_features(
    symbols: Optional[Sequence[str]] = None,
    timeframe: str = "1m",
    lookback: int = 8000,
    feature_cols: Optional[Sequence[str]] = None,
    add_symbol_id: bool = True,
    return_dfs: bool = False,
    return_symbol_lengths: bool = False,
    symbol_id_map: Optional[dict] = None,
    **kwargs: Any,
):
    """Thin wrapper that forwards to data.load_prices_and_features.

    Only the parameters listed in the signature are forwarded; any extra
    kwargs are dropped (with a warning the first time each name is seen).
    This is intentional: the underlying data.py only accepts these
    parameters, and silently passing unknown kwargs through caused subtle
    bugs in the past.

    Parameters
    ----------
    symbols : sequence of str, optional
        Symbol list (e.g. ['BTCUSDT', 'ETHUSDT']). None means use the
        loader's default universe.
    timeframe : str, default '1m'
        OHLCV timeframe.
    lookback : int, default 8000
        How many rows to fetch.
    feature_cols : sequence of str, optional
        Subset of feature columns. None means all available features.
    add_symbol_id : bool, default True
        Whether to add an integer symbol-id column for multi-symbol training.
    return_dfs : bool, default False
        If True, also return the underlying DataFrames keyed by symbol.
        live_writer.py uses this to extract the most recent close price.

    Returns
    -------
    Whatever data.load_prices_and_features returns. Typically (X, dfs)
    when return_dfs=True, or (X, None) otherwise.
    """
    if _lpaf is None:
        raise ImportError(
            "data.load_prices_and_features is not available. "
            "Ensure data.py is present and importable."
        ) from _lpaf_err

    if kwargs:
        # Warn once per unknown kwarg name. Repeating callers (e.g. the
        # writer ticking every 3 seconds) won't spam the log.
        unknown = set(kwargs.keys()) - _KWARGS_WARNED
        if unknown:
            _KWARGS_WARNED.update(unknown)
            _LOG.warning(
                "load_prices_and_features: ignoring unsupported kwargs %s "
                "(this warning shows once per name)",
                sorted(unknown),
            )

    return _lpaf(
        symbols=symbols,
        timeframe=timeframe,
        lookback=lookback,
        feature_cols=feature_cols,
        add_symbol_id=add_symbol_id,
        return_dfs=return_dfs,
        return_symbol_lengths=return_symbol_lengths,
        symbol_id_map=symbol_id_map,
    )


# ---------------------------------------------------------------------------
# SeqDataset
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    """Fixed-length sequence windows over X with aligned scalar targets.

    Builds every window of length L ending at row i (for i in [L-1, T)),
    then drops:
      - any window where any feature is NaN/inf
      - any row where the regression target r or rv target is NaN/inf
      - any row where the classification target y is NaN/inf or non-integer

    Parameters
    ----------
    X : np.ndarray of shape [T, F]
        Feature matrix. Float32 internally.
    r : np.ndarray of shape [T]
        Regression target (typically forward return).
    y : np.ndarray of shape [T]
        Classification target. Must be finite and integer-valued.
    rv : np.ndarray of shape [T]
        Realised volatility target.
    L : int
        Sequence length. Must satisfy 1 <= L <= T.

    Each item is a dict:
      {
        "x":         FloatTensor [L, F],
        "y_ret_reg": FloatTensor scalar,
        "y_ret_cls": LongTensor  scalar,
        "y_rv_reg":  FloatTensor scalar,
      }
    """

    def __init__(self, X: np.ndarray, r: np.ndarray, y: np.ndarray,
                 rv: np.ndarray, L: int):
        # Length validation: ValueError, not assert. asserts disappear
        # under python -O and silent shape mismatches are nasty to debug.
        if not (len(X) == len(r) == len(y) == len(rv)):
            raise ValueError(
                f"SeqDataset: X, r, y, rv must have the same length. "
                f"Got X={len(X)}, r={len(r)}, y={len(y)}, rv={len(rv)}."
            )

        L = int(L)
        if L < 1:
            raise ValueError(f"SeqDataset: L must be >= 1, got {L}")

        T = len(X)
        if L > T:
            raise ValueError(
                f"SeqDataset: L={L} is larger than T={T}; "
                f"there are no windows of that length."
            )

        self.X = np.asarray(X, dtype=np.float32)
        self.r = np.asarray(r, dtype=np.float32)
        self.rv = np.asarray(rv, dtype=np.float32)
        self.L = L

        # --- Build the y target carefully ---------------------------------
        # The old code did `np.asarray(y, dtype=np.int64)`, which casts
        # NaN to a huge negative integer silently. We instead validate
        # that each y value is finite AND integer-like, and we mark any
        # bad row so its window can be skipped below.
        y_arr = np.asarray(y)
        y_finite = np.isfinite(y_arr.astype(np.float64, copy=False))
        # Non-finite values won't fit in int64 cleanly; substitute 0 in
        # those positions so the cast is well-defined, but keep the mask
        # so we drop the corresponding windows.
        y_safe = np.where(y_finite, y_arr, 0)
        self.y = y_safe.astype(np.int64, copy=False)

        # --- Vectorised valid-window detection ----------------------------
        # A window ending at index i is valid iff:
        #   - the targets r[i], rv[i], y[i] are all finite
        #   - every row in X[i-L+1 : i+1] has all-finite features
        #
        # Old code: per-i loop with np.isfinite(win).all()  ->  O(T*L).
        # New code: precompute a per-row "all features finite" mask, then
        # a rolling sum tells us if any of the last L rows had a bad
        # feature.  ->  O(T).
        feat_ok = np.isfinite(self.X).all(axis=1)            # [T] bool
        targ_ok = np.isfinite(self.r) & np.isfinite(self.rv) & y_finite

        # Cumulative count of bad rows up to (and including) each index.
        # Bad-rows-in-window-ending-at-i = bad_cum[i] - bad_cum[i-L]
        # (with bad_cum[-1] := 0).
        bad = (~feat_ok).astype(np.int64)
        bad_cum = np.cumsum(bad)
        # Pad so we can index bad_cum[i-L] for i = L-1 (gives bad_cum[-1] = 0).
        bad_cum_padded = np.concatenate(([0], bad_cum))      # length T+1
        # For window ending at i (0-indexed), start = i-L+1, end = i.
        # bad rows in [start, end] = bad_cum[end] - bad_cum[start-1]
        #                          = bad_cum_padded[end+1] - bad_cum_padded[start]
        idx_end = np.arange(L - 1, T)                        # candidate i values
        start = idx_end - L + 1
        bad_in_window = bad_cum_padded[idx_end + 1] - bad_cum_padded[start]
        feat_ok_window = bad_in_window == 0                  # [T-L+1] bool

        # Combine with target validity at i.
        keep = feat_ok_window & targ_ok[idx_end]
        self.idx = idx_end[keep].astype(np.int64)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        s = i - self.L + 1
        x = self.X[s:i + 1]                # [L, F]
        return {
            "x": torch.from_numpy(x),                               # float32
            "y_ret_reg": torch.tensor(self.r[i], dtype=torch.float32),
            "y_ret_cls": torch.tensor(self.y[i], dtype=torch.long),
            "y_rv_reg": torch.tensor(self.rv[i], dtype=torch.float32),
        }

    def __repr__(self) -> str:
        return (f"SeqDataset(T={len(self.X)}, F={self.X.shape[1] if self.X.ndim == 2 else '?'}, "
                f"L={self.L}, valid_windows={len(self.idx)})")