import numpy as np


def next_k_logret(prices: np.ndarray, k: int) -> np.ndarray:
    r = np.full_like(prices, fill_value=np.nan, dtype=float)
    r[:-k] = np.log(prices[k:]) - np.log(prices[:-k])
    return r


def next_k_rv(log_prices: np.ndarray, k: int) -> np.ndarray:
    rv = np.full_like(log_prices, np.nan, dtype=float)
    diffs = np.diff(log_prices)
    sq = diffs ** 2
    csum = np.cumsum(sq)
    rv[:-(k)] = np.sqrt(csum[k - 1:] - np.concatenate(([0.0], csum[:-k])))
    return rv


def binarize_return(r: np.ndarray, tau: float = 0.0) -> np.ndarray:
    return np.where(r >= tau, 1, 0)


def triple_barrier_label(
    prices: np.ndarray,
    pt: float = 0.008,
    sl: float = 0.008,
    max_hold: int = 60,
) -> np.ndarray:
    """Triple-barrier labeling: 1 if TP hit first, 0 if SL hit first, NaN if timeout.

    Timeout (neither barrier hit within max_hold) is NaN so SeqDataset drops the sample.
    Last max_hold bars are also NaN (no forward window).
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    y = np.full(T, np.nan, dtype=np.float64)
    for i in range(T - max_hold):
        entry = prices[i]
        window = prices[i + 1: i + max_hold + 1]
        up = np.nonzero(window >= entry * (1.0 + pt))[0]
        dn = np.nonzero(window <= entry * (1.0 - sl))[0]
        if len(up) == 0 and len(dn) == 0:
            continue  # timeout - leave as NaN, SeqDataset will drop
        up_t = int(up[0]) if len(up) else max_hold
        dn_t = int(dn[0]) if len(dn) else max_hold
        y[i] = 1 if up_t < dn_t else 0
    return y
