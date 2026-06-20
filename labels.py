import pandas as pd
import numpy as np

def triple_barrier_labels(df: pd.DataFrame, pt=0.01, sl=0.01, max_h=20):
    """
    df must have 'close'. Returns 1 (tp hit), 0 (sl hit), or NaN (timeout/no hit).
    pt/sl are fractions, max_h is horizon in bars.
    """
    close = df['close'].values
    n = len(close)
    y = np.full(n, np.nan)

    for i in range(n - max_h):
        entry = close[i]
        up = entry * (1 + pt)
        dn = entry * (1 - sl)
        for j in range(1, max_h+1):
            px = close[i+j]
            if px >= up:
                y[i] = 1
                break
            if px <= dn:
                y[i] = 0
                break
        # if neither, stays NaN (timeout)

    return pd.Series(y, index=df.index, name='y_tb')
