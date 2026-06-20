from __future__ import annotations
import pandas as pd

def to_utc_ms(ts) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)
