from typing import Iterator, Tuple

def rolling_windows(T:int, train_len:int, val_len:int, step:int) -> Iterator[Tuple[slice,slice]]:
    """
    Yields (train_slice, val_slice); test comes from next roll or live.
    Example: train_len=60d, val_len=7d, step=7d in bar units.
    """
    start = 0
    while start + train_len + val_len <= T:
        tr = slice(start, start+train_len)
        va = slice(start+train_len, start+train_len+val_len)
        yield tr, va
        start += step
