from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out


def rolling_origin_train_sizes(
    n: int, horizon: int, min_train_size: int, step: int
) -> list[int]:
    """Position-based version of the split above (works on plain arrays).

    Each returned number is how many points a fold may train on. The fold then
    forecasts the next `horizon` points. The training part always grows and always
    ends before the points being forecast, so nothing from the future is used.

    Example: n=110, horizon=24, min_train_size=60, step=4 -> [60, 64, ..., 84]
    (the last fold forecasts points 84..107, so the final 2 points are never used
    as a forecast target; that is fine, it only means a few points are skipped).
    """
    if min(horizon, min_train_size, step) <= 0:
        raise ValueError("horizon, min_train_size and step must be positive")
    sizes = []
    size = min_train_size
    while size + horizon <= n:
        sizes.append(size)
        size += step
    return sizes
