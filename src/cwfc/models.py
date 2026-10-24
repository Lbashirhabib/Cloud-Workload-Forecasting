"""The two forecasting models: a persistence baseline and damped-trend ETS.

Both take a 1-D array of past values and a horizon, and return a ForecastResult.
Neither can see anything after the end of the array it is given, which is what
keeps the validation honest.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np

from .data import clean_for_fit

MIN_ETS_OBSERVATIONS = 12  # below this the ETS fit is not trustworthy


@dataclass(frozen=True)
class ForecastResult:
    mean: np.ndarray
    lower: np.ndarray  # lower edge of the prediction interval (NaN if the model has none)
    upper: np.ndarray  # upper edge of the prediction interval (NaN if the model has none)
    method: str


def _clip(values, bounds):
    if bounds is None:
        return values
    low, high = bounds
    return np.clip(values, low, high)


def persistence(y, horizon: int, bounds=None) -> ForecastResult:
    """Baseline: the next `horizon` values all equal the last observed value."""
    y = clean_for_fit(y)
    if len(y) == 0:
        raise ValueError("cannot forecast from an empty series")
    mean = _clip(np.full(horizon, y[-1], dtype=float), bounds)
    nan = np.full(horizon, np.nan)
    return ForecastResult(mean, nan, nan.copy(), "persistence")


def damped_ets(y, horizon: int, bounds=None, alpha: float = 0.10) -> ForecastResult:
    """Main model: exponential smoothing with a damped additive trend.

    "Damped" means any upward or downward trend fades out over the horizon instead of
    continuing in a straight line. `alpha=0.10` gives a 90% prediction interval.

    If the series is too short, has no variation at all, or the fit fails, we fall back
    to persistence (with a zero-width interval) and say so in `method`.
    """
    y = clean_for_fit(y)
    if len(y) < MIN_ETS_OBSERVATIONS or np.ptp(y) < 1e-12:
        return _persistence_fallback(y, horizon, bounds, "too short or constant")

    # imported here so that the baseline and the tests of it do not need statsmodels
    import pandas as pd
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # a pandas Series (not a bare array) is needed for get_prediction()
            model = ETSModel(
                pd.Series(y), error="add", trend="add", damped_trend=True,
                initialization_method="estimated",
            )
            fitted = model.fit(disp=False)
            frame = fitted.get_prediction(
                start=len(y), end=len(y) + horizon - 1
            ).summary_frame(alpha=alpha)
        mean = frame["mean"].to_numpy(dtype=float)
        lower = frame["pi_lower"].to_numpy(dtype=float)
        upper = frame["pi_upper"].to_numpy(dtype=float)
    except Exception as error:  # statsmodels can fail to converge on odd series
        return _persistence_fallback(y, horizon, bounds, f"fit failed: {type(error).__name__}")

    if not (np.isfinite(mean).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        return _persistence_fallback(y, horizon, bounds, "non-finite forecast")

    return ForecastResult(
        _clip(mean, bounds), _clip(lower, bounds), _clip(upper, bounds), "damped_ets"
    )


def _persistence_fallback(y, horizon, bounds, reason: str) -> ForecastResult:
    mean = _clip(np.full(horizon, y[-1], dtype=float), bounds)
    return ForecastResult(mean, mean.copy(), mean.copy(), f"persistence (ets fallback: {reason})")


BASELINE_NAME = "persistence"
MAIN_MODEL_NAME = "damped_ets"
MODELS = {BASELINE_NAME: persistence, MAIN_MODEL_NAME: damped_ets}


def make_forecasters(bounds=None, alpha: float = 0.10) -> dict:
    """Both models with the target's valid range baked in."""
    return {
        BASELINE_NAME: partial(persistence, bounds=bounds),
        MAIN_MODEL_NAME: partial(damped_ets, bounds=bounds, alpha=alpha),
    }
