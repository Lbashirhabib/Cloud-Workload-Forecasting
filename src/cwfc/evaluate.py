"""Rolling-origin evaluation, scoring, and choosing between the two models.

The "rolling origin" idea in plain English: pretend it is time T, give the model only
data up to T, ask for the next 24 steps, and compare with what really happened. Then
move T forward and repeat. All of this happens inside the training window; the real
holdout is never touched here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mae, rmse, smape
from .models import BASELINE_NAME, MAIN_MODEL_NAME, make_forecasters
from .validation import rolling_origin_train_sizes

# defaults chosen for a 110-point training window (see the notebook for the reasoning)
DEFAULT_MIN_TRAIN = 60
DEFAULT_STEP = 4
DEFAULT_PEAK_QUANTILE = 0.75


def default_min_train(n_train: int, horizon: int = 24, step: int = DEFAULT_STEP) -> int:
    """First fold's training size: 60 normally, smaller if the series is short.

    For the real 110-point window this returns 60. It only shrinks (never below 12)
    when the data is too short for 60, so the pipeline also runs on tiny smoke data.
    """
    return int(max(12, min(DEFAULT_MIN_TRAIN, n_train - horizon - step)))


def rolling_origin_forecasts(
    y,
    forecasters: dict,
    horizon: int = 24,
    min_train: int = DEFAULT_MIN_TRAIN,
    step: int = DEFAULT_STEP,
    peak_quantile: float = DEFAULT_PEAK_QUANTILE,
) -> pd.DataFrame:
    """Run every forecaster on every fold; return one row per (model, fold, step).

    A step counts as "peak" if the actual value is at or above the `peak_quantile`
    quantile of that fold's own training data, so peak labels use only the past.
    """
    y = np.asarray(y, dtype=float)
    frames = []
    for fold, n_train in enumerate(rolling_origin_train_sizes(len(y), horizon, min_train, step)):
        train = y[:n_train]
        actual = y[n_train : n_train + horizon]
        threshold = np.nanquantile(train, peak_quantile)
        for name, forecaster in forecasters.items():
            result = forecaster(train, horizon)
            frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "method": result.method,  # shows if ETS had to fall back
                        "fold": fold,
                        "n_train": n_train,
                        "step_ahead": np.arange(1, horizon + 1),
                        "position": np.arange(n_train, n_train + horizon),
                        "actual": actual,
                        "forecast": result.mean,
                        "lower": result.lower,
                        "upper": result.upper,
                        "peak_threshold": threshold,
                        "is_peak": actual >= threshold,
                    }
                )
            )
    if not frames:
        return pd.DataFrame(
            columns=["model", "method", "fold", "n_train", "step_ahead", "position", "actual",
                     "forecast", "lower", "upper", "peak_threshold", "is_peak"]
        )
    return pd.concat(frames, ignore_index=True)


def score_table(results: pd.DataFrame, baseline: str = BASELINE_NAME) -> pd.DataFrame:
    """One row per model: MAE, RMSE, sMAPE, peak MAE, and comparison with the baseline.

    sMAPE divides by |actual| + |forecast|, floored at 1e-8, so a near-zero denominator
    can never blow the metric up (two zeros count as zero error).
    `MAE_ratio` is model MAE divided by baseline MAE (below 1 is better than the
    baseline). When the baseline's MAE is exactly 0 the ratio is undefined, so it is NaN.
    """
    rows = []
    for name, group in results.groupby("model", sort=False):
        group = group.dropna(subset=["actual", "forecast"])
        peak = group[group["is_peak"]]
        rows.append(
            {
                "model": name,
                "MAE": mae(group["actual"], group["forecast"]),
                "RMSE": rmse(group["actual"], group["forecast"]),
                "sMAPE_%": smape(group["actual"], group["forecast"]),
                "peak_MAE": mae(peak["actual"], peak["forecast"]) if len(peak) else np.nan,
                "folds": int(group["fold"].nunique()),
                "points": int(len(group)),
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    if baseline in table.index:
        base_mae = table.loc[baseline, "MAE"]
        base_peak = table.loc[baseline, "peak_MAE"]
        table["MAE_ratio"] = table["MAE"] / base_mae if base_mae > 0 else np.nan
        table["improvement_%"] = (1 - table["MAE_ratio"]) * 100
        table["peak_MAE_ratio"] = table["peak_MAE"] / base_peak if base_peak > 0 else np.nan
    return table


def error_by_step(results: pd.DataFrame) -> pd.DataFrame:
    """MAE for each 'how far ahead' step (1 = five minutes ahead ... 24 = two hours)."""
    return results.groupby(["step_ahead", "model"])[["actual", "forecast"]].apply(
        lambda g: mae(g["actual"], g["forecast"])
    ).unstack("model")


def error_by_fold(results: pd.DataFrame) -> pd.DataFrame:
    """MAE for each fold, so you can see which stretch of time was hard to forecast."""
    return results.groupby(["fold", "model"])[["actual", "forecast"]].apply(
        lambda g: mae(g["actual"], g["forecast"])
    ).unstack("model")


def interval_coverage(results: pd.DataFrame, model: str = MAIN_MODEL_NAME) -> float:
    """Share of actual values that fell inside the model's prediction interval (in %)."""
    g = results[results["model"] == model].dropna(subset=["actual", "lower", "upper"])
    if g.empty:
        return float("nan")
    inside = (g["actual"] >= g["lower"]) & (g["actual"] <= g["upper"])
    return float(inside.mean() * 100)


def select_model(
    y_train,
    horizon: int = 24,
    bounds=None,
    min_train: int = DEFAULT_MIN_TRAIN,
    step: int = DEFAULT_STEP,
):
    """Pick the model with the lower rolling-origin MAE. Uses training data only.

    Returns (chosen_name, scores, results). The baseline wins ties because it is
    simpler. If the training window is too short for even one fold, we cannot
    compare, so we keep the baseline.
    """
    y_train = np.asarray(y_train, dtype=float)
    forecasters = make_forecasters(bounds)
    results = rolling_origin_forecasts(y_train, forecasters, horizon, min_train, step)
    if results.empty:
        return BASELINE_NAME, pd.DataFrame(), results
    scores = score_table(results)
    chosen = MAIN_MODEL_NAME if scores.loc[MAIN_MODEL_NAME, "MAE"] < scores.loc[BASELINE_NAME, "MAE"] else BASELINE_NAME
    return chosen, scores, results
