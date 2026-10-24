import numpy as np
import pandas as pd

from cwfc.evaluate import (
    default_min_train,
    interval_coverage,
    rolling_origin_forecasts,
    score_table,
    select_model,
)
from cwfc.models import ForecastResult
from cwfc.policy import margin_tradeoff, smallest_margin_for_coverage
from cwfc.validation import rolling_origin_train_sizes


def test_train_sizes_for_the_real_window():
    assert rolling_origin_train_sizes(110, horizon=24, min_train_size=60, step=4) == [
        60, 64, 68, 72, 76, 80, 84
    ]


def test_default_min_train_is_60_for_110_points_and_shrinks_for_short_data():
    assert default_min_train(110, 24, 4) == 60
    assert default_min_train(50, 24, 4) == 22
    assert default_min_train(10, 24, 4) == 12  # never below 12


def test_each_fold_trains_only_on_the_past():
    seen = []

    def spy(train, horizon):
        seen.append(len(train))
        return ForecastResult(np.zeros(horizon), np.zeros(horizon), np.zeros(horizon), "spy")

    y = np.arange(110, dtype=float)
    results = rolling_origin_forecasts(y, {"spy": spy}, horizon=24, min_train=60, step=4)
    assert seen == [60, 64, 68, 72, 76, 80, 84]
    # the values being forecast start exactly where training ends, never earlier
    for fold, group in results.groupby("fold"):
        assert group["position"].min() == group["n_train"].iloc[0]
    # the peak threshold of a fold is built from that fold's training data only
    first = results[results["fold"] == 0].iloc[0]
    assert first["peak_threshold"] == np.quantile(y[:60], 0.75)


def test_score_table_ratio_is_nan_when_baseline_is_perfect():
    results = pd.DataFrame(
        {
            "model": ["persistence"] * 3 + ["m"] * 3,
            "fold": 0,
            "actual": [1.0, 1.0, 1.0] * 2,
            "forecast": [1.0, 1.0, 1.0, 1.1, 1.1, 1.1],
            "is_peak": False,
        }
    )
    table = score_table(results)
    assert table.loc["persistence", "MAE"] == 0
    assert np.isnan(table.loc["m", "MAE_ratio"])


def test_score_table_shows_improvement_over_baseline():
    results = pd.DataFrame(
        {
            "model": ["persistence"] * 2 + ["m"] * 2,
            "fold": 0,
            "actual": [10.0, 10.0, 10.0, 10.0],
            "forecast": [12.0, 12.0, 11.0, 11.0],
            "is_peak": False,
        }
    )
    table = score_table(results)
    assert table.loc["m", "MAE_ratio"] == 0.5
    assert table.loc["m", "improvement_%"] == 50.0


def test_select_model_keeps_baseline_when_there_is_no_room_for_a_fold():
    chosen, scores, results = select_model(np.arange(30, dtype=float), horizon=24)
    assert chosen == "persistence" and scores.empty and results.empty


def test_select_model_prefers_baseline_on_a_tie():
    chosen, _, _ = select_model(np.full(110, 5.0), horizon=24)  # constant: both perfect
    assert chosen == "persistence"


def test_interval_coverage_and_margin_tradeoff():
    results = pd.DataFrame(
        {
            "model": "m",
            "actual": [1.0, 2.0, 3.0, 4.0],
            "forecast": [1.0, 2.0, 3.0, 3.0],
            "lower": [0.5, 1.5, 2.5, 2.5],
            "upper": [1.5, 2.5, 3.5, 3.5],
        }
    )
    assert interval_coverage(results, "m") == 75.0  # the last actual (4.0) is above 3.5
    table = margin_tradeoff(results, "m", basis="upper", margins=(0.0, 0.5))
    assert table.loc[0.0, "covered_%"] == 75.0
    assert table.loc[50.0, "covered_%"] == 100.0
    assert smallest_margin_for_coverage(table, 100.0) == 50.0
    assert smallest_margin_for_coverage(table, 100.1) is None
