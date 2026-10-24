import numpy as np

from cwfc.models import damped_ets, make_forecasters, persistence


def _noisy_series(n=100, seed=1):
    rng = np.random.default_rng(seed)
    return 9 + 0.5 * np.sin(np.arange(n) / 10) + rng.normal(0, 0.1, n)


def test_persistence_repeats_the_last_value():
    result = persistence([1.0, 2.0, 7.0], horizon=5)
    assert list(result.mean) == [7.0] * 5


def test_forecasts_are_clipped_to_valid_range():
    assert persistence([150.0], 3, bounds=(0, 100)).mean.max() == 100
    assert persistence([-5.0], 3, bounds=(0, 100)).mean.min() == 0


def test_damped_ets_returns_a_real_model_with_a_sensible_interval():
    result = damped_ets(_noisy_series(), horizon=24, bounds=(0, 100))
    assert result.method == "damped_ets"  # i.e. it did NOT silently fall back
    assert len(result.mean) == 24
    assert np.isfinite(result.mean).all()
    assert (result.lower <= result.mean).all() and (result.mean <= result.upper).all()
    # the interval should widen as we look further ahead
    assert (result.upper - result.lower)[-1] > (result.upper - result.lower)[0]


def test_damped_ets_falls_back_on_constant_or_short_series():
    constant = damped_ets(np.full(50, 10000.0), horizon=4)
    assert constant.method.startswith("persistence")
    assert list(constant.mean) == [10000.0] * 4
    short = damped_ets([1.0, 2.0, 3.0], horizon=2)
    assert short.method.startswith("persistence")


def test_forecast_only_depends_on_the_series_it_is_given():
    y = _noisy_series(120)
    first = damped_ets(y[:100], 24).mean
    # same first 100 points, different "future": identical forecast
    y_changed = y.copy()
    y_changed[100:] += 50
    assert np.array_equal(first, damped_ets(y_changed[:100], 24).mean)


def test_make_forecasters_has_both_models():
    assert set(make_forecasters()) == {"persistence", "damped_ets"}
