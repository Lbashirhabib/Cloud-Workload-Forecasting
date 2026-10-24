import numpy as np
import pandas as pd
import pytest

from cwfc.data import build_targets
from cwfc.predict import OUTPUT_COLUMNS, forecast_holdout, main, validate_predictions
from conftest import make_readings


def test_command_line_writes_a_valid_file(data_dir, readings, tmp_path):
    out = tmp_path / "predictions.csv"
    assert main(["--data", str(data_dir), "--output", str(out)]) == 0
    pred = pd.read_csv(out)
    assert list(pred.columns) == OUTPUT_COLUMNS
    assert len(pred) == 24
    last_24 = np.sort(readings["timestamp"].unique())[-24:]
    assert np.array_equal(pred["timestamp"].to_numpy(), last_24)
    validate_predictions(pred, last_24)  # raises if any SUBMISSION.md rule is broken


def test_changing_the_holdout_values_does_not_change_the_forecast(readings):
    """The leakage test: scramble the last 24 timestamps' readings, forecast again."""
    horizon = 24
    original, _ = forecast_holdout(build_targets(readings), horizon)

    last_24 = np.sort(readings["timestamp"].unique())[-horizon:]
    tampered = readings.copy()
    in_holdout = tampered["timestamp"].isin(last_24)
    tampered.loc[in_holdout, "avg_cpu"] = 99.0  # wildly different holdout CPU
    tampered = tampered[~(in_holdout & tampered["vm_id"].isin(["vm0", "vm1", "vm2"]))]  # and fewer VMs

    after, _ = forecast_holdout(build_targets(tampered), horizon)
    pd.testing.assert_frame_equal(original, after)


def test_pipeline_runs_on_a_tiny_dataset(tmp_path):
    tiny = make_readings(n_vms=5, n_steps=30)
    tiny.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    out = tmp_path / "p.csv"
    assert main(["--data", str(tmp_path), "--output", str(out)]) == 0
    assert len(pd.read_csv(out)) == 24


def test_missing_data_folder_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download"):
        main(["--data", str(tmp_path / "nope"), "--output", str(tmp_path / "p.csv")])


def _good(n=3):
    return pd.DataFrame(
        {"timestamp": np.arange(n) * 300, "active_vm_count": [10.0] * n,
         "aggregate_cpu_utilization": [9.0] * n}
    )


def test_validator_accepts_good_and_rejects_bad_files():
    good = _good()
    validate_predictions(good, good["timestamp"])

    for bad_change in (
        lambda d: d.assign(aggregate_cpu_utilization=[9.0, np.nan, 9.0]),
        lambda d: d.assign(aggregate_cpu_utilization=[9.0, 101.0, 9.0]),
        lambda d: d.assign(active_vm_count=[10.0, -1.0, 10.0]),
        lambda d: d.iloc[:2],
        lambda d: d.assign(timestamp=[0, 300, 300]),
        lambda d: d.rename(columns={"timestamp": "ts"}),
    ):
        with pytest.raises(ValueError):
            validate_predictions(bad_change(good.copy()), good["timestamp"])
