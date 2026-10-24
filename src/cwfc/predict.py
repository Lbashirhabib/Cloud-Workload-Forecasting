"""Command-line entry point: raw challenge data in, predictions.csv out.

    python -m cwfc.predict --data challenge_data --output predictions.csv

What it does, in order:
  1. load the readings and report data-quality checks;
  2. build the full target series (needs only the readings, no split yet);
  3. split chronologically: the last `horizon` timestamps are the holdout;
  4. for each target, pick persistence or damped ETS using rolling-origin MAE
     computed on the TRAINING part only;
  5. fit the chosen model on the whole training part and forecast the holdout;
  6. check the file against SUBMISSION.md and write it.

Only the holdout's timestamps are read from the holdout rows. Its target values are
never used to fit, tune or select anything (a unit test checks this by scrambling them).
`--evaluate` is optional and only prints a score AFTER the file is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    TARGET_BOUNDS,
    TARGETS,
    build_targets,
    check_data_quality,
    load_readings,
    split_train_holdout,
)
from .evaluate import DEFAULT_STEP, default_min_train, select_model
from .metrics import mae, rmse
from .models import BASELINE_NAME, make_forecasters

OUTPUT_COLUMNS = ["timestamp", *TARGETS]


def validate_predictions(pred: pd.DataFrame, expected_timestamps) -> None:
    """Raise ValueError unless `pred` satisfies every rule in SUBMISSION.md."""
    expected = np.asarray(expected_timestamps)
    if list(pred.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"columns must be exactly {OUTPUT_COLUMNS}, got {list(pred.columns)}")
    if len(pred) != len(expected):
        raise ValueError(f"expected {len(expected)} rows, got {len(pred)}")
    if not np.array_equal(pred["timestamp"].to_numpy(), expected):
        raise ValueError("timestamps do not match the holdout timestamps in order")
    if pred["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps")
    values = pred[list(TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictions contain NaN or infinite values")
    if (pred["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    cpu = pred["aggregate_cpu_utilization"]
    if ((cpu < 0) | (cpu > 100)).any():
        raise ValueError("aggregate_cpu_utilization must be within [0, 100]")


def forecast_holdout(targets: pd.DataFrame, horizon: int = 24):
    """Choose a model per target on training data only and forecast the holdout.

    Returns (predictions DataFrame, info dict about the choices made).
    """
    train, holdout = split_train_holdout(targets, horizon)  # holdout used for timestamps only
    columns = {"timestamp": holdout.index.to_numpy()}
    info = {"n_train": int(len(train)), "horizon": int(horizon), "targets": {}}

    for target in TARGETS:
        y_train = train[target].to_numpy(dtype=float)
        bounds = TARGET_BOUNDS[target]
        min_train = default_min_train(len(y_train), horizon, DEFAULT_STEP)
        chosen, scores, results = select_model(y_train, horizon, bounds, min_train, DEFAULT_STEP)
        forecast = make_forecasters(bounds)[chosen](y_train, horizon)
        columns[target] = forecast.mean
        ets_rows = results[results["model"] == "damped_ets"] if len(results) else results
        info["targets"][target] = {
            "chosen_model": chosen,
            "method_used": forecast.method,
            # how the ETS model actually ran during validation (fallbacks are visible here)
            "ets_validation_methods": ets_rows["method"].value_counts().to_dict() if len(ets_rows) else {},
            "validation_scores": scores.round(6).to_dict(orient="index") if len(scores) else {},
        }

    predictions = pd.DataFrame(columns)[OUTPUT_COLUMNS]
    validate_predictions(predictions, holdout.index.to_numpy())
    return predictions, info


def holdout_scores(targets: pd.DataFrame, predictions: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Final test: score the submitted forecast and the baseline on the real holdout.

    Call this only after predictions are final. It is reporting, not tuning.
    """
    train, holdout = split_train_holdout(targets, horizon)
    rows = []
    for target in TARGETS:
        actual = holdout[target].to_numpy(dtype=float)
        base = make_forecasters(TARGET_BOUNDS[target])[BASELINE_NAME](
            train[target].to_numpy(dtype=float), horizon
        ).mean
        mine = predictions[target].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "model_MAE": mae(actual, mine),
                "model_RMSE": rmse(actual, mine),
                "baseline_MAE": mae(actual, base),
                "baseline_RMSE": rmse(actual, base),
            }
        )
    return pd.DataFrame(rows).set_index("target")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Forecast the final holdout and write predictions.csv")
    parser.add_argument("--data", default="challenge_data", help="folder with vm_cpu_readings.csv.gz")
    parser.add_argument("--output", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--horizon", type=int, default=24, help="number of final timestamps to forecast")
    parser.add_argument("--summary-json", default=None, help="optional: save the model choices and scores")
    parser.add_argument("--evaluate", action="store_true",
                        help="after writing, also print the score on the holdout (reporting only)")
    args = parser.parse_args(argv)

    readings = load_readings(args.data)
    quality = check_data_quality(readings)
    print("Data checks:", json.dumps(quality))
    if quality["irregular_steps"] or quality["duplicate_vm_timestamp_rows"] or quality["missing_avg_cpu"]:
        print("WARNING: irregularities found; see README for how they are handled.", file=sys.stderr)

    targets = build_targets(readings)
    predictions, info = forecast_holdout(targets, args.horizon)

    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} rows to {args.output}")
    for target, details in info["targets"].items():
        print(f"  {target}: {details['method_used']}")

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps({"data_quality": quality, **info}, indent=2))
    if args.evaluate:
        print("\nFinal test on the holdout (reporting only):")
        print(holdout_scores(targets, predictions, args.horizon).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
