# Cloud Workload Forecasting: persistence baseline vs damped-trend ETS

Take-home solution for the Atomity forecasting challenge (Azure Public Dataset V2 subset, `data-v1`).
I am to forecasts the final 2 hours (24 five-minute steps) of two fleet-level series, `active_vm_count` and
`aggregate_cpu_utilization`, and turns the forecast into a simple capacity-planning policy.

**Headline findings** (details in [`reports/final_report.md`](reports/final_report.md) and the notebook)

* `active_vm_count` is exactly 10,000 at every step (no VM is created or deleted in the window), so persistence is exact.
* For CPU, **persistence beat damped-trend ETS** in rolling-origin validation (MAE 0.219 vs 0.239 CPU points, about 9% better). The pipeline picks the model by that rule, so the submitted forecast is the baseline.
* A plan of `latest fleet CPU x 1.05` was high enough in every validation step. On the real holdout it was high enough in 23 of 24 steps.

## 1. Install

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv\Scripts\Activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`requirements.txt` lists the exact versions this was developed and tested with (`pip install -r requirements.txt` then `pip install -e . --no-deps` if you want to match them exactly).

## 2. Get the data

The first time the forecasting code runs, it checks `challenge_data/`. If `vm_cpu_readings.csv.gz` is missing, it automatically downloads the `data-v1` release archive, verifies its SHA-256 checksum, and extracts it into that folder. You can also download and extract it manually:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
(cd challenge_data && sha256sum -c SHA256SUMS)     # all three should say OK
```

The data files are not committed (see `.gitignore`). The code reads `vm_cpu_readings.csv.gz` from the folder you pass with `--data`.

## 3. Generate the predictions (one command)

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This writes `predictions.csv` with columns `timestamp,active_vm_count,aggregate_cpu_utilization` and exactly 24 rows (the final 24 timestamps, in order). The file is checked against every rule in `SUBMISSION.md` before it is written. Takes about 5 seconds.

Optional flags: `--summary-json path.json` saves the model choices and validation scores; `--evaluate` prints the score on the holdout **after** the file is written (reporting only).

## 4. Re-run the analysis

```bash
jupyter lab notebooks/01_cloud_workload_forecasting.ipynb     # then Kernel -> Restart & Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_cloud_workload_forecasting.ipynb
```

The notebook walks through data checks, workload analysis, validation, results, error analysis, the capacity policy and the final test, in plain English. It takes about 20 seconds and saves its figures to `reports/figures/`. It also re-runs the command-line tool and checks that both give identical predictions.

## 5. Tests

```bash
python -m pytest -q
```

29 tests cover target construction, the models, rolling-origin folds, metrics, the policy table, the submission validator, a tiny-dataset smoke run, and a **leakage test** that scrambles the last 24 timestamps' readings and checks the forecast does not change.

## 6. Project layout

```text
src/cwfc/
  data.py        load files, data-quality checks, build the two targets, train/holdout split
  models.py      persistence baseline and damped-trend ETS (with prediction interval)
  validation.py  rolling-origin fold sizes (plus the starter's timestamp version)
  evaluate.py    run folds, score tables, error by step/fold, model selection rule
  policy.py      margin vs coverage/waste table for the capacity policy
  metrics.py     MAE, RMSE, sMAPE (from the starter, unchanged)
  predict.py     command-line entry point and submission validator
notebooks/01_cloud_workload_forecasting.ipynb   the full analysis (executed, with outputs)
reports/final_report.md                          4-6 page report;  reports/figures/ = its charts
tests/                                           pytest suite
```

## 7. Main modeling decisions

* **Targets.** Built from the full readings first, then split: first 110 steps train, last 24 holdout. A VM is counted once per timestamp; invalid readings (missing or outside 0-100) are dropped; nothing is resampled or invented.
* **Two models only.** Persistence (last value repeated) and damped-trend ETS. With about 9 hours of history there is no daily cycle to learn, so seasonal models are impossible and bigger ML models would overfit 110 points.
* **Validation.** Rolling-origin inside the training window only: first fold trains on 60 points, forecasts 24, then moves forward 4 steps (7 folds). No shuffling.
* **Model choice by rule.** Per target, pick the lower validation MAE (ties go to persistence). The same function runs in the notebook and the command-line tool.
* **Metrics.** MAE, RMSE, sMAPE (denominator floored at 1e-8), MAE ratio vs baseline, and MAE on peak steps (actual at or above the 75th percentile of that fold's own training data).
* **Leakage protection.** Only the holdout *timestamps* are read from the holdout rows; the holdout values are never used to fit, tune or select. They are scored once, in the final-test section, after everything is decided.
* **Capacity policy.** Plan for forecast x (1 + margin); the margin is chosen from a coverage/waste table on the validation folds.

## 8. Known limitations

About 9 hours of history (no daily pattern), a single hump in the series, seven overlapping validation folds, zero VM churn, and a fleet-average view that hides a hot tail of busy VMs. See section 9 of the report and the last section of the notebook.

## Attribution

Data: curated subset of the Microsoft Azure Public Dataset V2 (<https://github.com/Azure/AzurePublicDataset>); follow the upstream license and usage terms. Starter utilities (`metrics.py`, the original `validation.py` function, `baselines.py`) come from the challenge repository.
