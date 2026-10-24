"""Loading the challenge files and building the two forecasting targets.

Everything here works on the raw VM-level readings. Nothing in this file knows
about the train/holdout split, so no information from the holdout can leak in
through feature building (the split happens later, in predict.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

import numpy as np
import pandas as pd

READINGS_FILE = "vm_cpu_readings.csv.gz"
METADATA_FILE = "vm_metadata.csv.gz"
DATASET_URL = (
    "https://github.com/atomityhq/forecasting-challenge/releases/download/"
    "data-v1/data-v1.tar.gz"
)
DATASET_SHA256 = "d04d402ac1719c1f5b869e3699fc3d57f6f8273751f1ef19bedb29397b39caeb"
STEP_SECONDS = 300  # the data is sampled every five minutes
TARGETS = ("active_vm_count", "aggregate_cpu_utilization")

# valid range for each target, used to clip forecasts
TARGET_BOUNDS = {
    "active_vm_count": (0.0, None),
    "aggregate_cpu_utilization": (0.0, 100.0),
}


def _extract_dataset(archive_path: Path, data_dir: Path) -> None:
    """Extract regular files and directories without allowing archive paths out of data_dir."""
    root = data_dir.resolve()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or "\\" in member.name
                or ":" in member.name
                or ".." in member_path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe or unsupported path in dataset archive: {member.name!r}")
            destination = (data_dir / member.name).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Dataset archive path escapes the destination: {member.name!r}"
                ) from exc

        for member in members:
            destination = data_dir / member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read dataset archive member: {member.name!r}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def download_dataset(data_dir) -> None:
    """Download and extract the pinned data-v1 release into `data_dir`."""
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_dir:
        archive_path = Path(temporary_dir) / "data-v1.tar.gz"
        digest = hashlib.sha256()
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            with archive_path.open("wb") as archive_file:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    archive_file.write(chunk)
        if digest.hexdigest() != DATASET_SHA256:
            raise ValueError(
                "Downloaded dataset checksum does not match the pinned data-v1 release."
            )
        _extract_dataset(archive_path, destination)


def load_readings(data_dir) -> pd.DataFrame:
    """Read the VM-level five-minute CPU readings."""
    path = Path(data_dir) / READINGS_FILE
    if not path.exists():
        download_dataset(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found after downloading the data-v1 dataset.")
    readings = pd.read_csv(path)
    missing = {"timestamp", "vm_id", "avg_cpu"} - set(readings.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return readings


def load_metadata(data_dir) -> pd.DataFrame:
    """Read the VM metadata (category, core bucket, memory bucket, ...)."""
    path = Path(data_dir) / METADATA_FILE
    if not path.exists():
        download_dataset(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found after downloading the data-v1 dataset.")
    return pd.read_csv(path)


def check_data_quality(readings: pd.DataFrame) -> dict:
    """Count the things that can go wrong with a time-series table.

    Returns plain Python numbers so the result can be printed or saved as JSON.
    """
    timestamps = np.sort(readings["timestamp"].unique())
    gaps = np.diff(timestamps)
    per_vm = readings.groupby("vm_id").size()
    avg = readings["avg_cpu"]
    return {
        "rows": int(len(readings)),
        "vms": int(readings["vm_id"].nunique()),
        "timestamps": int(len(timestamps)),
        "first_timestamp": int(timestamps[0]),
        "last_timestamp": int(timestamps[-1]),
        "step_seconds_seen": {int(k): int(v) for k, v in pd.Series(gaps).value_counts().items()},
        "irregular_steps": int((gaps != STEP_SECONDS).sum()),
        "duplicate_vm_timestamp_rows": int(readings.duplicated(["timestamp", "vm_id"]).sum()),
        "missing_avg_cpu": int(avg.isna().sum()),
        "avg_cpu_outside_0_100": int(((avg < 0) | (avg > 100)).sum()),
        "readings_per_vm_min": int(per_vm.min()),
        "readings_per_vm_max": int(per_vm.max()),
    }


def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Turn VM-level readings into the two aggregate time series.

    active_vm_count            = number of distinct VMs with a valid reading at t
    aggregate_cpu_utilization  = mean of the VMs' avg_cpu at t, in percent

    How bad rows are treated (the "assumptions" the assignment asks us to state):
      * a reading with a missing avg_cpu, or avg_cpu outside [0, 100], is not valid
        and is dropped (it is not counted and not averaged);
      * if the same VM appears twice at one timestamp only the first row is kept,
        so each VM is counted at most once;
      * a timestamp that exists in the file but has no valid reading keeps its row
        with active_vm_count = 0 and aggregate_cpu_utilization = NaN. We do not
        invent a CPU value for it.
      * no resampling: the index is the timestamps that really occur in the file.

    The result is indexed by timestamp (seconds), sorted oldest to newest.
    """
    all_timestamps = np.sort(readings["timestamp"].unique())

    valid = readings.dropna(subset=["avg_cpu"])
    valid = valid[(valid["avg_cpu"] >= 0) & (valid["avg_cpu"] <= 100)]
    valid = valid.drop_duplicates(["timestamp", "vm_id"], keep="first")

    grouped = valid.groupby("timestamp")
    targets = pd.DataFrame(
        {
            "active_vm_count": grouped["vm_id"].nunique(),
            "aggregate_cpu_utilization": grouped["avg_cpu"].mean(),
        }
    ).reindex(all_timestamps)
    targets["active_vm_count"] = targets["active_vm_count"].fillna(0).astype(float)
    targets.index.name = "timestamp"
    return targets


def clean_for_fit(y) -> np.ndarray:
    """Fill NaNs in a training series using only earlier values (forward fill).

    A leading NaN has nothing before it, so it takes the first real value. This is
    only ever applied to training data and only matters if a timestamp had no valid
    readings at all.
    """
    series = pd.Series(np.asarray(y, dtype=float))
    return series.ffill().bfill().to_numpy()


def split_train_holdout(targets: pd.DataFrame, horizon: int = 24):
    """Chronological split: everything except the last `horizon` rows is training."""
    if len(targets) <= horizon:
        raise ValueError(
            f"need more than {horizon} timestamps to hold out {horizon}; got {len(targets)}"
        )
    return targets.iloc[:-horizon], targets.iloc[-horizon:]
