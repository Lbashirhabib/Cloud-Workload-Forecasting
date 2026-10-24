import gzip
import hashlib
import io
import tarfile

import numpy as np
import pandas as pd
import pytest

import cwfc.data as data_module
from cwfc.data import build_targets, check_data_quality, clean_for_fit, split_train_holdout


def _frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])


def test_each_vm_is_counted_once_per_timestamp():
    readings = _frame([(0, "a", 10.0), (0, "a", 90.0), (0, "b", 20.0)])  # "a" appears twice
    targets = build_targets(readings)
    assert targets.loc[0, "active_vm_count"] == 2
    assert targets.loc[0, "aggregate_cpu_utilization"] == pytest.approx(15.0)  # first "a" kept


def test_missing_and_out_of_range_readings_are_not_valid():
    readings = _frame([(0, "a", 10.0), (0, "b", np.nan), (0, "c", 150.0), (0, "d", -1.0)])
    targets = build_targets(readings)
    assert targets.loc[0, "active_vm_count"] == 1
    assert targets.loc[0, "aggregate_cpu_utilization"] == pytest.approx(10.0)


def test_timestamp_without_valid_readings_is_kept_not_invented():
    readings = _frame([(0, "a", 10.0), (300, "a", np.nan)])
    targets = build_targets(readings)
    assert list(targets.index) == [0, 300]
    assert targets.loc[300, "active_vm_count"] == 0
    assert np.isnan(targets.loc[300, "aggregate_cpu_utilization"])


def test_targets_are_sorted_by_time():
    readings = _frame([(600, "a", 1.0), (0, "a", 2.0), (300, "a", 3.0)])
    assert list(build_targets(readings).index) == [0, 300, 600]


def test_quality_check_finds_irregular_step_and_duplicates():
    readings = _frame([(0, "a", 1.0), (300, "a", 1.0), (900, "a", 1.0), (900, "a", 2.0)])
    report = check_data_quality(readings)
    assert report["irregular_steps"] == 1  # 300 -> 900 is a 600 s gap
    assert report["duplicate_vm_timestamp_rows"] == 1
    assert report["timestamps"] == 3


def test_clean_for_fit_only_looks_backwards():
    filled = clean_for_fit([np.nan, 1.0, np.nan, np.nan, 5.0])
    assert list(filled) == [1.0, 1.0, 1.0, 1.0, 5.0]


def test_split_is_chronological_and_sized():
    targets = pd.DataFrame({"x": range(134)}, index=np.arange(134) * 300)
    train, holdout = split_train_holdout(targets, 24)
    assert len(train) == 110 and len(holdout) == 24
    assert train.index.max() < holdout.index.min()


def test_split_needs_more_rows_than_horizon():
    with pytest.raises(ValueError):
        split_train_holdout(pd.DataFrame({"x": range(10)}), 24)


def test_load_readings_downloads_and_extracts_release_when_missing(tmp_path, monkeypatch):
    readings_csv = gzip.compress(b"timestamp,vm_id,avg_cpu\n0,vm1,12.5\n")
    metadata_csv = gzip.compress(b"vm_id,category\nvm1,test\n")
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for name, contents in (
            ("vm_cpu_readings.csv.gz", readings_csv),
            ("vm_metadata.csv.gz", metadata_csv),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    archive_bytes = archive_buffer.getvalue()

    monkeypatch.setattr(
        data_module.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(archive_bytes),
    )
    monkeypatch.setattr(data_module, "DATASET_SHA256", hashlib.sha256(archive_bytes).hexdigest())

    data_dir = tmp_path / "challenge_data"
    readings = data_module.load_readings(data_dir)

    assert readings.loc[0, "avg_cpu"] == pytest.approx(12.5)
    assert (data_dir / "vm_metadata.csv.gz").is_file()
