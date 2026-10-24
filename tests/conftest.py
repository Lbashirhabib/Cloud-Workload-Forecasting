"""Shared helpers: a small fake VM trace in the same format as the real data."""
import numpy as np
import pandas as pd
import pytest


def make_readings(n_vms: int = 40, n_steps: int = 140, seed: int = 0) -> pd.DataFrame:
    """Fake readings: every VM follows a shared slow wave plus its own noise."""
    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps)
    shared = 10 + 3 * np.sin(steps / 12)
    rows = []
    for vm in range(n_vms):
        level = rng.uniform(0.5, 1.5)
        avg = np.clip(level * shared + rng.normal(0, 2, n_steps), 0, 100)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps * 300,
                    "vm_id": f"vm{vm}",
                    "min_cpu": np.clip(avg - 1, 0, 100),
                    "max_cpu": np.clip(avg + 5, 0, 100),
                    "avg_cpu": avg,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def readings():
    return make_readings()


@pytest.fixture
def data_dir(tmp_path, readings):
    """A folder laid out like challenge_data/ with a fake readings file."""
    readings.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    return tmp_path
