"""A simple headroom policy, tested on the rolling-origin forecasts.

Policy: plan for  (forecast upper band) x (1 + margin).
  * Under-provisioning  = actual demand ends up above the planned level (risk of
    slowdowns / failed scale-ups).
  * Over-provisioning   = planned level is above what was needed (wasted money).
A bigger margin trades more waste for fewer shortfalls. `margin_tradeoff` measures
that trade on past folds so the margin can be chosen from evidence, not guessed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.20)


def margin_tradeoff(
    results: pd.DataFrame,
    model: str,
    basis: str = "upper",
    margins=DEFAULT_MARGINS,
) -> pd.DataFrame:
    """For each margin: how often demand stays covered and how much is wasted.

    basis="upper"    -> plan = upper edge of the 90% interval x (1 + margin)
    basis="forecast" -> plan = point forecast x (1 + margin)   (what you would do
                        with a plain forecast and no uncertainty band)
    Units are "CPU percentage points" for the CPU target.
    """
    g = results[results["model"] == model].dropna(subset=["actual", basis])
    rows = []
    for margin in margins:
        planned = g[basis] * (1 + margin)
        gap = planned - g["actual"]  # positive = spare room, negative = shortfall
        waste = gap.clip(lower=0)
        shortfall = (-gap).clip(lower=0)
        rows.append(
            {
                "margin_%": margin * 100,
                "covered_%": float((gap >= 0).mean() * 100),
                "avg_spare_pts": float(waste.mean()),
                "avg_spare_%_of_actual": float((waste / np.maximum(g["actual"], 1e-8)).mean() * 100),
                "avg_shortfall_pts": float(shortfall.mean()),
                "worst_shortfall_pts": float(shortfall.max()),
            }
        )
    return pd.DataFrame(rows).set_index("margin_%")


def smallest_margin_for_coverage(table: pd.DataFrame, target_covered: float = 99.0):
    """Smallest margin (in %) whose covered_% reaches the target, or None."""
    ok = table[table["covered_%"] >= target_covered]
    return None if ok.empty else float(ok.index.min())
