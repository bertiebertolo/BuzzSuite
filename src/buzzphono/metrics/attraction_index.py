"""Attraction index: (ON − pre) / (ON + pre) of the speaker-zone fraction, per hour-cycle.

The index is +1 when every speaker-zone rester appears only during the tone, −1 when they all
leave, 0 for no change. ``pre`` is the OFF window (minutes 28-38) unless another pre-window is
set. A cycle where both windows are zero has no index (NaN) and is counted, not dropped silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from buzzphono.metrics.cycles import paired_comparison

INDEX_COLUMN = "attraction_index"


def add_attraction_index(
    cycles: pd.DataFrame, on_col: str = "on_colony_frac_in_zone", pre_col: str = "off_colony_frac_in_zone"
) -> pd.DataFrame:
    """A copy of ``cycles`` with an ``attraction_index`` column (NaN where ON + pre = 0)."""
    out = cycles.copy()
    on = out[on_col].to_numpy(dtype=np.float64)
    pre = out[pre_col].to_numpy(dtype=np.float64)
    denom = on + pre
    with np.errstate(invalid="ignore", divide="ignore"):
        out[INDEX_COLUMN] = np.where(denom > 0, (on - pre) / denom, np.nan)
    return out


def attraction_summary(cycles: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    """Tests of the index against 0, pooled and per phase (same methods as the ON/OFF comparison)."""
    df = cycles.copy()
    df["_zero"] = 0.0
    n_undefined = int(df[INDEX_COLUMN].isna().sum())
    table = paired_comparison(df, "_zero", INDEX_COLUMN, label="attraction_index_vs_0", **kwargs)  # type: ignore[arg-type]
    table["baseline"] = "0"
    table["n_cycles_index_undefined"] = n_undefined
    for col in ("mean_off", "fold_change"):  # not meaningful against a constant 0
        if col in table:
            table = table.drop(columns=col)
    return table
