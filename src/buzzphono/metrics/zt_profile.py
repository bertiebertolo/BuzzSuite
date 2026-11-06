"""Time-of-day profiles: how the response and the baseline change over the light cycle.

* :func:`hour_of_day_profile` – per clock hour (pooled over days and cages): mean OFF, ON and
  Δ = ON − OFF of the speaker-zone fraction with cage-day cluster-bootstrap 95% intervals, plus flight
  activity; ZT is the hour counted from lights-on.
* :func:`minute_series` – the whole recording minute by minute (speaker-zone fraction, resting count
  and flight), with every stimulus window marked, for the multi-day plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from buzzphono.frames import Binned
from buzzphono.metrics.psth import cluster_band
from buzzphono.timebase import NS_PER_MIN, from_ns


def hour_of_day_profile(
    cycles: pd.DataFrame,
    *,
    lights_on_hour: int,
    on_col: str = "on_colony_frac_in_zone",
    off_col: str = "off_colony_frac_in_zone",
    n_resamples: int = 999,
    seed: int = 20261104,
) -> pd.DataFrame:
    """One row per clock hour with data: n cycles, mean OFF / ON / Δ with 95% cluster intervals."""
    rows = []
    for _, g in cycles.groupby("hour"):
        hour = int(g["hour"].iloc[0])
        clusters = g["cage_day"].to_numpy()
        values = np.stack(
            [
                g[off_col].to_numpy(dtype=np.float64),
                g[on_col].to_numpy(dtype=np.float64),
                (g[on_col] - g[off_col]).to_numpy(dtype=np.float64),
            ],
            axis=1,
        )
        mean, low, high = cluster_band(values, clusters, n_resamples, seed)
        rows.append(
            {
                "hour": hour,
                "zt_hour": (hour - lights_on_hour) % 24,
                "phase": g["phase"].iloc[0] if "phase" in g else "",
                "n_cycles": len(g),
                "n_cage_days": g["cage_day"].nunique(),
                "off_mean": mean[0],
                "off_ci_low": low[0],
                "off_ci_high": high[0],
                "on_mean": mean[1],
                "on_ci_low": low[1],
                "on_ci_high": high[1],
                "delta_mean": mean[2],
                "delta_ci_low": low[2],
                "delta_ci_high": high[2],
            }
        )
    return pd.DataFrame(rows)


def minute_series(
    binned: Binned, *, fps: float, windows: dict[str, tuple[int, int]], zone: str = "speaker"
) -> pd.DataFrame:
    """The whole recording per wall-clock minute, with the stimulus window each minute belongs to."""
    per_min = NS_PER_MIN // binned.bin_ns
    n = binned.n_bins // per_min
    if n == 0:
        return pd.DataFrame()
    z = binned.zone_index(zone)

    def per_minute(a: np.ndarray) -> np.ndarray:
        return a[: n * per_min].reshape(n, per_min).sum(1)

    rec = per_minute(binned.recorded)
    cap_t = per_minute(binned.cap_total)
    cap_i = per_minute(binned.cap_in[:, z])
    rest = per_minute(binned.rest_frames)
    raw_i = per_minute(binned.raw_in[:, z])
    fly = per_minute(binned.fly_px)
    starts = binned.t0_ns + np.arange(n, dtype=np.int64) * NS_PER_MIN
    minute_of_hour = (starts // NS_PER_MIN) % 60
    label = np.full(n, "", dtype=object)
    for name, (a, b) in windows.items():
        label[(minute_of_hour >= a) & (minute_of_hour < b) & (label == "")] = name
    with np.errstate(invalid="ignore", divide="ignore"):
        df = pd.DataFrame(
            {
                "minute_start": [from_ns(int(s)).strftime("%Y-%m-%d %H:%M") for s in starts],
                "minute_start_ns": starts,
                "window": label,
                "frames_recorded": rec,
                "colony_frac_in_zone": np.where(cap_t > 0, cap_i / cap_t, np.nan),
                "rest_in_zone_per_frame": np.where(rest > 0, raw_i / rest, np.nan),
                "fly_px_per_s": np.where(rec > 0, fly / rec * fps, np.nan),
            }
        )
    return df[df["frames_recorded"] > 0].reset_index(drop=True)
