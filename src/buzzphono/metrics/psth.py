"""Peri-stimulus time histograms: occupancy and flight in 30-s bins around every tone onset.

For each hour-cycle, the bins from 12 min before to 20 min after the onset (minute 40 by default)
are cut out of the binned counts; the same is done at the sham onset (minute 20) as a control.
Per bin and quantity the mean over events is given with a 95% band from a cluster bootstrap that
resamples cage-days (neighbouring hours of one cage-day are not independent). Each bin is also
compared with the event's own pre-onset baseline (−10 to −2 min) by a cage-day sign-flip test,
with Benjamini-Hochberg correction across bins.

Quantities (per bin):

* ``colony_frac_in_zone`` – the headline speaker-zone fraction;
* ``rest_in_zone_per_frame`` – mean resting detections in the speaker zone per frame;
* ``fly_px_per_s`` – flight activity, summed fast-step length per second (px/s);
* ``moving_per_frame`` – moving detections per recorded frame.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from buzzphono.frames import Binned
from buzzphono.stats import benjamini_hochberg, block_signflip_test
from buzzphono.timebase import NS_PER_MIN, NS_PER_S

QUANTITIES = ("colony_frac_in_zone", "rest_in_zone_per_frame", "fly_px_per_s", "moving_per_frame")
BASELINE_MIN = (-10.0, -2.0)


def bin_quantities(binned: Binned, fps: float, zone: str = "speaker") -> dict[str, np.ndarray]:
    """Each quantity per bin of the whole experiment (NaN where nothing was recorded)."""
    z = binned.zone_index(zone)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "colony_frac_in_zone": np.where(
                binned.cap_total > 0, binned.cap_in[:, z] / binned.cap_total, np.nan
            ),
            "rest_in_zone_per_frame": np.where(
                binned.rest_frames > 0, binned.raw_in[:, z] / binned.rest_frames, np.nan
            ),
            "fly_px_per_s": np.where(binned.recorded > 0, binned.fly_px / binned.recorded * fps, np.nan),
            "moving_per_frame": np.where(binned.recorded > 0, binned.moving_total / binned.recorded, np.nan),
        }


def event_matrix(
    values: np.ndarray, binned: Binned, onsets_ns: np.ndarray, before_min: float, after_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """(events × relative bins) cut from ``values``; also the relative bin starts in seconds."""
    per_min = NS_PER_MIN // binned.bin_ns
    lo = -round(before_min * per_min)
    hi = round(after_min * per_min)
    rel = np.arange(lo, hi)
    start = (np.asarray(onsets_ns, dtype=np.int64) - binned.t0_ns) // binned.bin_ns
    idx = start[:, None] + rel[None, :]
    ok = (idx >= 0) & (idx < values.size)
    out = np.full(idx.shape, np.nan)
    out[ok] = values[idx[ok]]
    return out, rel * (binned.bin_ns / NS_PER_S)


def cluster_band(
    matrix: np.ndarray, clusters: np.ndarray, n_resamples: int = 999, seed: int = 20261104
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean over events per column and a 95% percentile band from resampling clusters (NaN-aware)."""
    labels, inv = np.unique(clusters, return_inverse=True)
    g = labels.size
    finite = np.isfinite(matrix)
    sums = np.zeros((g, matrix.shape[1]))
    counts = np.zeros((g, matrix.shape[1]))
    np.add.at(sums, inv, np.where(finite, matrix, 0.0))
    np.add.at(counts, inv, finite.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums.sum(0) / counts.sum(0)
        if g < 2:
            nan = np.full(matrix.shape[1], np.nan)
            return mean, nan, nan
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, g, size=(n_resamples, g))
        weights = np.zeros((n_resamples, g))
        np.add.at(weights, (np.repeat(np.arange(n_resamples), g), draws.ravel()), 1.0)
        boot = (weights @ sums) / (weights @ counts)
        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return mean, low, high


def psth(
    binned: Binned,
    cycles: pd.DataFrame,
    *,
    fps: float,
    onset_min: int = 40,
    sham_onset_min: int = 20,
    before_min: float = 12.0,
    after_min: float = 20.0,
    zone: str = "speaker",
    by_phase: bool = True,
    test_bins: bool = True,
    n_resamples: int = 999,
    seed: int = 20261104,
) -> pd.DataFrame:
    """Tidy PSTH table: event, phase, quantity, rel_time_s, mean, ci_low, ci_high, n_events, p, q."""
    quantities = bin_quantities(binned, fps, zone)
    rows: list[pd.DataFrame] = []
    events: Mapping[str, int] = {"onset": onset_min, "sham_onset": sham_onset_min}
    groups: list[tuple[str, np.ndarray]] = [("all", np.ones(len(cycles), dtype=bool))]
    if by_phase and "phase" in cycles:
        groups += [(str(p), (cycles["phase"] == p).to_numpy()) for p in cycles["phase"].unique() if p]
    for event, minute in events.items():
        onsets = cycles["cycle_start_ns"].to_numpy(dtype=np.int64) + minute * NS_PER_MIN
        for qname, values in quantities.items():
            mat, rel_s = event_matrix(values, binned, onsets, before_min, after_min)
            base_cols = (rel_s >= BASELINE_MIN[0] * 60) & (rel_s < BASELINE_MIN[1] * 60)
            for group, mask in groups:
                if not mask.any():
                    continue
                sub = mat[mask]
                clusters = cycles["cage_day"].to_numpy()[mask]
                mean, low, high = cluster_band(sub, clusters, n_resamples, seed)
                table = pd.DataFrame(
                    {
                        "event": event,
                        "phase": group,
                        "quantity": qname,
                        "rel_time_s": rel_s,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_events": np.isfinite(sub).sum(0),
                    }
                )
                if test_bins and group == "all":
                    table["p_vs_baseline"], table["q_bh"] = _bin_tests(
                        sub, base_cols, clusters, n_resamples, seed
                    )
                rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bin_tests(
    mat: np.ndarray, base_cols: np.ndarray, clusters: np.ndarray, n_resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        base = np.nanmean(np.where(base_cols[None, :], mat, np.nan), axis=1) if base_cols.any() else None
    p = np.full(mat.shape[1], np.nan)
    if base is None:
        return p, p.copy()
    _, blocks = np.unique(clusters, return_inverse=True)
    for j in range(mat.shape[1]):
        if base_cols[j]:
            continue
        d = mat[:, j] - base
        ok = np.isfinite(d)
        if ok.sum() >= 3 and np.unique(blocks[ok]).size >= 2:
            p[j] = block_signflip_test(d[ok], blocks[ok], n_resamples=n_resamples, seed=seed).p_value
    q = np.full(p.size, np.nan)
    fin = np.isfinite(p)
    if fin.any():
        q[fin] = benjamini_hochberg(p[fin])
    return p, q
