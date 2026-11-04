"""Directed flight: do flying mosquitoes move towards the speaker when the tone plays?

Two levels, both from the moving tracks of *flyers* (at least 10 points, mean step at least
4 px/frame × camera scale – the thesis rule):

1. **Radial velocity** (needs no track identity beyond consecutive points; failed the control): for every
   flyer step, the speed towards each zone's centre (positive = approaching). Per window it is
   averaged over all flyer steps; the directed-flight measure is the speaker's value minus the
   mean of the control zones', so whole-cage arousal cancels.
2. **Approach bouts**: along each flyer track the distance to the speaker zone (0 inside it),
   median-filtered over 5 frames, is split at turning points with a hysteresis of ``jitter`` px
   (4 × the median absolute deviation of resting position jitter, at least 2 px). A bout is a
   fall from a local maximum to the next local minimum of at least ``max(min_drop, 15% of the
   cage width)`` lasting at least 0.4 s. Per bout: start time, closing distance, duration, speed,
   the minimum distance reached and whether it entered the zone.

Distances are in millimetres when the calibration gives the cage's physical size, otherwise in
pixels (the column names say which). Earlier per-track approach measures failed the positive
control, so every output carries the verdict in :data:`VALIDATION` from the Aedes positive-control
check: approach bouts passed it, radial velocity did not.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from buzzphono.errors import Cancelled
from buzzphono.flight import FLYER_STEP, MIN_FLYER_POINTS, camera_scale
from buzzphono.frames import ExperimentClock
from buzzphono.geometry import centroid, distance_to_polygon
from buzzphono.io.cache import load_npz
from buzzphono.io.discover import ExperimentLayout
from buzzphono.metrics.spatial import window_labels
from buzzphono.timebase import NS_PER_S

# Outcome of the positive-control check on the three Aedes thesis cages (475 Hz males, 2026-11-04).
# Radial velocity came out negative for ON vs OFF and also changed in the sham comparison: it is
# dominated by take-offs from the speaker zone, so it does not measure approach. Approach bouts per
# flight-second rose during the tone with a null sham. Approaches to the other walls are not
# compared, so a general rise in tortuous flight is not excluded.
VALIDATION: dict[str, str] = {
    "radial_velocity": "not validated (failed the Aedes positive control)",
    "approach_bouts": "passed the Aedes positive control (ON > OFF, sham null)",
}


@dataclass(frozen=True)
class ApproachSettings:
    min_points: int = MIN_FLYER_POINTS
    smoothing_frames: int = 5
    min_drop_mm: float = 20.0
    min_drop_cage_fraction: float = 0.15
    min_duration_s: float = 0.4


def resting_jitter_px(resting: Mapping[str, np.ndarray]) -> float:
    """4 × MAD of the position change between consecutive runs of one resting object (px)."""
    obj = resting["run_obj"]
    if obj.size < 3:
        return 2.0
    same = (obj[1:] == obj[:-1]) & (resting["run_f0"][1:] == resting["run_f0"][:-1] + resting["run_len"][:-1])
    d = np.hypot(*(resting["run_xy"][1:] - resting["run_xy"][:-1]).T)[same]
    if d.size < 3:
        return 2.0
    mad = float(np.median(np.abs(d - np.median(d)))) * 1.4826
    return max(4.0 * mad, 2.0)


def _flyer_tracks(
    moving: Mapping[str, np.ndarray], scale: float, min_points: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    own = moving["det_obj"]
    if own.size == 0:
        return []
    frames = moving["det_frame"]
    xy = moving["det_xy"]
    cuts = np.flatnonzero(np.diff(own)) + 1
    out = []
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, own.size], strict=True):
        if hi - lo < min_points:
            continue
        p = xy[lo:hi]
        step = np.hypot(*np.diff(p, axis=0).T)
        if step.mean() >= FLYER_STEP * scale:
            out.append((frames[lo:hi], p))
    return out


def turning_points(d: np.ndarray, hysteresis: float) -> list[tuple[int, int]]:
    """(start, end) index pairs of every fall from a local maximum to the next local minimum,
    turning only when the signal moves ``hysteresis`` away from the current extreme."""
    bouts: list[tuple[int, int]] = []
    if d.size < 2:
        return bouts
    falling = False
    ext_i = 0
    peak_i = 0
    for i in range(1, d.size):
        if not falling:
            if d[i] > d[ext_i]:
                ext_i = i
            elif d[ext_i] - d[i] >= hysteresis:
                falling, peak_i, ext_i = True, ext_i, i
        else:
            if d[i] < d[ext_i]:
                ext_i = i
            elif d[i] - d[ext_i] >= hysteresis:
                bouts.append((peak_i, ext_i))
                falling, ext_i = False, i
    if falling:
        bouts.append((peak_i, ext_i))
    return bouts


@dataclass
class ApproachResult:
    unit: str  # "mm" or "px"
    jitter: float
    min_drop: float
    bouts: pd.DataFrame  # one row per accepted approach bout
    steps: pd.DataFrame  # per cycle × window: radial-velocity sums and flight seconds


def approach_analysis(
    layout: ExperimentLayout,
    cache_dir: Path,
    eclock: ExperimentClock,
    usable_starts: np.ndarray,
    windows: Mapping[str, tuple[int, int]],
    *,
    speaker: np.ndarray,
    controls: Mapping[str, np.ndarray],
    cage: np.ndarray,
    to_mm: Callable[[np.ndarray], np.ndarray] | None = None,
    settings: ApproachSettings | None = None,
    fps: float = 25.0,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> ApproachResult:
    settings = settings or ApproachSettings()
    scale = camera_scale(cage)
    unit = "mm" if to_mm is not None else "px"

    def conv(xy: np.ndarray) -> np.ndarray:
        return to_mm(xy) if to_mm is not None else xy

    cage_c = conv(np.asarray(cage, dtype=np.float64))
    cage_width = float(0.5 * (np.hypot(*(cage_c[1] - cage_c[0])) + np.hypot(*(cage_c[2] - cage_c[3]))))
    min_drop_abs = (
        settings.min_drop_mm if unit == "mm" else settings.min_drop_mm / 0.4
    )  # 0.4 mm/px if unknown
    min_drop = max(min_drop_abs, settings.min_drop_cage_fraction * cage_width)
    zones = {"speaker": speaker, **controls}
    centres = {k: conv(centroid(np.asarray(v, dtype=np.float64))[None, :])[0] for k, v in zones.items()}
    speaker_c = conv(np.asarray(speaker, dtype=np.float64))
    starts = np.asarray(usable_starts, dtype=np.int64)
    names = list(windows)
    # resting position jitter, from up to 10 chunks spread over the recording
    sample = layout.chunks[:: max(1, len(layout.chunks) // 10)]
    jitters = [
        resting_jitter_px(load_npz(p))
        for c in sample
        if (p := cache_dir / "resting" / f"{c.chunk_id}.npz").exists()
    ]
    jitter_px = float(np.median(jitters)) if jitters else 2.0
    jitter = jitter_px if unit == "px" else jitter_px * _mm_per_px(cage, cage_c)
    bout_rows: list[dict[str, Any]] = []
    step_rows: list[pd.DataFrame] = []
    n = len(layout.chunks)
    for i, c in enumerate(layout.chunks):
        if cancel is not None and cancel():
            raise Cancelled("Cancelled during the approach analysis.")
        if progress is not None:
            progress(i / max(n, 1), f"Approach: chunk {i + 1} of {n}")
        mp = cache_dir / "moving" / f"{c.chunk_id}.npz"
        if c.chunk_id not in eclock.nf or not mp.exists():
            continue
        for frames, xy_px in _flyer_tracks(load_npz(mp), scale, settings.min_points):
            t = eclock.clock.frame_ns(c.name, frames)
            xy = conv(xy_px)
            dt = np.diff(t) / NS_PER_S
            dt[dt <= 0] = 1.0 / fps
            lab = window_labels(t[1:], windows, starts)
            keep = lab >= 0
            if keep.any():
                cyc = (t[1:] - t[1:] % (3600 * NS_PER_S))[keep]
                rec: dict[str, Any] = {
                    "cycle_start_ns": cyc,
                    "window": np.array(names)[lab[keep]],
                    "flight_s": dt[keep],
                }
                for z, cz in centres.items():
                    r = np.hypot(*(xy - cz).T)
                    rec[f"radial_sum__{z}"] = -np.diff(r)[keep]  # Σ v·dt: distance closed towards the zone
                step_rows.append(pd.DataFrame(rec))
            d = distance_to_polygon(speaker_c, xy) if unit == "mm" else distance_to_polygon(speaker, xy_px)
            if d.size >= settings.smoothing_frames:
                d = median_filter(d, size=settings.smoothing_frames, mode="nearest")
            bout_rows.extend(
                _bouts(d, t, c.chunk_id, windows, starts, names, min_drop, settings, jitter, unit)
            )
    bouts = pd.DataFrame(bout_rows)
    steps = pd.concat(step_rows, ignore_index=True) if step_rows else pd.DataFrame()
    if len(steps):
        steps = steps.groupby(["cycle_start_ns", "window"], as_index=False).sum()
    if progress is not None:
        progress(1.0, "Approach done")
    return ApproachResult(unit, jitter, min_drop, bouts, steps)


def _mm_per_px(cage_px: np.ndarray, cage_mm: np.ndarray) -> float:
    w_px = 0.5 * (np.hypot(*(cage_px[1] - cage_px[0])) + np.hypot(*(cage_px[2] - cage_px[3])))
    w_mm = 0.5 * (np.hypot(*(cage_mm[1] - cage_mm[0])) + np.hypot(*(cage_mm[2] - cage_mm[3])))
    return float(w_mm / w_px)


def _bouts(
    d: np.ndarray,
    t: np.ndarray,
    chunk: str,
    windows: Mapping[str, tuple[int, int]],
    starts: np.ndarray,
    names: list[str],
    min_drop: float,
    settings: ApproachSettings,
    jitter: float,
    unit: str,
) -> list[dict[str, Any]]:
    rows = []
    for a, b in turning_points(d, jitter):
        drop = float(d[a] - d[b])
        dur = (t[b] - t[a]) / NS_PER_S
        if drop < min_drop or dur < settings.min_duration_s:
            continue
        lab = int(window_labels(t[a : a + 1], windows, starts)[0])
        if lab < 0:
            continue
        rows.append(
            {
                "chunk": chunk,
                "cycle_start_ns": int(t[a] - t[a] % (3600 * NS_PER_S)),
                "window": names[lab],
                "start_ns": int(t[a]),
                "duration_s": dur,
                f"drop_{unit}": drop,
                f"speed_{unit}_per_s": drop / dur if dur > 0 else np.nan,
                f"min_distance_{unit}": float(d[b]),
                "entered_zone": bool(d[b] <= 0),
            }
        )
    return rows


def approach_cycle_table(result: ApproachResult, cycles: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    """Per usable cycle and window: flight seconds, radial velocity per zone and the directed measure,
    approach bouts per flight-second, mean speed and minimum distance, zone entries per minute."""
    u = result.unit
    if result.steps.empty:
        return pd.DataFrame()
    s = result.steps.copy()
    for col in [c for c in s.columns if c.startswith("radial_sum__")]:
        s[col.replace("radial_sum__", f"radial_velocity_{u}_per_s__")] = s[col] / s["flight_s"]
    ctrl_cols = [f"radial_velocity_{u}_per_s__{c}" for c in controls]
    s[f"directed_velocity_{u}_per_s"] = s[f"radial_velocity_{u}_per_s__speaker"] - (
        s[ctrl_cols].mean(axis=1) if ctrl_cols else 0.0
    )
    b = result.bouts
    if len(b):
        agg = b.groupby(["cycle_start_ns", "window"]).agg(
            n_bouts=("start_ns", "size"),
            mean_bout_speed=(f"speed_{u}_per_s", "mean"),
            mean_min_distance=(f"min_distance_{u}", "mean"),
            n_entries=("entered_zone", "sum"),
        )
        s = s.merge(agg.reset_index(), on=["cycle_start_ns", "window"], how="left")
        s[["n_bouts", "n_entries"]] = s[["n_bouts", "n_entries"]].fillna(0)
    else:
        s["n_bouts"] = 0
        s["n_entries"] = 0
    s["bouts_per_flight_s"] = s["n_bouts"] / s["flight_s"]
    meta = cycles[["cycle_start_ns", "cage", "cage_day", "hour_index", "phase"]].drop_duplicates(
        "cycle_start_ns"
    )
    out = s.merge(meta, on="cycle_start_ns", how="inner")
    out["validated_radial_velocity"] = VALIDATION["radial_velocity"]
    out["validated_bouts"] = VALIDATION["approach_bouts"]
    return out.drop(columns=[c for c in out.columns if c.startswith("radial_sum__")])


def wide_by_window(table: pd.DataFrame, value: str) -> pd.DataFrame:
    """One row per cycle with ``<window>_<value>`` columns, ready for paired comparisons."""
    w = table.pivot_table(
        index=["cycle_start_ns", "cage", "cage_day", "hour_index", "phase"], columns="window", values=value
    )
    w.columns = [f"{c}_{value}" for c in w.columns]
    return w.reset_index()
