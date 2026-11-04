"""Hour-cycle tables: every window of every stimulus cycle, with its coverage, as one tidy table.

A *cycle* is one clock hour of one cage. For each cycle and each named window (ON 40-50, OFF 28-38,
the sham windows ...), :func:`cycle_table` gives, per zone:

* ``colony_frac_in_zone`` – the headline: Σ min(n_in, total) / Σ min(n_rest, colony) over frames;
* ``resting_frac_in_zone`` – the uncapped Σ n_in / Σ n_rest;
* ``mean_frame_frac`` – the mean over frames of the capped per-frame fraction;
* ``in_zone_per_frame`` – mean resting detections in the zone per frame with resting detections;

and, per window, its coverage: the fraction of the window's frames that were recorded, the longest
run of empty 30-s bins, the minutes with resting detections and whether the coverage rule passes.
The paired comparisons of :func:`paired_comparison` then use one row per cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.frames import Binned
from buzzphono.io.playback_log import PlaybackLog, window_status
from buzzphono.stats import ASSUMPTIONS, Alternative, holm, summarise_paired
from buzzphono.timebase import NS_PER_HOUR, NS_PER_MIN, NS_PER_S, from_ns

METRICS = ("colony_frac_in_zone", "resting_frac_in_zone", "mean_frame_frac", "in_zone_per_frame")
EXCLUDED_STIMULUS = ("failed", "not played")


@dataclass(frozen=True)
class Coverage:
    """The default-mode coverage rule for one window (see ``CoverageRule`` in the settings)."""

    min_frame_fraction: float = 0.8
    max_gap_s: float = 120.0
    min_minutes_with_resting: int = 8
    enabled: bool = True


def _runs_of_zeros(a: np.ndarray) -> int:
    """Length of the longest run of zeros along the last axis."""
    best = np.zeros(a.shape[:-1], dtype=np.int64)
    cur = np.zeros(a.shape[:-1], dtype=np.int64)
    for k in range(a.shape[-1]):
        cur = np.where(a[..., k] == 0, cur + 1, 0)
        best = np.maximum(best, cur)
    return best  # type: ignore[return-value]


def cycle_table(
    binned: Binned,
    windows: Mapping[str, tuple[int, int]],
    *,
    fps: float,
    cage: str,
    phases: Mapping[int, str] | None = None,
    coverage: Coverage | None = None,
    playback: PlaybackLog | None = None,
    stimulus_window: str = "on",
) -> pd.DataFrame:
    """One row per hour-cycle with recorded frames; columns ``<window>_<metric>[__<zone>]``.

    The speaker zone's columns have no ``__zone`` suffix when the zone is called ``speaker``; the
    other zones' columns end in ``__<zone name>``.
    """
    coverage = coverage or Coverage()
    bph = binned.bins_per_hour
    n_cycles = binned.n_bins // bph
    if n_cycles == 0:
        return pd.DataFrame()
    per_min = bph // 60 if bph >= 60 else 1
    shape = (n_cycles, bph)

    def grid(a: np.ndarray) -> np.ndarray:
        return a[: n_cycles * bph].reshape(shape + a.shape[1:])

    rec, rest = grid(binned.recorded), grid(binned.rest_frames)
    cap_t, raw_t = grid(binned.cap_total), grid(binned.raw_total)
    cap_i, raw_i, fsum = grid(binned.cap_in), grid(binned.raw_in), grid(binned.frac_sum)
    starts = binned.t0_ns + np.arange(n_cycles, dtype=np.int64) * NS_PER_HOUR
    out: dict[str, Any] = {
        "cage": cage,
        "cycle_start_ns": starts,
        "cycle_start": [from_ns(int(s)).strftime("%Y-%m-%d %H:%M") for s in starts],
    }
    hours = np.array([from_ns(int(s)).hour for s in starts])
    dates = [from_ns(int(s)).strftime("%Y-%m-%d") for s in starts]
    out["hour"] = hours
    out["date"] = dates
    out["cage_day"] = [f"{cage} {d}" for d in dates]
    out["phase"] = [phases.get(int(h), "") if phases else "" for h in hours]
    any_recorded = rec.sum(1) > 0
    bin_s = binned.bin_ns / NS_PER_S
    for name, (a, b) in windows.items():
        lo, hi = a * per_min, b * per_min
        expected = (b - a) * 60 * fps
        r = rec[:, lo:hi]
        out[f"{name}_frames_recorded"] = r.sum(1)
        out[f"{name}_frame_fraction"] = r.sum(1) / expected
        out[f"{name}_max_gap_s"] = _runs_of_zeros(r) * bin_s
        minutes = cap_t[:, lo:hi].reshape(n_cycles, b - a, per_min).sum(2)
        out[f"{name}_minutes_with_resting"] = (minutes > 0).sum(1)
        ok = np.ones(n_cycles, dtype=bool)
        if coverage.enabled:
            ok &= out[f"{name}_frame_fraction"] >= coverage.min_frame_fraction
            ok &= out[f"{name}_max_gap_s"] <= coverage.max_gap_s
            ok &= out[f"{name}_minutes_with_resting"] >= coverage.min_minutes_with_resting
        ok &= cap_t[:, lo:hi].sum(1) > 0
        out[f"{name}_ok"] = ok
        rest_f = rest[:, lo:hi].sum(1)
        for z, zone in enumerate(binned.zones):
            suffix = "" if zone == "speaker" else f"__{zone}"
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{name}_colony_frac_in_zone{suffix}"] = cap_i[:, lo:hi, z].sum(1) / cap_t[:, lo:hi].sum(
                    1
                )
                out[f"{name}_resting_frac_in_zone{suffix}"] = raw_i[:, lo:hi, z].sum(1) / raw_t[:, lo:hi].sum(
                    1
                )
                out[f"{name}_mean_frame_frac{suffix}"] = fsum[:, lo:hi, z].sum(1) / rest_f
                out[f"{name}_in_zone_per_frame{suffix}"] = raw_i[:, lo:hi, z].sum(1) / rest_f
    df = pd.DataFrame(out)[any_recorded].reset_index(drop=True)
    first = int(df["cycle_start_ns"].min()) if len(df) else 0
    df.insert(4, "hour_index", ((df["cycle_start_ns"] - first) // NS_PER_HOUR).astype(np.int64))
    if playback is not None and stimulus_window in windows:
        a, b = windows[stimulus_window]
        status = [
            window_status(playback, from_ns(int(s) + a * NS_PER_MIN), from_ns(int(s) + b * NS_PER_MIN))
            for s in df["cycle_start_ns"]
        ]
        df["stimulus_status"] = [s for s, _ in status]
        df["stimulus_played_frac"] = [f for _, f in status]
    else:
        df["stimulus_status"] = "not checked (no playback log)"
        df["stimulus_played_frac"] = np.nan
    return df


def zone_column(window: str, metric: str, zone: str = "speaker") -> str:
    return f"{window}_{metric}" + ("" if zone == "speaker" else f"__{zone}")


def usable(cycles: pd.DataFrame, windows: tuple[str, ...]) -> pd.Series:
    """Cycles whose windows all pass coverage and whose stimulus was not known to have failed."""
    ok = pd.Series(True, index=cycles.index)
    for w in windows:
        ok &= cycles[f"{w}_ok"].astype(bool)
    if "stimulus_status" in cycles:
        ok &= ~cycles["stimulus_status"].isin(EXCLUDED_STIMULUS)
    return ok


# ----------------------------------------------------------------------------- paired comparisons
def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}_{k}" if prefix else str(k), v, out)
    elif isinstance(value, (tuple, list)):
        out[prefix] = "; ".join(map(str, value))
    else:
        out[prefix] = value


def flatten_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """A :func:`buzzphono.stats.summarise_paired` result as one flat row (``wilcoxon_p_value`` ...)."""
    row: dict[str, Any] = {}
    for key, value in summary.items():
        if key == "effect":
            _flatten("", value, row)
        elif value is not None:
            _flatten(key, value, row)
    return row


def paired_comparison(
    cycles: pd.DataFrame,
    a_col: str,
    b_col: str,
    *,
    label: str,
    alternative: Alternative = "two-sided",
    block_hours: int = 12,
    n_resamples: int = 9_999,
    seed: int = 20261104,
    with_mixed_model: bool = True,
    by_phase: bool = True,
) -> pd.DataFrame:
    """Every pre-registered test of ``b − a`` per cycle, pooled and per phase (Holm across phases).

    ``a_col`` is the baseline (e.g. OFF) and ``b_col`` the test window (e.g. ON). Rows with a
    missing value in either column are dropped and counted.
    """
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", cycles)]
    if by_phase and "phase" in cycles and cycles["phase"].astype(bool).any():
        groups += [(str(p), g) for p, g in cycles.groupby("phase", sort=False) if p]
    for group, df in groups:
        a = df[a_col].to_numpy(dtype=np.float64)
        b = df[b_col].to_numpy(dtype=np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        row: dict[str, Any] = {
            "comparison": label,
            "baseline": a_col,
            "test": b_col,
            "phase": group,
            "alternative": alternative,
            "n_cycles_dropped_missing": int((~finite).sum()),
            "n_cages": int(df.loc[finite, "cage"].nunique()) if len(df) else 0,
            "n_cage_days": int(df.loc[finite, "cage_day"].nunique()) if len(df) else 0,
        }
        if finite.sum() >= 2:
            summary = summarise_paired(
                a[finite],
                b[finite],
                df["cage_day"].to_numpy()[finite],
                df["hour_index"].to_numpy()[finite],
                cage=df["cage"].to_numpy()[finite] if len(set(df["cage"])) > 1 else None,
                phase=None,
                alternative=alternative,
                block_hours=block_hours,
                n_resamples=n_resamples,
                seed=seed,
                with_mixed_model=bool(with_mixed_model and finite.sum() >= 6),
            )
            row.update(flatten_summary(summary))
        rows.append(row)
    table = pd.DataFrame(rows)
    phase_rows = table["phase"] != "all"
    for test in ("wilcoxon_p_value", "block_signflip_p_value", "wild_cluster_p_value"):
        if test in table and phase_rows.sum() > 1:
            p = table.loc[phase_rows, test].to_numpy(dtype=np.float64)
            adj = np.full(p.size, np.nan)
            fin = np.isfinite(p)
            adj[fin] = holm(p[fin])
            table.loc[phase_rows, test.replace("_p_value", "_p_holm_phases")] = adj
    return table


def assumptions_table() -> pd.DataFrame:
    return pd.DataFrame({"method": list(ASSUMPTIONS), "assumptions": list(ASSUMPTIONS.values())})


def add_control_contrast(
    cycles: pd.DataFrame, binned: Binned, windows: tuple[str, ...], controls: list[str]
) -> pd.DataFrame:
    """A copy with, per window, ``<w>_control_area_matched`` – the mean control-zone fraction scaled to
    the speaker zone's area – and per cycle ``delta_speaker``, ``delta_control`` (ON − OFF of each)."""
    out = cycles.copy()
    if not controls or not len(out):
        return out
    area = dict(zip(binned.zones, binned.zone_area_px, strict=True))
    for w in windows:
        cols = [out[zone_column(w, "colony_frac_in_zone", c)] * area["speaker"] / area[c] for c in controls]
        out[f"{w}_control_area_matched"] = np.mean(np.vstack([c.to_numpy(dtype=float) for c in cols]), axis=0)
    if {"on", "off"} <= set(windows):
        out["delta_speaker"] = out["on_colony_frac_in_zone"] - out["off_colony_frac_in_zone"]
        out["delta_control"] = out["on_control_area_matched"] - out["off_control_area_matched"]
    return out
