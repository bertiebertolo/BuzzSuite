"""Exact reproduction of the published (2026 thesis) speaker-zone analysis.

The thesis numbers (Figs 4.8-4.12) came from ``plot_speaker_distance_phonotaxis.py
--resting-only`` (per-cage CSVs) and ``plot_thesis_figures.py`` (figures and statistics). This
module recomputes both from the tracking data with the same definitions, so a published result
can be regenerated at any time and the improved analyses can be compared with it:

* the thesis clock (:class:`buzzphono.timebase.ThesisClock`) and file selection (names without an
  ``-attempt`` suffix);
* the four per-cage CSV files, value for value and byte for byte (LF line endings);
* the pooled per-cycle Wilcoxon test (Fig 4.9) and the per-phase estimator (Fig 4.12), with the
  statistics of SciPy 1.10.1.

Two published details are kept on purpose, because they define the published numbers:

1. The per-phase table averages the per-minute proportions of
   ``resting_speaker_proportion_timeseries.csv`` within each window (minutes with resting
   detections only), whereas the paired test uses the ratio of the capped sums.
2. The per-phase table groups minutes by their clock hour across all cages pooled, so two cages
   recorded at the same hour would be merged. (The thesis cages ran on different dates.)
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from buzzphono.errors import DataError
from buzzphono.geometry import as_polygon
from buzzphono.io.cache import load_npz
from buzzphono.io.discover import ExperimentLayout, discover
from buzzphono.io.ingest import IngestReport, ingest
from buzzphono.io.naming import legacy_recognises
from buzzphono.metrics.resting_fraction import RestingGrids, frame_counts, polygon_test
from buzzphono.stats import WilcoxonResult, wilcoxon_scipy110
from buzzphono.timebase import NS_PER_MIN, ThesisClock, from_ns, to_ns

OFF_WINDOW = (28, 38)
ON_WINDOW = (40, 50)
COLONY = 30
# Penzance photoperiod as published: lights on 05:00, off 17:00, one-hour ramps.
THESIS_PHASES: tuple[tuple[str, int, int], ...] = (
    ("dawn (05-06h)", 5, 6),
    ("day (06-16h)", 6, 16),
    ("dusk (16-17h)", 16, 17),
    ("night (17-05h)", 17, 5),
)
LEGACY_CSVS = (
    "speaker_proportion_by_cycle.csv",
    "resting_speaker_proportion_timeseries.csv",
    "speaker_phase_minute_of_hour.csv",
    "recruitment_resting_fold.csv",
)


def phase_of_hour(hour: int, phases: Iterable[tuple[str, int, int]] = THESIS_PHASES) -> str | None:
    for name, start, end in phases:
        inside = start <= hour < end if start <= end else (hour >= start or hour < end)
        if inside:
            return name
    return None


# ----------------------------------------------------------------------------- per-cage tables
def paired_by_cycle(
    grids: RestingGrids, off: tuple[int, int] = OFF_WINDOW, on: tuple[int, int] = ON_WINDOW
) -> pd.DataFrame:
    """``speaker_proportion_by_cycle.csv``: per hour-cycle OFF and ON proportions (ratio of sums)."""
    off_in, off_tot = grids.in_zone[:, off[0] : off[1]].sum(1), grids.total[:, off[0] : off[1]].sum(1)
    on_in, on_tot = grids.in_zone[:, on[0] : on[1]].sum(1), grids.total[:, on[0] : on[1]].sum(1)
    valid = (off_tot > 0) & (on_tot > 0)
    return pd.DataFrame(
        {
            "cycle": np.flatnonzero(valid),
            "prop_off": off_in[valid] / off_tot[valid],
            "prop_on": on_in[valid] / on_tot[valid],
            "off_total": off_tot[valid],
            "on_total": on_tot[valid],
        }
    )


def minute_of_hour_profile(grids: RestingGrids) -> pd.DataFrame:
    """``speaker_phase_minute_of_hour.csv``: mean ± SEM (ddof 0) of per-cell proportions."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-empty minute columns give NaN, as published
        prop = np.where(grids.total > 0, grids.in_zone / grids.total, np.nan)
        n_per_bin = np.sum(np.isfinite(prop), axis=0)
        mean = np.nanmean(np.where(n_per_bin > 0, prop, np.nan), axis=0)
        sem = np.nanstd(prop, axis=0) / np.sqrt(np.maximum(n_per_bin, 1))
    return pd.DataFrame({"minute": np.arange(60) + 0.5, "mean": mean, "sem": sem, "n_cycles": n_per_bin})


def recruitment(grids: RestingGrids, off: tuple[int, int] = OFF_WINDOW) -> pd.DataFrame:
    """``recruitment_resting_fold.csv``: per-minute fraction minus the cycle's OFF baseline."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frac = np.where(grids.total > 0, grids.in_zone / grids.total, np.nan)
        off_in = grids.in_zone[:, off[0] : off[1]].sum(1)
        off_tot = grids.total[:, off[0] : off[1]].sum(1)
        base = np.where(off_tot > 0, off_in / off_tot, np.nan)
        rec = (frac - base[:, None])[np.isfinite(base)]
        mean = np.nanmean(rec, axis=0)
        k = np.sum(np.isfinite(rec), axis=0)
        sem = np.nanstd(rec, axis=0) / np.sqrt(np.maximum(k, 1))
    return pd.DataFrame({"minute": np.arange(60), "recruit_mean": mean, "recruit_sem": sem})


def minute_series(grids: RestingGrids) -> pd.DataFrame:
    """``resting_speaker_proportion_timeseries.csv``: one row per wall-clock minute (centre time)."""
    n = grids.abs_total.size
    with np.errstate(invalid="ignore", divide="ignore"):
        prop = np.where(grids.abs_total > 0, grids.abs_in / grids.abs_total, np.nan)
    centres = grids.t0_ns + (np.arange(n, dtype=np.int64) * NS_PER_MIN) + NS_PER_MIN // 2
    times = [from_ns(int(t)).strftime("%Y-%m-%d %H:%M:%S") for t in centres]
    return pd.DataFrame({"time": times, "speaker_proportion": prop, "n_resting": grids.abs_total})


def legacy_tables(grids: RestingGrids) -> dict[str, pd.DataFrame]:
    """The four published per-cage CSV tables (those the published code would skip are omitted)."""
    tables: dict[str, pd.DataFrame] = {}
    paired = paired_by_cycle(grids)
    if len(paired) >= 3:
        tables["speaker_proportion_by_cycle.csv"] = paired
    if grids.abs_total.sum() > 0:
        tables["resting_speaker_proportion_timeseries.csv"] = minute_series(grids)
    profile = minute_of_hour_profile(grids)
    if profile["n_cycles"].max() >= 2:
        tables["speaker_phase_minute_of_hour.csv"] = profile
    if int(np.isfinite(_off_baseline(grids)).sum()) >= 3:
        tables["recruitment_resting_fold.csv"] = recruitment(grids)
    return tables


def _off_baseline(grids: RestingGrids) -> np.ndarray:
    off_in = grids.in_zone[:, OFF_WINDOW[0] : OFF_WINDOW[1]].sum(1)
    off_tot = grids.total[:, OFF_WINDOW[0] : OFF_WINDOW[1]].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(off_tot > 0, off_in / off_tot, np.nan)


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = out_dir / name
        table.to_csv(path, index=False, lineterminator="\n")
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------- pooled statistics
@dataclass(frozen=True)
class PhaseResult:
    phase: str
    n: int
    delta_mean: float
    test: WilcoxonResult | None  # one-sided ON > OFF, only when n >= 4 (as published)


def pooled_paired_test(paired_tables: Iterable[pd.DataFrame]) -> WilcoxonResult:
    """Fig 4.9: one-sided Wilcoxon (ON > OFF) on all per-cycle rows of all cages."""
    df = pd.concat(list(paired_tables), ignore_index=True)
    df = df[(df["off_total"] > 0) & (df["on_total"] > 0)]
    return wilcoxon_scipy110(df["prop_on"].to_numpy(), df["prop_off"].to_numpy(), alternative="greater")


def phase_table(minute_tables: Iterable[pd.DataFrame]) -> list[PhaseResult]:
    """Fig 4.12: per photoperiod phase, Δ = mean over cycles of (mean ON minute − mean OFF minute)."""
    df = pd.concat(list(minute_tables), ignore_index=True)
    df = df[df["n_resting"] > 0].copy()
    t = pd.to_datetime(df["time"])
    df["moh"] = t.dt.minute
    df["hour"] = t.dt.hour
    df["cycle_uid"] = t.dt.floor("h").astype(str)
    df["phase"] = df["hour"].map(phase_of_hour)
    on = df[(df["moh"] >= ON_WINDOW[0]) & (df["moh"] < ON_WINDOW[1])]
    off = df[(df["moh"] >= OFF_WINDOW[0]) & (df["moh"] < OFF_WINDOW[1])]
    on_c = on.groupby(["cycle_uid", "phase"])["speaker_proportion"].mean().reset_index(name="on")
    off_c = off.groupby(["cycle_uid", "phase"])["speaker_proportion"].mean().reset_index(name="off")
    cyc = on_c.merge(off_c, on=["cycle_uid", "phase"])
    results = []
    for name, _, _ in THESIS_PHASES:
        sub = cyc[cyc["phase"] == name]
        if sub.empty:
            continue
        test = None
        if len(sub) >= 4:
            test = wilcoxon_scipy110(sub["on"].to_numpy(), sub["off"].to_numpy(), alternative="greater")
        results.append(PhaseResult(name, len(sub), float((sub["on"] - sub["off"]).mean()), test))
    return results


# ----------------------------------------------------------------------------- running it
@dataclass
class ThesisRun:
    """One experiment reproduced with the published method."""

    experiment: Path
    grids: RestingGrids
    tables: dict[str, pd.DataFrame]
    chunks_used: int
    chunks_skipped: list[tuple[str, str]] = field(default_factory=list)  # (chunk, reason)
    ingest: IngestReport | None = None


def load_zone(zones_file: Path, zone: str = "speaker") -> np.ndarray:
    """A polygon from a ``custom_zones.json`` (``{"speaker": {"polygon": [[x, y], ...]}}``)."""
    try:
        zones = json.loads(zones_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataError(
            f"Zone file not found: {zones_file}", hint="Define the speaker zone first (buzzphono calibrate)."
        ) from exc
    except ValueError as exc:
        raise DataError(f"Zone file {zones_file} is not valid JSON ({exc}).") from exc
    if zone not in zones:
        raise DataError(f"Zone '{zone}' is not in {zones_file} (it has: {', '.join(map(str, zones))}).")
    spec = zones[zone]
    vertices = spec.get("polygon") if isinstance(spec, dict) else spec
    try:
        return as_polygon(vertices)
    except ValueError as exc:
        raise DataError(f"Zone '{zone}' in {zones_file}: {exc}.") from exc


def run_thesis(
    experiment_dir: str | Path,
    start: datetime,
    end: datetime,
    *,
    zones_file: Path | None = None,
    zone: str = "speaker",
    cache_root: Path | None = None,
    workers: int | None = None,
    layout: ExperimentLayout | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> ThesisRun:
    """Recompute one experiment's published per-cage tables from its tracking data."""
    exp = Path(experiment_dir)
    layout = layout or discover(exp)
    polygon = load_zone(zones_file or exp / "custom_zones.json", zone)
    report = ingest(
        layout, tiers=("resting",), cache_root=cache_root, workers=workers, progress=progress, cancel=cancel
    )
    t_start, t_end = to_ns(start), to_ns(end)
    grids = RestingGrids.empty(t_start, t_end, COLONY)
    clock = ThesisClock()
    inside = polygon_test(polygon)
    used, skipped = 0, []
    errors = {chunk: msg for chunk, tier, msg in report.errors if tier == "resting"}
    for chunk in layout.chunks:
        path = chunk.files.get("resting")
        if path is None:
            continue
        if not legacy_recognises("resting", path.name):
            skipped.append((chunk.chunk_id, "name not recognised by the published code (-attempt)"))
            continue
        if chunk.chunk_id in errors:
            skipped.append((chunk.chunk_id, errors[chunk.chunk_id]))
            continue
        arrays = load_npz(report.cache_dir / "resting" / f"{chunk.chunk_id}.npz")
        counts = frame_counts(arrays, inside)
        if counts.frames.size == 0:
            continue
        if grids.add(counts, clock.frame_ns(chunk.name, counts.frames), t_start, t_end):
            used += 1
    return ThesisRun(exp, grids, legacy_tables(grids), used, skipped, report)
