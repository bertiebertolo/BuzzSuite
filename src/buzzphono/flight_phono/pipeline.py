"""Run the flight analyses on one experiment: per-chunk arrays -> clock -> per-minute -> cycles -> statistics.

Two time bases:

* ``legacy=True`` reproduces the original flight package exactly: every chunk is ``nf`` frames
  (the largest of three sampled chunks), chunk *k* starts ``k × nf`` frames into its session, the
  clock is fitted on the bursts inside the analysis window, sessions with fewer than 3 bursts of
  their own are left out, and times are rounded as pandas 2.0 did.
* ``legacy=False`` (default) uses BuzzPhono's corrected clock (see :mod:`buzzphono.frames`):
  every chunk's real frame count, end-aligned short first chunks, and the median offset for
  sessions with too few bursts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.calibration import Calibration
from buzzphono.errors import Cancelled, DataError
from buzzphono.flight_phono import build as fb
from buzzphono.flight_phono.extract import chunk_arrays, nmax_for
from buzzphono.flight_phono.geometry import FlightGeometry, build_geometry, role_order
from buzzphono.frames import experiment_clock
from buzzphono.io.cache import CacheIndex, load_npz
from buzzphono.io.discover import ExperimentLayout
from buzzphono.io.naming import legacy_recognises
from buzzphono.stimulus import burst_onsets, burst_series, fit_clock
from buzzphono.timebase import (
    NS_PER_S,
    ClockModel,
    CorrectedClock,
    SessionFrames,
    legacy_seconds_to_ns,
    to_ns,
)

FPS = 25.0
CHUNK_MARGIN_H = 2.0


@dataclass
class FlightResult:
    label: str
    window: tuple[pd.Timestamp, pd.Timestamp]
    nf: int | dict[str, int]
    clock: ClockModel
    bursts: pd.DataFrame
    role_zones: list[str]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    chunks_used: int = 0
    problems: list[tuple[str, str]] = field(default_factory=list)


def geometry_from_calibration(cal: Calibration, params: fb.FlightParams) -> FlightGeometry:
    zones: dict[str, Any] = {"speaker": cal.speaker.polygon}
    for z in cal.zones:
        if z.role != "speaker":
            zones[z.name] = z.polygon
    return build_geometry(cal.cage, zones, params.r_fracs())


def _legacy_nf(layout: ExperimentLayout, index: CacheIndex, chunks: list[tuple[str, str, int]]) -> int:
    """Largest tracker frame count of three chunks a quarter, half and three quarters through."""
    n = len(chunks)
    sample = sorted({chunks[n // 4][0], chunks[n // 2][0], chunks[(3 * n) // 4][0]})
    lens = []
    for stem in sample:
        e = index.entry(stem, "tracks")
        if e and e.get("status") == "ok" and isinstance(e.get("nf"), int):
            lens.append(int(e["nf"]))
    if lens and max(lens) > 1000:
        return max(lens)
    mx = max(
        (
            int(e["max_frame"])
            for s in sample
            if (e := index.entry(s, "moving")) and e.get("max_frame") is not None
        ),
        default=-1,
    )
    if mx > 1000:
        return mx + 1
    raise DataError("Could not work out the frames per chunk from the tracking data.")


def run_flight(
    layout: ExperimentLayout,
    cache_dir: Path,
    cal: Calibration,
    *,
    params: fb.FlightParams | None = None,
    legacy: bool = False,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    label: str | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> FlightResult:
    """The whole single-experiment flight analysis (the cache must hold resting, moving and tracks)."""
    params = params or fb.FlightParams()
    label = label or layout.root.name
    geom = geometry_from_calibration(cal, params)
    order, roles = role_order(geom)
    index = CacheIndex.open(cache_dir, layout.root)

    def say(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    files = sorted(
        (
            c
            for c in layout.chunks
            if "moving" in c.files and (not legacy or legacy_recognises("moving", c.files["moving"].name))
        ),
        key=lambda c: c.files["moving"].name,
    )
    if not files:
        raise DataError(f"No moving-tracking files in {layout.root}.", hint="Track the experiment first.")
    sess_key = {c.chunk_id: f"{c.name.date}_{c.name.time}" if legacy else c.name.session for c in files}
    chunks_all = [(c.chunk_id, sess_key[c.chunk_id], c.name.index) for c in files]
    starts_ns = {c.chunk_id: to_ns(begin) for c in files if (begin := c.name.session_start()) is not None}

    eclock = None
    nf: int | dict[str, int]
    nf_int = 0
    if legacy:
        nf_int = _legacy_nf(layout, index, chunks_all)
        nf = nf_int
        nmax = nmax_for(nf_int)
        spans = [
            (
                starts_ns[s] + round(k * nf_int / FPS * NS_PER_S),
                starts_ns[s] + round((k + 1) * nf_int / FPS * NS_PER_S),
            )
            for s, _, k in chunks_all
        ]
    else:
        eclock = experiment_clock(layout, cache_dir, mode="nominal", fps=FPS)
        nf_map = {s: eclock.nf[s][0] for s, _, _ in chunks_all if s in eclock.nf}
        nf = nf_map
        chunks_all = [c for c in chunks_all if c[0] in nf_map]
        nmax = nmax_for(max(nf_map.values()))
        spans = []
        for s, _, _ in chunks_all:
            t = eclock.clock.frame_ns(
                next(c.name for c in files if c.chunk_id == s), np.array([0, nf_map[s]])
            )
            spans.append((int(t[0]), int(t[1])))
    data_start = pd.Timestamp(min(a for a, _ in spans)).floor("h")
    data_end = pd.Timestamp(max(b for _, b in spans)).ceil("h")
    w0, w1 = start or data_start, end or data_end
    if w1 <= w0:
        raise DataError(f"The analysis end ({w1}) must be after its start ({w0}).")
    margin = pd.Timedelta(hours=CHUNK_MARGIN_H)
    chunks = [
        c
        for c, (a, b) in zip(chunks_all, spans, strict=True)
        if pd.Timestamp(b) >= w0 - margin and pd.Timestamp(a) <= w1 + margin
    ]

    # ---- per-chunk arrays from the cache
    arrays: dict[str, dict[str, Any]] = {}
    problems: list[tuple[str, str]] = []
    for i, (stem, _, _) in enumerate(chunks):
        if cancel is not None and cancel():
            raise Cancelled("Flight analysis cancelled.")
        say(0.7 * i / max(len(chunks), 1), f"Flight arrays: chunk {i + 1} of {len(chunks)}")

        def tier(name: str, stem: str = stem) -> dict[str, np.ndarray] | None:
            p = cache_dir / name / f"{stem}.npz"
            e = index.entry(stem, name)
            return load_npz(p) if p.exists() and e is not None and e.get("status") == "ok" else None

        mv, rs, tr = tier("moving"), tier("resting"), tier("tracks")
        if mv is None:
            problems.append((stem, "moving cache missing or unreadable"))
        if tr is None:
            problems.append((stem, "no assembled tracks (final_tracking_data)"))
        arrays[stem] = chunk_arrays(mv, rs, tr, geom, nmax, params.flyer_mult, params.min_median)

    def load(kind: str, stem: str) -> dict[str, np.ndarray] | None:
        return arrays.get(stem, {}).get(kind)

    # ---- clock
    say(0.72, "Calibrating the recording clock on the tone-onset flight burst")
    by_session: dict[str, dict[int, str]] = {}
    for stem, sess, k in chunks:
        by_session.setdefault(sess, {})[k] = stem
    if legacy:
        rows: list[dict[str, Any]] = []
        sessions: dict[str, SessionFrames] = {}
        for sess, by_k in sorted(by_session.items()):
            s0 = starts_ns[by_k[min(by_k)]]
            sf = SessionFrames(sess, s0, nf_int, dict.fromkeys(by_k, nf_int), "legacy", FPS)
            sessions[sess] = sf
            if (
                pd.Timestamp(s0 + round((max(by_k) + 1) * nf_int / FPS * NS_PER_S)) < w0
                or pd.Timestamp(s0) > w1
            ):
                continue
            activity = {
                k: m["fly_px"][:nf_int] for k, s in by_k.items() if (m := load("main", s)) is not None
            }
            rows += burst_onsets(burst_series(sf, activity), sf, w0.value, w1.value, params.on[0])
        bursts = pd.DataFrame(
            rows, columns=["session", "hour_ns", "onset_recording_s", "offset_s", "days", "peak_ratio"]
        )
        fit = fit_clock(bursts, sorted(by_session))
        model = fit.model
        a = (
            {s: model.offset_s[s] for s in model.offset_s if model.source.get(s) == "fitted"}
            if model.fitted
            else dict.fromkeys(by_session, 0.0)
        )

        def frame_times(sess: str, k: int, frames: np.ndarray) -> pd.DatetimeIndex | None:
            if sess not in a:
                return None
            secs = (k * nf_int + np.asarray(frames, float)) / FPS
            s0 = sessions[sess].start_ns
            return pd.DatetimeIndex(
                s0 + legacy_seconds_to_ns(secs - a[sess] - model.drift_s_per_day * secs / 86400.0)
            )

    else:
        corrected = experiment_clock(layout, cache_dir, mode="corrected", cage=cal.cage_polygon(), fps=FPS)
        clock = corrected.clock
        assert isinstance(clock, CorrectedClock)
        model = clock.model
        bursts = corrected.fit.bursts if corrected.fit is not None else pd.DataFrame()
        names = {c.chunk_id: c.name for c in files}
        stem_of = {(sess, k): stem for stem, sess, k in chunks}

        def frame_times(sess: str, k: int, frames: np.ndarray) -> pd.DatetimeIndex | None:
            return pd.DatetimeIndex(clock.frame_ns(names[stem_of[(sess, k)]], frames))

    # ---- per-minute, cycles, statistics
    say(0.8, "Per-minute flight metrics")
    window = (w0, w1)
    acc = fb.accumulate(load, chunks, frame_times, order, params, window, nf, nmax, geom.s, cancel)
    if acc["chunks_used"] == 0:
        raise DataError("No chunk could be placed in the analysis window on the calibrated clock.")
    cm = fb.cycle_metrics(acc, params)
    st = fb.stat_table(cm, label, params)
    say(0.88, "Take-off and landing links")
    lacc = fb.accumulate_links(
        load, chunks, frame_times, order, geom, params, window, nf, acc["minutes"].time, cancel
    )
    lcm = fb.links_cycle_metrics(lacc, params)
    lst = fb.stat_table(lcm, label, params)
    say(0.93, "Distance from each wall")
    dacc = fb.accumulate_dist(load, chunks, frame_times, order, window, nf, cancel)
    dcm = fb.dist_cycle_metrics(dacc, params)
    dst = fb.stat_table(dcm, label, params)
    traj = fb.trajectory_pass_summary(load, chunks, frame_times, order, geom, params, window, cancel)
    say(1.0, "Flight analysis done")
    tables = {
        "flight_per_minute": acc["minutes"].drop(columns=["hour", "moh"]),
        "flight_cycle_metrics": cm,
        "links_cycle_metrics": lcm,
        "distance_cycle_metrics": dcm,
        "trajectory_pass_summary": traj,
        "stats_flight_on_vs_off": st,
        "stats_links_on_vs_off": lst,
        "stats_distance_on_vs_off": dst,
    }
    return FlightResult(label, window, nf, model, bursts, roles, tables, acc["chunks_used"], problems)
