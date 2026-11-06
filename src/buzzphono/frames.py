"""Per-frame counts of one experiment, summed into short time bins: the input of every window analysis.

Every chunk's frames are placed on the wall clock by the chosen time base, and each frame's counts
are added to a 30-second bin:

* recorded frames (from the chunk's frame count, whether or not anything was detected);
* resting detections, uncapped and capped at the colony size, in total and inside every zone;
* the sum of per-frame capped zone fractions (for the mean-of-frames variant);
* flight activity (``fly_px``, see :mod:`buzzphono.flight`) and moving detections in each zone.

Windows of the stimulus cycle (minutes 40-50 and so on) are whole minutes, so they are sums of
whole bins; finer analyses (the PSTH) use the bins directly. Bins have no gaps or overlaps, so on
the corrected clock every recorded frame is counted exactly once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from buzzphono.errors import Cancelled, DataError
from buzzphono.flight import camera_scale, fly_px_per_frame
from buzzphono.geometry import area, contains
from buzzphono.io.cache import CacheIndex, load_npz
from buzzphono.io.discover import ExperimentLayout
from buzzphono.stimulus import ClockFit, data_span_hours, find_bursts, fit_clock
from buzzphono.timebase import (
    NS_PER_HOUR,
    NS_PER_S,
    ClockModel,
    CorrectedClock,
    SessionFrames,
    ThesisClock,
    build_sessions,
    frames_per_chunk,
)

BIN_S = 30
BIN_NS = BIN_S * NS_PER_S
Clock = CorrectedClock | ThesisClock
# A zone is a polygon (published membership rule) or any function of (n, 2) positions -> bool mask.
ZoneTest = np.ndarray | Callable[[np.ndarray], np.ndarray]


def zone_mask(zone: ZoneTest, xy: np.ndarray) -> np.ndarray:
    if callable(zone):
        return np.asarray(zone(xy), dtype=bool)
    return contains(np.asarray(zone, dtype=np.float64), xy)


def walking_runs(
    resting: Mapping[str, np.ndarray], max_px_per_s: float, fps: float, lag_s: float = 2.0
) -> np.ndarray:
    """Mask of resting runs whose object moved faster than ``max_px_per_s`` over the last ``lag_s``.

    A resting detection's position jitters by a pixel or so; comparing it with the same object's
    position ``lag_s`` earlier measures real walking rather than that jitter. Runs whose object did
    not exist ``lag_s`` earlier are never flagged.
    """
    obj = resting["run_obj"].astype(np.int64)
    f0 = resting["run_f0"].astype(np.int64)
    ln = resting["run_len"].astype(np.int64)
    xy = resting["run_xy"]
    n = obj.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((f0, obj))
    big = np.int64(1) << 32
    keys = obj[order] * big + f0[order]
    lag = max(1, round(lag_s * fps))
    j = np.searchsorted(keys, obj * big + (f0 - lag), side="right") - 1
    ok = j >= 0
    jj = order[np.clip(j, 0, n - 1)]
    ok &= (obj[jj] == obj) & (f0 - lag < f0[jj] + ln[jj]) & (f0 - lag >= 0)
    dist = np.hypot(*(xy - xy[jj]).T)
    return ok & (dist / (lag / fps) > max_px_per_s)


# ----------------------------------------------------------------------------- the clock
@dataclass
class ExperimentClock:
    """The time base of one experiment and how it was obtained."""

    clock: Clock
    sessions: dict[str, SessionFrames]
    fit: ClockFit | None = None  # the burst fit (corrected mode only)
    nf: dict[str, tuple[int, str]] = field(default_factory=dict)  # chunk_id -> (frames, source)

    @property
    def mode(self) -> str:
        return self.clock.name

    @property
    def message(self) -> str:
        if isinstance(self.clock, CorrectedClock):
            return self.clock.model.message
        return (
            "Thesis clock: session start + 1200 s per chunk + frame / 25 (reproduces the published numbers)."
        )


def activity_by_session(
    layout: ExperimentLayout,
    cache_dir: Path,
    sessions: Mapping[str, SessionFrames],
    cage: np.ndarray,
    fps: float,
    min_median: float = 0.0,
) -> dict[str, dict[int, np.ndarray]]:
    """Flight activity per frame of every chunk with moving data, grouped by session."""
    scale = camera_scale(cage)
    out: dict[str, dict[int, np.ndarray]] = {}
    for c in layout.chunks:
        sf = sessions.get(c.name.session)
        path = cache_dir / "moving" / f"{c.chunk_id}.npz"
        if sf is None or c.name.index not in sf.nf or not path.exists():
            continue
        moving = load_npz(path)
        out.setdefault(sf.session, {})[c.name.index] = fly_px_per_frame(
            moving, sf.nf[c.name.index], scale, fps=fps, min_median=min_median
        )
    return out


def experiment_clock(
    layout: ExperimentLayout,
    cache_dir: Path,
    *,
    mode: str = "corrected",
    cage: np.ndarray | None = None,
    fps: float = 25.0,
    on_start_min: int = 40,
    rig_drift_s_per_day: float | None = None,
    manual_offset_s: Mapping[str, float] | None = None,
) -> ExperimentClock:
    """Build the time base: ``thesis``, ``nominal`` (real frame counts, no fit) or ``corrected``.

    The corrected clock needs the moving tier in the cache and the cage corners (for the camera
    scale of the flight-burst detector); without them it falls back to the nominal clock and says so.
    """
    index = CacheIndex.open(cache_dir, layout.root)
    nf = frames_per_chunk(index.data["chunks"])
    names = [c.name for c in layout.chunks if c.chunk_id in nf]
    sessions = build_sessions(names, nf, fps=fps)
    if mode == "thesis":
        return ExperimentClock(ThesisClock(fps=fps), sessions, None, nf)
    if not sessions:
        raise DataError(
            f"No chunk of {layout.root} has a known frame count, so its frames cannot be placed in time.",
            hint="Check that the tracking files were read (buzzphono validate).",
        )
    fit: ClockFit | None = None
    if mode == "nominal":
        model = ClockModel.nominal(
            sorted(sessions), "Nominal clock: real frame counts at the nominal frame rate."
        )
    elif mode == "corrected":
        if cage is None:
            model = ClockModel.nominal(
                sorted(sessions),
                "Nominal clock: no cage corners, so the tone-onset bursts were not searched.",
            )
        else:
            activity = activity_by_session(layout, cache_dir, sessions, cage, fps)
            fit = fit_clock(
                find_bursts(sessions, activity, on_start_min),
                list(sessions),
                rig_drift_s_per_day=rig_drift_s_per_day,
            )
            model = fit.model
    else:
        raise ValueError(f"unknown clock mode {mode!r}")
    if manual_offset_s:
        unknown = sorted(set(manual_offset_s) - set(sessions))
        if unknown:
            raise DataError(
                f"Manual clock offsets name sessions that are not in this experiment: {', '.join(unknown)}.",
                hint=f"Sessions here: {', '.join(sorted(sessions))}.",
            )
        offsets = dict(model.offset_s)
        source = dict(model.source)
        for s, v in manual_offset_s.items():
            offsets[s] = float(v)
            source[s] = "manual offset"
        model = ClockModel(
            offsets,
            model.drift_s_per_day,
            source,
            model.fitted,
            model.message + " Manual offsets set for: " + ", ".join(sorted(manual_offset_s)) + ".",
            model.n_bursts,
            model.n_kept,
            model.residual_sd_s,
        )
    return ExperimentClock(CorrectedClock(sessions, model), sessions, fit, nf)


# ----------------------------------------------------------------------------- binned counts
@dataclass
class Binned:
    """One experiment's per-frame counts summed into ``bin_ns`` bins from ``t0_ns`` (a whole hour)."""

    t0_ns: int
    bin_ns: int
    zones: tuple[str, ...]
    zone_area_px: np.ndarray  # (n_zones,)
    colony: int
    recorded: np.ndarray  # (n,) frames recorded
    rest_frames: np.ndarray  # (n,) frames with at least one resting detection
    cap_total: np.ndarray  # (n,) sum of min(n_rest, colony)
    raw_total: np.ndarray  # (n,) resting detections
    cap_in: np.ndarray  # (n, n_zones) sum of min(n_in, capped total)
    raw_in: np.ndarray  # (n, n_zones) resting detections in the zone
    frac_sum: np.ndarray  # (n, n_zones) sum over frames of capped in / capped total
    fly_px: np.ndarray  # (n,) flight activity, px
    moving_total: np.ndarray  # (n,) moving detections
    moving_in: np.ndarray  # (n, n_zones) moving detections in the zone
    chunks_used: int = 0
    chunks_skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_bins(self) -> int:
        return int(self.recorded.size)

    @property
    def bins_per_hour(self) -> int:
        return NS_PER_HOUR // self.bin_ns

    def zone_index(self, name: str) -> int:
        try:
            return self.zones.index(name)
        except ValueError:
            raise DataError(f"No zone named '{name}' (zones: {', '.join(self.zones)}).") from None

    def bin_start_ns(self) -> np.ndarray:
        return self.t0_ns + np.arange(self.n_bins, dtype=np.int64) * self.bin_ns

    @classmethod
    def empty(
        cls,
        t0_ns: int,
        t1_ns: int,
        zones: Mapping[str, ZoneTest],
        colony: int,
        bin_ns: int = BIN_NS,
        zone_areas: Mapping[str, float] | None = None,
    ) -> Binned:
        if NS_PER_HOUR % bin_ns:
            raise ValueError("the bin length must divide one hour")
        n = max(0, -(-(t1_ns - t0_ns) // bin_ns))
        nz = len(zones)
        return cls(
            t0_ns=t0_ns,
            bin_ns=bin_ns,
            zones=tuple(zones),
            zone_area_px=np.array(
                [
                    (zone_areas or {}).get(
                        k, np.nan if callable(p) else area(np.asarray(p, dtype=np.float64))
                    )
                    for k, p in zones.items()
                ],
                dtype=np.float64,
            ),
            colony=colony,
            recorded=np.zeros(n),
            rest_frames=np.zeros(n),
            cap_total=np.zeros(n),
            raw_total=np.zeros(n),
            cap_in=np.zeros((n, nz)),
            raw_in=np.zeros((n, nz)),
            frac_sum=np.zeros((n, nz)),
            fly_px=np.zeros(n),
            moving_total=np.zeros(n),
            moving_in=np.zeros((n, nz)),
        )

    def _bins(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = (np.asarray(t_ns, dtype=np.int64) - self.t0_ns) // self.bin_ns
        ok = (b >= 0) & (b < self.n_bins)
        return b[ok].astype(np.intp), ok

    def _add(self, target: np.ndarray, idx: np.ndarray, weights: np.ndarray | None = None) -> None:
        if weights is None or weights.ndim == 1:
            target += np.bincount(idx, weights=weights, minlength=self.n_bins)
        else:
            for z in range(weights.shape[1]):
                target[:, z] += np.bincount(idx, weights=weights[:, z], minlength=self.n_bins)

    def add_recorded(self, frame_ns: np.ndarray) -> None:
        idx, _ = self._bins(frame_ns)
        self._add(self.recorded, idx)

    def add_resting(self, frame_ns: np.ndarray, n_rest: np.ndarray, n_in: np.ndarray) -> None:
        """Frames with resting detections: ``n_rest`` (frames,), ``n_in`` (frames, zones)."""
        idx, ok = self._bins(frame_ns)
        n_rest = n_rest[ok].astype(np.float64)
        n_in = n_in[ok].astype(np.float64)
        cap_total = np.minimum(n_rest, float(self.colony))
        cap_in = np.minimum(n_in, cap_total[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(cap_total[:, None] > 0, cap_in / cap_total[:, None], 0.0)
        self._add(self.rest_frames, idx, (n_rest > 0).astype(np.float64))
        self._add(self.cap_total, idx, cap_total)
        self._add(self.raw_total, idx, n_rest)
        self._add(self.cap_in, idx, cap_in)
        self._add(self.raw_in, idx, n_in)
        self._add(self.frac_sum, idx, frac)

    def add_moving(self, det_ns: np.ndarray, in_zone: np.ndarray) -> None:
        idx, ok = self._bins(det_ns)
        self._add(self.moving_total, idx)
        self._add(self.moving_in, idx, in_zone[ok].astype(np.float64))

    def add_flight(self, frame_ns: np.ndarray, fly_px: np.ndarray) -> None:
        idx, ok = self._bins(frame_ns)
        self._add(self.fly_px, idx, fly_px[ok])


def resting_frame_counts(
    resting: Mapping[str, np.ndarray], polygons: list[ZoneTest], keep_runs: np.ndarray | None = None
) -> tuple[np.ndarray, ...]:
    """Per frame with resting detections: (frames, n_rest, n_in[frames, zones]) for several zones."""
    nz = len(polygons)
    f0 = resting["run_f0"].astype(np.int64)
    ln = resting["run_len"].astype(np.int64)
    run_xy = resting["run_xy"]
    if keep_runs is not None:
        f0, ln, run_xy = f0[keep_runs], ln[keep_runs], run_xy[keep_runs]
    frames: list[np.ndarray] = []
    totals: list[np.ndarray] = []
    inside: list[np.ndarray] = []
    if f0.size:
        lo = int(np.min(f0))
        hi = int(np.max(f0 + ln))
        w = np.stack([zone_mask(p, run_xy) for p in polygons], axis=1).astype(np.int64)
        size = hi - lo + 1
        a, b = f0 - lo, f0 + ln - lo  # +1 where a run starts, -1 just after it ends
        tot = np.bincount(a, minlength=size) - np.bincount(b, minlength=size)
        inz = np.stack(
            [
                np.bincount(a, weights=w[:, z], minlength=size)
                - np.bincount(b, weights=w[:, z], minlength=size)
                for z in range(nz)
            ],
            axis=1,
        ).astype(np.int64)
        tot = np.cumsum(tot)[:-1]
        inz = np.cumsum(inz, axis=0)[:-1]
        present = np.flatnonzero(tot > 0)
        frames.append((present + lo).astype(np.float64))
        totals.append(tot[present])
        inside.append(inz[present])
    exc = resting["exc_frame"]
    if exc.size:
        uniq, inv = np.unique(exc, return_inverse=True)
        frames.append(uniq.astype(np.float64))
        totals.append(np.bincount(inv, minlength=uniq.size).astype(np.int64))
        inside.append(
            np.stack(
                [
                    np.bincount(inv, weights=zone_mask(p, resting["exc_xy"]), minlength=uniq.size)
                    for p in polygons
                ],
                axis=1,
            ).astype(np.int64)
        )
    if not frames:
        return np.empty(0), np.empty(0, dtype=np.int64), np.empty((0, nz), dtype=np.int64)
    f = np.concatenate(frames)
    t = np.concatenate(totals)
    i = np.concatenate(inside)
    if len(frames) > 1:  # a frame can be both in a run and an exception: merge
        uniq, inv = np.unique(f, return_inverse=True)
        t = np.bincount(inv, weights=t, minlength=uniq.size).astype(np.int64)
        i = np.stack([np.bincount(inv, weights=i[:, z], minlength=uniq.size) for z in range(nz)], axis=1)
        f, i = uniq, i.astype(np.int64)
    return f, t, i


def experiment_span(eclock: ExperimentClock, layout: ExperimentLayout) -> tuple[int, int]:
    """Whole hours covering every chunk on the experiment's clock."""
    if isinstance(eclock.clock, CorrectedClock):
        lo, hi = data_span_hours(eclock.sessions)
        # a fitted offset can move the data across an hour boundary: allow one hour each side
        return lo - NS_PER_HOUR, hi + NS_PER_HOUR
    starts = []
    for c in layout.chunks:
        if c.name.session_start() is not None:
            starts.append(eclock.clock.chunk_start_ns(c.name))
    if not starts:
        raise DataError(f"No chunk of {layout.root} has a usable time stamp.")
    lo, hi = min(starts), max(starts) + 2 * NS_PER_HOUR
    return lo - lo % NS_PER_HOUR, hi - hi % NS_PER_HOUR


def bin_experiment(
    layout: ExperimentLayout,
    cache_dir: Path,
    eclock: ExperimentClock,
    zones: Mapping[str, ZoneTest],
    *,
    colony: int = 30,
    zone_areas: Mapping[str, float] | None = None,
    walking_px_per_s: float | None = None,
    cage: np.ndarray | None = None,
    t_start_ns: int | None = None,
    t_end_ns: int | None = None,
    with_moving: bool = True,
    min_median: float = 0.0,
    bin_ns: int = BIN_NS,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    arrays_cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> Binned:
    """Sum every chunk's per-frame counts into bins (see the module docstring).

    ``arrays_cache`` (optional) keeps the loaded resting arrays between calls, for analyses that
    bin the same experiment several times.
    """
    lo, hi = experiment_span(eclock, layout)
    t0 = lo if t_start_ns is None else t_start_ns - t_start_ns % NS_PER_HOUR
    t1 = hi if t_end_ns is None else t_end_ns
    binned = Binned.empty(t0, t1, zones, colony, bin_ns, zone_areas)
    polygons = list(zones.values())
    scale = camera_scale(cage) if cage is not None else None
    fps = eclock.sessions[next(iter(eclock.sessions))].fps if eclock.sessions else 25.0
    n = len(layout.chunks)
    for i, c in enumerate(layout.chunks):
        if cancel is not None and cancel():
            raise Cancelled("Cancelled while counting detections.")
        if progress is not None:
            progress(i / max(n, 1), f"Counting chunk {i + 1} of {n}")
        if c.chunk_id not in eclock.nf:
            binned.chunks_skipped.append((c.chunk_id, "no readable tracking file (frame count unknown)"))
            continue
        nf, _ = eclock.nf[c.chunk_id]
        try:
            frame_ns = eclock.clock.frame_ns(c.name, np.arange(nf))
        except DataError as exc:
            binned.chunks_skipped.append((c.chunk_id, exc.message))
            continue
        binned.add_recorded(frame_ns[_mask(frame_ns, t_start_ns, t_end_ns)])
        rest_path = cache_dir / "resting" / f"{c.chunk_id}.npz"
        if rest_path.exists():
            if arrays_cache is not None:
                resting = arrays_cache.get(c.chunk_id) or arrays_cache.setdefault(
                    c.chunk_id, load_npz(rest_path)
                )
            else:
                resting = load_npz(rest_path)
            keep_runs = (
                ~walking_runs(resting, walking_px_per_s, fps) if walking_px_per_s is not None else None
            )
            f, n_rest, n_in = resting_frame_counts(resting, polygons, keep_runs)
            if f.size:
                t = eclock.clock.frame_ns(c.name, f)
                keep = _mask(t, t_start_ns, t_end_ns)
                binned.add_resting(t[keep], n_rest[keep], n_in[keep])
        mov_path = cache_dir / "moving" / f"{c.chunk_id}.npz"
        if with_moving and mov_path.exists():
            moving = load_npz(mov_path)
            det_frame = moving["det_frame"]
            if det_frame.size:
                t = eclock.clock.frame_ns(c.name, det_frame)
                keep = _mask(t, t_start_ns, t_end_ns)
                xy = moving["det_xy"][keep]
                in_zone = np.stack([zone_mask(p, xy) for p in polygons], axis=1)
                binned.add_moving(t[keep], in_zone)
            if scale is not None:
                fly = fly_px_per_frame(moving, nf, scale, fps=fps, min_median=min_median)
                keep = _mask(frame_ns, t_start_ns, t_end_ns)
                binned.add_flight(frame_ns[keep], fly[keep])
        binned.chunks_used += 1
    if progress is not None:
        progress(1.0, "Counting done")
    return binned


def _mask(t: np.ndarray, start: int | None, end: int | None) -> np.ndarray:
    keep = np.ones(t.size, dtype=bool)
    if start is not None:
        keep &= t >= start
    if end is not None:
        keep &= t < end
    return keep
