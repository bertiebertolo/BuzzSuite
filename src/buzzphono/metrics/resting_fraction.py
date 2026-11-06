"""Speaker-zone resting fraction: per-frame counts and the hour-cycle × minute grid.

Definitions (per video frame):

* ``n_rest`` = resting detections in the frame; ``n_in`` = those inside the speaker zone.
* With the colony size C (30 mosquitoes by default), the *capped* counts are
  ``total = min(n_rest, C)`` and ``in_zone = min(n_in, total)``. The tracker over-detects (one
  crowded mosquito can give several detections, and static specks look like resters), so without
  the cap the false positives inflate the denominator only.

Per window (e.g. minutes 40-50 of one hour), the headline value, as published, is the ratio of
the summed capped counts::

    colony_frac_in_zone = sum(in_zone) / sum(total)

Because ``total`` sits at C in almost every frame, this is in practice "speaker resters ÷ C".
The grid also keeps the uncapped sums (``resting_frac_in_zone = sum(n_in) / sum(n_rest)``) and the
sum of per-frame capped fractions (for the mean-of-frames variant), so the effect of the cap is
always visible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from buzzphono.geometry import contains
from buzzphono.timebase import NS_PER_HOUR, NS_PER_MIN, floor_hour

Arrays = dict[str, np.ndarray]
MINUTES = 60


@dataclass(frozen=True)
class FrameCounts:
    """Resting counts for every frame of one chunk that has at least one resting detection."""

    frames: np.ndarray  # float64 frame numbers (whole numbers except rare tracker oddities)
    n_rest: np.ndarray  # int64 resting detections in the frame
    n_in: np.ndarray  # int64 of those inside the zone


def frame_counts(resting: Arrays, inside: Callable[[np.ndarray], np.ndarray]) -> FrameCounts:
    """Per-frame resting counts from a chunk's resting cache arrays.

    ``inside(xy)`` returns a boolean mask for an (n, 2) array of positions (a zone test).
    """
    f0 = resting["run_f0"].astype(np.int64)
    ln = resting["run_len"].astype(np.int64)
    parts_f: list[np.ndarray] = []
    parts_t: list[np.ndarray] = []
    parts_i: list[np.ndarray] = []
    if f0.size:
        lo = int(np.min(f0))
        hi = int(np.max(f0 + ln))
        w_in = inside(resting["run_xy"]).astype(np.int64)
        tot = np.zeros(hi - lo + 1, dtype=np.int64)
        inz = np.zeros(hi - lo + 1, dtype=np.int64)
        np.add.at(tot, f0 - lo, 1)
        np.add.at(tot, f0 + ln - lo, -1)
        np.add.at(inz, f0 - lo, w_in)
        np.add.at(inz, f0 + ln - lo, -w_in)
        tot = np.cumsum(tot)[:-1]
        inz = np.cumsum(inz)[:-1]
        present = np.flatnonzero(tot > 0)
        parts_f.append((present + lo).astype(np.float64))
        parts_t.append(tot[present])
        parts_i.append(inz[present])
    exc_frame = resting["exc_frame"]
    if exc_frame.size:
        uniq, inv = np.unique(exc_frame, return_inverse=True)
        parts_f.append(uniq)
        parts_t.append(np.bincount(inv, minlength=uniq.size).astype(np.int64))
        parts_i.append(
            np.bincount(inv, weights=inside(resting["exc_xy"]), minlength=uniq.size).astype(np.int64)
        )
    if not parts_f:
        empty_i = np.empty(0, dtype=np.int64)
        return FrameCounts(np.empty(0), empty_i, empty_i)
    return FrameCounts(np.concatenate(parts_f), np.concatenate(parts_t), np.concatenate(parts_i))


def polygon_test(polygon: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def inside(xy: np.ndarray) -> np.ndarray:
        return contains(polygon, xy)

    return inside


@dataclass
class RestingGrids:
    """Resting counts summed into (hour-cycle × minute-of-hour) cells and wall-clock minutes.

    Cycle c covers ``[t0 + c h, t0 + (c+1) h)`` where ``t0`` is the start of the clock hour that
    contains the analysis start.
    """

    t0_ns: int
    n_cycles: int
    colony: int
    in_zone: np.ndarray  # (n_cycles, 60) sum of capped in-zone counts
    total: np.ndarray  # (n_cycles, 60) sum of capped totals
    raw_in: np.ndarray  # (n_cycles, 60) uncapped in-zone detections
    raw_total: np.ndarray  # (n_cycles, 60) uncapped resting detections
    frac_sum: np.ndarray  # (n_cycles, 60) sum over frames of in_zone/total
    n_frames: np.ndarray  # (n_cycles, 60) frames with at least one resting detection
    abs_in: np.ndarray = field(init=False)  # (n_cycles * 60,) capped in-zone per wall-clock minute
    abs_total: np.ndarray = field(init=False)  # (n_cycles * 60,) capped total per wall-clock minute

    def __post_init__(self) -> None:
        self.abs_in = np.zeros(self.n_cycles * MINUTES)
        self.abs_total = np.zeros(self.n_cycles * MINUTES)

    @classmethod
    def empty(cls, t_start_ns: int, t_end_ns: int, colony: int) -> RestingGrids:
        t0 = int(floor_hour(t_start_ns))
        n_cycles = -(-(t_end_ns - t0) // NS_PER_HOUR) + 1  # ceil, +1 as in the published code
        shape = (n_cycles, MINUTES)
        return cls(
            t0_ns=t0,
            n_cycles=n_cycles,
            colony=colony,
            in_zone=np.zeros(shape),
            total=np.zeros(shape),
            raw_in=np.zeros(shape),
            raw_total=np.zeros(shape),
            frac_sum=np.zeros(shape),
            n_frames=np.zeros(shape, dtype=np.int64),
        )

    def add(self, counts: FrameCounts, frame_ns: np.ndarray, t_start_ns: int, t_end_ns: int) -> int:
        """Add one chunk's frames that fall in ``[t_start, t_end)``. Returns how many were added."""
        keep = (frame_ns >= t_start_ns) & (frame_ns < t_end_ns)
        if not keep.any():
            return 0
        t = frame_ns[keep]
        n_rest = counts.n_rest[keep].astype(np.float64)
        n_in = counts.n_in[keep].astype(np.float64)
        cap_total = np.minimum(n_rest, float(self.colony))
        cap_in = np.minimum(n_in, cap_total)
        cyc = (floor_hour(t) - self.t0_ns) // NS_PER_HOUR
        minute = (t // NS_PER_MIN) % MINUTES
        ok = (cyc >= 0) & (cyc < self.n_cycles)
        flat = (cyc[ok] * MINUTES + minute[ok]).astype(np.intp)
        size = self.n_cycles * MINUTES
        self.total += np.bincount(flat, weights=cap_total[ok], minlength=size).reshape(self.total.shape)
        self.in_zone += np.bincount(flat, weights=cap_in[ok], minlength=size).reshape(self.total.shape)
        self.raw_total += np.bincount(flat, weights=n_rest[ok], minlength=size).reshape(self.total.shape)
        self.raw_in += np.bincount(flat, weights=n_in[ok], minlength=size).reshape(self.total.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(cap_total[ok] > 0, cap_in[ok] / cap_total[ok], 0.0)
        self.frac_sum += np.bincount(flat, weights=frac, minlength=size).reshape(self.total.shape)
        self.n_frames += np.bincount(flat, minlength=size).reshape(self.total.shape)
        amin = (t[ok] - self.t0_ns) // NS_PER_MIN
        a_ok = (amin >= 0) & (amin < self.abs_total.size)
        idx = amin[a_ok].astype(np.intp)
        self.abs_total += np.bincount(idx, weights=cap_total[ok][a_ok], minlength=self.abs_total.size)
        self.abs_in += np.bincount(idx, weights=cap_in[ok][a_ok], minlength=self.abs_in.size)
        return int(ok.sum())
