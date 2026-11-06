"""Flight measures shared by the clock calibration and the flight analyses.

Distances and speeds scale with the camera: the published flight rules were set on the Aedes
camera, where the cage is 780 pixels wide, so every pixel threshold is multiplied by the camera
scale ``s = cage width / 780`` (cage width = mean length of the top and bottom cage edges).

* Flight activity, ``fly_px``: per frame, the summed length of all moving-track steps faster than
  300 px/s × s (the thesis activity measure).
* A *flyer* track: at least 10 points with a mean step of at least 4 px/frame × s (the thesis
  "real flyer" rule); optionally also a median step of at least ``min_median`` px/frame × s, which
  rejects stationary tracks with a few mislinked jumps in crowded cages.
"""

from __future__ import annotations

import numpy as np

REF_CAGE_WIDTH_PX = 780.0
FLY_PX_SPEED = 300.0  # px/s on the reference camera
FLYER_STEP = 4.0  # px/frame on the reference camera
MIN_FLYER_POINTS = 10
NOMINAL_FPS = 25.0


def camera_scale(cage: np.ndarray) -> float:
    """``s`` for a cage given as 4 corners (top-left, top-right, bottom-right, bottom-left)."""
    c = np.asarray(cage, dtype=np.float64).reshape(4, 2)
    top = float(np.hypot(*(c[1] - c[0])))
    bottom = float(np.hypot(*(c[2] - c[3])))
    return 0.5 * (top + bottom) / REF_CAGE_WIDTH_PX


def frame_limit(nf: int) -> int:
    """Frames per chunk array the published flight code kept (a little slack past the chunk end)."""
    return max(30200, int(nf) + 140)


def track_steps(moving: dict[str, np.ndarray], n_limit: int) -> tuple[np.ndarray, ...]:
    """Consecutive steps of every moving track, as the published flight code formed them.

    Returns ``(owner, frame_to, step_px, frames_between)``: for each step between consecutive points
    of one track (frame numbers truncated to integers; points outside 0..n_limit-1 dropped first),
    its track index, the later point's frame, its length and the frame difference.
    """
    obj = moving["det_obj"]
    fi = moving["det_frame"].astype(np.int64)
    xy = moving["det_xy"]
    ok = (fi >= 0) & (fi < n_limit)
    obj, fi, xy = obj[ok], fi[ok], xy[ok]
    if obj.size < 2:
        empty = np.empty(0)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), empty, empty
    same = obj[1:] == obj[:-1]
    d = xy[1:] - xy[:-1]
    step = np.hypot(d[:, 0], d[:, 1])[same]
    dframes = np.diff(fi).astype(np.float64)[same]
    return obj[1:][same].astype(np.int64), fi[1:][same], step, dframes


def fly_px_per_frame(
    moving: dict[str, np.ndarray], nf: int, scale: float, fps: float = NOMINAL_FPS, min_median: float = 0.0
) -> np.ndarray:
    """Flight activity per frame of one chunk (length ``nf``): summed length of fast steps."""
    owner, frame_to, step, dframes = track_steps(moving, frame_limit(nf))
    out = np.zeros(nf)
    if step.size == 0:
        return out
    dt = dframes / fps
    dt[dt <= 0] = 1.0 / fps
    fast = (step / dt) >= FLY_PX_SPEED * scale
    if min_median > 0:
        # a track counts only if its median step is sustained (crowded-cage filter)
        n_tracks = int(owner.max()) + 1
        order = np.lexsort((step, owner))
        counts = np.bincount(owner, minlength=n_tracks)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        med = np.full(n_tracks, np.nan)
        has = counts > 0
        lo = starts[has] + (counts[has] - 1) // 2
        hi = starts[has] + counts[has] // 2
        s_sorted = step[order]
        med[has] = 0.5 * (s_sorted[lo] + s_sorted[hi])
        fast &= med[owner] >= min_median * scale
    keep = fast & (frame_to < nf)
    np.add.at(out, frame_to[keep], step[keep])
    return out
