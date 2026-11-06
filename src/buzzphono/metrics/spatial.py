"""Where the mosquitoes are: occupancy maps, the radial profile around the speaker, sample tracks.

Only the usable hour-cycles (coverage rule passed, stimulus not known to have failed) are used.

* :func:`occupancy_maps` – resting detections per frame in square cells (``cell_px``) for every
  window (ON, OFF ...), so maps are comparable whatever the window length; the ON − OFF map uses
  a symmetric diverging scale when plotted.
* :func:`radial_profile` – from those maps: occupancy per frame and per unit cage area as a
  function of the distance from the speaker zone (cells inside the zone are at distance 0), so a
  ring that is mostly outside the cage does not look empty just because it is small.
* :func:`sample_tracks` – a seeded random sample of moving tracks that start in each window, for
  trajectory overlays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from buzzphono.errors import Cancelled
from buzzphono.frames import ExperimentClock
from buzzphono.geometry import contains, signed_distance
from buzzphono.io.cache import load_npz
from buzzphono.io.discover import ExperimentLayout
from buzzphono.timebase import NS_PER_HOUR, NS_PER_MIN


@dataclass
class OccupancyMaps:
    cell_px: float
    width_px: int
    height_px: int
    maps: dict[str, np.ndarray]  # window -> (ny, nx) resting detections per frame per cell
    frames: dict[str, float]  # window -> frames recorded in usable cycles

    def difference(self, a: str = "on", b: str = "off") -> np.ndarray:
        return self.maps[a] - self.maps[b]

    def tidy(self) -> pd.DataFrame:
        """Long table: window, x_px, y_px (cell centres), detections_per_frame."""
        ny, nx = next(iter(self.maps.values())).shape
        xc = (np.arange(nx) + 0.5) * self.cell_px
        yc = (np.arange(ny) + 0.5) * self.cell_px
        gx, gy = np.meshgrid(xc, yc)
        parts = []
        for name, m in self.maps.items():
            parts.append(
                pd.DataFrame(
                    {
                        "window": name,
                        "x_px": gx.ravel(),
                        "y_px": gy.ravel(),
                        "detections_per_frame": m.ravel(),
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)


def window_labels(
    t_ns: np.ndarray, windows: Mapping[str, tuple[int, int]], usable_starts: np.ndarray
) -> np.ndarray:
    """Per time: index into ``windows`` of the window it falls in (−1 = none or unusable cycle)."""
    t = np.asarray(t_ns, dtype=np.int64)
    label = np.full(t.size, -1, dtype=np.int64)
    in_usable = np.isin(t - t % NS_PER_HOUR, usable_starts)
    minute = (t // NS_PER_MIN) % 60
    for k, (a, b) in enumerate(windows.values()):
        label[in_usable & (minute >= a) & (minute < b) & (label < 0)] = k
    return label


def occupancy_maps(
    layout: ExperimentLayout,
    cache_dir: Path,
    eclock: ExperimentClock,
    usable_starts: np.ndarray,
    windows: Mapping[str, tuple[int, int]],
    frame_size: tuple[int, int],
    *,
    cell_px: float = 10.0,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> OccupancyMaps:
    w, h = frame_size
    nx, ny = int(np.ceil(w / cell_px)), int(np.ceil(h / cell_px))
    names = list(windows)
    counts = np.zeros((len(names), ny * nx))
    frames = np.zeros(len(names))
    starts = np.asarray(usable_starts, dtype=np.int64)
    n = len(layout.chunks)
    for i, c in enumerate(layout.chunks):
        if cancel is not None and cancel():
            raise Cancelled("Cancelled while building occupancy maps.")
        if progress is not None:
            progress(i / max(n, 1), f"Mapping chunk {i + 1} of {n}")
        if c.chunk_id not in eclock.nf:
            continue
        nf, _ = eclock.nf[c.chunk_id]
        lab = window_labels(eclock.clock.frame_ns(c.name, np.arange(nf)), windows, starts)
        frames += np.bincount(lab[lab >= 0], minlength=len(names))
        path = cache_dir / "resting" / f"{c.chunk_id}.npz"
        if not path.exists():
            continue
        r = load_npz(path)
        f0, ln, xy = r["run_f0"].astype(np.int64), r["run_len"].astype(np.int64), r["run_xy"]
        if f0.size:
            # each run sits still: weight its cell by the frames it spends in each window
            frame_label = np.full(int(np.max(f0 + ln)) + 1, -1, dtype=np.int64)
            upto = min(frame_label.size, lab.size)
            frame_label[:upto] = lab[:upto]
            beyond = np.arange(upto, frame_label.size)
            if beyond.size:
                frame_label[upto:] = window_labels(eclock.clock.frame_ns(c.name, beyond), windows, starts)
            cell = _cells(xy, cell_px, nx, ny)
            for k in range(len(names)):
                cum = np.concatenate([[0], np.cumsum(frame_label == k)])
                weight = cum[f0 + ln] - cum[f0]
                ok = (cell >= 0) & (weight > 0)
                counts[k] += np.bincount(cell[ok], weights=weight[ok], minlength=ny * nx)
        if r["exc_frame"].size:
            lab_e = window_labels(eclock.clock.frame_ns(c.name, r["exc_frame"]), windows, starts)
            cell = _cells(r["exc_xy"], cell_px, nx, ny)
            for k in range(len(names)):
                ok = (lab_e == k) & (cell >= 0)
                counts[k] += np.bincount(cell[ok], minlength=ny * nx)
    if progress is not None:
        progress(1.0, "Maps done")
    with np.errstate(invalid="ignore", divide="ignore"):
        maps = {
            name: (counts[k] / frames[k] if frames[k] > 0 else np.full(ny * nx, np.nan)).reshape(ny, nx)
            for k, name in enumerate(names)
        }
    return OccupancyMaps(cell_px, w, h, maps, dict(zip(names, frames.tolist(), strict=True)))


def _cells(xy: np.ndarray, cell_px: float, nx: int, ny: int) -> np.ndarray:
    ix = np.floor(xy[:, 0] / cell_px).astype(np.int64)
    iy = np.floor(xy[:, 1] / cell_px).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(xy).all(1)
    return np.where(ok, iy * nx + ix, -1)


def radial_profile(
    maps: OccupancyMaps, speaker: np.ndarray, cage: np.ndarray, *, ring_px: float = 20.0
) -> pd.DataFrame:
    """Occupancy per frame per 1,000 px² of cage, by distance from the speaker zone (rings of ``ring_px``)."""
    ny, nx = next(iter(maps.maps.values())).shape
    xc = (np.arange(nx) + 0.5) * maps.cell_px
    yc = (np.arange(ny) + 0.5) * maps.cell_px
    gx, gy = np.meshgrid(xc, yc)
    centres = np.column_stack([gx.ravel(), gy.ravel()])
    in_cage = contains(cage, centres)
    dist = np.maximum(signed_distance(speaker, centres), 0.0)
    ring = np.floor(dist / ring_px).astype(np.int64)
    cell_area = maps.cell_px**2
    rows = []
    for r in np.unique(ring[in_cage]):
        sel = in_cage & (ring == r)
        ring_area = sel.sum() * cell_area
        row = {
            "distance_from_px": r * ring_px,
            "distance_to_px": (r + 1) * ring_px,
            "cage_area_px2": ring_area,
        }
        for name, m in maps.maps.items():
            row[f"{name}_per_frame_per_1000px2"] = float(np.nansum(m.ravel()[sel]) / ring_area * 1000.0)
        rows.append(row)
    return pd.DataFrame(rows)


def sample_tracks(
    layout: ExperimentLayout,
    cache_dir: Path,
    eclock: ExperimentClock,
    usable_starts: np.ndarray,
    windows: Mapping[str, tuple[int, int]],
    *,
    per_window: int = 30,
    min_points: int = 10,
    seed: int = 20261104,
) -> pd.DataFrame:
    """Up to ``per_window`` random moving tracks starting in each window (seeded, reproducible)."""
    starts = np.asarray(usable_starts, dtype=np.int64)
    candidates: dict[int, list[tuple[str, int]]] = {k: [] for k in range(len(windows))}
    for c in layout.chunks:
        path = cache_dir / "moving" / f"{c.chunk_id}.npz"
        if c.chunk_id not in eclock.nf or not path.exists():
            continue
        m = load_npz(path)
        obj = m["det_obj"]
        if obj.size == 0:
            continue
        ids, first, n_pts = np.unique(obj, return_index=True, return_counts=True)
        long = n_pts >= min_points
        lab = window_labels(eclock.clock.frame_ns(c.name, m["det_frame"][first[long]]), windows, starts)
        for k, oid in zip(lab, ids[long], strict=True):
            if k >= 0:
                candidates[int(k)].append((c.chunk_id, int(oid)))
    rng = np.random.default_rng(seed)
    picked: dict[str, list[tuple[int, int]]] = {}
    names = list(windows)
    for k, cands in candidates.items():
        cands.sort()
        take = rng.permutation(len(cands))[:per_window] if cands else np.empty(0, dtype=np.int64)
        for j in take:
            chunk_id, oid = cands[int(j)]
            picked.setdefault(chunk_id, []).append((oid, k))
    rows = []
    for chunk_id, items in sorted(picked.items()):
        m = load_npz(cache_dir / "moving" / f"{chunk_id}.npz")
        for oid, k in items:
            sel = m["det_obj"] == oid
            xy = m["det_xy"][sel]
            rows.append(
                pd.DataFrame(
                    {
                        "window": names[k],
                        "chunk": chunk_id,
                        "track": oid,
                        "frame": m["det_frame"][sel],
                        "x_px": xy[:, 0],
                        "y_px": xy[:, 1],
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["window", "chunk", "track", "frame", "x_px", "y_px"])
    return pd.concat(rows, ignore_index=True)
