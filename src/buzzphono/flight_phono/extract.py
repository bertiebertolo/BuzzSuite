"""Per-chunk flight arrays, computed from BuzzPhono's tracking cache (no pickles are read here).

Three sets of per-frame arrays per chunk, indexed by frame within the chunk (clock-free):

* ``main``  – flyers, take-offs, flyers near each zone, entries, wall sectors, landings, resting
  totals and in-zone counts, flight px and 10-s heat maps of flyer positions;
* ``links`` – flight bouts (state-1 runs of assembled tracks) with take-off and landing sites;
* ``traj``  – flyer coordinates and per-frame summed distances to each zone centre (cage widths).

The arithmetic, including the order of floating-point accumulation, is that of the original
package, so the arrays are identical to its caches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from buzzphono.flight_phono.geometry import FlightGeometry

GRID = 40  # heat-map bins per axis over the cage bounding box
BLOCK = 250  # frames per heat-map block (10 s)
FPS = 25.0
FLYER_STEP = 4.0
FLY_PX_SPEED = 300.0
MIN_FLYER_PTS = 10
R_MAIN = 1
MAXGAP = 25  # frames: a state change across a longer gap is not a trusted take-off / landing link
NMAX_MIN = 30200

Arrays = dict[str, np.ndarray]


def nmax_for(nf: int) -> int:
    return max(NMAX_MIN, int(nf) + 140)


def nblk_for(nmax: int) -> int:
    return nmax // BLOCK + 1


def _is_sustained(step: np.ndarray, min_median: float, s: float) -> bool:
    return min_median <= 0 or float(np.median(step)) >= min_median * s


def _tracks(moving: Mapping[str, np.ndarray], n_limit: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(object index, integer frames, xy) of every moving track, points outside 0..n_limit-1 dropped."""
    own = moving["det_obj"]
    if own.size == 0:
        return []
    fi_all = moving["det_frame"].astype(np.int64)
    xy_all = moving["det_xy"]
    cuts = np.flatnonzero(np.diff(own)) + 1
    out = []
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, own.size], strict=True):
        fi, co = fi_all[lo:hi], xy_all[lo:hi]
        ok = (fi >= 0) & (fi < n_limit)
        out.append((int(own[lo]), fi[ok], co[ok]))
    return out


def resting_counts(
    resting: Mapping[str, np.ndarray], geom: FlightGeometry, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: resting detections in total and in each zone (frames truncated to integers)."""
    nz = len(geom.names)
    rest_tot = np.zeros(n, np.int32)
    rest_in = np.zeros((nz, n), np.int32)
    f0 = resting["run_f0"].astype(np.int64)
    ln = resting["run_len"].astype(np.int64)
    xy = resting["run_xy"]
    if f0.size:
        frames = np.repeat(f0, ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
        run = np.repeat(np.arange(f0.size), ln)
        good = (frames >= 0) & (frames < n)
        np.add.at(rest_tot, frames[good], 1)
        for z in range(nz):
            inside = geom.paths[z].contains_points(xy)[run]
            np.add.at(rest_in[z], frames[good & inside], 1)
    ef = resting["exc_frame"]
    if ef.size:
        fi = ef.astype(np.int64)
        good = (fi >= 0) & (fi < n)
        np.add.at(rest_tot, fi[good], 1)
        for z in range(nz):
            inside = geom.paths[z].contains_points(resting["exc_xy"])
            np.add.at(rest_in[z], fi[good & inside], 1)
    return rest_tot, rest_in


def extract_main(
    moving: Mapping[str, np.ndarray],
    resting: Mapping[str, np.ndarray] | None,
    geom: FlightGeometry,
    nmax: int,
    mult: float,
    min_median: float = 0.0,
) -> Arrays:
    s, names, nz = geom.s, geom.names, len(geom.names)
    n_all = nmax
    nblk = nblk_for(nmax)
    nr_ = len(geom.rpx)
    x0, x1, y0, y1 = geom.bbox
    near_n = np.zeros((nr_, nz, n_all), np.int32)
    fly_n = np.zeros(n_all, np.int32)
    all_n = np.zeros(n_all, np.int32)
    fly_px = np.zeros(n_all, np.float64)
    takeoff = np.zeros(n_all, np.int32)
    takeoff_rest = np.zeros(n_all, np.int32)
    sector_n = np.zeros((nz, n_all), np.int32)
    entries = np.zeros((nz, n_all), np.int32)
    path_near = np.zeros((nz, n_all), np.float32)
    landing_in = np.zeros((nz, n_all), np.int32)
    land_rest_in = np.zeros((nz, n_all), np.int32)
    heat = np.zeros(nblk * GRID * GRID, np.int32)
    start_rest = moving.get("obj_start_resting")
    end_rest = moving.get("obj_end_resting")
    rows: list[list[float]] = []
    for obj, fi, co in _tracks(moving, n_all):
        n = len(fi)
        if n == 0:
            continue
        np.add.at(all_n, fi, 1)
        if n < 2:
            continue
        d = co[1:] - co[:-1]
        step = np.hypot(d[:, 0], d[:, 1])
        dt = np.diff(fi).astype(float) / FPS
        dt[dt <= 0] = 1.0 / FPS
        fast = (step / dt) >= FLY_PX_SPEED * s
        sustained = _is_sustained(step, min_median, s)
        if fast.any() and sustained:
            np.add.at(fly_px, fi[1:][fast], step[fast])
        mean_step = float(step.mean())
        if n < MIN_FLYER_PTS or mean_step < FLYER_STEP * s * mult or not sustained:
            continue
        st_rest = bool(start_rest[obj]) if start_rest is not None else False
        en_rest = bool(end_rest[obj]) if end_rest is not None else False
        np.add.at(fly_n, fi, 1)
        takeoff[fi[0]] += 1
        if st_rest:
            takeoff_rest[fi[0]] += 1
        dist = np.stack([geom.distance(z, co) for z in range(nz)])
        for r, rpx in enumerate(geom.rpx):
            nr = dist <= rpx
            for z in range(nz):
                if nr[z].any():
                    np.add.at(near_n[r, z], fi[nr[z]], 1)
        nm = dist <= geom.rpx[R_MAIN]
        ent = nm[:, 1:] & ~nm[:, :-1]
        inside_start = nm[:, :-1]
        for z in range(nz):
            if ent[z].any():
                np.add.at(entries[z], fi[1:][ent[z]], 1)
            if inside_start[z].any():
                np.add.at(path_near[z], fi[1:][inside_start[z]], step[inside_start[z]].astype(np.float32))
        ang = np.arctan2(co[:, 1] - geom.centre[1], co[:, 0] - geom.centre[0])
        pr = (geom.sign * (ang - geom.cang0)) % (2 * np.pi)
        sec = np.searchsorted(geom.rel[1:], pr, side="right")
        zsec = geom.sec2zone[sec]
        for z in range(nz):
            sel = zsec == z
            if sel.any():
                np.add.at(sector_n[z], fi[sel], 1)
        ex, ey = co[-1]
        for z in range(nz):
            if geom.paths[z].contains_point((ex, ey)):
                landing_in[z][fi[-1]] += 1
                if en_rest:
                    land_rest_in[z][fi[-1]] += 1
        gx = ((co[:, 0] - x0) / (x1 - x0) * GRID).astype(np.int64)
        gy = ((co[:, 1] - y0) / (y1 - y0) * GRID).astype(np.int64)
        inb = (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID)
        flat = (fi[inb] // BLOCK) * GRID * GRID + gy[inb] * GRID + gx[inb]
        heat += np.bincount(flat, minlength=heat.size).astype(np.int32)
        rows.append(
            [fi[0], fi[-1], n, mean_step, float(step.sum()), float(st_rest), float(en_rest), ex, ey]
            + [float(dist[z].min()) for z in range(nz)]
            + [int(nm[z].sum()) for z in range(nz)]
        )
    if resting is not None:
        rest_tot, rest_in = resting_counts(resting, geom, n_all)
    else:
        rest_tot, rest_in = np.zeros(n_all, np.int32), np.zeros((nz, n_all), np.int32)
    return {
        "fly_n": fly_n,
        "all_n": all_n,
        "fly_px": fly_px,
        "takeoff": takeoff,
        "takeoff_rest": takeoff_rest,
        "near_n": near_n,
        "entries": entries,
        "path_near": path_near,
        "sector_n": sector_n,
        "landing_in": landing_in,
        "land_rest_in": land_rest_in,
        "heat": heat.reshape(nblk, GRID, GRID),
        "rest_tot": rest_tot,
        "rest_in": rest_in,
        "zone_names": np.array(names),
        "tracks": np.array(rows, float).reshape(-1, 9 + 2 * nz),
    }


def expand_tracks(tracks: Mapping[str, np.ndarray]) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """(track id, start frame, state codes, xy) per assembled track from the tracks cache tier."""
    out = []
    off = tracks["trk_run_off"]
    run_len = tracks["run_len"].astype(np.int64)
    for i, tid in enumerate(tracks["trk_id"]):
        lo, hi = int(off[i]), int(off[i + 1])
        ln = run_len[lo:hi]
        out.append(
            (
                int(tid),
                int(tracks["trk_start"][i]),
                np.repeat(tracks["run_state"][lo:hi], ln),
                np.repeat(tracks["run_xy"][lo:hi], ln, axis=0),
            )
        )
    return out


def extract_links(tracks: Mapping[str, np.ndarray], geom: FlightGeometry, nmax: int) -> Arrays:
    nz, n_all = len(geom.names), nmax
    rest_tot = np.zeros(n_all, np.int32)
    rest_in = np.zeros((nz, n_all), np.int32)
    bouts: list[list[float]] = []
    for tid, start, code, c in expand_tracks(tracks):
        ok = (code >= 0) & np.isfinite(c).all(axis=1)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        ss, cc, fr = code[idx].astype(float), c[idx], start + idx
        rm = ss == 0
        if rm.any():
            f0 = fr[rm]
            good = (f0 >= 0) & (f0 < n_all)
            f0, c0 = f0[good], cc[rm][good]
            np.add.at(rest_tot, f0, 1)
            for z in range(nz):
                ins = geom.paths[z].contains_points(c0)
                if ins.any():
                    np.add.at(rest_in[z], f0[ins], 1)
        if not (ss == 1).any():
            continue
        bnd = np.flatnonzero(np.diff(ss)) + 1
        starts = np.r_[0, bnd]
        ends = np.r_[bnd, len(ss)] - 1
        for a, b in zip(starts, ends, strict=True):
            if ss[a] != 1:
                continue
            ls = a > 0 and (fr[a] - fr[a - 1]) <= MAXGAP
            le = b < len(ss) - 1 and (fr[b + 1] - fr[b]) <= MAXGAP
            p0 = cc[a - 1] if ls else cc[a]
            p1 = cc[b + 1] if le else cc[b]
            seg = cc[a : b + 1]
            step = np.hypot(*(seg[1:] - seg[:-1]).T) if len(seg) > 1 else np.zeros(1)
            bouts.append(
                [
                    tid,
                    fr[a],
                    fr[b],
                    p0[0],
                    p0[1],
                    p1[0],
                    p1[1],
                    float(ls),
                    float(le),
                    float(step.sum()),
                    len(seg),
                    float(step.mean()),
                ]
            )
    return {
        "bouts": np.array(bouts, float).reshape(-1, 12),
        "rest_tot": rest_tot,
        "rest_in": rest_in,
        "zone_names": np.array(geom.names),
    }


def extract_traj(
    moving: Mapping[str, np.ndarray], geom: FlightGeometry, nmax: int, mult: float, min_median: float = 0.0
) -> Arrays:
    nz, n_all = len(geom.names), nmax
    width, s, cents = geom.width, geom.s, geom.cents
    dist_sum = np.zeros((nz, n_all), np.float64)
    cnt = np.zeros(n_all, np.int32)
    xy_list: list[np.ndarray] = []
    fr_list: list[np.ndarray] = []
    lens: list[int] = []
    for _, fi, co in _tracks(moving, n_all):
        n = len(fi)
        if n < MIN_FLYER_PTS:
            continue
        step = np.hypot(*(co[1:] - co[:-1]).T)
        if step.mean() < FLYER_STEP * s * mult or not _is_sustained(step, min_median, s):
            continue
        np.add.at(cnt, fi, 1)
        for z in range(nz):
            np.add.at(dist_sum[z], fi, np.hypot(co[:, 0] - cents[z, 0], co[:, 1] - cents[z, 1]) / width)
        xy_list.append(co.astype(np.float32))
        fr_list.append(fi.astype(np.int32))
        lens.append(n)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return {
        "dist_sum": dist_sum,
        "cnt": cnt,
        "off": off,
        "xy": np.concatenate(xy_list) if xy_list else np.zeros((0, 2), np.float32),
        "fr": np.concatenate(fr_list) if fr_list else np.zeros(0, np.int32),
        "zone_names": np.array(geom.names),
    }


def chunk_arrays(
    moving: Mapping[str, np.ndarray] | None,
    resting: Mapping[str, np.ndarray] | None,
    tracks: Mapping[str, np.ndarray] | None,
    geom: FlightGeometry,
    nmax: int,
    mult: float = 1.0,
    min_median: float = 0.0,
) -> dict[str, Any]:
    """All three array sets for one chunk (a set is None when its input tier is missing)."""
    return {
        "main": extract_main(moving, resting, geom, nmax, mult, min_median) if moving is not None else None,
        "links": extract_links(tracks, geom, nmax) if tracks is not None else None,
        "traj": extract_traj(moving, geom, nmax, mult, min_median) if moving is not None else None,
    }
