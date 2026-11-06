"""Read-only per-chunk pass over the tracking pickles -> small per-frame .npz caches (spawn-safe workers).

Ported from fp_extract.py, fp_links_extract.py and fp_traj_extract.py. One worker task per chunk loads
the chunk's tracking_moving pickle once and writes up to three caches:

  main   (fp_extract)       flyers, take-offs, near-zone counts, entries, wall sectors, landings, resting
                            totals / in-zone counts, flight px, 10-s heat maps
  links  (fp_links_extract) take-off / landing bouts from the assembled final_tracking_data tracks
  traj   (fp_traj_extract)  flyer coordinates + per-frame distance sums to each zone centre

Arrays are indexed by frame-within-chunk (clock-free); the clock model is applied later.
Dropped on purpose: the approach index (sumcos / ncos), which failed the Ae. aegypti positive control.
A cache is only written when its source pickles were read without error (a transient read failure, e.g.
a drive dropping out, is reported and retried next run rather than cached as zeros).
"""
import os

import numpy as np

from . import config as fc


def _cache_ok(out, sources):
    """cache exists and is at least as new as every source file that exists."""
    if not os.path.isfile(out):
        return False
    t = os.path.getmtime(out)
    for s in sources:
        if os.path.isfile(s) and os.path.getmtime(s) > t:
            return False
    return True


def _save(out, **arrays):
    d = os.path.dirname(out)
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)


# ------------------------------------------------------------------------------------------- main
def is_sustained(step, min_median, s):
    """optional dense-cage filter: a track whose median step is below min_median*s px/frame is mostly stationary
    with a few big jumps (linking to a neighbouring mosquito), not a flight. min_median <= 0 = off."""
    return min_median <= 0 or float(np.median(step)) >= min_median * s


def extract_main(obj_moving, obj_resting, G, nmax, mult, min_median=0.0):
    """fp_extract.one, minus the approach index. Returns the dict of arrays saved to the main cache.
    min_median > 0 additionally requires a sustained median step for the flyer arrays and fly_px."""
    s, names, nz = G["s"], G["names"], len(G["names"])
    N = nmax
    nblk = fc.nblk_for(nmax)
    nr_ = len(G["Rpx"])
    x0, x1, y0, y1 = G["g"]["bbox"]
    near_n = np.zeros((nr_, nz, N), np.int32)
    fly_n = np.zeros(N, np.int32)
    all_n = np.zeros(N, np.int32)
    fly_px = np.zeros(N, np.float64)
    takeoff = np.zeros(N, np.int32)
    takeoff_rest = np.zeros(N, np.int32)
    sector_n = np.zeros((nz, N), np.int32)
    entries = np.zeros((nz, N), np.int32)
    path_near = np.zeros((nz, N), np.float32)
    landing_in = np.zeros((nz, N), np.int32)
    land_rest_in = np.zeros((nz, N), np.int32)
    heat = np.zeros(nblk * fc.GRID * fc.GRID, np.int32)
    rows = []
    for td in (obj_moving.objects.values() if getattr(obj_moving, "objects", None) else []):
        ts_raw, co_raw = td.get("time_stamp", []), td.get("coordinates", [])
        n0 = min(len(ts_raw), len(co_raw))
        if n0 == 0:
            continue
        ts = np.asarray(ts_raw[:n0], float)
        co = np.asarray(co_raw[:n0], float)
        ok = np.isfinite(ts) & np.isfinite(co).all(axis=1)
        ts, co = ts[ok], co[ok]
        fi = ts.astype(np.int64)
        ok = (fi >= 0) & (fi < N)
        fi, co = fi[ok], co[ok]
        n = len(fi)
        if n == 0:
            continue
        np.add.at(all_n, fi, 1)
        if n < 2:
            continue
        d = co[1:] - co[:-1]
        step = np.hypot(d[:, 0], d[:, 1])
        dtf = np.diff(fi).astype(float)
        dt = dtf / fc.FPS
        dt[dt <= 0] = 1.0 / fc.FPS
        fast = (step / dt) >= fc.FLY_PX_SPEED * s
        sustained = is_sustained(step, min_median, s)
        if fast.any() and sustained:
            np.add.at(fly_px, fi[1:][fast], step[fast])
        mean_step = float(step.mean())
        if n < fc.MIN_FLYER_PTS or mean_step < fc.FLYER_STEP * s * mult or not sustained:
            continue
        # ---- flyer track ----
        st_rest = td.get("start_type") == "resting"
        en_rest = td.get("end_type") == "resting"
        np.add.at(fly_n, fi, 1)
        takeoff[fi[0]] += 1
        if st_rest:
            takeoff_rest[fi[0]] += 1
        xi = np.clip((co[:, 0] / fc.DS).astype(np.int64), 0, G["gw"] - 1)
        yi = np.clip((co[:, 1] / fc.DS).astype(np.int64), 0, G["gh"] - 1)
        D = np.stack([dm[yi, xi] for dm in G["dms"]])            # (nz, n) distance to each polygon
        for r, Rpx in enumerate(G["Rpx"]):
            nr = D <= Rpx
            for z in range(nz):
                if nr[z].any():
                    np.add.at(near_n[r, z], fi[nr[z]], 1)
        nm = D <= G["Rpx"][fc.R_MAIN]
        ent = nm[:, 1:] & ~nm[:, :-1]
        inside_start = nm[:, :-1]
        for z in range(nz):
            if ent[z].any():
                np.add.at(entries[z], fi[1:][ent[z]], 1)
            if inside_start[z].any():
                np.add.at(path_near[z], fi[1:][inside_start[z]], step[inside_start[z]].astype(np.float32))
        # wall sector of each point
        ang = np.arctan2(co[:, 1] - G["ctr"][1], co[:, 0] - G["ctr"][0])
        pr = (G["sign"] * (ang - G["cang0"])) % (2 * np.pi)
        sec = np.searchsorted(G["rel"][1:], pr, side="right")
        zsec = G["sec2zone"][sec]
        for z in range(nz):
            sel = zsec == z
            if sel.any():
                np.add.at(sector_n[z], fi[sel], 1)
        # landings: track end inside the zone polygon
        ex, ey = co[-1]
        for z in range(nz):
            if G["paths"][z].contains_point((ex, ey)):
                landing_in[z][fi[-1]] += 1
                if en_rest:
                    land_rest_in[z][fi[-1]] += 1
        # heat map of flyer positions per 10-s block
        gx = ((co[:, 0] - x0) / (x1 - x0) * fc.GRID).astype(np.int64)
        gy = ((co[:, 1] - y0) / (y1 - y0) * fc.GRID).astype(np.int64)
        inb = (gx >= 0) & (gx < fc.GRID) & (gy >= 0) & (gy < fc.GRID)
        flat = (fi[inb] // fc.BLOCK) * fc.GRID * fc.GRID + gy[inb] * fc.GRID + gx[inb]
        heat += np.bincount(flat, minlength=heat.size).astype(np.int32)
        rows.append([fi[0], fi[-1], n, mean_step, float(step.sum()), float(st_rest), float(en_rest), ex, ey]
                    + [float(D[z].min()) for z in range(nz)] + [int(nm[z].sum()) for z in range(nz)])
    rest_tot = np.zeros(N, np.int32)
    rest_in = np.zeros((nz, N), np.int32)
    if obj_resting is not None:
        for m in (obj_resting.objects.values() if getattr(obj_resting, "objects", None) else []):
            ts, co = m.get("time_stamp", []), m.get("coordinates", [])
            n = min(len(ts), len(co))
            if n == 0:
                continue
            ts = np.asarray(ts[:n], float)
            co = np.asarray(co[:n], float)
            ok = np.isfinite(ts) & np.isfinite(co).all(axis=1)
            if not ok.any():
                continue
            fi = ts[ok].astype(np.int64)
            co = co[ok]
            good = (fi >= 0) & (fi < N)
            fi, co = fi[good], co[good]
            np.add.at(rest_tot, fi, 1)
            for z in range(nz):
                inside = G["paths"][z].contains_points(co)
                np.add.at(rest_in[z], fi[inside], 1)
    return dict(fly_n=fly_n, all_n=all_n, fly_px=fly_px, takeoff=takeoff, takeoff_rest=takeoff_rest,
                near_n=near_n, entries=entries, path_near=path_near, sector_n=sector_n,
                landing_in=landing_in, land_rest_in=land_rest_in, heat=heat.reshape(nblk, fc.GRID, fc.GRID),
                rest_tot=rest_tot, rest_in=rest_in, zone_names=np.array(names),
                tracks=np.array(rows, float).reshape(-1, 9 + 2 * nz))


# ------------------------------------------------------------------------------------------ links
def extract_links(mt, G, nmax):
    """fp_links_extract.one: flight bouts (state 1 runs) with take-off / landing sites + resting counts."""
    nz, N = len(G["names"]), nmax
    rest_tot = np.zeros(N, np.int32)
    rest_in = np.zeros((nz, N), np.int32)
    bouts = []
    for tid, t in mt.objects.items():
        s = np.asarray(t["state"], float)
        c = np.asarray(t["coordinates"], float)
        n0 = min(len(s), len(c))
        s, c = s[:n0], c[:n0]
        ok = np.isfinite(s) & np.isfinite(c).all(axis=1)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        ss, cc, fr = s[idx], c[idx], int(t["start"]) + idx
        rm = ss == 0
        if rm.any():
            f0 = fr[rm]
            good = (f0 >= 0) & (f0 < N)
            f0, c0 = f0[good], cc[rm][good]
            np.add.at(rest_tot, f0, 1)
            for z in range(nz):
                ins = G["paths"][z].contains_points(c0)
                if ins.any():
                    np.add.at(rest_in[z], f0[ins], 1)
        if not (ss == 1).any():
            continue
        bnd = np.flatnonzero(np.diff(ss)) + 1
        starts = np.r_[0, bnd]
        ends = np.r_[bnd, len(ss)] - 1
        for a, b in zip(starts, ends):
            if ss[a] != 1:
                continue
            ls = a > 0 and (fr[a] - fr[a - 1]) <= fc.MAXGAP
            le = b < len(ss) - 1 and (fr[b + 1] - fr[b]) <= fc.MAXGAP
            p0 = cc[a - 1] if ls else cc[a]
            p1 = cc[b + 1] if le else cc[b]
            seg = cc[a:b + 1]
            step = np.hypot(*(seg[1:] - seg[:-1]).T) if len(seg) > 1 else np.zeros(1)
            bouts.append([tid, fr[a], fr[b], p0[0], p0[1], p1[0], p1[1], float(ls), float(le),
                          float(step.sum()), len(seg), float(step.mean())])
    return dict(bouts=np.array(bouts, float).reshape(-1, 12), rest_tot=rest_tot, rest_in=rest_in,
                zone_names=np.array(G["names"]))


# ------------------------------------------------------------------------------------------- traj
def extract_traj(obj_moving, G, nmax, mult, min_median=0.0):
    """fp_traj_extract.one: flyer coordinates + per-frame sums of distance to each zone centre (cage widths).
    The flyer threshold multiplier and the optional sustained-movement filter are applied here too (the research
    script always used x1 and no median filter)."""
    nz, N = len(G["names"]), nmax
    width, s = G["g"]["width"], G["s"]
    cents = G["cents"]
    dist_sum = np.zeros((nz, N), np.float64)
    cnt = np.zeros(N, np.int32)
    xy_list, fr_list, lens = [], [], []
    for td in (obj_moving.objects.values() if getattr(obj_moving, "objects", None) else []):
        ts_raw, co_raw = td.get("time_stamp", []), td.get("coordinates", [])
        n0 = min(len(ts_raw), len(co_raw))
        if n0 < fc.MIN_FLYER_PTS:
            continue
        ts = np.asarray(ts_raw[:n0], float)
        co = np.asarray(co_raw[:n0], float)
        ok = np.isfinite(ts) & np.isfinite(co).all(axis=1)
        ts, co = ts[ok], co[ok]
        fi = ts.astype(np.int64)
        ok = (fi >= 0) & (fi < N)
        fi, co = fi[ok], co[ok]
        n = len(fi)
        if n < fc.MIN_FLYER_PTS:
            continue
        step = np.hypot(*(co[1:] - co[:-1]).T)
        if step.mean() < fc.FLYER_STEP * s * mult or not is_sustained(step, min_median, s):
            continue
        np.add.at(cnt, fi, 1)
        for z in range(nz):
            np.add.at(dist_sum[z], fi, np.hypot(co[:, 0] - cents[z, 0], co[:, 1] - cents[z, 1]) / width)
        xy_list.append(co.astype(np.float32))
        fr_list.append(fi.astype(np.int32))
        lens.append(n)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return dict(dist_sum=dist_sum, cnt=cnt, off=off,
                xy=np.concatenate(xy_list) if xy_list else np.zeros((0, 2), np.float32),
                fr=np.concatenate(fr_list) if fr_list else np.zeros(0, np.int32), zone_names=np.array(G["names"]))


# ----------------------------------------------------------------------------------- worker entry
def chunk_jobs(ctx, chunks, kinds=("main", "links", "traj")):
    """(job, needs) for every chunk that has at least one stale / missing cache of the requested kinds."""
    exp = ctx["exp"]
    jobs = []
    n_cached = 0
    for stem, _sess, _k in chunks:
        mv, rs, fn = fc.moving_path(exp, stem), fc.resting_path(exp, stem), fc.final_path(exp, stem)
        need = []
        if "main" in kinds and not _cache_ok(os.path.join(ctx["dirs"]["main"], stem + ".npz"), [mv, rs]):
            need.append("main")
        if "links" in kinds and not _cache_ok(os.path.join(ctx["dirs"]["links"], stem + ".npz"), [fn]):
            need.append("links")
        if "traj" in kinds and not _cache_ok(os.path.join(ctx["dirs"]["traj"], stem + ".npz"), [mv]):
            need.append("traj")
        if need:
            jobs.append((ctx, stem, tuple(need)))
        else:
            n_cached += 1
    return jobs, n_cached


def process_chunk(job):
    """Top-level (spawn-safe) worker: build the requested caches for one chunk. Returns (stem, {kind: status})."""
    ctx, stem, need = job
    exp = ctx["exp"]
    G = fc.setup(ctx["geom"], ctx["r_fracs"])
    nmax, mult, med = ctx["nmax"], ctx["mult"], ctx.get("min_median", 0.0)
    status = {}
    obj_m = None
    if "main" in need or "traj" in need:
        try:
            obj_m = fc.load_pickle(fc.moving_path(exp, stem))
        except Exception as exc:  # noqa: BLE001
            for k in ("main", "traj"):
                if k in need:
                    status[k] = "moving pickle unreadable: %r" % (exc,)
    if "main" in need and "main" not in status:
        try:
            obj_r = fc.load_pickle(fc.resting_path(exp, stem))
        except Exception as exc:  # noqa: BLE001
            obj_r = None
            status["main"] = "resting pickle unreadable: %r" % (exc,)
        if obj_r is not None:
            try:
                arr = extract_main(obj_m, obj_r, G, nmax, mult, med)
                _save(os.path.join(ctx["dirs"]["main"], stem + ".npz"), **arr)
                status["main"] = "ok"
            except Exception as exc:  # noqa: BLE001
                status["main"] = "error: %r" % (exc,)
            del obj_r
    if "traj" in need and "traj" not in status:
        try:
            arr = extract_traj(obj_m, G, nmax, mult, med)
            _save(os.path.join(ctx["dirs"]["traj"], stem + ".npz"), **arr)
            status["traj"] = "ok"
        except Exception as exc:  # noqa: BLE001
            status["traj"] = "error: %r" % (exc,)
    obj_m = None
    if "links" in need:
        p = fc.final_path(exp, stem)
        if not os.path.isfile(p):
            status["links"] = "no final_tracking_data file"
        else:
            try:
                mt = fc.load_pickle(p)
                arr = extract_links(mt, G, nmax)
                del mt
                _save(os.path.join(ctx["dirs"]["links"], stem + ".npz"), **arr)
                status["links"] = "ok"
            except Exception as exc:  # noqa: BLE001
                status["links"] = "error: %r" % (exc,)
    return stem, status


def exit_with_parent():
    """Pool initializer: end this worker as soon as the process that created the pool is gone (app closed
    or killed mid-run) instead of reading files as an orphan. Same as batch_runner._exit_with_parent. Never raises."""
    try:
        import threading
        from multiprocessing import parent_process
        parent = parent_process()
        if parent is None:
            return

        def _watch():
            parent.join()
            os._exit(1)

        threading.Thread(target=_watch, name="exit-with-parent", daemon=True).start()
    except Exception:  # noqa: BLE001
        pass


def run_extraction(ctx, chunks, workers=1, progress=None, cancel=None, log=None, kinds=("main", "links", "traj")):
    """Build every missing / stale cache. progress(done, total); raises fc.Cancelled when cancel is set.
    Returns {stem: {kind: status}} for the chunks that were (re)processed, and the number already cached."""
    jobs, n_cached = chunk_jobs(ctx, chunks, kinds)
    total = len(jobs)
    results = {}
    if log:
        log("extraction: %d chunk(s) to read, %d already cached" % (total, n_cached))
    if progress:
        progress(0, total)
    if not jobs:
        return results, n_cached
    if workers and workers > 1 and total > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(processes=min(workers, total), maxtasksperchild=8,
                                            initializer=exit_with_parent)
        try:
            for i, (stem, st) in enumerate(pool.imap_unordered(process_chunk, jobs)):
                results[stem] = st
                if progress:
                    progress(i + 1, total)
                if cancel is not None and cancel.is_set():
                    raise fc.Cancelled()
            pool.close()
        except BaseException:
            pool.terminate()
            _remove_partials(ctx)
            raise
        finally:
            pool.join()
    else:
        for i, job in enumerate(jobs):
            if cancel is not None and cancel.is_set():
                raise fc.Cancelled()
            stem, st = process_chunk(job)
            results[stem] = st
            if progress:
                progress(i + 1, total)
    return results, n_cached


def _remove_partials(ctx):
    """delete half-written *.tmp.npz files left by terminated workers (finished caches are complete)."""
    for d in ctx["dirs"].values():
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".tmp.npz"):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass
