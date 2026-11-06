"""Clock-corrected per-minute aggregation, per hour-cycle window metrics and paired ON-vs-OFF statistics.

Ported from fp_build.py, fp_links_build.py, fp_dist_build.py and the counting part of fp_traj_figs.scan.
Zone slots are re-ordered to roles by geometry: speaker, right (= the wall opposite the speaker, shown as
"Opposite wall"), top, bottom (image orientation).

Dropped on purpose (both failed the Ae. aegypti positive control): the approach index (mean cos of the step
direction toward a zone) and the per-bout approach gain (start vs end distance, 'toward / away').
"""
import os
import warnings

import numpy as np

from . import clock as fk
from . import config as fc

ROLES = fc.ROLES


def _grid(params, window):
    import pandas as pd
    w0, w1 = window
    t0 = w0.floor("h")
    n_min = int(np.ceil((w1 - t0) / pd.Timedelta(minutes=1)))
    return t0, n_min


def _minutes_frame(A, t0, n_min, times=None):
    import pandas as pd
    centers = times if times is not None else t0 + pd.to_timedelta(np.arange(n_min) + 0.5, unit="m")
    cols = {"time": centers}
    for nm, v in A.items():
        if v.ndim == 1:
            cols[nm] = v
        else:
            for i, r in enumerate(ROLES):
                cols["%s|%s" % (nm, r)] = v[i]
    M = pd.DataFrame(cols)
    M["hour"] = M.time.dt.floor("h")
    M["moh"] = M.time.dt.minute
    return M


# ------------------------------------------------------------------------------------- main metrics
def accumulate(main_dir, chunks, clock, order, params, window, nf, nmax, s, cancel=None):
    """per-minute sums of every main-cache array on the calibrated clock (fp_build.accumulate)."""
    import pandas as pd
    w0, w1 = window
    t0, n_min = _grid(params, window)
    nr_ = len(params.r_fracs())
    nz = 4
    on, off = params.on, params.off
    A = dict(frames=np.zeros(n_min), fly_n=np.zeros(n_min), all_n=np.zeros(n_min), fly_px=np.zeros(n_min),
             takeoff=np.zeros(n_min), rest_tot=np.zeros(n_min))
    for nm in ("entries", "path_near", "sector", "landing", "rest_in"):
        A[nm] = np.zeros((nz, n_min))
    for r in range(nr_):
        A["near_r%d" % r] = np.zeros((nz, n_min))
    heat = {}
    nblk = {}
    used = 0
    for stem, sess, k in chunks:
        if cancel is not None and cancel.is_set():
            raise fc.Cancelled()
        times = fk.frame_times(clock, sess, k, np.arange(nf))
        if times is None or times[-1] < w0 or times[0] >= w1:
            continue
        p = os.path.join(main_dir, stem + ".npz")
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        amin = np.floor((times - t0) / pd.Timedelta(minutes=1)).astype(int)
        inw = (times >= w0) & (times < w1) & (amin >= 0) & (amin < n_min)
        a = amin[inw]

        def bc(src):
            return np.bincount(a, weights=src[:nf][inw].astype(float), minlength=n_min)[:n_min]
        A["frames"] += np.bincount(a, minlength=n_min)[:n_min]
        for nm in ("fly_n", "all_n", "fly_px", "takeoff"):
            A[nm] += bc(z[nm])
        rt = np.minimum(z["rest_tot"][:nf].astype(float), fc.COLONY)
        A["rest_tot"] += bc(np.concatenate([rt, np.zeros(nmax - nf)]))
        for i, zi in enumerate(order):
            ri = np.minimum(z["rest_in"][zi][:nf].astype(float), rt)
            A["rest_in"][i] += bc(np.concatenate([ri, np.zeros(nmax - nf)]))
            for nm, src in (("entries", "entries"), ("path_near", "path_near"), ("sector", "sector_n"),
                            ("landing", "landing_in")):
                A[nm][i] += bc(z[src][zi])
            for r in range(nr_):
                A["near_r%d" % r][i] += bc(z["near_n"][r, zi])
        # heat-map blocks (10 s) by calibrated block-centre time
        H = z["heat"]
        for b in range(nf // fc.BLOCK):
            tm = times[b * fc.BLOCK + fc.BLOCK // 2]
            if not (w0 <= tm < w1):
                continue
            moh = tm.minute
            win = "ON" if on[0] <= moh < on[1] else ("OFF" if off[0] <= moh < off[1] else None)
            if win is None:
                continue
            kk = (params.light_of(tm.hour), win)
            heat[kk] = heat.get(kk, 0) + H[b]
            nblk[kk] = nblk.get(kk, 0) + 1
        used += 1
    M = _minutes_frame(A, t0, n_min)
    return dict(minutes=M, heat=heat, nblk=nblk, s=s, chunks_used=used,
                minutes_with_data=int((M.frames > 0).sum()))


def window_sums(M, lo, hi):
    sel = M[(M.moh >= lo) & (M.moh < hi)]
    cols = [c for c in M.columns if c not in ("time", "hour", "moh")]
    return sel.groupby("hour")[cols].sum()


def window_metrics(S, s, n_radii=3):
    import pandas as pd
    mins = S.frames / (60.0 * fc.FPS_EFF)
    X = pd.DataFrame(index=S.index)
    nan = np.nan
    X["flyers_per_frame"] = S.fly_n / S.frames
    X["alldet_per_frame"] = S.all_n / S.frames
    X["flight_px_per_frame"] = S.fly_px / S.frames
    X["takeoffs_per_min"] = S.takeoff / mins
    sec_tot = sum(S["sector|%s" % r] for r in ROLES)
    for r in ROLES:
        for i in range(n_radii):
            X["share_near_r%d|%s" % (i, r)] = S["near_r%d|%s" % (i, r)] / S.fly_n.replace(0, nan)
        X["share_sector|%s" % r] = S["sector|%s" % r] / sec_tot.replace(0, nan)
        X["entries_per_min|%s" % r] = S["entries|%s" % r] / mins
        X["dwell_frames_per_visit|%s" % r] = S["near_r1|%s" % r] / S["entries|%s" % r].replace(0, nan)
        X["speed_in_nbhd|%s" % r] = S["path_near|%s" % r] / S["near_r1|%s" % r].replace(0, nan) / s
        X["landings_per_min|%s" % r] = S["landing|%s" % r] / mins
        X["rest_prop|%s" % r] = S["rest_in|%s" % r] / S.rest_tot.replace(0, nan)
    return X


def _cycles(M, params, metrics_fn, extra_mask=None):
    """long table: hour x window -> metrics, only windows with >= 90% frame coverage."""
    import pandas as pd
    out = []
    for w, (lo, hi) in params.windows().items():
        S = window_sums(M, lo, hi)
        keep = S.frames >= 0.9 * (hi - lo) * 60 * fc.FPS_EFF
        if extra_mask is not None:
            keep = keep & extra_mask(S)
        S = S[keep]
        X = metrics_fn(S)
        X.insert(0, "window", w)
        out.append(X.reset_index())
    return pd.concat(out, ignore_index=True)


def cycle_metrics(acc, params):
    return _cycles(acc["minutes"], params, lambda S: window_metrics(S, acc["s"], len(params.r_fracs())))


# ------------------------------------------------------------------------------------- statistics
def _wilcoxon(d, p, q):
    from scipy.stats import wilcoxon
    if len(d) < 4 or not (d != 0).any():
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return (wilcoxon(p, q, alternative="greater").pvalue, wilcoxon(p, q).pvalue)
        except ValueError:
            return np.nan, np.nan


def stat_table(CM, label, params):
    """paired tests per (pair, metric, zone, grouping) (fp_build.stat_table): one-sided Wilcoxon ON > OFF
    (two-sided also), dz = mean(ON-OFF)/SD(ON-OFF), and difference-in-differences of the speaker vs the mean of
    the controls and vs each control. CM may be pooled over experiments when it carries a 'cyc' column."""
    import pandas as pd
    if "cyc" not in CM.columns:
        CM = CM.assign(cyc=CM.hour.astype(str))
    metric_cols = [c for c in CM.columns if c not in ("hour", "window", "cyc", "exp")]
    base_metrics = sorted(set(c.split("|")[0] for c in metric_cols))
    rows = []
    windows = params.windows()
    wide = dict((w, CM[CM.window == w].set_index("cyc")) for w in windows)
    LIGHT, DARK = params.light_label, params.dark_label
    for pair, won, woff in params.pairs():
        on_t, off_t = wide[won], wide[woff]
        idx = on_t.index.intersection(off_t.index)
        on_t, off_t = on_t.loc[idx], off_t.loc[idx]
        hrs = pd.to_datetime(on_t.hour).dt.hour
        groups = [("all", "all", np.ones(len(idx), bool))]
        for lab in (LIGHT, DARK):
            groups.append(("light", lab, hrs.map(params.light_of).values == lab))
        for pn, _ in params.phases():
            groups.append(("phase", pn, hrs.map(params.phase_of).values == pn))
        for bm in base_metrics:
            zoned = ("%s|speaker" % bm) in on_t.columns
            series = {}
            if zoned:
                for r in ROLES:
                    series[r] = (on_t["%s|%s" % (bm, r)].values, off_t["%s|%s" % (bm, r)].values)
                d = dict((r, series[r][0] - series[r][1]) for r in ROLES)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")      # all-NaN columns (no controls data in a cycle)
                    ctrl = np.nanmean(np.vstack([d[r] for r in ROLES[1:]]), axis=0)
                series["speaker - mean(controls)"] = (d["speaker"] - ctrl, np.zeros(len(idx)))
                for r in ROLES[1:]:
                    series["speaker - %s" % r] = (d["speaker"] - d[r], np.zeros(len(idx)))
            else:
                series["all"] = (on_t[bm].values, off_t[bm].values)
            for zname, (von, voff) in series.items():
                for gname, gval, gmask in groups:
                    m = gmask & np.isfinite(von) & np.isfinite(voff)
                    n = int(m.sum())
                    if n < 3:
                        continue
                    a, b = von[m], voff[m]
                    dd = a - b
                    pg, p2 = _wilcoxon(dd, a, b)
                    sd = dd.std(ddof=1) if n > 1 else np.nan
                    rows.append(dict(exp=label, pair=pair, metric=bm, zone=zname, zone_label=fc.role_label(zname),
                                     grouping=gname, group=gval,
                                     n_cycles=n, mean_off=b.mean(), mean_on=a.mean(), delta=dd.mean(),
                                     fold=(a.mean() / b.mean()) if b.mean() not in (0, np.nan) and zname in ROLES + ["all"] else np.nan,
                                     dz=dd.mean() / sd if sd and sd > 0 else np.nan, n_on_gt_off=int((dd > 0).sum()),
                                     p_on_gt_off=pg, p_two_sided=p2))
    return pd.DataFrame(rows)


def pool_cycle_metrics(tables, params, label="pooled"):
    """Hook for the (not yet ported) multi-experiment comparison: pool several experiments' cycle-metric tables
    with each (experiment, hour) as a replicate, like the research run's pooled Ae. aegypti cages.
    tables: list of (experiment_label, cycle_metrics DataFrame). Returns (pooled CM, stats table)."""
    import pandas as pd
    frames = []
    for exp_label, cm in tables:
        cm = cm.copy()
        cm["exp"] = exp_label
        cm["cyc"] = exp_label + "_" + cm.hour.astype(str)
        frames.append(cm)
    pool = pd.concat(frames, ignore_index=True)
    return pool, stat_table(pool, label, params)


# ------------------------------------------------------------------------------------------ links
def accumulate_links(links_dir, chunks, clock, order, G, params, window, nf, base_times, cancel=None):
    """per-minute take-off / landing link sums (fp_links_build.accumulate, minus the approach gain)."""
    import pandas as pd
    w0, w1 = window
    n_min = len(base_times)
    t0 = base_times.iloc[0].floor("h")
    nz = 4
    A = dict(frames=np.zeros(n_min), takeoffs=np.zeros(n_min), landings=np.zeros(n_min), bouts_both=np.zeros(n_min),
             rest_tot_frames=np.zeros(n_min))
    for nm in ("takeoff_poly", "landing_poly", "takeoff_nb", "landing_nb", "bout_start_out", "bout_arrive",
               "rest_frames_poly"):
        A[nm] = np.zeros((nz, n_min))
    dms = [G["dms"][i] for i in order]
    paths = [G["paths"][i] for i in order]
    rnb = G["Rpx"][fc.R_MAIN]
    thr = fc.FLYER_STEP * G["s"] * params.flyer_mult
    used = 0
    for stem, sess, k in chunks:
        if cancel is not None and cancel.is_set():
            raise fc.Cancelled()
        p = os.path.join(links_dir, stem + ".npz")
        times = fk.frame_times(clock, sess, k, np.arange(nf))
        if times is None or times[-1] < w0 or times[0] >= w1 or not os.path.isfile(p):
            continue
        z = np.load(p)
        amin_f = np.floor((times - t0) / pd.Timedelta(minutes=1)).astype(int)
        inw_f = (times >= w0) & (times < w1) & (amin_f >= 0) & (amin_f < n_min)

        def minute_of(frames):
            f = np.clip(frames.astype(int), 0, nf - 1)
            return amin_f[f], inw_f[f]
        a = amin_f[inw_f]
        A["frames"] += np.bincount(a, minlength=n_min)[:n_min]
        A["rest_tot_frames"] += np.bincount(a, weights=z["rest_tot"][:nf][inw_f].astype(float), minlength=n_min)[:n_min]
        for i, zi in enumerate(order):
            A["rest_frames_poly"][i] += np.bincount(a, weights=z["rest_in"][zi][:nf][inw_f].astype(float), minlength=n_min)[:n_min]
        b = z["bouts"]
        if len(b):
            fly = (b[:, 10] >= fc.MIN_FLYER_PTS) & (b[:, 11] >= thr)
            b = b[fly]
            ls, le = b[:, 7] == 1, b[:, 8] == 1

            def dist_to(xy, zidx):
                xi = np.clip((xy[:, 0] / fc.DS).astype(int), 0, G["gw"] - 1)
                yi = np.clip((xy[:, 1] / fc.DS).astype(int), 0, G["gh"] - 1)
                return dms[zidx][yi, xi]
            m_to, ok_to = minute_of(b[:, 1])
            m_ld, ok_ld = minute_of(b[:, 2])
            p0, p1 = b[:, 3:5], b[:, 5:7]
            tk = ls & ok_to
            ld = le & ok_ld
            both = ls & le & ok_to
            A["takeoffs"] += np.bincount(m_to[tk], minlength=n_min)[:n_min]
            A["landings"] += np.bincount(m_ld[ld], minlength=n_min)[:n_min]
            A["bouts_both"] += np.bincount(m_to[both], minlength=n_min)[:n_min]
            for i in range(nz):
                d0, d1 = dist_to(p0, i), dist_to(p1, i)
                in0 = paths[i].contains_points(p0) if tk.any() else np.zeros(len(b), bool)
                in1 = paths[i].contains_points(p1) if ld.any() else np.zeros(len(b), bool)
                A["takeoff_poly"][i] += np.bincount(m_to[tk & in0], minlength=n_min)[:n_min]
                A["landing_poly"][i] += np.bincount(m_ld[ld & in1], minlength=n_min)[:n_min]
                A["takeoff_nb"][i] += np.bincount(m_to[tk & (d0 <= rnb)], minlength=n_min)[:n_min]
                A["landing_nb"][i] += np.bincount(m_ld[ld & (d1 <= rnb)], minlength=n_min)[:n_min]
                st_out = both & (d0 > rnb)
                A["bout_start_out"][i] += np.bincount(m_to[st_out], minlength=n_min)[:n_min]
                A["bout_arrive"][i] += np.bincount(m_to[st_out & (d1 <= rnb)], minlength=n_min)[:n_min]
        used += 1
    M = _minutes_frame(A, t0, n_min, times=base_times.values)
    return dict(minutes=M, chunks_used=used, takeoffs=float(A["takeoffs"].sum()),
                landings=float(A["landings"].sum()), bouts_both=float(A["bouts_both"].sum()))


def links_window_metrics(S):
    import pandas as pd
    mins = S.frames / (60.0 * fc.FPS_EFF)
    X = pd.DataFrame(index=S.index)
    X["takeoffs_per_min"] = S.takeoffs / mins
    X["landings_per_min"] = S.landings / mins
    X["linked_bouts_per_min"] = S.bouts_both / mins
    for r in ROLES:
        X["takeoff_poly_per_min|%s" % r] = S["takeoff_poly|%s" % r] / mins
        X["landing_poly_per_min|%s" % r] = S["landing_poly|%s" % r] / mins
        rest_min = S["rest_frames_poly|%s" % r] / (60.0 * fc.FPS_EFF)
        X["takeoff_hazard|%s" % r] = S["takeoff_poly|%s" % r] / rest_min.where(rest_min >= 5)
        X["takeoff_share_nb|%s" % r] = S["takeoff_nb|%s" % r] / S.takeoffs.replace(0, np.nan)
        X["landing_share_nb|%s" % r] = S["landing_nb|%s" % r] / S.landings.replace(0, np.nan)
        so = S["bout_start_out|%s" % r].where(S["bout_start_out|%s" % r] >= 5)
        X["arrival_prob|%s" % r] = S["bout_arrive|%s" % r] / so
    return X


def links_cycle_metrics(acc, params):
    return _cycles(acc["minutes"], params, links_window_metrics)


# --------------------------------------------------------------------------------------- distance
def accumulate_dist(traj_dir, chunks, clock, order, params, window, nf, cancel=None):
    """per-minute flyer count and summed distance to each zone centre, cage widths (fp_dist_build.accumulate)."""
    import pandas as pd
    w0, w1 = window
    t0, n_min = _grid(params, window)
    frames, cnt = np.zeros(n_min), np.zeros(n_min)
    dsum = np.zeros((4, n_min))
    used = 0
    for stem, sess, k in chunks:
        if cancel is not None and cancel.is_set():
            raise fc.Cancelled()
        p = os.path.join(traj_dir, stem + ".npz")
        times = fk.frame_times(clock, sess, k, np.arange(nf))
        if times is None or times[-1] < w0 or times[0] >= w1 or not os.path.isfile(p):
            continue
        z = np.load(p)
        amin = np.floor((times - t0) / pd.Timedelta(minutes=1)).astype(int)
        inw = (times >= w0) & (times < w1) & (amin >= 0) & (amin < n_min)
        a = amin[inw]
        frames += np.bincount(a, minlength=n_min)[:n_min]
        cnt += np.bincount(a, weights=z["cnt"][:nf][inw].astype(float), minlength=n_min)[:n_min]
        for i, zi in enumerate(order):
            dsum[i] += np.bincount(a, weights=z["dist_sum"][zi][:nf][inw], minlength=n_min)[:n_min]
        used += 1
    A = dict(frames=frames, cnt=cnt, dist_sum=dsum)
    return dict(minutes=_minutes_frame(A, t0, n_min), chunks_used=used)


def dist_cycle_metrics(acc, params):
    import pandas as pd

    def fn(S):
        X = pd.DataFrame(index=S.index)
        for r in ROLES:
            X["mean_dist|%s" % r] = S["dist_sum|%s" % r] / S.cnt
        return X
    return _cycles(acc["minutes"], params, fn, extra_mask=lambda S: S.cnt > 0)


# ---------------------------------------------------------------------------- trajectory pass-by
def trajectory_pass_summary(traj_dir, chunks, clock, order, G, params, window, cancel=None):
    """Share of flight tracks that come within the neighbourhood radius of each wall's zone, per light phase
    x tone window (counting part of fp_traj_figs.scan; a track's window is that of its first point)."""
    import pandas as pd
    w0, w1 = window
    rnb = G["Rpx"][fc.R_MAIN]
    dms = [G["dms"][i] for i in order]
    ls_, le_ = params.light_hours
    conds = [(lab, w) for lab in (params.light_label, params.dark_label) for w in ("OFF", "ON")]
    wins = {"OFF": params.off, "ON": params.on}
    res = dict((c, dict(n=0, **dict(("vis_%s" % r, 0) for r in ROLES))) for c in conds)
    for stem, sess, k in chunks:
        if cancel is not None and cancel.is_set():
            raise fc.Cancelled()
        p = os.path.join(traj_dir, stem + ".npz")
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        off = z["off"]
        if len(off) < 2:
            continue
        xy, fr = z["xy"], z["fr"]
        times = fk.frame_times(clock, sess, k, fr[off[:-1]])
        if times is None:
            continue
        inw = np.asarray((times >= w0) & (times < w1))
        moh, hr = times.minute.values, times.hour.values
        lightm = (hr >= ls_) & (hr < le_)
        xi = np.clip((xy[:, 0] / fc.DS).astype(int), 0, G["gw"] - 1)
        yi = np.clip((xy[:, 1] / fc.DS).astype(int), 0, G["gh"] - 1)
        tid = np.repeat(np.arange(len(off) - 1), np.diff(off))
        vis = dict((r, np.bincount(tid, weights=dms[i][yi, xi] <= rnb, minlength=len(off) - 1) > 0)
                   for i, r in enumerate(ROLES))
        for lab, lm in ((params.light_label, lightm), (params.dark_label, ~lightm)):
            for win, (lo, hi) in wins.items():
                idx = np.nonzero(inw & lm & (moh >= lo) & (moh < hi))[0]
                if not len(idx):
                    continue
                r_ = res[(lab, win)]
                r_["n"] += len(idx)
                for r in ROLES:
                    r_["vis_%s" % r] += int(vis[r][idx].sum())
    rows = []
    for (lab, win), r_ in res.items():
        n = max(r_["n"], 1)
        row = dict(phase=lab, tone=win, n_flights=r_["n"])
        for r in ROLES:
            row["pct_pass_%s" % r] = 100.0 * r_["vis_%s" % r] / n
        rows.append(row)
    return pd.DataFrame(rows)
