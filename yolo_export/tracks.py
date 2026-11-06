"""tracks: one segment's assembled tracks -> aligned per-point arrays, motion labels, per-frame
activity and behaviour events.

Alignment (see DEFINITIONS["alignment"]): a flying point's tracker index is its video frame + 2,
a resting point's is its own frame. Points are re-indexed to video frames, and a track with both a
resting and a flying point on one frame (around a take-off) keeps the flying one.

Motion: tracker state 1 is split by smoothed speed into real flight, slow movers (walking, which
the moving detector also picks up) and holds (a lost or landed mosquito held in place); state 0 is
resting unless it is displaced (walking). Only real flight gets the YOLO class "flying".

Events follow the validated flight-phonotaxis analysis (fp_links_extract / fp_links_build, DEVLOG
2026-10-20): bouts are runs of state 1 in tracker order that pass the flyer rule; a take-off /
landing is linked when the neighbouring resting point is <= link_max_gap_frames away; zone metrics
use per-zone distance maps (0.1 cage widths = "near").
"""
import numpy as np

from yolo_export import common as C

_KEY_SHIFT = 21          # key = track_id << 21 | frame  (frames < 2,097,152)

EVENT_COLUMNS = ["event", "video_frame", "start_frame", "end_frame", "track_id", "bout_id", "x", "y",
                 "zone", "near", "dist_cw", "value", "n_points", "mean_step_cw", "linked_start",
                 "linked_end"]           # + site_occupied, verified (site_checks)


def raw_points(mt):
    """Every valid point of every assembled track, in (track, tracker index) order.
    Returns dict of arrays tid, idx, x, y, state (int8), plus n_tracks and n_bad (skipped tracks)."""
    tids, idxs, xs, ys, sts = [], [], [], [], []
    n_bad = 0
    for tid, obj in mt.objects.items():
        coords = obj.get("coordinates") or []
        state = obj.get("state") or []
        n = min(len(coords), len(state))
        if n == 0:
            continue
        try:
            start = int(obj.get("start"))
            c = np.asarray(coords[:n], dtype=float).reshape(n, -1)
            s = np.asarray(state[:n], dtype=float)
        except (TypeError, ValueError):
            n_bad += 1
            continue
        if c.shape[1] < 2:
            n_bad += 1
            continue
        ok = np.isfinite(c[:, 0]) & np.isfinite(c[:, 1]) & ((s == 0) | (s == 1))
        k = np.nonzero(ok)[0]
        if len(k) == 0:
            continue
        tids.append(np.full(len(k), int(tid), np.int64))
        idxs.append(start + k.astype(np.int64))
        xs.append(c[k, 0])
        ys.append(c[k, 1])
        sts.append(s[k].astype(np.int8))
    if not tids:
        e = np.zeros(0)
        return {"tid": e.astype(np.int64), "idx": e.astype(np.int64), "x": e, "y": e,
                "state": e.astype(np.int8), "n_tracks": 0, "n_bad": n_bad}
    return {"tid": np.concatenate(tids), "idx": np.concatenate(idxs), "x": np.concatenate(xs),
            "y": np.concatenate(ys), "state": np.concatenate(sts), "n_tracks": len(tids), "n_bad": n_bad}


def keys_of(tid, frame):
    return (np.asarray(tid).astype(np.int64) << _KEY_SHIFT) + np.asarray(frame).astype(np.int64)


def lookup(keys_sorted, want):
    """Positions of `want` in the sorted key array (-1 where absent)."""
    want = np.asarray(want)
    if len(keys_sorted) == 0:
        return np.full(want.shape, -1, np.int64)
    pos = np.searchsorted(keys_sorted, want)
    pos_c = np.minimum(pos, len(keys_sorted) - 1)
    hit = (pos < len(keys_sorted)) & (keys_sorted[pos_c] == want)
    return np.where(hit, pos_c, -1)


def _pair_dist(P, i, j, ok):
    d = np.full(len(i), np.nan)
    d[ok] = np.hypot(P["x"][i[ok]] - P["x"][j[ok]], P["y"][i[ok]] - P["y"][j[ok]])
    return d


def aligned_points(raw, defs, cw, centroid, zmaps):
    """Aligned per-point arrays sorted by (tid, video frame), with motion labels.

    Keys: tid, vf (video frame), idx (tracker index), x, y, state, speed (px/frame), motion
    (0 rest / 1 walk / 2 fly / 3 hold), real (real flight), cls (YOLO class), zone (-1 or index of
    the zone containing the point), rdist (px to the cage centroid), keys (sorted lookup keys)."""
    b = defs["behaviour"]
    off = int(defs["alignment"]["flying_offset_frames"])
    state = raw["state"]
    vf_all = raw["idx"] - off * (state == 1)
    order = np.lexsort((-state.astype(np.int16), vf_all, raw["tid"]))
    tid_o, vf_o = raw["tid"][order], vf_all[order]
    keep = np.ones(len(order), bool)
    keep[1:] = (tid_o[1:] != tid_o[:-1]) | (vf_o[1:] != vf_o[:-1])
    sel = order[keep]
    P = {"tid": raw["tid"][sel], "vf": vf_all[sel], "idx": raw["idx"][sel], "x": raw["x"][sel],
         "y": raw["y"][sel], "state": state[sel]}
    n = len(sel)
    keys = keys_of(P["tid"], P["vf"])
    P["keys"] = keys
    st1 = P["state"] == 1

    # smoothed speed |p(t+h) - p(t-h)| / 2h (BuzzSwarm smoothed_motion, h = 2); +-1 as a fallback
    h = int(b["speed_half_window_frames"])
    ip, im = lookup(keys, keys + h), lookup(keys, keys - h)
    speed = _pair_dist(P, ip, im, (ip >= 0) & (im >= 0)) / (2.0 * h)
    miss = ~np.isfinite(speed)
    if miss.any():
        ip1, im1 = lookup(keys, keys + 1), lookup(keys, keys - 1)
        both = (ip1 >= 0) & (im1 >= 0)
        s1 = _pair_dist(P, ip1, im1, both) / 2.0
        one = ~both & ((ip1 >= 0) | (im1 >= 0))
        nb = np.where(ip1 >= 0, ip1, im1)
        s1[one] = np.hypot(P["x"][one] - P["x"][nb[one]], P["y"][one] - P["y"][nb[one]])
        speed[miss] = s1[miss]
    P["speed"] = speed

    # walking (state 0): displaced >= walking_disp_cw between t-12 and t+12, both points resting
    w = int(b["walking_half_window_frames"])
    ip, im = lookup(keys, keys + w), lookup(keys, keys - w)
    ok = (ip >= 0) & (im >= 0) & ~st1
    ok[ok] &= (P["state"][ip[ok]] == 0) & (P["state"][im[ok]] == 0)
    disp = _pair_dist(P, ip, im, ok)
    walk0 = ok & (np.nan_to_num(disp, nan=0.0) >= b["walking_disp_cw"] * cw)

    v_fly = b["real_flight_speed_cw_per_frame"] * cw
    v_hold = b["hold_speed_cw_per_frame"] * cw
    known = np.isfinite(speed)
    motion = np.zeros(n, np.int8)
    motion[walk0] = 1
    motion[st1] = 2
    motion[st1 & known & (speed < v_fly)] = 1
    motion[st1 & known & (speed < v_hold)] = 3
    P["motion"] = motion
    P["real"] = st1 & known & (speed >= v_fly)
    P["cls"] = (motion == 2).astype(np.int8)
    P["rdist"] = np.hypot(P["x"] - centroid[0], P["y"] - centroid[1])
    if zmaps is not None and zmaps["names"] and n:
        P["zone"] = C.map_lookup(zmaps["inside"], zmaps["gw"], zmaps["gh"], P["x"], P["y"]).astype(np.int8) - 1
    else:
        P["zone"] = np.full(n, -1, np.int8)
    return P


def activity(P, T, defs, cw, n_zones):
    """Per-frame arrays (length T): counts per motion class, tracker-flying, real flyers, swarm
    (real flyers in the central disc), flight pixels (clock burst series) and resting per zone."""
    b = defs["behaviour"]
    vf = P["vf"]
    inr = (vf >= 0) & (vf < T)

    def count(mask):
        return np.bincount(vf[inr & mask], minlength=T)[:T].astype(np.int16)

    out = {"n_rest": count(P["motion"] == 0), "n_walk": count(P["motion"] == 1),
           "n_fly": count(P["motion"] == 2), "n_hold": count(P["motion"] == 3),
           "n_state1": count(P["state"] == 1), "n_real": count(P["real"]),
           "n_swarm": count(P["real"] & (P["rdist"] <= b["swarm_radius_cw"] * cw))}
    prev = lookup(P["keys"], P["keys"] - 1)
    ok = (prev >= 0) & (P["state"] == 1)
    ok[ok] &= P["state"][prev[ok]] == 1
    step = np.zeros(len(vf))
    step[ok] = np.hypot(P["x"][ok] - P["x"][prev[ok]], P["y"][ok] - P["y"][prev[ok]])
    fast = ok & (step >= b["fly_px_step_cw_per_s"] * cw / C.NOMINAL_FPS) & inr
    out["fly_px"] = np.bincount(vf[fast], weights=step[fast], minlength=T)[:T].astype(np.float32)
    rz = np.zeros((max(n_zones, 0), T), np.int16)
    for z in range(n_zones):
        rz[z] = count(((P["motion"] == 0) | (P["motion"] == 1)) & (P["zone"] == z))
    out["rest_zone"] = rz
    return out


def _runs(mask_values, group):
    """(start, end) index pairs of maximal runs where `group` is constant and the mask is True."""
    m = np.asarray(mask_values, bool)
    if not m.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    new = np.ones(len(m), bool)
    new[1:] = (group[1:] != group[:-1]) | (m[1:] != m[:-1])
    starts = np.nonzero(new)[0]
    ends = np.r_[starts[1:], len(m)] - 1
    keep = m[starts]
    return starts[keep], ends[keep]


def _frame(cols):
    import pandas as pd
    n = max((len(v) for v in cols.values() if hasattr(v, "__len__") and not isinstance(v, str)), default=0)
    out = {}
    for c in EVENT_COLUMNS:
        v = cols.get(c)
        if v is None:
            v = {"zone": "", "near": ""}.get(c, np.nan)
        out[c] = v if (hasattr(v, "__len__") and not isinstance(v, str)) else [v] * n
    return pd.DataFrame(out, columns=EVENT_COLUMNS)


def events(raw, P, act, defs, cw, zmaps):
    """Behaviour events of one segment as a DataFrame (EVENT_COLUMNS; frames are video frames)."""
    import pandas as pd
    b = defs["behaviour"]
    off = int(defs["alignment"]["flying_offset_frames"])
    names = list(zmaps["names"]) if zmaps is not None else []
    R = b["zone_radius_cw"] * cw
    parts = []

    tid, idx, x, y, st = raw["tid"], raw["idx"], raw["x"], raw["y"], raw["state"]
    n = len(tid)
    a, e = _runs(st == 1, tid * 2 + st)
    if len(a):
        steps = np.zeros(n)
        if n > 1:
            same = tid[1:] == tid[:-1]
            steps[1:][same] = np.hypot(np.diff(x)[same], np.diff(y)[same])
        csum = np.cumsum(steps)
        npts = e - a + 1
        path = csum[e] - csum[a]
        mean_step = np.where(npts > 1, path / np.maximum(npts - 1, 1), 0.0)
        fl = (npts >= b["flyer_min_points"]) & (mean_step >= b["flyer_mean_step_cw"] * cw)
        bout_id = np.nonzero(fl)[0]
        a, e, npts, mean_step = a[fl], e[fl], npts[fl], mean_step[fl]
    if len(a):
        maxgap = int(b["link_max_gap_frames"])
        am, ep = np.maximum(a - 1, 0), np.minimum(e + 1, n - 1)
        ls = (a > 0) & (tid[am] == tid[a]) & (st[am] == 0) & (idx[a] - idx[am] <= maxgap)
        le = (e < n - 1) & (tid[ep] == tid[e]) & (st[ep] == 0) & (idx[ep] - idx[e] <= maxgap)
        p0x, p0y = np.where(ls, x[am], x[a]), np.where(ls, y[am], y[a])
        p1x, p1y = np.where(le, x[ep], x[e]), np.where(le, y[ep], y[e])
        f_take = idx[a] - off
        f_land = np.where(le, idx[ep], idx[e] - off)
        base = {"start_frame": f_take, "end_frame": f_land, "track_id": tid[a], "bout_id": bout_id,
                "n_points": npts, "mean_step_cw": mean_step / cw, "linked_start": ls, "linked_end": le}

        def sub(mask, **kw):
            cols = {k: (v[mask] if hasattr(v, "__len__") else v) for k, v in base.items()}
            for k, v in kw.items():
                cols[k] = v[mask] if (hasattr(v, "__len__") and not isinstance(v, str)) else v
            return _frame(cols)

        in0 = np.full(len(a), "", object)
        in1 = np.full(len(a), "", object)
        near0 = np.full(len(a), "", object)
        near1 = np.full(len(a), "", object)
        d0, d1 = {}, {}
        if names:
            z0 = C.map_lookup(zmaps["inside"], zmaps["gw"], zmaps["gh"], p0x, p0y).astype(int) - 1
            z1 = C.map_lookup(zmaps["inside"], zmaps["gw"], zmaps["gh"], p1x, p1y).astype(int) - 1
            nm = np.array(names + [""], object)
            in0, in1 = nm[z0], nm[z1]
            for zi, zn in enumerate(names):
                d0[zn] = C.map_lookup(zmaps["dists"][zi], zmaps["gw"], zmaps["gh"], p0x, p0y)
                d1[zn] = C.map_lookup(zmaps["dists"][zi], zmaps["gw"], zmaps["gh"], p1x, p1y)
            near0 = np.array([",".join(zn for zn in names if d0[zn][i] <= R) for i in range(len(a))], object)
            near1 = np.array([",".join(zn for zn in names if d1[zn][i] <= R) for i in range(len(a))], object)
        parts.append(sub(ls, event="take_off", video_frame=f_take, x=p0x, y=p0y, zone=in0, near=near0))
        parts.append(sub(le, event="landing", video_frame=f_land, x=p1x, y=p1y, zone=in1, near=near1))
        if names:
            parts.append(sub(le & (in1 != ""), event="zone_landing", video_frame=f_land, x=p1x, y=p1y,
                             zone=in1, near=near1, dist_cw=np.zeros(len(a))))
            bounds = np.ravel(np.column_stack([a, e + 1]))
            for zi, zn in enumerate(names):
                d_all = np.r_[C.map_lookup(zmaps["dists"][zi], zmaps["gw"], zmaps["gh"], x, y), np.inf]
                dmin = np.minimum.reduceat(d_all, bounds)[::2]
                hit = np.nonzero(dmin <= R)[0]
                if len(hit):
                    jj = np.array([a[i] + int(np.argmin(d_all[a[i]:e[i] + 1])) for i in hit])
                    m = np.zeros(len(a), bool)
                    m[hit] = True
                    vf_c = np.zeros(len(a), np.int64)
                    xc, yc = np.zeros(len(a)), np.zeros(len(a))
                    vf_c[hit], xc[hit], yc[hit] = idx[jj] - off, x[jj], y[jj]
                    parts.append(sub(m, event="zone_pass", video_frame=vf_c, x=xc, y=yc, zone=zn,
                                     dist_cw=dmin / cw))
                arr = ls & le & (d0[zn] > R) & (d1[zn] <= R)
                parts.append(sub(arr, event="arrival", video_frame=f_land, x=p1x, y=p1y, zone=zn,
                                 dist_cw=d1[zn] / cw, value=d0[zn] / cw))

    # walking bouts: consecutive walking points of one track (aligned; gaps <= 2 frames)
    mot = P["motion"]
    if len(mot):
        brk = np.ones(len(mot), bool)
        brk[1:] = (P["tid"][1:] != P["tid"][:-1]) | (P["vf"][1:] - P["vf"][:-1] > 2)
        ws, we = _runs(mot == 1, np.cumsum(brk))
        long_ = (P["vf"][we] - P["vf"][ws] + 1) >= b["walk_bout_min_frames"]
        ws, we = ws[long_], we[long_]
        if len(ws):
            names_arr = np.array(names + [""], object)
            parts.append(_frame({
                "event": "walk_bout", "video_frame": P["vf"][ws], "start_frame": P["vf"][ws],
                "end_frame": P["vf"][we], "track_id": P["tid"][ws], "bout_id": np.full(len(ws), -1),
                "n_points": we - ws + 1, "x": P["x"][ws], "y": P["y"][ws],
                "zone": names_arr[P["zone"][ws].astype(int)] if names else np.full(len(ws), "", object),
                "value": np.hypot(P["x"][we] - P["x"][ws], P["y"][we] - P["y"][ws]) / cw,
                "linked_start": np.zeros(len(ws), bool), "linked_end": np.zeros(len(ws), bool)}))

    # swarm episodes: >= swarm_min_flyers real flyers in the central disc for >= swarm_min_frames
    sw = act["n_swarm"] >= b["swarm_min_flyers"]
    ss, se = _runs(sw, np.zeros(len(sw), np.int8))
    keep = (se - ss + 1) >= b["swarm_min_frames"]
    ss, se = ss[keep], se[keep]
    if len(ss):
        parts.append(_frame({
            "event": "swarm", "video_frame": ss, "start_frame": ss, "end_frame": se,
            "track_id": np.full(len(ss), -1), "bout_id": np.full(len(ss), -1), "n_points": se - ss + 1,
            "value": np.array([act["n_swarm"][i:j + 1].max() for i, j in zip(ss, se)], float),
            "linked_start": np.zeros(len(ss), bool), "linked_end": np.zeros(len(ss), bool)}))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS + ["site_occupied", "site_ok", "verified"])
    df = pd.concat(parts, ignore_index=True)
    df = site_checks(df, P, defs, cw)
    return df.sort_values(["video_frame", "event"], kind="stable").reset_index(drop=True)


def site_checks(df, P, defs, cw):
    """Verify take-offs and landings against the resting points around the site.

    After a real take-off the site empties; before a real landing it was empty. site_occupied is
    the share of frames in [+5, +25] after a take-off (or [-25, -5] before a landing / zone
    landing / arrival) with a resting point (any track) within site_radius_cw of the site; an
    event is verified when that share is below site_max_occupied. The tracker's assembly sometimes
    links a resting mosquito to a flyer that only passed by -- those take-offs leave the site
    occupied."""
    b = defs["behaviour"]
    r = b["site_radius_cw"] * cw
    w0, w1 = int(b["site_window_frames"][0]), int(b["site_window_frames"][1])
    rest = (P["motion"] == 0) | (P["motion"] == 1)
    vf, x, y = P["vf"][rest], P["x"][rest], P["y"][rest]
    o = np.argsort(vf, kind="stable")
    vf, x, y = vf[o], x[o], y[o]
    occ = np.full(len(df), np.nan)
    kinds = df.event.values
    for i in np.nonzero(np.isin(kinds, ["take_off", "landing", "zone_landing", "arrival"]))[0]:
        f = int(df.video_frame.values[i])
        lo_f, hi_f = (f + w0, f + w1) if kinds[i] == "take_off" else (f - w1, f - w0)
        lo, hi = np.searchsorted(vf, lo_f), np.searchsorted(vf, hi_f, side="right")
        if hi <= lo:
            occ[i] = 0.0
            continue
        near = np.hypot(x[lo:hi] - df.x.values[i], y[lo:hi] - df.y.values[i]) <= r
        occ[i] = len(np.unique(vf[lo:hi][near])) / float(hi_f - lo_f + 1)
    df["site_occupied"] = occ
    df["site_ok"] = np.where(np.isnan(occ), True, occ < b["site_max_occupied"])
    df["verified"] = df["site_ok"]          # verify.py adds the pixel check (px_ok) and updates this
    return df
