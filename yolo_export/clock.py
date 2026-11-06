"""clock: corrected wall-clock time for every frame, plus stimulus and light-phase flags.

Port of the flight-phonotaxis fp_clock.py (DEVLOG 2026-10-20). Chunks of one rpicam --segment
session are consecutive, so a frame's recording-clock time is secs = (k * NF + frame) / 25 after the
session start S0. The camera's true rate is slightly off 25 fps, so the flight burst evoked by the
tone at real :40:00 drifts. Fitted per experiment on those bursts:
    real time = S0 + secs - a_session - b * secs / 86400
(one intercept per session, one drift b per experiment, robust least squares). Without a stimulus or
enough bursts the clock falls back to a = 0 and the rig rate from selection.yaml (rig_fps).

Writes clock/<alias>.json (+ bursts csv), timeseries/<rig>/<alias>.parquet (per-second activity)
and events/<rig>/<alias>.parquet (all segment events with time, stim_on, stim_edge, light_phase).
"""
import os

import numpy as np

from yolo_export import annotate as AN
from yolo_export import common as C
from yolo_export import inventory as I

FPS = C.NOMINAL_FPS


def _s0(sess):
    import pandas as pd
    return pd.Timestamp("%s-%s-%s %s:%s:%s" % (sess[:4], sess[4:6], sess[6:8], sess[9:11], sess[11:13], sess[13:15]))


def _session_series(seg_dir, chunks, nf, bin_s):
    import pandas as pd
    kmax = max(k for k, _, _ in chunks)
    nb = int(np.ceil((kmax + 1) * nf / FPS / bin_s)) + 1
    px, cnt = np.zeros(nb), np.zeros(nb)
    for k, stem, _ in chunks:
        z = np.load(os.path.join(seg_dir, stem + ".activity.npz"))
        fp = z["fly_px"][:nf].astype(float)
        b = np.floor((k * nf + np.arange(len(fp))) / FPS / bin_s).astype(np.int64)
        ok = b < nb
        px += np.bincount(b[ok], weights=fp[ok], minlength=nb)[:nb]
        cnt += np.bincount(b[ok], minlength=nb)[:nb]
    ser = pd.Series(px, index=np.arange(nb) * bin_s)
    ser[cnt == 0] = np.nan
    return ser


def _burst_onsets(ser, s0, t_first, t_last, on_min):
    """fp_clock.burst_onsets with the window anchored on the stimulus onset minute."""
    import pandas as pd
    rows = []
    h = t_first.floor("h")
    while h < t_last:
        base_s = (h + pd.Timedelta(minutes=on_min - 10) - s0).total_seconds()
        end_s = (h + pd.Timedelta(minutes=on_min + 12) - s0).total_seconds()
        if base_s >= 0 and end_s <= ser.index[-1]:
            w = ser.loc[base_s:end_s]
            if len(w) and w.notna().mean() > 0.9:
                sm = w.rolling(3, center=True, min_periods=1).mean()
                base = sm.loc[:base_s + 8 * 60]
                thr = base.median() + 6 * (base.quantile(0.75) - base.quantile(0.25) + 1)
                cand = sm.loc[base_s + 8 * 60:]
                above = cand[cand > max(thr, 3 * base.median())]
                onset = None
                for t in above.index.values:
                    if (cand.loc[t:t + 30] > thr).mean() > 0.8:
                        onset = t
                        break
                if onset is not None:
                    rec = s0 + pd.Timedelta(seconds=float(onset))
                    rows.append({"hour": h, "onset_rec": rec,
                                 "offset_s": (rec - (h + pd.Timedelta(minutes=on_min))).total_seconds(),
                                 "days": (h - s0).total_seconds() / 86400,
                                 "peak_ratio": float(cand.max() / max(base.median(), 1))})
        h += pd.Timedelta(hours=1)
    return rows


def fit_experiment(row, out_dir, selection, log=C.log_default):
    """Fit (or fall back) the clock of one experiment; writes and returns clock/<alias>.json."""
    import pandas as pd
    rig, alias = row["rig_id"], row["alias"]
    infos = AN.segment_infos(out_dir, rig, alias, I.segment_set(row))
    if not infos:
        raise RuntimeError("no annotated segments for %s (run annotate first)" % alias)
    sessions = {}
    for stem, info in infos:
        sessions.setdefault(info["session"], []).append((int(info["k"]), stem, int(info["n_frames"])))
    lengths = [n for ch in sessions.values() for _, _, n in ch]
    nf = int(pd.Series(lengths).mode().iloc[0])
    fps_fb = float((selection.get("rig_fps") or {}).get(rig, FPS))
    res = {"alias": alias, "rig": rig, "nf": nf, "fitted": False, "a": {s: 0.0 for s in sessions},
           "b": 86400.0 * (1.0 - FPS / fps_fb), "fallback_fps": fps_fb, "n_bursts": 0,
           "resid_sd": float("nan"), "n_kept": 0, "a_imputed": [], "stim_windows": row["stim_windows"],
           "sessions": {s: min(k for k, _, _ in ch) for s, ch in sessions.items()}}
    windows = C.parse_windows(row["stim_windows"])
    df = pd.DataFrame()
    if windows:
        defs = C.definitions()
        seg_dir = AN.seg_dir(out_dir, rig, alias)
        rows = []
        for sess, chunks in sorted(sessions.items()):
            s0 = _s0(sess)
            ser = _session_series(seg_dir, chunks, nf, defs["clock"]["burst_bin_s"])
            t_last = s0 + pd.Timedelta(seconds=(max(k for k, _, _ in chunks) + 1) * nf / FPS)
            for r in _burst_onsets(ser, s0, s0, t_last, windows[0][0]):
                r["session"] = sess
                rows.append(r)
        df = pd.DataFrame(rows)
        res["n_bursts"] = int(len(df))
        keep_sess = [s for s in sorted(set(df.session))] if len(df) else []
        keep_sess = [s for s in keep_sess if (df.session == s).sum() >= 3]
        if len(df) >= 8 and keep_sess:
            df = df[df.session.isin(keep_sess)].reset_index(drop=True)
            X = np.zeros((len(df), len(keep_sess) + 1))
            for i, s in enumerate(keep_sess):
                X[:, i] = (df.session == s).values
            X[:, -1] = df.days.values
            y = df.offset_s.values
            keep = np.ones(len(df), bool)
            for _ in range(6):
                beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
                r = y - X @ beta
                mad = np.median(np.abs(r[keep] - np.median(r[keep]))) * 1.4826
                keep = np.abs(r) < 3 * max(mad, 5)
            df["resid"], df["kept"] = r, keep
            a = {s: float(beta[i]) for i, s in enumerate(keep_sess)}
            med = float(np.median(list(a.values())))
            res["a_imputed"] = [s for s in sessions if s not in a]
            for s in res["a_imputed"]:
                a[s] = med
            res.update(a=a, b=float(beta[-1]), resid_sd=float(np.std(r[keep])), n_kept=int(keep.sum()),
                       fitted=True)
    res["effective_fps"] = FPS * (1 + res["b"] / 86400.0)
    C.write_json(os.path.join(out_dir, "clock", "%s.json" % alias), res)
    if len(df):
        df.to_csv(os.path.join(out_dir, "clock", "%s_bursts.csv" % alias), index=False)
    log("  clock %-48s %s  b=%+.1f s/day (%.4f fps)  bursts=%d kept=%d resid_sd=%.1f s"
        % (alias, "FITTED " if res["fitted"] else "fallback", res["b"], res["effective_fps"],
           res["n_bursts"], res["n_kept"], res["resid_sd"]))
    return res


def load_clock(out_dir, alias):
    return C.read_json(os.path.join(out_dir, "clock", "%s.json" % alias))


def frame_times(clock, sess, k, frames):
    """Corrected wall-clock times (numpy datetime64[ms]) of frames of chunk k of a session."""
    s0 = np.datetime64(_s0(sess).to_pydatetime(), "ms")
    a = float(clock["a"].get(sess, 0.0))
    secs = (int(k) * int(clock["nf"]) + np.asarray(frames, dtype=float)) / FPS
    real = secs - a - float(clock["b"]) * secs / 86400.0
    return s0 + np.round(real * 1000.0).astype("int64").astype("timedelta64[ms]")


def schedule_of(row):
    """(night_start_hour, night_end_hour, {date: 'LD'|'DD'}) from ld_dd_schedule.json or defaults."""
    import ld_dd
    try:
        sch = ld_dd.load_schedule(row["exp_dir"])
    except Exception:  # noqa: BLE001
        sch = {}
    ns = float(sch.get("night_start_hour", ld_dd.DEFAULT_NIGHT[0]))
    ne = float(sch.get("night_end_hour", ld_dd.DEFAULT_NIGHT[1]))
    return ns, ne, dict(sch.get("dates") or {})


def flags(times, windows, schedule, defs):
    """stim_on, stim_edge (bool), light_phase (str), day_tag (str) for datetime64 times."""
    t = np.asarray(times, dtype="datetime64[ms]")
    day = t.astype("datetime64[D]")
    sec_of_day = (t - day).astype("timedelta64[ms]").astype(np.int64) / 1000.0
    hour = sec_of_day / 3600.0
    moh = (sec_of_day % 3600.0) / 60.0
    edge_min = defs["clock"]["edge_s"] / 60.0
    on = np.zeros(len(t), bool)
    edge = np.zeros(len(t), bool)
    for a, b in windows:
        on |= (moh >= a) & (moh < b)
        edge |= (np.abs(moh - a) <= edge_min) | (np.abs(moh - b) <= edge_min)
    ns, ne, tags = schedule
    ramp_up = defs["light"]["dawn_ramp_minutes"] / 60.0
    ramp_dn = defs["light"]["dusk_ramp_minutes"] / 60.0
    if ns > ne:
        night = (hour >= ns) | (hour < ne)
    else:
        night = (hour >= ns) & (hour < ne)
    phase = np.full(len(t), "day", object)
    phase[night] = "night"
    phase[(hour >= ne) & (hour < ne + ramp_up)] = "dawn"
    phase[(hour >= ns - ramp_dn) & (hour < ns)] = "dusk"
    days = day.astype(str)
    day_tag = np.array([tags.get(d, "") for d in days], object) if tags else np.full(len(t), "", object)
    return on, edge, phase, day_tag


def build_tables(row, out_dir, clock, log=C.log_default):
    """Per-experiment timeseries (per second) and events tables with corrected time + flags."""
    import pandas as pd
    defs = C.definitions()
    rig, alias = row["rig_id"], row["alias"]
    seg_dir = AN.seg_dir(out_dir, rig, alias)
    windows = C.parse_windows(row["stim_windows"])
    sched = schedule_of(row)
    ts_parts, ev_parts = [], []
    for stem, info in AN.segment_infos(out_dir, rig, alias, I.segment_set(row)):
        sess, k, T = info["session"], int(info["k"]), int(info["n_frames"])
        times = frame_times(clock, sess, k, np.arange(T))
        z = np.load(os.path.join(seg_dir, stem + ".activity.npz"))
        tsec = times.astype("datetime64[s]").astype(np.int64)
        u, inv = np.unique(tsec, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        cols = {"time": u.astype("datetime64[s]"), "stem": stem, "n_frames": cnt.astype(np.int16)}
        for key in ("n_rest", "n_walk", "n_fly", "n_hold", "n_state1", "n_real", "n_swarm"):
            cols[key] = (np.bincount(inv, weights=z[key][:T].astype(float)) / cnt).astype(np.float32)
        cols["fly_px"] = np.bincount(inv, weights=z["fly_px"][:T].astype(float)).astype(np.float32)
        for zi, zn in enumerate(info.get("zone_names") or []):
            cols["rest_%s" % zn] = (np.bincount(inv, weights=z["rest_zone"][zi][:T].astype(float)) / cnt).astype(np.float32)
        on, edge, phase, tag = flags(u.astype("datetime64[ms]"), windows, sched, defs)
        cols.update(stim_on=on, stim_edge=edge, light_phase=phase, day_tag=tag)
        ts_parts.append(pd.DataFrame(cols))
        ev = C.read_table(os.path.join(seg_dir, stem + ".events.parquet"))
        if ev is not None and len(ev):
            vf = np.clip(ev["video_frame"].values.astype(np.int64), 0, T - 1)
            et = frame_times(clock, sess, k, vf)
            on, edge, phase, tag = flags(et, windows, sched, defs)
            ev.insert(0, "stem", stem)
            ev.insert(1, "time", et)
            ev["stim_on"], ev["stim_edge"], ev["light_phase"], ev["day_tag"] = on, edge, phase, tag
            ev_parts.append(ev)
    if ts_parts:
        ts = pd.concat(ts_parts, ignore_index=True)
        C.write_table(ts, os.path.join(out_dir, "timeseries", rig, alias + ".parquet"))
    if ev_parts:
        ev = pd.concat(ev_parts, ignore_index=True)
        C.write_table(ev, os.path.join(out_dir, "events", rig, alias + ".parquet"))
        log("  events  %-48s %s" % (alias, ev.event.value_counts().to_dict()))
    return len(ts_parts)
