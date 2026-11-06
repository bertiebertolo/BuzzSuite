"""Recording-clock calibration on the sound-evoked flight burst at tone onset (ported from fp_clock.py).

Chunks come from one continuous rpicam --segment stream per session, so a frame's recording-clock time is
secs = (k*NF + frame)/25 after the session start S0 (NF = frames per chunk, read from the data). The camera's
true rate is slightly off 25 fps, so the burst (sound ON at real :40:00) drifts:
offset(h) = rec_onset - real_onset = a_session + b * days_since_S0.
Fit: robust (MAD) least squares, one b per experiment, one intercept per session (sessions with >= 3 bursts).
Real time of a frame = S0 + secs - a_session - b * secs/86400. With fewer than 8 bursts the nominal clock is
used (a = b = 0) and the result says so.
"""
import os

import numpy as np

from . import config as fc

BIN = 5.0
MAX_DRIFT_S_PER_DAY = 300.0   # larger fitted drifts are treated as a failed fit
MIN_KEPT_BURSTS = 8           # a fit must keep at least this many bursts ...
MIN_KEPT_FRACTION = 0.5       # ... and at least this fraction of those it was given


def session_series(main_dir, stems_by_k, nf):
    """5-s binned flight px along the continuous recording clock of one session (NaN where no chunk cache)."""
    import pandas as pd
    kmax = max(stems_by_k)
    nb = int(np.ceil((kmax + 1) * nf / fc.FPS / BIN)) + 1
    px = np.zeros(nb)
    cnt = np.zeros(nb)
    for k, stem in stems_by_k.items():
        p = os.path.join(main_dir, stem + ".npz")
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        fp = z["fly_px"][:nf]
        b = np.floor((k * nf + np.arange(nf)) / fc.FPS / BIN).astype(np.int64)
        px += np.bincount(b, weights=fp, minlength=nb)[:nb]
        cnt += np.bincount(b, minlength=nb)[:nb]
    ser = pd.Series(px, index=np.arange(nb) * BIN)
    ser[cnt == 0] = np.nan
    return ser


def burst_onsets(ser, s0, t_first, t_last, on_start=40):
    """flight-burst onset per hour, searched from on_start-10 to on_start+12 min (30-52 at the default)."""
    import pandas as pd
    rows = []
    h = t_first.floor("h")
    while h < t_last:
        base_s = (h + pd.Timedelta(minutes=on_start - 10) - s0).total_seconds()
        end_s = (h + pd.Timedelta(minutes=on_start + 12) - s0).total_seconds()
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
                                 "offset_s": (rec - (h + pd.Timedelta(minutes=on_start))).total_seconds(),
                                 "days": (h - s0).total_seconds() / 86400,
                                 "peak_ratio": float(cand.max() / max(base.median(), 1))})
        h += pd.Timedelta(hours=1)
    return rows


def calibrate(main_dir, chunks, nf, t_first, t_last, on_start=40):
    """Fit the clock model. Returns (clock dict, bursts DataFrame)."""
    import pandas as pd
    sessions = {}
    for stem, sess, k in chunks:
        sessions.setdefault(sess, {})[k] = stem
    rows = []
    for sess, by_k in sorted(sessions.items()):
        s0 = fc.session_start(sess)
        # skip sessions that cannot overlap the window
        if s0 + pd.Timedelta(seconds=(max(by_k) + 1) * nf / fc.FPS) < t_first or s0 > t_last:
            continue
        ser = session_series(main_dir, by_k, nf)
        for r in burst_onsets(ser, s0, t_first, t_last, on_start):
            r["session"] = sess
            rows.append(r)
    df = pd.DataFrame(rows)
    res = dict(n_bursts=int(len(df)))
    sess_list = sorted(set(df.session)) if len(df) else []
    keep_sess = [s for s in sess_list if (df.session == s).sum() >= 3]
    ok = len(df) >= 8 and len(keep_sess) >= 1
    reject = ""
    if ok:
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
        n_kept = int(keep.sum())
        # plausibility guard: with few or noisy bursts (e.g. mostly noise in a dense cage) the robust loop can
        # collapse onto a couple of points and return a nonsense drift with ~0 residual. Real cameras drift by
        # tens of s/day (51 s/day Penzance, ~0 Bangkok); require most bursts to agree with the fit.
        if n_kept < MIN_KEPT_BURSTS or n_kept < MIN_KEPT_FRACTION * len(df):
            ok, reject = False, "the fit kept only %d of %d bursts" % (n_kept, len(df))
        elif abs(float(beta[-1])) > MAX_DRIFT_S_PER_DAY:
            ok, reject = False, "the fitted drift (%.0f s/day) is implausible" % float(beta[-1])
        else:
            res.update(a=dict((s, float(beta[i])) for i, s in enumerate(keep_sess)), b=float(beta[-1]),
                       resid_sd=float(np.std(r[keep])), n_kept=n_kept, fitted=True)
    if not ok:
        sess_all = sorted(sessions)
        res.update(a=dict((s, 0.0) for s in sess_all), b=0.0, resid_sd=float("nan"), n_kept=0, fitted=False)
    if res["fitted"]:
        res["message"] = ("Clock calibrated on %d tone-onset bursts (drift %.1f s/day, residual SD %.1f s)."
                          % (res["n_kept"], res["b"], res["resid_sd"]))
    elif reject:
        res["message"] = ("Not enough consistent tone-onset bursts (%d found; %s): using the nominal clock "
                          "(25 fps from each session start)." % (res["n_bursts"], reject))
    else:
        res["message"] = ("Not enough tone-onset bursts (%d found; at least 8, with 3 in one session, are needed): "
                          "using the nominal clock (25 fps from each session start)." % res["n_bursts"])
    res["sessions"] = dict((s, min(v)) for s, v in sessions.items())
    res["nf"] = int(nf)
    res["effective_fps"] = fc.FPS * (1 + res["b"] / 86400.0)
    res["dropped_sessions"] = sorted(s for s in sessions if s not in res["a"])
    return res, df


def frame_times(clock, sess, k, frames):
    """real wall-clock time (pandas DatetimeIndex) of frames-within-chunk k of a session, or None when the
    session has no clock intercept (fewer than 3 bursts while the rest of the experiment was fitted)."""
    import pandas as pd
    a = clock["a"].get(sess)
    if a is None:
        return None
    s0 = fc.session_start(sess)
    secs = (k * clock["nf"] + np.asarray(frames, float)) / fc.FPS
    return s0 + pd.to_timedelta(secs - a - clock["b"] * secs / 86400.0, unit="s")
