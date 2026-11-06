"""When the tone was really on, in video frames.

Three sources of truth, best first:

1. the Pi's playback log (``sound.log`` / journald export of ``buzzwatch-sound``), which says when
   each window opened and closed and which ones failed (:mod:`buzzphono.io.playback_log`);
2. the Pi's ``config.ini`` ``[sound]`` schedule (:mod:`buzzphono.io.pi_config`);
3. the BuzzPhono configuration (minutes 40-50 of every hour by default).

All three are in Pi wall-clock time. Turning them into frames needs the recording clock, which
this module calibrates on the flight burst every tone evokes: the onset of that burst, found in
each hour, is regressed on time to give one clock offset per session and one drift per experiment
(the camera's frame rate is not exactly 25 fps). This is the method of the 2026 flight-phonotaxis
analysis, with the same guards: a fit must keep at least 8 bursts and half of those found, and a
drift above 300 s/day is rejected as implausible. BuzzPhono adds one guard: the kept bursts must
agree within 30 s (residual SD); half-noise burst sets otherwise pass, because the noise inflates the
robust threshold itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from buzzphono.timebase import NS_PER_HOUR, NS_PER_S, ClockModel, SessionFrames, from_ns

BURST_BIN_S = 5.0
MAX_DRIFT_S_PER_DAY = 300.0
MIN_KEPT_BURSTS = 8
MIN_KEPT_FRACTION = 0.5
MIN_SESSION_BURSTS = 3
MAX_RESIDUAL_SD_S = 30.0  # real fits: 1.7-3.3 s; a larger scatter means the bursts do not agree


def burst_series(session: SessionFrames, chunk_activity: Mapping[int, np.ndarray]) -> pd.Series:
    """Flight activity in 5-s bins along the session's recording clock (NaN where nothing was recorded)."""
    if not chunk_activity:
        return pd.Series(dtype=np.float64)
    end = max(session.first_frame(k) + v.size for k, v in chunk_activity.items()) / session.fps
    nb = int(np.ceil(end / BURST_BIN_S)) + 1
    px = np.zeros(nb)
    cnt = np.zeros(nb)
    for k, activity in chunk_activity.items():
        secs = session.recording_seconds(k, np.arange(activity.size))
        b = np.floor(secs / BURST_BIN_S).astype(np.int64)
        px += np.bincount(b, weights=activity, minlength=nb)[:nb]
        cnt += np.bincount(b, minlength=nb)[:nb]
    ser = pd.Series(px, index=np.arange(nb) * BURST_BIN_S)
    ser[cnt == 0] = np.nan
    return ser


def burst_onsets(
    ser: pd.Series, session: SessionFrames, first_hour_ns: int, end_ns: int, on_start_min: int = 40
) -> list[dict[str, float | int | str]]:
    """The tone-evoked flight burst's onset in each hour, on the recording clock.

    For each clock hour, a baseline is taken from ``on_start − 10`` to ``on_start − 2`` minutes;
    the burst is the first 5-s bin (searched up to ``on_start + 12``) that is above
    ``median + 6 × (IQR + 1)`` and three times the baseline median, with more than 80% of the next
    30 s also above the threshold. Hours with less than 90% of the search window recorded are
    skipped.
    """
    rows: list[dict[str, float | int | str]] = []
    if ser.empty:
        return rows
    last_label = float(ser.index[-1])
    h = first_hour_ns - (first_hour_ns % NS_PER_HOUR)
    while h < end_ns:
        base_s = (h + (on_start_min - 10) * 60 * NS_PER_S - session.start_ns) / NS_PER_S
        end_s = (h + (on_start_min + 12) * 60 * NS_PER_S - session.start_ns) / NS_PER_S
        if base_s >= 0 and end_s <= last_label:
            w = ser.loc[base_s:end_s]
            if len(w) and w.notna().mean() > 0.9:
                sm = w.rolling(3, center=True, min_periods=1).mean()
                base = sm.loc[: base_s + 8 * 60]
                thr = base.median() + 6 * (base.quantile(0.75) - base.quantile(0.25) + 1)
                cand = sm.loc[base_s + 8 * 60 :]
                above = cand[cand > max(thr, 3 * base.median())]
                onset = None
                for t in above.index.to_numpy():
                    if (cand.loc[t : t + 30] > thr).mean() > 0.8:
                        onset = float(t)
                        break
                if onset is not None:
                    expected_s = (h + on_start_min * 60 * NS_PER_S - session.start_ns) / NS_PER_S
                    rows.append(
                        {
                            "session": session.session,
                            "hour_ns": int(h),
                            "onset_recording_s": onset,
                            "offset_s": onset - expected_s,
                            "days": (h - session.start_ns) / (86400 * NS_PER_S),
                            "peak_ratio": float(cand.max() / max(base.median(), 1)),
                        }
                    )
        h += NS_PER_HOUR
    return rows


@dataclass(frozen=True)
class ClockFit:
    model: ClockModel
    bursts: pd.DataFrame  # one row per burst, with residual and kept flag when fitted


def fit_clock(
    bursts: pd.DataFrame,
    sessions: Iterable[str],
    *,
    min_kept: int = MIN_KEPT_BURSTS,
    min_kept_fraction: float = MIN_KEPT_FRACTION,
    max_drift: float = MAX_DRIFT_S_PER_DAY,
    rig_drift_s_per_day: float | None = None,
) -> ClockFit:
    """Fit offset per session + one drift to the burst offsets (robust least squares, 6 passes).

    Sessions with fewer than 3 bursts of their own get the median offset of the fitted ones. If
    the fit fails, every session uses the nominal clock, or the rig's known drift when given.
    """
    sessions = sorted(sessions)
    df = bursts.copy()
    n_bursts = len(df)
    keep_sess = (
        sorted(s for s in set(df["session"]) if (df["session"] == s).sum() >= MIN_SESSION_BURSTS)
        if n_bursts
        else []
    )
    reject = ""
    if n_bursts >= min_kept and keep_sess:
        df = df[df["session"].isin(keep_sess)].reset_index(drop=True)
        x = np.zeros((len(df), len(keep_sess) + 1))
        for i, s in enumerate(keep_sess):
            x[:, i] = (df["session"] == s).to_numpy()
        x[:, -1] = df["days"].to_numpy()
        y = df["offset_s"].to_numpy(dtype=np.float64)
        keep = np.ones(len(df), dtype=bool)
        beta = np.zeros(x.shape[1])
        resid = y.copy()
        for _ in range(6):
            beta = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
            resid = y - x @ beta
            mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) * 1.4826
            keep = np.abs(resid) < 3 * max(mad, 5)
        df["residual_s"], df["kept"] = resid, keep
        n_kept = int(keep.sum())
        if n_kept < min_kept or n_kept < min_kept_fraction * len(df):
            reject = f"the fit kept only {n_kept} of {len(df)} bursts"
        elif float(np.std(resid[keep])) > MAX_RESIDUAL_SD_S:
            reject = f"the kept bursts scatter by {float(np.std(resid[keep])):.0f} s (residual SD)"
        elif abs(float(beta[-1])) > max_drift:
            reject = f"the fitted drift ({float(beta[-1]):.0f} s/day) is implausible"
        else:
            offsets = {s: float(beta[i]) for i, s in enumerate(keep_sess)}
            source = dict.fromkeys(keep_sess, "fitted")
            if offsets:
                med = float(np.median(list(offsets.values())))
                for s in sessions:
                    if s not in offsets:
                        offsets[s] = med
                        source[s] = "median offset of the fitted sessions (too few bursts of its own)"
            sd = float(np.std(resid[keep]))
            msg = (
                f"Clock calibrated on {n_kept} tone-onset bursts (drift {float(beta[-1]):.1f} s/day, "
                f"residual SD {sd:.1f} s)."
            )
            model = ClockModel(offsets, float(beta[-1]), source, True, msg, n_bursts, n_kept, sd)
            return ClockFit(model, df)
    if rig_drift_s_per_day is not None:
        why = f"; {reject}" if reject else ""
        msg = (
            f"Not enough consistent tone-onset bursts ({n_bursts} found{why}): using the rig's known drift "
            f"({rig_drift_s_per_day:.1f} s/day) with no session offsets."
        )
        model = ClockModel(
            dict.fromkeys(sessions, 0.0),
            float(rig_drift_s_per_day),
            dict.fromkeys(sessions, "rig drift profile"),
            False,
            msg,
            n_bursts,
        )
        return ClockFit(model, df)
    if reject:
        msg = (
            f"Not enough consistent tone-onset bursts ({n_bursts} found; {reject}): using the nominal clock "
            "(the nominal frame rate from each session start)."
        )
    else:
        msg = (
            f"Not enough tone-onset bursts ({n_bursts} found; at least {min_kept}, with "
            f"{MIN_SESSION_BURSTS} in one session, are needed): using the nominal clock "
            "(the nominal frame rate from each session start)."
        )
    return ClockFit(ClockModel.nominal(sessions, msg), df)


def data_span_hours(sessions: Mapping[str, SessionFrames]) -> tuple[int, int]:
    """Whole hours covering every chunk on the nominal recording clock (first hour start, end)."""
    starts, ends = [], []
    for sf in sessions.values():
        for k, n in sf.nf.items():
            first = sf.start_ns + round(sf.first_frame(k) / sf.fps * NS_PER_S)
            starts.append(first)
            ends.append(first + round(n / sf.fps * NS_PER_S))
    if not starts:
        return 0, 0
    lo = min(starts)
    hi = max(ends)
    return lo - lo % NS_PER_HOUR, hi - hi % NS_PER_HOUR + (NS_PER_HOUR if hi % NS_PER_HOUR else 0)


def find_bursts(
    sessions: Mapping[str, SessionFrames],
    activity: Mapping[str, Mapping[int, np.ndarray]],
    on_start_min: int = 40,
) -> pd.DataFrame:
    """Burst onsets of every session over the whole recording (one row per detected burst)."""
    first, last = data_span_hours(sessions)
    rows: list[dict[str, float | int | str]] = []
    for name, sf in sorted(sessions.items()):
        rows += burst_onsets(burst_series(sf, activity.get(name, {})), sf, first, last, on_start_min)
    return pd.DataFrame(
        rows, columns=["session", "hour_ns", "onset_recording_s", "offset_s", "days", "peak_ratio"]
    )


def cross_fitted_models(
    bursts: pd.DataFrame, sessions: Iterable[str], rig_drift_s_per_day: float | None = None
) -> dict[int, ClockModel]:
    """Clock models for latency analyses, each fitted without the bursts it will be used on.

    Returns {0: model fitted on odd hours (use for even hours), 1: fitted on even hours (use for odd)},
    so a burst never helps place itself in time.
    """
    sessions = list(sessions)
    hours = (bursts["hour_ns"] // NS_PER_HOUR) % 2 if len(bursts) else pd.Series(dtype=int)
    out = {}
    for parity in (0, 1):
        other = bursts[hours != parity] if len(bursts) else bursts
        out[parity] = fit_clock(other, sessions, rig_drift_s_per_day=rig_drift_s_per_day).model
    return out


def describe_bursts(fit: ClockFit) -> pd.DataFrame:
    """Burst table for output: wall-clock hour, onset, offset, residual and whether it was kept."""
    df = fit.bursts.copy()
    if df.empty:
        return df
    df.insert(1, "hour", [from_ns(int(h)).strftime("%Y-%m-%d %H:%M") for h in df["hour_ns"]])
    return df.drop(columns=["hour_ns"])
