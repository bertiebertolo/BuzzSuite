"""Per-minute sums on the calibrated clock, per hour-cycle window metrics and paired ON-vs-OFF tests.

Ported from the original flight package's ``build.py``. Zones are re-ordered to roles: speaker,
right (the wall opposite the speaker, labelled "Opposite wall"), top, bottom. Statistics use the
Wilcoxon test exactly as SciPy 1.10.1 computed it for the original results.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.errors import Cancelled, ConfigError
from buzzphono.flight_phono.extract import BLOCK, FLYER_STEP, MIN_FLYER_PTS, R_MAIN
from buzzphono.flight_phono.geometry import ROLES, FlightGeometry, role_label
from buzzphono.stats import wilcoxon_scipy110

COLONY = 30.0
FPS_EFF = 25.015  # effective camera rate, for coverage thresholds and per-minute rates

Loader = Callable[[str, str], dict[str, np.ndarray] | None]  # (kind, chunk stem) -> arrays
FrameTimes = Callable[[str, int, np.ndarray], pd.DatetimeIndex | None]  # (session, k, frames) -> times
CancelFn = Callable[[], bool] | None


@dataclass
class FlightParams:
    """The analysis choices. The defaults reproduce the 2026 research run."""

    on: tuple[int, int] = (40, 50)
    off: tuple[int, int] = (28, 38)
    light_hours: tuple[int, int] = (5, 17)
    flyer_mult: float = 1.0
    radius: float = 0.1
    onset_minutes: int = 2
    sham: tuple[int, int] = (10, 20)
    early: tuple[int, int] = (10, 30)
    min_median: float = 0.0

    def __post_init__(self) -> None:
        for name, (lo, hi) in (
            ("ON", self.on),
            ("OFF", self.off),
            ("sham", self.sham),
            ("early", self.early),
        ):
            if not (0 <= lo < hi <= 60):
                raise ConfigError(
                    f"The flight {name} window must be minutes 0-60 with start < end; got {lo}-{hi}."
                )
        if self.on[1] - self.on[0] <= self.onset_minutes:
            raise ConfigError(f"The ON window must be longer than {self.onset_minutes} minutes.")
        ls, le = self.light_hours
        if not (0 <= ls < le <= 24) or le - ls < 3:
            raise ConfigError(
                f"Light hours must be start < end within 0-24 h and at least 3 h; got {ls}-{le}."
            )

    def windows(self) -> OrderedDict[str, tuple[int, int]]:
        on0, on1 = self.on
        k = on0 + self.onset_minutes
        return OrderedDict(
            [
                ("OFF", self.off),
                ("ON", self.on),
                ("ON_onset", (on0, k)),
                ("ON_sust", (k, on1)),
                ("SHAM", self.sham),
                ("EARLY", self.early),
            ]
        )

    def pairs(self) -> list[tuple[str, str, str]]:
        on0, on1 = self.on
        k = on0 + self.onset_minutes
        return [
            ("ON vs OFF", "ON", "OFF"),
            (f"ON-onset ({on0}-{k}) vs OFF", "ON_onset", "OFF"),
            (f"ON-sustained ({k}-{on1}) vs OFF", "ON_sust", "OFF"),
            (f"sham: OFF {self.off[0]}-{self.off[1]} vs {self.sham[0]}-{self.sham[1]}", "OFF", "SHAM"),
            (f"ON vs early baseline ({self.early[0]}-{self.early[1]})", "ON", "EARLY"),
        ]

    @property
    def light_label(self) -> str:
        return f"light ({self.light_hours[0]:02d}-{self.light_hours[1]:02d}h)"

    @property
    def dark_label(self) -> str:
        return f"dark ({self.light_hours[1]:02d}-{self.light_hours[0]:02d}h)"

    def light_of(self, h: int) -> str:
        ls, le = self.light_hours
        return self.light_label if ls <= h < le else self.dark_label

    def phases(self) -> list[tuple[str, set[int]]]:
        ls, le = self.light_hours
        night = set(range(24)) - set(range(ls, le))
        return [
            (f"dawn ({ls:02d}-{ls + 1:02d}h)", {ls}),
            (f"day ({ls + 1:02d}-{le - 1:02d}h)", set(range(ls + 1, le - 1))),
            (f"dusk ({le - 1:02d}-{le:02d}h)", {le - 1}),
            (f"night ({le:02d}-{ls:02d}h)", night),
        ]

    def phase_of(self, h: int) -> str | None:
        for name, hrs in self.phases():
            if h in hrs:
                return name
        return None

    def r_fracs(self) -> tuple[float, float, float]:
        return (self.radius / 2.0, self.radius, self.radius * 2.0)


def _check(cancel: CancelFn) -> None:
    if cancel is not None and cancel():
        raise Cancelled("Flight analysis cancelled.")


def _grid(window: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[pd.Timestamp, int]:
    w0, w1 = window
    t0 = w0.floor("h")
    return t0, int(np.ceil((w1 - t0) / pd.Timedelta(minutes=1)))


def _minutes_frame(a: dict[str, np.ndarray], t0: pd.Timestamp, n_min: int, times: Any = None) -> pd.DataFrame:
    centers = times if times is not None else t0 + pd.to_timedelta(np.arange(n_min) + 0.5, unit="m")
    cols: dict[str, Any] = {"time": centers}
    for nm, v in a.items():
        if v.ndim == 1:
            cols[nm] = v
        else:
            for i, r in enumerate(ROLES):
                cols[f"{nm}|{r}"] = v[i]
    m = pd.DataFrame(cols)
    m["hour"] = m.time.dt.floor("h")
    m["moh"] = m.time.dt.minute
    return m


Chunks = list[tuple[str, str, int]]  # (stem, session, chunk index)
FramesPerChunk = int | Mapping[str, int]  # one count for all chunks, or per chunk stem


def _nf(nf: FramesPerChunk, stem: str) -> int:
    return nf if isinstance(nf, int) else int(nf[stem])


def accumulate(
    load: Loader,
    chunks: Chunks,
    frame_times: FrameTimes,
    order: list[int],
    params: FlightParams,
    window: tuple[pd.Timestamp, pd.Timestamp],
    nf: FramesPerChunk,
    nmax: int,
    s: float,
    cancel: CancelFn = None,
) -> dict[str, Any]:
    w0, w1 = window
    t0, n_min = _grid(window)
    nr_ = len(params.r_fracs())
    nz = 4
    on, off = params.on, params.off
    acc: dict[str, np.ndarray] = {
        k: np.zeros(n_min) for k in ("frames", "fly_n", "all_n", "fly_px", "takeoff", "rest_tot")
    }
    for nm in ("entries", "path_near", "sector", "landing", "rest_in"):
        acc[nm] = np.zeros((nz, n_min))
    for r in range(nr_):
        acc[f"near_r{r}"] = np.zeros((nz, n_min))
    heat: dict[tuple[str, str], np.ndarray] = {}
    nblk: dict[tuple[str, str], int] = {}
    used = 0
    for stem, sess, k in chunks:
        n_f = _nf(nf, stem)
        _check(cancel)
        times = frame_times(sess, k, np.arange(n_f))
        if times is None or times[-1] < w0 or times[0] >= w1:
            continue
        z = load("main", stem)
        if z is None:
            continue
        amin = np.floor((times - t0) / pd.Timedelta(minutes=1)).astype(int)
        inw = np.asarray((times >= w0) & (times < w1) & (amin >= 0) & (amin < n_min))
        a = np.asarray(amin)[inw]

        def bc(src: np.ndarray, inw: np.ndarray = inw, a: np.ndarray = a, n_f: int = n_f) -> np.ndarray:
            return np.bincount(a, weights=src[:n_f][inw].astype(float), minlength=n_min)[:n_min]

        acc["frames"] += np.bincount(a, minlength=n_min)[:n_min]
        for nm in ("fly_n", "all_n", "fly_px", "takeoff"):
            acc[nm] += bc(z[nm])
        rt = np.minimum(z["rest_tot"][:n_f].astype(float), COLONY)
        acc["rest_tot"] += bc(np.concatenate([rt, np.zeros(nmax - n_f)]))
        for i, zi in enumerate(order):
            ri = np.minimum(z["rest_in"][zi][:n_f].astype(float), rt)
            acc["rest_in"][i] += bc(np.concatenate([ri, np.zeros(nmax - n_f)]))
            for nm, src in (
                ("entries", "entries"),
                ("path_near", "path_near"),
                ("sector", "sector_n"),
                ("landing", "landing_in"),
            ):
                acc[nm][i] += bc(z[src][zi])
            for r in range(nr_):
                acc[f"near_r{r}"][i] += bc(z["near_n"][r, zi])
        hm = z["heat"]
        for b in range(n_f // BLOCK):
            tm = times[b * BLOCK + BLOCK // 2]
            if not (w0 <= tm < w1):
                continue
            moh = tm.minute
            win = "ON" if on[0] <= moh < on[1] else ("OFF" if off[0] <= moh < off[1] else None)
            if win is None:
                continue
            kk = (params.light_of(tm.hour), win)
            heat[kk] = heat.get(kk, 0) + hm[b]
            nblk[kk] = nblk.get(kk, 0) + 1
        used += 1
    m = _minutes_frame(acc, t0, n_min)
    return {
        "minutes": m,
        "heat": heat,
        "nblk": nblk,
        "s": s,
        "chunks_used": used,
        "minutes_with_data": int((m.frames > 0).sum()),
    }


def window_sums(m: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    sel = m[(m.moh >= lo) & (m.moh < hi)]
    cols = [c for c in m.columns if c not in ("time", "hour", "moh")]
    return sel.groupby("hour")[cols].sum()


def window_metrics(sums: pd.DataFrame, s: float, n_radii: int = 3) -> pd.DataFrame:
    mins = sums.frames / (60.0 * FPS_EFF)
    x = pd.DataFrame(index=sums.index)
    nan = np.nan
    x["flyers_per_frame"] = sums.fly_n / sums.frames
    x["alldet_per_frame"] = sums.all_n / sums.frames
    x["flight_px_per_frame"] = sums.fly_px / sums.frames
    x["takeoffs_per_min"] = sums.takeoff / mins
    sec_tot = sums[[f"sector|{r}" for r in ROLES]].sum(axis=1)
    for r in ROLES:
        for i in range(n_radii):
            x[f"share_near_r{i}|{r}"] = sums[f"near_r{i}|{r}"] / sums.fly_n.replace(0, nan)
        x[f"share_sector|{r}"] = sums[f"sector|{r}"] / sec_tot.replace(0, nan)
        x[f"entries_per_min|{r}"] = sums[f"entries|{r}"] / mins
        x[f"dwell_frames_per_visit|{r}"] = sums[f"near_r1|{r}"] / sums[f"entries|{r}"].replace(0, nan)
        x[f"speed_in_nbhd|{r}"] = sums[f"path_near|{r}"] / sums[f"near_r1|{r}"].replace(0, nan) / s
        x[f"landings_per_min|{r}"] = sums[f"landing|{r}"] / mins
        x[f"rest_prop|{r}"] = sums[f"rest_in|{r}"] / sums.rest_tot.replace(0, nan)
    return x


def _cycles(
    m: pd.DataFrame,
    params: FlightParams,
    metrics_fn: Callable[[pd.DataFrame], pd.DataFrame],
    extra_mask: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """Long table hour × window -> metrics; only windows with at least 90% of their frames."""
    out = []
    for w, (lo, hi) in params.windows().items():
        sums = window_sums(m, lo, hi)
        keep = sums.frames >= 0.9 * (hi - lo) * 60 * FPS_EFF
        if extra_mask is not None:
            keep = keep & extra_mask(sums)
        sums = sums[keep]
        x = metrics_fn(sums)
        x.insert(0, "window", w)
        out.append(x.reset_index())
    return pd.concat(out, ignore_index=True)


def cycle_metrics(acc: dict[str, Any], params: FlightParams) -> pd.DataFrame:
    return _cycles(acc["minutes"], params, lambda sums: window_metrics(sums, acc["s"], len(params.r_fracs())))


def _wilcoxon(d: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(d) < 4 or not (d != 0).any():
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return (
                wilcoxon_scipy110(a, b, alternative="greater").p_value,
                wilcoxon_scipy110(a, b).p_value,
            )
        except ValueError:
            return np.nan, np.nan


def stat_table(cm: pd.DataFrame, label: str, params: FlightParams) -> pd.DataFrame:
    """Paired tests per (pair, metric, zone, grouping): one-sided Wilcoxon ON > OFF (and two-sided), dz,
    and the speaker's change minus each control's (difference in differences)."""
    if "cyc" not in cm.columns:
        cm = cm.assign(cyc=cm.hour.astype(str))
    metric_cols = [c for c in cm.columns if c not in ("hour", "window", "cyc", "exp")]
    base_metrics = sorted({c.split("|")[0] for c in metric_cols})
    rows = []
    wide = {w: cm[cm.window == w].set_index("cyc") for w in params.windows()}
    light, dark = params.light_label, params.dark_label
    for pair, won, woff in params.pairs():
        on_t, off_t = wide[won], wide[woff]
        idx = on_t.index.intersection(off_t.index)
        on_t, off_t = on_t.loc[idx], off_t.loc[idx]
        hrs = pd.to_datetime(on_t.hour).dt.hour
        groups: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(idx), bool))]
        for lab in (light, dark):
            groups.append(("light", lab, hrs.map(params.light_of).to_numpy() == lab))
        for pn, _ in params.phases():
            groups.append(("phase", pn, hrs.map(params.phase_of).to_numpy() == pn))
        for bm in base_metrics:
            zoned = f"{bm}|speaker" in on_t.columns
            series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if zoned:
                for r in ROLES:
                    series[r] = (on_t[f"{bm}|{r}"].to_numpy(), off_t[f"{bm}|{r}"].to_numpy())
                d = {r: series[r][0] - series[r][1] for r in ROLES}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ctrl = np.nanmean(np.vstack([d[r] for r in ROLES[1:]]), axis=0)
                series["speaker - mean(controls)"] = (d["speaker"] - ctrl, np.zeros(len(idx)))
                for r in ROLES[1:]:
                    series[f"speaker - {r}"] = (d["speaker"] - d[r], np.zeros(len(idx)))
            else:
                series["all"] = (on_t[bm].to_numpy(), off_t[bm].to_numpy())
            for zname, (von, voff) in series.items():
                for gname, gval, gmask in groups:
                    msk = gmask & np.isfinite(von) & np.isfinite(voff)
                    n = int(msk.sum())
                    if n < 3:
                        continue
                    a, b = von[msk], voff[msk]
                    dd = a - b
                    pg, p2 = _wilcoxon(dd, a, b)
                    sd = dd.std(ddof=1) if n > 1 else np.nan
                    rows.append(
                        {
                            "exp": label,
                            "pair": pair,
                            "metric": bm,
                            "zone": zname,
                            "zone_label": role_label(zname),
                            "grouping": gname,
                            "group": gval,
                            "n_cycles": n,
                            "mean_off": b.mean(),
                            "mean_on": a.mean(),
                            "delta": dd.mean(),
                            "fold": (a.mean() / b.mean())
                            if b.mean() not in (0, np.nan) and zname in (*ROLES, "all")
                            else np.nan,
                            "dz": dd.mean() / sd if sd and sd > 0 else np.nan,
                            "n_on_gt_off": int((dd > 0).sum()),
                            "p_on_gt_off": pg,
                            "p_two_sided": p2,
                        }
                    )
    return pd.DataFrame(rows)


def pool_cycle_metrics(
    tables: list[tuple[str, pd.DataFrame]], params: FlightParams, label: str = "pooled"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool several experiments' cycle metrics, each (experiment, hour) a replicate."""
    frames = []
    for exp_label, cm in tables:
        cm = cm.copy()
        cm["exp"] = exp_label
        cm["cyc"] = exp_label + "_" + cm.hour.astype(str)
        frames.append(cm)
    pool = pd.concat(frames, ignore_index=True)
    return pool, stat_table(pool, label, params)


def accumulate_links(
    load: Loader,
    chunks: Chunks,
    frame_times: FrameTimes,
    order: list[int],
    geom: FlightGeometry,
    params: FlightParams,
    window: tuple[pd.Timestamp, pd.Timestamp],
    nf: FramesPerChunk,
    base_times: pd.Series,
    cancel: CancelFn = None,
) -> dict[str, Any]:
    w0, w1 = window
    n_min = len(base_times)
    t0 = base_times.iloc[0].floor("h")
    nz = 4
    acc: dict[str, np.ndarray] = {
        k: np.zeros(n_min) for k in ("frames", "takeoffs", "landings", "bouts_both", "rest_tot_frames")
    }
    for nm in (
        "takeoff_poly",
        "landing_poly",
        "takeoff_nb",
        "landing_nb",
        "bout_start_out",
        "bout_arrive",
        "rest_frames_poly",
    ):
        acc[nm] = np.zeros((nz, n_min))
    paths = [geom.paths[i] for i in order]
    rnb = geom.rpx[R_MAIN]
    thr = FLYER_STEP * geom.s * params.flyer_mult
    used = 0
    for stem, sess, k in chunks:
        n_f = _nf(nf, stem)
        _check(cancel)
        times = frame_times(sess, k, np.arange(n_f))
        z = load("links", stem)
        if times is None or times[-1] < w0 or times[0] >= w1 or z is None:
            continue
        amin_f = np.asarray(np.floor((times - t0) / pd.Timedelta(minutes=1)).astype(int))
        inw_f = np.asarray((times >= w0) & (times < w1)) & (amin_f >= 0) & (amin_f < n_min)

        def minute_of(
            frames: np.ndarray, amin_f: np.ndarray = amin_f, inw_f: np.ndarray = inw_f, n_f: int = n_f
        ) -> tuple[np.ndarray, np.ndarray]:
            f = np.clip(frames.astype(int), 0, n_f - 1)
            return amin_f[f], inw_f[f]

        a = amin_f[inw_f]
        acc["frames"] += np.bincount(a, minlength=n_min)[:n_min]
        acc["rest_tot_frames"] += np.bincount(
            a, weights=z["rest_tot"][:n_f][inw_f].astype(float), minlength=n_min
        )[:n_min]
        for i, zi in enumerate(order):
            acc["rest_frames_poly"][i] += np.bincount(
                a, weights=z["rest_in"][zi][:n_f][inw_f].astype(float), minlength=n_min
            )[:n_min]
        b = z["bouts"]
        if len(b):
            fly = (b[:, 10] >= MIN_FLYER_PTS) & (b[:, 11] >= thr)
            b = b[fly]
            ls, le = b[:, 7] == 1, b[:, 8] == 1
            m_to, ok_to = minute_of(b[:, 1])
            m_ld, ok_ld = minute_of(b[:, 2])
            p0, p1 = b[:, 3:5], b[:, 5:7]
            tk = ls & ok_to
            ld = le & ok_ld
            both = ls & le & ok_to
            acc["takeoffs"] += np.bincount(m_to[tk], minlength=n_min)[:n_min]
            acc["landings"] += np.bincount(m_ld[ld], minlength=n_min)[:n_min]
            acc["bouts_both"] += np.bincount(m_to[both], minlength=n_min)[:n_min]
            for i in range(nz):
                d0, d1 = geom.distance(order[i], p0), geom.distance(order[i], p1)
                in0 = paths[i].contains_points(p0) if tk.any() else np.zeros(len(b), bool)
                in1 = paths[i].contains_points(p1) if ld.any() else np.zeros(len(b), bool)
                acc["takeoff_poly"][i] += np.bincount(m_to[tk & in0], minlength=n_min)[:n_min]
                acc["landing_poly"][i] += np.bincount(m_ld[ld & in1], minlength=n_min)[:n_min]
                acc["takeoff_nb"][i] += np.bincount(m_to[tk & (d0 <= rnb)], minlength=n_min)[:n_min]
                acc["landing_nb"][i] += np.bincount(m_ld[ld & (d1 <= rnb)], minlength=n_min)[:n_min]
                st_out = both & (d0 > rnb)
                acc["bout_start_out"][i] += np.bincount(m_to[st_out], minlength=n_min)[:n_min]
                acc["bout_arrive"][i] += np.bincount(m_to[st_out & (d1 <= rnb)], minlength=n_min)[:n_min]
        used += 1
    m = _minutes_frame(acc, t0, n_min, times=base_times.to_numpy())
    return {
        "minutes": m,
        "chunks_used": used,
        "takeoffs": float(acc["takeoffs"].sum()),
        "landings": float(acc["landings"].sum()),
        "bouts_both": float(acc["bouts_both"].sum()),
    }


def links_window_metrics(sums: pd.DataFrame) -> pd.DataFrame:
    mins = sums.frames / (60.0 * FPS_EFF)
    x = pd.DataFrame(index=sums.index)
    x["takeoffs_per_min"] = sums.takeoffs / mins
    x["landings_per_min"] = sums.landings / mins
    x["linked_bouts_per_min"] = sums.bouts_both / mins
    for r in ROLES:
        x[f"takeoff_poly_per_min|{r}"] = sums[f"takeoff_poly|{r}"] / mins
        x[f"landing_poly_per_min|{r}"] = sums[f"landing_poly|{r}"] / mins
        rest_min = sums[f"rest_frames_poly|{r}"] / (60.0 * FPS_EFF)
        x[f"takeoff_hazard|{r}"] = sums[f"takeoff_poly|{r}"] / rest_min.where(rest_min >= 5)
        x[f"takeoff_share_nb|{r}"] = sums[f"takeoff_nb|{r}"] / sums.takeoffs.replace(0, np.nan)
        x[f"landing_share_nb|{r}"] = sums[f"landing_nb|{r}"] / sums.landings.replace(0, np.nan)
        so = sums[f"bout_start_out|{r}"].where(sums[f"bout_start_out|{r}"] >= 5)
        x[f"arrival_prob|{r}"] = sums[f"bout_arrive|{r}"] / so
    return x


def links_cycle_metrics(acc: dict[str, Any], params: FlightParams) -> pd.DataFrame:
    return _cycles(acc["minutes"], params, links_window_metrics)


def accumulate_dist(
    load: Loader,
    chunks: Chunks,
    frame_times: FrameTimes,
    order: list[int],
    window: tuple[pd.Timestamp, pd.Timestamp],
    nf: FramesPerChunk,
    cancel: CancelFn = None,
) -> dict[str, Any]:
    w0, w1 = window
    t0, n_min = _grid(window)
    frames, cnt = np.zeros(n_min), np.zeros(n_min)
    dsum = np.zeros((4, n_min))
    used = 0
    for stem, sess, k in chunks:
        n_f = _nf(nf, stem)
        _check(cancel)
        times = frame_times(sess, k, np.arange(n_f))
        z = load("traj", stem)
        if times is None or times[-1] < w0 or times[0] >= w1 or z is None:
            continue
        amin = np.asarray(np.floor((times - t0) / pd.Timedelta(minutes=1)).astype(int))
        inw = np.asarray((times >= w0) & (times < w1)) & (amin >= 0) & (amin < n_min)
        a = amin[inw]
        frames += np.bincount(a, minlength=n_min)[:n_min]
        cnt += np.bincount(a, weights=z["cnt"][:n_f][inw].astype(float), minlength=n_min)[:n_min]
        for i, zi in enumerate(order):
            dsum[i] += np.bincount(a, weights=z["dist_sum"][zi][:n_f][inw], minlength=n_min)[:n_min]
        used += 1
    return {
        "minutes": _minutes_frame({"frames": frames, "cnt": cnt, "dist_sum": dsum}, t0, n_min),
        "chunks_used": used,
    }


def dist_cycle_metrics(acc: dict[str, Any], params: FlightParams) -> pd.DataFrame:
    def fn(sums: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=sums.index)
        for r in ROLES:
            x[f"mean_dist|{r}"] = sums[f"dist_sum|{r}"] / sums.cnt
        return x

    return _cycles(acc["minutes"], params, fn, extra_mask=lambda sums: sums.cnt > 0)


def trajectory_pass_summary(
    load: Loader,
    chunks: Chunks,
    frame_times: FrameTimes,
    order: list[int],
    geom: FlightGeometry,
    params: FlightParams,
    window: tuple[pd.Timestamp, pd.Timestamp],
    cancel: CancelFn = None,
) -> pd.DataFrame:
    """Share of flight tracks passing within the neighbourhood radius of each wall's zone, per light phase
    and tone window (a track's window is that of its first point)."""
    w0, w1 = window
    rnb = geom.rpx[R_MAIN]
    ls_, le_ = params.light_hours
    conds = [(lab, w) for lab in (params.light_label, params.dark_label) for w in ("OFF", "ON")]
    wins = {"OFF": params.off, "ON": params.on}
    res: dict[tuple[str, str], dict[str, int]] = {
        c: {"n": 0, **{f"vis_{r}": 0 for r in ROLES}} for c in conds
    }
    for stem, sess, k in chunks:
        _check(cancel)
        z = load("traj", stem)
        if z is None:
            continue
        off = z["off"]
        if len(off) < 2:
            continue
        xy, fr = z["xy"], z["fr"]
        times = frame_times(sess, k, fr[off[:-1]])
        if times is None:
            continue
        inw = np.asarray((times >= w0) & (times < w1))
        moh, hr = np.asarray(times.minute), np.asarray(times.hour)
        lightm = (hr >= ls_) & (hr < le_)
        tid = np.repeat(np.arange(len(off) - 1), np.diff(off))
        vis = {
            r: np.bincount(tid, weights=geom.distance(order[i], xy) <= rnb, minlength=len(off) - 1) > 0
            for i, r in enumerate(ROLES)
        }
        for lab, lm in ((params.light_label, lightm), (params.dark_label, ~lightm)):
            for win, (lo, hi) in wins.items():
                idx = np.nonzero(inw & lm & (moh >= lo) & (moh < hi))[0]
                if not len(idx):
                    continue
                r_ = res[(lab, win)]
                r_["n"] += len(idx)
                for r in ROLES:
                    r_[f"vis_{r}"] += int(vis[r][idx].sum())
    rows = []
    for (lab, win), r_ in res.items():
        n = max(r_["n"], 1)
        row: dict[str, Any] = {"phase": lab, "tone": win, "n_flights": r_["n"]}
        for r in ROLES:
            row[f"pct_pass_{r}"] = 100.0 * r_[f"vis_{r}"] / n
        rows.append(row)
    return pd.DataFrame(rows)
