"""Figures for the flight-phonotaxis analysis (ported from fp_figs.py; thesis style: ON orange, OFF grey).

Every builder returns a matplotlib ``Figure`` made with the object-oriented API (no pyplot), so it is safe
to build on a worker thread and save with ``save()``, and the GUI can embed the same builder's output in a
``FigureCanvasTkAgg`` on the main thread. ``scale`` shrinks fonts for the small inline view.
"""
import os

import numpy as np

from . import config as fc

STIM_COLOR, OFF_COLOR = "#ff8c00", "#444444"
RC = {"speaker": "#b5179e", "right": "#4575b4", "top": "#1a9850", "bottom": "#7f7f7f"}
DPI = 160
SIG_KEY = "Wilcoxon signed-rank (ON > OFF per cycle):   ns p>=0.05    * p<0.05    ** p<0.01    *** p<0.001"


def _fs(scale):
    return dict(title=14 * scale, label=12 * scale, tick=10 * scale, small=8 * scale)


def _new_figure(figsize, dpi):
    from matplotlib.figure import Figure
    return Figure(figsize=figsize, dpi=dpi)


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def _clean(ax, fs):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.tick_params(labelsize=fs["tick"])


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def valid_minutes(M):
    """minute rows of hours with >= 90% frame coverage over the whole hour."""
    hf = M.groupby("hour").frames.transform("sum")
    return M[hf >= 0.9 * 60 * 60 * fc.FPS_EFF].copy()


def minute_course(M, series_fn, params):
    """mean/SEM over hours of series_fn(M) per (light, minute-of-hour)."""
    import pandas as pd
    V = valid_minutes(M)
    V["v"] = series_fn(V)
    V["light"] = V.hour.dt.hour.map(params.light_of)
    g = V.groupby(["light", "moh"]).v
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.count()}).reset_index()


def _band(ax, params):
    ax.axvspan(params.on[0], params.on[1], color=STIM_COLOR, alpha=0.18)
    ax.axvspan(params.off[0], params.off[1], color=OFF_COLOR, alpha=0.08)


def _minute_axes(axes_row, fs):
    for ax in axes_row:
        ax.set_xlabel("Minute of hour", fontsize=fs["label"])
        ax.set_xlim(0, 59)
        ax.set_xticks(range(0, 61, 5))


def _empty(fig, msg, fs):
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=fs["label"], wrap=True)
    return fig


def fig_arousal(M, label, params, scale=1.0, dpi=DPI, figsize=(13, 8)):
    """flyers airborne (x the hour's mean) and take-offs per min through the hour, light vs dark."""
    import pandas as pd
    fs = _fs(scale)
    fig = _new_figure(figsize, dpi)
    V = valid_minutes(M)
    if V.empty:
        return _empty(fig, "No hour with >= 90% frame coverage in the analysis window.", fs)
    axes = fig.subplots(2, 2, sharex=True)
    hm = V.groupby("hour").apply(lambda d: d.fly_n.sum() / d.frames.sum())
    if isinstance(hm, pd.DataFrame):
        hm = hm.iloc[:, 0]
    rel = lambda D: D.fly_n / D.frames / D.hour.map(hm).replace(0, np.nan)  # noqa: E731
    tk = lambda D: D.takeoff / (D.frames / (60 * fc.FPS_EFF)).replace(0, np.nan)  # noqa: E731
    for j, ln in enumerate((params.light_label, params.dark_label)):
        for i, (fn, yl, col) in enumerate(((rel, "Flyers airborne\n(x the hour's mean, +-SEM)", "#2ca02c"),
                                           (tk, "Flight-track starts per min\n(mean +-SEM over hours)", "#1f77b4"))):
            c = minute_course(M, fn, params)
            c = c[c.light == ln]
            ax = axes[i, j]
            _band(ax, params)
            ax.plot(c.moh, c["mean"], color=col, lw=1.8)
            ax.fill_between(c.moh, c["mean"] - c["sem"], c["mean"] + c["sem"], color=col, alpha=0.2)
            if i == 0:
                ax.axhline(1.0, color="gray", ls="--", lw=1)
                ax.set_title("%s  (n=%d h)" % (ln, int(c.n.max()) if len(c) else 0), fontsize=fs["label"])
            ax.set_ylabel(yl, fontsize=fs["label"])
            _clean(ax, fs)
    _minute_axes(axes[1], fs)
    fig.suptitle("%s: arousal (do they fly more during the tone? ON %d-%d orange, OFF %d-%d grey)"
                 % ((label,) + params.on + params.off), fontsize=fs["title"])
    fig.tight_layout()
    return fig


def fig_share_timecourse(M, label, params, scale=1.0, dpi=DPI, figsize=(13, 8)):
    """share of flyers near each wall's zone and in each wall sector, through the hour."""
    fs = _fs(scale)
    fig = _new_figure(figsize, dpi)
    if valid_minutes(M).empty:
        return _empty(fig, "No hour with >= 90% frame coverage in the analysis window.", fs)
    axes = fig.subplots(2, 2, sharex=True)
    rows = (("near_r1", "Share of flyers within %g cage widths\nof the zone (mean +-SEM)" % params.radius),
            ("sector", "Share of flyers in the zone's wall sector\n(mean +-SEM)"))
    for j, ln in enumerate((params.light_label, params.dark_label)):
        for i, (nm, yl) in enumerate(rows):
            ax = axes[i, j]
            _band(ax, params)
            for r in fc.ROLES:
                if nm == "near_r1":
                    fn = lambda D, r=r: D["near_r1|%s" % r] / D.fly_n.replace(0, np.nan)  # noqa: E731
                else:
                    tot = lambda D: sum(D["sector|%s" % q] for q in fc.ROLES).replace(0, np.nan)  # noqa: E731
                    fn = lambda D, r=r, tot=tot: D["sector|%s" % r] / tot(D)  # noqa: E731
                c = minute_course(M, fn, params)
                c = c[c.light == ln]
                ax.plot(c.moh, c["mean"], color=RC[r], lw=2.0 if r == "speaker" else 1.2, label=fc.ROLE_LABELS[r])
                ax.fill_between(c.moh, c["mean"] - c["sem"], c["mean"] + c["sem"], color=RC[r], alpha=0.14)
            if i == 0:
                ax.set_title(ln, fontsize=fs["label"])
            ax.set_ylabel(yl, fontsize=fs["label"] - 1)
            _clean(ax, fs)
    _minute_axes(axes[1], fs)
    axes[0, 1].legend(fontsize=fs["small"] + 1, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.suptitle("%s: where does the flight go? (flight shares, tone ON = orange)" % label, fontsize=fs["title"])
    fig.tight_layout()
    return fig


def fig_distance_timecourse(DM, label, params, scale=1.0, dpi=DPI, figsize=(13, 4.8)):
    """average distance of flyers from each wall's zone centre (cage widths) through the hour."""
    fs = _fs(scale)
    fig = _new_figure(figsize, dpi)
    if valid_minutes(DM).empty:
        return _empty(fig, "No hour with >= 90% frame coverage in the analysis window.", fs)
    axes = fig.subplots(1, 2, sharey=True)
    for j, ln in enumerate((params.light_label, params.dark_label)):
        ax = axes[j]
        _band(ax, params)
        for r in fc.ROLES:
            fn = lambda D, r=r: D["dist_sum|%s" % r] / D.cnt.replace(0, np.nan)  # noqa: E731
            c = minute_course(DM, fn, params)
            c = c[c.light == ln]
            ax.plot(c.moh, c["mean"], color=RC[r], lw=2.0 if r == "speaker" else 1.2, label=fc.ROLE_LABELS[r])
            ax.fill_between(c.moh, c["mean"] - c["sem"], c["mean"] + c["sem"], color=RC[r], alpha=0.14)
        ax.set_title(ln, fontsize=fs["label"])
        if j == 0:
            ax.set_ylabel("Average distance of flyers from\nthe wall's zone (cage widths)", fontsize=fs["label"] - 1)
        _clean(ax, fs)
    _minute_axes(axes, fs)
    axes[1].legend(fontsize=fs["small"] + 1, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.suptitle("%s: average distance of flying mosquitoes from each wall (lower = closer)" % label,
                 fontsize=fs["title"])
    fig.tight_layout()
    return fig


DEFAULT_BOX_METRICS = (("share_near_r1", "Flyers near zone\n(share of all flyers)"),
                       ("share_sector", "Flyers in zone's wall sector\n(share)"),
                       ("entries_per_min", "Entries into zone\nneighbourhood (per min)"),
                       ("landings_per_min", "Flight tracks ending in\nzone polygon (per min)"))


def fig_boxes(CM, ST, label, params, metrics=None, scale=1.0, dpi=DPI, figsize=None, title=None):
    """per-cycle ON vs OFF boxes for the speaker and each control wall (fp_figs.fig_boxes)."""
    import matplotlib.patches as mpatches
    import pandas as pd
    fs = _fs(scale)
    metrics = metrics or DEFAULT_BOX_METRICS
    fig = _new_figure(figsize or (13, 4.0 * len(metrics)), dpi)
    if "cyc" not in CM.columns:
        CM = CM.assign(cyc=CM.hour.astype(str))
    on = CM[CM.window == "ON"].set_index("cyc")
    off = CM[CM.window == "OFF"].set_index("cyc")
    idx = on.index.intersection(off.index)
    if len(idx) == 0:
        return _empty(fig, "No complete ON/OFF hour-cycles in the analysis window.", fs)
    on, off = on.loc[idx], off.loc[idx]
    hrs = pd.to_datetime(on.hour).dt.hour
    axes = np.atleast_2d(fig.subplots(len(metrics), 2, sharey="row"))
    rng = np.random.default_rng(42)
    for i, (m, yl) in enumerate(metrics):
        for j, ln in enumerate((params.light_label, params.dark_label)):
            ax = axes[i, j]
            msk = (hrs.map(params.light_of) == ln).values
            for k, r in enumerate(fc.ROLES):
                a, b = on["%s|%s" % (m, r)].values[msk], off["%s|%s" % (m, r)].values[msk]
                for x, vals, col, al in ((k - 0.21, b, OFF_COLOR, 0.5), (k + 0.21, a, STIM_COLOR, 0.85)):
                    v = vals[np.isfinite(vals)]
                    if len(v) == 0:
                        continue
                    bp = ax.boxplot([v], positions=[x], widths=0.34, patch_artist=True, showfliers=False,
                                    medianprops=dict(color="black", lw=1.5))
                    bp["boxes"][0].set_facecolor(col)
                    bp["boxes"][0].set_alpha(al)
                    ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, color=col, s=6, alpha=0.55, linewidths=0, zorder=5)
                row = ST[(ST.pair == "ON vs OFF") & (ST.metric == m) & (ST.zone == r) & (ST.grouping == "light")
                         & (ST.group == ln)] if len(ST) else ST
                if len(row):
                    rr = row.iloc[0]
                    both = np.concatenate([a, b])
                    if np.isfinite(both).any():
                        ax.text(k, np.nanmax(both), "d=%+.3g\n%s" % (rr.delta, stars(rr.p_on_gt_off)),
                                ha="center", va="bottom", fontsize=fs["small"])
            ax.set_xticks(range(4))
            ax.set_xticklabels([fc.ROLE_LABELS[r] for r in fc.ROLES], fontsize=fs["tick"])
            if j == 0:
                ax.set_ylabel(yl, fontsize=fs["label"] - 1)
            if i == 0:
                ax.set_title("%s  (n=%d cycles)" % (ln, int(msk.sum())), fontsize=fs["label"])
            _clean(ax, fs)
    offp = mpatches.Rectangle((0, 0), 1, 1, fc=OFF_COLOR, alpha=0.5, ec="black", lw=0.8)
    onp = mpatches.Rectangle((0, 0), 1, 1, fc=STIM_COLOR, alpha=0.85, ec="black", lw=0.8)
    axes[0, 1].legend([offp, onp], ["OFF %d-%d min" % params.off, "ON %d-%d min" % params.on],
                      fontsize=fs["small"] + 1, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.text(0.01, 0.003, SIG_KEY, fontsize=fs["small"] - 0.5, style="italic", color="#444444")
    fig.suptitle(title or ("%s: ON vs OFF per hour-cycle, speaker vs the three other walls" % label),
                 fontsize=fs["title"], y=1.0)
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    return fig


def background_image(exp):
    """RGB background frame (background_with_borders.png, else a middle individual_images frame), or None."""
    import cv2
    p = os.path.join(exp, "background_with_borders.png")
    img = cv2.imread(p) if os.path.isfile(p) else None
    if img is None:
        d = os.path.join(exp, "individual_images")
        if os.path.isdir(d):
            imgs = sorted(f for f in os.listdir(d) if f.lower().endswith(".png") and not f.startswith("."))
            if imgs:
                img = cv2.imread(os.path.join(d, imgs[len(imgs) // 2]))
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_geometry_png(G, order, exp, path):
    """zones by role, their neighbourhoods and the wall sectors over the cage image (fp_figs.fig_geometry).
    Check this first: it shows which zone the analysis treats as speaker / opposite / top / bottom."""
    import cv2
    g = G["g"]
    img = background_image(exp)
    if img is None:
        x1, y1 = int(g["bbox"][1]) + 100, int(g["bbox"][3]) + 100
        img = np.zeros((y1, x1, 3), np.uint8)
    img = img.copy()
    cols = {"speaker": (181, 23, 158), "right": (69, 117, 180), "top": (26, 152, 80), "bottom": (127, 127, 127)}
    for r, zi in zip(fc.ROLES, order):
        n = G["names"][zi]
        poly = np.round(g["zones"][n]).astype(np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [poly], cols[r])
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        cv2.polylines(img, [poly], True, cols[r], 3)
        mask = (G["dms"][zi] <= G["Rpx"][fc.R_MAIN]).astype(np.uint8) * 255
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            cv2.polylines(img, [(c * fc.DS).astype(np.int32)], True, cols[r], 2, cv2.LINE_AA)
        c0 = poly.mean(axis=0).astype(int)
        txt = "%s (%s)" % (fc.ROLE_LABELS[r], n)
        cv2.putText(img, txt, (int(c0[0]) - 60, int(c0[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * max(g["s"], 0.8),
                    (255, 255, 255), 2)
    ctr = tuple(int(v) for v in np.round(g["centre"]))
    cage = np.round(g["cage"]).astype(np.int32)
    for p in cage:
        cv2.line(img, ctr, (int(p[0]), int(p[1])), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.polylines(img, [cage], True, (0, 255, 0), 2)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


# name -> (builder kind, file name); the GUI offers the same list inline
FIGURES = [("arousal", "Arousal through the hour", "arousal_timecourse.png"),
           ("boxes", "Speaker vs the other walls (per cycle)", "zone_comparison_flight.png"),
           ("share", "Flight share near each wall", "flight_share_timecourse.png"),
           ("distance", "Distance of flyers from each wall", "distance_to_walls_timecourse.png")]


def build_figure(kind, result, scale=1.0, dpi=DPI, figsize=None):
    """build one of FIGURES from a pipeline result dict."""
    p = result["params_obj"]
    label = result["label"]
    kw = dict(scale=scale, dpi=dpi)
    if figsize is not None:
        kw["figsize"] = figsize
    if kind == "arousal":
        return fig_arousal(result["minutes"], label, p, **kw)
    if kind == "share":
        return fig_share_timecourse(result["minutes"], label, p, **kw)
    if kind == "distance":
        return fig_distance_timecourse(result["dist_minutes"], label, p, **kw)
    if kind == "boxes":
        if figsize is None and scale < 1.0:
            kw["figsize"] = (13, 7)
            return fig_boxes(result["cycle_metrics"], result["stats"], label, p,
                             metrics=DEFAULT_BOX_METRICS[:2], **kw)
        return fig_boxes(result["cycle_metrics"], result["stats"], label, p, **kw)
    raise KeyError(kind)
