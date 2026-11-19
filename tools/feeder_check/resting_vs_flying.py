#!/usr/bin/env python3
"""Where mosquitoes REST vs where they FLY, plus resting density in each zone by hour.

Resting = every state-0 point in final_tracking_data (one point per mosquito per frame). Flying =
the real-flight points the BuzzSwarm heatmap uses. The hourly panel compares each zone with the
other walls at the same distance from the cage centre, per unit area, so a feeder visit shows up
as a zone line above the grey walls line.

Usage (buzzwatch_env):
    python tools/feeder_check/resting_vs_flying.py <experiment_dir> -o resting_vs_flying.png
        [--days ...] [--workers 6] [--cell 10] [--ring 230 354] [--night 17 5]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

W, H = 1280, 720


def work(args):
    path, zones, ring, edges = args
    from matplotlib.path import Path
    mt = c.m.load_tracking_file(path)
    if mt is None or not mt.time_stamp:
        return None
    ts = mt.time_stamp
    hrs = np.array([t.hour for t in ts])
    nts = len(ts)
    frames_h = np.bincount(hrs, minlength=24).astype(float)
    pts, ph = [], []
    for o in mt.objects.values():
        cc = o.get("coordinates", []) or []
        st = o.get("state", []) or []
        s = int(o.get("start", 0))
        n = min(len(cc), len(st))
        if not n:
            continue
        idx = np.nonzero(np.asarray(st[:n]) == 0)[0]
        if not len(idx):
            continue
        xy = np.asarray([cc[k] for k in idx], float).reshape(-1, 2)
        fa = s + idx + c.FRAME_OFFSET
        ok = np.isfinite(xy).all(1) & (fa >= 0) & (fa < nts) & (xy[:, 0] > 0) & (xy[:, 1] > 0)
        pts.append(xy[ok])
        ph.append(hrs[fa[ok]])
    if not pts:
        return None
    P = np.concatenate(pts)
    Hh = np.concatenate(ph)
    hist = np.histogram2d(P[:, 0], P[:, 1], edges)[0]
    anyz = np.zeros(len(P), bool)
    per = {}
    for name, poly in zones.items():
        inz = Path(poly).contains_points(P)
        anyz |= inz
        per[name] = np.bincount(Hh[inz], minlength=24).astype(float)
    r = np.hypot(P[:, 0] - ring[0], P[:, 1] - ring[1])
    inw = (~anyz) & (r > ring[2]) & (r < ring[3]) & Path(ring[4]).contains_points(P)
    per["walls"] = np.bincount(Hh[inw], minlength=24).astype(float)
    return hist, frames_h, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cell", type=int, default=10)
    ap.add_argument("--ring", type=float, nargs=2, default=c.DEFAULT_RING,
                    help="comparison walls band, px from the cage centre")
    ap.add_argument("--night", type=int, nargs=2, default=None,
                    help="lights-off and lights-on hour (default: the experiment's LD schedule)")
    a = ap.parse_args()

    from multiprocessing import get_context
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap

    exp = a.experiment
    geo, zones = c.load_geometry(exp)
    days = c.resolve_days(exp, a.days)
    sched = c.ld_dd.load_schedule(exp) or {}
    night = a.night or (sched.get("night_start_hour", 17), sched.get("night_end_hour", 5))
    cx, cy = geo["centroid"]
    ring = (cx, cy, a.ring[0], a.ring[1], geo["polygon"])
    edges = [np.arange(0, W + a.cell, a.cell), np.arange(0, H + a.cell, a.cell)]
    segs = c.ca.list_segments(exp)
    files = sorted(set(f for d in days for f in c.ca.day_files(segs, d)))
    print("days %s, %d tracking files" % (", ".join(days), len(files)))
    with get_context("spawn").Pool(max(1, a.workers), maxtasksperchild=8) as pool:
        res = [r for r in pool.imap(work, [(f, zones, ring, edges) for f in files]) if r is not None]
    h_rest = sum(r[0] for r in res)
    frames_h = sum(r[1] for r in res)
    per = {k: sum(r[2][k] for r in res) for k in res[0][2]}
    h_fly = np.zeros_like(h_rest)
    for d in days:
        p = c.real_flight_coords(c.load_day_cache(exp, d))
        h_fly += np.histogram2d(p[:, 0], p[:, 1], edges)[0]

    inside, zm, wallm = c.grid_masks(geo, zones, edges, a.ring)

    def rel(Hm):
        R = Hm / Hm[inside].mean()
        R[~inside] = np.nan
        return R

    Rr, Rf = rel(h_rest), rel(h_fly)
    area = {k: zm[k].sum() * a.cell ** 2 for k in zm}
    area["walls"] = wallm.sum() * a.cell ** 2
    occ = {k: per[k] / np.maximum(frames_h, 1) / area[k] * 10000 for k in per}
    light = np.array([not (h >= night[0] or h < night[1]) for h in range(24)])
    print("resting mosquitoes per 10,000 px^2 (mean per frame):   lights on | lights off")
    for k in per:
        print("  %-12s %6.2f | %6.2f" % (k, occ[k][light].mean(), occ[k][~light].mean()))
    for h in range(24):
        print("  %02d:00  " % h + "  ".join("%s %.2f" % (k, occ[k][h]) for k in per))

    c.style_matplotlib()
    seq = LinearSegmentedColormap.from_list("seq", c.SEQ_STEPS)
    seq.set_bad(c.SURF)
    fig = plt.figure(figsize=(13, 9.6), facecolor=c.SURF, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ext = [0, W, H, 0]
    norm = LogNorm(vmin=0.05, vmax=20)
    maps = []
    for i, (R, t) in enumerate([(Rr, "RESTING: where mosquitoes sit"),
                                (Rf, "FLYING: real flight (as in the BuzzSwarm heatmap)")]):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(R.T, extent=ext, cmap=seq, norm=norm, interpolation="nearest")
        c.draw_outlines(ax, geo, zones)
        ax.set_title(t, loc="left", fontsize=11)
        maps.append(ax)
    cb = fig.colorbar(im, ax=maps, shrink=0.8, pad=0.01, ticks=[0.1, 0.3, 1, 3, 10])
    cb.ax.set_yticklabels(["0.1x", "0.3x", "1x", "3x", "10x"])
    cb.set_label("density vs cage average (log)")
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    hrs = np.arange(24) + 0.5
    for h in range(24):
        if not light[h]:
            ax.axvspan(h, h + 1, color="#f0efec", lw=0)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical order; walls stay grey
    series = [(k, palette[i % len(palette)], "%s zone" % k) for i, k in enumerate(zones)]
    series.append(("walls", c.MUTED, "other walls, same distance"))
    for k, col, lab in series:
        ax.plot(hrs, occ[k], color=col, lw=2, marker="o", ms=4, label=lab)
        ax.annotate(lab, (hrs[-1], occ[k][-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=c.INK2, fontsize=9)
    ax.set_xlim(0, 27.5)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels(["%02d:00" % h for h in range(0, 25, 3)])
    ax.set_ylabel("mosquitoes resting per 10,000 px²\n(mean per frame)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=c.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=len(series))
    ax.set_title("Resting density by hour (shaded = lights off)", loc="left", fontsize=11)
    fig.suptitle("%s\nResting vs flying positions: %s, all hours, %d px cells" %
                 (geo.get("experiment", os.path.basename(exp)), c.days_label(days), a.cell),
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(a.out, dpi=130, facecolor=c.SURF)
    print("saved", a.out)


if __name__ == "__main__":
    main()
