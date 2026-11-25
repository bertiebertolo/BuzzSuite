#!/usr/bin/env python3
"""Heatmap of moving-tracker detections BEFORE vs AFTER BuzzSwarm's filtering.

Panels: cage background with zones | every raw moving-tracker detection | the real-flight points
the BuzzSwarm heatmap plots | the share of detections the filters keep per cell, relative to the
cage-wide share. Colour = density relative to the cage average (log), so the walls stay visible.
Prints, per zone, the density before/after and the kept share.

Usage (buzzwatch_env):
    python tools/feeder_check/heatmap_before_after.py <experiment_dir> -o before_after.png
        [--days 2026-07-05 2026-07-06] [--workers 6] [--cell 10]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

W, H = 1280, 720


def raw_hist(args):
    path, tunables, edges = args
    for k, v in tunables:
        setattr(c.m, k, v)
    mt = c.m.load_tracking_file(path)
    if mt is None or not mt.time_stamp:
        return None
    pts = [np.asarray(fl["coords"], float) for fl in c.m.extract_raw_flights(mt) if len(fl["coords"])]
    if not pts:
        return None
    p = np.concatenate(pts)
    return np.histogram2d(p[:, 0], p[:, 1], edges)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cell", type=int, default=10, help="cell size in px")
    a = ap.parse_args()

    from multiprocessing import get_context
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap, TwoSlopeNorm

    exp = a.experiment
    geo, zones = c.load_geometry(exp)
    days = c.resolve_days(exp, a.days)
    edges = [np.arange(0, W + a.cell, a.cell), np.arange(0, H + a.cell, a.cell)]
    segs = c.ca.list_segments(exp)
    files = sorted(set(f for d in days for f in c.ca.day_files(segs, d)))
    tun = c.m._session_cache_signature()
    print("days %s, %d tracking files" % (", ".join(days), len(files)))
    with get_context("spawn").Pool(max(1, a.workers), maxtasksperchild=8) as pool:
        before = sum(h for h in pool.imap(raw_hist, [(f, tun, edges) for f in files]) if h is not None)
    after = np.zeros_like(before)
    for d in days:
        p = c.real_flight_coords(c.load_day_cache(exp, d), c.fly_px_frame(exp))
        after += np.histogram2d(p[:, 0], p[:, 1], edges)[0]

    inside, zm, _walls = c.grid_masks(geo, zones, edges)

    def rel(Hm):
        R = Hm / Hm[inside].mean()
        R[~inside] = np.nan
        return R

    Rb, Ra = rel(before), rel(after)
    with np.errstate(divide="ignore", invalid="ignore"):
        kept = (after / before) / (after[inside].sum() / before[inside].sum())
    kept[~inside | (before < 50)] = np.nan
    print("detections before %.3g, after %.3g, kept %.1f%%" %
          (before[inside].sum(), after[inside].sum(), 100 * after[inside].sum() / before[inside].sum()))
    for k, msk in zm.items():
        print("%-10s density before %.2fx, after %.2fx cage average; kept %.2fx the cage-wide share" %
              (k, np.nanmean(Rb[msk]), np.nanmean(Ra[msk]), np.nanmean(kept[msk])))

    c.style_matplotlib()
    seq = LinearSegmentedColormap.from_list("seq", c.SEQ_STEPS)
    seq.set_bad(c.SURF)
    div = LinearSegmentedColormap.from_list("div", c.DIV_STEPS)
    div.set_bad(c.SURF)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.4), facecolor=c.SURF, layout="constrained")
    ext = [0, W, H, 0]
    a0 = axs[0, 0]
    bgp = os.path.join(exp, "background_with_borders.png")
    if os.path.isfile(bgp):
        a0.imshow(plt.imread(bgp), extent=ext)
    c.draw_outlines(a0, geo, zones, color="#eb6834")
    a0.set_title("Cage background with zones", loc="left", fontsize=11)
    hidden = fig.colorbar(plt.cm.ScalarMappable(cmap=seq), ax=a0, shrink=0.85, pad=0.01)
    hidden.ax.set_visible(False)
    norm = LogNorm(vmin=0.05, vmax=20)
    for ax, R, t in [(axs[0, 1], Rb, "BEFORE: every raw moving-tracker detection"),
                     (axs[1, 0], Ra, "AFTER: real flight only (what the heatmap plots)")]:
        im = ax.imshow(R.T, extent=ext, cmap=seq, norm=norm, interpolation="nearest")
        c.draw_outlines(ax, geo, zones)
        ax.set_title(t, loc="left", fontsize=11)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01, ticks=[0.1, 0.3, 1, 3, 10])
        cb.ax.set_yticklabels(["0.1x", "0.3x", "1x", "3x", "10x"])
        cb.set_label("density vs cage average (log)")
        cb.outline.set_visible(False)
    ax = axs[1, 1]
    im = ax.imshow(kept.T, extent=ext, cmap=div, norm=TwoSlopeNorm(1.0, 0.25, 1.75), interpolation="nearest")
    c.draw_outlines(ax, geo, zones)
    ax.set_title("Share of detections the filters keep", loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01, ticks=[0.25, 0.5, 1, 1.5])
    cb.ax.set_yticklabels(["0.25x", "0.5x", "1x", "1.5x"])
    cb.set_label("kept vs cage average (red = removed more)")
    cb.outline.set_visible(False)
    fig.suptitle("%s\nMoving-tracker detections before vs after filtering: %s, all hours, %d px cells" %
                 (geo.get("experiment", os.path.basename(exp)), c.days_label(days), a.cell),
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(a.out, dpi=130, facecolor=c.SURF)
    print("saved", a.out)


if __name__ == "__main__":
    main()
