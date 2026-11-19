"""Shared helpers for the feeder-zone tracking checks (see README.md in this folder).

Everything here is read-only with respect to the experiment: it reads final_tracking_data/,
custom_zones.json, plots/buzzswarm/geometry.json and plots/buzzswarm/day_cache/, and the tools only
write the output file(s) you name. Python 3.8 (runs in the buzzwatch_env conda env).
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from buzzswarm import clock_analysis as ca  # noqa: E402  (needs REPO on sys.path)
import ld_dd  # noqa: E402

m = ca.m

# The tracker stores a track's coordinate k at video frame (track start + k + FRAME_OFFSET).
# Same convention as VideoManager.overlay_tracking (coordinates[frame - start + 2]); confirmed on
# video: detections land on the darkest pixels at -2 and nowhere near them at -1 or -3.
FRAME_OFFSET = -2

# Same-distance-from-centre comparison band used for the feeder zones (px from the cage centroid).
# Matches the radius span of the Penzance sugar/speaker polygons; override with --ring.
DEFAULT_RING = (230.0, 354.0)

# Chart palette (sequential blue, diverging red <-> blue, ink and surfaces).
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ_STEPS = [SURF, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIV_STEPS = ["#b3261e", "#e34948", "#f0efec", "#3987e5", "#184f95"]


def load_geometry(exp):
    """(geometry.json dict, {zone name: polygon}) for an experiment that BuzzSwarm has run on."""
    gpath = os.path.join(exp, "plots", "buzzswarm", "geometry.json")
    zpath = os.path.join(exp, "custom_zones.json")
    if not os.path.isfile(gpath):
        sys.exit("No %s - run the BuzzSwarm clock analysis on this experiment first." % gpath)
    if not os.path.isfile(zpath):
        sys.exit("No %s - draw the zones (BuzzPhono custom zones) first." % zpath)
    with open(gpath) as f:
        geo = json.load(f)
    with open(zpath) as f:
        zones = {k: v["polygon"] for k, v in json.load(f).items() if "polygon" in v}
    return geo, zones


def resolve_days(exp, days):
    """Dates to use: the ones given, else BuzzSwarm's saved day selection, else every full day."""
    if days:
        return list(days)
    cov = ca.date_coverage(exp)
    return ca.default_days(cov, ld_dd.load_schedule(exp))


def days_label(days):
    """'2026-07-05 to 2026-07-07 (3 days)' for figure titles."""
    if len(days) == 1:
        return days[0]
    return "%s to %s (%d days)" % (min(days), max(days), len(days))


def segment_path(exp, segment):
    return os.path.join(exp, "final_tracking_data", "forward_mosq_tracks_" + segment)


def grid_masks(geo, zones, edges, ring=DEFAULT_RING):
    """Cell-centre masks on a histogram grid: inside cage, each zone, and the comparison walls band."""
    from matplotlib.path import Path
    bx = edges[0][1] - edges[0][0]
    by = edges[1][1] - edges[1][0]
    X, Y = np.meshgrid(edges[0][:-1] + bx / 2.0, edges[1][:-1] + by / 2.0, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel()]
    inside = Path(geo["polygon"]).contains_points(pts).reshape(X.shape)
    zm = {k: Path(p).contains_points(pts).reshape(X.shape) & inside for k, p in zones.items()}
    cx, cy = geo["centroid"]
    r = np.hypot(X - cx, Y - cy)
    anyz = np.any(list(zm.values()), 0) if zm else np.zeros_like(inside)
    walls = inside & ~anyz & (r > ring[0]) & (r < ring[1])
    return inside, zm, walls


def real_flight_coords(day):
    """The points the BuzzSwarm heatmap uses: flight pieces' points with smoothed speed >= FLY_PX_FRAME."""
    sp = np.nan_to_num(ca.smoothed_motion(day["flight"])[0], nan=0.0)
    return day["flight"]["coords"][sp >= ca.FLY_PX_FRAME]


def load_day_cache(exp, date):
    import pickle
    path = os.path.join(exp, "plots", "buzzswarm", "day_cache", "%s.pkl" % date)
    if not os.path.isfile(path):
        sys.exit("No day cache %s - run the BuzzSwarm clock analysis for %s first." % (path, date))
    with open(path, "rb") as f:
        return pickle.load(f)


def style_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.edgecolor": MUTED,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})


def draw_outlines(ax, geo, zones, color=INK, xlim=(150, 990), ylim=(720, 0)):
    poly = np.asarray(geo["polygon"], float)
    ax.plot(*np.vstack([poly, poly[:1]]).T, color=MUTED, lw=1)
    for k, v in zones.items():
        P = np.vstack([v, v[:1]])
        ax.plot(P[:, 0], P[:, 1], color=color, lw=1.6, ls="--")
        ax.text(P[:, 0].mean(), P[:, 1].min() - 8, k, color=color, ha="center", va="bottom", fontsize=9,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
