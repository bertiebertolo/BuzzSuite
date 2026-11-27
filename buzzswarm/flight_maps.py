#!/usr/bin/env python3
"""Fruitless-paper style BuzzSwarm figures, drawn from the day caches: smooth heatmaps, flight
paths, the longest flights and GIFs of each clock window (e.g. dawn, midday, dusk).

They use the engine's strict flights (the day cache's 'tracks': pieces of >= MIN_FLIGHT_FRAMES
frames after the step split, the speed bounds and the density-artifact filter), cut to the window
-- the same flights as the fru2 paper figures (flights_in_zt_window). Unlike clock_analysis's
real-flight metrics, nothing here depends on the swarm-zone radius.

  heatmaps_paper     rows = windows, columns = groups: every flight point of the group's days in
                     the window, counted in BINS x BINS bins across the cage (the paper: 1000),
                     shown as log(1 + count) scaled to the panel's own 99th percentile, white ->
                     group colour (or white -> one chosen colour, or a colormap), smoothed
                     (bilinear)
  flights_*          per experiment-day: every flight in each window, one panel per window
  top*_*             per experiment-day and window: the longest flights (most frames) first, one
                     panel each (the paper's individual-trajectory montage, 30 per page)
  gifs/*.gif         per experiment-day and window: the flights replayed, sped up, the paths
                     building up over each other until the last frame shows them all (make_gif)

Coordinates are image pixels relative to the cage centroid, in image orientation (camera below),
so several experiments share one frame. Figures are plain matplotlib Figure objects (no pyplot),
safe to draw in a worker thread.
"""

import math
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

try:
    from buzzswarm import clock_analysis as ca
    from buzzswarm import clock_plots as cp
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import clock_analysis as ca
    import clock_plots as cp


HEATMAP_BINS = 1000            # bins across the cage (fru2's HEATMAP_BINS)
MARGIN_PX = 20                 # around the cage, as fru2's heatmap bounds
GROUP_COLOUR = cp.GROUP_COLOUR  # white -> the group's colour (the paper's heatmaps)
# the choices offered: each group's colour, one colour (white -> it), or a colormap; any colour
# name or code (e.g. #1f77b4) and any matplotlib colormap name are accepted too
COLOURS = (GROUP_COLOUR, 'black', 'red', 'blue', 'green', 'purple', 'orange',
           'magma', 'inferno', 'viridis', 'plasma', 'cividis', 'hot', 'turbo',
           'Greys', 'Blues', 'Reds', 'YlOrRd')
TOP_FLIGHTS = 30               # longest flights per window (fru2: 30 per montage page)
TOP_COLS = 5
GIF_SPEEDUP = 60               # a 30-min window plays in 30 s
GIF_FPS = 10
GIF_BUILT_ALPHA = 0.25         # flights already flown stay drawn this faintly and overlay each other
GIF_HOLD_MS = 2500             # the last frame (every flight of the window) stays up this long
GIF_PHOTO_FADE = 0.7           # the cage picture behind a GIF keeps 70% of its contrast (paths show)
_FRAME_RE = re.compile(r'^.+_(\d{8})_(\d{6})_(\d{5})\.png$', re.IGNORECASE)   # individual_images/
GIF_MAX_FRAMES = 3000
GIF_INCHES, GIF_DPI = 5.0, 90
_NS = 1_000_000_000


def check_bins(bins):
    """Heatmap bins across the cage: a whole number from 2 to 2000."""
    try:
        n = int(bins)
    except (TypeError, ValueError):
        raise ValueError(f'heatmap bins must be a whole number, not {bins!r}') from None
    if not 2 <= n <= 2000:
        raise ValueError(f'heatmap bins must be between 2 and 2000, not {n}')
    return n


# the heatmap colour, shared with the grid heatmaps: GROUP_COLOUR (white -> each group's colour,
# as fru2's sex_cmap), one colour (white -> it) or a matplotlib colormap
check_colour = cp.check_heat_colour
heat_cmap = cp.heat_colormap


def window_flights(tracks, date_str, window):
    """The strict flights' points inside a clock window, as a list of (xy (n, 2) float, t_ns (n,))
    with n >= 2 -- fru2's flights_in_zt_window: each flight cut to its points in [start, end)."""
    _label, lo_min, hi_min = ca.parse_window(window)
    d0 = np.datetime64(datetime.strptime(date_str, '%Y-%m-%d'), 'ns').astype(np.int64)
    lo, hi = d0 + lo_min * 60 * _NS, d0 + hi_min * 60 * _NS
    t = tracks['t_ns']
    sel = np.nonzero((t >= lo) & (t < hi))[0]
    if not len(sel):
        return []
    piece = np.searchsorted(tracks['offsets'], sel, side='right') - 1
    out = []
    for idx in np.split(sel, np.nonzero(np.diff(piece))[0] + 1):
        if len(idx) >= 2:
            out.append((tracks['coords'][idx], t[idx]))
    return out


def _centred(flights, centroid):
    c = np.asarray(centroid, dtype=float)
    return [(xy - c, t) for xy, t in flights]


def _path_px(xy):
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def _blank(ax, half):
    ax.set_xlim(-half, half)
    ax.set_ylim(half, -half)                       # image orientation
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _window_file(window):
    return ca.parse_window(window)[0].replace(':', '')


class PaperFigures:
    """Collects the paper-style figures one day at a time (a day is ~150 MB, so it is never kept):
    add_day() draws that day's flight paths, top flights and GIFs and adds its points to the
    heatmaps; finish() draws the heatmaps and returns every file written.

    experiments: run()'s list of dicts {'folder', 'alias', 'group', 'days'}. windows: clock windows
    ('HH:MM-HH:MM'). Writes into out_dir (flight figures in flights/, GIFs in gifs/)."""

    def __init__(self, experiments, windows, out_dir, bins=HEATMAP_BINS, colour=GROUP_COLOUR,
                 heatmaps=True, flights=True, gifs=False, speedup=GIF_SPEEDUP, top=TOP_FLIGHTS,
                 dpi=120, cancel=None, progress=None):
        self.windows = [ca.parse_window(w)[0] for w in windows]
        self.out_dir = out_dir
        self.bins = check_bins(bins)
        self.colour = check_colour(colour)
        self.do_heat, self.do_flights, self.do_gifs = heatmaps, flights, gifs
        self.speedup = float(speedup)
        if gifs:
            gif_frames(self.windows, self.speedup)        # fails now, not after an hour of work
        self.top = int(top)
        self.dpi = dpi
        self.cancel = cancel
        self.progress = progress
        self.cage = {}
        for e in experiments:
            alias = e.get('alias') or ca.default_alias(e['folder'])
            self.cage[alias] = ca.cage_geometry(e['folder'])
        self.half = max(float(np.abs(np.asarray(poly) - np.asarray(cen, dtype=float)).max())
                        for cen, poly in self.cage.values()) + MARGIN_PX
        self.edges = np.linspace(-self.half, self.half, self.bins + 1)
        self.groups = list(dict.fromkeys(e.get('group') or e.get('alias') or ca.default_alias(e['folder'])
                                         for e in experiments))
        self.colors = cp.group_colors(self.groups)
        self.heat = {}            # (window, group) -> counts [x bin, y bin]
        self.heat_days = {}       # (window, group) -> days added
        self.flight_paths, self.gif_paths = [], []

    # ---------------------------------------------------------------- per day
    def add_day(self, exp, date_str, day):
        """One experiment-day: exp is a dict with 'alias' and 'group' (run()'s on_day hook)."""
        alias = exp.get('alias') or ca.default_alias(exp['folder'])
        group = exp.get('group') or alias
        centroid, polygon = self.cage[alias]
        per_window = []
        for w in self.windows:
            fl = _centred(window_flights(day['tracks'], date_str, w), centroid)
            per_window.append((w, fl))
            if self.do_heat and fl:
                pts = np.concatenate([xy for xy, _t in fl])
                H = np.histogram2d(pts[:, 0], pts[:, 1], [self.edges, self.edges])[0]
                key = (w, group)
                self.heat[key] = self.heat.get(key, 0) + H
                self.heat_days[key] = self.heat_days.get(key, 0) + 1
        tag = f'{cp.safe_name(alias)}_{date_str}'
        cage = np.asarray(polygon, dtype=float) - np.asarray(centroid, dtype=float)
        if self.do_flights:
            fdir = os.path.join(self.out_dir, 'flights')
            os.makedirs(fdir, exist_ok=True)
            fig, table = flight_paths(per_window, self.colors[group], self.half, f'{alias}  {date_str}')
            path = os.path.join(fdir, f'flights_{tag}.png')
            fig.savefig(path, dpi=self.dpi)
            table.to_csv(os.path.join(fdir, f'flights_{tag}.csv'), index=False)
            self.flight_paths.append(path)
            for w, fl in per_window:
                fig = top_flights(fl, self.colors[group], cage, self.top)
                if fig is None:
                    continue
                path = os.path.join(fdir, f'top{self.top}_{tag}_{_window_file(w)}.png')
                fig.savefig(path, dpi=self.dpi)
                self.flight_paths.append(path)
        if self.do_gifs:
            gdir = os.path.join(self.out_dir, 'gifs')
            os.makedirs(gdir, exist_ok=True)
            for w, fl in per_window:
                path = os.path.join(gdir, f'{tag}_{_window_file(w)}.gif')
                picture = gif_picture(exp['folder'], date_str, w) if exp.get('folder') else None
                make_gif(fl, date_str, w, path, self.colors[group], cage, self.half, self.speedup,
                         title=f'{alias}  {date_str}', cancel=self.cancel, picture=picture,
                         centroid=centroid)
                self.gif_paths.append(path)

    # ---------------------------------------------------------------- end
    def finish(self):
        """Draw the heatmaps; every file written, heatmaps first."""
        paths = []
        if self.do_heat and self.heat:
            fig = paper_heatmaps(self.heat, self.heat_days, self.windows, self.groups, self.colors,
                                 self.edges, self.colour)
            path = os.path.join(self.out_dir, 'heatmaps_paper.png')
            os.makedirs(self.out_dir, exist_ok=True)
            fig.savefig(path, dpi=max(self.dpi, 200))
            arrays = {'edges_px': self.edges}
            for i, (key, H) in enumerate(sorted(self.heat.items())):
                arrays[f'counts_{i}'] = H.astype(np.int64)
                arrays[f'label_{i}'] = np.array([key[0], key[1], str(self.heat_days[key])])
            np.savez_compressed(os.path.join(self.out_dir, 'heatmaps_paper_counts.npz'), **arrays)
            paths.append(path)
        return list(dict.fromkeys(paths + self.flight_paths + self.gif_paths))


def gif_frames(windows, speedup):
    """Frames per GIF for the longest window; ValueError when the speed-up is out of range."""
    if not (np.isfinite(speedup) and speedup > 0):
        raise ValueError(f'the GIF speed-up must be a positive number, not {speedup!r}')
    longest = max(hi - lo for _w, lo, hi in (ca.parse_window(w) for w in windows))
    n = int(math.ceil(longest * 60.0 / speedup * GIF_FPS))
    if n > GIF_MAX_FRAMES:
        raise ValueError(f'a {longest}-min window at {speedup:g}x would be {n} GIF frames; use a '
                         f'speed-up of at least {math.ceil(longest * 60.0 * GIF_FPS / GIF_MAX_FRAMES)}x')
    return n


# ==== figures ====

def paper_heatmaps(heat, heat_days, windows, groups, colors, edges, colour=GROUP_COLOUR):
    """Rows = windows, columns = groups, drawn as fru2's plot_heatmap_single."""
    half = float(edges[-1])
    fig = Figure(figsize=(max(4.0 * len(groups) + 0.4, 6.0), 4.0 * len(windows) + 1.0),
                 layout='constrained')
    axes = fig.subplots(len(windows), len(groups), squeeze=False)
    for ri, w in enumerate(windows):
        for ci, g in enumerate(groups):
            ax = axes[ri, ci]
            _blank(ax, half)
            H = heat.get((w, g))
            n_days = heat_days.get((w, g), 0)
            ax.set_title(f'{w}  {g}' + (f'  ({cp._n_days(n_days)})' if n_days else ''), fontsize=9)
            if H is None or not H.any():
                ax.text(0.5, 0.5, 'no flights', ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='0.5')
                continue
            Hvis = np.log1p(H)
            p99 = np.percentile(Hvis, 99) if Hvis.max() > 0 else 1.0
            if p99 > 0:
                Hvis = np.clip(Hvis / p99, 0, 1)
            ax.imshow(Hvis.T, origin='upper', extent=[-half, half, half, -half],
                      cmap=heat_cmap(colour, colors[g]), interpolation='bilinear')
    fig.suptitle('Where they fly: every flight >= 2 s in the window\n(seen from below; '
                 f'{len(edges) - 1} x {len(edges) - 1} bins, log scale,\neach panel scaled to its '
                 'own 99th percentile)', fontsize=10)
    return fig


def flight_paths(per_window, color, half, title):
    """One experiment-day: every flight in each window, as fru2's per-session combined
    trajectories. Returns (figure, one row per flight)."""
    fig = Figure(figsize=(4.2 * len(per_window), 4.7), layout='constrained')
    axes = fig.subplots(1, len(per_window), squeeze=False)[0]
    rows = []
    for ax, (w, fl) in zip(axes, per_window):
        _blank(ax, half)
        if fl:
            ax.add_collection(LineCollection([xy for xy, _t in fl], colors=color, alpha=0.45,
                                             linewidths=0.8))
        ax.set_title(f'{w}  ({len(fl)} flights)', fontsize=9)
        for i, (xy, t) in enumerate(fl):
            rows.append({'window': w, 'flight': i, 'start': pd.Timestamp(int(t[0])),
                         'end': pd.Timestamp(int(t[-1])), 'n_frames': len(xy),
                         'duration_s': (t[-1] - t[0]) / 1e9, 'path_px': _path_px(xy)})
    fig.suptitle(f'{title}: every flight >= 2 s', fontsize=10)
    cols = ['window', 'flight', 'start', 'end', 'n_frames', 'duration_s', 'path_px']
    return fig, pd.DataFrame(rows, columns=cols)


def top_flights(flights, color, cage, n=TOP_FLIGHTS):
    """The n longest flights (most frames) first, one panel each on the whole cage, as fru2's
    trajectory montage. None when there are no flights."""
    if not flights:
        return None
    order = sorted(range(len(flights)), key=lambda i: len(flights[i][0]), reverse=True)[:n]
    cols = min(TOP_COLS, len(order))
    rows = int(math.ceil(len(order) / float(cols)))
    lo, hi = cage.min(axis=0) - MARGIN_PX, cage.max(axis=0) + MARGIN_PX
    fig = Figure(figsize=(cols * 2.6, rows * 2.6), layout='constrained')
    axes = fig.subplots(rows, cols, squeeze=False)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(hi[1], lo[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        if k >= len(order):
            ax.set_axis_off()
            continue
        xy, t = flights[order[k]]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.9, alpha=0.9)
        ax.text(0.03, 0.97, '%.1f s' % ((t[-1] - t[0]) / 1e9), transform=ax.transAxes, ha='left',
                va='top', fontsize=7, color='0.45')
    return fig


# ==== GIFs ====

def gif_picture(experiment_dir, date_str, window):
    """The camera frame a GIF of this window is drawn on: the frame in individual_images/ (one per
    video segment) from the segment nearest the window's middle on that day, else the experiment's
    reference frame (clock_analysis.cage_reference_image). None if there is none."""
    _label, lo_min, hi_min = ca.parse_window(window)
    mid = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(minutes=(lo_min + hi_min) / 2.0)
    folder = os.path.join(experiment_dir, 'individual_images')
    best = None
    names = os.listdir(folder) if os.path.isdir(folder) else []
    for name in names:
        m = _FRAME_RE.match(name)
        if not m:
            continue
        try:
            start = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        except ValueError:
            continue
        middle = start + timedelta(seconds=ca.SEGMENT_SECONDS * (int(m.group(3)) + 0.5))
        gap = abs((middle - mid).total_seconds())
        if best is None or gap < best[0]:
            best = (gap, os.path.join(folder, name))
    if best is not None and best[0] <= 6 * 3600:
        return best[1]
    return ca.cage_reference_image(experiment_dir)


def _load_picture(path):
    """A camera frame as grey values 0-1 (contrast stretched, then faded towards white by
    GIF_PHOTO_FADE so the paths stand out), or None when it cannot be read."""
    if not path:
        return None
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as img:
            grey = ImageOps.autocontrast(img.convert('L'), cutoff=1)
            g = np.asarray(grey, dtype=float) / 255.0
    except Exception:
        return None
    return 1.0 - GIF_PHOTO_FADE * (1.0 - g)


def _gif_palette(color):
    """A fixed palette so every frame maps the same way: greys (the cage picture, text), white ->
    colour, the colour over grey at the alphas the paths build up to, and black -> colour (dots)."""
    from PIL import Image
    c = np.array(to_rgb(color))
    a = np.linspace(0, 1, 48)[:, None]
    greys = np.linspace(1.0 - GIF_PHOTO_FADE, 1.0, 12)
    alphas = 1.0 - (1.0 - GIF_BUILT_ALPHA) ** np.arange(1, 10)        # 1, 2, ... 9 paths on top
    alphas = np.append(alphas[:8], 0.95)
    over = [g * (1 - al) + c * al for g in greys for al in alphas]
    cols = np.vstack([np.linspace(1, 0, 64)[:, None].repeat(3, 1), 1 - a * (1 - c), np.array(over),
                      np.linspace(0, 1, 36)[:, None] * c])
    pal = np.clip(np.rint(cols * 255), 0, 255).astype(np.uint8)[:256]
    img = Image.new('P', (1, 1))
    img.putpalette(pal.ravel().tolist() + [0] * (768 - pal.size))
    return img


def make_gif(flights, date_str, window, path, color, cage, half, speedup=GIF_SPEEDUP, title='',
             cancel=None, picture=None, centroid=None):
    """Replay one window's flights (centred coordinates) as a GIF, speedup x real time, GIF_FPS
    frames a second. The paths build up: everything flown so far stays drawn faintly
    (GIF_BUILT_ALPHA), so the flights overlay each other and busy places darken towards the full
    colour; the last frame shows every flight of the window and is held GIF_HOLD_MS. The latest
    step is drawn strongly, with a dark dot where each flyer is. picture: a camera frame (path,
    e.g. gif_picture()) drawn behind in grey, with centroid the cage centre in its pixels; without
    one the cage outline is drawn on white. A set `cancel` Event raises Cancelled."""
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    label, lo_min, hi_min = ca.parse_window(window)
    n_frames = gif_frames([window], speedup)
    d0 = np.datetime64(datetime.strptime(date_str, '%Y-%m-%d'), 'ns').astype(np.int64)
    lo = d0 + lo_min * 60 * _NS
    step = int(round(speedup * _NS / GIF_FPS))
    if flights:
        xy = np.concatenate([f[0] for f in flights])
        t = np.concatenate([f[1] for f in flights])
        fid = np.repeat(np.arange(len(flights)), [len(f[1]) for f in flights])
        o = np.argsort(t, kind='stable')
        xy, t, fid = xy[o], t[o], fid[o]
    else:
        xy, t, fid = np.empty((0, 2)), np.empty(0, np.int64), np.empty(0, int)
    # each point's previous point in its own flight (-1 for a flight's first), so that a step's
    # path joins on to the step before
    prev = np.full(len(t), -1)
    if len(t):
        k = np.lexsort((np.arange(len(t)), fid))       # by flight, then in time order
        same = fid[k][1:] == fid[k][:-1]
        prev[k[1:][same]] = k[:-1][same]

    fig = Figure(figsize=(GIF_INCHES, GIF_INCHES + 0.35), dpi=GIF_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.01, 0.01, 0.98, GIF_INCHES / (GIF_INCHES + 0.35) - 0.02])
    grey = _load_picture(picture) if centroid is not None else None
    if grey is not None:                               # the camera's view of the cage, in grey
        cx, cy = (float(v) for v in centroid)
        h, w = grey.shape
        ax.imshow(grey, cmap='gray', vmin=0, vmax=1, origin='upper', interpolation='bilinear',
                  extent=[-0.5 - cx, w - 0.5 - cx, h - 0.5 - cy, -0.5 - cy], zorder=0)
    else:
        ring = np.vstack([cage, cage[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color='0.75', lw=0.8)
    _blank(ax, half)
    fig.text(0.98, 0.985, f'{title}  ({speedup:g}x)', ha='right', va='top', fontsize=8, color='0.35')
    # drawn by hand on top of a kept background (blitting), not by canvas.draw()
    built = LineCollection([], colors=color, alpha=GIF_BUILT_ALPHA, linewidths=0.7, animated=True)
    new = LineCollection([], colors=color, alpha=0.95, linewidths=1.3, animated=True)
    ax.add_collection(built)
    ax.add_collection(new)
    heads = ax.scatter([], [], s=10, color=np.array(to_rgb(color)) * 0.5, zorder=3, animated=True)
    text = fig.text(0.02, 0.985, '', ha='left', va='top', fontsize=9, animated=True)
    palette = _gif_palette(color)
    dither = getattr(getattr(Image, 'Dither', Image), 'NONE', 0)
    frames = []

    def segments(a, b):
        """The flights' paths through the points [a, b), each joined to its point before a."""
        if b <= a:
            return []
        idx = np.arange(a, b)[np.argsort(fid[a:b], kind='stable')]
        out = []
        for g in np.split(idx, np.nonzero(np.diff(fid[idx]))[0] + 1):
            if prev[g[0]] >= 0:
                g = np.concatenate([[prev[g[0]]], g])
            if len(g) >= 2:
                out.append(xy[g])
        return out

    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)      # the cage and title; then the built-up paths
    done = 0                                           # points already in the background
    for i in range(n_frames):
        if cancel is not None and i % 25 == 0 and cancel.is_set():
            raise ca.Cancelled()
        t1 = lo + (i + 1) * step
        b = int(np.searchsorted(t, t1))
        canvas.restore_region(background)
        if b > done:                                   # this step joins the built-up paths
            built.set_segments(segments(done, b))
            ax.draw_artist(built)
            background = canvas.copy_from_bbox(fig.bbox)
            new.set_segments(segments(done, b))
            ax.draw_artist(new)
            last = np.unique(fid[done:b][::-1], return_index=True)[1]
            heads.set_offsets(xy[done:b][::-1][last])
            ax.draw_artist(heads)
        done = b
        clock = pd.Timestamp(int(t1)).strftime('%H:%M:%S')
        text.set_text(f'{clock}   {label}')
        fig.draw_artist(text)
        rgb = Image.fromarray(np.array(canvas.buffer_rgba())[..., :3])
        frames.append(rgb.quantize(palette=palette, dither=dither))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    durations = [int(1000 / GIF_FPS)] * len(frames)
    durations[-1] = GIF_HOLD_MS
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return path


# ==== from the day caches (Redraw, Make GIFs) ====

def from_caches(figures, experiments, cache_root=None, workers=None, progress=None, cancel=None):
    """Feed every experiment-day to `figures` (a PaperFigures) from its day cache (built if
    missing, as Run analysis does), one day at a time. progress(fraction, message)."""
    units = [(e, d) for e in experiments for d in e['days']]
    for i, (e, date_str) in enumerate(units):
        if cancel is not None and cancel.is_set():
            raise ca.Cancelled()
        alias = e.get('alias') or ca.default_alias(e['folder'])
        if progress:
            progress(i / float(len(units)), f'{alias} | {date_str} | flight figures')
        segments = ca.list_segments(e['folder'])
        n_workers = workers or ca.auto_workers(len(ca.day_files(segments, date_str)))
        day = ca.load_or_build_day(e['folder'], date_str, ca.day_cache_dir(e['folder'], alias, cache_root),
                                   segments=segments, workers=n_workers, cancel=cancel)
        figures.add_day(dict(e, alias=alias), date_str, day)
        del day
    if progress:
        progress(1.0, 'flight figures done')
    return figures
