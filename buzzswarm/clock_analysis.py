#!/usr/bin/env python3
"""Real clock-time aggregation / swarm metrics for multi-day BuzzSwarm recordings (e.g. Flare).

The fru2 engine (fru2_zt_normalized_analysis.py) treats a whole experiment as ONE 12-h day and
squashes it onto ZT 0-12. Flare recordings run continuously for days, so this module instead:

  - analyses one CALENDAR DAY at a time, as its own session (fru2 had one session per 12-h day),
    so the per-session density-artifact filter keeps its original meaning;
  - streams that day's tracking files one at a time (a whole day held as Python objects needs
    several GB) and keeps compact numpy arrays, cached per day in
    plots/buzzswarm/day_cache/{date}.pkl;
  - bins by real clock time and scores editable per-day clock windows (e.g. 16:30-17:00).

Two flight populations per day, both from the engine's own filter steps (step split, average
speed bounds, density artifacts):
  tracks   pieces >= MIN_FLIGHT_FRAMES (50 frames, 2 s) = exactly the engine's strict flights.
           Used for per-flight metrics (r50 / prop_within use the engine's definitions).
  flight   pieces >= FLIGHT_MIN_FRAMES (5 frames, 0.2 s). Frame-level metrics use only its REAL
           flight points: smoothed speed (displacement i-2 -> i+2 inside one piece, / 4 frames)
           >= FLY_PX_FRAME. Speeds are bimodal (frozen positions at 0, flight at ~8-15 px/frame);
           near the walls most tracker "flight" points are frozen and would swamp the metrics.

Camera geometry (Flare): filmed from below looking up, so the image centre is the cage's central
column (at any height) and the centre square is the ceiling. Image area is therefore not a true
"chance" baseline for central metrics: compare periods or groups rather than reading 1 = chance.

The tracker merges two flyers closer than ~30 px into one detection, so swarm counts are lower
bounds in a dense swarm, and "flying together" (near_ratio) is measured just outside that blind
zone against a time-shifted control (flyers paired with the others' positions 1-2 min away).

Metrics per clock bin / window (pixels; distances from the cage centroid):
  n_flying          mean mosquitoes in real flight per recorded frame
  swarm_mean        mean number in real flight inside the swarm zone (swarm_radius_px disc,
                    default SWARM_RADIUS_PX) at the same moment
  swarm_ge3_frac, swarm_ge5_frac   share of recorded frames with >= 3 / >= 5 in the swarm zone
  swarm_p95         swarm size reached or exceeded in the busiest 5% of frames
  central_enrichment  share of real flight within CENTRAL_RADIUS_PX / share of cage area there
  core_density      real-flight density within CORE_PX of the centre per unit area (1 = even)
  shape_index       core density / densest 20-px ring out to SHAPE_MAX_PX (1 = filled centre,
                    low = ring around a quieter middle)
  speed_px_s        median real-flight speed
  turning_deg_s     median turning rate of real flight (smoothed 2-frame steps)
  nn_px             mean nearest-neighbour distance between simultaneous real flyers
  near_ratio        pairs 45-100 px apart, observed / time-shifted control (> 1 = together)
  activity          flights >= 2 s per recorded minute (tracks)
  r50, prop_within_100, prop_within_150   engine definitions (tracks)
  straightness      median net displacement / path length of flights >= 2 s (tracks)
  n_traj, recorded_frac   counts / how much of the bin was recorded
Per window also: the swarm-size distribution, a centre profile (real-flight density per 20-px ring)
and a heatmap (real flight per 20-px cell around the centroid). clock_plots.py draws all of these
from the CSVs.

Read-only with respect to tracking: nothing here changes the tracker, the .pkl format or the engine.
"""

import json
import os
import re
import sys
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

try:
    from buzzswarm import fru2_zt_normalized_analysis as m
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fru2_zt_normalized_analysis as m


SEGMENT_SECONDS = 1200                 # nominal segment length (filename index x this = offset)
DAY_MARGIN = timedelta(minutes=30)     # load segments this far past each day edge (clock drift)
FULL_DAY_COVERAGE = 0.95               # a date counts as a full day at >= this fraction of 24 h
CACHE_SCHEMA = 2
FLIGHT_MIN_FRAMES = 5                  # shortest flight piece kept for frame-level metrics
FLY_PX_FRAME = 3.0                     # smoothed speed for "real flight" (3 px/frame = 75 px/s)
SWARM_RADIUS_PX = 100                  # swarm zone (default): core disc around the cage centroid.
                                       # Inside the ~110-150 px female flight ring, so it separates a
                                       # filled male swarm from flight circling the centre.
CENTRAL_RADIUS_PX = 150                # central_enrichment disc (= population_data's _within_150)
CORE_PX = 60                           # core for core_density / shape_index
SHAPE_RING_PX = 20                     # ring width for shape_index
SHAPE_MAX_PX = 240                     # rings considered for shape_index (ignore the walls)
NEAR_BAND_PX = (45, 100)               # "together" band: outside the ~30 px blind zone + its edge
SHIFTS_S = (-120, -60, 60, 120)        # time-shifted control offsets
HEAT_CELL_PX = 20                      # heatmap cell size (windows only)
MIN_POINTS = 500                       # fewer real-flight points than this -> spatial metrics NaN
MIN_PAIRS = 100                        # fewer simultaneous pairs than this -> pair metrics NaN
MIN_SAMPLES = 100                      # fewer speed / turning samples than this -> NaN
SWARM_MAX_COUNT = 40                   # swarm-size distribution columns 0..SWARM_MAX_COUNT

METRICS = [
    # (column, axis label)
    ('n_flying',           'Mosquitoes in flight (mean per frame)'),
    ('swarm_mean',         'Swarm size: in flight in the centre at once (mean)'),
    ('swarm_ge3_frac',     'Share of time with >= 3 in the swarm'),
    ('swarm_ge5_frac',     'Share of time with >= 5 in the swarm'),
    ('swarm_p95',          'Swarm size at busy moments (95th percentile)'),
    ('central_enrichment', 'Central enrichment (share of flight / share of area)'),
    ('core_density',       'Flight density at the centre (1 = even)'),
    ('shape_index',        'Shape: centre / densest ring (1 = filled, low = ring)'),
    ('speed_px_s',         'Flight speed (px/s, median)'),
    ('turning_deg_s',      'Turning rate (deg/s, median)'),
    ('nn_px',              'Nearest-neighbour distance (px)'),
    ('near_ratio',         'Pairs 45-100 px apart, observed / control (> 1 = together)'),
    ('activity',           'Flights >= 2 s per minute'),
    ('r50',                'r50 (px from cage centre)'),
    ('prop_within_150',    'Fraction of flight within 150 px of centre'),
    ('straightness',       'Straightness of flights >= 2 s (net / path, median)'),
]

_SEG_RE = re.compile(r'^forward_mosq_tracks_(.+)_(\d{8})_(\d{6})_(\d{5})$')
_NS = 1_000_000_000


class Cancelled(Exception):
    """Raised when the caller's cancel Event is set. No partial cache or CSV is written."""


# ==== Segments, dates, windows ====

def list_segments(experiment_dir):
    """[(path, nominal_start, nominal_end)] for every tracked segment, sorted by start. The start
    comes from the filename (session start + index x SEGMENT_SECONDS). Real timestamps drift a few
    seconds per segment, so these are only used to pick files and estimate day coverage."""
    tdir = os.path.join(experiment_dir, 'final_tracking_data')
    out = []
    if not os.path.isdir(tdir):
        return out
    for name in os.listdir(tdir):
        match = _SEG_RE.match(name)
        if not match:
            continue
        try:
            session = datetime.strptime(match.group(2) + match.group(3), '%Y%m%d%H%M%S')
        except ValueError:
            continue
        start = session + timedelta(seconds=SEGMENT_SECONDS * int(match.group(4)))
        out.append((os.path.join(tdir, name), start, start + timedelta(seconds=SEGMENT_SECONDS)))
    out.sort(key=lambda s: s[1])
    return out


def date_coverage(experiment_dir, segments=None):
    """{'YYYY-MM-DD': fraction of that day's 24 h covered by tracked segments}, from filenames
    only (cheap enough to fill a date table without loading any tracking data)."""
    if segments is None:
        segments = list_segments(experiment_dir)
    cov = {}
    for _path, start, end in segments:
        t = start
        while t < end:
            day0 = datetime(t.year, t.month, t.day)
            chunk_end = min(end, day0 + timedelta(days=1))
            key = day0.strftime('%Y-%m-%d')
            cov[key] = cov.get(key, 0.0) + (chunk_end - t).total_seconds() / 86400.0
            t = chunk_end
    return {k: min(1.0, v) for k, v in sorted(cov.items())}


def full_days(experiment_dir, segments=None):
    """Dates covered for at least FULL_DAY_COVERAGE of their 24 h."""
    return [d for d, c in date_coverage(experiment_dir, segments).items() if c >= FULL_DAY_COVERAGE]


def default_days(coverage, schedule=None):
    """Days to analyse by default (tab and CLI): the experiment's saved BuzzSwarm selection if
    there is one, else every full day not tagged DD in its light schedule. `coverage` is
    date_coverage()'s dict, `schedule` ld_dd.load_schedule()'s."""
    try:
        from ld_dd import BUZZSWARM_DAYS_KEY
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ld_dd import BUZZSWARM_DAYS_KEY
    schedule = schedule or {}
    saved = schedule.get(BUZZSWARM_DAYS_KEY)
    if isinstance(saved, list):
        return [d for d in coverage if d in saved]
    tags = schedule.get('dates') or {}
    return [d for d, c in coverage.items() if c >= FULL_DAY_COVERAGE and tags.get(d) != 'DD']


def day_files(segments, date_str):
    """Tracking files whose nominal span overlaps the day (plus DAY_MARGIN on each side)."""
    d0 = datetime.strptime(date_str, '%Y-%m-%d')
    d1 = d0 + timedelta(days=1)
    return [p for p, s, e in segments if s < d1 + DAY_MARGIN and e > d0 - DAY_MARGIN]


def parse_window(text):
    """'16:30-17:00' -> ('16:30-17:00', 990, 1020) as minutes after midnight. Windows may not
    cross midnight (a day is analysed on its own)."""
    parts = text.replace('–', '-').split('-')
    if len(parts) != 2:
        raise ValueError(f'window {text!r} is not HH:MM-HH:MM')
    mins = []
    for p in parts:
        hh, mm = p.strip().split(':')
        mins.append(int(hh) * 60 + int(mm))
    lo, hi = mins
    if not (0 <= lo < hi <= 1440):
        raise ValueError(f'window {text!r} must start before it ends, within one day')
    return f'{lo // 60:02d}:{lo % 60:02d}-{hi // 60:02d}:{hi % 60:02d}', lo, hi


def default_windows(night_start=17, night_end=5):
    """30-min clock windows from the light schedule: just after lights-on ('dawn'), centred on the
    middle of the light phase (fru2's midday window) and just before lights-off (fru2's dusk)."""
    off = int(night_start) * 60
    on = int(night_end) * 60
    light_len = (off - on) % 1440 or 1440
    mid = (on + light_len // 2) % 1440

    def fmt(a, b):
        return f'{a // 60:02d}:{a % 60:02d}-{b // 60:02d}:{b % 60:02d}'
    out = []
    if on + 30 <= 1440:
        out.append(fmt(on, on + 30))
    if mid >= 15 and mid + 15 <= 1440:
        out.append(fmt(mid - 15, mid + 15))
    if off >= 30:
        out.append(fmt(off - 30, off))
    return out


# ==== Cage geometry ====

def cage_geometry(experiment_dir):
    """(centroid, polygon (k, 2) array) from the experiment's buzzwatch_track_settings.yml."""
    path = os.path.join(experiment_dir, 'buzzwatch_track_settings.yml')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'no buzzwatch_track_settings.yml in {experiment_dir}')
    with open(path, 'r') as f:
        settings = yaml.safe_load(f) or {}
    pts = settings.get('cage_border_points')
    if not pts or len(pts) < 3:
        raise ValueError(f'no cage border drawn for {experiment_dir}')
    return m.calculate_cage_centroid(pts), np.asarray(pts, dtype=float)


def _area_shares(polygon, centroid, swarm_radius, n=400_000, seed=0):
    """Share of the cage polygon's area inside the swarm disc, the central disc, the core disc and
    each 20-px ring out to SHAPE_MAX_PX (uniform Monte Carlo, fixed seed)."""
    from matplotlib.path import Path
    rng = np.random.default_rng(seed)
    pts = np.empty((0, 2))
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    path = Path(polygon)
    while len(pts) < n:
        cand = rng.uniform(lo, hi, size=(n, 2))
        pts = np.vstack([pts, cand[path.contains_points(cand)]])
    r = np.hypot(pts[:n, 0] - centroid[0], pts[:n, 1] - centroid[1])
    rings = np.arange(0, SHAPE_MAX_PX + SHAPE_RING_PX, SHAPE_RING_PX)
    return {'swarm': float(np.mean(r < swarm_radius)), 'central': float(np.mean(r < CENTRAL_RADIUS_PX)),
            'core': float(np.mean(r < CORE_PX)),
            'ring_edges': rings, 'rings': np.histogram(r, rings)[0] / float(n)}


def experiment_geometry(experiment_dir, swarm_radius_px=SWARM_RADIUS_PX):
    centroid, polygon = cage_geometry(experiment_dir)
    half = np.ceil(np.abs(polygon - np.asarray(centroid, dtype=float)).max() / HEAT_CELL_PX) * HEAT_CELL_PX
    return {'centroid': centroid, 'polygon': polygon, 'swarm_radius': float(swarm_radius_px),
            'areas': _area_shares(polygon, centroid, float(swarm_radius_px)),
            'heat_edges': np.arange(-half, half + HEAT_CELL_PX, HEAT_CELL_PX)}


# ==== Per-day build (streamed, cached) ====

def _to_ns(timestamps):
    return np.array(timestamps, dtype='datetime64[us]').astype('datetime64[ns]').astype(np.int64)


def _extract_file(path, tunables):
    """Worker: load one tracking file and return every flight piece that passes the engine's
    per-flight filter steps (step split, average-speed bounds) with at least
    min(FLIGHT_MIN_FRAMES, MIN_FLIGHT_FRAMES) frames, as concatenated arrays. Pieces >=
    MIN_FLIGHT_FRAMES are exactly the engine's strict_candidates, in the same order."""
    for k, v in tunables:
        setattr(m, k, v)
    mt = m.load_tracking_file(path)
    if mt is None or not mt.time_stamp:
        return None
    ts = mt.time_stamp
    fps = None
    if len(ts) >= 2:
        dt = (ts[1] - ts[0]).total_seconds()
        if dt > 0:
            fps = 1.0 / dt
    span = (int(_to_ns(ts[:1])[0]), int(_to_ns(ts[-1:])[0]))
    raw = m.extract_raw_flights(mt)
    del mt
    min_frames = min(FLIGHT_MIN_FRAMES, m.MIN_FLIGHT_FRAMES)
    coords, times, lengths = [], [], []
    for fl in raw:
        zts = [None] * len(fl['coords'])
        if m.USE_STEP_SPLIT:
            pieces = m._zip_split_step(fl['coords'], fl['timestamps'], zts,
                                       m.STEP_SPLIT_MIN, m.STEP_SPLIT_MAX)
        else:
            pieces = [(fl['coords'], fl['timestamps'], zts)]
        for c, t, _z in pieces:
            if len(c) < min_frames:
                continue
            if not (m.SPEED_FILTER_MIN <= m.average_step_speed(c) <= m.SPEED_FILTER_MAX):
                continue
            coords.append(np.asarray(c, dtype=float))
            times.append(_to_ns(t))
            lengths.append(len(c))
    return {
        'fps': fps, 'span': span,
        'coords': np.concatenate(coords) if coords else np.empty((0, 2)),
        't_ns': np.concatenate(times) if times else np.empty(0, dtype=np.int64),
        'lengths': np.asarray(lengths, dtype=np.int64),
    }


def _extract_file_star(args):
    return _extract_file(*args)


def _exit_with_parent():
    """Pool initializer: end this worker as soon as the process that created the pool is gone
    (app closed or killed mid-run), instead of loading files as an orphan. Same as
    batch_runner._exit_with_parent, copied so workers don't import the tracker. Never raises."""
    try:
        import threading
        from multiprocessing import parent_process
        parent = parent_process()
        if parent is None:
            return

        def _watch():
            parent.join()
            os._exit(1)

        threading.Thread(target=_watch, name='exit-with-parent', daemon=True).start()
    except Exception:
        pass


def auto_workers(n_files, per_worker_gb=1.5, reserve_gb=3.0, cap=6):
    """Parallel file loaders that fit in the RAM free right now (each holds one ~100 MB tracking
    pickle, ~0.8 GB once unpickled). 1 if psutil is unavailable."""
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        return 1
    by_mem = int((avail_gb - reserve_gb) // per_worker_gb)
    return max(1, min(cap, (os.cpu_count() or 2) - 1, by_mem, n_files))


def _day_signature(files):
    newest = 0.0
    for p in files:
        try:
            newest = max(newest, os.path.getmtime(p))
        except OSError:
            pass
    return (CACHE_SCHEMA, m._session_cache_signature(), tuple(os.path.basename(p) for p in files),
            newest, SEGMENT_SECONDS, DAY_MARGIN.total_seconds(), FLIGHT_MIN_FRAMES)


def _offsets(lengths):
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def _subset(coords, t, offsets, piece_mask):
    """The pieces where piece_mask is True, as (coords, t, offsets)."""
    lengths = np.diff(offsets)
    point_mask = np.repeat(piece_mask, lengths)
    return coords[point_mask], t[point_mask], _offsets(lengths[piece_mask])


def drop_artifact_pieces(coords, offsets):
    """Vectorised m.drop_density_artifacts over concatenated pieces: returns a keep mask per piece.
    Same histogram (m._density_artifact_mask over all points) and same rule (drop a piece when
    >= DENSITY_DROP_FRAC of its points fall in artifact bins)."""
    n = len(offsets) - 1
    if n == 0:
        return np.zeros(0, dtype=bool)
    mask_func, n_artifact = m._density_artifact_mask(coords)
    if n_artifact == 0:
        return np.ones(n, dtype=bool)
    lengths = np.diff(offsets)
    pid = np.repeat(np.arange(n), lengths)
    frac = np.bincount(pid, weights=mask_func(coords), minlength=n) / lengths
    return frac < m.DENSITY_DROP_FRAC


def _day_population(coords, t, offsets, piece_mask, d0, d1):
    """Filter a population: candidate pieces -> density artifacts (over all candidates loaded for
    the day, margins included) -> pieces whose midpoint falls on the day."""
    c, tt, off = _subset(coords, t, offsets, piece_mask)
    keep = drop_artifact_pieces(c, off)
    mid = (tt[off[:-1]] + tt[off[1:] - 1]) // 2 if len(off) > 1 else np.empty(0, np.int64)
    c, tt, off = _subset(c, tt, off, keep & (mid >= d0) & (mid < d1))
    return {'coords': c, 't_ns': tt, 'offsets': off}


def load_or_build_day(experiment_dir, date_str, cache_dir, segments=None, workers=1,
                      progress=None, cancel=None):
    """Both flight populations for one calendar day, as compact arrays:
        tracks / flight: {'coords' (N, 2) float64, 't_ns' (N,) int64, 'offsets' (P+1,) int64}
        spans (F, 2) int64 first/last timestamp of every loaded file, fps, date, files.
    Cached to {cache_dir}/{date}.pkl; rebuilt when the day's files, their mtimes or the engine's
    filter tunables change. progress(done_files, total_files) is called from this thread; a set
    `cancel` Event raises Cancelled (nothing is cached)."""
    if segments is None:
        segments = list_segments(experiment_dir)
    files = day_files(segments, date_str)
    if not files:
        raise ValueError(f'no tracked segments on {date_str} in {experiment_dir}')
    sig = _day_signature(files)
    cache_path = os.path.join(cache_dir, f'{date_str}.pkl')
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get('signature') == sig:
                return cached
        except Exception:
            pass

    tunables = m._session_cache_signature()
    jobs = [(p, tunables) for p in files]
    results = []
    if progress:
        progress(0, len(files))

    def take(i, result):
        if result is not None:
            results.append(result)
        if progress:
            progress(i + 1, len(files))

    if workers and workers > 1:
        import multiprocessing as mp
        pool = mp.get_context('spawn').Pool(processes=min(workers, len(jobs)), maxtasksperchild=8,
                                            initializer=_exit_with_parent)
        try:
            for i, result in enumerate(pool.imap(_extract_file_star, jobs)):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                take(i, result)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
    else:
        for i, job in enumerate(jobs):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            take(i, _extract_file(*job))

    coords = np.concatenate([r['coords'] for r in results]) if results else np.empty((0, 2))
    t = np.concatenate([r['t_ns'] for r in results]) if results else np.empty(0, np.int64)
    lengths = np.concatenate([r['lengths'] for r in results]) if results else np.empty(0, np.int64)
    offsets = _offsets(lengths)
    fps_list = [r['fps'] for r in results if r['fps']]
    d0 = np.datetime64(datetime.strptime(date_str, '%Y-%m-%d'), 'ns').astype(np.int64)
    d1 = d0 + 86400 * _NS
    day = {
        'schema': CACHE_SCHEMA, 'signature': sig, 'experiment_dir': experiment_dir,
        'date': date_str, 'files': [os.path.basename(p) for p in files],
        'fps': float(np.median(fps_list)) if fps_list else 25.0,
        'spans': np.array([r['span'] for r in results], dtype=np.int64).reshape(-1, 2),
        'tracks': _day_population(coords, t, offsets, lengths >= m.MIN_FLIGHT_FRAMES, d0, d1),
        'flight': _day_population(coords, t, offsets, lengths >= FLIGHT_MIN_FRAMES, d0, d1),
    }
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cache_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(day, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)
    return day


# ==== Metrics ====

def _track_table(tracks, centroid):
    """Per-flight arrays for the >= 2 s tracks, sorted by flight midpoint."""
    coords, t, off = tracks['coords'], tracks['t_ns'], tracks['offsets']
    n_traj = len(off) - 1
    lengths = np.diff(off)
    seg = np.repeat(np.arange(n_traj), lengths)
    dists = np.hypot(coords[:, 0] - centroid[0], coords[:, 1] - centroid[1])
    first, last = off[:-1], off[1:] - 1
    step = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    step[seg[1:] != seg[:-1]] = 0.0
    path = np.bincount(seg[1:], weights=step, minlength=n_traj) if len(step) else np.zeros(n_traj)
    net = np.hypot(coords[last, 0] - coords[first, 0], coords[last, 1] - coords[first, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        straight = np.where(path > 0, net / path, np.nan)
    mid = (t[first] + t[last]) // 2
    order = np.argsort(mid, kind='stable')
    dist_list = np.split(dists, off[1:-1]) if n_traj else []
    return {
        'mid': mid[order],
        'dists': [dist_list[i] for i in order],
        'prop100': (np.bincount(seg, weights=(dists <= 100), minlength=n_traj) / lengths)[order],
        'prop150': (np.bincount(seg, weights=(dists <= 150), minlength=n_traj) / lengths)[order],
        'straight': straight[order],
    }


def smoothed_motion(flight):
    """Per point: smoothed speed (px/frame, displacement i-2 -> i+2 / 4) and turning (deg/frame,
    between the 2-frame displacements before and after the point, only where both are >= 2 x
    FLY_PX_FRAME). NaN where the piece is too short around the point."""
    c, off = flight['coords'], flight['offsets']
    pid = np.repeat(np.arange(len(off) - 1), np.diff(off))
    n = len(pid)
    speed = np.full(n, np.nan)
    turn = np.full(n, np.nan)
    if n > 4:
        ok = pid[4:] == pid[:-4]
        speed[2:-2][ok] = np.hypot(c[4:, 0] - c[:-4, 0], c[4:, 1] - c[:-4, 1])[ok] / 4.0
        ax_, ay_ = c[2:-2, 0] - c[:-4, 0], c[2:-2, 1] - c[:-4, 1]
        bx_, by_ = c[4:, 0] - c[2:-2, 0], c[4:, 1] - c[2:-2, 1]
        ok &= (np.hypot(ax_, ay_) >= 2 * FLY_PX_FRAME) & (np.hypot(bx_, by_) >= 2 * FLY_PX_FRAME)
        ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(by_, bx_) - np.arctan2(ay_, ax_))))))
        turn[2:-2][ok] = ang[ok] / 2.0
    return speed, turn, pid


def _flight_table(flight, fps, d0, geo):
    """Real-flight points sorted by time, plus per-frame counts for the whole day: all real
    flyers and those inside the swarm zone, and which frames were recorded at all."""
    speed, turn, pid = smoothed_motion(flight)
    real = np.nan_to_num(speed, nan=0.0) >= FLY_PX_FRAME
    c, t = flight['coords'][real], flight['t_ns'][real]
    frame_ns = _NS / float(fps)
    f = np.rint((t - d0) / frame_ns).astype(np.int64)
    centroid = geo['centroid']
    r = np.hypot(c[:, 0] - centroid[0], c[:, 1] - centroid[1])
    o = np.argsort(t, kind='stable')
    n_frames = int(round(86400 * fps))
    in_day = (f >= 0) & (f < n_frames)
    counts_all = np.bincount(f[in_day], minlength=n_frames)[:n_frames]
    zone = in_day & (r < geo['swarm_radius'])
    counts_zone = np.bincount(f[zone], minlength=n_frames)[:n_frames]
    return {'t': t[o], 'f': f[o], 'x': c[o, 0], 'y': c[o, 1], 'r': r[o], 'pid': pid[real][o],
            'speed': speed[real][o], 'turn': turn[real][o],
            'counts_all': counts_all, 'counts_zone': counts_zone, 'n_frames': n_frames}


def _recorded_frames(spans, d0, fps, n_frames):
    """Boolean per frame of the day: covered by a loaded tracking file."""
    rec = np.zeros(n_frames, dtype=bool)
    frame_ns = _NS / float(fps)
    for a, b in spans:
        fa = max(0, int(np.floor((a - d0) / frame_ns)))
        fb = min(n_frames, int(np.ceil((b - d0) / frame_ns)) + 1)
        if fb > fa:
            rec[fa:fb] = True
    return rec


def _pair_metrics(ft, a, b, f_lo, f_hi, shifts):
    """nn_px and near_ratio for the real-flight points ft[a:b] (frames f_lo..f_hi)."""
    out = {'nn_px': np.nan, 'near_ratio': np.nan, 'n_pairs': 0}
    if b - a < 2:
        return out
    cols = ('f', 'x', 'y', 'pid')
    A = pd.DataFrame({k: ft[k][a:b] for k in cols})
    A['i'] = np.arange(b - a)
    obs = A.merge(A, on='f', suffixes=('a', 'b'))
    obs = obs[obs.ia.values < obs.ib.values]
    out['n_pairs'] = len(obs)
    if len(obs) < MIN_PAIRS:
        return out
    d = np.hypot(obs.xa.values - obs.xb.values, obs.ya.values - obs.yb.values)
    nn = np.full(b - a, np.inf)
    np.minimum.at(nn, obs.ia.values, d)
    np.minimum.at(nn, obs.ib.values, d)
    out['nn_px'] = float(np.mean(nn[np.isfinite(nn)]))

    def in_band(dist):
        return int(np.sum((dist >= NEAR_BAND_PX[0]) & (dist < NEAR_BAND_PX[1])))

    o_tot, o_band = len(d), in_band(d)
    c_tot = c_band = 0
    for s in shifts:
        sa, sb = np.searchsorted(ft['f'], [f_lo + s, f_hi + s])
        if sb <= sa:
            continue
        B = pd.DataFrame({k: ft[k][sa:sb] for k in cols})
        B['f'] = B['f'] - s
        ctl = A.merge(B, on='f', suffixes=('a', 'b'))
        ctl = ctl[ctl.pida.values != ctl.pidb.values]
        dc = np.hypot(ctl.xa.values - ctl.xb.values, ctl.ya.values - ctl.yb.values)
        c_tot += len(dc)
        c_band += in_band(dc)
    if c_tot and c_band:
        out['near_ratio'] = (o_band / float(o_tot)) / (c_band / float(c_tot))
    return out


def _swarm_counts(ft, rec, f0, f1):
    """Swarm-zone counts for the recorded frames in [f0, f1)."""
    f0, f1 = max(0, f0), min(ft['n_frames'], f1)
    sel = rec[f0:f1]
    return ft['counts_all'][f0:f1][sel], ft['counts_zone'][f0:f1][sel]


def _ring_density(r, areas):
    """Real-flight density per unit cage area in each SHAPE_RING_PX ring (1 = spread evenly)."""
    rings = np.histogram(r, areas['ring_edges'])[0] / float(len(r))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(areas['rings'] > 1e-4, rings / areas['rings'], np.nan)


def _window_maps(ft, geo, lo, hi, n_rec):
    """Centre profile (density per ring, as in shape_index) and heatmap cells (real flight per
    HEAT_CELL_PX cell, centred on the cage centroid, image orientation) for [lo, hi)."""
    pa, pb = np.searchsorted(ft['t'], [lo, hi])
    if pb - pa < MIN_POINTS or not n_rec:
        return [], []
    edges = geo['areas']['ring_edges']
    dens = _ring_density(ft['r'][pa:pb], geo['areas'])
    profile = [{'r_lo_px': float(edges[i]), 'r_hi_px': float(edges[i + 1]), 'density': float(dens[i])}
               for i in range(len(dens))]
    he = geo['heat_edges']
    cx, cy = geo['centroid']
    H = np.histogram2d(ft['x'][pa:pb] - cx, ft['y'][pa:pb] - cy, [he, he])[0]
    ix, iy = np.nonzero(H)
    half, n = HEAT_CELL_PX / 2.0, float(pb - pa)
    heat = [{'x_px': float(he[i] + half), 'y_px': float(he[j] + half),
             'share': H[i, j] / n, 'per_frame': H[i, j] / float(n_rec)} for i, j in zip(ix, iy)]
    return profile, heat


def _interval_metrics(tr, ft, rec, geo, day, lo, hi, d0):
    fps = day['fps']
    frame_ns = _NS / float(fps)
    f_lo, f_hi = int(round((lo - d0) / frame_ns)), int(round((hi - d0) / frame_ns))
    all_c, zone_c = _swarm_counts(ft, rec, f_lo, f_hi)
    n_rec = len(all_c)
    out = {'recorded_frac': n_rec / float(max(1, f_hi - f_lo))}
    # ---- real flight: numbers and swarm size ----
    if n_rec:
        out['n_flying'] = float(all_c.mean())
        out['swarm_mean'] = float(zone_c.mean())
        out['swarm_ge3_frac'] = float(np.mean(zone_c >= 3))
        out['swarm_ge5_frac'] = float(np.mean(zone_c >= 5))
        out['swarm_p95'] = float(np.percentile(zone_c, 95))
    else:
        for k in ('n_flying', 'swarm_mean', 'swarm_ge3_frac', 'swarm_ge5_frac', 'swarm_p95'):
            out[k] = np.nan
    # ---- real flight: where and how ----
    pa, pb = np.searchsorted(ft['t'], [lo, hi])
    areas = geo['areas']
    r = ft['r'][pa:pb]
    if pb - pa >= MIN_POINTS:
        out['central_enrichment'] = float(np.mean(r < CENTRAL_RADIUS_PX)) / areas['central']
        core = float(np.mean(r < CORE_PX)) / areas['core']
        ring_density = _ring_density(r, areas)
        out['core_density'] = core
        out['shape_index'] = core / float(np.nanmax(ring_density)) if np.any(ring_density > 0) else np.nan
    else:
        out['central_enrichment'] = out['core_density'] = out['shape_index'] = np.nan
    sp = ft['speed'][pa:pb]
    out['speed_px_s'] = float(np.median(sp)) * fps if len(sp) >= MIN_SAMPLES else np.nan
    tn = ft['turn'][pa:pb]
    tn = tn[np.isfinite(tn)]
    out['turning_deg_s'] = float(np.median(tn)) * fps if len(tn) >= MIN_SAMPLES else np.nan
    shifts = [int(round(s * fps)) for s in SHIFTS_S]
    out.update(_pair_metrics(ft, pa, pb, f_lo, f_hi, shifts))
    # ---- tracks (>= 2 s) ----
    a, b = np.searchsorted(tr['mid'], [lo, hi])
    n = int(b - a)
    out['n_traj'] = n
    out['activity'] = n / (n_rec / fps / 60.0) if n_rec else np.nan
    if n >= m.MIN_TRAJECTORIES_PER_BIN:
        out['r50'] = m.weighted_r50(tr['dists'][a:b])
        out['prop_within_100'] = float(np.mean(tr['prop100'][a:b]))
        out['prop_within_150'] = float(np.mean(tr['prop150'][a:b]))
        out['straightness'] = float(np.nanmedian(tr['straight'][a:b]))
    else:
        for k in ('r50', 'prop_within_100', 'prop_within_150', 'straightness'):
            out[k] = np.nan
    return out, zone_c


def day_metrics(day, geo, bin_minutes=10, windows=()):
    """One day -> {'bins', 'windows', 'swarm_size', 'minutes', 'profile', 'heatmap'}: lists of
    row dicts. swarm_size: per window, the share of recorded frames with exactly k
    (0..SWARM_MAX_COUNT) flyers in the swarm zone. minutes: per clock minute, real flyers and
    swarm size (mean, max) -- a light series for plotting swarm size through the day. profile /
    heatmap: per window, where the real flight is (see _window_maps)."""
    d0 = np.datetime64(datetime.strptime(day['date'], '%Y-%m-%d'), 'ns').astype(np.int64)
    fps = day['fps']
    tr = _track_table(day['tracks'], geo['centroid'])
    ft = _flight_table(day['flight'], fps, d0, geo)
    rec = _recorded_frames(day['spans'], d0, fps, ft['n_frames'])
    bin_ns = int(bin_minutes) * 60 * _NS
    n_bins = int(np.ceil(1440 / float(bin_minutes)))
    bins = []
    for b in range(n_bins):
        lo = d0 + b * bin_ns
        hi = min(lo + bin_ns, d0 + 86400 * _NS)
        row = {'bin_start': pd.Timestamp(lo), 'tod_h': b * bin_minutes / 60.0}
        row.update(_interval_metrics(tr, ft, rec, geo, day, lo, hi, d0)[0])
        bins.append(row)
    wins, sizes, profiles, heat = [], [], [], []
    for text in windows:
        label, lo_min, hi_min = parse_window(text)
        lo, hi = d0 + lo_min * 60 * _NS, d0 + hi_min * 60 * _NS
        metrics, zone_c = _interval_metrics(tr, ft, rec, geo, day, lo, hi, d0)
        wins.append(dict({'window': label}, **metrics))
        if len(zone_c):
            dist = np.bincount(np.minimum(zone_c, SWARM_MAX_COUNT), minlength=SWARM_MAX_COUNT + 1)
            share = dist / float(len(zone_c))
            sizes.extend({'window': label, 'swarm_size': k, 'share_of_frames': float(share[k])}
                         for k in range(SWARM_MAX_COUNT + 1))
        prof, cells = _window_maps(ft, geo, lo, hi, len(zone_c))
        profiles.extend(dict({'window': label}, **p) for p in prof)
        heat.extend(dict({'window': label}, **c) for c in cells)
    minutes = []
    for mi in range(1440):
        f0, f1 = int(round(mi * 60 * fps)), int(round((mi + 1) * 60 * fps))
        all_c, zone_c = _swarm_counts(ft, rec, f0, f1)
        if not len(all_c):
            continue
        minutes.append({'minute_start': pd.Timestamp(d0 + mi * 60 * _NS), 'tod_h': mi / 60.0,
                        'n_flying': float(all_c.mean()), 'swarm_mean': float(zone_c.mean()),
                        'swarm_max': int(zone_c.max()), 'swarm_ge3_frac': float(np.mean(zone_c >= 3))})
    return {'bins': bins, 'windows': wins, 'swarm_size': sizes, 'minutes': minutes,
            'profile': profiles, 'heatmap': heat}


# ==== Orchestration ====

def experiment_output_dir(experiment_dir):
    return os.path.join(experiment_dir, 'plots', 'buzzswarm')


def default_alias(experiment_dir):
    """'{Location}_{Experiment}' -- the experiment JSON's alias convention. The folder name alone
    is not unique (the same batch name is reused across locations/cages)."""
    folder = os.path.normpath(experiment_dir)
    return f'{os.path.basename(os.path.dirname(folder))}_{os.path.basename(folder)}'


def run(experiments, bin_minutes=10, windows=None, cache_root=None, out_root=None, workers=None,
        swarm_radius_px=SWARM_RADIUS_PX, progress=None, cancel=None):
    """Compute clock-time metrics for several experiments.

    experiments: list of dicts {'folder', 'group', 'days': ['YYYY-MM-DD', ...],
                                'tags': {date: 'LD'|'DD'} (optional), 'alias' (optional)}.
    Per experiment, writes bins_{N}min.csv, windows.csv, swarm_size.csv, swarm_minutes.csv,
    profile.csv, heatmap.csv and geometry.json to {out_root}/{alias}/ if out_root is given, else
    to the experiment's own plots/buzzswarm/ (load_outputs reads them back). Day
    caches go to {cache_root}/{alias}/ or plots/buzzswarm/day_cache/. swarm_radius_px sets the
    swarm zone (default SWARM_RADIUS_PX; metrics only, caches are unaffected). workers=None picks a
    memory-safe count per day. progress(fraction 0-1, message) is called from the calling
    thread; a set `cancel` Event raises Cancelled.
    Returns a dict of DataFrames for all experiments together: bins, windows, swarm_size, minutes,
    profile, heatmap; plus 'geometry': {alias: geometry.json payload} (same shape as load_outputs)."""
    if windows is None:
        windows = default_windows()
    units = [(e, d) for e in experiments for d in e['days']]
    if not units:
        raise ValueError('no days selected')
    tables = {k: [] for k in output_names(bin_minutes)}
    geometry = {}
    done = 0
    for exp in experiments:
        folder = exp['folder']
        alias = exp.get('alias') or default_alias(folder)
        group = exp.get('group') or alias
        tags = exp.get('tags') or {}
        geo = experiment_geometry(folder, swarm_radius_px)
        segments = list_segments(folder)
        cache_dir = (os.path.join(cache_root, alias) if cache_root
                     else os.path.join(experiment_output_dir(folder), 'day_cache'))
        exp_rows = {k: [] for k in tables}
        for date_str in exp['days']:
            if cancel is not None and cancel.is_set():
                raise Cancelled()

            def file_progress(i, n, _done=done, _alias=alias, _date=date_str):
                if progress:
                    progress((_done + (i / float(n) if n else 1.0)) / len(units),
                             f'{_alias} | {_date} | segment {i}/{n}')
            files = day_files(segments, date_str)
            n_workers = workers if workers else auto_workers(len(files))
            day = load_or_build_day(folder, date_str, cache_dir, segments=segments,
                                    workers=n_workers, progress=file_progress, cancel=cancel)
            if progress:
                progress((done + 1.0) / len(units), f'{alias} | {date_str} | computing metrics')
            ident = {'experiment': alias, 'group': group, 'date': date_str,
                     'ld_dd': tags.get(date_str, 'LD'), 'swarm_radius_px': geo['swarm_radius']}
            for key, rows in day_metrics(day, geo, bin_minutes, windows).items():
                exp_rows[key].extend(dict(ident, **r) for r in rows)
            del day
            done += 1
        out_dir = os.path.join(out_root, alias) if out_root else experiment_output_dir(folder)
        os.makedirs(out_dir, exist_ok=True)
        names = output_names(bin_minutes)
        for key, rows in exp_rows.items():
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out_dir, names[key]), index=False)
            tables[key].append(df)
        geometry[alias] = _write_geometry(out_dir, alias, group, geo)
    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in tables.items()}
    out['geometry'] = geometry
    return out


def output_names(bin_minutes):
    return {'bins': f'bins_{int(bin_minutes)}min.csv', 'windows': 'windows.csv',
            'swarm_size': 'swarm_size.csv', 'minutes': 'swarm_minutes.csv',
            'profile': 'profile.csv', 'heatmap': 'heatmap.csv'}


def _write_geometry(out_dir, alias, group, geo):
    """Cage outline and zone radii (pixels), so plots can draw them without the experiment."""
    cx, cy = (float(v) for v in geo['centroid'])
    payload = {'experiment': alias, 'group': group, 'centroid': [cx, cy],
               'polygon': [[float(x), float(y)] for x, y in geo['polygon']],
               'swarm_radius_px': geo['swarm_radius'], 'central_radius_px': CENTRAL_RADIUS_PX,
               'core_px': CORE_PX}
    with open(os.path.join(out_dir, 'geometry.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    return payload


def load_outputs(out_dirs, bin_minutes):
    """Read run()'s CSVs back from several output folders -> the same dict run() returns, plus
    'geometry': {experiment alias: geometry.json dict}. Missing files are skipped."""
    names = output_names(bin_minutes)
    tables = {k: [] for k in names}
    geometry = {}
    for d in out_dirs:
        for key, name in names.items():
            path = os.path.join(d, name)
            if os.path.isfile(path) and os.path.getsize(path) > 2:
                try:
                    tables[key].append(pd.read_csv(path))
                except pd.errors.EmptyDataError:
                    pass
        gpath = os.path.join(d, 'geometry.json')
        if os.path.isfile(gpath):
            with open(gpath, 'r') as f:
                g = json.load(f)
            geometry[g.get('experiment', os.path.basename(d))] = g
    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in tables.items()}
    for key, col in (('bins', 'bin_start'), ('minutes', 'minute_start')):
        if col in out[key]:
            out[key][col] = pd.to_datetime(out[key][col])
    for df in out.values():
        if 'date' in df:
            df['date'] = df['date'].astype(str)
    out['geometry'] = geometry
    return out
