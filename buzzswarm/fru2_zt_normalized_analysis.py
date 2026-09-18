#!/usr/bin/env python3
"""
ZT-normalized r50 / Sholl analysis engine for BuzzSwarm.

Core BuzzSwarm engine: loads per-experiment tracking (process_session_from_dir /
load_or_build_session), strict-filters flights, normalizes each session to ZT0-ZT12
(05:00-17:00), and provides the r50 / Sholl-CDF metrics + a parameterized plotting API
(plot_all_day_r50, run_sholl_analysis, run_heatmaps, run_trajectories, run_prop_near_centre,
run_filter_validation, ...), each taking an explicit output directory.

Used by the GUI (buzzswarm_tab_manager) and CLI (buzzsuite_cli) via the run(experiment_dir, ...)
wrappers in fru2_sholl_csv / fru2_pooled_analysis / fru2_per_trajectory_plots. Tracking numerics
and the .pkl format are frozen — do not change the math here.
"""

import math
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Make BuzzSuite's local buzzwatch_data_analysis package importable. Prefers the full
# package shipped with BuzzSuite; the bundled _compat_shim (legacy .pkl unpickling stub)
# is appended after it so it only takes effect if the full package can't be found.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUZZSUITE_ROOT = os.path.dirname(_HERE)
_COMPAT_SHIM = os.path.join(_HERE, '_compat_shim')
if _BUZZSUITE_ROOT not in sys.path:
    sys.path.insert(0, _BUZZSUITE_ROOT)
if _COMPAT_SHIM not in sys.path:
    sys.path.append(_COMPAT_SHIM)

from buzzwatch_data_analysis.flight_center_distance_analysis import calculate_cage_centroid

# ==== Constants ====
ZT0_HOUR = 5    # ZT0 = 05:00
ZT12_HOUR = 17  # ZT12 = 17:00
BIN_MINUTES = 10

# The recording pipeline wrote videos slowed down:
# 32 recorded minutes = 20 REAL minutes of biological time  ->  factor 32/20 = 1.6.
# All time-window widths (bins, last/middle 30-min windows) are multiplied by this
# factor when interpreting recorded timestamps, so each "30 min" in the analysis
# = 30 REAL minutes = 48 recorded minutes.
# Each session is rescaled so its recorded duration maps to exactly 12 real hours.
# session_speedup(sd) = recorded_duration_seconds / (12 * 3600).
# All time-window widths multiply by this per-session factor when interpreting
# recorded timestamps, so each "30 real min" in the analysis  = 30 / 12 = 1/24 of
# the session's recorded duration.

def session_speedup(session_data):
    total_s = (session_data['last_ts'] - session_data['first_ts']).total_seconds()
    return total_s / (12.0 * 3600.0) if total_s > 0 else 1.0

SHOLL_WINDOWS = [(5.75, 6.25), (11.5, 12.0)]
HEATMAP_WINDOW = (11.5, 12.0)
TRAJECTORY_WINDOW = (11.5, 12.0)
HEATMAP_BINS = 1000
HEATMAP_DPI = 300

SPEED_FILTER_MIN = 1.0       # px/frame  (avg-step lower bound: removes stuck blotches)
SPEED_FILTER_MAX = 40.0      # px/frame  (avg-step upper bound)
TELEPORT_MAX_PX = 100.0      # px/frame  (per-step safety net: split only on clear teleports)
STEP_SPLIT_MAX = 40.0        # px/frame  (per-step split: original strict)
STEP_SPLIT_MIN = 0.0         # px/frame  (per-step lower bound)
USE_STEP_SPLIT = True        # toggle the per-step split filter
MEDIAN_SMOOTH_WINDOW = 1     # 1 = effectively disabled (returns coords as-is)
# Density-outlier artifact filter (per session, applied to all candidate trajectories):
# Build a 2D histogram of all flight points with fine pixel bins. Stuck-tracker
# pixels show as bins with extreme density (same coord repeated thousands of times),
# while real flight (including male centre hovering) spreads over a much larger
# area. Bins whose count exceeds DENSITY_OUTLIER_FACTOR x median non-empty count
# are flagged as artifact bins; trajectories with at least DENSITY_DROP_FRAC of
# their points in artifact bins are dropped.
DENSITY_BIN_PX = 5
DENSITY_OUTLIER_FACTOR = 100.0
DENSITY_DROP_FRAC = 0.50
MIN_FLIGHT_FRAMES = 50
MAX_DISPLACEMENT_PX = 150

PROP_NEAR_RADIUS_PX = 150
SHOLL_RADIUS_STEP_PX = 5

# Minimum number of trajectories required in a bin for that bin to be plotted.
# Bins below this are returned as NaN (line skips), which removes single-trajectory
# spikes that visually look like "blobs" in the all-day plots.
MIN_TRAJECTORIES_PER_BIN = 5

SEX_COLORS = {
    'Female': '#DC3977',
    'FruM':   '#F85525',
    'Male':   '#0070FF',
}
SEX_ORDER = ['Female', 'Male', 'FruM']  # matches sample plot ordering (♀, ♂, fruM♂)

# ==== Low-level utilities ====

def valid_point(p):
    return p is not None and len(p) == 2 and not (np.isnan(p[0]) or np.isnan(p[1]))


def distance_to_center(pt, centroid):
    return math.hypot(pt[0] - centroid[0], pt[1] - centroid[1])


def average_step_speed(coords):
    if len(coords) < 2:
        return 0.0
    dists = [math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
             for i in range(1, len(coords))]
    return float(np.mean(dists)) if dists else 0.0


def step_speeds(coords):
    if len(coords) < 2:
        return np.array([])
    return np.array([math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
                     for i in range(1, len(coords))])


def split_on_large_displacement(coords, max_disp):
    segs, cur = [], []
    for p in coords:
        if not valid_point(p):
            if cur:
                segs.append(cur); cur = []
            continue
        if not cur:
            cur = [p]; continue
        if math.hypot(p[0] - cur[-1][0], p[1] - cur[-1][1]) > max_disp:
            segs.append(cur); cur = [p]
        else:
            cur.append(p)
    if cur:
        segs.append(cur)
    return segs


def split_on_step_speed(coords, smin, smax):
    if len(coords) < 2:
        return [coords] if coords else []
    segs, cur = [], [coords[0]]
    for i in range(1, len(coords)):
        v = math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
        if smin <= v <= smax:
            cur.append(coords[i])
        else:
            if len(cur) > 1:
                segs.append(cur)
            cur = [coords[i]]
    if len(cur) > 1:
        segs.append(cur)
    return segs


# ==== Loading ====

def load_tracking_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except (ModuleNotFoundError, ImportError, NameError):
        class CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)
        try:
            with open(filepath, 'rb') as f:
                return CompatUnpickler(f).load()
        except Exception as e:
            print(f'  ! load failed {filepath}: {e}')
            return None
    except Exception as e:
        print(f'  ! load failed {filepath}: {e}')
        return None


# ==== ZT time utilities ====
# Normalization: each session is linearly mapped onto ZT0-ZT12.
# The first timestamp of the session -> ZT 0, the last timestamp -> ZT 12.
# Per-point ZT = 12 * (ts - first_ts) / (last_ts - first_ts).

def zt_from_timestamp(ts, first_ts, last_ts):
    total = (last_ts - first_ts).total_seconds()
    if total <= 0:
        return 0.0
    return 12.0 * (ts - first_ts).total_seconds() / total


# ==== Extraction: flight coordinates per session ====

def extract_raw_flights(mt):
    """Extract all 'flying' runs from tracking object, with per-frame timestamps.
    Returns list of dicts: {fly_id, coords, timestamps, start_frame_abs}.
    Applies only: valid_point masking, large-displacement split. NO speed filter."""
    timestamps = mt.time_stamp
    if not timestamps or len(timestamps) < 2:
        return []

    flights = []
    for fly_id, obj in mt.objects.items():
        coords = obj.get('coordinates', []) or []
        state = obj.get('state', []) or []
        if not coords or not state:
            continue
        abs_start = int(obj.get('start', 0))

        i = 0
        n = min(len(coords), len(state))
        while i < n:
            if state[i] != 1:
                i += 1; continue
            j = i
            while j < n and state[j] == 1:
                j += 1
            run_coords = coords[i:j]
            run_ts = []
            for k in range(i, j):
                abs_idx = abs_start + k
                if 0 <= abs_idx < len(timestamps):
                    run_ts.append(timestamps[abs_idx])
                else:
                    run_ts.append(None)

            # split on large displacement (same mask applied to timestamps)
            cur_c, cur_t = [], []
            for pt, ts in zip(run_coords, run_ts):
                if not valid_point(pt) or ts is None:
                    if len(cur_c) >= 2:
                        flights.append({
                            'fly_id': fly_id, 'coords': cur_c, 'timestamps': cur_t,
                            'start_frame_abs': abs_start + i,
                        })
                    cur_c, cur_t = [], []
                    continue
                if cur_c:
                    prev = cur_c[-1]
                    if math.hypot(pt[0] - prev[0], pt[1] - prev[1]) > MAX_DISPLACEMENT_PX:
                        if len(cur_c) >= 2:
                            flights.append({
                                'fly_id': fly_id, 'coords': cur_c, 'timestamps': cur_t,
                                'start_frame_abs': abs_start + i,
                            })
                        cur_c, cur_t = [], []
                cur_c.append(pt); cur_t.append(ts)
            if len(cur_c) >= 2:
                flights.append({
                    'fly_id': fly_id, 'coords': cur_c, 'timestamps': cur_t,
                    'start_frame_abs': abs_start + i,
                })
            i = j
    return flights


def rolling_median_smooth(coords, window=MEDIAN_SMOOTH_WINDOW):
    """Smooth each (x, y) with a rolling median over `window` frames (centered).
    This suppresses high-frequency zigzag jitter without dropping any frames or
    splitting trajectories. Real fast flight passes through nearly unchanged
    because median over a small window of consistent motion stays consistent."""
    if window < 2 or len(coords) < window:
        return [tuple(p) for p in coords]
    arr = np.asarray(coords, dtype=float)  # (n, 2)
    half = window // 2
    n = len(arr)
    out = np.empty_like(arr)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        out[i, 0] = float(np.median(arr[a:b, 0]))
        out[i, 1] = float(np.median(arr[a:b, 1]))
    return [tuple(p) for p in out]


def split_on_step_speed_with_meta(coords, timestamps, zt_values, smin, smax):
    """Split a single trajectory at any per-step speed outside [smin, smax]."""
    if len(coords) < 2:
        return [(coords, timestamps, zt_values)] if coords else []
    segs = []
    cur_c = [coords[0]]; cur_t = [timestamps[0]]; cur_z = [zt_values[0]]
    for i in range(1, len(coords)):
        v = math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
        if smin <= v <= smax:
            cur_c.append(coords[i]); cur_t.append(timestamps[i]); cur_z.append(zt_values[i])
        else:
            if len(cur_c) > 1:
                segs.append((cur_c, cur_t, cur_z))
            cur_c = [coords[i]]; cur_t = [timestamps[i]]; cur_z = [zt_values[i]]
    if len(cur_c) > 1:
        segs.append((cur_c, cur_t, cur_z))
    return segs


def _density_artifact_mask(all_coords, bin_px=DENSITY_BIN_PX,
                            factor=DENSITY_OUTLIER_FACTOR):
    """Return (mask_function, n_artifact_bins).
    Builds a 2D histogram of all_coords with `bin_px` bins; flags any bin whose
    count exceeds `factor * median(non-empty bin counts)` as an artifact bin.
    The returned mask_function takes coords (np.array N,2) and returns a boolean
    array (length N) marking which points fall in artifact bins.
    Returns (lambda c: zeros, 0) if there are too few points to compute."""
    pts = np.asarray(all_coords, dtype=float)
    if len(pts) < 100:
        return (lambda c: np.zeros(len(c), dtype=bool)), 0
    xmin, xmax = float(pts[:, 0].min()) - 1, float(pts[:, 0].max()) + 1
    ymin, ymax = float(pts[:, 1].min()) - 1, float(pts[:, 1].max()) + 1
    nx = max(1, int(math.ceil((xmax - xmin) / bin_px)))
    ny = max(1, int(math.ceil((ymax - ymin) / bin_px)))
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nx, ny],
                               range=[[xmin, xmax], [ymin, ymax]])
    nonzero = H[H > 0]
    if len(nonzero) == 0:
        return (lambda c: np.zeros(len(c), dtype=bool)), 0
    median_count = float(np.median(nonzero))
    threshold = factor * median_count
    artifact_bins = H >= threshold
    n_artifact = int(np.sum(artifact_bins))

    bw_x = (xmax - xmin) / nx
    bw_y = (ymax - ymin) / ny

    def mask_func(coords):
        c = np.asarray(coords, dtype=float)
        ix = np.clip(((c[:, 0] - xmin) / bw_x).astype(int), 0, nx - 1)
        iy = np.clip(((c[:, 1] - ymin) / bw_y).astype(int), 0, ny - 1)
        return artifact_bins[ix, iy]

    return mask_func, n_artifact


def strict_filter_flights(raw_flights):
    """Density-outlier artifact filter:
    1. MIN_FLIGHT_FRAMES + average-step-speed in [SPEED_FILTER_MIN, SPEED_FILTER_MAX].
    2. Pool all candidate flight points; build a 2D histogram with DENSITY_BIN_PX bins.
    3. Mark bins with count >= DENSITY_OUTLIER_FACTOR * median(non-empty bin count) as
       artifact bins (stuck-tracker pixels).
    4. Drop trajectories with >= DENSITY_DROP_FRAC of their points in artifact bins.
    Real flight - including male centre hovering - spreads over many bins so its
    per-bin density is moderate; only stuck-pixel patterns trigger the threshold."""
    candidates = []
    for fl in raw_flights:
        zts = fl.get('zt_values', [None] * len(fl['coords']))
        if USE_STEP_SPLIT:
            segments = _zip_split_step(fl['coords'], fl['timestamps'], zts,
                                        STEP_SPLIT_MIN, STEP_SPLIT_MAX)
        else:
            segments = [(fl['coords'], fl['timestamps'], zts)]
        for seg_c, seg_t, seg_z in segments:
            if len(seg_c) < MIN_FLIGHT_FRAMES:
                continue
            if not (SPEED_FILTER_MIN <= average_step_speed(seg_c) <= SPEED_FILTER_MAX):
                continue
            candidates.append({
                'fly_id': fl['fly_id'],
                'coords': seg_c,
                'timestamps': seg_t,
                'zt_values': seg_z,
                'start_frame_abs': fl['start_frame_abs'],
            })

    if not candidates:
        return []

    all_pts = [pt for c in candidates for pt in c['coords']]
    mask_func, n_artifact = _density_artifact_mask(all_pts)
    if n_artifact == 0:
        return candidates

    out = []
    for c in candidates:
        in_artifact = mask_func(c['coords'])
        if float(np.mean(in_artifact)) >= DENSITY_DROP_FRAC:
            continue
        out.append(c)
    return out


def _zip_split_step(coords, timestamps, zt_values, smin, smax):
    """Same logic as split_on_step_speed but carries timestamps and zt_values alongside coords."""
    if len(coords) < 2:
        return [(coords, timestamps, zt_values)] if coords else []
    segs = []
    cur_c = [coords[0]]; cur_t = [timestamps[0]]; cur_z = [zt_values[0]]
    for i in range(1, len(coords)):
        v = math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
        if smin <= v <= smax:
            cur_c.append(coords[i]); cur_t.append(timestamps[i]); cur_z.append(zt_values[i])
        else:
            if len(cur_c) > 1:
                segs.append((cur_c, cur_t, cur_z))
            cur_c = [coords[i]]; cur_t = [timestamps[i]]; cur_z = [zt_values[i]]
    if len(cur_c) > 1:
        segs.append((cur_c, cur_t, cur_z))
    return segs


# ==== Higher-level: load session, return filtered flights + cage centroid ====

def _build_session_from_files(files, centroid, sex, strain, batch, verbose=True):
    """Shared session loader: given a list of tracking files + a cage centroid, load and
    strict-filter the flights and return the standard session dict. This is the single
    implementation of the per-session loading numerics, used by process_session_from_dir
    (the GUI/CLI path that takes an explicit experiment folder). Do not change the math here."""
    raw_flights = []
    fps = 25.0
    first_ts = None
    last_ts = None

    for fp in files:
        mt = load_tracking_file(fp)
        if mt is None:
            continue
        ts = mt.time_stamp
        if not ts:
            continue
        if ts and len(ts) >= 2:
            dt = (ts[1] - ts[0]).total_seconds()
            if dt > 0:
                fps = 1.0 / dt
        if first_ts is None or ts[0] < first_ts:
            first_ts = ts[0]
        if last_ts is None or ts[-1] > last_ts:
            last_ts = ts[-1]
        file_flights = extract_raw_flights(mt)
        raw_flights.extend(file_flights)
        if verbose:
            print(f'    {os.path.basename(fp)}: {len(file_flights)} raw flights')
        # Process and discard one tracking .pkl at a time — never hold all of them in memory.
        del mt

    if first_ts is None or last_ts is None or first_ts >= last_ts:
        print(f'  ! invalid time bounds for {sex}/{batch}')
        return None

    # Tag every point with its session-normalized ZT value (0-12)
    for fl in raw_flights:
        fl['zt_values'] = [zt_from_timestamp(ts, first_ts, last_ts) if ts is not None else None
                           for ts in fl['timestamps']]

    strict_flights = strict_filter_flights(raw_flights)
    # strict_filter_flights preserves zt_values via _zip_split_step
    print(f'  {sex}/{batch}: {len(raw_flights)} raw -> {len(strict_flights)} strict flights  '
          f'(span {first_ts} -> {last_ts})')

    return {
        'sex': sex, 'strain': strain, 'batch': batch,
        'cage_centroid': centroid,
        'first_ts': first_ts, 'last_ts': last_ts,
        'raw_flights': raw_flights,
        'flights': strict_flights,
        'fps': fps,
    }


def _list_tracking_files_in_dir(tracking_dir):
    """List forward_mosq_tracks_* files in an explicit final_tracking_data directory."""
    if not os.path.isdir(tracking_dir):
        return []
    files = []
    for name in sorted(os.listdir(tracking_dir)):
        if name.startswith('forward_mosq_tracks_') and not name.startswith('.'):
            files.append(os.path.join(tracking_dir, name))
    return files


def _centroid_from_settings(settings_path):
    """Load cage_border_points from a per-experiment settings YAML and return the cage
    centroid (or None), for an explicit settings path."""
    if not settings_path or not os.path.exists(settings_path):
        return None
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    cage_pts = settings.get('cage_border_points') if settings else None
    if not cage_pts:
        return None
    return calculate_cage_centroid(cage_pts)


def process_session_from_dir(experiment_dir, tracking_dir=None, settings_path=None,
                             cage_centroid=None, sex='experiment', strain='experiment',
                             batch='experiment', verbose=True):
    """GUI/CLI-facing single-experiment session loader.

    Takes an explicit BuzzSuite experiment folder and resolves:
      - tracking_dir:  <experiment_dir>/final_tracking_data by default
      - cage centroid: explicit cage_centroid, else cage_border_points from settings_path
                       (default <experiment_dir>/buzzwatch_track_settings.yml)
    Returns the standard session dict, or None on failure."""
    if tracking_dir is None:
        tracking_dir = os.path.join(experiment_dir, 'final_tracking_data')
    if settings_path is None:
        settings_path = os.path.join(experiment_dir, 'buzzwatch_track_settings.yml')

    centroid = cage_centroid if cage_centroid is not None else _centroid_from_settings(settings_path)
    if centroid is None:
        print(f'  ! no cage centroid for {experiment_dir} (looked in {settings_path})')
        return None

    files = _list_tracking_files_in_dir(tracking_dir)
    if not files:
        print(f'  ! no tracking files in {tracking_dir}')
        return None

    return _build_session_from_files(files, centroid, sex, strain, batch, verbose=verbose)


# ==== Per-experiment session/CDF cache (Step 5) ====
# The expensive part of every BuzzSwarm run is loading + strict-filtering the per-segment
# tracking .pkls (process_session_from_dir). Cache that derived session dict to
# plots/sholl_cache.pkl so repeated runs (r50 / Sholl / pooled / per-trajectory) reuse it. The
# cache is invalidated when any final_tracking_data file is newer than it OR when the filter
# tunables that shape the session change — so this never returns a stale/wrong result and does
# not alter numerics (it is the same computed session, just not rebuilt).
_SESSION_CACHE_TUNABLES = [
    'SPEED_FILTER_MIN', 'SPEED_FILTER_MAX', 'STEP_SPLIT_MAX', 'STEP_SPLIT_MIN', 'USE_STEP_SPLIT',
    'TELEPORT_MAX_PX', 'MIN_FLIGHT_FRAMES', 'MAX_DISPLACEMENT_PX', 'MEDIAN_SMOOTH_WINDOW',
    'DENSITY_BIN_PX', 'DENSITY_OUTLIER_FACTOR', 'DENSITY_DROP_FRAC',
]


def _session_cache_signature():
    return tuple((k, globals().get(k)) for k in _SESSION_CACHE_TUNABLES)


def _newest_tracking_mtime(tracking_dir):
    newest = 0.0
    if os.path.isdir(tracking_dir):
        for name in os.listdir(tracking_dir):
            if name.startswith('forward_mosq_tracks_') and not name.startswith('.'):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(tracking_dir, name)))
                except OSError:
                    pass
    return newest


def load_or_build_session(experiment_dir, cache_path=None, force_rebuild=False, verbose=False):
    """Return the session dict for an experiment, cached to plots/sholl_cache.pkl.

    Rebuilds via process_session_from_dir when the cache is missing, older than the newest
    final_tracking_data file, for a different experiment, or built with different filter
    tunables. Identical numerics to process_session_from_dir — this only avoids re-loading and
    re-filtering the .pkls. Labels are the process_session_from_dir defaults; callers that need a
    group label set sd['sex']/['strain']/['batch'] on the returned dict."""
    tracking_dir = os.path.join(experiment_dir, 'final_tracking_data')
    if cache_path is None:
        cache_path = os.path.join(experiment_dir, 'plots', 'sholl_cache.pkl')
    sig = _session_cache_signature()
    if not force_rebuild and os.path.isfile(cache_path):
        try:
            if os.path.getmtime(cache_path) >= _newest_tracking_mtime(tracking_dir):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if (isinstance(cached, dict) and cached.get('signature') == sig
                        and cached.get('experiment_dir') == experiment_dir
                        and cached.get('session') is not None):
                    if verbose:
                        print(f'  [sholl_cache] hit: {cache_path}')
                    return cached['session']
        except Exception:
            pass
    session = process_session_from_dir(experiment_dir, verbose=verbose)
    if session is not None:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'signature': sig, 'experiment_dir': experiment_dir, 'session': session}, f)
            if verbose:
                print(f'  [sholl_cache] wrote: {cache_path}')
        except Exception:
            pass
    return session


# ==== Time-window filtering helpers ====

def flights_in_zt_window(session_data, zt_start, zt_end, use_raw=False):
    """Return flights trimmed to points within a real-time window, anchored per window label:
        ZT 11.5-12.0  -> last 30 real minutes of recording (anchored to last_ts)
        ZT 5.5-6.0    -> middle 30 minutes of recording (centered at session midpoint)
    Other windows default to end-anchored.
    Set use_raw=True to use raw (loose-filtered) flights instead of strict-filtered flights."""
    flights = session_data['raw_flights'] if use_raw else session_data['flights']
    first_ts = session_data['first_ts']
    last_ts = session_data['last_ts']

    # Per-session rescaling: each session is treated as exactly 12 real hours.
    # ZT t (in [0, 12]) corresponds to fraction t/12 through the recorded duration.
    label = (round(zt_start, 2), round(zt_end, 2))
    span = last_ts - first_ts
    if label == (5.75, 6.25):
        mid_ts = first_ts + span / 2  # middle 30 real min = middle 1/24 of recording
        half = span / 48              # 15 real min = 1/48 of recording
        t_start = mid_ts - half
        t_end = mid_ts + half
    else:
        # ZT t -> first_ts + (t/12) * span
        t_start = first_ts + span * (zt_start / 12.0)
        t_end = first_ts + span * (zt_end / 12.0)
    out = []
    for fl in flights:
        keep_c, keep_t = [], []
        for pt, ts in zip(fl['coords'], fl['timestamps']):
            if ts is None:
                continue
            if t_start <= ts < t_end:
                keep_c.append(pt); keep_t.append(ts)
        if len(keep_c) >= 2:
            out.append({
                'fly_id': fl['fly_id'], 'coords': keep_c, 'timestamps': keep_t,
                'start_frame_abs': fl['start_frame_abs'],
            })
    return out


def all_flights(session_data, use_raw=False):
    """Return all flights (no time filtering) — used for step-filter validation & full-session views."""
    return session_data['raw_flights'] if use_raw else session_data['flights']


# ==== Plot 1: all-day r50 (30-min bins), sex merged w/ faint session lines ====

def compute_session_bins(session_data):
    """Bin TRAJECTORIES in REAL 30-min windows, END-ALIGNED:
        - last bin  = last 30 REAL min (matches Sholl ZT 11.5-12)
        - each trajectory is assigned to the bin containing its MIDPOINT timestamp
    For each bin, compute:
        r50     : median of per-trajectory medians (median-of-medians)
        prop100 : proportion of trajectories whose median distance <= 100 px
        prop150 : proportion of trajectories whose median distance <= 150 px
        activity: number of trajectories in the bin per MINUTE of real time
    Returns (lower_frac, upper_frac, r50, prop100, prop150, activity).
    Multiply lower/upper_frac by 12 to get squashed x on 0-12."""
    centroid = session_data['cage_centroid']
    first_ts = session_data['first_ts']
    last_ts = session_data['last_ts']
    total_s = (last_ts - first_ts).total_seconds()
    empty = tuple(np.array([]) for _ in range(6))
    if total_s <= 0:
        return empty

    # Per-session rescaling: each session = exactly 12 real hours.
    # Number of bins = 12 hours / BIN_MINUTES; each bin = total_s / n_bins recorded seconds.
    n_bins = int(round(12 * 60 / BIN_MINUTES))   # e.g. 10 min -> 72 bins, 30 min -> 24 bins
    bin_s = total_s / n_bins

    # Per bin we collect, per trajectory: its sorted distance array and its size.
    # The new "r50" metric per bin = radius R at which the AVERAGE per-trajectory
    # fraction within R equals 0.5.
    bin_traj_dists = [[] for _ in range(n_bins)]    # list of np.arrays
    traj_prop100   = [[] for _ in range(n_bins)]
    traj_prop150   = [[] for _ in range(n_bins)]

    for fl in session_data['flights']:
        dists = np.asarray([distance_to_center(pt, centroid) for pt in fl['coords']])
        if len(dists) == 0:
            continue
        ts_list = [ts for ts in fl['timestamps'] if ts is not None]
        if len(ts_list) < 2:
            continue
        mid_elapsed = ((ts_list[0] - first_ts).total_seconds()
                       + (ts_list[-1] - first_ts).total_seconds()) / 2.0
        if mid_elapsed < 0 or mid_elapsed > total_s:
            continue
        from_end = total_s - mid_elapsed
        b_from_end = int(from_end / bin_s)
        b = min(n_bins - 1, max(0, n_bins - 1 - b_from_end))
        bin_traj_dists[b].append(dists)
        traj_prop100[b].append(float(np.mean(dists <= 100)))
        traj_prop150[b].append(float(np.mean(dists <= 150)))

    r50 = np.empty(n_bins); prop100 = np.empty(n_bins); prop150 = np.empty(n_bins)
    n_traj = np.zeros(n_bins)
    for b in range(n_bins):
        td = bin_traj_dists[b]
        if len(td) < MIN_TRAJECTORIES_PER_BIN:
            r50[b] = np.nan; prop100[b] = np.nan; prop150[b] = np.nan
            n_traj[b] = len(td)
            continue
        # Build a weighted sample where each point has weight 1/(n_traj * n_pts_in_its_traj)
        # so total weight per trajectory = 1/n_traj, and total weight = 1.0.
        nT = len(td)
        all_d_parts = []
        all_w_parts = []
        for d_arr in td:
            n_pts = len(d_arr)
            all_d_parts.append(d_arr)
            all_w_parts.append(np.full(n_pts, 1.0 / (nT * n_pts)))
        all_d = np.concatenate(all_d_parts)
        all_w = np.concatenate(all_w_parts)
        order = np.argsort(all_d)
        sd = all_d[order]; sw = all_w[order]
        cumw = np.cumsum(sw)
        # Solve cumw(R) = 0.5 by linear interpolation between adjacent sorted distances
        if cumw[-1] < 0.5:
            r50[b] = np.nan
        else:
            idx = int(np.searchsorted(cumw, 0.5))
            if idx == 0:
                r50[b] = float(sd[0])
            else:
                d0, d1 = sd[idx - 1], sd[idx]
                c0, c1 = cumw[idx - 1], cumw[idx]
                r50[b] = float(d0 + (0.5 - c0) / (c1 - c0) * (d1 - d0)) if c1 > c0 else float(d1)
        prop100[b] = float(np.mean(traj_prop100[b]))
        prop150[b] = float(np.mean(traj_prop150[b]))
        n_traj[b] = nT

    upper_edge_s = np.array([total_s - (n_bins - 1 - b) * bin_s for b in range(n_bins)])
    upper_edge_s = np.minimum(upper_edge_s, total_s)
    lower_edge_s = np.maximum(0.0, upper_edge_s - bin_s)
    # Each bin = BIN_MINUTES real minutes
    activity = n_traj / float(BIN_MINUTES)

    return lower_edge_s / total_s, upper_edge_s / total_s, r50, prop100, prop150, activity


# Backwards-compatible alias (if anything still calls it, returns the 4-tuple it expects)
def compute_r50_per_session(session_data):
    lo, up, r50, _p100, _p150, activity = compute_session_bins(session_data)
    return lo, up, r50, activity


def _hybrid_x_transform(t_frac_upper, total_s):
    """Unused. Kept to avoid breaking references."""
    last_frac = (1800.0) / total_s
    pre_end = 1.0 - last_frac
    t = np.asarray(t_frac_upper, dtype=float)
    out = np.where(
        t > pre_end + 1e-12,
        11.5 + 0.5 * (t - pre_end) / last_frac,          # last 30 min -> [11.5, 12]
        11.5 * t / pre_end if pre_end > 0 else t * 0.0,  # pre-period -> [0, 11.5]
    )
    return out


def _step_xy(lower_x, upper_x, values):
    """Build step-style x,y arrays from per-bin lower/upper edges and values.
    Result: horizontal segment at value[i] from lower_x[i] to upper_x[i] for each bin."""
    xs = np.empty(2 * len(lower_x)); ys = np.empty_like(xs)
    xs[0::2] = lower_x; xs[1::2] = upper_x
    ys[0::2] = values; ys[1::2] = values
    return xs, ys


_METRIC_INDEX = {'r50': 2, 'prop100': 3, 'prop150': 4, 'activity': 5}


def _plot_binned_metric(all_sessions_data, value_key, ylabel, outdir, out_name,
                        ylim=None):
    """Generic all-day squashed plot for a per-session, per-bin trajectory metric.
    value_key is one of 'r50', 'prop100', 'prop150', 'activity'."""
    os.makedirs(outdir, exist_ok=True)
    common_x = np.arange(0.0, 12.0 + 1e-9, 0.25)  # 49 pts every 15 ZT min

    session_bins = {}
    session_interp = {}
    session_sex = {}

    metric_idx = _METRIC_INDEX[value_key]

    for sd in all_sessions_data:
        sid = f"{sd['sex']}_{sd['batch']}"
        session_sex[sid] = sd['sex']
        bins = compute_session_bins(sd)
        if len(bins[0]) == 0:
            continue
        lo_frac, up_frac = bins[0], bins[1]
        values = bins[metric_idx]
        lower_x = lo_frac * 12.0
        upper_x = up_frac * 12.0
        session_bins[sid] = (lower_x, upper_x, values)

        # For the sex mean: use each bin's UPPER EDGE so the last point lands at x=12
        # (and the curve extends across the full common grid).
        mask = ~np.isnan(values)
        if mask.sum() >= 2:
            interp = np.interp(common_x, upper_x[mask], values[mask],
                               left=values[mask][0], right=values[mask][-1])
        else:
            interp = np.full_like(common_x, np.nan, dtype=float)
        session_interp[sid] = interp

    fig, ax = plt.subplots(figsize=(11, 6))

    # Faint per-session step lines
    for sid, (lx, ux, v) in session_bins.items():
        sex = session_sex[sid]
        mask = ~np.isnan(v)
        if mask.sum() >= 1:
            xs, ys = _step_xy(lx[mask], ux[mask], v[mask])
            ax.plot(xs, ys, color=SEX_COLORS[sex], alpha=0.35, lw=1.0,
                    label='_nolegend_')

    # Sex mean on common grid
    sex_sessions = defaultdict(list)
    for sid, sx in session_sex.items():
        sex_sessions[sx].append(sid)
    wide = {'zt_hour_common': common_x}
    for sex in SEX_ORDER:
        sids = sex_sessions.get(sex, [])
        if not sids:
            continue
        arr = np.array([session_interp[s] for s in sids])
        for sid in sids:
            wide[f'{value_key}_{sid}'] = session_interp[sid]
        mean = np.nanmean(arr, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0).astype(float)
        n_valid[n_valid < 2] = np.nan
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid)
        wide[f'{value_key}_{sex}_mean'] = mean
        wide[f'{value_key}_{sex}_sem'] = sem
        m = ~np.isnan(mean)
        ax.plot(common_x[m], mean[m], color=SEX_COLORS[sex], lw=2.8,
                label=sex, zorder=5)
        ax.fill_between(common_x[m], (mean - sem)[m], (mean + sem)[m],
                        color=SEX_COLORS[sex], alpha=0.15, zorder=1)

    ax.set_xlabel('ZT  (session squashed to 0-12)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 12)
    ax.set_xticks(np.arange(0, 12.1, 1))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{out_name}.png'), dpi=200)
    plt.close(fig)

    pd.DataFrame(wide).to_csv(os.path.join(outdir, f'{out_name}.csv'),
                              index=False, float_format='%.4f')

    rows = []
    for sd in all_sessions_data:
        sid = f"{sd['sex']}_{sd['batch']}"
        if sid not in session_bins:
            continue
        lx, ux, v = session_bins[sid]
        # Per-session rescale: session = 12 real hours, so real_min = ZT * 60
        for i in range(len(lx)):
            rows.append({
                'session_id': sid, 'sex': sd['sex'], 'batch': sd['batch'],
                'real_minutes_from_start_low':  float(lx[i]) * 60.0,
                'real_minutes_from_start_high': float(ux[i]) * 60.0,
                'zt_hour_squashed_low':  float(lx[i]),
                'zt_hour_squashed_high': float(ux[i]),
                value_key: float(v[i]) if not np.isnan(v[i]) else np.nan,
            })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f'{out_name}_individual_sessions.csv'),
                              index=False, float_format='%.4f')


def plot_all_day_r50(all_sessions_data, outdir):
    _plot_binned_metric(
        all_sessions_data, 'r50',
        'Radius at which mean per-trajectory fraction within = 0.5  (px)',
        outdir, 'plot_r50_by_sex_all_day_ZT',
    )
    print(f'  wrote r50 plot and CSVs -> {outdir}')


def plot_all_day_activity(all_sessions_data, outdir):
    _plot_binned_metric(
        all_sessions_data, 'activity',
        'Flight activity  (trajectories per minute of real time)',
        outdir, 'plot_activity_by_sex_all_day_ZT',
    )
    print(f'  wrote activity plot and CSVs -> {outdir}')


def plot_all_day_no_squash(all_sessions_data, outdir, value_key, ylabel, out_name,
                            ylim=None):
    """All-day plot with NO time squashing — x-axis = real elapsed hours from session start.
    Each session's line ends where its real recording duration ends (so different lengths)."""
    os.makedirs(outdir, exist_ok=True)
    metric_idx = _METRIC_INDEX[value_key]

    session_lines = {}  # sid -> (x_real_hours_lower, x_real_hours_upper, values)
    session_sex = {}
    max_real_h = 0.0

    for sd in all_sessions_data:
        sid = f"{sd['sex']}_{sd['batch']}"
        session_sex[sid] = sd['sex']
        bins = compute_session_bins(sd)
        if len(bins[0]) == 0:
            continue
        lo_frac, up_frac = bins[0], bins[1]
        values = bins[metric_idx]
        total_s = (sd['last_ts'] - sd['first_ts']).total_seconds()
        # No-squash plot: x-axis = RECORDED hours from start (uncorrected by per-session
        # rescaling). Sessions of different recorded lengths end at different x positions.
        lo_h = lo_frac * total_s / 3600.0
        up_h = up_frac * total_s / 3600.0
        session_lines[sid] = (lo_h, up_h, values)
        max_real_h = max(max_real_h, float(up_h[-1]))

    fig, ax = plt.subplots(figsize=(13, 6))

    # Faint per-session step lines
    for sid, (lx, ux, v) in session_lines.items():
        sex = session_sex[sid]
        mask = ~np.isnan(v)
        if mask.sum() >= 1:
            xs, ys = _step_xy(lx[mask], ux[mask], v[mask])
            ax.plot(xs, ys, color=SEX_COLORS[sex], alpha=0.35, lw=1.0,
                    label='_nolegend_')

    # Sex mean: interpolate each session onto a common real-hours grid (every 0.25h)
    common_x = np.arange(0.0, max_real_h + 0.001, 0.25)
    sex_sessions = defaultdict(list)
    for sid, sx in session_sex.items():
        sex_sessions[sx].append(sid)
    wide = {'real_hours_from_start': common_x}
    for sex in SEX_ORDER:
        sids = sex_sessions.get(sex, [])
        if not sids:
            continue
        per_sess = []
        for sid in sids:
            lx, ux, v = session_lines[sid]
            mask = ~np.isnan(v)
            if mask.sum() >= 2:
                interp = np.interp(common_x, ux[mask], v[mask],
                                   left=v[mask][0], right=np.nan)
                interp = np.where(common_x > float(ux[mask][-1]), np.nan, interp)
            else:
                interp = np.full_like(common_x, np.nan, dtype=float)
            per_sess.append(interp)
            wide[f'{value_key}_{sid}'] = interp
        arr = np.array(per_sess)
        mean = np.nanmean(arr, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0).astype(float)
        n_valid[n_valid < 2] = np.nan
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid)
        wide[f'{value_key}_{sex}_mean'] = mean
        wide[f'{value_key}_{sex}_sem'] = sem
        m = ~np.isnan(mean)
        ax.plot(common_x[m], mean[m], color=SEX_COLORS[sex], lw=2.8,
                label=sex, zorder=5)
        ax.fill_between(common_x[m], (mean - sem)[m], (mean + sem)[m],
                        color=SEX_COLORS[sex], alpha=0.15, zorder=1)

    ax.set_xlabel('Recorded hours from session start  (no rescaling)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max_real_h)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{out_name}.png'), dpi=200)
    plt.close(fig)
    pd.DataFrame(wide).to_csv(os.path.join(outdir, f'{out_name}.csv'),
                              index=False, float_format='%.4f')


def plot_all_day_prop_within(all_sessions_data, outdir, radius_px):
    key = f'prop{radius_px}'
    _plot_binned_metric(
        all_sessions_data, key,
        f'Mean fraction of trajectory within {radius_px} px of centre',
        outdir, f'plot_prop_within_{radius_px}_by_sex_all_day_ZT',
        ylim=(-0.02, 1.02),
    )
    print(f'  wrote prop-within-{radius_px} plot and CSVs -> {outdir}')


# ==== Per-trajectory metrics (new) ====
#
# For every surviving trajectory in every session, emit one row containing its
# own r50 (median distance-to-centre), prop_within_100, prop_within_150, and
# the ZT (0-12) of its midpoint timestamp. This preserves all trajectory-level
# variation that the per-bin pooled r50 collapses to one number.

def extract_per_trajectory_metrics(all_sessions_data):
    """Return a DataFrame: one row per surviving trajectory.
    Columns: session_id, sex, batch, fly_id, traj_idx, n_frames,
             zt_midpoint, real_minutes_midpoint, r50, prop_within_100,
             prop_within_150.
    r50 is computed as np.median(distances_to_centre) — i.e. each trajectory's
    own CDF crossing 0.5 with linear interpolation between adjacent sorted
    distances.
    """
    rows = []
    for sd in all_sessions_data:
        sex = sd['sex']
        batch = sd['batch']
        sid = f"{sex}_{batch}"
        centroid = sd['cage_centroid']
        first_ts = sd['first_ts']
        last_ts = sd['last_ts']
        total_s = (last_ts - first_ts).total_seconds()
        if total_s <= 0:
            continue
        for traj_idx, fl in enumerate(sd['flights']):
            dists = np.asarray([distance_to_center(pt, centroid) for pt in fl['coords']])
            if len(dists) == 0:
                continue
            ts_list = [ts for ts in fl['timestamps'] if ts is not None]
            if len(ts_list) < 2:
                continue
            mid_elapsed = ((ts_list[0] - first_ts).total_seconds()
                           + (ts_list[-1] - first_ts).total_seconds()) / 2.0
            if mid_elapsed < 0 or mid_elapsed > total_s:
                continue
            zt_mid = 12.0 * mid_elapsed / total_s
            r50 = float(np.median(dists))
            p100 = float(np.mean(dists <= 100))
            p150 = float(np.mean(dists <= 150))
            rows.append({
                'session_id': sid,
                'sex': sex,
                'batch': batch,
                'fly_id': fl.get('fly_id', -1),
                'traj_idx': traj_idx,
                'n_frames': int(len(dists)),
                'zt_midpoint': zt_mid,
                'real_minutes_midpoint': mid_elapsed * (12.0 * 60.0 / total_s),
                'r50': r50,
                'prop_within_100': p100,
                'prop_within_150': p150,
            })
    return pd.DataFrame(rows)


def _pertraj_bin_summary(df, value_key, bin_minutes):
    """For each (sex, bin) compute mean, std, SEM, n over the per-trajectory
    values pooled across that sex's sessions. Returns dict[sex] -> dict with
    keys lower_zt, upper_zt, mean, sem, n."""
    if df.empty:
        return {}
    n_bins = int(round(12 * 60 / bin_minutes))
    bin_width = 12.0 / n_bins
    edges = np.linspace(0.0, 12.0, n_bins + 1)
    out = {}
    for sex in SEX_ORDER:
        sub = df[df['sex'] == sex]
        if sub.empty:
            continue
        idx = np.clip(np.floor(sub['zt_midpoint'].values / bin_width).astype(int),
                      0, n_bins - 1)
        vals = sub[value_key].values
        mean = np.full(n_bins, np.nan)
        sem = np.full(n_bins, np.nan)
        n = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            m = idx == b
            n[b] = int(m.sum())
            if n[b] >= MIN_TRAJECTORIES_PER_BIN:
                v = vals[m]
                mean[b] = float(np.mean(v))
                sem[b] = float(np.std(v, ddof=1) / np.sqrt(n[b])) if n[b] > 1 else np.nan
        out[sex] = {
            'lower_zt': edges[:-1],
            'upper_zt': edges[1:],
            'mean': mean,
            'sem': sem,
            'n': n,
        }
    return out


def plot_per_trajectory_scatter(df, value_key, ylabel, outdir, out_name,
                                ylim=None, bin_minutes=None):
    """Scatter of per-trajectory values vs ZT midpoint, overlaid with a per-sex
    mean + SEM line built from binning the trajectory pool across that sex's
    sessions (bin width = bin_minutes, default = BIN_MINUTES)."""
    os.makedirs(outdir, exist_ok=True)
    if bin_minutes is None:
        bin_minutes = BIN_MINUTES

    fig, ax = plt.subplots(figsize=(11, 6))

    # Scatter — one dot per trajectory, coloured by sex
    for sex in SEX_ORDER:
        sub = df[df['sex'] == sex]
        if sub.empty:
            continue
        ax.scatter(sub['zt_midpoint'].values, sub[value_key].values,
                   s=8, alpha=0.25, color=SEX_COLORS[sex], edgecolors='none',
                   label='_nolegend_', zorder=2)

    # Mean + SEM line per sex from binned pool
    summary = _pertraj_bin_summary(df, value_key, bin_minutes)
    csv_rows = []
    for sex, s in summary.items():
        mask = ~np.isnan(s['mean'])
        if not mask.any():
            continue
        centers = 0.5 * (s['lower_zt'] + s['upper_zt'])
        ax.plot(centers[mask], s['mean'][mask], color=SEX_COLORS[sex], lw=2.8,
                label=sex, zorder=5)
        # SEM band — only where SEM is finite
        sem_mask = mask & ~np.isnan(s['sem'])
        if sem_mask.any():
            ax.fill_between(centers[sem_mask],
                            s['mean'][sem_mask] - s['sem'][sem_mask],
                            s['mean'][sem_mask] + s['sem'][sem_mask],
                            color=SEX_COLORS[sex], alpha=0.18, zorder=1)
        for i in range(len(centers)):
            csv_rows.append({
                'sex': sex,
                'zt_lower': float(s['lower_zt'][i]),
                'zt_upper': float(s['upper_zt'][i]),
                'zt_center': float(centers[i]),
                'n_trajectories': int(s['n'][i]),
                f'{value_key}_mean': float(s['mean'][i]) if not np.isnan(s['mean'][i]) else np.nan,
                f'{value_key}_sem':  float(s['sem'][i])  if not np.isnan(s['sem'][i])  else np.nan,
            })

    ax.set_xlabel('ZT  (session squashed to 0-12)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 12)
    ax.set_xticks(np.arange(0, 12.1, 1))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{out_name}.png'), dpi=200)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(outdir, f'{out_name}_bin_summary.csv'),
        index=False, float_format='%.4f')


def plot_per_trajectory_violin(df, value_key, ylabel, outdir, out_name,
                               ylim=None, bin_minutes=30):
    """Violin plot: one violin per (sex, ZT bin) for the per-trajectory values.
    Default bin width = 30 min (24 bins) — coarser than the scatter so each
    violin has enough trajectories."""
    os.makedirs(outdir, exist_ok=True)
    if df.empty:
        return

    n_bins = int(round(12 * 60 / bin_minutes))
    bin_width = 12.0 / n_bins
    edges = np.linspace(0.0, 12.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(13, 6))
    sex_offsets = {sex: (i - 1) * (bin_width * 0.25) for i, sex in enumerate(SEX_ORDER)}
    violin_width = bin_width * 0.22

    csv_rows = []
    for sex in SEX_ORDER:
        sub = df[df['sex'] == sex]
        if sub.empty:
            continue
        idx = np.clip(np.floor(sub['zt_midpoint'].values / bin_width).astype(int),
                      0, n_bins - 1)
        vals = sub[value_key].values
        data_per_bin, positions = [], []
        for b in range(n_bins):
            m = idx == b
            n_in = int(m.sum())
            v = vals[m] if n_in >= MIN_TRAJECTORIES_PER_BIN else np.array([])
            csv_rows.append({
                'sex': sex,
                'zt_lower': float(edges[b]),
                'zt_upper': float(edges[b + 1]),
                'zt_center': float(centers[b]),
                'n_trajectories': n_in,
                f'{value_key}_mean':   float(np.mean(v))   if v.size else np.nan,
                f'{value_key}_median': float(np.median(v)) if v.size else np.nan,
                f'{value_key}_std':    float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
                f'{value_key}_q25':    float(np.percentile(v, 25)) if v.size else np.nan,
                f'{value_key}_q75':    float(np.percentile(v, 75)) if v.size else np.nan,
            })
            if v.size:
                data_per_bin.append(v)
                positions.append(centers[b] + sex_offsets[sex])
        if data_per_bin:
            parts = ax.violinplot(data_per_bin, positions=positions,
                                  widths=violin_width, showmeans=False,
                                  showmedians=True, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(SEX_COLORS[sex])
                body.set_edgecolor(SEX_COLORS[sex])
                body.set_alpha(0.45)
            if 'cmedians' in parts:
                parts['cmedians'].set_color(SEX_COLORS[sex])
                parts['cmedians'].set_linewidth(1.5)
        # Proxy legend handle
        ax.plot([], [], color=SEX_COLORS[sex], lw=6, alpha=0.6, label=sex)

    ax.set_xlabel('ZT  (session squashed to 0-12)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 12)
    ax.set_xticks(np.arange(0, 12.1, 1))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, axis='y')
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{out_name}.png'), dpi=200)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(outdir, f'{out_name}_bin_stats.csv'),
        index=False, float_format='%.4f')


def run_per_trajectory_analysis(all_sessions_data, outdir):
    """Emit per-trajectory r50, prop_within_100, prop_within_150 outputs
    (scatter + violin + CSVs) into outdir."""
    os.makedirs(outdir, exist_ok=True)
    df = extract_per_trajectory_metrics(all_sessions_data)

    # Master CSV: one row per surviving trajectory.
    df.to_csv(os.path.join(outdir, 'per_trajectory_metrics.csv'),
              index=False, float_format='%.4f')

    metrics = [
        ('r50',             'Per-trajectory r50  (median distance to centre, px)', None),
        ('prop_within_100', 'Per-trajectory fraction within 100 px of centre',     (-0.02, 1.02)),
        ('prop_within_150', 'Per-trajectory fraction within 150 px of centre',     (-0.02, 1.02)),
    ]
    for key, ylabel, ylim in metrics:
        plot_per_trajectory_scatter(
            df, key, ylabel, outdir,
            f'scatter_{key}_by_sex_all_day_ZT', ylim=ylim,
        )
        plot_per_trajectory_violin(
            df, key, ylabel, outdir,
            f'violin_{key}_by_sex_all_day_ZT', ylim=ylim,
        )

    print(f'  wrote per-trajectory analysis ({len(df)} trajectories) -> {outdir}')


# ==== Plot 2: Sholl profiles ====

def sholl_profile(distances, step_px=SHOLL_RADIUS_STEP_PX):
    """Legacy point-based Sholl profile (kept for back-compat). Treats input as a flat
    list of point distances and returns a per-point CDF."""
    if len(distances) == 0:
        return None
    d = np.asarray(distances, dtype=float)
    max_d = float(np.max(d))
    max_r = step_px * int(np.ceil(max_d / step_px)) if max_d > 0 else step_px
    radii = np.arange(step_px, max_r + step_px, step_px)
    cdf = np.array([float(np.mean(d <= r)) for r in radii])
    r25, r50, r75 = np.percentile(d, [25, 50, 75])
    return {
        'radii_px': radii, 'cdf': cdf,
        'r25_px': float(r25), 'r50_px': float(r50), 'r75_px': float(r75),
        'n_points': int(len(d)),
    }


def sholl_profile_trajectory(trajectory_dists, step_px=SHOLL_RADIUS_STEP_PX):
    """Per-trajectory-mean Sholl profile.
    Input: list of np.arrays, one per trajectory (point distances to centre).
    For each radius R, the CDF value is the MEAN across trajectories of
    (fraction of that trajectory's points within R).
    The reported r25/r50/r75 are the radii where this mean fraction crosses 0.25/0.5/0.75.
    """
    trajectory_dists = [np.asarray(t, dtype=float) for t in trajectory_dists if len(t) > 0]
    if not trajectory_dists:
        return None
    n_traj = len(trajectory_dists)
    max_d = float(max(np.max(t) for t in trajectory_dists))
    max_r = step_px * int(np.ceil(max_d / step_px)) if max_d > 0 else step_px
    radii = np.arange(step_px, max_r + step_px, step_px)

    # Build weighted distance distribution (each trajectory contributes total weight 1/n_traj)
    parts_d = [t for t in trajectory_dists]
    parts_w = [np.full(len(t), 1.0 / (n_traj * len(t))) for t in trajectory_dists]
    all_d = np.concatenate(parts_d)
    all_w = np.concatenate(parts_w)
    order = np.argsort(all_d)
    sd = all_d[order]; sw = all_w[order]
    cumw = np.cumsum(sw)

    # CDF at the radii grid: cumw at the largest sd <= r
    idx_at = np.searchsorted(sd, radii, side='right') - 1
    cdf = np.where(idx_at >= 0, cumw[np.clip(idx_at, 0, len(cumw) - 1)], 0.0)
    cdf = np.where(idx_at >= 0, cdf, 0.0)

    def _radius_at(p):
        if cumw[-1] < p:
            return float('nan')
        i = int(np.searchsorted(cumw, p))
        if i == 0:
            return float(sd[0])
        d0, d1 = sd[i - 1], sd[i]
        c0, c1 = cumw[i - 1], cumw[i]
        return float(d0 + (p - c0) / (c1 - c0) * (d1 - d0)) if c1 > c0 else float(d1)

    return {
        'radii_px': radii, 'cdf': cdf,
        'r25_px': _radius_at(0.25),
        'r50_px': _radius_at(0.5),
        'r75_px': _radius_at(0.75),
        'n_trajectories': n_traj,
        'n_points': int(sum(len(t) for t in trajectory_dists)),
    }


def plot_sholl_single(profile, color, title, out_png, out_csv, label=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(profile['radii_px'], profile['cdf'], color=color, lw=2.2, label=label or '')
    ax.axvline(profile['r50_px'], ls='--', color=color, alpha=0.7,
               label=f"r50 = {profile['r50_px']:.1f} px")
    for ref in (125, 150, 175):
        ax.axvline(ref, color='grey', alpha=0.25, lw=0.8)
    ax.set_xlabel('Radius from cage centre (px)')
    ax.set_ylabel('Mean per-trajectory fraction within radius')
    ax.set_ylim(0, 1.02)
    # title removed
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    pd.DataFrame({
        'radius_px': profile['radii_px'],
        'mean_per_trajectory_fraction_within': profile['cdf'],
    }).to_csv(out_csv, index=False, float_format='%.4f')


def plot_sholl_combined(sex_profiles, title, out_png, out_csv):
    """sex_profiles: dict sex -> profile."""
    fig, ax = plt.subplots(figsize=(9, 6))
    data = {}
    # union of radii: use the longest
    max_radii = None
    for sex, prof in sex_profiles.items():
        if prof is None:
            continue
        if max_radii is None or len(prof['radii_px']) > len(max_radii):
            max_radii = prof['radii_px']
    if max_radii is None:
        return
    data['radius_px'] = max_radii
    for sex in SEX_ORDER:
        prof = sex_profiles.get(sex)
        if prof is None:
            continue
        cdf = np.interp(max_radii, prof['radii_px'], prof['cdf'],
                        left=0.0, right=1.0)
        data[f'cdf_{sex}'] = cdf
        ax.plot(max_radii, cdf, color=SEX_COLORS[sex], lw=2.5,
                label=f"{sex}  (r50={prof['r50_px']:.1f} px, n={prof['n_points']})")
        ax.axvline(prof['r50_px'], ls='--', color=SEX_COLORS[sex], alpha=0.55, lw=1.2)
    for ref in (125, 150, 175):
        ax.axvline(ref, color='grey', alpha=0.25, lw=0.8)
    ax.set_xlabel('Radius from cage centre (px)')
    ax.set_ylabel('Mean per-trajectory fraction within radius')
    ax.set_ylim(0, 1.02)
    # title removed
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    pd.DataFrame(data).to_csv(out_csv, index=False, float_format='%.4f')


def run_sholl_analysis(all_sessions_data, outdir_base):
    for zt_start, zt_end in SHOLL_WINDOWS:
        label = f'ZT{zt_start}-{zt_end}'
        outdir = os.path.join(outdir_base, label)
        os.makedirs(outdir, exist_ok=True)

        # Per-sex combined profile (pool TRAJECTORIES across sessions of that sex)
        sex_profiles = {}
        for sex in SEX_ORDER:
            traj_dists = []
            for sd in all_sessions_data:
                if sd['sex'] != sex:
                    continue
                centroid = sd['cage_centroid']
                for fl in flights_in_zt_window(sd, zt_start, zt_end):
                    arr = np.asarray([distance_to_center(pt, centroid) for pt in fl['coords']])
                    if len(arr) > 0:
                        traj_dists.append(arr)
            prof = sholl_profile_trajectory(traj_dists) if traj_dists else None
            sex_profiles[sex] = prof

        plot_sholl_combined(
            sex_profiles,
            f'Sholl CDF  ({label})  — all sessions pooled (per-trajectory mean)',
            os.path.join(outdir, f'sholl_{label}_combined.png'),
            os.path.join(outdir, f'sholl_{label}_combined.csv'),
        )

        # Per-session individual plots
        for sd in all_sessions_data:
            sex = sd['sex']
            sex_dir = os.path.join(outdir, sex)
            os.makedirs(sex_dir, exist_ok=True)
            sid = f"{sex}_{sd['batch']}"
            centroid = sd['cage_centroid']
            traj_dists = []
            for fl in flights_in_zt_window(sd, zt_start, zt_end):
                arr = np.asarray([distance_to_center(pt, centroid) for pt in fl['coords']])
                if len(arr) > 0:
                    traj_dists.append(arr)
            prof = sholl_profile_trajectory(traj_dists) if traj_dists else None
            if prof is None:
                print(f'    ! no data for {sid} in {label}')
                continue
            plot_sholl_single(
                prof, SEX_COLORS[sex],
                f'Sholl CDF  — {sid}  ({label})  (per-trajectory mean)',
                os.path.join(sex_dir, f'sholl_{label}_{sid}.png'),
                os.path.join(sex_dir, f'sholl_{label}_{sid}.csv'),
                label=sid,
            )
        print(f'  wrote Sholl plots for {label} -> {outdir}')


# ==== Plot 3: heatmaps (ZT11.5-12) ====

def sex_cmap(sex):
    base = SEX_COLORS[sex]
    return LinearSegmentedColormap.from_list(f'{sex}_cmap', ['white', base])


def plot_heatmap_single(xs, ys, cage_centroid, cmap, title, out_png, out_csv, bins=HEATMAP_BINS):
    if len(xs) == 0:
        print(f'    ! no points for heatmap {title}')
        return
    xs = np.asarray(xs); ys = np.asarray(ys)
    # bounds: cage range + margin
    xmin, xmax = float(np.min(xs)) - 20, float(np.max(xs)) + 20
    ymin, ymax = float(np.min(ys)) - 20, float(np.max(ys)) + 20
    H, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    Hvis = np.log1p(H)
    p99 = np.percentile(Hvis, 99) if Hvis.max() > 0 else 1.0
    if p99 > 0:
        Hvis = np.clip(Hvis / p99, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(Hvis.T, origin='lower',
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
              cmap=cmap, interpolation='bilinear', aspect='equal')
    # No centre cross, no axis ticks/labels (match individual-trajectory style);
    # keep a thin frame box only.
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=HEATMAP_DPI)
    plt.close(fig)

    # CSV: bin centers + normalized density
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(xc, yc, indexing='ij')
    df = pd.DataFrame({
        'x_bin_center': XX.ravel(),
        'y_bin_center': YY.ravel(),
        'count': H.ravel(),
        'density_normalized': Hvis.ravel(),
    })
    # drop all-zero bins to keep CSV manageable (optional)
    df = df[df['count'] > 0]
    df.to_csv(out_csv, index=False, float_format='%.4f')


def run_heatmaps(all_sessions_data, outdir):
    os.makedirs(outdir, exist_ok=True)
    zt_s, zt_e = HEATMAP_WINDOW

    # gather per-sex pooled points and per-session points
    sex_xs = defaultdict(list); sex_ys = defaultdict(list)
    sex_centroid_any = {}

    for sd in all_sessions_data:
        sex = sd['sex']
        sid = f"{sex}_{sd['batch']}"
        centroid = sd['cage_centroid']
        fl_in = flights_in_zt_window(sd, zt_s, zt_e)
        xs, ys = [], []
        for fl in fl_in:
            for pt in fl['coords']:
                xs.append(pt[0]); ys.append(pt[1])

        # per-session heatmap
        sex_dir = os.path.join(outdir, sex)
        os.makedirs(sex_dir, exist_ok=True)
        plot_heatmap_single(
            xs, ys, centroid, sex_cmap(sex),
            f'Heatmap  — {sid}  (ZT{zt_s}-{zt_e})',
            os.path.join(sex_dir, f'heatmap_{sid}_{HEATMAP_BINS}x{HEATMAP_BINS}.png'),
            os.path.join(sex_dir, f'heatmap_{sid}_{HEATMAP_BINS}x{HEATMAP_BINS}.csv'),
        )

        sex_xs[sex].extend(xs); sex_ys[sex].extend(ys)
        sex_centroid_any[sex] = centroid

    # combined per-sex heatmaps (pooled sessions per sex)
    for sex in SEX_ORDER:
        xs = sex_xs.get(sex, []); ys = sex_ys.get(sex, [])
        if not xs:
            continue
        plot_heatmap_single(
            xs, ys, sex_centroid_any.get(sex), sex_cmap(sex),
            f'Heatmap combined  — {sex}  (ZT{zt_s}-{zt_e})',
            os.path.join(outdir, f'heatmap_combined_{sex}_{HEATMAP_BINS}x{HEATMAP_BINS}.png'),
            os.path.join(outdir, f'heatmap_combined_{sex}_{HEATMAP_BINS}x{HEATMAP_BINS}.csv'),
        )
    print(f'  wrote heatmaps -> {outdir}')


# ==== Plot 4 & 5: trajectories (combined per session, individual per flight) ====

def plot_session_combined_trajectories(sd, out_png, out_csv):
    zt_s, zt_e = TRAJECTORY_WINDOW
    fl_in = flights_in_zt_window(sd, zt_s, zt_e)
    sex = sd['sex']; sid = f"{sex}_{sd['batch']}"
    if not fl_in:
        print(f'    ! no flights in {sid} for ZT{zt_s}-{zt_e}')
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    rows = []
    for traj_idx, fl in enumerate(fl_in):
        xs = [p[0] for p in fl['coords']]
        ys = [p[1] for p in fl['coords']]
        ax.plot(xs, ys, color=SEX_COLORS[sex], alpha=0.45, lw=0.8)
        for pt, ts in zip(fl['coords'], fl['timestamps']):
            rows.append({
                'trajectory_idx': traj_idx, 'fly_id': fl['fly_id'],
                'timestamp': ts.isoformat() if ts is not None else '',
                'x_px': pt[0], 'y_px': pt[1],
            })
    ax.set_aspect('equal', adjustable='datalim')
    ax.invert_yaxis()  # image-style
    # No centre cross, no axis ticks/labels (match individual-trajectory style);
    # keep a thin frame box only.
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format='%.4f')


INDIVIDUAL_TRAJ_GRID_COLS = 5
INDIVIDUAL_TRAJ_GRID_ROWS = 6  # 30 per page
INDIVIDUAL_TRAJ_PER_PAGE = INDIVIDUAL_TRAJ_GRID_COLS * INDIVIDUAL_TRAJ_GRID_ROWS


def plot_trajectory_montage(flights, sex, cage_bounds, out_png, title):
    """Grid of trajectories, one subplot each. No start/end/centre markers.
    cage_bounds: (xmin, xmax, ymin, ymax) shared across all subplots for visual comparability."""
    n = len(flights)
    cols = INDIVIDUAL_TRAJ_GRID_COLS
    rows = max(1, int(math.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.6, rows * 2.6),
                             squeeze=False)
    xmin, xmax, ymin, ymax = cage_bounds
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        if idx < n:
            fl = flights[idx]
            xs = [p[0] for p in fl['coords']]
            ys = [p[1] for p in fl['coords']]
            ax.plot(xs, ys, color=SEX_COLORS[sex], lw=0.9, alpha=0.9)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)  # invert y (image coords)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    # title removed
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_trajectories(all_sessions_data, combined_outdir, individual_outdir):
    os.makedirs(combined_outdir, exist_ok=True)
    os.makedirs(individual_outdir, exist_ok=True)
    zt_s, zt_e = TRAJECTORY_WINDOW

    for sd in all_sessions_data:
        sex = sd['sex']; sid = f"{sex}_{sd['batch']}"
        # combined per session
        sex_dir_c = os.path.join(combined_outdir, sex)
        os.makedirs(sex_dir_c, exist_ok=True)
        plot_session_combined_trajectories(
            sd,
            os.path.join(sex_dir_c, f'{sid}_combined.png'),
            os.path.join(sex_dir_c, f'{sid}_combined.csv'),
        )

        # individual montage(s) per session
        sess_dir_i = os.path.join(individual_outdir, sex, sid)
        os.makedirs(sess_dir_i, exist_ok=True)
        fl_in = flights_in_zt_window(sd, zt_s, zt_e)
        # Sort: largest (most points) first, so page 1 has the most interesting trajectories
        fl_in = sorted(fl_in, key=lambda f: len(f['coords']), reverse=True)

        # Shared bounds for visual comparability across subplots in a session
        if fl_in:
            all_x = [p[0] for fl in fl_in for p in fl['coords']]
            all_y = [p[1] for fl in fl_in for p in fl['coords']]
            pad = 20
            cage_bounds = (min(all_x) - pad, max(all_x) + pad,
                           min(all_y) - pad, max(all_y) + pad)
        else:
            cage_bounds = (0, 1, 0, 1)

        # Paginate
        n = len(fl_in)
        per_page = INDIVIDUAL_TRAJ_PER_PAGE
        n_pages = max(1, int(math.ceil(n / per_page)))
        for page in range(n_pages):
            a = page * per_page
            b = min(n, a + per_page)
            chunk = fl_in[a:b]
            title = f'{sid}  trajectories {a+1}-{b} / {n}  (ZT{zt_s}-{zt_e})'
            page_suffix = '' if n_pages == 1 else f'_p{page+1}'
            plot_trajectory_montage(
                chunk, sex, cage_bounds,
                os.path.join(sess_dir_i, f'{sid}_trajectories{page_suffix}.png'),
                title,
            )

        # Single session-level CSV: one row per point, indexed by trajectory_idx
        rows = []
        for idx, fl in enumerate(fl_in):
            for pt, ts in zip(fl['coords'], fl['timestamps']):
                rows.append({
                    'trajectory_idx': idx,
                    'fly_id': fl['fly_id'],
                    'n_points': len(fl['coords']),
                    'timestamp': ts.isoformat() if ts is not None else '',
                    'x_px': pt[0], 'y_px': pt[1],
                })
        pd.DataFrame(rows).to_csv(
            os.path.join(sess_dir_i, f'{sid}_trajectories.csv'),
            index=False, float_format='%.4f')

        print(f'  {sid}: wrote {n} trajectories in {n_pages} montage page(s)')


# ==== Plot 6: prop-near-centre strip plot ====

def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def run_prop_near_centre(all_sessions_data, outdir):
    os.makedirs(outdir, exist_ok=True)
    zt_s, zt_e = TRAJECTORY_WINDOW

    rows = []
    for sd in all_sessions_data:
        sex = sd['sex']; sid = f"{sex}_{sd['batch']}"
        centroid = sd['cage_centroid']
        fl_in = flights_in_zt_window(sd, zt_s, zt_e)
        for idx, fl in enumerate(fl_in, start=1):
            dists = np.array([distance_to_center(p, centroid) for p in fl['coords']])
            rows.append({
                'trajectory_id': f'{sid}_{idx:03d}',
                'session_id': sid, 'sex': sex, 'batch': sd['batch'],
                'fly_id': fl['fly_id'],
                'n_points': int(len(fl['coords'])),
                'prop_within_100': float(np.mean(dists <= 100.0)),
                'prop_within_150': float(np.mean(dists <= 150.0)),
            })

    # One CSV, one row per trajectory (individual points), BOTH p100 and p150.
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'prop_near_centre_stripplot.csv'),
              index=False, float_format='%.4f')

    # Strip plot: one panel per radius (100 px, 150 px).
    metrics = [('prop_within_100', 'within 100 px'), ('prop_within_150', 'within 150 px')]
    positions = {sex: i for i, sex in enumerate(SEX_ORDER)}
    tick_labels = {'Female': '♀', 'Male': '♂', 'FruM': 'fruM♂'}
    pairs = [('Female', 'Male'), ('Male', 'FruM'), ('Female', 'FruM')]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    stat_rows = []
    for ax, (col, label) in zip(axes, metrics):
        for sex in SEX_ORDER:
            sub = df[df['sex'] == sex]
            if len(sub) == 0:
                continue
            x = np.full(len(sub), positions[sex], dtype=float) + rng.uniform(-0.22, 0.22, len(sub))
            ax.scatter(x, sub[col], color=SEX_COLORS[sex], s=18, alpha=0.55, edgecolor='none')
            med = float(np.median(sub[col]))
            ax.hlines(med, positions[sex] - 0.28, positions[sex] + 0.28,
                      colors='black', lw=2.5, zorder=5)

        ax.set_xticks([positions[s] for s in SEX_ORDER])
        ax.set_xticklabels([tick_labels[s] for s in SEX_ORDER], fontsize=14)
        ax.set_title(f'Proportion of trajectory {label} of centre')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # pairwise Mann-Whitney U (two-sided)
        y_top = 0.93
        for i, (a, b) in enumerate(pairs):
            sa = df.loc[df['sex'] == a, col].to_numpy()
            sb = df.loc[df['sex'] == b, col].to_numpy()
            if len(sa) > 0 and len(sb) > 0:
                try:
                    res = stats.mannwhitneyu(sa, sb, alternative='two-sided')
                    p = float(res.pvalue); u = float(res.statistic)
                except Exception:
                    p = 1.0; u = float('nan')
            else:
                p = 1.0; u = float('nan')
            stat_rows.append({'metric': col, 'group_a': a, 'n_a': len(sa),
                              'group_b': b, 'n_b': len(sb),
                              'U_statistic': u, 'p_value': p, 'sig': stars(p)})
            if a in positions and b in positions:
                x1, x2 = positions[a], positions[b]
                y = y_top + i * 0.045
                ax.plot([x1, x1, x2, x2], [y, y + 0.015, y + 0.015, y], color='black', lw=1.1)
                ax.text((x1 + x2) / 2.0, y + 0.02, stars(p), ha='center', va='bottom', fontsize=12)
        ax.set_ylim(-0.02, y_top + len(pairs) * 0.045 + 0.06)

    axes[0].set_ylabel('Proportion of trajectory frames within radius')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'prop_near_centre_stripplot.png'), dpi=200)
    plt.close(fig)

    pd.DataFrame(stat_rows).to_csv(os.path.join(outdir, 'prop_near_centre_stats.csv'),
                                   index=False, float_format='%.6g')
    print(f'  wrote prop-near-centre (p100+p150) strip plot -> {outdir}')


# ==== Validation: before vs after strict step filter ====

def run_filter_validation(all_sessions_data, outdir):
    os.makedirs(outdir, exist_ok=True)
    summary_rows = []
    for sd in all_sessions_data:
        sex = sd['sex']; sid = f"{sex}_{sd['batch']}"
        sex_dir = os.path.join(outdir, sex)
        os.makedirs(sex_dir, exist_ok=True)

        # Restrict to ZT0-ZT12 for visual fairness
        raw = sd['raw_flights']
        strict = sd['flights']

        # Side-by-side trajectory overlay
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, fls, sub in zip(axes, [raw, strict], ['BEFORE strict filter', 'AFTER strict filter']):
            for fl in fls:
                xs = [p[0] for p in fl['coords']]; ys = [p[1] for p in fl['coords']]
                ax.plot(xs, ys, color=SEX_COLORS[sex], alpha=0.25, lw=0.5)
            if sd.get('cage_centroid') is not None:
                cx, cy = sd['cage_centroid']
                ax.plot(cx, cy, '+', color='black', ms=12, mew=1.5)
            ax.set_aspect('equal', adjustable='datalim')
            ax.invert_yaxis()
            ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
        fig.tight_layout()
        fig.savefig(os.path.join(sex_dir, f'{sid}_filter_comparison.png'),
                    dpi=160, bbox_inches='tight')
        plt.close(fig)

        # Stats
        def flat_speeds(fls):
            v = []
            for fl in fls:
                v.extend(step_speeds(fl['coords']).tolist())
            return np.array(v) if v else np.array([0.0])

        raw_n_pts = sum(len(fl['coords']) for fl in raw)
        strict_n_pts = sum(len(fl['coords']) for fl in strict)
        pct_removed = (100.0 * (raw_n_pts - strict_n_pts) / raw_n_pts) if raw_n_pts else 0.0
        vs_raw = flat_speeds(raw); vs_strict = flat_speeds(strict)

        stats_row = {
            'session_id': sid, 'sex': sex, 'batch': sd['batch'],
            'n_trajectories_before': len(raw), 'n_trajectories_after': len(strict),
            'n_points_before': raw_n_pts, 'n_points_after': strict_n_pts,
            'pct_points_removed': pct_removed,
            'step_speed_mean_before':   float(np.mean(vs_raw)),
            'step_speed_median_before': float(np.median(vs_raw)),
            'step_speed_p99_before':    float(np.percentile(vs_raw, 99)),
            'step_speed_mean_after':    float(np.mean(vs_strict)),
            'step_speed_median_after':  float(np.median(vs_strict)),
            'step_speed_p99_after':     float(np.percentile(vs_strict, 99)),
        }
        pd.DataFrame([stats_row]).to_csv(
            os.path.join(sex_dir, f'{sid}_filter_stats.csv'),
            index=False, float_format='%.4f')
        summary_rows.append(stats_row)
        print(f'    {sid}: {raw_n_pts} -> {strict_n_pts} pts  ({pct_removed:.1f}% removed)')

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(outdir, 'summary_all_sessions.csv'),
        index=False, float_format='%.4f')
    print(f'  wrote filter validation -> {outdir}')
