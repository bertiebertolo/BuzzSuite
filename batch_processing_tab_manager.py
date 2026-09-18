import os
from multiprocessing import current_process
if current_process().name != "MainProcess":
    os.environ.setdefault("BUZZSUITE_HEADLESS_WORKER", "1")
    # Cap BLAS/OpenMP threads inside each worker process so N parallel workers don't each fan out
    # to all cores (oversubscription thrash). Set before numpy/cv2 import below so the thread pools
    # honor it. OpenCV (the hot path) is additionally pinned per-run via cv2.setNumThreads().
    for _thr_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_thr_var, "1")

from common_imports import *
import cv2   # explicit: cv2 is no longer re-exported by common_imports (kept lean)
from logger import MultiLogger

import threading
from threading import Thread
from contextlib import contextmanager

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
import pickle
import sys
import warnings
from experiment_manager import ExperimentManager
from tooltip import add_tooltip

# Suppress expected NumPy warnings from empty data windows
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

# Cross-job worker-count coordination. Every Pool-consuming operation in the app (tracking via
# batch_runner.run_tracking_job_for_folder, concatenation, and the speed filter) independently
# sized itself up to cpu_count()-1 workers -- fine for one operation at a time, but the Experiment
# Dashboard runs several experiments concurrently in separate threads, so two experiments tracking
# together could spawn ~2x cpu_count()-1 worker processes fighting over the same cores (seen live:
# 18 processes on a 10-core machine from 2 concurrent dashboard jobs). pool_concurrency_slot() is a
# process-wide (not per-experiment) counter of how many such operations are currently running, so
# each one can divide the machine's budget by however many siblings are active *at the moment it
# starts*. Not fully dynamic -- a Pool's size is fixed at creation and won't shrink/grow as
# siblings start or finish later -- but it fixes the common case of starting several experiments
# around the same time. A plain threading.Lock is correct here: all callers run as threads inside
# the one GUI process, never across separate OS processes.
_pool_slot_lock = threading.Lock()
_active_pool_ops = 0


@contextmanager
def pool_concurrency_slot():
    """Yields the number of Pool-consuming operations (including this one) currently active."""
    global _active_pool_ops
    with _pool_slot_lock:
        _active_pool_ops += 1
        count = _active_pool_ops
    try:
        yield count
    finally:
        with _pool_slot_lock:
            _active_pool_ops = max(0, _active_pool_ops - 1)


def _default_batch_worker_count(config=None, concurrency=1):
    """Worker process count for one experiment's tracking Pool. A user-configured
    ``batch_processing_workers`` in config.json is honoured exactly (still divided across
    concurrently-running experiments and capped at cpu_count()-1) -- it used to be silently
    overridden by `max(configured, recommended)` any time the configured value was LOWER than the
    auto-recommended one, which meant there was no way to dial worker count down from the GUI (see
    DEVLOG: this let enough simultaneous heavy segments pile up to exhaust memory on a 32-core box).
    Falls back to `recommended` only when unset or not a valid positive int."""
    cpu_total = cpu_count() or 1
    concurrency = max(1, int(concurrency))
    recommended = max(1, (cpu_total - 2) // concurrency)
    upper_bound = max(1, (cpu_total - 1) // concurrency)

    configured = None
    if isinstance(config, dict):
        configured = config.get('batch_processing_workers')

    try:
        configured = max(1, int(configured) // concurrency) if configured is not None else recommended
    except (TypeError, ValueError):
        configured = recommended

    return max(1, min(configured, upper_bound))


def _tracking_output_path(folder_analysis, video_name):
    return os.path.join(folder_analysis, "final_tracking_data", f"forward_mosq_tracks_{video_name}")


def _should_skip_video(video_name, folder_analysis):
    if os.environ.get('BUZZSUITE_FORCE_REPROCESS', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return False
    return os.path.exists(_tracking_output_path(folder_analysis, video_name))


# Segments the user chose to Skip in the app. Kept in their own small file (not the experiment JSON,
# which ExperimentManager rewrites from its own fields and would silently drop an extra key).
# Skipped segments are never tracked and are simply absent from analyzed_data.pkl, which is built
# from final_tracking_data/ only. The typical case is the first segment of a recording, where the
# camera is still focusing: it produces hundreds of false detections per frame.
EXCLUDED_SEGMENTS_FILE = "excluded_segments.json"


def load_excluded_segments(folder_analysis):
    """Set of segment names skipped for this experiment (empty if none / unreadable)."""
    import json
    try:
        with open(os.path.join(folder_analysis, EXCLUDED_SEGMENTS_FILE), 'r', encoding='utf-8') as f:
            return set(json.load(f).get('segments', []))
    except (OSError, ValueError, AttributeError, TypeError):
        return set()


def set_segment_excluded(folder_analysis, video_name, excluded=True):
    """Add (or with ``excluded=False`` remove) one segment. Written via a temp file + os.replace so
    a worker process reading it at the same moment never sees a half-written file. On Windows,
    os.replace raises PermissionError while another process has the target open (workers read it at
    the start of each segment), so the swap is retried briefly."""
    import json
    import time
    segments = load_excluded_segments(folder_analysis)
    if excluded:
        segments.add(video_name)
    else:
        segments.discard(video_name)
    path = os.path.join(folder_analysis, EXCLUDED_SEGMENTS_FILE)
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'segments': sorted(segments),
                   'note': "Segments skipped in the app: not tracked, and left out of "
                           "analyzed_data.pkl. Remove a name (or this file) to track it again."},
                  f, indent=2)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)
    return segments

# --- Compatibility shim for legacy NumPy internal module paths ---
def _patch_numpy_compat():
    """
    Ensure legacy module paths that sometimes appear in pickles (e.g. 'numpy._core')
    are available in sys.modules so unpickling succeeds in worker processes.
    Call this at module import time so each multiprocessing worker inherits it.
    """
    try:
        import numpy as _np
        # map numpy._core -> numpy.core
        sys.modules.setdefault('numpy._core', getattr(_np, 'core', _np))
        # map common internal extension modules if present
        if hasattr(_np.core, '_multiarray_umath'):
            sys.modules.setdefault('numpy._core._multiarray_umath', _np.core._multiarray_umath)
            sys.modules.setdefault('numpy.core._multiarray_umath', _np.core._multiarray_umath)
        elif hasattr(_np.core, 'multiarray'):
            sys.modules.setdefault('numpy._core._multiarray_umath', _np.core.multiarray)
            sys.modules.setdefault('numpy.core._multiarray_umath', _np.core.multiarray)
    except Exception:
        # Non-fatal: if numpy cannot be imported or attributes missing, let unpickling report errors later
        pass

# Run the patch so it applies in the main process and in worker processes on import
_patch_numpy_compat()

DEFAULT_VIDEO_SEGMENT_SECONDS = int(os.environ.get('BUZZSUITE_VIDEO_SEGMENT_SECONDS', '1200'))
ROLLING_RESAMPLE_INTERVAL = os.environ.get('BUZZSUITE_ROLLING_RESAMPLE_INTERVAL', '1min')  # Resample interval (e.g., '1min', '5min')
ROLLING_WINDOW_SIZE_MINUTES = int(os.environ.get('BUZZSUITE_ROLLING_WINDOW_MINUTES', '20'))  # Rolling average window in minutes
ZT0_HOUR = int(os.environ.get('BUZZSUITE_ZT0_HOUR', '5'))  # Hour of day corresponding to ZT0 (lights on)

# Plot figure configurations
FIGSIZE_ROLLING_PLOT = tuple(map(int, os.environ.get('BUZZSUITE_FIGSIZE_ROLLING', '10,6').split(',')))  # (width, height)
FIGSIZE_SUMMARY_PLOT = tuple(map(int, os.environ.get('BUZZSUITE_FIGSIZE_SUMMARY', '3,3').split(',')))  # (width, height)
FIGSIZE_HISTOGRAM_PLOT = tuple(map(int, os.environ.get('BUZZSUITE_FIGSIZE_HISTOGRAM', '10,6').split(',')))  # (width, height)
FIGSIZE_LOCATION_PLOT = tuple(map(int, os.environ.get('BUZZSUITE_FIGSIZE_LOCATION', '20,6').split(',')))  # (width, height)
SPEAKER_SIDE_MAPPING_VERSION = 2


def _extract_version_index(video_name):
    match = re.search(r'_v(\d+)(?:_|$)', str(video_name))
    return int(match.group(1)) if match else 1


def _extract_clip_sequence_index(video_name):
    match = re.search(r'_(\d{5})(?:_|$)', str(video_name))
    return int(match.group(1)) if match else None


def _parse_start_datetime_from_video_name(video_name, segment_seconds=None):
    """Parse start datetime from common BuzzSuite video name formats."""
    if segment_seconds is None or segment_seconds <= 0:
        segment_seconds = DEFAULT_VIDEO_SEGMENT_SECONDS

    clip_sequence_index = _extract_clip_sequence_index(video_name)
    version_index = _extract_version_index(video_name) if clip_sequence_index is None else None
    if clip_sequence_index is not None:
        version_offset_seconds = clip_sequence_index * int(segment_seconds)
    else:
        version_offset_seconds = max(0, (version_index or 1) - 1) * int(segment_seconds)

    # YYYYMMDD_HHMMSS
    m = re.search(r'(\d{8})_(\d{6})(?:_|$)', video_name)
    if m:
        ymd, hms = m.group(1), m.group(2)
        t = pd.Timestamp(
            int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]),
            int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
        )
        return t + pd.Timedelta(seconds=version_offset_seconds)

    # YYMMDD_HHMMSS
    m = re.search(r'(\d{6})_(\d{6})(?:_|$)', video_name)
    if m:
        ymd, hms = m.group(1), m.group(2)
        t = pd.Timestamp(
            2000 + int(ymd[0:2]), int(ymd[2:4]), int(ymd[4:6]),
            int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
        )
        return t + pd.Timedelta(seconds=version_offset_seconds)

    return None


def _filter_by_time_range(video_names, start_str, end_str, segment_seconds=None):
    """Return the subset of video_names whose parsed start datetime (via
    _parse_start_datetime_from_video_name) falls within [start_str, end_str], inclusive. Segments
    whose start cannot be parsed are excluded. A blank/unparseable bound is simply not applied."""
    try:
        start_ts = pd.Timestamp(start_str) if start_str and str(start_str).strip() else None
    except Exception:
        start_ts = None
    try:
        end_ts = pd.Timestamp(end_str) if end_str and str(end_str).strip() else None
    except Exception:
        end_ts = None
    result = []
    for name in video_names:
        ts = _parse_start_datetime_from_video_name(name, segment_seconds)
        if ts is None:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        result.append(name)
    return result


def _reanchor_index_to_video_start(df, parsed_start):
    """Shift a datetime index so its first timestamp matches parsed_start."""
    if df is None or len(df) == 0 or parsed_start is None:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    shifted = df.copy()
    current_start = shifted.index.min()
    if pd.isna(current_start):
        return shifted
    delta = parsed_start - current_start
    shifted.index = shifted.index + delta
    return shifted


def _concat_flight_metrics_data(frames, log_fn=None):
    """Concatenate non-empty DataFrames while tolerating invalid entries."""
    valid_frames = [
        frame for frame in frames
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not valid_frames:
        return pd.DataFrame()

    try:
        return pd.concat(valid_frames, ignore_index=True)
    except (TypeError, ValueError) as exc:
        if callable(log_fn):
            log_fn(f"[WARN] Failed to concatenate flight metrics data: {exc}")
        return pd.DataFrame()


def _safe_concat_dataframes(frames):
    """Concatenate DataFrames while ignoring None/invalid items."""
    valid_frames = [frame for frame in frames if isinstance(frame, pd.DataFrame)]
    if not valid_frames:
        return pd.DataFrame()
    try:
        return pd.concat(valid_frames)
    except Exception:
        return pd.DataFrame()


def _compute_total_activity_from_objects(video_data):
    """Reconstruct numb_mosquitos_flying/numb_mosquitos_resting directly from the raw per-track
    `objects` + `time_stamp`, for older-schema tracking files that never had `population_variables`
    or `flight_population_activity` attached. Confirmed live (2026-09-16, Bangkok/Aedes_aegypti_F):
    these files carry `individual_variables`/`resting_variables` instead and their raw `objects`
    data is fully valid (thousands of tracks, millions of state frames) -- tracking worked fine,
    this one derived summary attribute was simply never computed for them. Mirrors the frozen
    tracker's own per-frame headcount logic (single_video_analysis.py::
    extract_mosquito_population_variables: a count of state==1/state==0 tracks per frame, with no
    cage-zone restriction) but returns RAW per-frame rows, not pre-resampled -- the caller
    (process_video_files) already applies its own `.resample('1s', label='right').mean()` to
    whatever this returns, so aggregating here too would double-resample and shift every timestamp
    by up to 1 second relative to the label='right' convention the frozen tracker's own output uses.
    Returns None if `objects`/`time_stamp` aren't usable."""
    objects = getattr(video_data, 'objects', None)
    raw_timestamps = getattr(video_data, 'time_stamp', None)
    if not isinstance(objects, dict) or raw_timestamps is None:
        return None

    track_index = pd.to_datetime(raw_timestamps, errors='coerce')
    n = len(track_index)
    if n == 0:
        return None

    flying = np.zeros(n, dtype=float)
    resting = np.zeros(n, dtype=float)
    for obj in objects.values():
        states = obj.get('state', []) if isinstance(obj, dict) else []
        start = int(obj.get('start', 0)) if isinstance(obj, dict) else 0
        for i, state in enumerate(states):
            idx = start + i
            if idx < 0 or idx >= n:
                continue
            try:
                state_val = float(state)
            except Exception:
                continue
            if not np.isfinite(state_val):
                continue
            if abs(state_val - 1.0) < 1e-6:
                flying[idx] += 1.0
            elif abs(state_val - 0.0) < 1e-6:
                resting[idx] += 1.0

    valid_mask = ~pd.isna(track_index)
    if not valid_mask.any():
        return None

    return pd.DataFrame({'numb_mosquitos_flying': flying[valid_mask],
                          'numb_mosquitos_resting': resting[valid_mask]},
                         index=pd.DatetimeIndex(track_index[valid_mask]))


def _get_population_dataframe(video_data):
    """Return a normalized population DataFrame from modern or legacy track objects.

    Falls back to reconstructing flying/resting headcounts directly from the raw per-track
    `objects` + `time_stamp` (_compute_total_activity_from_objects) when neither
    `population_variables` nor `flight_population_activity` is attached -- see that function's
    docstring for why this is safe (the raw tracking data is confirmed valid, only this one
    derived summary was never computed). When cage-zone border settings are also available, the
    existing zone breakdown (_compute_side_state_counts_from_video_data) is merged in too, for
    experiments (e.g. BuzzPhono) that use the zone-specific columns."""
    population_data = getattr(video_data, 'population_variables', None)
    if population_data is None:
        population_data = getattr(video_data, 'flight_population_activity', None)

    used_fallback = False
    if population_data is None:
        population_data = _compute_total_activity_from_objects(video_data)
        used_fallback = population_data is not None

    if population_data is None:
        return pd.DataFrame()

    if not isinstance(population_data, pd.DataFrame):
        try:
            population_data = pd.DataFrame(population_data)
        except Exception:
            return pd.DataFrame()

    if population_data.empty:
        return pd.DataFrame()

    population_data = population_data.copy()

    if 'numb_mosquitos_hs' not in population_data.columns and 'numb_mosquitos_control' in population_data.columns:
        population_data['numb_mosquitos_hs'] = population_data['numb_mosquitos_control']

    for missing_col in ['numb_mosquitos_left_ctrl', 'numb_mosquitos_right_ctrl']:
        if missing_col not in population_data.columns:
            population_data[missing_col] = np.nan

    if used_fallback:
        try:
            _, side_flying_df = _compute_side_state_counts_from_video_data(video_data)
        except Exception:
            side_flying_df = None
        if side_flying_df is not None:
            floored = pd.DatetimeIndex(population_data.index).floor('s')
            aligned = side_flying_df.reindex(floored)
            for col in ('numb_mosquitos_sugar', 'numb_mosquitos_hs',
                        'numb_mosquitos_left_ctrl', 'numb_mosquitos_right_ctrl'):
                if col in aligned.columns:
                    population_data[col] = aligned[col].to_numpy()

    return population_data


def _compute_side_state_counts_from_video_data(video_data):
    objects = getattr(video_data, 'objects', None)
    raw_timestamps = getattr(video_data, 'time_stamp', None)
    settings = getattr(video_data, 'settings', None)

    if not isinstance(objects, dict) or raw_timestamps is None:
        return None, None

    if not isinstance(settings, dict):
        settings = {
            'sugar_border_points': getattr(video_data, 'sugar_border_points', []),
            'control_border_points': getattr(video_data, 'control_border_points', []),
            'square_4_border_points': getattr(video_data, 'square_4_border_points', []),
            'square_3_border_points': getattr(video_data, 'square_3_border_points', []),
        }

    track_index = pd.to_datetime(raw_timestamps, errors='coerce')
    if len(track_index) == 0:
        return None, None

    side_polygons = {
        'numb_mosquitos_sugar': settings.get('sugar_border_points', []),
        'numb_mosquitos_hs': settings.get('control_border_points', []),
        'numb_mosquitos_left_ctrl': settings.get('square_3_border_points', []),
        'numb_mosquitos_right_ctrl': settings.get('square_4_border_points', []),
    }

    polygon_arrays = {}
    for key, polygon in side_polygons.items():
        try:
            arr = np.array(polygon, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                polygon_arrays[key] = arr[:, :2]
        except Exception:
            continue

    if len(polygon_arrays) < 1:
        return None, None

    n = len(track_index)
    resting_counts = {k: np.zeros(n, dtype=float) for k in side_polygons.keys()}
    flying_counts = {k: np.zeros(n, dtype=float) for k in side_polygons.keys()}

    for obj in objects.values():
        coordinates = obj.get('coordinates', [])
        states = obj.get('state', [])
        start = int(obj.get('start', 0))
        max_len = min(len(coordinates), len(states))
        if max_len <= 0:
            continue

        for i in range(max_len):
            absolute_idx = start + i
            if absolute_idx < 0 or absolute_idx >= n:
                continue

            coord = coordinates[i]
            if coord is None or len(coord) < 2:
                continue

            x = float(coord[0])
            y = float(coord[1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            state = states[i]
            try:
                state_val = float(state)
            except Exception:
                continue

            if not np.isfinite(state_val):
                continue

            if abs(state_val - 0.0) < 1e-6:
                state_class = 0
            elif abs(state_val - 1.0) < 1e-6:
                state_class = 1
            else:
                continue

            matched_side = None
            best_dist = -np.inf
            for side_key, poly in polygon_arrays.items():
                # Use signed distance to choose the best-fitting side region when
                # polygons touch/overlap near region boundaries.
                dist = cv2.pointPolygonTest(poly, (x, y), True)
                if dist >= 0 and dist > best_dist:
                    best_dist = dist
                    matched_side = side_key

            if matched_side is None:
                continue

            if state_class == 0:
                resting_counts[matched_side][absolute_idx] += 1.0
            else:
                flying_counts[matched_side][absolute_idx] += 1.0

    valid_mask = ~pd.isna(track_index)
    valid_idx = pd.DatetimeIndex(track_index[valid_mask]).floor('s')
    if len(valid_idx) == 0:
        return None, None

    resting_df = pd.DataFrame({k: v[valid_mask] for k, v in resting_counts.items()}, index=valid_idx)
    flying_df = pd.DataFrame({k: v[valid_mask] for k, v in flying_counts.items()}, index=valid_idx)

    resting_df = resting_df.groupby(resting_df.index).sum(numeric_only=True)
    flying_df = flying_df.groupby(flying_df.index).sum(numeric_only=True)

    for side_key in side_polygons.keys():
        if side_key not in resting_df.columns:
            resting_df[side_key] = 0.0
        if side_key not in flying_df.columns:
            flying_df[side_key] = 0.0

    ordered_cols = [k for k in side_polygons.keys()]
    resting_df = resting_df[ordered_cols].astype(float)
    flying_df = flying_df[ordered_cols].astype(float)
    return resting_df, flying_df


def _compute_pixels_moved_from_video_data(video_data):
    """Total pixels moved per frame, summed across all tracks, counting only movement while flying.

    Net-new movement-magnitude activity metric (the frozen tracker's population_variables only has
    flying/resting COUNTS, never displacement). Walks the same raw per-track coordinates that
    tracking_export.py reads for its dx/dy/step_px columns: for each track, the step between two
    consecutive frames (hypot of the coordinate delta) is attributed to the later frame when BOTH
    frames are flying (state==1) with finite coords. Mirrors _compute_side_state_counts_from_video_data's
    access pattern/guards. Returns a per-frame pandas Series indexed by the raw frame timestamp (which
    the caller resamples to per-second), or None if the data is unusable."""
    objects = getattr(video_data, 'objects', None)
    raw_timestamps = getattr(video_data, 'time_stamp', None)
    if not isinstance(objects, dict) or raw_timestamps is None:
        return None

    track_index = pd.to_datetime(raw_timestamps, errors='coerce')
    n = len(track_index)
    if n == 0:
        return None

    pixels_per_frame = np.zeros(n, dtype=float)

    for obj in objects.values():
        coordinates = obj.get('coordinates', [])
        states = obj.get('state', [])
        start = int(obj.get('start', 0))
        max_len = min(len(coordinates), len(states))
        if max_len <= 0:
            continue

        prev_xy = None       # (x, y) of the previous frame, if it was a valid flying point
        for i in range(max_len):
            absolute_idx = start + i
            coord = coordinates[i]
            state = states[i]

            xy = None
            if coord is not None and len(coord) >= 2:
                x = float(coord[0])
                y = float(coord[1])
                if np.isfinite(x) and np.isfinite(y):
                    try:
                        is_flying = abs(float(state) - 1.0) < 1e-6
                    except Exception:
                        is_flying = False
                    if is_flying:
                        xy = (x, y)

            if xy is not None and prev_xy is not None and 0 <= absolute_idx < n:
                pixels_per_frame[absolute_idx] += float(np.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1]))
            prev_xy = xy   # None (invalid/resting) breaks the chain so no step spans a gap

    valid_mask = ~pd.isna(track_index)
    if not valid_mask.any():
        return None
    return pd.Series(pixels_per_frame[valid_mask], index=pd.DatetimeIndex(track_index[valid_mask]))


def delete_mac_metafiles(folder_path):
    """
    Delete Mac metafiles that start with '.' (e.g., .DS_Store, ._ files).

    Args:
        folder_path (str): The directory path to clean up.
    """
    if not os.path.exists(folder_path):
        return

    try:
        for item in os.listdir(folder_path):
            if item.startswith('.'):
                item_path = os.path.join(folder_path, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Warning: Could not delete {item_path}: {e}")
    except Exception as e:
        print(f"Warning: Error cleaning up metafiles in {folder_path}: {e}")


# Define the function at module level
def process_video_files(video_file, folder_analysis):
    """
    Process a single video file to extract necessary data.

    Args:
        video_file (str): The name of the video file to process (without extension).
        folder_analysis (str): The directory where analysis data is stored.

    Returns:
        tuple or None: Returns the processed data tuple or None if an error occurs.
    """
    final_data_path = os.path.join(folder_analysis, "final_tracking_data", f"forward_mosq_tracks_{video_file}")
    if not os.path.exists(final_data_path):
        print(f"File not found: {final_data_path}")
        return None

    # Try to load pickle with robust fallbacks for module name mismatches
    video_data = None
    try:
        with open(final_data_path, 'rb') as f:
            video_data = pickle.load(f)
    except (ModuleNotFoundError, ImportError) as e_mod:
        # Attempt to patch sys.modules for known numpy legacy paths and retry
        print(f"ModuleNotFoundError/ImportError while loading {final_data_path}: {e_mod}. Trying compatibility fallback...")
        _patch_numpy_compat()
        try:
            with open(final_data_path, 'rb') as f:
                video_data = pickle.load(f)
        except Exception as e_retry:
            # As a last resort, try a custom CompatUnpickler that remaps legacy module names
            try:
                import io
                class CompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # remap a few known legacy numpy module paths -> current equivalents
                        if module.startswith("numpy._core"):
                            module = module.replace("numpy._core", "numpy.core")
                        if module.startswith("numpy.core._multiarray_umath"):
                            module = module.replace("numpy.core._multiarray_umath", "numpy.core._multiarray_umath")
                        return super().find_class(module, name)

                with open(final_data_path, 'rb') as f:
                    f.seek(0)
                    video_data = CompatUnpickler(f).load()
            except Exception as e_compat:
                print(f"Compatibility unpickler failed for {final_data_path}: {e_compat}")
                return None
    except (OSError, pickle.PickleError) as e:
        # Log the error and return None on exception to prevent crashing
        print(f"Error loading {final_data_path}: {e}")
        return None
    except Exception as e_other:
        print(f"Unexpected error loading {final_data_path}: {e_other}")
        return None

    if video_data is None:
        print(f"No data loaded from {final_data_path}")
        return None

    try:
        population_data = _get_population_dataframe(video_data)
        if population_data.empty:
            print(f"Error processing data from {final_data_path}: population/activity data missing")
            return None

        if not isinstance(population_data.index, pd.DatetimeIndex):
            population_data.index = pd.to_datetime(population_data.index, errors='coerce')
        population_data = population_data[~pd.isna(population_data.index)]
        if population_data.empty:
            print(f"Error processing data from {final_data_path}: population/activity index is invalid")
            return None

        population_data = population_data.resample('1s', label='right').mean(numeric_only=True)

        # Net-new movement-magnitude metric: total pixels moved per second (summed across tracks,
        # flying-only). Resampled to match population_data's right-labeled 1s index, so it flows
        # through concatenation/pickling as just another population_data column. SUM (not mean) is
        # the right aggregation for a displacement total; missing seconds mean no flight movement -> 0.
        pixels_series = _compute_pixels_moved_from_video_data(video_data)
        if pixels_series is not None and len(pixels_series):
            pixels_per_second = pixels_series.resample('1s', label='right').sum()
            population_data['total_pixels_moved'] = pixels_per_second.reindex(
                population_data.index, fill_value=0.0)
        else:
            population_data['total_pixels_moved'] = 0.0

        # Derive speaker-side analysis from raw tracks as a dedicated dataset.
        speaker_side_data = pd.DataFrame(index=population_data.index)
        side_resting_df, side_flying_df = _compute_side_state_counts_from_video_data(video_data)
        if side_resting_df is not None and side_flying_df is not None:
            aligned_idx = pd.DatetimeIndex(population_data.index).floor('s')
            side_resting_df = side_resting_df.reindex(aligned_idx, fill_value=0.0)
            side_flying_df = side_flying_df.reindex(aligned_idx, fill_value=0.0)

            speaker_side_data['side_resting_speaker'] = side_resting_df['numb_mosquitos_sugar'].to_numpy()
            speaker_side_data['side_resting_non_speaker_1'] = side_resting_df['numb_mosquitos_hs'].to_numpy()
            speaker_side_data['side_resting_non_speaker_2'] = side_resting_df['numb_mosquitos_left_ctrl'].to_numpy()
            speaker_side_data['side_resting_non_speaker_3'] = side_resting_df['numb_mosquitos_right_ctrl'].to_numpy()

            speaker_side_data['side_flying_speaker'] = side_flying_df['numb_mosquitos_sugar'].to_numpy()
            speaker_side_data['side_flying_non_speaker_1'] = side_flying_df['numb_mosquitos_hs'].to_numpy()
            speaker_side_data['side_flying_non_speaker_2'] = side_flying_df['numb_mosquitos_left_ctrl'].to_numpy()
            speaker_side_data['side_flying_non_speaker_3'] = side_flying_df['numb_mosquitos_right_ctrl'].to_numpy()

        population_data['video'] = video_file  # add video identifier
        speaker_side_data['video'] = video_file

        individual_data = getattr(video_data, 'individual_variables', pd.DataFrame())
        if not isinstance(individual_data, pd.DataFrame):
            try:
                individual_data = pd.DataFrame(individual_data)
            except Exception:
                individual_data = pd.DataFrame()

        flight_metrics_data = video_data.flight_metrics_around_resting if hasattr(video_data, 'flight_metrics_around_resting') else pd.DataFrame()
        resting_data = getattr(video_data, 'resting_variables', pd.DataFrame())
        if not isinstance(resting_data, pd.DataFrame):
            resting_data = pd.DataFrame()

        segment_seconds_est = None
        try:
            ts = getattr(video_data, 'time_stamp', None)
            if ts is not None and len(ts) >= 2:
                ts_start = pd.to_datetime(ts[0])
                ts_end = pd.to_datetime(ts[-1])
                duration_sec = (ts_end - ts_start).total_seconds()
                if duration_sec > 0:
                    segment_seconds_est = int(round(duration_sec))
        except Exception:
            segment_seconds_est = None

        if not segment_seconds_est or segment_seconds_est <= 0:
            segment_seconds_est = DEFAULT_VIDEO_SEGMENT_SECONDS

        # Always use the nominal segment length for the clip-index offset, not the
        # actual measured duration of this segment (partial/last segments are shorter
        # and would produce wrong offsets when multiplied by the clip index).
        parsed_start = _parse_start_datetime_from_video_name(video_file, segment_seconds=DEFAULT_VIDEO_SEGMENT_SECONDS)
        if parsed_start is not None:
            population_data = _reanchor_index_to_video_start(population_data, parsed_start)
            if isinstance(individual_data, pd.DataFrame):
                individual_data = _reanchor_index_to_video_start(individual_data, parsed_start)
            if isinstance(resting_data, pd.DataFrame):
                resting_data = _reanchor_index_to_video_start(resting_data, parsed_start)
            if isinstance(flight_metrics_data, pd.DataFrame) and 'start_time' in flight_metrics_data.columns:
                try:
                    flight_metrics_data = flight_metrics_data.copy()
                    flight_metrics_data['start_time'] = pd.to_datetime(flight_metrics_data['start_time'], errors='coerce')
                    if flight_metrics_data['start_time'].notna().any():
                        base_start = flight_metrics_data['start_time'].min()
                        flight_metrics_data['start_time'] = flight_metrics_data['start_time'] + (parsed_start - base_start)
                except Exception:
                    pass

        start_time = video_data.time_stamp[0] if hasattr(video_data, 'time_stamp') and len(video_data.time_stamp) > 0 else None
        avg_fraction_flying = population_data['numb_mosquitos_flying'].mean() if 'numb_mosquitos_flying' in population_data.columns else None
        total_nb_tracks = len(video_data.objects.keys()) if hasattr(video_data, 'objects') else 0

        return (population_data, individual_data, resting_data, flight_metrics_data,
            [start_time, avg_fraction_flying, total_nb_tracks], speaker_side_data)
    except Exception as e:
        # Log the error and return None on processing error
        print(f"Error processing data from {final_data_path}: {e}")
        return None


def concatenate_and_save_experiment_data(folder_analysis, config=None, log_fn=None, force=False):
    """Concatenate every tracked segment for one experiment into analyzed_data.pkl (incl. the
    total_pixels_moved metric computed per-segment in process_video_files). Runs its own
    multiprocessing.Pool -- call off the Tk main thread. Module-level (not a method) so both the
    Section 2 tracking queue (BatchProcessingTabManager._run_queue) and the Experiment Dashboard
    (experiment_dashboard.py::_run_activity) can call it right after tracking finishes -- there is
    no longer a manual "Concatenate and Save Data" button; this always keeps analyzed_data.pkl in
    sync with final_tracking_data/.

    Skips the rebuild (unless `force=True`) when analyzed_data.pkl already exists and is not older
    than the newest final_tracking_data/ file -- this is what lets Section 2 and the Dashboard call
    this unconditionally after every tracking run without wastefully re-concatenating hundreds of
    segments that haven't changed. Known limitation: mtime can't see a code change to
    process_video_files itself, or an mtime-preserving file copy -- that's what `force=True` is for.

    Segments the user skipped (excluded_segments.json) are left out even if they have tracking
    output, and a change to that list also counts as "newer" so the data is rebuilt without them.

    Returns 'saved', 'up_to_date', 'no_data' or 'failed' (callers may ignore it).
    """
    log = log_fn or (lambda msg: None)

    final_tracking_folder = os.path.join(folder_analysis, "final_tracking_data")
    if not os.path.exists(final_tracking_folder):
        log(f"Tracking data folder not found: {final_tracking_folder}")
        return 'no_data'

    excluded = load_excluded_segments(folder_analysis)
    if not force:
        final_output_path = os.path.join(folder_analysis, "analyzed_data.pkl")
        try:
            existing_mtime = os.path.getmtime(final_output_path)
        except OSError:
            existing_mtime = None
        if existing_mtime is not None:
            from path_utils import newest_mtime
            newest = newest_mtime(final_tracking_folder, "forward_mosq_tracks_")
            try:
                newest = max(newest, os.path.getmtime(
                    os.path.join(folder_analysis, EXCLUDED_SEGMENTS_FILE)))
            except OSError:
                pass
            if existing_mtime >= newest:
                log("analyzed_data.pkl is already up to date -- skipping concatenation.")
                return 'up_to_date'

    video_files = []
    for f in os.listdir(final_tracking_folder):
        if f.startswith("forward_mosq_tracks_") and not f.endswith(('.png', '.csv', '.txt')):
            video_files.append(f.replace("forward_mosq_tracks_", ""))
    left_out = [v for v in video_files if v in excluded]
    if left_out:
        video_files = [v for v in video_files if v not in excluded]
        log(f"Leaving out {len(left_out)} skipped segment(s).")

    if not video_files:
        log("No tracking data found for concatenation.")
        return 'no_data'

    video_files.sort()
    log(f"Concatenating {len(video_files)} tracked segment(s)…")

    with pool_concurrency_slot() as concurrency:
        processes_to_use = _default_batch_worker_count(config, concurrency=concurrency)
        with Pool(processes=processes_to_use) as pool:
            results_raw = pool.starmap(process_video_files, [(video_name, folder_analysis) for video_name in video_files])

    successful_videos = [v for v, r in zip(video_files, results_raw) if r is not None]
    failed_videos = [v for v, r in zip(video_files, results_raw) if r is None]
    if failed_videos:
        log(f"{len(successful_videos)} succeeded, {len(failed_videos)} failed to concatenate.")

    results = [result for result in results_raw if result is not None]
    if not results:
        log("No data found for concatenation.")
        return 'failed'

    all_population_data, all_individual_data, all_resting_data, all_flight_metrics_data, summary_data, all_speaker_side_data = zip(*results)

    full_population_data = _safe_concat_dataframes(all_population_data)
    full_individual_data = _safe_concat_dataframes(all_individual_data)
    full_resting_data = _safe_concat_dataframes(all_resting_data)
    full_flight_metrics_data = _concat_flight_metrics_data(all_flight_metrics_data, log_fn=log)
    full_speaker_side_data = _safe_concat_dataframes(all_speaker_side_data)
    summary_df = pd.DataFrame(summary_data, columns=['start_time', 'avg_fraction_flying', 'total_nb_tracks'])

    final_output_path = os.path.join(folder_analysis, "analyzed_data.pkl")
    try:
        with open(final_output_path, 'wb') as f:
            pickle.dump({
                'population_data': full_population_data,
                'individual_data': full_individual_data,
                'resting_data': full_resting_data,
                'summary_data': summary_df,
                'flight_metrics_data': full_flight_metrics_data,
                'speaker_side_data': full_speaker_side_data,
                'speaker_side_mapping_version': SPEAKER_SIDE_MAPPING_VERSION,
                'concat_debug': {
                    'requested_videos': video_files,
                    'successful_videos': successful_videos,
                    'failed_videos': failed_videos,
                }
            }, f)
    except Exception as e:
        log(f"Failed to save data: {e}")
        return 'failed'

    log(f"Concatenation complete. Data saved to: {final_output_path}")
    return 'saved'


class BatchProcessingTabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.analyzed_data = None
        self.loaded_data_path = None
        self.data_load_mode = 'auto'
        self.last_speed_before = None
        self.last_speed_after = None

    def init_tracking_tab(self, tab):
        """'2 · Analysis' tab: pick what to track (specific segments / time range / all
        untracked) + raw video playback (folded in from the old Video Inspection tab) + a
        tracking queue with a verbose inline progress panel."""
        self.tab = tab
        self.tab.grid_rowconfigure(0, weight=1)
        self.tab.grid_columnconfigure(0, weight=0, minsize=320)
        self.tab.grid_columnconfigure(1, weight=1)
        self.tab.grid_columnconfigure(2, weight=1, minsize=320)
        self.create_tracking_controls()

    def create_tracking_controls(self):
        controls_frame = tk.Frame(self.tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        tk.Label(controls_frame, text="What to track", font=("TkDefaultFont", 11, "bold")
                 ).pack(side=tk.TOP, anchor="w")

        self.track_mode_var = tk.StringVar(value="all")
        modes_frame = tk.Frame(controls_frame)
        modes_frame.pack(side=tk.TOP, fill=tk.X, pady=(4, 4))
        tk.Radiobutton(modes_frame, text="All untracked (batch)", variable=self.track_mode_var,
                       value="all", command=self._on_track_mode_change).pack(anchor="w")
        tk.Radiobutton(modes_frame, text="Specific segments", variable=self.track_mode_var,
                       value="specific", command=self._on_track_mode_change).pack(anchor="w")
        tk.Radiobutton(modes_frame, text="Time range", variable=self.track_mode_var,
                       value="range", command=self._on_track_mode_change).pack(anchor="w")

        # Fixed-position holder so toggling which sub-frame is visible doesn't reorder the
        # widgets packed below it (worker/GPU/run controls).
        mode_holder = tk.Frame(controls_frame)
        mode_holder.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        self.segment_frame = tk.LabelFrame(mode_holder, text="Specific segments")
        self.segment_listbox = tk.Listbox(self.segment_frame, selectmode=tk.EXTENDED, width=32, height=8)
        self.segment_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        tk.Button(self.segment_frame, text="Refresh segment list",
                  command=self.update_segment_listbox).pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))

        self.range_frame = tk.LabelFrame(mode_holder, text="Time range")
        tk.Label(self.range_frame, text="Start").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.range_start_var = tk.StringVar(value="")
        self.range_start_combo = ttk.Combobox(self.range_frame, textvariable=self.range_start_var,
                                              width=20, state="readonly")
        self.range_start_combo.grid(row=0, column=1, padx=4, pady=2)
        tk.Label(self.range_frame, text="End").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.range_end_var = tk.StringVar(value="")
        self.range_end_combo = ttk.Combobox(self.range_frame, textvariable=self.range_end_var,
                                            width=20, state="readonly")
        self.range_end_combo.grid(row=1, column=1, padx=4, pady=2)
        tk.Button(self.range_frame, text="Refresh date range",
                  command=self.update_range_options).grid(row=2, column=0, columnspan=2,
                                                            sticky="ew", padx=4, pady=(4, 4))

        self._on_track_mode_change()

        # Worker count selector
        worker_frame = tk.Frame(controls_frame)
        worker_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)
        tk.Label(worker_frame, text="Worker Processes:").pack(side=tk.LEFT, padx=5)

        # Prefer an adaptive default that uses most available CPU cores without oversubscribing.
        default_workers = _default_batch_worker_count(self.experiment_manager.config)
        self.worker_count_var = tk.IntVar(value=default_workers)
        self.worker_spinbox = tk.Spinbox(worker_frame, from_=1, to=32, textvariable=self.worker_count_var, width=5)
        self.worker_spinbox.pack(side=tk.LEFT, padx=5)
        tk.Label(worker_frame, text="(1-32)").pack(side=tk.LEFT, padx=5)
        add_tooltip(self.worker_spinbox,
                    "How many videos to track in parallel (one CPU process each). More = faster but "
                    "heavier on RAM/CPU. A good default is (number of CPU cores − 1).")

        # The overhauled tracking config (WS1+WS3+WS4+WS5) is now the only tracking path
        # (no old-vs-new toggle). ExperimentManager defaults use_tracking_overhaul=True and
        # apply_tracking_overhaul injects OVERHAUL_TRACKING into the in-memory settings at run
        # time, never the YAML. Kept as a single source of truth here for clarity.
        self.experiment_manager.use_tracking_overhaul = True

        # (There used to be a "Use GPU (OpenCL)" checkbox here. It was removed: benchmarked on real
        # data it computed byte-identical results but ran ~1.9x SLOWER than plain CPU, because these
        # per-frame ops are too cheap at cage-video resolutions to pay for the host<->device
        # transfer. See DEVLOG 2026-07-11 — the vision ops now call cv2 directly.)

        self._create_button(controls_frame, "Run tracking", self._on_run_tracking_clicked)

        # Optional post-tracking data prep (speed filter only -- concatenation into
        # analyzed_data.pkl is now automatic, right after "Run tracking"/the queue finishes; see
        # concatenate_and_save_experiment_data). Re-homed here from the old Activity Plotting tab's
        # column, removed in the 2026-07-15 Plotting redesign. build_plot_section is reused
        # verbatim (now trimmed to the speed filter only).
        prep_frame = tk.LabelFrame(controls_frame, text="Speed filter (optional)")
        prep_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 4))
        self.build_plot_section(prep_frame)

        # Raw playback panel folds in here (replaces the standalone Video Inspection tab).
        playback_frame = tk.Frame(self.tab)
        playback_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.ui_manager.video_tab_manager.build_playback_section(playback_frame)

        # Tracking queue + verbose inline progress panel (workflow redesign, Section 2).
        queue_panel_frame = tk.Frame(self.tab)
        queue_panel_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self._build_tracking_queue_panel(queue_panel_frame)

    def _build_tracking_queue_panel(self, parent):
        """An experiment tracking queue (Add to queue / Run queue / Cancel / Clear) plus a
        non-blocking, scrolling verbose log + progress bar showing exactly what's being tracked
        and what's finished. "Run tracking" enqueues the current experiment/scope and runs the
        queue immediately (a single run is just a one-item queue); "Add to queue" only enqueues,
        so several experiments' scopes can be queued before running them all together."""
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.queue_items = []       # dicts: folder_analysis/folder_videos/alias/scope_label/video_names
        self._queue_cancel = None   # threading.Event while a queue run is active
        self._queue_running = False
        self._queue_current_pool = None   # the currently-executing item's Pool, for Cancel to terminate

        queue_frame = tk.LabelFrame(parent, text="Tracking queue")
        queue_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        btn_row = tk.Frame(queue_frame)
        btn_row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))
        add_queue_btn = tk.Button(btn_row, text="Add to queue", command=self._on_add_to_queue_clicked)
        add_queue_btn.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(add_queue_btn, "Queue the current experiment with the scope selected on the "
                    "left, without running it yet. Add more experiments' scopes (open a different "
                    "one first), then Run queue to track them all in sequence.")
        run_queue_btn = tk.Button(btn_row, text="Run queue", command=self._on_run_queue_clicked)
        run_queue_btn.pack(side=tk.LEFT, padx=4)
        add_tooltip(run_queue_btn, "Track every queued experiment/scope in sequence, printing "
                    "progress below.")
        cancel_queue_btn = tk.Button(btn_row, text="Cancel", command=self._on_cancel_queue_clicked)
        cancel_queue_btn.pack(side=tk.LEFT, padx=4)
        add_tooltip(cancel_queue_btn, "Stop the running queue: terminates the current experiment's "
                    "in-progress tracking right away and skips any experiments still queued after it.")
        clear_queue_btn = tk.Button(btn_row, text="Clear", command=self._on_clear_queue_clicked)
        clear_queue_btn.pack(side=tk.LEFT, padx=4)
        add_tooltip(clear_queue_btn, "Remove all not-yet-run items from the queue.")

        self.queue_listbox = tk.Listbox(queue_frame, height=4)
        self.queue_listbox.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))

        progress_frame = tk.LabelFrame(parent, text="Tracking progress")
        progress_frame.grid(row=1, column=0, sticky="nsew")
        progress_frame.grid_rowconfigure(1, weight=1)
        progress_frame.grid_columnconfigure(0, weight=1)
        self.queue_progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.queue_progress.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))

        log_row = tk.Frame(progress_frame)
        log_row.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        log_row.grid_rowconfigure(0, weight=1)
        log_row.grid_columnconfigure(0, weight=1)
        self.queue_log = tk.Text(log_row, height=12, width=34, state=tk.DISABLED, wrap=tk.WORD)
        self.queue_log.grid(row=0, column=0, sticky="nsew")
        qlog_scroll = tk.Scrollbar(log_row, command=self.queue_log.yview)
        qlog_scroll.grid(row=0, column=1, sticky="ns")
        self.queue_log.configure(yscrollcommand=qlog_scroll.set)

    # ------------------------------------------------------------- tracking queue
    def _resolve_video_names_for_current_mode(self):
        """Video names + a human scope label for whichever scope radio is currently selected, or
        None if the selection is invalid (an error has already been shown to the user). Shared by
        "Add to queue" so enqueuing uses exactly the same scope resolution as the old immediate
        run_tracking_all_untracked/_specific/_time_range handlers did."""
        mode = self.track_mode_var.get()
        if mode == "specific":
            selection = self.segment_listbox.curselection()
            if not selection:
                self.ui_manager.show_error("Batch processing",
                                           "No segments selected. Pick one or more from the list.")
                return None
            video_names = [self.segment_listbox.get(i) for i in selection]
            # Picking a previously skipped segment by hand is an explicit request to track it.
            folder_analysis = self.experiment_manager.folder_analysis
            unskip = [v for v in video_names if v in load_excluded_segments(folder_analysis)]
            for v in unskip:
                set_segment_excluded(folder_analysis, v, excluded=False)
            if unskip:
                self.log(f"Tracking {len(unskip)} segment(s) you had skipped, since you picked them.")
            return video_names, "Single video / segment(s)"
        elif mode == "range":
            all_names = self._list_all_video_names()
            video_names = _filter_by_time_range(all_names, self.range_start_var.get(), self.range_end_var.get())
            if not video_names:
                self.ui_manager.show_error("Batch processing",
                                           "No segments fall within the selected time range.")
                return None
            return video_names, "Time range"
        else:
            video_names = self._list_all_video_names()
            if not video_names:
                self.ui_manager.show_error(
                    "Batch processing",
                    "No .mp4 videos were found for this experiment.\n\n"
                    f"Video folder:\n{self.experiment_manager.folder_videos}\n\n"
                    "Check that the folder exists, the drive is mounted, and it contains "
                    "converted .mp4 segments (use the H264 → MP4 Converter if needed).")
                return None
            folder_analysis = self.experiment_manager.folder_analysis
            excluded = load_excluded_segments(folder_analysis)
            pending = [v for v in video_names
                       if not _should_skip_video(v, folder_analysis) and v not in excluded]
            skipped = [v for v in video_names if v not in pending and v not in excluded]
            if skipped:
                self.log(f"Skipping {len(skipped)} already processed video(s).")
            n_excluded = sum(1 for v in video_names if v in excluded)
            if n_excluded:
                self.log(f"Leaving out {n_excluded} segment(s) you chose to skip.")
            if not pending:
                # Every segment is already tracked -- still queue this experiment with an empty
                # video list so the run reaches concatenate_and_save_experiment_data (which itself
                # skips the rebuild if analyzed_data.pkl is already up to date). Without this, a
                # fully-tracked experiment whose analyzed_data.pkl is missing or stale (e.g. it was
                # never generated, or a segment was re-tracked outside Section 2) had NO way in the
                # GUI to (re)build it -- the run had nothing to enqueue and stopped here.
                self.log("All videos are already processed — will rebuild analyzed_data.pkl if needed.")
                return [], "Whole experiment (already tracked — rebuild data only)"
            return pending, "Whole experiment"

    def _refresh_queue_listbox(self):
        self.queue_listbox.delete(0, tk.END)
        for item in self.queue_items:
            self.queue_listbox.insert(
                tk.END, f"{item['alias']} — {item['scope_label']} ({len(item['video_names'])} video(s))")

    def _progress_log(self, message):
        """Append one line to the verbose queue-progress panel. Main-thread only — callers running
        off-thread marshal via root.after()."""
        self.queue_log.configure(state=tk.NORMAL)
        self.queue_log.insert(tk.END, message + "\n")
        self.queue_log.see(tk.END)
        self.queue_log.configure(state=tk.DISABLED)

    def _add_to_queue(self):
        """Enqueue the currently-loaded experiment with the currently-selected scope. Returns True
        if something was queued, False otherwise (an error/info message was already shown/logged)."""
        if not self.experiment_manager.experiment:
            self.ui_manager.show_error(
                "Batch processing",
                "No experiment is loaded. Open or create an experiment "
                "(1 · Setup section) before running batch tracking.")
            return False
        resolved = self._resolve_video_names_for_current_mode()
        if resolved is None:
            return False
        video_names, scope_label = resolved

        folder_analysis = self.experiment_manager.folder_analysis
        folder_videos = self.experiment_manager.folder_videos
        delete_mac_metafiles(folder_videos)
        delete_mac_metafiles(folder_analysis)

        # Background-image precondition (the #1 setup trap): the tracker SILENTLY skips any
        # segment lacking its Initialization background image images_mortality/<video>.png. Ask
        # up front, on the main thread (this is a button click), same prompt run_tracking() used
        # to show interactively — so the user isn't surprised later when the queue just skips them.
        mort_dir = os.path.join(folder_analysis, "images_mortality")
        missing_bg = [v for v in video_names if not os.path.isfile(os.path.join(mort_dir, v + ".png"))]
        if missing_bg:
            examples = ", ".join(missing_bg[:3]) + (" ..." if len(missing_bg) > 3 else "")
            ready = len(video_names) - len(missing_bg)
            if ready == 0:
                run_now = messagebox.askyesno(
                    "Initialization required",
                    "None of the {n} selected segment(s) have a background image in\n"
                    "images_mortality/. The tracker needs one per segment and will\n"
                    "SILENTLY SKIP any segment without one — so queuing this scope would\n"
                    "add nothing.\n\n"
                    "Run Initialization now to compute the background image for each\n"
                    "segment? This runs in the background; add to the queue again once\n"
                    "it finishes.\n\n"
                    "Missing e.g.: {ex}".format(n=len(video_names), ex=examples),
                    parent=self.root)
                if run_now:
                    self._start_initialization()
                else:
                    self.log("Add to queue cancelled: no segments have background images "
                             "(run the 1 · Setup section's image/background extraction first).")
                return False
            proceed = messagebox.askyesno(
                "Some segments not initialized",
                "{m} of {n} selected segment(s) have NO background image in "
                "images_mortality/ and will be SILENTLY SKIPPED by the tracker.\n\n"
                "Missing e.g.: {ex}\n\n"
                "Queue the {r} ready segment(s) anyway?".format(
                    m=len(missing_bg), n=len(video_names), ex=examples, r=ready),
                parent=self.root)
            if not proceed:
                self.log("Add to queue cancelled — initialize the missing segments first.")
                return False
            video_names = [v for v in video_names if v not in missing_bg]

        alias = (getattr(self.experiment_manager, 'experiment_alias', None)
                or os.path.basename(os.path.normpath(folder_analysis)))
        self.queue_items.append({
            'folder_analysis': folder_analysis, 'folder_videos': folder_videos,
            'alias': alias, 'scope_label': scope_label, 'video_names': list(video_names),
        })
        self._refresh_queue_listbox()
        self._progress_log(f"Added to queue: {alias} — {scope_label} ({len(video_names)} video(s)).")
        return True

    def _on_add_to_queue_clicked(self):
        self._add_to_queue()

    def _on_run_tracking_clicked(self):
        # A "Run tracking" click enqueues the current experiment/scope and runs the queue right
        # away — running a single experiment now is just a one-item queue.
        if not self._add_to_queue():
            return
        if not self._queue_running:
            self._run_queue()

    def _on_run_queue_clicked(self):
        if self._queue_running:
            self.log("A tracking queue is already running.")
            return
        if not self.queue_items:
            self.log("Tracking queue is empty — use \"Add to queue\" first.")
            return
        self._run_queue()

    def _on_cancel_queue_clicked(self):
        if self._queue_cancel is not None:
            self._queue_cancel.set()
            self._progress_log("Cancelling…")
            # Terminate the currently-running experiment's Pool immediately (mirrors the
            # Experiment Dashboard's _cancel) -- without this, imap_unordered has already
            # dispatched every video in this experiment to a worker, so merely setting the
            # cancel Event would only stop the *next queued experiment* from starting while the
            # current one's whole batch ran to completion regardless.
            pool = self._queue_current_pool
            if pool is not None:
                try:
                    pool.terminate()
                except Exception:
                    pass

    def _on_clear_queue_clicked(self):
        if self._queue_running:
            self.ui_manager.show_error("Batch processing",
                                       "Cannot clear the queue while it is running. Cancel first.")
            return
        self.queue_items = []
        self._refresh_queue_listbox()

    def _run_queue(self):
        """Run every queued experiment/scope in sequence on a background thread, via the shared
        batch_runner engine (the same one the Experiment Dashboard uses), printing a verbose,
        per-experiment/per-video log. Non-blocking: the rest of the app stays usable while this
        runs; only this queue's own Run/Add/Clear are guarded against overlap."""
        if self._queue_running or not self.queue_items:
            return
        items = list(self.queue_items)
        self.queue_items = []
        self._refresh_queue_listbox()
        self._queue_running = True
        cancel = threading.Event()
        self._queue_cancel = cancel
        self.queue_progress['value'] = 0
        n_experiments = len(items)

        # Snapshot the "Worker Processes" spinbox now, on the main thread, before worker() (which
        # runs on a background Thread) is ever dispatched -- this was previously read nowhere at
        # all: run_tracking_job_for_folder always passed None for its worker-count override, so
        # this control had no effect regardless of what the user set it to, and every tracking run
        # silently used the auto-computed default instead (on a 32-core/48GB machine, that default
        # picked 30 parallel workers, which turned out to be enough simultaneous heavy segments to
        # exhaust system memory and crash the whole run -- see DEVLOG). Reading the IntVar here
        # (not inside worker()) keeps the Tk access on the main thread where it belongs.
        try:
            requested_workers = int(self.worker_count_var.get())
            if requested_workers < 1:
                requested_workers = None
        except Exception:
            requested_workers = None

        def worker():
            from batch_runner import run_tracking_job_for_folder
            cfg_mgr = getattr(self.experiment_manager, 'config_manager', None)

            for exp_idx, item in enumerate(items, start=1):
                if cancel.is_set():
                    break
                alias = item['alias']
                total_videos = len(item['video_names'])
                self.root.after(0, lambda a=alias, n=total_videos:
                                self._progress_log(f"▶ Experiment {a}: {n} videos to track"))

                def on_step(text, a=alias):
                    self.root.after(0, lambda: self._progress_log(f"   … {text}"))

                def on_video_done(result, idx, total, a=alias, ei=exp_idx):
                    name = result.get('video_name', '?') if isinstance(result, dict) else '?'
                    ok = isinstance(result, dict) and result.get('ok')
                    mark = "✓" if ok else "✗"
                    reason = ""
                    if not ok and isinstance(result, dict) and result.get('error'):
                        reason = " — " + str(result['error']).replace("\n", " ")[:180]

                    def _update():
                        self._progress_log(f"   {mark} {name} ({idx}/{total}){reason}")
                        frac = ((ei - 1) + idx / max(1, total)) / max(1, n_experiments) * 100
                        self.queue_progress['value'] = frac
                    self.root.after(0, _update)

                def on_pool(pool):
                    item['_pool'] = pool
                    # Tracked so Cancel can terminate whichever Pool is currently running (a plain
                    # attribute write is fine here — no Tk touched, and CPython's GIL makes this
                    # single assignment safe to read from the main thread in _on_cancel_queue_clicked).
                    self._queue_current_pool = pool

                try:
                    result = run_tracking_job_for_folder(
                        item['folder_analysis'], cfg_mgr, video_names=item['video_names'],
                        on_step=on_step, on_video_done=on_video_done, on_pool=on_pool, cancel=cancel,
                        requested_workers=requested_workers)
                except Exception as exc:
                    # Mirrors the Experiment Dashboard's per-analysis try/except (_run_job): one
                    # experiment failing (including a Cancel-triggered pool.terminate() raising out
                    # of imap_unordered mid-iteration) must not strand the whole queue — log it and
                    # move on to the next queued experiment, so _finish() below always still runs
                    # and _queue_running is always cleared.
                    self.root.after(0, lambda a=alias, e=exc: self._progress_log(f"✗ {a} failed: {e}"))
                    continue

                def _report_done(a=alias, r=result):
                    if r['total']:
                        self._progress_log(f"✔ {a} complete ({r['done']}/{r['total']})")
                    else:
                        self._progress_log(f"✔ {a}: nothing to track")
                self.root.after(0, _report_done)

                # Auto-concatenate: keep analyzed_data.pkl (incl. total_pixels_moved) in sync with
                # final_tracking_data/ right after each experiment finishes — no manual button.
                # Own try/except (mirrors the tracking call above): a concatenation failure on one
                # queued experiment must not crash worker() and strand _queue_running as True
                # forever (_finish() below would never run).
                def concat_log(msg, a=alias):
                    self.root.after(0, lambda m=msg: self._progress_log(f"   [{a}] {m}"))
                try:
                    concatenate_and_save_experiment_data(
                        item['folder_analysis'], config=self.experiment_manager.config, log_fn=concat_log)
                except Exception as exc:
                    self.root.after(0, lambda a=alias, e=exc:
                                    self._progress_log(f"✗ {a}: concatenation failed: {e}"))

            def _finish():
                self._queue_running = False
                self._queue_cancel = None
                if cancel.is_set():
                    self._progress_log("Queue cancelled.")
                else:
                    self._progress_log("All queued experiments tracked.")
                    self.queue_progress['value'] = 100
            self.root.after(0, _finish)

        Thread(target=worker, daemon=True).start()

    def build_plot_section(self, parent):
        """Speed-filter data-prep controls, built into `parent`. Called by the tracking tab.

        "Concatenate and Save Data" used to be a button here; it is now automatic — the tracking
        queue (_run_queue below) and the Experiment Dashboard both call
        concatenate_and_save_experiment_data() right after each experiment finishes tracking, so
        analyzed_data.pkl is always kept in sync with final_tracking_data/ with no manual step."""
        speed_filter_frame = tk.Frame(parent)
        speed_filter_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)

        tk.Label(speed_filter_frame, text="Speed Filter:").pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(speed_filter_frame, text="Min").pack(side=tk.LEFT)
        self.min_speed_var = tk.StringVar(value="1.0")
        self.min_speed_entry = tk.Entry(speed_filter_frame, textvariable=self.min_speed_var, width=6)
        self.min_speed_entry.pack(side=tk.LEFT, padx=(2, 8))
        add_tooltip(self.min_speed_entry,
                    "Lower speed cut-off (pixels/frame): slower track points are treated as "
                    "resting/jitter and excluded from flight metrics. Typical: 1.0.")

        tk.Label(speed_filter_frame, text="Max").pack(side=tk.LEFT)
        self.max_speed_var = tk.StringVar(value="40.0")
        self.max_speed_entry = tk.Entry(speed_filter_frame, textvariable=self.max_speed_var, width=6)
        self.max_speed_entry.pack(side=tk.LEFT, padx=(2, 8))
        add_tooltip(self.max_speed_entry,
                    "Upper speed cut-off (pixels/frame): faster points are treated as tracking "
                    "errors and excluded. Typical: 40.0.")

        self.auto_speed_detect_var = tk.IntVar(value=1)
        auto_speed_check = tk.Checkbutton(speed_filter_frame, text="Auto Detect", variable=self.auto_speed_detect_var)
        auto_speed_check.pack(side=tk.LEFT, padx=(2, 2))
        add_tooltip(auto_speed_check,
                    "When ticked, the speed filter bounds are estimated from the data instead of "
                    "using the Min/Max boxes above.")

        self.speed_filter_status_label = tk.Label(parent, text="Speed filter active: auto (default 1.0 to 40.0)", fg='gray')
        self.speed_filter_status_label.pack(side=tk.TOP, pady=(0, 5), anchor='w')

        rerun_filter_button = self._create_button(
            parent, "Run Speed Filter",
            lambda: self._run_off_thread(self.run_speed_filter, "Run Speed Filter"))
        add_tooltip(rerun_filter_button,
                    "Filter tracked flights by speed range (auto-detected by default, or the "
                    "Min/Max above) and save analyzed_data_filtered.pkl / "
                    "analyzed_data_speed_filtered.pkl. Runs in the background.")

        # Data prep only. The old aggregate-plot buttons (Rolling Average / Flying-Resting-Combined /
        # Summary / Histograms / Scatter-Bar) and the three Load-Data buttons were removed in the
        # 2026-07-15 Plotting redesign: plotting now lives in ActivityPlotManager
        # (activity_plot_manager.py), driven by an explicit experiment/date picker. The load_data*
        # methods are kept (apply_speed_filter still calls load_data() internally); export_rolling_
        # data_to_csv is kept unwired as an Export-tab seed alongside analysis_tab_manager's exports.
        return parent

    def _create_button(self, parent, text, command):
        button = tk.Button(parent, text=text, command=command)
        button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)
        return button

    # ------------------------------------------------------- background execution
    def _run_off_thread(self, work, label):
        """Run one long, Tk-free operation on a background thread so the UI stays responsive.

        These handlers used to run synchronously in the button callback, i.e. ON the Tk main
        thread — `batch_speed_filter` (and, before it was moved off the button entirely,
        concatenation) each drive a full multiprocessing Pool over *every* tracked segment, so on
        a 100+ segment experiment the whole app froze (unresponsive window, no repaint) for minutes.

        Only the heavy DATA work is routed through here (Pools, pickle load/save, CSV export); each
        function was audited to touch no Tk widget directly. The `plot_*` handlers deliberately do
        NOT use this: with matplotlib's TkAgg backend, figure creation/drawing must stay on the main
        thread, so threading them would trade a freeze for a crash.

        `self.log` and `ui_manager.show_error` are temporarily swapped for marshalling versions, so
        a log line or error dialog raised from inside the worker is re-dispatched onto the main
        thread via `root.after()` rather than touching Tk off-thread.
        """
        if getattr(self, '_bg_busy', False):
            self.ui_manager.show_error(
                label, "Another long operation is still running. Wait for it to finish first.")
            return

        real_log = self.log
        real_show_error = self.ui_manager.show_error

        def safe_log(msg):
            self.root.after(0, lambda m=msg: real_log(m))

        def safe_show_error(title, message):
            self.root.after(0, lambda t=title, m=message: real_show_error(t, m))

        def runner():
            try:
                work()
            except Exception as exc:
                safe_log("%s failed: %s" % (label, exc))
            finally:
                def restore():
                    self.log = real_log
                    self.ui_manager.show_error = real_show_error
                    self._bg_busy = False
                    self.ui_manager.set_status(action="%s: finished" % label, busy=False)
                self.root.after(0, restore)

        self._bg_busy = True
        self.log = safe_log
        self.ui_manager.show_error = safe_show_error
        self.ui_manager.set_status(action="%s: running…" % label, busy=True)
        real_log("%s: started (running in the background — the app stays usable)." % label)
        Thread(target=runner, daemon=True).start()

    # ------------------------------------------------------------- selection modes
    def _on_track_mode_change(self):
        mode = self.track_mode_var.get()
        self.segment_frame.pack_forget()
        self.range_frame.pack_forget()
        if mode == "specific":
            self.segment_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.update_segment_listbox()
        elif mode == "range":
            self.range_frame.pack(side=tk.TOP, fill=tk.X)
            self.update_range_options()

    def _list_all_video_names(self):
        """All .mp4 segment names for the loaded experiment (video_names resolution shared by
        every selection mode)."""
        video_names = []
        try:
            if hasattr(self.experiment_manager.experiment, 'list_video_name'):
                video_names = list(self.experiment_manager.experiment.list_video_name)
        except Exception:
            video_names = []
        if not video_names and self.experiment_manager.folder_videos:
            try:
                video_names = [os.path.splitext(f)[0] for f in os.listdir(self.experiment_manager.folder_videos)
                               if f.endswith('.mp4') and not f.startswith('.')]
                video_names.sort()
            except Exception:
                video_names = []
        return video_names

    def update_segment_listbox(self):
        self.segment_listbox.delete(0, tk.END)
        for name in self._list_all_video_names():
            self.segment_listbox.insert(tk.END, name)

    def update_range_options(self):
        names = self._list_all_video_names()
        timestamps = sorted(set(
            ts for ts in (_parse_start_datetime_from_video_name(n) for n in names) if ts is not None))
        values = [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps]
        self.range_start_combo['values'] = values
        self.range_end_combo['values'] = values
        if values and not self.range_start_var.get():
            self.range_start_var.set(values[0])
        if values and not self.range_end_var.get():
            self.range_end_var.set(values[-1])

    def _start_initialization(self):
        """Run the full Initialization (extract one reference frame per segment, then compute the
        per-segment background image) off the main thread, so the batch gate's 'Run Initialization
        now?' offer does not freeze the UI. Both steps skip files that already exist
        (force_rerun=0), so this only does the missing work, then the user re-runs batch tracking.
        Result-neutral: exactly what the Setup tab's "Extract Images from Video" / "Get
        Background from Images" buttons do, just chained here."""
        em = self.experiment_manager
        if not getattr(em, 'experiment', None):
            self.ui_manager.show_error("Initialization", "No experiment is loaded.")
            return
        self.log("Initialization started — extracting a reference frame and computing a background "
                 "image per segment. This runs in the background; start batch processing again when "
                 "it finishes.")

        def safe_log(msg):
            # Marshal every log line back to the main thread (this worker is off it).
            self.root.after(0, lambda m=msg: self.log(m))

        def worker():
            saved_em_log = em.log
            saved_exp_log = getattr(em.experiment, 'log', None)
            em.log = safe_log
            try:
                em.experiment.log = safe_log
            except Exception:
                pass
            try:
                em.get_images_from_video(0)      # -> individual_images/ (skips existing)
                em.get_background_from_images(0)  # -> images_mortality/  (skips existing)
                safe_log("Initialization finished. Click \"Start batch processing\" again to track "
                         "the segments.")
            except Exception as exc:
                safe_log("Initialization failed: {0}".format(exc))
            finally:
                em.log = saved_em_log
                try:
                    em.experiment.log = saved_exp_log
                except Exception:
                    pass

        Thread(target=worker, daemon=True).start()

    def load_data(self):
        """Auto-load data preferring speed-filtered files when available."""
        self.data_load_mode = 'auto'
        self._load_data_by_mode()

    def load_data_normal(self):
        """Load only unfiltered analyzed_data.pkl."""
        self.data_load_mode = 'normal'
        self._load_data_by_mode()

    def load_data_filtered(self):
        """Load speed-filtered analyzed data (canonical/legacy)."""
        self.data_load_mode = 'filtered'
        self._load_data_by_mode()

    def _load_data_by_mode(self):
        folder_analysis = self.experiment_manager.folder_analysis
        if not folder_analysis or not os.path.isdir(folder_analysis):
            self.ui_manager.show_error(
                "Load data",
                "No experiment is loaded (or its analysis folder is missing). "
                "Open or create an experiment first, then try again.")
            return
        try:
            self._load_data_by_mode_impl(folder_analysis)
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            self.ui_manager.show_error(
                "Load data",
                "The saved analysis file appears to be corrupt or incomplete:\n"
                f"{getattr(self, 'loaded_data_path', '') or folder_analysis}\n\n{exc}\n\n"
                "Re-run batch processing / merge to regenerate it.")
        except Exception as exc:
            self.ui_manager.show_error("Load data", f"Could not load analysis data:\n{exc}")

    def _load_data_by_mode_impl(self, folder_analysis):
        filtered_path = os.path.join(folder_analysis, "analyzed_data_filtered.pkl")
        legacy_filtered_path = os.path.join(folder_analysis, "analyzed_data_speed_filtered.pkl")
        default_path = os.path.join(folder_analysis, "analyzed_data.pkl")

        mode = getattr(self, 'data_load_mode', 'auto')

        if mode == 'normal':
            if os.path.exists(default_path):
                with open(default_path, 'rb') as f:
                    self.analyzed_data = pickle.load(f)
                self.loaded_data_path = default_path
                self.log("Normal data successfully loaded.")
            else:
                self.log("Normal data file not found (analyzed_data.pkl).")
            return

        if mode == 'filtered':
            if os.path.exists(filtered_path):
                with open(filtered_path, 'rb') as f:
                    self.analyzed_data = pickle.load(f)
                self.loaded_data_path = filtered_path
                self.log("Speed-filtered data successfully loaded.")
            elif os.path.exists(legacy_filtered_path):
                with open(legacy_filtered_path, 'rb') as f:
                    self.analyzed_data = pickle.load(f)
                self.loaded_data_path = legacy_filtered_path
                self.log("Speed-filtered data successfully loaded.")
            else:
                self.log("Speed-filtered data file not found.")
            return

        if os.path.exists(filtered_path):
            with open(filtered_path, 'rb') as f:
                self.analyzed_data = pickle.load(f)
            self.loaded_data_path = filtered_path
            self.log("Speed-filtered data successfully loaded.")
        elif os.path.exists(legacy_filtered_path):
            with open(legacy_filtered_path, 'rb') as f:
                self.analyzed_data = pickle.load(f)
            self.loaded_data_path = legacy_filtered_path
            self.log("Speed-filtered data successfully loaded.")
        elif os.path.exists(default_path):
            with open(default_path, 'rb') as f:
                self.analyzed_data = pickle.load(f)
            self.loaded_data_path = default_path
            self.log("Data successfully loaded.")
        else:
            self.log("Data files not found. Please merge the data first.")

    def _ensure_preferred_plot_data_loaded(self):
        folder_analysis = self.experiment_manager.folder_analysis
        if not folder_analysis:
            return

        mode = getattr(self, 'data_load_mode', 'auto')
        default_path = os.path.join(folder_analysis, "analyzed_data.pkl")
        filtered_path = os.path.join(folder_analysis, "analyzed_data_filtered.pkl")
        legacy_filtered_path = os.path.join(folder_analysis, "analyzed_data_speed_filtered.pkl")
        preferred_path = None

        if mode == 'normal':
            if os.path.exists(default_path):
                preferred_path = default_path
        elif mode == 'filtered':
            if os.path.exists(filtered_path):
                preferred_path = filtered_path
            elif os.path.exists(legacy_filtered_path):
                preferred_path = legacy_filtered_path
        else:
            if os.path.exists(filtered_path):
                preferred_path = filtered_path
            elif os.path.exists(legacy_filtered_path):
                preferred_path = legacy_filtered_path
            elif os.path.exists(default_path):
                preferred_path = default_path

        if preferred_path is not None:
            if self.loaded_data_path != preferred_path:
                self.load_data()
            return

        if self.analyzed_data is None:
            self._load_data_by_mode()

    def _auto_detect_speed_range(self):
        min_default = 1.0
        max_default = 40.0

        folder_analysis = self.experiment_manager.folder_analysis
        if not folder_analysis:
            self.min_speed_var.set(f"{min_default:.2f}")
            self.max_speed_var.set(f"{max_default:.2f}")
            return min_default, max_default

        source_path = os.path.join(folder_analysis, "analyzed_data.pkl")
        if not os.path.exists(source_path):
            self.min_speed_var.set(f"{min_default:.2f}")
            self.max_speed_var.set(f"{max_default:.2f}")
            self.log("Auto-detect fallback: using default speed range 1.0 to 40.0.")
            return min_default, max_default

        try:
            with open(source_path, 'rb') as f:
                base_data = pickle.load(f)
        except Exception as e:
            self.log(f"Auto-detect fallback: could not read analyzed_data.pkl ({e}).")
            self.min_speed_var.set(f"{min_default:.2f}")
            self.max_speed_var.set(f"{max_default:.2f}")
            return min_default, max_default

        individual_data = base_data.get('individual_data', pd.DataFrame())
        if 'average_speed' not in individual_data.columns:
            self.log("Auto-detect fallback: 'average_speed' not found, using defaults 1.0 to 40.0.")
            self.min_speed_var.set(f"{min_default:.2f}")
            self.max_speed_var.set(f"{max_default:.2f}")
            return min_default, max_default

        speed_series = pd.to_numeric(individual_data['average_speed'], errors='coerce').dropna()
        speed_series = speed_series[np.isfinite(speed_series)]
        if speed_series.empty:
            self.log("Auto-detect fallback: no valid speed values, using defaults 1.0 to 40.0.")
            self.min_speed_var.set(f"{min_default:.2f}")
            self.max_speed_var.set(f"{max_default:.2f}")
            return min_default, max_default

        q01 = float(speed_series.quantile(0.01))
        q99 = float(speed_series.quantile(0.99))
        detected_min = max(min_default, q01)
        detected_max = min(max_default, q99)

        if detected_min >= detected_max:
            detected_min, detected_max = min_default, max_default

        self.min_speed_var.set(f"{detected_min:.2f}")
        self.max_speed_var.set(f"{detected_max:.2f}")
        self.log(f"Auto-detected speed range: {detected_min:.2f} to {detected_max:.2f}.")
        self._set_speed_filter_status(detected_min, detected_max, mode="auto")
        return detected_min, detected_max

    def _set_speed_filter_status(self, min_speed, max_speed, mode="manual"):
        try:
            self.speed_filter_status_label.config(
                text=f"Speed filter active: {mode} ({float(min_speed):.2f} to {float(max_speed):.2f})"
            )
        except Exception:
            pass

    def _save_filtered_analyzed_data(self, filtered_analyzed_data):
        """Save filtered analyzed data to canonical and legacy filenames.

        Canonical path: analyzed_data_filtered.pkl
        Legacy compatibility path: analyzed_data_speed_filtered.pkl
        """
        folder_analysis = self.experiment_manager.folder_analysis
        canonical_path = os.path.join(folder_analysis, "analyzed_data_filtered.pkl")
        legacy_path = os.path.join(folder_analysis, "analyzed_data_speed_filtered.pkl")

        with open(canonical_path, 'wb') as f:
            pickle.dump(filtered_analyzed_data, f)

        if legacy_path != canonical_path:
            with open(legacy_path, 'wb') as f:
                pickle.dump(filtered_analyzed_data, f)

        return canonical_path, legacy_path

    def _filter_population_data_by_individual_time(self, population_data, filtered_individual_data):
        before_rows = len(population_data) if hasattr(population_data, '__len__') else 0
        default_stats = {
            'before_rows': before_rows,
            'after_rows': before_rows,
            'dropped_rows': 0,
            'reason': 'invalid_input'
        }

        if population_data is None or not hasattr(population_data, 'index'):
            return population_data, default_stats

        if filtered_individual_data is None or not hasattr(filtered_individual_data, 'index'):
            stats = dict(default_stats)
            stats['reason'] = 'invalid_filtered_individual_data'
            return population_data, stats

        pop_index = population_data.index
        ind_index = filtered_individual_data.index

        pop_was_datetime = isinstance(pop_index, pd.DatetimeIndex)
        ind_was_datetime = isinstance(ind_index, pd.DatetimeIndex)

        try:
            pop_index_dt = pd.to_datetime(pop_index, errors='coerce')
            ind_index_dt = pd.to_datetime(ind_index, errors='coerce')
        except Exception:
            stats = dict(default_stats)
            stats['reason'] = 'datetime_conversion_failed'
            return population_data, stats

        if not pop_was_datetime and getattr(pop_index_dt, 'isna', lambda: pd.Series([True]))().any():
            stats = dict(default_stats)
            stats['reason'] = 'population_non_datetime_index'
            return population_data, stats

        if not ind_was_datetime and getattr(ind_index_dt, 'isna', lambda: pd.Series([True]))().any():
            stats = dict(default_stats)
            stats['reason'] = 'individual_non_datetime_index'
            return population_data, stats

        if getattr(pop_index_dt, 'isna', lambda: pd.Series([True]))().all():
            stats = dict(default_stats)
            stats['reason'] = 'population_non_datetime_index'
            return population_data, stats

        if getattr(ind_index_dt, 'isna', lambda: pd.Series([True]))().all():
            stats = dict(default_stats)
            stats['reason'] = 'individual_non_datetime_index'
            return population_data, stats

        pop_floor = pd.DatetimeIndex(pop_index_dt).floor('s')
        ind_floor = pd.DatetimeIndex(ind_index_dt).floor('s')

        if 'flight_duration' in filtered_individual_data.columns:
            duration_series = pd.to_numeric(filtered_individual_data['flight_duration'], errors='coerce')
        else:
            duration_series = pd.Series(1.0, index=filtered_individual_data.index)

        duration_series = duration_series.fillna(1.0).clip(lower=1.0, upper=14400.0)
        duration_seconds = np.ceil(duration_series.to_numpy(dtype=float)).astype(np.int64)
        duration_seconds = np.clip(duration_seconds, 1, 14400)

        # Preserve timeline continuity in plots: zero-fill excluded minutes instead of dropping rows.
        count_columns = [
            col for col in getattr(population_data, 'columns', [])
            if isinstance(col, str) and col.startswith('numb_mosquitos_')
        ]
        if not count_columns:
            count_columns = population_data.select_dtypes(include=[np.number]).columns.tolist()

        window_minutes = max(1, int(ROLLING_WINDOW_SIZE_MINUTES))
        window_freq = f'{window_minutes}min'

        active_windows = set()
        for ts, dur in zip(ind_floor, duration_seconds):
            if pd.isna(ts):
                continue
            end_ts = ts + pd.Timedelta(seconds=int(dur))
            start_window = ts.floor(window_freq)
            end_window = end_ts.floor(window_freq)
            window_range = pd.date_range(start=start_window, end=end_window, freq=window_freq)
            active_windows.update(window_range)

        if not active_windows:
            filtered_population_data = population_data.copy()
            if count_columns:
                filtered_population_data.loc[:, count_columns] = 0
            stats = {
                'before_rows': before_rows,
                'after_rows': before_rows,
                'dropped_rows': 0,
                'reason': f'no_active_{window_minutes}min_windows_zero_filled'
            }
            return filtered_population_data, stats

        pop_window = pop_floor.floor(window_freq)
        keep_mask = pop_window.isin(active_windows)
        filtered_population_data = population_data.copy()
        if count_columns:
            filtered_population_data.loc[~keep_mask, count_columns] = 0

        after_rows = before_rows
        reason = (
            f'{window_minutes}min_window_zero_filled'
            if (~keep_mask).any()
            else f'{window_minutes}min_window_zero_filled_no_change'
        )
        stats = {
            'before_rows': before_rows,
            'after_rows': after_rows,
            'dropped_rows': 0,
            'reason': reason
        }
        return filtered_population_data, stats

    def apply_speed_filter(self):
        if not self.experiment_manager.folder_analysis:
            self.log("No analysis folder found. Please load an experiment first.")
            return

        if self.analyzed_data is None:
            self.load_data()
            if self.analyzed_data is None:
                self.log("No analyzed data loaded. Please run concatenation first.")
                return

        if self.auto_speed_detect_var.get():
            min_speed, max_speed = self._auto_detect_speed_range()
        else:
            try:
                min_speed = float(self.min_speed_var.get())
                max_speed = float(self.max_speed_var.get())
                self._set_speed_filter_status(min_speed, max_speed, mode="manual")
            except ValueError:
                self.log("Invalid speed filter values. Please enter numeric min/max speeds.")
                return

        if min_speed >= max_speed:
            self.log("Invalid speed range: min speed must be less than max speed.")
            return

        if 'individual_data' not in self.analyzed_data or self.analyzed_data['individual_data'] is None:
            self.log("No individual data found in analyzed data.")
            return

        individual_data = self.analyzed_data['individual_data']
        if 'average_speed' not in individual_data.columns:
            self.log("Column 'average_speed' not found in individual data.")
            return

        self.last_speed_before = pd.to_numeric(individual_data['average_speed'], errors='coerce')
        self.last_speed_before = self.last_speed_before[np.isfinite(self.last_speed_before)]

        original_count = len(individual_data)
        filtered_individual_data = individual_data[
            individual_data['average_speed'].between(min_speed, max_speed, inclusive='both')
        ].copy()
        filtered_count = len(filtered_individual_data)

        self.last_speed_after = pd.to_numeric(filtered_individual_data['average_speed'], errors='coerce')
        self.last_speed_after = self.last_speed_after[np.isfinite(self.last_speed_after)]

        filtered_population_data, population_time_mask_stats = self._filter_population_data_by_individual_time(
            self.analyzed_data.get('population_data'),
            filtered_individual_data
        )
        filtered_speaker_side_data, _ = self._filter_population_data_by_individual_time(
            self.analyzed_data.get('speaker_side_data'),
            filtered_individual_data
        )

        filtered_analyzed_data = dict(self.analyzed_data)
        filtered_analyzed_data['population_data'] = filtered_population_data
        filtered_analyzed_data['individual_data'] = filtered_individual_data
        filtered_analyzed_data['speaker_side_data'] = filtered_speaker_side_data
        filtered_analyzed_data['speaker_side_mapping_version'] = SPEAKER_SIDE_MAPPING_VERSION
        filtered_analyzed_data['speed_filter'] = {
            'min_speed': min_speed,
            'max_speed': max_speed,
            'rows_before': original_count,
            'rows_after': filtered_count,
            'population_time_mask': population_time_mask_stats,
        }

        pop_before = self._get_population_range(self.analyzed_data.get('population_data')) if isinstance(self.analyzed_data, dict) else None

        try:
            canonical_path, legacy_path = self._save_filtered_analyzed_data(filtered_analyzed_data)
        except Exception as e:
            self.log(f"Failed to save speed-filtered data: {e}")
            return

        self.analyzed_data = filtered_analyzed_data
        kept_percent = (filtered_count / original_count * 100.0) if original_count else 0.0
        self.log(
            f"Speed filter applied ({min_speed} to {max_speed}). "
            f"Kept {filtered_count}/{original_count} rows ({kept_percent:.1f}%)."
        )
        self.log(
            f"Population time-mask rows: {population_time_mask_stats['before_rows']} -> "
            f"{population_time_mask_stats['after_rows']} (reason: {population_time_mask_stats['reason']})."
        )
        pop_after = self._get_population_range(filtered_analyzed_data.get('population_data')) if isinstance(filtered_analyzed_data, dict) else None
        if pop_before and pop_after:
            self.log(
                f"[DEBUG] Speed filter timeline check | before: {pop_before['rows']} rows, {pop_before['min']} -> {pop_before['max']} | "
                f"after: {pop_after['rows']} rows, {pop_after['min']} -> {pop_after['max']}"
            )
        self.log(f"Speed-filtered data saved to: {canonical_path}")
        if legacy_path != canonical_path:
            self.log(f"Compatibility copy saved to: {legacy_path}")

    def run_speed_filter(self):
        """Unified speed filter action.
        Prefers full batch rebuild from tracking files; falls back to loaded-data filtering.
        """
        folder_analysis = self.experiment_manager.folder_analysis
        if not folder_analysis:
            self.log("No analysis folder found. Please load an experiment first.")
            return

        final_tracking_folder = os.path.join(folder_analysis, "final_tracking_data")
        if os.path.exists(final_tracking_folder):
            self.log("Running speed filter using all tracking files (batch mode)...")
            self.batch_speed_filter()
        else:
            self.log("Tracking folder not found; running speed filter on currently loaded analyzed data.")
            self.apply_speed_filter()

    def batch_speed_filter(self):
        folder_analysis = self.experiment_manager.folder_analysis
        if not folder_analysis:
            self.log("No analysis folder found. Please load an experiment first.")
            return

        if self.auto_speed_detect_var.get():
            min_speed, max_speed = self._auto_detect_speed_range()
        else:
            try:
                min_speed = float(self.min_speed_var.get())
                max_speed = float(self.max_speed_var.get())
                self._set_speed_filter_status(min_speed, max_speed, mode="manual")
            except ValueError:
                self.log("Invalid speed filter values. Please enter numeric min/max speeds.")
                return

        if min_speed >= max_speed:
            self.log("Invalid speed range: min speed must be less than max speed.")
            return

        final_tracking_folder = os.path.join(folder_analysis, "final_tracking_data")
        if not os.path.exists(final_tracking_folder):
            self.log(f"Tracking data folder not found: {final_tracking_folder}")
            return

        video_files = []
        for f in os.listdir(final_tracking_folder):
            if f.startswith("forward_mosq_tracks_") and not f.endswith((".png", ".csv", ".txt")):
                video_name = f.replace("forward_mosq_tracks_", "")
                video_files.append(video_name)

        if not video_files:
            self.log("No tracking data found for batch speed filtering. Please run batch processing first.")
            return

        total_found = len(video_files)
        video_files.sort()
        self.log(
            f"Starting batch speed filter for {len(video_files)} tracking files "
            f"with range [{min_speed}, {max_speed}]..."
        )

        with pool_concurrency_slot() as concurrency:
            processes_to_use = _default_batch_worker_count(self.experiment_manager.config, concurrency=concurrency)
            with Pool(processes=processes_to_use) as pool:
                results_raw = pool.starmap(process_video_files, [(video_name, folder_analysis) for video_name in video_files])

        successful_videos = []
        failed_videos = []
        for video_name, result in zip(video_files, results_raw):
            if result is None:
                failed_videos.append(video_name)
            else:
                successful_videos.append(video_name)

        results = [result for result in results_raw if result is not None]
        if not results:
            self.log("No data found for batch speed filtering.")
            return

        all_population_data = []
        all_individual_data = []
        all_resting_data = []
        all_flight_metrics_data = []
        all_speaker_side_data = []
        summary_data = []

        total_before = 0
        total_after = 0
        total_population_before = 0
        total_population_after = 0
        skipped_files = 0
        all_before_speed = []
        all_after_speed = []
        population_reason_counts = {}

        for result in results:
            population_data, individual_data, resting_data, flight_metrics_data, summary_row, speaker_side_data = result
            all_resting_data.append(resting_data)
            all_flight_metrics_data.append(flight_metrics_data)
            summary_data.append(summary_row)

            if 'average_speed' not in individual_data.columns:
                skipped_files += 1
                all_individual_data.append(individual_data)
                filtered_population = population_data
                filtered_speaker_side = speaker_side_data
                before_rows = len(population_data) if hasattr(population_data, '__len__') else 0
                population_stats = {
                    'before_rows': before_rows,
                    'after_rows': before_rows,
                    'dropped_rows': 0,
                    'reason': 'skipped_missing_average_speed'
                }
                all_population_data.append(filtered_population)
                all_speaker_side_data.append(filtered_speaker_side)
                total_population_before += population_stats['before_rows']
                total_population_after += population_stats['after_rows']
                population_reason_counts[population_stats['reason']] = population_reason_counts.get(population_stats['reason'], 0) + 1
                continue

            before_series = pd.to_numeric(individual_data['average_speed'], errors='coerce')
            before_series = before_series[np.isfinite(before_series)]
            if not before_series.empty:
                all_before_speed.append(before_series)

            before_count = len(individual_data)
            filtered_individual = individual_data[
                individual_data['average_speed'].between(min_speed, max_speed, inclusive='both')
            ].copy()
            after_count = len(filtered_individual)

            after_series = pd.to_numeric(filtered_individual['average_speed'], errors='coerce')
            after_series = after_series[np.isfinite(after_series)]
            if not after_series.empty:
                all_after_speed.append(after_series)

            total_before += before_count
            total_after += after_count
            all_individual_data.append(filtered_individual)

            filtered_population, population_stats = self._filter_population_data_by_individual_time(
                population_data,
                filtered_individual
            )
            filtered_speaker_side, _ = self._filter_population_data_by_individual_time(
                speaker_side_data,
                filtered_individual
            )
            all_population_data.append(filtered_population)
            all_speaker_side_data.append(filtered_speaker_side)
            total_population_before += population_stats['before_rows']
            total_population_after += population_stats['after_rows']
            population_reason_counts[population_stats['reason']] = population_reason_counts.get(population_stats['reason'], 0) + 1

        if all_before_speed:
            self.last_speed_before = pd.concat(all_before_speed, ignore_index=True)
        else:
            self.last_speed_before = pd.Series(dtype=float)

        if all_after_speed:
            self.last_speed_after = pd.concat(all_after_speed, ignore_index=True)
        else:
            self.last_speed_after = pd.Series(dtype=float)

        full_population_data = _safe_concat_dataframes(all_population_data)
        full_individual_data = _safe_concat_dataframes(all_individual_data)
        full_resting_data = _safe_concat_dataframes(all_resting_data)

        full_flight_metrics_data = _concat_flight_metrics_data(
            all_flight_metrics_data,
            log_fn=self.log
        )
        full_speaker_side_data = _safe_concat_dataframes(all_speaker_side_data)

        summary_df = pd.DataFrame(summary_data, columns=['start_time', 'avg_fraction_flying', 'total_nb_tracks'])

        filtered_analyzed_data = {
            'population_data': full_population_data,
            'individual_data': full_individual_data,
            'resting_data': full_resting_data,
            'summary_data': summary_df,
            'flight_metrics_data': full_flight_metrics_data,
            'speaker_side_data': full_speaker_side_data,
            'speaker_side_mapping_version': SPEAKER_SIDE_MAPPING_VERSION,
            'speed_filter': {
                'min_speed': min_speed,
                'max_speed': max_speed,
                'rows_before': total_before,
                'rows_after': total_after,
                'population_rows_before': total_population_before,
                'population_rows_after': total_population_after,
                'population_time_mask': {
                    'reason_counts': population_reason_counts
                },
                'skipped_files_missing_average_speed': skipped_files,
                'mode': 'batch_from_tracking_files'
            },
            'concat_debug': {
                'requested_videos': video_files,
                'successful_videos': successful_videos,
                'failed_videos': failed_videos,
            }
        }

        try:
            canonical_path, legacy_path = self._save_filtered_analyzed_data(filtered_analyzed_data)
        except Exception as e:
            self.log(f"Failed to save batch speed-filtered data: {e}")
            return

        self.analyzed_data = filtered_analyzed_data
        kept_percent = (total_after / total_before * 100.0) if total_before else 0.0
        self.log(
            f"Batch speed filter complete ({min_speed} to {max_speed}). "
            f"Kept {total_after}/{total_before} rows ({kept_percent:.1f}%)."
        )
        pop_after = self._get_population_range(filtered_analyzed_data.get('population_data'))
        if pop_after:
            self.log(
                f"[DEBUG] Batch speed filter timeline check | population_data rows {total_population_before} -> "
                f"{total_population_after} | timeline: {pop_after['rows']} rows, {pop_after['min']} -> {pop_after['max']}"
            )
        self.log(
            f"Population time-mask rows across videos: {total_population_before} -> {total_population_after} "
            f"(reasons: {population_reason_counts})."
        )
        if skipped_files > 0:
            self.log(f"Note: {skipped_files} file(s) did not contain 'average_speed' and were not speed-filtered.")
        self.log(f"Batch speed-filtered data saved to: {canonical_path}")
        if legacy_path != canonical_path:
            self.log(f"Compatibility copy saved to: {legacy_path}")

    def export_rolling_data_to_csv(self):
        self._ensure_preferred_plot_data_loaded()
        if self.analyzed_data is None:
            self.log("Data not loaded. Please load the data first.")
            return

        full_population_data = self.analyzed_data['population_data']
        self._log_plot_video_debug(full_population_data)
        full_population_data = self._prepare_population_time_series(full_population_data)
        if full_population_data is None or full_population_data.empty:
            self.log("Population data has no valid datetime rows after parsing.")
            return

        self.log(
            f"[DEBUG] Export rolling CSV from {self.loaded_data_path or 'in-memory data'} | "
            f"rows={len(full_population_data)} | "
            f"range={full_population_data.index.min()} -> {full_population_data.index.max()}"
        )

        # Resample and compute rolling average
        resampled_population_data = full_population_data.resample(ROLLING_RESAMPLE_INTERVAL).mean(numeric_only=True)  # Resample intervals
        rolling_population_data = resampled_population_data.rolling(window=ROLLING_WINDOW_SIZE_MINUTES, min_periods=1).mean(numeric_only=True)  # Rolling avg window
        valid_mask = resampled_population_data.notna().any(axis=1)
        rolling_population_data = rolling_population_data.where(valid_mask)
        rolling_population_data = rolling_population_data.dropna(how='all')  # Remove rows with all NaN

        # Define output path
        folder_analysis = self.experiment_manager.folder_analysis
        output_path = os.path.join(folder_analysis, "rolling_population_data.csv")

        try:
            rolling_population_data.to_csv(output_path)
            self.log(f"Rolling population data exported to: {output_path}")
        except Exception as e:
            self.log(f"Failed to export rolling data CSV: {e}")

    def _prepare_population_time_series(self, population_data):
        """Normalize population_data index for robust time-based plotting."""
        if population_data is None or len(population_data) == 0:
            return None

        df = population_data.copy()
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~pd.isna(df.index)]
        if df.empty:
            return None

        df = df.sort_index()
        if df.index.duplicated().any():
            df = df.groupby(df.index).mean(numeric_only=True)

        return df

    def _get_population_range(self, population_data):
        try:
            df = self._prepare_population_time_series(population_data)
            if df is None or df.empty:
                return None
            return {
                'rows': len(df),
                'min': df.index.min(),
                'max': df.index.max(),
            }
        except Exception:
            return None

    def _log_plot_video_debug(self, population_data):
        """Log which videos are represented in the plotting dataset and which failed in concatenation."""
        try:
            videos_plotted = []
            if isinstance(population_data, pd.DataFrame) and 'video' in population_data.columns:
                videos_plotted = sorted([str(v) for v in population_data['video'].dropna().unique().tolist()])

            if videos_plotted:
                self.log(f"[DEBUG] Videos represented in plot data ({len(videos_plotted)}):")
                for idx, name in enumerate(videos_plotted, start=1):
                    self.log(f"  [PLOT {idx}/{len(videos_plotted)}] {name}")
            else:
                self.log("[DEBUG] Plot data does not include a 'video' column for per-video listing.")

            concat_debug = self.analyzed_data.get('concat_debug', {}) if isinstance(self.analyzed_data, dict) else {}
            failed_videos = concat_debug.get('failed_videos', []) if isinstance(concat_debug, dict) else []
            if failed_videos:
                self.log(f"[DEBUG] Videos failed during concatenation ({len(failed_videos)}):")
                for idx, name in enumerate(failed_videos, start=1):
                    self.log(f"  [FAIL {idx}/{len(failed_videos)}] {name}")
            else:
                self.log("[DEBUG] No failed videos recorded in concatenation metadata.")
        except Exception as e:
            self.log(f"[DEBUG] Could not list plotted/failed videos: {e}")


def _run_batch_analysis(args):
    video_name, settings_file, folder_videos, folder_analysis, experiment_path, preloaded_settings, use_overhaul, cv2_threads = args

    # Pin this worker's OpenCV thread count so the parallel workers share the cores instead of each
    # fanning out to all of them (the env-var BLAS caps at module import handle numpy/scipy).
    try:
        cv2.setNumThreads(int(cv2_threads))
    except Exception:
        pass

    def dummy_log(message):
        pass

    try:
        if _should_skip_video(video_name, folder_analysis):
            return {'ok': True, 'video_name': video_name, 'fps': None, 'skipped': True}
        if video_name in load_excluded_segments(folder_analysis):
            return {'ok': True, 'video_name': video_name, 'fps': None, 'skipped': True,
                    'excluded': True}

        experiment_manager = ExperimentManager(dummy_log)
        experiment_manager.folder_videos = folder_videos
        experiment_manager.folder_analysis = folder_analysis
        # Honored by run_tracking_analysis: injects the overhaul into the per-run in-memory settings.
        experiment_manager.use_tracking_overhaul = bool(use_overhaul)

        # Use pre-loaded settings if available to avoid file contention
        if preloaded_settings is not None:
            experiment_manager.settings = preloaded_settings
            experiment_manager.settings_file = settings_file

        # Load experiment (which will also load settings if not already loaded)
        if experiment_path:
            if str(experiment_path).lower().endswith('.json'):
                # Skip settings load if already pre-loaded
                experiment_manager.load_experiment_from_json(experiment_path)
            else:
                experiment_manager.load_experiment(experiment_path)
        elif settings_file and preloaded_settings is None:
            # Fallback: if no experiment path and no pre-loaded settings, load settings
            experiment_manager.load_settings(settings_file)

        if not experiment_manager.experiment:
            error_msg = 'Experiment object is None after loading. This should not happen.'
            if not experiment_path:
                error_msg = 'No experiment path provided and no experiment loaded'
            elif not experiment_manager.settings:
                error_msg = 'Settings failed to load'
            return {'ok': False, 'video_name': video_name, 'error': error_msg}

        fps = experiment_manager.detect_and_store_fps(video_name, persist=False)
        experiment_manager.run_tracking_analysis(video_name)
        
        # Verify that tracking data was actually saved
        tracking_file_path = _tracking_output_path(folder_analysis, video_name)
        if not os.path.exists(tracking_file_path):
            return {'ok': False, 'video_name': video_name, 'error': 'Tracking analysis completed but output file was not created (video may be corrupted or too short)'}
        
        return {'ok': True, 'video_name': video_name, 'fps': fps}
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return {'ok': False, 'video_name': video_name, 'error': error_details}
