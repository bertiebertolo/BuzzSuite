########################## IMPORT ALL NECESSARY PACKAGES ####################
import numpy as np
import cv2
import os
import pickle
from buzzwatch_data_analysis.resting_obj_tracker import *
from buzzwatch_data_analysis.moving_obj_tracker import *
from buzzwatch_data_analysis.misc_functions import *
from buzzwatch_data_analysis.mosquito_obj_tracker import *
from buzzwatch_data_analysis.flight_center_distance_analysis import calculate_cage_centroid
import bisect
import copy
import yaml
import time
from scipy.spatial import distance as dist
import scipy.sparse.csgraph as graph
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta
import re
import warnings
from logger import MultiLogger

# The per-frame vision ops (threshold / dilate / distance transform) call cv2 directly. An opt-in
# OpenCV UMat/OpenCL "GPU" indirection layer (gpu_ops) used to wrap them; it was removed after being
# benchmarked on real data: it computed byte-identical results but ran ~1.9x SLOWER than plain CPU,
# because at typical cage-video resolutions these ops are so cheap that the host<->device transfer
# costs more than the compute. See DEVLOG (2026-07-11) — do not reintroduce it.

DEFAULT_VIDEO_SEGMENT_SECONDS = int(os.environ.get('BUZZSUITE_VIDEO_SEGMENT_SECONDS', '1200'))

# Suppress expected NumPy warnings from empty data windows
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

# ------------------------------------------------------------------------------------------------
# Tracking-overhaul feature flags.
# All flags live under an optional ``tracking:`` block in buzzwatch_track_settings.yml.
# Defaults below reproduce the ORIGINAL behaviour, so any settings file without the block
# (or with the block absent) behaves exactly as before. Turn features on per-experiment.
# ------------------------------------------------------------------------------------------------
_TRACKING_FLAG_DEFAULTS = {
    "resting_reference": "median",        # "median" (original) | "bright_percentile" (WS1)
    "split_blobs": False,                 # watershed splitting of merged clusters (WS2)
    "assignment": "greedy",               # "greedy" (original) | "hungarian" (WS3)
    "motion_model": "none",               # "none" (original) | "constant_velocity" (WS3)
    "use_backward_pass": False,           # forward+backward merge (WS4)
    "resting_adaptive": False,            # per-pixel adaptive resting threshold (WS5)
    "resting_adaptive_k": 6.0,            # threshold = max(floor, k * bright-frame std); bright-frame
                                          # std is clean (empty-cage only), so k can be higher and
                                          # only suppresses genuine structure, never speaker resters
    "resting_std_percentile": 75,         # empty-cage cutoff for the bright-frame std: std is taken
                                          # over frames where the pixel is >= this percentile, so a
                                          # mosquito resting up to ~this fraction of the clip is
                                          # excluded (protects long/low-activity resters from suppression)
    "resting_threshold_floor": 12,        # floor (grey levels); = the working flat threshold so
                                          # stable pixels are unchanged and k*std only raises the
                                          # bar at high-variance structure edges
    "resting_enhance_dark": False,        # WS6: gamma-lift the dark structure regions (e.g. the dark
                                          # bottom-left vent/panel) before differencing, so a dark-on-
                                          # dark mosquito there gets more contrast. Off = unchanged.
    "resting_enhance_gamma": 0.5,         # gamma applied to dark pixels (<1 brightens; 0.5 = sqrt)
    "resting_enhance_cutoff": 90,         # only lift pixels where the empty-cage reference is darker
                                          # than this (leaves the bright floor untouched)
    "resting_reference_window": 0,        # WS7: seconds for a time-local (rolling) reference. 0 =
                                          # one global reference (original WS1). >0 recomputes the
                                          # empty-cage reference/std in temporal blocks of this many
                                          # seconds, so it tracks slow drift/shimmer (dark corners).
    "bg_session_cache": False,            # opt-in: share ONE within-video bright-percentile resting
                                          # reference across all segments of a recording session
                                          # (temp_data/bg_cache_{session_key}.pkl) instead of one per
                                          # segment. Off = per-segment (original, byte-identical).
}

# Scale used to persist the per-pixel temporal-std map as an 8-bit PNG (WS5). The raw std is
# small (~1 on stable pixels, up to ~40 at high-contrast structure edges), so we store
# round(std * _REST_STD_SCALE) and divide on load to keep sub-integer resolution.
_REST_STD_SCALE = 4.0


def tracking_flag(settings, key):
    """Read a tracking-overhaul feature flag, falling back to the original-behaviour default."""
    default = _TRACKING_FLAG_DEFAULTS.get(key)
    try:
        block = settings.get("tracking", {}) or {}
        return block.get(key, default)
    except Exception:
        return default


def split_merged_contour(shape, contour, single_area, min_length):
    """Split one over-sized contour into multiple centroids (WS2).

    Mosquitoes crowd at the speaker and merge into a single blob that is then *rejected* by the
    ``max_length`` size filter, so the whole cluster vanishes. Here we recover the individuals via
    local-maxima detection on the distance transform: each peak that is locally highest within a
    ~mosquito-radius neighbourhood is one centroid. cv2-only, no skimage.

    Returns a list of (x, y) centroids; falls back to the bounding-box centre if it cannot split.

    Works on a padded crop around the contour instead of the whole frame. The full-frame version
    cost ~20 ms per contour on 1920x1080 video whatever the contour's size, so frames with hundreds
    of large blobs (a recording's first segment, while the camera is still focusing) took hours per
    segment. The crop gives bit-identical output, in the same order (DEVLOG 2026-09-17: 6,880 real
    contours / 44,028 centroids compared, 0 differences):
      * pad = k + 2 >= the dilation radius + the 5x5 distance mask's reach, and every pixel outside
        the contour is 0 in both versions, so distances and peaks on the contour are identical;
      * the crop origin is rounded DOWN to even coordinates -- connectedComponents labels 2x2
        blocks in raster order, and an odd origin shifts that grid and reorders the components;
      * OpenCV's centroid is sum(coordinate) / area, so the exact integer sum is rebuilt in frame
        coordinates before dividing (crop centroid + offset can differ in the last bit).
    """
    import math
    (x, y, w, h) = cv2.boundingRect(contour)
    fallback = [(x + w / 2.0, y + h / 2.0)]
    area = cv2.contourArea(contour)
    if not single_area or single_area <= 0 or area < 1.6 * single_area:
        return fallback
    # Local-maxima detection: a pixel is a peak if it equals the dilation within a
    # neighbourhood whose radius matches the expected single-mosquito radius.  This finds each
    # mosquito's core independently of how deep the overall blob is (fixes the dense-crowd case
    # where the old global threshold 0.5*max was too high and merged many peaks into one).
    radius = max(2, int(math.sqrt(single_area / math.pi)))
    k = max(3, 2 * radius - 1)   # odd kernel, ~mosquito diameter
    pad = k + 2
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x0 -= x0 % 2
    y0 -= y0 % 2
    x1 = min(shape[1], x + w + pad)
    y1 = min(shape[0], y + h + pad)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED, offset=(-x0, -y0))
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist_map.max() <= 0:
        return fallback
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(dist_map, kernel)
    floor = max(1.0, radius * 0.4)   # ignore peaks shallower than ~half a mosquito radius
    local_max = ((dist_map == dilated) & (dist_map > floor)).astype(np.uint8) * 255
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max)
    pts = []
    for label in range(1, n_labels):  # skip background label 0
        if stats[label, cv2.CC_STAT_AREA] >= max(1, (min_length * min_length) / 4.0):
            n = int(stats[label, cv2.CC_STAT_AREA])
            cx, cy = centroids[label]
            pts.append(((int(round(float(cx) * n)) + n * x0) / n,
                        (int(round(float(cy) * n)) + n * y0) / n))
    return pts if pts else fallback

##################################################################################################################################
########################## Memory-bounded replacement for the dense NxN/NxM matching matrices ##################################
class _SparseFillMatrix:
    """Drop-in replacement for ``np.full((n_rows, n_cols), default_value)``, used ONLY by
    ``assemble_unmatched_moving_tracks``/``assemble_resting_and_moving_tracks`` and consumed ONLY
    by ``find_optimal_matching_resting_moving_from_distance``. Supports exactly the API surface
    those functions use (``m[i][j] = v``, ``m[i][j]``, ``m[r, c]``, ``.shape``, ``.min(axis=1)``,
    ``.argmin(axis=1)``) -- it is NOT a general numpy substitute.

    Why: on the densest real recording sessions, ``list_of_ID_moving``/``list_of_ID_still`` can
    reach 20,000+ tracks (see DEVLOG 2026-09-11 -- almost certainly itself a symptom of noisy
    over-fragmented detections on those sessions, not genuine mosquito counts, but that is a
    separate question from this fix). A dense NxN float64 array at N=21554 is 3.46 GiB; the
    matching int array is another ~1.6 GiB; two such pairs (dist+time, and often two directions,
    landing+takeoff) routinely exceeded available RAM and crashed the worker outright (confirmed
    live: `Unable to allocate 3.46 GiB for an array with shape (21554, 21554)`, and separately
    `... shape (20975, 20975) ... data type int32`). But the *fill fraction* of these matrices is
    tiny: an entry is only ever set when a track pair passes a restrictive temporal-window check,
    which excludes the vast majority of the N**2 possible pairs by construction. Storing only the
    entries that are ever actually set (a dict keyed by (row, col)) uses memory proportional to the
    number of real candidate matches, not N**2 -- typically orders of magnitude less.

    Numerically IDENTICAL to the dense array it replaces for every consumer in this file: the
    nested loops that populate it are completely unchanged (same iteration order, same conditions,
    same side effects on the track dicts) -- only the storage container changed. ``.min()``/
    ``.argmin()`` are defined to match ``np.ndarray``'s exact semantics for an array that starts
    entirely filled with ``default_value`` and has a sparse subset of cells overwritten to smaller
    values: an unset row's min/argmin is (default_value, column 0), matching what
    ``np.full((n,m), default_value).argmin(axis=1)`` returns for an all-equal row (index 0, numpy's
    "first occurrence of the minimum" tie-break). See DEVLOG for the parity test against a real
    dense array across random and edge-case (empty row, single-entry row, tied values) inputs.
    """
    __slots__ = ('n_rows', 'n_cols', 'default', 'data')

    def __init__(self, n_rows, n_cols, default_value):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.default = default_value
        self.data = {}  # (row, col) -> value, only for cells ever explicitly written

    @property
    def shape(self):
        return (self.n_rows, self.n_cols)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.data.get(key, self.default)
        return _SparseMatrixRow(self, key)

    def __setitem__(self, key, value):
        # m[i, j] = v style (not currently used by this file's callers, kept for API completeness
        # so this class is a safe substitute anywhere the m[i][j] pattern might be written
        # differently in the future).
        i, j = key
        self.data[(i, j)] = value

    def min(self, axis=1):
        assert axis == 1, "_SparseFillMatrix.min only supports axis=1 (the only usage in this file)"
        row_best = {}
        for (i, j), v in self.data.items():
            if i not in row_best or v < row_best[i]:
                row_best[i] = v
        out = np.empty(self.n_rows, dtype=float)
        for i in range(self.n_rows):
            out[i] = min(self.default, row_best.get(i, self.default))
        return out

    def argmin(self, axis=1):
        assert axis == 1, "_SparseFillMatrix.argmin only supports axis=1 (the only usage in this file)"
        # dict iteration is insertion order (Python 3.7+); the populating loops always write
        # column j in increasing order within a row, so "first strictly-smaller value wins, ties
        # keep the earlier (lower-column) entry" naturally reproduces np.argmin's left-to-right
        # first-occurrence-of-the-minimum tie-break without needing an explicit column comparison.
        row_best = {}  # i -> (best_value, best_col)
        for (i, j), v in self.data.items():
            if i not in row_best or v < row_best[i][0]:
                row_best[i] = (v, j)
        out = np.zeros(self.n_rows, dtype=np.intp)
        for i in range(self.n_rows):
            best = row_best.get(i)
            # Strictly less than default: a real entry beats (or ties from below, impossible here
            # since ties are resolved by insertion order above) the implicit all-default row.
            # Equal-to-default or no entry at all: matches numpy's all-default-row behavior, which
            # always returns column 0 (first occurrence of the minimum among equal values).
            out[i] = best[1] if (best is not None and best[0] < self.default) else 0
        return out

    def to_dense(self):
        """Materialize the full dense array -- only safe to call when n_rows*n_cols is known to be
        small (the 'optimal'/Hungarian matching path needs a real array; it is not the default
        matching_method and is guarded by a size check before calling this, see
        find_optimal_matching_resting_moving_from_distance)."""
        out = np.full((self.n_rows, self.n_cols), self.default, dtype=float)
        for (i, j), v in self.data.items():
            out[i, j] = v
        return out


class _SparseMatrixRow:
    """Row proxy returned by ``_SparseFillMatrix.__getitem__(i)`` so ``m[i][j] = v`` and ``m[i][j]``
    both work exactly like indexing a numpy row view -- writes go straight into the parent's dict,
    keyed by (row, col), with no intermediate row array ever materialized."""
    __slots__ = ('_parent', '_row')

    def __init__(self, parent, row):
        self._parent = parent
        self._row = row

    def __setitem__(self, col, value):
        self._parent.data[(self._row, col)] = value

    def __getitem__(self, col):
        return self._parent.data.get((self._row, col), self._parent.default)


########################## Class that manages tracking of a single video ##############################################
class single_video_analysis:

    # dist_moving_obj = 30
    # dist_resting_obj = 15
    # min_duration_resting_traj = 125

    def __init__(self,experiment_class,video_name,debug_mode):
        self.video_mp4_name = video_name+".mp4"#experiment_class.list_videos_files[f]
        self.folder_analysis = experiment_class.folder_analysis
        self.video_name = video_name#os.path.splitext(self.video_mp4_name)[0]
        self.video_path = os.path.join(experiment_class.folder_videos , self.video_mp4_name)
        self.background_path = experiment_class.background_path
        self.folder_temp = experiment_class.folder_temp_data
        self.folder_final = experiment_class.folder_final
        self.settings = experiment_class.settings # copy setting file
        self.folder_test_tracking  = experiment_class.folder_test_tracking
        self.control_border_points = self.settings["control_border_points"]
        self.cage_border_points = self.settings["cage_border_points"]
        self.sugar_border_points = self.settings["sugar_border_points"]
        self.mosquito_tracks = None
        self.settings_file = experiment_class.settings_file
        # Memoized blob-split params per seg sub-block (resolved once, not recomputed per frame).
        self._blob_split_cache = {}
        self.fps = 25.0
        try:
            self.fps = float(self.settings.get("fps", 25.0))
        except Exception:
            self.fps = 25.0
        if self.fps <= 0:
            self.fps = 25.0

        try:

            cap = cv2.VideoCapture(self.video_path)
            n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
            cap.release()
            self.total_number_frames = n_frame
        except Exception:
            self.total_number_frames = 30000

        #objs_resting_tracked = resting_tracker(se)
        #self.maxobjdisappear_resting = objs_resting_tracked.maxDisappeared
        if debug_mode:
            self.video_object_path = self.folder_temp+"/temp_data_"+self.video_name+".pkl"
            with open(self.video_object_path, 'wb') as f:
                pickle.dump(self, f)
        
        print("Starting analysis of "+self.video_name) 

########################## Run video and segment both resting and moving points ####################
    def segment_resting_and_moving_objects(self,step_to_force_analyze,debug_mode):
        """ 
        Segment each frame of video for resting and moving objects
        """
        if hasattr(self, 'resting_objects')==0 or step_to_force_analyze<=1:
            print("Start running segmentation")

            # Parameters
            dist_moving_obj = self.settings["seg_moving"]["dist_moving_obj"] # Can be changed
            dist_resting_obj = self.settings["seg_resting"]["dist_resting_obj"] # Can be changed

            # WS2: with blob-splitting on, one centroid per (split) contour makes the per-frame
            # distance suppression redundant and harmful (it erases crowded mosquitoes at the
            # speaker). Shrink it to a true-duplicate merge; otherwise keep original behaviour.
            if tracking_flag(self.settings, "split_blobs"):
                dedup_moving = 3
                dedup_resting = 3
            else:
                dedup_moving = dist_moving_obj
                dedup_resting = dist_resting_obj

            # Load background
            #
            path_back = self.folder_analysis+"/images_mortality/"+self.video_name+".png"
            #path_back = self.background_path
            check_file = os.path.isfile(path_back)
            if check_file: # If the background file exists
                #background = cv2.imread(self.background_path)
                background = cv2.imread(path_back)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

                # WS1: optional within-video bright-percentile "empty-cage" reference for resting
                # detection. Directional subtraction (reference - frame) catches stationary
                # mosquitoes regardless of how long they rest, matches this video's mesh exactly
                # (no structural residual), and excludes dark resting mosquitoes from the
                # reference. Falls back to the median background (original behaviour) when off.
                rest_reference = None
                rest_thr_map = None
                if tracking_flag(self.settings, "resting_reference") == "bright_percentile":
                    # WS5: per-pixel adaptive resting threshold. Only meaningful on top of the
                    # within-video reference (it thresholds the directional rest_gray = reference -
                    # frame). When on, the flat gray_tresh is replaced by max(floor, k * temporal
                    # std), raising the bar ONLY where the background fluctuates (structure edges).
                    adaptive = bool(tracking_flag(self.settings, "resting_adaptive"))
                    rest_reference, rest_std = self._get_within_video_rest_stats(need_std=adaptive)
                    if adaptive and rest_std is not None:
                        k = float(tracking_flag(self.settings, "resting_adaptive_k"))
                        floor = float(tracking_flag(self.settings, "resting_threshold_floor"))
                        rest_thr_map = np.maximum(floor, k * rest_std).astype(np.float32)

                # WS6: optional dark-region brightening. A dark mosquito on the dark vent/panel
                # corners is low-contrast (dark-on-dark) and gets missed. A gamma lift applied ONLY
                # where the empty-cage reference is dark amplifies the directional difference there
                # without compressing the bright floor (reference and each frame are lifted
                # identically, so the already-bright floor is untouched).
                rest_enh_lut = None
                rest_dark_mask = None
                rest_reference_enh = None
                if rest_reference is not None and tracking_flag(self.settings, "resting_enhance_dark"):
                    gamma = float(tracking_flag(self.settings, "resting_enhance_gamma"))
                    cutoff = float(tracking_flag(self.settings, "resting_enhance_cutoff"))
                    rest_enh_lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
                    rest_dark_mask = rest_reference < cutoff
                    rest_reference_enh = np.where(rest_dark_mask, cv2.LUT(rest_reference, rest_enh_lut), rest_reference)

                # WS7: optional time-local (rolling) reference. Build per-block references/std/thr_map
                # (and enhanced ref) so the loop can pick the block matching each frame, tracking
                # slow drift a single global reference cannot. Off (window<=0) keeps the global ref.
                rest_blocks = None
                wframes = 0
                window = int(tracking_flag(self.settings, "resting_reference_window") or 0)
                if rest_reference is not None and window > 0:
                    wframes = max(int(window * self.fps), 250)
                    rest_blocks = self._get_block_rest_stats(wframes, need_std=adaptive)
                    if rest_blocks:
                        # Apply the rolling reference ONLY in dark regions (the drifting structure
                        # corners). Keep the stable global reference on the bright floor/walls, so
                        # long resters there are not absorbed/fragmented by the shorter window.
                        dark_cut = float(tracking_flag(self.settings, "resting_enhance_cutoff"))
                        ws7_dark = rest_reference < dark_cut
                        for b in rest_blocks:
                            b["ref_use"] = np.where(ws7_dark, b["ref"], rest_reference)
                            if adaptive and b["std"] is not None and rest_thr_map is not None:
                                blk_thr = np.maximum(floor, k * b["std"]).astype(np.float32)
                                b["thr_use"] = np.where(ws7_dark, blk_thr, rest_thr_map)
                            else:
                                b["thr_use"] = rest_thr_map
                            if rest_enh_lut is not None:
                                b["dark_mask"] = b["ref_use"] < cutoff
                                b["ref_enh"] = np.where(b["dark_mask"], cv2.LUT(b["ref_use"], rest_enh_lut), b["ref_use"])
                use_blocks = rest_blocks is not None

                # Initialize video variables
                cap = cv2.VideoCapture(self.video_path)

                moving_objects_forward = [[] for i in range(self.total_number_frames)]
                moving_objects_backward = [[] for i in range(self.total_number_frames)]
                resting_objects = [[] for i in range(self.total_number_frames)]

                ####### Start reading the video 
                frame_idx = 0       
                while frame_idx < self.total_number_frames-1:
                    suc,frame = cap.read()
                    if frame_idx%1000 == 0:
                        if debug_mode:
                            #multi_logger = MultiLogger(None, 'logfile.log')
                            progress_bar(frame_idx, self.total_number_frames, bar_length=20)
                    
                    if suc == True:
                        frame_gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_gray = cv2.absdiff(frame_gray_raw,background)

                        # WS1: resting difference image. Directional subtraction against the
                        # bright empty-cage reference when enabled, else the original absdiff frame.
                        # WS7: when active, pick the time-local block reference for this frame.
                        if use_blocks:
                            blk = rest_blocks[min(frame_idx // wframes, len(rest_blocks) - 1)]
                            cur_ref, cur_thr = blk["ref_use"], blk["thr_use"]
                            cur_ref_enh, cur_dmask = blk.get("ref_enh"), blk.get("dark_mask")
                        else:
                            cur_ref, cur_thr = rest_reference, rest_thr_map
                            cur_ref_enh, cur_dmask = rest_reference_enh, rest_dark_mask

                        if cur_ref is not None:
                            if rest_enh_lut is not None and cur_ref_enh is not None:
                                frame_enh = np.where(cur_dmask, cv2.LUT(frame_gray_raw, rest_enh_lut), frame_gray_raw)
                                rest_gray = cv2.subtract(cur_ref_enh, frame_enh)
                            else:
                                rest_gray = cv2.subtract(cur_ref, frame_gray_raw)
                        else:
                            rest_gray = frame_gray

                        if frame_idx > 1:
                            # Get the moving objects forward
                            moving = self.get_centroids_moving_objects(frame_gray,prev_gray,self.settings["seg_moving"])
                            if len(moving)>0:
                                for centroid_moving in moving:
                                    if self.point_inside_cage(self.cage_border_points,centroid_moving):
                                        if len(moving_objects_forward[frame_idx+1])>0:
                                            D = dist.cdist(np.array(centroid_moving).reshape(1,-1), moving_objects_forward[frame_idx+1])
                                            if D[0].min(axis=0) > dedup_moving:
                                                moving_objects_forward[frame_idx+1].append(centroid_moving)
                                        else:
                                            moving_objects_forward[frame_idx+1].append(centroid_moving)

                            # Get the moving objects backward
                            moving = self.get_centroids_moving_objects(prev_gray,frame_gray,self.settings["seg_moving"])
                            if len(moving)>0:
                                for centroid_moving in moving:
                                    if self.point_inside_cage(self.cage_border_points,centroid_moving):
                                        if len(moving_objects_backward[frame_idx])>0:
                                            D = dist.cdist(np.array(centroid_moving).reshape(1,-1), moving_objects_backward[frame_idx])
                                            if D[0].min(axis=0) > dedup_moving :
                                                moving_objects_backward[frame_idx].append(centroid_moving)
                                        else:
                                            moving_objects_backward[frame_idx].append(centroid_moving)

                            # Get resting objects
                            centroids_still = self.get_centroids_still_objects(rest_gray,self.settings["seg_resting"],cur_thr)
                            if len(centroids_still)>0:
                                for centroid in centroids_still:
                                    if self.point_inside_cage(self.cage_border_points,centroid):
                                        if len(resting_objects[frame_idx])>0:
                                            D = dist.cdist(np.array(centroid).reshape(1,-1), resting_objects[frame_idx])
                                            if D[0].min(axis=0) > dedup_resting :
                                                resting_objects[frame_idx].append(centroid)
                                        else:
                                            resting_objects[frame_idx].append(centroid)        
                            
                        frame_idx += 1
                        prev_gray = frame_gray
                    else:
                        # A failed read must still advance frame_idx, or this loop spins on
                        # cap.read() forever: `frame_idx < self.total_number_frames-1` never
                        # becomes false once frame_idx stops moving. This happens for real on any
                        # truncated/corrupted video whose container metadata (CAP_PROP_FRAME_COUNT,
                        # read at __init__ time into self.total_number_frames) overstates how many
                        # frames actually exist -- confirmed live: a 96MB segment whose header still
                        # claims a full 300MB/20-min segment's frame count hung this exact loop for
                        # 40+ minutes across 6 separate attempts, spinning >16 million cap.read()
                        # calls in 20 seconds once real data ran out, frame_idx frozen throughout
                        # (see DEVLOG). Skipping the frame (no detections recorded for it, same as
                        # if it were transiently unreadable) lets a bad file fail fast instead of
                        # hanging the whole worker -- and this branch is simply never taken for any
                        # video that reads cleanly to self.total_number_frames-1, so it changes
                        # nothing for already-valid tracking output.
                        frame_idx += 1
                cap.release()
                

                # Save intermediate variables

                self.resting_objects = resting_objects
                self.moving_objects_forw = moving_objects_forward
                moving_objects_backward.reverse()
                self.moving_objects_back = moving_objects_backward

                if debug_mode:
                    with open(self.video_object_path, 'wb') as f:
                        pickle.dump(self, f)
                print("Finished segmentation of video without error")
            else:
                print("background image missing")

        else: #Skip segmentation
            print("Segmentation already done")
            
########################## Check if a point inside the user-defined cage borders
    def point_inside_cage(self,border,coord):
        if border is None:
            return False

        # The cage border is constant for a whole video, but this runs for every detected centroid
        # on every frame (millions of calls). Build the int32 polygon once and cache it (keyed by the
        # border object's identity); only the cv2.pointPolygonTest below then runs per call.
        cache = getattr(self, "_cage_poly_cache", None)
        if cache is None or cache[0] is not border:
            points = []
            try:
                for p in border:
                    if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                        points.append((int(p[0]), int(p[1])))
            except Exception:
                points = []
            poly = None
            # Need at least 3 points for a valid polygon.
            if len(points) >= 3:
                try:
                    poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                except Exception:
                    poly = None
            cache = (border, poly)
            self._cage_poly_cache = cache

        poly = cache[1]
        if poly is None:
            return False
        try:
            return cv2.pointPolygonTest(poly, (float(coord[0]), float(coord[1])), False) >= 0
        except Exception:
            return False

########################## Within-video resting reference + std (WS1/WS5) ####################
    def _session_key(self):
        """Recording-session key = this segment's video name with the trailing _NNNNN segment
        index stripped, i.e. {Cage}_{YYYYMMDD}_{HHMMSS}. Segments of one session share it."""
        return re.sub(r'_\d{5}$', '', self.video_name)

    def _session_bg_cache_path(self):
        return os.path.join(self.folder_temp, "bg_cache_" + self._session_key() + ".pkl")

    def _load_session_bg_cache(self, path, need_std):
        try:
            if not os.path.isfile(path):
                return None, None
            with open(path, 'rb') as f:
                data = pickle.load(f) or {}
            ref = data.get('ref')
            std = data.get('std')
            if need_std and std is None:
                return None, None
            return ref, std
        except Exception:
            return None, None

    def _save_session_bg_cache(self, path, ref, std):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            existing = {}
            if os.path.isfile(path):
                try:
                    with open(path, 'rb') as f:
                        existing = pickle.load(f) or {}
                except Exception:
                    existing = {}
            existing['ref'] = ref
            if std is not None or 'std' not in existing:
                existing['std'] = std
            with open(path, 'wb') as f:
                pickle.dump(existing, f)
        except Exception:
            pass

    def _get_within_video_rest_stats(self, need_std=False):
        """Session-cache wrapper around the per-video resting-stat builder.

        Default (``bg_session_cache`` off): delegates straight through -> the original per-video
        behaviour, byte-identical. When on: reuse ONE bright-percentile reference/std across all
        segments of a recording session via ``temp_data/bg_cache_{session_key}.pkl`` instead of
        building one per segment. That is a deliberate speed/consistency tradeoff the user opts
        into (segments 2..N then use the session reference rather than their own, which differs
        from per-segment), so it is off by default to keep the frozen numerics unchanged.
        """
        if not tracking_flag(self.settings, "bg_session_cache"):
            return self._get_within_video_rest_stats_impl(need_std=need_std)
        cache_path = self._session_bg_cache_path()
        ref, std = self._load_session_bg_cache(cache_path, need_std)
        if ref is not None and (not need_std or std is not None):
            return ref, std
        ref, std = self._get_within_video_rest_stats_impl(need_std=need_std)
        if ref is not None:
            self._save_session_bg_cache(cache_path, ref, std)
        return ref, std

    def _get_within_video_rest_stats_impl(self, need_std=False):
        """Build (and cache) the within-video resting statistics.

        Samples frames across this video once and derives, per pixel:
          - the bright percentile -> mosquito-free ("empty cage") value (WS1). Because mosquitoes
            are dark, the high percentile is the empty value at every pixel, matching this video's
            exact mesh/structure. Cached to images_mortality/<video>_rest.png.
          - the temporal std (WS5, only when ``need_std``) -> how much each pixel naturally
            fluctuates. Open mesh/floor barely vary; high-contrast cage-structure edges vary a lot.
            Cached to images_mortality/<video>_reststd.png, scaled by _REST_STD_SCALE.

        Either cache file can be deleted to force that statistic to rebuild after tuning. Returns
        ``(reference_uint8_or_None, std_float32_or_None)``; ``std`` is None unless ``need_std``.
        """
        img_dir = self.folder_analysis + "/images_mortality/"
        rest_path = img_dir + self.video_name + "_rest.png"
        std_path = img_dir + self.video_name + "_reststd.png"

        ref = None
        if os.path.isfile(rest_path):
            ref = cv2.cvtColor(cv2.imread(rest_path), cv2.COLOR_BGR2GRAY)
        std = None
        if need_std and os.path.isfile(std_path):
            std = cv2.cvtColor(cv2.imread(std_path), cv2.COLOR_BGR2GRAY).astype(np.float32) / _REST_STD_SCALE

        need_ref_build = ref is None
        need_std_build = need_std and std is None
        if not need_ref_build and not need_std_build:
            return ref, std

        block = self.settings.get("tracking", {}) or {}
        pct = block.get("resting_reference_percentile", 90)
        n_samples = block.get("resting_reference_samples", 80)
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            cap.release()
            print("Cannot build resting statistics for " + self.video_name + " (no frames)")
            return ref, std
        idxs = np.linspace(0, total - 2, min(int(n_samples), total - 1)).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap.release()
        if len(frames) < 3:
            return ref, std
        stack = np.stack(frames).astype(np.float32)
        if need_ref_build:
            ref = np.percentile(stack, pct, axis=0).astype(np.uint8)
            try:
                cv2.imwrite(rest_path, ref)
            except Exception:
                pass
            print("Built within-video resting reference (p%d over %d frames) for %s" % (pct, len(frames), self.video_name))
        if need_std_build:
            # WS5: BRIGHT-frame std = variation of the EMPTY cage only, NOT the raw temporal std.
            # A mosquito that repeatedly perches on one spot (esp. the speaker net) makes that pixel
            # flick dark<->bright, inflating the raw std so k*std would SUPPRESS the very mosquito we
            # want. Restricting the std to frames where the pixel is at/above its median (mosquitoes
            # are dark, so >= median ~= mosquito-free) measures only genuine background/structure
            # flicker: habitual resters survive while flickering cage structure is still suppressed.
            # The cutoff percentile controls how long a rest is tolerated: std is taken over frames
            # where the pixel is >= its per-pixel `std_pct` percentile (empty-cage frames). p50
            # (median) only protects resters present <50% of the clip and so progressively loses
            # long/low-activity resters; p75 protects up to ~75% occupancy.
            std_pct = block.get("resting_std_percentile", 75)
            cut = np.percentile(stack, std_pct, axis=0, keepdims=True)
            mask = stack >= cut
            cnt = np.maximum(mask.sum(axis=0), 1)
            mean_up = (stack * mask).sum(axis=0) / cnt
            std = np.sqrt((((stack - mean_up) ** 2) * mask).sum(axis=0) / cnt)
            try:
                cv2.imwrite(std_path, np.clip(std * _REST_STD_SCALE, 0, 255).astype(np.uint8))
            except Exception:
                pass
            print("Built within-video resting BRIGHT-frame std map over %d frames for %s" % (len(frames), self.video_name))
        return ref, std

    def _get_within_video_rest_reference(self):
        """Back-compat wrapper (WS1): return only the bright-percentile resting reference."""
        ref, _ = self._get_within_video_rest_stats(need_std=False)
        return ref

    def _get_block_rest_stats(self, window_frames, need_std=False):
        """WS7: time-local resting reference/std in temporal blocks of ``window_frames``.

        Samples frames within each contiguous block and computes that block's bright-percentile
        reference (and bright-frame std), so the 'empty cage' tracks slow drift/shimmer that a
        single global reference cannot. Cached per window to images_mortality/<video>_blocks_w*.pkl.
        Returns an ordered list of dicts {start, end, ref, std} (std None unless need_std), or None.
        """
        cache = (self.folder_analysis + "/images_mortality/" + self.video_name +
                 "_blocks_w%d_s%d.pkl" % (window_frames, 1 if need_std else 0))
        if os.path.isfile(cache):
            try:
                with open(cache, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        block = self.settings.get("tracking", {}) or {}
        pct = block.get("resting_reference_percentile", 90)
        std_pct = block.get("resting_std_percentile", 75)
        per_block = min(int(block.get("resting_reference_samples", 80)), 60)
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            cap.release()
            return None
        n_blocks = max(1, int(np.ceil(total / float(window_frames))))
        blocks = []
        for bi in range(n_blocks):
            s = bi * window_frames
            e = min((bi + 1) * window_frames, total)
            if e - s < 3:
                continue
            idxs = np.linspace(s, e - 1, min(per_block, e - s)).astype(int)
            frames = []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if ok:
                    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            if len(frames) < 3:
                continue
            stack = np.stack(frames).astype(np.float32)
            ref = np.percentile(stack, pct, axis=0).astype(np.uint8)
            std = None
            if need_std:
                cut = np.percentile(stack, std_pct, axis=0, keepdims=True)
                mask = stack >= cut
                cnt = np.maximum(mask.sum(axis=0), 1)
                mean_up = (stack * mask).sum(axis=0) / cnt
                std = np.sqrt((((stack - mean_up) ** 2) * mask).sum(axis=0) / cnt)
            blocks.append({"start": s, "end": e, "ref": ref, "std": std})
        cap.release()
        if not blocks:
            return None
        try:
            with open(cache, "wb") as f:
                pickle.dump(blocks, f)
        except Exception:
            pass
        print("Built WS7 time-local reference: %d blocks of %d frames for %s" %
              (len(blocks), window_frames, self.video_name))
        return blocks

########################## Resolve blob-splitting params (WS2) ####################
    def _blob_split_params(self, settings, max_length):
        """Return (split_enabled, single_mosquito_area) for the given seg sub-settings.

        ``single_blob_area`` may be set per seg block in the YAML (the overhaul sets 150);
        otherwise it is estimated from ``max_length``. Splitting is only active when
        ``tracking.split_blobs`` is true. Resolved ONCE per seg sub-block and memoized (keyed by
        the sub-settings dict identity) so it is not recomputed on every frame — same value,
        just computed once at setup.
        """
        cache = getattr(self, '_blob_split_cache', None)
        if cache is None:
            cache = self._blob_split_cache = {}
        key = id(settings)
        if key in cache:
            return cache[key]
        if not tracking_flag(self.settings, "split_blobs"):
            result = (False, None)
        else:
            single_area = settings.get("single_blob_area", None)
            if not single_area or single_area <= 0:
                single_area = max(8.0, (max_length ** 2) / 4.0)
            result = (True, single_area)
        cache[key] = result
        return result

########################## Segment "resting" object ####################
    def get_centroids_still_objects(self,frame_gray,settings,thr_map=None):
        # Parameters resting :
        gray_tresh = settings["gray_tresh"]
        max_elongation_ratio = settings["max_elongation_ratio"]
        max_length = settings["max_length"]
        min_length = settings["min_length"]

        # Conventio
        if thr_map is not None:
            # WS5: per-pixel adaptive threshold (max(floor, k*std)) instead of the flat gray_tresh.
            thresh = ((frame_gray > thr_map).astype(np.uint8)) * 255
        else:
            _, thresh = cv2.threshold(frame_gray, gray_tresh, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        split, single_area = self._blob_split_params(settings, max_length)
        inputCentroids = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Save the centroid of the bounding rectangle
            if max([w/h,h/w])< max_elongation_ratio and max([h,w])<max_length and min([h,w])>min_length:
                inputCentroids.append((x+w/2, y+h/2))
            elif split and max([w/h,h/w]) < max_elongation_ratio and max([h,w]) >= max_length and min([h,w]) > min_length:
                # Too large for a single mosquito -> likely a merged cluster; split it (WS2).
                inputCentroids.extend(split_merged_contour(dilated.shape, contour, single_area, min_length))

        return inputCentroids

########################## Segment "moving" objects ####################
    def get_centroids_moving_objects(self,frame_gray,prev_gray,settings):
        # Parameters moving :
        gray_tresh = settings["gray_tresh"]
        max_elongation_ratio = settings["max_elongation_ratio"]
        max_length = settings["max_length"]
        min_length = settings["min_length"]

        diff_gray = cv2.subtract(prev_gray,frame_gray)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, gray_tresh, 255, cv2.THRESH_BINARY) # Modified the thresh value from 12 to 5
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        split, single_area = self._blob_split_params(settings, max_length)
        inputCentroids = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Save the centroid of the bounding rectangle
            if max([w/h,h/w])< max_elongation_ratio and max([h,w])<max_length and min([h,w])>min_length:
                inputCentroids.append((x+w/2, y+h/2))
            elif split and max([w/h,h/w]) < max_elongation_ratio and max([h,w]) >= max_length and min([h,w]) > min_length:
                # Too large for a single mosquito -> likely a merged cluster; split it (WS2).
                inputCentroids.extend(split_merged_contour(dilated.shape, contour, single_area, min_length))
        return inputCentroids

########################## Track resting objects ####################
    def track_resting_obj(self,step_to_force_analyze,debug_mode):
        """ 
        Track resting objects (forward and backward) and update moving_obj
        input : self.resting_objects_forw
        output : self.mosquito_rest_tracks_forw , self.mosquito_moving_objects_forw
        """

        if hasattr(self,'mosquito_rest_tracks_forw')==0 or step_to_force_analyze<=2:

            resting_objects = copy.deepcopy(self.resting_objects)
            for direction in ["forward"]:#["forward","backward"]:
                print("Starting tracking resting objects "+direction)

                # Load parameters
                min_duration_resting_traj = self.settings["track_resting"]["min_duration_resting_traj"]
                dist_moving_obj = self.settings["seg_moving"]["dist_moving_obj"]
                maxDisappeared = self.settings["track_resting"]["maxDisappeared"]

                # Load variables
                
                if direction == "forward":
                    moving_objects = self.moving_objects_forw
                else:
                    resting_objects.reverse()
                    moving_objects = self.moving_objects_back

                # Initialize new data structure for tracking. Resting stays on the original
                # greedy + tight-gate tracker: it is correct for stationary objects (the motion
                # model / a looser gate fragmented resting tracks in A/B). Resting gains come
                # from the detection side (within-video reference), not the tracker.
                objs_resting_tracked = resting_tracker(self.settings["track_resting"])
                mosquito_tracks = mosquito_obj_tracker("resting")


                # Start tracking loop
                frame_idx = 0
                while frame_idx < len(resting_objects):

                    if frame_idx%1000 == 0:
                        if debug_mode:
                            progress_bar(frame_idx, len(resting_objects), bar_length=20)
                        
                    centroids_still = resting_objects[frame_idx]

                    ########## Track resting objects
                    objects_resting,new_IDs,lost_IDs,lost_objects = objs_resting_tracked.update(centroids_still)
                    
                    ########### Initialize new IDs
                    if len(new_IDs)>0: 
                        for new_ID in new_IDs:
                            mosquito_tracks.add_new_track(new_ID,[],[],np.nan,np.nan,np.nan,np.nan)
                            #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)
                    ######## Save resting object tracked #######
                    if len(objects_resting)>0:
                        for (objectID, centroid) in objects_resting.items():
                            mosquito_tracks.add_points_to_track(objectID,centroid,frame_idx) # Add point to trajectory      
                            #add_points_to_track(self,track_ID,coord_to_add,time_stamp_to_add)       
 
                    ###### Keep or remove finished IDs
                    if len(lost_IDs)>0:
                        for lost_ID in lost_IDs:
                            if len(mosquito_tracks.objects[lost_ID]["time_stamp"]) < maxDisappeared + min_duration_resting_traj: # If the object was there for less that 2 seconds
                                
                                #Save only the coord before dissapearing to moving objects
                                for i in np.arange(len(mosquito_tracks.objects[lost_ID]["time_stamp"])-maxDisappeared): 
                                    t = mosquito_tracks.objects[lost_ID]["time_stamp"][i]
                                    obj_to_add = mosquito_tracks.objects[lost_ID]["coordinates"][i]
                                    if len(moving_objects[t])>0:
                                        D = dist.cdist(np.array(obj_to_add).reshape(1,-1), moving_objects[t])
                                        if D[0].min(axis=0) > dist_moving_obj :
                                                moving_objects[t].append(obj_to_add)
                                    else:
                                        moving_objects[t].append(obj_to_add)

                                # Delete the track
                                mosquito_tracks.remove_track(lost_ID)
                            else:
                                mosquito_tracks.remove_points_from_track(lost_ID,[-maxDisappeared,-1]) # Trim the traj with the dissapeared tail

                    frame_idx += 1

                if direction == "forward":
                    self.mosquito_rest_tracks_forw = mosquito_tracks
                    self.moving_objects = moving_objects
                    
                    # Save intermediate resting tracks for comparison
                    tracking_resting_folder = os.path.join(self.folder_analysis, "tracking_resting")
                    os.makedirs(tracking_resting_folder, exist_ok=True)
                    resting_file = os.path.join(tracking_resting_folder, f"resting_{self.video_name}.pkl")
                    with open(resting_file, 'wb') as f:
                        pickle.dump(mosquito_tracks, f)
                    print(f"Saved intermediate resting tracks to {resting_file}")
                else:
                    self.mosquito_rest_tracks_back = mosquito_tracks
                    self.moving_objects_back = moving_objects
                # Save

                if debug_mode:
                    with open(self.video_object_path, 'wb') as f:
                        pickle.dump(self, f)
                print("Finished tracking resting objects "+direction)
        else:
            print("Tracking resting objects already done")

########################## Track moving objects ####################
    def track_moving_obj(self,step_to_force_analyze,debug_mode):
        """ 
        Track moving objects (forward and backward) 
        input : self.moving_objects_forw
        output : self.mosquito_mov_tracks_forw
        """

        if hasattr(self,'mosquito_mov_tracks_forw')==0 or step_to_force_analyze<=2:
            for direction in ["forward"]:#["forward","backward"]:

                print("Starting tracking moving objects "+direction)

                # Load parameters
                min_duration_moving_traj = self.settings["track_moving"]["min_duration_moving_traj"]
                maxDisappeared = self.settings["track_moving"]["maxDisappeared"]

                # Load variables
                if direction == "forward":
                    moving_objects = self.moving_objects_forw
                else:
                    moving_objects = self.moving_objects_back

                # Initialize new data structure
                objs_moving_tracked = moving_tracker(
                    self.settings.get("track_moving"),
                    assignment=tracking_flag(self.settings, "assignment"),
                    motion_model=tracking_flag(self.settings, "motion_model"),
                )
                mosquito_tracks = mosquito_obj_tracker("moving")

                ### Start tracking
                frame_idx = 0
                while frame_idx < len(moving_objects):

                    # Run the tracker
                    centroids_moving = moving_objects[frame_idx]
                    objects_moving,new_IDs,lost_IDs,lost_objects = objs_moving_tracked.update(centroids_moving)
                    
                    if len(new_IDs)>0: ########### Start tracking new disappeared IDs               
                        for new_ID in new_IDs:
                            mosquito_tracks.add_new_track(new_ID,[],[],np.nan,np.nan,np.nan,np.nan)
                            #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)


                    if len(objects_moving)>0:
                        for (objectID, centroid) in objects_moving.items():            
                            mosquito_tracks.add_points_to_track(objectID,centroid,frame_idx) # Add point to trajectory        
                
                    if len(lost_IDs)>0:
                        for lost_ID in lost_IDs:
                            if len(mosquito_tracks.objects[lost_ID]["time_stamp"]) < maxDisappeared + min_duration_moving_traj: # If the object was there for less that 2 seconds
                                # Delete the track
                                mosquito_tracks.remove_track(lost_ID)
                            else:
                                mosquito_tracks.remove_points_from_track(lost_ID,[-maxDisappeared,-1]) # Trim the traj with the dissapeared tail
                                
                    frame_idx += 1

                if direction == "forward":
                    self.mosquito_mov_tracks_forw = mosquito_tracks
                    
                    # Save intermediate moving tracks for comparison
                    tracking_moving_folder = os.path.join(self.folder_analysis, "tracking_moving")
                    os.makedirs(tracking_moving_folder, exist_ok=True)
                    moving_file = os.path.join(tracking_moving_folder, f"moving_{self.video_name}.pkl")
                    with open(moving_file, 'wb') as f:
                        pickle.dump(mosquito_tracks, f)
                    print(f"Saved intermediate moving tracks to {moving_file}")
                else:
                    self.mosquito_mov_tracks_back = mosquito_tracks
                # Save 
                if debug_mode:
                    with open(self.video_object_path, 'wb') as f:
                        pickle.dump(self, f)
                print("Finished tracking moving objects "+direction)

            # WS4: recover moving objects the forward-only pass missed.
            self.merge_forward_backward_tracks(debug_mode)

        else:
            print("Tracking moving object already done")

########################## Merge forward + backward moving tracks (WS4) ####################
    def merge_forward_backward_tracks(self, debug_mode):
        """Recover moving objects missed by the forward-only tracker.

        The forward pass cannot start a track for an object that is already moving fast when it
        first appears. The segmentation already produced time-reversed detections
        (``self.moving_objects_back``); here we track them, express the resulting tracks back in
        real-frame coordinates, and add any backward track that is not already covered (in time
        and space) by a forward track. Gated by ``tracking.use_backward_pass`` (off => no-op,
        original behaviour). Runs before the assembly stages so recovered tracks are assembled.
        """
        if not tracking_flag(self.settings, "use_backward_pass"):
            return
        if getattr(self, "mosquito_mov_tracks_forw", None) is None or not hasattr(self, "moving_objects_back"):
            return

        print("Starting backward moving pass (WS4)")
        min_duration_moving_traj = self.settings["track_moving"]["min_duration_moving_traj"]
        maxDisappeared = self.settings["track_moving"]["maxDisappeared"]
        gate = self.settings["track_moving"]["maxdisttracking"]

        objs_back = moving_tracker(
            self.settings.get("track_moving"),
            assignment=tracking_flag(self.settings, "assignment"),
            motion_model=tracking_flag(self.settings, "motion_model"),
        )
        back_tracks = mosquito_obj_tracker("moving")
        n = len(self.moving_objects_back)
        for p in range(n):
            objects_moving, new_IDs, lost_IDs, lost_objects = objs_back.update(self.moving_objects_back[p])
            for new_ID in new_IDs:
                back_tracks.add_new_track(new_ID, [], [], np.nan, np.nan, np.nan, np.nan)
            for (objectID, centroid) in objects_moving.items():
                back_tracks.add_points_to_track(objectID, centroid, p)
            for lost_ID in lost_IDs:
                if len(back_tracks.objects[lost_ID]["time_stamp"]) < maxDisappeared + min_duration_moving_traj:
                    back_tracks.remove_track(lost_ID)
                else:
                    back_tracks.remove_points_from_track(lost_ID, [-maxDisappeared, -1])

        # Index forward track positions by real frame (with +/-1 tolerance applied at lookup).
        forward_by_frame = {}
        for tr in self.mosquito_mov_tracks_forw.objects.values():
            for coord, t in zip(tr["coordinates"], tr["time_stamp"]):
                forward_by_frame.setdefault(t, []).append(coord)

        existing_ids = list(self.mosquito_mov_tracks_forw.objects.keys())
        next_id = (max(existing_ids) + 1) if existing_ids else 0
        added = 0
        for tr in back_tracks.objects.values():
            ts = tr["time_stamp"]
            if len(ts) == 0:
                continue
            # Reversed detection index p -> real frame (n-1-p); reverse to chronological order.
            real_frames = [n - 1 - p for p in ts][::-1]
            real_coords = list(tr["coordinates"])[::-1]
            covered = 0
            for coord, rf in zip(real_coords, real_frames):
                fwd_pts = []
                for f in (rf - 1, rf, rf + 1):
                    fwd_pts.extend(forward_by_frame.get(f, []))
                if fwd_pts and dist.cdist(np.array(coord).reshape(1, -1), np.array(fwd_pts)).min() <= gate:
                    covered += 1
            if covered / len(real_frames) < 0.5:  # mostly new -> add as a recovered track
                self.mosquito_mov_tracks_forw.add_new_track(
                    next_id, real_coords, real_frames, np.nan, np.nan, np.nan, np.nan)
                next_id += 1
                added += 1
        print("Backward pass recovered " + str(added) + " moving tracks not seen forward")

        if debug_mode:
            with open(self.video_object_path, 'wb') as f:
                pickle.dump(self, f)

########################## Assemble resting and moving tracks (find taking off and landing match) ####################
    def assemble_resting_and_moving_tracks(self,step_to_force_analyze,time_window_search,max_distance_search,debug_mode):
        """ 
        Track moving objects (forward and backward) 
        input : self.mosquito_rest_tracks_forw self.mosquito_mov_tracks_forw
        output : self.mosquito_tracks_forw
        """

        # #        To remove after        ##
        # settings_file = self.folder_analysis+"buzzwatch_track_settings.yml"
        # if os.path.isfile(settings_file):
        #     with open(settings_file, 'r') as file:
        #         settings = yaml.safe_load(file)
        #         self.settings = settings
        # ##                              ##

        if hasattr(self,'mosquito_tracks_forw')==0 or step_to_force_analyze<=4:
            print(f"DEBUG: Entering assembly - step_to_force_analyze={step_to_force_analyze}, hasattr mosquito_tracks_forw={hasattr(self,'mosquito_tracks_forw')}")
            for direction in ["forward"]:#["forward","backward"]:
                print("Start assembling resting and moving tracks "+direction)

                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                    print(f"DEBUG: Loaded tracks - resting: {resting_object_tracks.number_objects}, moving: {moving_object_tracks.number_objects}")
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back


                # Get useable indexes of all tracks
                list_of_ID_still = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())
                print(f"DEBUG: About to loop - resting IDs: {len(list_of_ID_still)}, moving IDs: {len(list_of_ID_moving)}")

                # Initialize distance and time matrices. _SparseFillMatrix (not np.full): on the
                # densest sessions len(list_of_ID_still)/len(list_of_ID_moving) can exceed 20,000,
                # where a dense array here is multiple GB and has crashed workers outright -- see
                # DEVLOG 2026-09-11. Numerically identical to the dense array for every consumer
                # (same fill-in loop below, unchanged; only the storage container differs).
                landing_matrix_dist = _SparseFillMatrix(len(list_of_ID_still),len(list_of_ID_moving),10000.0)
                takeoff_matrix_dist = _SparseFillMatrix(len(list_of_ID_still),len(list_of_ID_moving),10000.0)
                landing_matrix_time = _SparseFillMatrix(len(list_of_ID_still),len(list_of_ID_moving),0)
                takeoff_matrix_time = _SparseFillMatrix(len(list_of_ID_still),len(list_of_ID_moving),0)

                # Perf: precompute timestamp -> index for every moving track once, instead of
                # repeating an O(len(track)) list `.index()`/`in` scan inside the O(N*M) loop
                # below (was the dominant cost on dense experiments: validated numerically
                # equivalent to the original per-pair scan, see DEVLOG).
                mov_ts_index = {
                    mov_id: {t: idx for idx, t in enumerate(moving_object_tracks.objects[mov_id]["time_stamp"])}
                    for mov_id in list_of_ID_moving
                }
                half_window = int(time_window_search/2)

                # Loop over the resting objects
                for i,rest_id in enumerate(list_of_ID_still):
                    if debug_mode:
                        progress_bar(i, len(list_of_ID_still), bar_length=20)

                    rest_track = resting_object_tracks.objects[rest_id]

                    # Take-off
                    # If already matched
                    if np.isnan(rest_track["end_boot"])==0:
                        t_takeoff_tar = np.nan
                    else:
                        # If resting tracks goes to the end of the video
                        if rest_track["time_stamp"][-1] == self.total_number_frames-1: # Goes to the end of the video
                            rest_track["end_boot"] = -1 # Mark as goes to the end
                            t_takeoff_tar = np.nan
                        else:
                            # time of the end of rest_track
                            t_takeoff_tar = rest_track["time_stamp"][-1]

                    # Landing
                    # If already matched
                    if np.isnan(rest_track["start_boot"])==0:
                        t_landing_tar = np.nan
                        #print("resting id "+str(rest_id)+" was already resting at t=0")
                    else:
                        # rest_tracks starts at the start of the video
                        if rest_track["time_stamp"][0] < 10:
                            t_landing_tar = np.nan
                            # Mark as done
                            rest_track["start_boot"] = -1
                        else:
                            # time of the start of rest_track
                            t_landing_tar = rest_track["time_stamp"][0]

                    rest_first_ts = rest_track["time_stamp"][0]
                    rest_last_ts = rest_track["time_stamp"][-1]
                    rest_first_coord = rest_track["coordinates"][0]
                    rest_last_coord = rest_track["coordinates"][-1]

                    landing_active = np.isnan(t_landing_tar) == 0
                    takeoff_active = np.isnan(t_takeoff_tar) == 0
                    if landing_active:
                        landing_lo = int(t_landing_tar) - half_window
                        landing_hi = int(t_landing_tar) + half_window
                    if takeoff_active:
                        takeoff_lo = int(t_takeoff_tar) - half_window
                        takeoff_hi = int(t_takeoff_tar) + half_window

                    # Loop over the moving objects
                    for j,mov_id in enumerate(list_of_ID_moving):
                        mov_track = moving_object_tracks.objects[mov_id]
                        ts_index = mov_ts_index[mov_id]

                        # Look for landing match : Look over all time points centered on t_landing_tar
                        # mov_track_end is free
                        if landing_active and np.isnan(mov_track["end_boot"]) == 1 :
                            mov_ts = mov_track["time_stamp"]
                            # If the rest_track and mov_track are well positioned for landing
                            if len(mov_ts)>0:
                                if mov_ts[0] < rest_first_ts:
                                    already_found = 0
                                    for t_landing in range(landing_lo, landing_hi):
                                        t_landing_mov = ts_index.get(t_landing)
                                        if t_landing_mov is not None:
                                            # The potential mov_track end  is before the end of the rest_tracl
                                            if t_landing < rest_last_ts:
                                                mov_coord = mov_track["coordinates"][t_landing_mov]
                                                dx = rest_first_coord[0] - mov_coord[0]
                                                dy = rest_first_coord[1] - mov_coord[1]
                                                D_landing = (dx*dx + dy*dy) ** 0.5
                                                if already_found == 0:
                                                    D_landing_min = D_landing
                                                    t_landing_min = t_landing_mov
                                                    already_found = 1
                                                else:
                                                    if D_landing < D_landing_min:
                                                        D_landing_min = D_landing
                                                        t_landing_min = t_landing_mov

                                    if already_found == 1:
                                        landing_matrix_dist[i][j] = D_landing_min
                                        landing_matrix_time[i][j] = t_landing_min

                        # Look for taking-off match : Look over all time points centered on t_takeoff_tar

                        if takeoff_active and np.isnan(mov_track["start_boot"]) == 1:
                            already_found = 0
                            mov_ts = mov_track["time_stamp"]
                            # If the rest_track and mov_track are well positioned for take-off
                            if len(mov_ts)>0:
                                if mov_ts[-1] > rest_last_ts:
                                    for t_takeoff in range(takeoff_lo, takeoff_hi):
                                        t_takeoff_mov = ts_index.get(t_takeoff)
                                        if t_takeoff_mov is not None:
                                            # The potential mov_track end  is after the start of the rest_tracl
                                            if t_takeoff > rest_first_ts:
                                                mov_coord = mov_track["coordinates"][t_takeoff_mov]
                                                dx = rest_last_coord[0] - mov_coord[0]
                                                dy = rest_last_coord[1] - mov_coord[1]
                                                D_takeoff = (dx*dx + dy*dy) ** 0.5
                                                if already_found == 0:
                                                    D_takeoff_min = D_takeoff
                                                    t_takeoff_min = t_takeoff_mov
                                                    already_found = 1
                                                else:
                                                    if D_takeoff < D_takeoff_min:
                                                        D_takeoff_min = D_takeoff
                                                        t_takeoff_min = t_takeoff_mov

                                    if already_found == 1:
                                        takeoff_matrix_dist[i][j] = D_takeoff_min
                                        takeoff_matrix_time[i][j] = t_takeoff_min

                # Initialize temporal tracking for trajectory consistency
                if not hasattr(self, '_previous_takeoff_matches'):
                    self._previous_takeoff_matches = {}
                if not hasattr(self, '_previous_landing_matches'):
                    self._previous_landing_matches = {}

                pairing_takeoff = self.find_optimal_matching_resting_moving_from_distance(takeoff_matrix_dist,takeoff_matrix_time,previous_matches=self._previous_takeoff_matches)
                pairing_landing = self.find_optimal_matching_resting_moving_from_distance(landing_matrix_dist,landing_matrix_time,previous_matches=self._previous_landing_matches)

                # Updating the tracks with taking-off
                for (i,j,dist_c,time_c) in pairing_takeoff:
                    if dist_c < max_distance_search:
                        rest_id = list_of_ID_still[i]
                        mov_id = list_of_ID_moving[j]

                        # Mark the tracks as attached
                        resting_object_tracks.objects[rest_id]["end_boot"] = mov_id
                        resting_object_tracks.objects[rest_id]["end_type"] = "moving"
                        moving_object_tracks.objects[mov_id]["start_boot"] = rest_id
                        moving_object_tracks.objects[mov_id]["start_type"] = "resting"

                        # Find the right
                        t_rest = resting_object_tracks.objects[rest_id]["time_stamp"][-1]
                        time_c = 0
                        if moving_object_tracks.objects[mov_id]["time_stamp"][time_c] < t_rest:               
                            while moving_object_tracks.objects[mov_id]["time_stamp"][time_c] < t_rest and time_c < len(moving_object_tracks.objects[mov_id]["time_stamp"])-1:
                                time_c += 1
                                #print(time_c)

                        # Create new moving track from the mov_track before taking-off
                        new_coord = moving_object_tracks.objects[mov_id]["coordinates"][0:time_c]
                        new_time_stamp = moving_object_tracks.objects[mov_id]["time_stamp"][0:time_c]
                        new_id = max(list(moving_object_tracks.objects.keys()))+1 # Create non existing index
                        new_start_boot = np.nan #moving_object_tracks.objects[mov_id]["start_boot"]
                        new_start_type = np.nan #moving_object_tracks.objects[mov_id]["start_type"]
                        new_end_boot = np.nan
                        new_end_type = np.nan
                        #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_type,new_start_boot,new_end_boot):
                        moving_object_tracks.add_new_track(new_id,new_coord,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)

                        # Cut-off the mov_track before taking-off
                        moving_object_tracks.remove_points_from_track(mov_id,[0,time_c])

                # Store matches for temporal consistency in next iteration
                for (i,j,dist_c,time_c) in pairing_takeoff:
                    self._previous_takeoff_matches[i] = j

                # Updating the tracks with landing
                for (i,j,dist_c,time_c) in pairing_landing:
                    if dist_c < max_distance_search:
                        rest_id = list_of_ID_still[i]
                        mov_id = list_of_ID_moving[j]

                        # Mark the tracks as attached
                        resting_object_tracks.objects[rest_id]["start_boot"] = mov_id
                        resting_object_tracks.objects[rest_id]["start_type"] = "moving"
                        moving_object_tracks.objects[mov_id]["end_boot"] = rest_id
                        moving_object_tracks.objects[mov_id]["end_type"] = "resting"

                        # Find the right
                        t_rest = resting_object_tracks.objects[rest_id]["time_stamp"][0]

                        time_c = len(moving_object_tracks.objects[mov_id]["time_stamp"])-1
                        if moving_object_tracks.objects[mov_id]["time_stamp"][time_c] > t_rest:               
                            while moving_object_tracks.objects[mov_id]["time_stamp"][time_c] > t_rest and time_c >0:
                                time_c += -1
                                

                        # Create new moving track from the mov_track before taking-off
                        new_coord = moving_object_tracks.objects[mov_id]["coordinates"][time_c:]
                        new_time_stamp = moving_object_tracks.objects[mov_id]["time_stamp"][time_c:]
                        new_id = max(list(moving_object_tracks.objects.keys()))+1 # Create non existing index
                        new_start_boot = np.nan
                        new_start_type = np.nan
                        new_end_boot = np.nan # moving_object_tracks.objects[mov_id]["end_boot"]
                        new_end_type = np.nan #moving_object_tracks.objects[mov_id]["end_type"]
                        #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_type,new_start_boot,new_end_boot):
                        moving_object_tracks.add_new_track(new_id,new_coord,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)

                        # Cut-off the mov_track after landing
                        moving_object_tracks.remove_points_from_track(mov_id,[time_c,-1])

                # Store matches for temporal consistency in next iteration
                for (i,j,dist_c,time_c) in pairing_landing:
                    self._previous_landing_matches[i] = j

                # Save the data 
                if direction == "forward":
                    self.mosquito_rest_tracks_forw = resting_object_tracks
                    self.mosquito_mov_tracks_forw = moving_object_tracks
                else:
                    self.mosquito_rest_tracks_back = resting_object_tracks
                    self.mosquito_mov_tracks_back = moving_object_tracks 
                print("Finished assembling resting-moving "+direction)

            #with open(self.video_object_path, 'wb') as f:
            #    pickle.dump(self, f)

########################## Sort matching indexes resting and moving ####################
    def find_optimal_matching_resting_moving_from_distance(self, distance_matrix, time_matrix, previous_matches=None):
        """
        Track moving objects (forward and backward) 
        input : distance matrix, time matrix, optional previous_matches dict
        output : pairs of resting and moving id to match
        
        Settings (from self.settings):
        - matching_method: 'greedy' or 'optimal' (default: 'greedy')
        - temporal_penalty_weight: cost penalty for switching mosquito ID (default: 10.0, pixels equivalent)
        
        previous_matches: dict mapping rest_id -> mov_id from previous call (for temporal smoothing)
        """
        # Get matching method from settings (default to greedy for backward compatibility)
        matching_method = self.settings.get('matching_method', 'greedy')

        # 'optimal' needs a real dense array (linear_sum_assignment has no sparse-with-implicit-
        # default-fill mode). distance_matrix may now be a _SparseFillMatrix (see
        # assemble_unmatched_moving_tracks/assemble_resting_and_moving_tracks) specifically because
        # densifying it can be multiple GB on the densest sessions -- np.array(distance_matrix,...)
        # on that object wouldn't even produce the right values (it has no numpy buffer protocol),
        # so this must be handled explicitly rather than left to fail. 'optimal' is not the default
        # matching_method and isn't configured anywhere in this codebase's settings today; falling
        # back to greedy on a huge sparse input preserves "always produce a result" without
        # reintroducing the exact OOM this fix removes.
        if matching_method == 'optimal' and isinstance(distance_matrix, _SparseFillMatrix):
            n_rows, n_cols = distance_matrix.shape
            if n_rows * n_cols > 4_000_000:  # ~32MB as float64 -- generous, not a hard hardware limit
                print("Warning: matching_method='optimal' requested but the matrix is %dx%d "
                      "(too large to densify safely) -- falling back to greedy for this call."
                      % (n_rows, n_cols))
                matching_method = 'greedy'
            else:
                distance_matrix = distance_matrix.to_dense()

        if matching_method == 'optimal':
            # Try Hungarian algorithm with optional temporal smoothing
            try:
                from scipy.optimize import linear_sum_assignment

                # Build cost matrix
                cost = np.array(distance_matrix, dtype=float, copy=True)
                large_value = 1e9
                
                # Replace NaN/inf with large value so they won't be selected
                cost[~np.isfinite(cost)] = large_value
                
                # Apply temporal penalty to encourage ID consistency
                if previous_matches is not None and len(previous_matches) > 0:
                    temporal_penalty = self.settings.get('temporal_penalty_weight', 10.0)
                    # For each resting row, if it was previously matched to a moving col,
                    # reduce cost for matching to the same col again
                    for rest_idx in range(cost.shape[0]):
                        if rest_idx in previous_matches:
                            prev_mov_idx = previous_matches[rest_idx]
                            if prev_mov_idx < cost.shape[1]:
                                # Add penalty to all columns except the previous match
                                for mov_idx in range(cost.shape[1]):
                                    if mov_idx != prev_mov_idx and cost[rest_idx, mov_idx] < large_value:
                                        cost[rest_idx, mov_idx] += temporal_penalty
                
                # Solve assignment problem
                row_ind, col_ind = linear_sum_assignment(cost)
                
                # Build matches list, filtering out invalid assignments
                matches = []
                for r, c in zip(row_ind, col_ind):
                    # Skip if cost was forbidden (NaN/inf in original matrix)
                    if cost[r, c] >= large_value:
                        continue
                    dist_val = float(distance_matrix[r, c])
                    time_val = int(time_matrix[r, c])
                    matches.append([int(r), int(c), dist_val, time_val])
                
                return matches
                
            except ImportError:
                # SciPy not available, fall back to greedy
                print("Warning: scipy not available, falling back to greedy matching")
                matching_method = 'greedy'
            except Exception as e:
                # Any other error, fall back to greedy
                print(f"Warning: optimal matching failed ({e}), falling back to greedy")
                matching_method = 'greedy'
        
        # Greedy algorithm (original implementation)
        if matching_method == 'greedy':
            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            rest_id = []
            mov_id = []
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                usedRows.add(row)
                usedCols.add(col)
                rest_id.append(row)
                mov_id.append(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, distance_matrix.shape[0])).difference(usedRows)
            unusedCols = set(range(0, distance_matrix.shape[1])).difference(usedCols)

            # return the indexs matched, with distance and time_diff
            return [[rest_id[f],mov_id[f],distance_matrix[rest_id[f]][mov_id[f]],time_matrix[rest_id[f]][mov_id[f]]] for f in np.arange(len(rest_id))]

########################## Assemble together the unmatched moving trajectories ##############
    def assemble_unmatched_moving_tracks(self,step_to_force_analyze,time_window_search,max_distance_search,debug_mode):
        """
        Track moving objects (forward and backward) 
        input : distance matrix
        output : pairs of resting and moving id to match
        """
         #        To remove after        ##
        # settings_file = self.folder_analysis+"buzzwatch_track_settings.yml"
        # if os.path.isfile(settings_file):
        #     with open(settings_file, 'r') as file:
        #         settings = yaml.safe_load(file)
        #         self.settings = settings
        ##                              ##
        if hasattr(self,'mosquito_tracks_forw')==0 or step_to_force_analyze<=4:
            for direction in ["forward"]:#["forward","backward"]:
                print("Start assembling umatched moving tracks together "+direction)
                

                # Load variables
                if direction == "forward":
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                else:
                    moving_object_tracks = self.mosquito_mov_tracks_back

                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                # _SparseFillMatrix (not np.full): same reasoning as assemble_resting_and_moving_tracks
                # above -- this matrix is square in len(list_of_ID_moving), which is exactly the
                # dimension observed exceeding 20,000 on the densest sessions (see DEVLOG 2026-09-11).
                landing_matrix_dist = _SparseFillMatrix(len(list_of_ID_moving),len(list_of_ID_moving),100000.0)
                landing_matrix_time = _SparseFillMatrix(len(list_of_ID_moving),len(list_of_ID_moving),0)

                # Loop over the moving objects
                for i,mov_id_1 in enumerate(list_of_ID_moving):
                    if debug_mode:
                        progress_bar(i, len(list_of_ID_moving), bar_length=20)

                    # If non empty trajectory (### to remove ###)
                    if len(moving_object_tracks.objects[mov_id_1]["time_stamp"])>0:
                        # If track is already matched
                        if np.isnan(moving_object_tracks.objects[mov_id_1]["end_boot"])==0: # If boot it not matched yet.
                            t_end_tar = np.nan
                        else:
                            # If moving tracks goes to the end of the video 
                            if moving_object_tracks.objects[mov_id_1]["time_stamp"][-1] == self.total_number_frames-1:
                                t_end_tar = np.nan
                                # Mark the track as done
                                moving_object_tracks.objects[mov_id_1]["end_boot"] = -1
                            else:
                                # Get the time of the end of the mov_track_1
                                t_end_tar = moving_object_tracks.objects[mov_id_1]["time_stamp"][-1]

                            # Loop over the moving objects (look for mov_track_2)
                            for j,mov_id_2 in enumerate(list_of_ID_moving):
                                # If non empty trajectory (### to remove ###)
                                if len(moving_object_tracks.objects[mov_id_2]["time_stamp"])>0: 
                                    #If same id or already matched
                                    if mov_id_1 == mov_id_2 or np.isnan(moving_object_tracks.objects[mov_id_2]["start_boot"])==0:
                                        t_start_tar = np.nan
                                    else:
                                        # If moving to match tracks starts at the start of the video
                                        if moving_object_tracks.objects[mov_id_2]["time_stamp"][0] < 10:
                                            t_start_tar = np.nan
                                            # Mark the track as done
                                            moving_object_tracks.objects[mov_id_2]["start_boot"] = -1
                                        else:
                                             # Get the time of the start of the mov_track_2
                                            t_start_tar = moving_object_tracks.objects[mov_id_2]["time_stamp"][0]

                                            # If not too far in time
                                            if np.abs(t_start_tar-t_end_tar) < time_window_search:
                                                # If end of mov_track_2 is after and of mov_track_1
                                                if moving_object_tracks.objects[mov_id_2]["time_stamp"][-1]-t_end_tar>0:
                                                    # If start of mov_track_1 is after and of mov_track_1
                                                    if moving_object_tracks.objects[mov_id_1]["time_stamp"][0] < moving_object_tracks.objects[mov_id_2]["time_stamp"][0]:
                                                        # If start of mov_track_1 is after and of mov_track_1
                                                        if moving_object_tracks.objects[mov_id_1]["time_stamp"][-1] < moving_object_tracks.objects[mov_id_2]["time_stamp"][-1]:
                                                            # If start of mov_track_1 is before end of mov_track_2
                                                            if t_start_tar < t_end_tar:
                                                                # Find the delay in term of mov_track_2 index.
                                                                # Perf: time_stamp is sorted, so this is the same
                                                                # index the original `while ... t_diff += 1` linear
                                                                # scan converges to (first index >= t_end_tar), just
                                                                # via binary search instead of O(len(track)) steps.
                                                                t_diff = bisect.bisect_left(moving_object_tracks.objects[mov_id_2]["time_stamp"], t_end_tar)
                                                                # Save the time delay
                                                                landing_matrix_time[i][j] = t_diff # Shift in the next traj
                                                            else:
                                                                # No time delay
                                                                landing_matrix_time[i][j] = 0
                                                            # Save the euclidian distance end of mov_track_1 and start of mov_track_2
                                                            # (perf: direct formula instead of a single-pair scipy.cdist call --
                                                            # mathematically identical, validated numerically, see DEVLOG)
                                                            coord_1 = moving_object_tracks.objects[mov_id_1]["coordinates"][-1]
                                                            coord_2 = moving_object_tracks.objects[mov_id_2]["coordinates"][0]
                                                            D_landing = ((coord_1[0]-coord_2[0])**2 + (coord_1[1]-coord_2[1])**2) ** 0.5
                                                            landing_matrix_dist[i][j] = D_landing

                # Initialize temporal tracking for moving-to-moving trajectory consistency
                if not hasattr(self, '_previous_moving_matches'):
                    self._previous_moving_matches = {}

                # Sort the matches based of the distance
                pairing_takeoff = self.find_optimal_matching_resting_moving_from_distance(landing_matrix_dist,landing_matrix_time,previous_matches=self._previous_moving_matches)

                # Updating the tracks with taking-off
                for (i,j,dist_c,time_c) in pairing_takeoff:
                    if dist_c < max_distance_search:
                        mov_id_1 = list_of_ID_moving[i]
                        mov_id_2 = list_of_ID_moving[j]

                        # Mark the end of mov_track_1 and start of mov_track_2 as attached
                        moving_object_tracks.objects[mov_id_1]["end_boot"] = mov_id_2
                        moving_object_tracks.objects[mov_id_1]["end_type"] = "moving"

                        moving_object_tracks.objects[mov_id_2]["start_boot"] = mov_id_1
                        moving_object_tracks.objects[mov_id_2]["start_type"] = "moving"
                        # Cut-off the mov_track_2 that overlaps with mov_track_1
                        if time_c > 0:
                            moving_object_tracks.remove_points_from_track(mov_id_2,[0,time_c])

                # Store matches for temporal consistency in next iteration
                for (i,j,dist_c,time_c) in pairing_takeoff:
                    self._previous_moving_matches[i] = j

            # Display the results
            # Save results 
                if direction == "forward":
                    self.mosquito_mov_tracks_forw = moving_object_tracks
                else:
                    self.mosquito_mov_tracks_back = moving_object_tracks 
                print("Finished assembling moving-moving "+direction)

            #with open(self.video_object_path, 'wb') as f:
            #    pickle.dump(self, f)

########################## Display video with tracking ##############
    def display_video_with_tracking(self,direction,starting_frame,time_btw_frames):

        # Load variables
        if direction == "forward":
            resting_object_tracks = self.mosquito_rest_tracks_forw
            moving_object_tracks = self.mosquito_mov_tracks_forw
            matched_ids = self.matched_ids_forw
        else:
            resting_object_tracks = self.mosquito_rest_tracks_back
            moving_object_tracks = self.mosquito_mov_tracks_back
            matched_ids = self.matched_ids_back
        
        # Defensive check
        if matched_ids is None or (isinstance(matched_ids, tuple) and len(matched_ids) < 2):
            print(f"WARNING: No matched IDs available for display_video_with_tracking")
            return

        # Initialiaze video
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        f_i = starting_frame

        list_of_ID_still = list(resting_object_tracks.objects.keys())
        list_of_ID_moving = list(moving_object_tracks.objects.keys())

        while True:
            suc,frame = cap.read()
            time.sleep(time_btw_frames)

            if suc == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame.copy()

                if f_i > 1:
                    if direction =="forward":
                        frame_idx = f_i
                    else:
                        frame_idx = n_frame-f_i


                    # Show resting tracks
                    n_rest = 0
                    for k,id in enumerate(list_of_ID_still):
                        if frame_idx in resting_object_tracks.objects[id]["time_stamp"]:
                            t_frame = resting_object_tracks.objects[id]["time_stamp"].index(frame_idx)
                            if hasattr(self,"matched_ids_forw"):
                                #text = "i {} s {} e {}".format(id,resting_object_tracks.objects[id]["start_boot"],resting_object_tracks.objects[id]["end_boot"])
                                text = "id {}".format(matched_ids[1][k])
                            else:
                                text = "i {} s {} e {}".format(id,resting_object_tracks.objects[id]["start_boot"],resting_object_tracks.objects[id]["end_boot"])
                            #if matched_ids[1][k] == 6:
                                #print("ID" + str(id))
                                #print(resting_object_tracks.objects[id]["start_boot"])
                                #print(resting_object_tracks.objects[id]["end_boot"])
                            centroid = resting_object_tracks.objects[id]["coordinates"][t_frame]
                            cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, (0, 0, 255), 1)
                            n_rest += 1
                    # Show moving tracks
                    n_mov = 0
                    for k,id in enumerate(list_of_ID_moving):
                        if frame_idx in moving_object_tracks.objects[id]["time_stamp"]:
                            t_frame = moving_object_tracks.objects[id]["time_stamp"].index(frame_idx)
                            if hasattr(self,"matched_ids_forw"):
                                #text = "i {} s {} e {}".format(id,moving_object_tracks.objects[id]["start_boot"],moving_object_tracks.objects[id]["end_boot"])
                                text = "id {}".format(matched_ids[1][k+len(list_of_ID_still)])
                            else:
                                text = "ID {} take {} land {}".format(id,moving_object_tracks.objects[id]["start_boot"],moving_object_tracks.objects[id]["end_boot"])
                            centroid = moving_object_tracks.objects[id]["coordinates"][t_frame]
                            cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, (0, 255, 0), 1)
                            n_mov += 1
                    #os.system('clear')
                    print(str(int(n_rest))+" resting and "+str(int(n_mov))+" moving", end="\r")
                cv2.imshow("frame",img)
                f_i += 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if f_i>n_frame-2:
                    break
        cap.release()
        cv2.destroyAllWindows()


########################## Clean tracks and return the number of umatched tracks ##############
    def clean_tracks(self,step_to_force_analyze,time_window_search,max_distance_search,debug_mode):
        #print("Cleaning tracks", end="\r")

        if step_to_force_analyze<=4:
            for direction in ["forward"]:#["forward","backward"]:
                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back

                list_of_ID_resting = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                min_duration_moving_traj = self.settings["track_moving"]["min_duration_moving_traj"]

                # Check moving objects
                for j,mov_id in enumerate(list_of_ID_moving):
                    #progress_bar(j, len(list_of_ID_moving), bar_length=20)

                    # If not already attached to another tracks
                    if np.isnan(moving_object_tracks.objects[mov_id]["start_boot"]) == 1 and np.isnan(moving_object_tracks.objects[mov_id]["end_boot"]) == 1:
                        if len(moving_object_tracks.objects[mov_id]["time_stamp"]) < time_window_search and self.is_flying(moving_object_tracks.objects[mov_id])==0:
                            moving_object_tracks.remove_track(mov_id)
                        else:
                            if len(moving_object_tracks.objects[mov_id]["time_stamp"]) < min_duration_moving_traj:
                                moving_object_tracks.remove_track(mov_id)


                # Count the percentage of matched tracks
                start_mov = [np.isnan(moving_object_tracks.objects[mov_id]["start_boot"]) for mov_id in moving_object_tracks.objects.keys()]
                perc_matched_mov_start = np.round((1-np.mean(start_mov))*100,3) if len(start_mov) > 0 else 0.0
                end_mov = [np.isnan(moving_object_tracks.objects[mov_id]["end_boot"]) for mov_id in moving_object_tracks.objects.keys()]
                perc_matched_mov_end = np.round((1-np.mean(end_mov))*100,3) if len(end_mov) > 0 else 0.0
                start_rest = [np.isnan(resting_object_tracks.objects[mov_id]["start_boot"]) for mov_id in resting_object_tracks.objects.keys()]
                perc_matched_rest_start = np.round((1-np.mean(start_rest))*100,3) if len(start_rest) > 0 else 0.0
                end_rest = [np.isnan(resting_object_tracks.objects[mov_id]["end_boot"]) for mov_id in resting_object_tracks.objects.keys()]
                perc_matched_rest_end = np.round((1-np.mean(end_rest))*100,3) if len(end_rest) > 0 else 0.0

                print("Finished leaning tracks")
                print("rest_start_"+str(perc_matched_rest_start)+"%")
                print("rest_end_"+str(perc_matched_rest_end)+"%")
                print("mov_start_"+str(perc_matched_mov_start)+"%")
                print("mov_end_"+str(perc_matched_mov_end)+"%")

                # Save results 
                if direction == "forward":
                    self.mosquito_mov_tracks_forw = moving_object_tracks
                else:
                    self.mosquito_mov_tracks_back = moving_object_tracks 
        if debug_mode:
            with open(self.video_object_path, 'wb') as f:
                pickle.dump(self, f)

            
# ########################## Assemble id from matched tracks ###################
    def assemble_tracks_ids(self,step_to_force_analyze,debug_mode):
         
        if step_to_force_analyze<=5:
            for direction in ["forward"]:#["forward","backward"]:
                print("Assembling IDs")
                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back

                list_of_ID_resting = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                matching_matrix = np.zeros((len(list_of_ID_resting)+len(list_of_ID_moving),len(list_of_ID_resting)+len(list_of_ID_moving)))

                # Loop through resting objects
                for i,rest_id in enumerate(list_of_ID_resting):
                    #Check start boot
                    if np.isnan(resting_object_tracks.objects[rest_id]["start_boot"])==0:
                        if resting_object_tracks.objects[rest_id]["start_type"] == "moving":
                            
                            j = list_of_ID_moving.index(resting_object_tracks.objects[rest_id]["start_boot"])+len(list_of_ID_resting)
                            matching_matrix[i][j] = -1
                        elif resting_object_tracks.objects[rest_id]["start_type"] == "resting":
                            
                            j = list_of_ID_resting.index(resting_object_tracks.objects[rest_id]["start_boot"])
                            matching_matrix[i][j] = -1

                    #Check end boot
                    if np.isnan(resting_object_tracks.objects[rest_id]["end_boot"])==0:
                        if resting_object_tracks.objects[rest_id]["end_type"] == "moving":
                            j = list_of_ID_moving.index(resting_object_tracks.objects[rest_id]["end_boot"])+len(list_of_ID_resting)
                            matching_matrix[i][j] = 1
                        elif resting_object_tracks.objects[rest_id]["end_type"] == "resting":
                            
                            j = list_of_ID_resting.index(resting_object_tracks.objects[rest_id]["end_boot"])
                            matching_matrix[i][j] = 1

                # Loop through moving objects
                for i,mov_id in enumerate(list_of_ID_moving):
                    #Check start boot
                    if np.isnan(moving_object_tracks.objects[mov_id]["start_boot"])==0:
                        if moving_object_tracks.objects[mov_id]["start_type"] == "moving":
                            j = list_of_ID_moving.index(moving_object_tracks.objects[mov_id]["start_boot"])+len(list_of_ID_resting)
                            #print(j)
                            matching_matrix[i+len(list_of_ID_resting)][j] = -1
                        elif moving_object_tracks.objects[mov_id]["start_type"] == "resting":
                            j = list_of_ID_resting.index(moving_object_tracks.objects[mov_id]["start_boot"])
                            matching_matrix[i+len(list_of_ID_resting)][j] = -1

                    #Check end boot
                    if np.isnan(moving_object_tracks.objects[mov_id]["end_boot"])==0:
                        if moving_object_tracks.objects[mov_id]["end_type"] == "moving":
                            j = list_of_ID_moving.index(moving_object_tracks.objects[mov_id]["end_boot"])+len(list_of_ID_resting)
                            matching_matrix[i+len(list_of_ID_resting)][j] = 1
                        elif moving_object_tracks.objects[mov_id]["end_type"] == "resting":
                            j = list_of_ID_resting.index(moving_object_tracks.objects[mov_id]["end_boot"])
                            matching_matrix[i+len(list_of_ID_resting)][j] = 1

                matching_matrix_graph = np.where((matching_matrix==-1) | (matching_matrix==1),1,0)
                #print(matching_matrix_graph)
                
                matched_ids = graph.connected_components(matching_matrix_graph)
                
                # Defensive check
                if matched_ids is None or (isinstance(matched_ids, tuple) and len(matched_ids) < 2):
                    print(f"WARNING: No tracks assembled for {self.video_name} in {direction} direction")
                    matched_ids = (0, np.array([]))

                print("Total number of tracks: "+str(matched_ids[0]))
                if direction == "forward":
                    self.matched_ids_forw = matched_ids
                else:
                    self.matched_ids_back = matched_ids
            if debug_mode:
                with open(self.video_object_path, 'wb') as f:
                    pickle.dump(self, f)

# ########################## Compute activity and trajectories from the video. ###################
    def extract_complete_trajectories_from_video(self, debug_mode, skip_plotting=False):
        for direction in ["forward"]:#["forward","backward"]:
            print("Finalizing trajectories")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                    matched_ids = self.matched_ids_forw
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back
                    matched_ids = self.matched_ids_back
                
                # Defensive check
                if matched_ids is None or (isinstance(matched_ids, tuple) and (len(matched_ids) < 2 or matched_ids[0] == 0)):
                    print(f"WARNING: No matched IDs to process for {self.video_name} in {direction} direction, skipping trajectory extraction")
                    continue

                list_of_ID_still = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                # Initialiaze data structure for all trajectories of the video.
                mosquito_tracks = mosquito_obj_tracker("mixed")

                # Get the time_stamp from the video in the right format.
                t_start = self.get_datetime_from_file_name()
                if t_start is None:
                    # Fallback to file mtime to keep pipeline running even with unexpected names.
                    try:
                        t_start = datetime.fromtimestamp(os.path.getmtime(self.video_path))
                    except Exception:
                        t_start = datetime.now()
                print(t_start)
                mosquito_tracks.time_stamp = [t_start + timedelta(milliseconds=int((1000.0 / self.fps) * i)) for i in np.arange(self.total_number_frames)]
                
                
                # Loop over all the Ids. 
                for id in np.arange(int(matched_ids[0])):
                    if debug_mode:
                        progress_bar(id, int(matched_ids[0]), bar_length=20)


                    # Initialize empty NaN track of the lenght of the video.
                    trajectory = [(np.nan,np.nan) for t in np.arange(self.total_number_frames)]
                    state = [np.nan for t in np.arange(self.total_number_frames)]
                    
                    # Loop over all the resting and moving ids of the track
                    list_ids_tracks = np.where(matched_ids[1] == id)
                    #print(list_ids_tracks[0])
                    for sub_id in list_ids_tracks[0]:
                        #If resting track
                        if sub_id < len(list_of_ID_still):
                            for t,frame in enumerate(resting_object_tracks.objects[list_of_ID_still[sub_id]]["time_stamp"]):
                                trajectory[frame] = resting_object_tracks.objects[list_of_ID_still[sub_id]]["coordinates"][t]
                                state[frame] = 0 # Mark as resting object
                        #If moving track
                        else:
                            sub_id = sub_id - len(list_of_ID_still)
                            if self.is_flying(moving_object_tracks.objects[list_of_ID_moving[sub_id]]) == True:
                                for t,frame in enumerate(moving_object_tracks.objects[list_of_ID_moving[sub_id]]["time_stamp"]):
                                    trajectory[frame] = moving_object_tracks.objects[list_of_ID_moving[sub_id]]["coordinates"][t]
                                    state[frame] = 1 # Mark as moving/flying track

                    # Smallest index with nan (beginning of the track)
                    if np.isnan(state).all()==0:
                        # Add a new empty track
                        mosquito_tracks.add_mosquito_track(id)
                        t_start = np.min(np.argwhere(np.isnan(state)==0))
                        t_end = np.max(np.argwhere(np.isnan(state)==0))
                        #print("t_start "+str(t_start)+" t_end"+str(t_end))
                        mosquito_tracks.objects[id]["coordinates"] = trajectory[t_start:t_end]
                        mosquito_tracks.objects[id]["state"] = state[t_start:t_end]
                        mosquito_tracks.objects[id]["start"] = t_start
                        mosquito_tracks.objects[id]["end"] = t_end


                # Save sugar feeding and flight activity
                sugar_all = []
                control_all = []
                fly_all = []
                #print(files_tracking[video_idx])
                for id in mosquito_tracks.objects.keys():

                    # Get main stats
                    sugar = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
                    control = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
                    fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]

                    coordinates = mosquito_tracks.objects[id]["coordinates"]
                    fly[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = mosquito_tracks.objects[id]["state"]
                    
                    is_sugar = [self.point_inside_cage(self.sugar_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
                    is_control = [self.point_inside_cage(self.control_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]

                    sugar[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_sugar
                    control[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_control

                    del sugar[0:5]
                    sugar.pop()

                    del control[0:5]
                    control.pop()

                    del fly[0:5]
                    fly.pop()

                    control_all.append(np.array(control))
                    sugar_all.append(np.array(sugar))
                    fly_all.append(np.array(fly))

                # print(len(np.nansum(control_all,axis=0)))
                # print(len(np.nansum(sugar_all,axis=0)))
                data = {'time': mosquito_tracks.time_stamp[5:-1],
                        'numb_mosquitos_flying': np.nansum(fly_all,axis=0),
                        'numb_mosquitos_sugar': np.nansum(sugar_all,axis=0),
                        'numb_mosquitos_control': np.nansum(control_all,axis=0)}
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                #df.resample('1T', label='right').mean()
                mosquito_tracks.flight_population_activity = df

                #plt.plot_date(mosquito_tracks.flight_population_activity.index,mosquito_tracks.flight_population_activity["numb_mosquitos_flying"],linestyle="-",marker = None)
                #plt.plot_date(mosquito_tracks.flight_population_activity.index,mosquito_tracks.flight_population_activity["numb_mosquitos_control"],linestyle="-",marker = None)
                #plt.plot_date(mosquito_tracks.flight_population_activity.index,mosquito_tracks.flight_population_activity["numb_mosquitos_sugar"],linestyle="-",marker = None)

                #plt.show()

                # Save flight speed and duration (skip plotting if running in background thread)
                if not skip_plotting:
                    create_folder(self.folder_analysis+"/plots_trajectories")
                    # Plot configuration for single video analysis
                    svg_dpi = int(os.environ.get('BUZZSUITE_PLOT_DPI_SINGLE_VIDEO', '200'))
                    svg_figheight = int(os.environ.get('BUZZSUITE_PLOT_FIGHEIGHT_SINGLE_VIDEO', '20'))
                    svg_figwidth = int(os.environ.get('BUZZSUITE_PLOT_FIGWIDTH_SINGLE_VIDEO', '20'))
                    fig, axes = plt.subplots(3, 3, dpi=svg_dpi)
                    fig.set_figheight(svg_figheight)
                    fig.set_figwidth(svg_figwidth)
                    axes  = axes.reshape(-1)
                else:
                    axes = None
                    nb_plotted = 9  # Skip all plotting

                all_speed = []
                all_start_time = []
                all_duration = []
                nb_plotted = 0
                for k,id in enumerate(mosquito_tracks.objects.keys()):
                    state_v = mosquito_tracks.objects[id]["state"]
                    coord_v = mosquito_tracks.objects[id]["coordinates"]

                    state_v = np.array([1-i for i in state_v])
                    runs = zero_runs(state_v)
                    nb_tracks = len(runs[:,0])

                    x = [a for a,b in  coord_v]
                    y = [b for a,b in  coord_v]

                    if nb_tracks>0:
                        for k in np.arange(nb_tracks):
                            t_i = runs[k,0]
                            t_f = runs[k,1]

                            d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
                            d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

                            d_x_2 = [np.square(z) for z in d_x]
                            d_y_2 = [np.square(z) for z in d_y]

                            dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]
                            if t_f - t_i > self.fps * 5:
                                #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
                                all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]] + timedelta(milliseconds=int((1000.0 / self.fps) * t_i)))
                                all_speed.append(np.mean(dist)) # starting time
                                all_duration.append((t_f - t_i) / self.fps) # duration in seconds

                                if not skip_plotting and nb_plotted < 9 and np.mean(dist)>5:
                                    ax = self.plot_flight_trajectory(x[t_i+1:t_f],y[t_i+1:t_f],axes[nb_plotted])
                                    nb_plotted += 1
                
                if not skip_plotting:
                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.ioff()
                    plt.savefig(self.folder_analysis+"/plots_trajectories/__"+self.video_name+'.png',bbox_inches='tight')
                    
                #print(self.folder_analysis+"plots_trajectories/__"+self.video_name+'.png')
                data = {'time': all_start_time,
                        'average_speed': all_speed,
                        'duration' : all_duration}
                df = pd.DataFrame(data)

                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                mosquito_tracks.flight_trajectories = df



                # Save the mosquito_tracks object in the "final_tracking" dir
                with open(self.folder_final+"/"+direction+"_mosq_tracks_"+self.video_name, 'wb') as f:
                    pickle.dump(mosquito_tracks, f)

                print("Finished finalizing trajectories")



########################## Complete trajectories ################
    def extract_complete_trajectories_from_video_V2(self,debug_mode):
        print("Start Extracting compplete trajectories")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Load variables

            resting_object_tracks = self.mosquito_rest_tracks_forw
            moving_object_tracks = self.mosquito_mov_tracks_forw
            matched_ids = self.matched_ids_forw
            
            # Defensive check
            if matched_ids is None or (isinstance(matched_ids, tuple) and (len(matched_ids) < 2 or matched_ids[0] == 0)):
                print(f"WARNING: No matched IDs to process for {self.video_name}, skipping trajectory extraction V2")
                return


            list_of_ID_still = list(resting_object_tracks.objects.keys())
            list_of_ID_moving = list(moving_object_tracks.objects.keys())

            # Initialiaze data structure for all trajectories of the video.
            mosquito_tracks = mosquito_obj_tracker("mixed")

            # Get the time_stamp from the video in the right format.
            t_start = self.get_datetime_from_file_name()
            if t_start is None:
                # Fallback to file mtime to keep pipeline running even with unexpected names.
                try:
                    t_start = datetime.fromtimestamp(os.path.getmtime(self.video_path))
                except Exception:
                    t_start = datetime.now()
            #print(t_start)
            mosquito_tracks.time_stamp = [t_start + timedelta(milliseconds=int((1000.0 / self.fps) * i)) for i in np.arange(self.total_number_frames)]
            
            # Loop over all the Ids. 
            for id in np.arange(int(matched_ids[0])):
                if debug_mode:
                    progress_bar(id, int(matched_ids[0]), bar_length=20)

                # Initialize empty NaN track of the lenght of the video.
                trajectory = [(np.nan,np.nan) for t in np.arange(self.total_number_frames)]
                state = [np.nan for t in np.arange(self.total_number_frames)]
                
                # Loop over all the resting and moving ids of the track
                list_ids_tracks = np.where(matched_ids[1] == id)
                #print(list_ids_tracks[0])
                for sub_id in list_ids_tracks[0]:
                    #If resting track
                    if sub_id < len(list_of_ID_still):
                        for t,frame in enumerate(resting_object_tracks.objects[list_of_ID_still[sub_id]]["time_stamp"]):
                            trajectory[frame] = resting_object_tracks.objects[list_of_ID_still[sub_id]]["coordinates"][t]
                            state[frame] = 0 # Mark as resting object
                    #If moving track
                    else:
                        sub_id = sub_id - len(list_of_ID_still)
                        if self.is_flying(moving_object_tracks.objects[list_of_ID_moving[sub_id]]) == True:
                            for t,frame in enumerate(moving_object_tracks.objects[list_of_ID_moving[sub_id]]["time_stamp"]):
                                trajectory[frame] = moving_object_tracks.objects[list_of_ID_moving[sub_id]]["coordinates"][t]
                                state[frame] = 1 # Mark as moving/flying track

                # Smallest index with nan (beginning of the track)
                if np.isnan(state).all()==0:
                    # Add a new empty track
                    mosquito_tracks.add_mosquito_track(id)
                    t_start = np.min(np.argwhere(np.isnan(state)==0))
                    t_end = np.max(np.argwhere(np.isnan(state)==0))
                    #print("t_start "+str(t_start)+" t_end"+str(t_end))
                    mosquito_tracks.objects[id]["coordinates"] = trajectory[t_start:t_end]
                    mosquito_tracks.objects[id]["state"] = state[t_start:t_end]
                    mosquito_tracks.objects[id]["start"] = t_start
                    mosquito_tracks.objects[id]["end"] = t_end

        self.mosquito_tracks = mosquito_tracks
        print("Completed Extracting compplete trajectories")


############# Extract the poppulation and single mosquito statistics from the mosquito_tracks objects

    def extract_mosquito_population_variables(self):
        print("Start Extracting population variable")
        
        hs_border_points = self.settings["control_border_points"]
        cage_border_points = self.settings["cage_border_points"]
        sugar_border_points = self.settings["sugar_border_points"]
        left_ctrl_border_points = self.settings["square_3_border_points"]
        right_ctrl_border_points = self.settings["square_4_border_points"]
        cage_centroid = calculate_cage_centroid(cage_border_points)

        mosquito_tracks = self.mosquito_tracks

        # Save sugar feeding and flight activity
        sugar_all = []
        hs_all = []
        left_ctrl_all = []
        right_ctrl_all = []

        sugar_fly_all = []
        hs_fly_all = []
        left_ctrl_fly_all = []
        right_ctrl_fly_all = []

        fly_all = []
        center_distances_by_t = [[] for _ in np.arange(len(mosquito_tracks.time_stamp))]

        #print(files_tracking[video_idx])
        for id in mosquito_tracks.objects.keys():

            # Get main stats
            sugar = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            hs = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]

            left_ctrl = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            right_ctrl = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]

            sugar_fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            hs_fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            left_ctrl_fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            right_ctrl_fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]

            coordinates = mosquito_tracks.objects[id]["coordinates"]
            track_start = mosquito_tracks.objects[id]["start"]
            fly[track_start:mosquito_tracks.objects[id]["end"]] = mosquito_tracks.objects[id]["state"]

            if cage_centroid is not None:
                states = mosquito_tracks.objects[id]["state"]
                for idx, coord in enumerate(coordinates):
                    abs_idx = track_start + idx
                    if abs_idx < 0 or abs_idx >= len(mosquito_tracks.time_stamp):
                        continue
                    if idx >= len(states) or states[idx] != 1:
                        continue
                    if coord is None or len(coord) != 2:
                        continue
                    if np.isnan(coord[0]) or np.isnan(coord[1]):
                        continue

                    dx = float(coord[0]) - float(cage_centroid[0])
                    dy = float(coord[1]) - float(cage_centroid[1])
                    center_distances_by_t[abs_idx].append(float(np.sqrt(dx * dx + dy * dy)))
            
            #is_sugar = [point_inside_cage(sugar_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
            #is_control = [point_inside_cage(control_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
            is_sugar = [self.point_inside_cage(sugar_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
            is_hs = [self.point_inside_cage(hs_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0 for i,coord in enumerate(coordinates)]

            is_left_ctrl = [self.point_inside_cage(left_ctrl_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0 for i,coord in enumerate(coordinates)]
            is_right_ctrl = [self.point_inside_cage(right_ctrl_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0 for i,coord in enumerate(coordinates)]

            is_sugar_fly = [self.point_inside_cage(sugar_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==1 for i,coord in enumerate(coordinates)]
            is_hs_fly = [self.point_inside_cage(hs_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==1 for i,coord in enumerate(coordinates)]
            is_left_ctrl_fly = [self.point_inside_cage(left_ctrl_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==1 for i,coord in enumerate(coordinates)]
            is_right_ctrl_fly = [self.point_inside_cage(right_ctrl_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==1 for i,coord in enumerate(coordinates)]

            sugar[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_sugar # Sugar feeder
            hs[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_hs # Host seeking


            left_ctrl[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_left_ctrl 
            right_ctrl[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_right_ctrl 

            sugar_fly[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_sugar_fly
            hs_fly[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_hs_fly
            left_ctrl_fly[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_left_ctrl_fly
            right_ctrl_fly[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_right_ctrl_fly

            del left_ctrl[0:5]
            left_ctrl.pop()

            del right_ctrl[0:5]
            right_ctrl.pop()

            del sugar[0:5]
            sugar.pop()

            del hs[0:5]
            hs.pop()

            del fly[0:5]
            fly.pop()

            del sugar_fly[0:5]
            sugar_fly.pop()

            del hs_fly[0:5]
            hs_fly.pop()

            del left_ctrl_fly[0:5]
            left_ctrl_fly.pop()

            del right_ctrl_fly[0:5]
            right_ctrl_fly.pop()

            hs_all.append(np.array(hs))
            sugar_all.append(np.array(sugar))
            left_ctrl_all.append(np.array(left_ctrl))
            right_ctrl_all.append(np.array(right_ctrl))
            fly_all.append(np.array(fly))

            sugar_fly_all.append(np.array(sugar_fly))
            hs_fly_all.append(np.array(hs_fly))
            left_ctrl_fly_all.append(np.array(left_ctrl_fly))
            right_ctrl_fly_all.append(np.array(right_ctrl_fly))

        mean_center_distance_px = []
        r50_center_distance_px = []
        prop_within_150 = []
        for distances_t in center_distances_by_t[5:-1]:
            if len(distances_t) == 0:
                mean_center_distance_px.append(np.nan)
                r50_center_distance_px.append(np.nan)
                prop_within_150.append(np.nan)
                continue

            distances_arr = np.array(distances_t, dtype=float)
            mean_center_distance_px.append(float(np.mean(distances_arr)))
            r50_center_distance_px.append(float(np.median(distances_arr)))
            prop_within_150.append(float(np.mean(distances_arr <= 150.0)))


        # print(len(np.nansum(control_all,axis=0)))
        # print(len(np.nansum(sugar_all,axis=0)))
        data = {'time': mosquito_tracks.time_stamp[5:-1],
                'numb_mosquitos_flying': np.nansum(fly_all,axis=0),
                'numb_mosquitos_sugar': np.nansum(sugar_all,axis=0),
                'numb_mosquitos_hs': np.nansum(hs_all,axis=0),
                'numb_mosquitos_left_ctrl': np.nansum(left_ctrl_all,axis=0),
            'numb_mosquitos_right_ctrl': np.nansum(right_ctrl_all,axis=0),
            'center_mean_distance_px': mean_center_distance_px,
            'center_r50_px': r50_center_distance_px,
            'center_prop_within_150': prop_within_150
                }
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df.resample('1s', label='right').mean(numeric_only=True)

        mosquito_tracks.population_variables = df

        self.mosquito_tracks = mosquito_tracks
        print("End Extracting population variable")


    def extract_mosquito_individual_variables(self):
        print("Start Extracting individual variable")
        mosquito_tracks = self.mosquito_tracks

        all_start_time = []
        #all_resting_duration = []
        all_flight_duration = []
        all_speed = []
        #print(mosquito_tracks.time_stamp[0])
        for k,id in enumerate(mosquito_tracks.objects.keys()):
            state_v = mosquito_tracks.objects[id]["state"]
            # #coord_v = mosquito_tracks.objects[id]["coordinates"]
            # state_v = np.array([i for i in state_v])
            # runs = zero_runs(state_v)
            # nb_tracks = len(runs[:,0])
            # if nb_tracks>0:
            #     for k in np.arange(nb_tracks):
            #         t_i = runs[k,0]
            #         t_f = runs[k,1]

            #         if t_f-t_i>25*5: # Filter for the not to short resting times
            #             #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
            #             #all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]]+timedelta(milliseconds=int(40*t_i)))
            #             all_start_time.append(mosquito_tracks.time_stamp[0])
            #             all_duration.append((t_f-t_i)/25)
            #             #print((t_f-t_i)/25)
            #             #all_duration.append(1)
            #         state_v = mosquito_tracks.objects[id]["state"]

            # # Extract the flight coordinates        
            coord_v = mosquito_tracks.objects[id]["coordinates"]
            state_v = np.array([1-i for i in state_v])
            runs = zero_runs(state_v)
            nb_tracks = len(runs[:,0])
            x = [a for a,b in  coord_v]
            y = [b for a,b in  coord_v]
            if nb_tracks>0:
                for k in np.arange(nb_tracks):
                    t_i = runs[k,0]
                    t_f = runs[k,1]

                    d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
                    d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

                    d_x_2 = [np.square(z) for z in d_x]
                    d_y_2 = [np.square(z) for z in d_y]
                    
                    dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]
                    if t_f - t_i > self.fps * 2 and np.mean(dist) > 5: # Flight trajectory at least 2sec long and not so slow (speed>5)
                        #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
                        #all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]]+timedelta(milliseconds=int(40*t_i)))
                        all_start_time.append(mosquito_tracks.time_stamp[0])
                        #all_duration.append((t_f-t_i)/25)
                        all_speed.append(np.mean(dist)) # starting time
                        all_flight_duration.append((t_f - t_i) / self.fps)
                        #print((t_f-t_i)/25)
                        #all_duration.append(1)
        #print(len(all_duration))

        data = {'time': all_start_time,
                #'duration' : all_duration}
                'flight_duration' : all_flight_duration,
                'average_speed' : all_speed}
        df = pd.DataFrame(data)

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        mosquito_tracks.individual_variables = df
        self.mosquito_tracks = mosquito_tracks

        print("End Extracting individual variable")

    def extract_mosquito_resting_variables(self):
        print("Start Extracting resting variable")
        mosquito_tracks = self.mosquito_tracks

        all_start_time = []
        all_resting_duration = []

        #print(mosquito_tracks.time_stamp[0])
        for k,id in enumerate(mosquito_tracks.objects.keys()):
            state_v = mosquito_tracks.objects[id]["state"]
            #coord_v = mosquito_tracks.objects[id]["coordinates"]
            state_v = np.array([i for i in state_v])
            runs = zero_runs(state_v)
            nb_tracks = len(runs[:,0])
            if nb_tracks>0:
                for k in np.arange(nb_tracks):
                    t_i = runs[k,0]
                    t_f = runs[k,1]

                    if t_f - t_i > self.fps * 5: # Filter for the not too short resting times
                        #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
                        #all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]]+timedelta(milliseconds=int(40*t_i)))
                        all_start_time.append(mosquito_tracks.time_stamp[0])
                        all_resting_duration.append((t_f - t_i) / self.fps)
                        #print((t_f-t_i)/25)
                        #all_duration.append(1)
                   
        data = {'time': all_start_time,
                #'duration' : all_duration}
                'resting_duration' : all_resting_duration}
        df = pd.DataFrame(data)

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        mosquito_tracks.resting_variables = df
        self.mosquito_tracks = mosquito_tracks

        print("End Extracting resting variable")

    def save_tracking_results(self):
        # Save the mosquito_tracks object in the "final_tracking" dir
        self.mosquito_tracks.settings = self.settings
        self.mosquito_tracks.video_name = self.video_name

        with open(f"{self.folder_final}/forward_mosq_tracks_{self.video_name}", 'wb') as f:
            pickle.dump(self.mosquito_tracks, f)

        print("Saving tracking results (mosquito_tracks object)")

# ########################## Plot single coplete tracks (assembled resting+moving) from a video ###################

    #def plot_sample_flight_trajectories_from_video(axes,mosquito_tracks):

    
    def plot_flight_trajectory(self,x,y,ax):
        # Set plot and add background
        back_path = self.video_path = os.path.join(self.folder_analysis ,"images_mortality", self.video_name)+".png"
        im = plt.imread(back_path)
        ax.imshow(im,zorder=1,cmap = "gray")

        x_f = uniform_filter1d(x, size=5)
        y_f = uniform_filter1d(y, size=5)

        c_f = np.arange(len(x_f))
        c_f = np.array([c_f[i] / self.fps for i in np.arange(len(c_f))])

        points = np.array([x_f, y_f]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(c_f.min(), c_f.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)

        # Set the values used for colormapping
        lc.set_array(c_f)
        lc.set_linewidth(4)
        line = ax.add_collection(lc)

        ax.set_xlim([0 ,im.shape[0]])
        ax.set_ylim([0 ,im.shape[1]])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return ax

# ########################## Filter out moving_tracks that don't look like flying mosquitos ###################
    def is_flying(self,moving_track):
        coord = moving_track["coordinates"]
        time_s = moving_track["time_stamp"]

        max_vel = 40
        min_vel = 1.0
        min_len = 40

        velocity = []
        for t,centroid in enumerate(coord):
            if t>0:
                velocity.append(dist.cdist(np.array(coord[t-1]).reshape(1,-1),np.array(coord[t]).reshape(1,-1)))

        average_speed = np.mean(velocity)
        unicity = len(np.unique(velocity))
        #print("Av_speed_"+str(average_speed)+" unicity_"+str(unicity))
        #print(average_speed)
        # WS3b: the unicity>10 gate (needs >11 distinct speeds) drops short but valid flights.
        # With the motion-aware tracker enabled, relax it; original behaviour otherwise.
        unicity_min = 3 if tracking_flag(self.settings, "assignment") == "hungarian" else 10
        return average_speed < max_vel and average_speed > min_vel and unicity > unicity_min

# ########################## Plot single complete tracks (assembled resting+moving) from a video ###################
    def plot_trajectories_from_video(self,direction):

        plt.rcParams.update({'font.size': 6})
        create_folder(self.folder_analysis+"plots_trajectories")

        # Load the trajectories
        direction = "forward"
        with open(self.folder_final+"/"+direction+"_mosq_tracks_"+self.video_name, 'rb') as f:
            mosquito_tracks = pickle.load(f)  

        
        for id in mosquito_tracks.objects.keys():
            coord =  mosquito_tracks.objects[id]["coordinates"]
            state =  mosquito_tracks.objects[id]["state"]

            if len(coord)>10:
                # Set plot and add background
                # Plot configuration for single video flight plot
                flight_dpi = int(os.environ.get('BUZZSUITE_PLOT_DPI_SINGLE_VIDEO_FLIGHT', '200'))
                flight_figheight = int(os.environ.get('BUZZSUITE_PLOT_FIGHEIGHT_SINGLE_VIDEO_FLIGHT', '16'))
                flight_figwidth = int(os.environ.get('BUZZSUITE_PLOT_FIGWIDTH_SINGLE_VIDEO_FLIGHT', '8'))
                fig, axes = plt.subplots(3, 1, dpi=flight_dpi)
                fig.set_figheight(flight_figheight)
                fig.set_figwidth(flight_figwidth)


            
                # Plot the resting points
                ax = axes[0]
                plt.tight_layout()
                im = plt.imread(self.background_path)
                ax.imshow(im,zorder=1,cmap = "gray")

                resting_coord_idx =[]
                for i,s in enumerate(state):
                    if s==0:
                        resting_coord_idx.append(i)
                x = [coord[k][0] for k in resting_coord_idx]
                y = [coord[k][1] for k in resting_coord_idx]
                ax.scatter(x,y,s=50,color="tab:red")
                ax.set_xlim([0 ,im.shape[0]])
                ax.set_ylim([0 ,im.shape[1]])
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                
                ax = axes[1]
                plt.tight_layout()
                im = plt.imread(self.background_path)
                ax.imshow(im,zorder=1,cmap = "gray")
                x = [coord[k][0] for k in np.arange(len(coord))]
                y = [coord[k][1] for k in np.arange(len(coord))]

                x_f = uniform_filter1d(x, size=5)
                y_f = uniform_filter1d(y, size=5)

                c_f = np.arange(len(x_f))
                c_f = np.array([c_f[i] / self.fps for i in np.arange(len(c_f))])

                points = np.array([x_f, y_f]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create a continuous norm to map from data points to colors
                norm = plt.Normalize(c_f.min(), c_f.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm)

                # Set the values used for colormapping
                lc.set_array(c_f)
                lc.set_linewidth(1)
                line = ax.add_collection(lc)
                cbar = fig.colorbar(line, ax=ax,fraction=0.046, pad=0.04,orientation="horizontal")

                ax.set_xlim([0 ,im.shape[0]])
                ax.set_ylim([0 ,im.shape[1]])
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                ax  =axes[2]
                ax.plot(mosquito_tracks.objects[id]["state"])

                plt.ioff()
                plt.savefig(self.folder_analysis+"plots_trajectories/__"+self.video_name+"_plot_ID_"+str(id)+'.png',bbox_inches='tight')



                        
    def get_datetime_from_file_name(self):
        try:
            print(self.video_name)

            clip_sequence_match = re.search(r'_(\d{5})(?:_|$)', self.video_name)
            clip_sequence_index = int(clip_sequence_match.group(1)) if clip_sequence_match else None

            version_match = re.search(r'_v(\d+)(?:_|$)', self.video_name)
            segment_seconds = self._estimate_segment_seconds()
            if clip_sequence_index is not None:
                version_offset_seconds = clip_sequence_index * segment_seconds
            else:
                version_index = int(version_match.group(1)) if version_match else 1
                version_offset_seconds = max(0, version_index - 1) * segment_seconds

            # First, support names ending with _YYYYMMDD_HHMMSS (e.g. Cage..._20260318_041430)
            ymd_hms = re.search(r'(\d{8})_(\d{6})(?:_|$)', self.video_name)
            if ymd_hms:
                ymd = ymd_hms.group(1)
                hms = ymd_hms.group(2)
                t = datetime(
                    int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]),
                    int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
                )
                return t + timedelta(seconds=version_offset_seconds)

            # Support names containing _YYMMDD_HHMMSS (e.g. Cage..._251030_040119_v02)
            yymmdd_hms = re.search(r'(\d{6})_(\d{6})(?:_|$)', self.video_name)
            if yymmdd_hms:
                ymd = yymmdd_hms.group(1)
                hms = yymmdd_hms.group(2)
                t = datetime(
                    2000 + int(ymd[0:2]), int(ymd[2:4]), int(ymd[4:6]),
                    int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
                )
                return t + timedelta(seconds=version_offset_seconds)

            # Legacy filename formats
            lower_name = self.video_name.lower()
            if "_raspberrypi_" in self.video_name:
                s = self.video_name.find('_raspberrypi_')
                l = 13
            elif "_mosquipi4_" in self.video_name:
                s = self.video_name.find('_mosquipi4_')
                l = 11
            elif "_mosquipi3_" in self.video_name:
                s = self.video_name.find('_mosquipi3_')
                l = 11
            elif "_mosquipi1_" in self.video_name:
                s = self.video_name.find('_mosquipi1_')
                l = 11
            elif "_moscam" in self.video_name:
                s = self.video_name.find('_moscam')
                l = 11
            elif '_dawn_' in lower_name:
                s = lower_name.find('_dawn_')
                l = 6
            elif '_dusk_' in lower_name:
                s = lower_name.find('_dusk_')
                l = 6
            else:
                raise ValueError("Unknown filename pattern")

            YY = int(self.video_name[s-6:s-4])
            MM = int(self.video_name[s-4:s-2])
            DD = int(self.video_name[s-2:s])
            HH = int(self.video_name[s+l:s+l+2])
            MI = int(self.video_name[s+l+2:s+l+4])
            SS = int(self.video_name[s+l+4:s+l+6])
            VV = int(self.video_name[s+l+8:s+l+10]) * 1204

            t = datetime(2000 + YY, MM, DD, HH, MI, SS)
            return t + timedelta(seconds=VV)
        except Exception:
            print("Incorrect name file, cannot find date")
            return None

    def _estimate_segment_seconds(self):
        """Estimate video segment duration in seconds for _vNN offsets.

        Prefers frame-count/fps when available; falls back to env/default.
        """
        try:
            if hasattr(self, 'total_number_frames') and hasattr(self, 'fps'):
                if self.total_number_frames and self.fps and self.fps > 0:
                    estimate = int(round(float(self.total_number_frames) / float(self.fps)))
                    if estimate > 0:
                        return estimate
        except Exception:
            pass

        return max(1, int(DEFAULT_VIDEO_SEGMENT_SECONDS))


    def extract_flight_metrics_around_resting(self):
        # Assuming the mosquito_tracks object has already been computed and exists.
        assert hasattr(self, 'mosquito_tracks'), "mosquito_tracks object not found. Run the extraction pipeline first."
        
        borders = {
            'sugar': self.settings["sugar_border_points"],
            'hs': self.settings["control_border_points"],
            'left_ctrl': self.settings["square_4_border_points"],
            'right_ctrl': self.settings["square_3_border_points"]
        }
        #print(borders)

        metrics = []
        fps = self.fps if self.fps > 0 else 25.0

        for id in self.mosquito_tracks.objects.keys():
            resting_times = []
            coords = self.mosquito_tracks.objects[id]["coordinates"]
            state = self.mosquito_tracks.objects[id]["state"]

            # Analyze the coordinates and states to check for resting points in each zone
            for zone, border in borders.items():
                is_in_zone = [self.point_inside_cage(border, coord) and state[i] == 0 for i, coord in enumerate(coords)]
                resting_indices = np.where(is_in_zone)[0]
                
                if len(resting_indices) == 0:
                    continue

                # Extracting landing metrics (if any)
                landing_duration, landing_speed = np.nan, np.nan
                if resting_indices[0] > 0:
                    # Find the end of the last flight segment before resting
                    landing_end_idx = resting_indices[0] - 1
                    
                    landing_start_idx = landing_end_idx
                    if landing_start_idx >= 0:
                        while landing_start_idx > 0 and state[landing_start_idx] == 1:
                            landing_start_idx -= 1
                        if state[landing_start_idx] == 0:
                            landing_start_idx += 1
                        # Debug prints
                        #print(f"(ID: {id}) Landing: Start={landing_start_idx}, End={landing_end_idx}")     
                        landing_coords = coords[landing_start_idx:landing_end_idx + 1]
                        if len(landing_coords) > 1:  # To prevent issues with 1D arrays
                            landing_dists = np.linalg.norm(np.diff(landing_coords, axis=0), axis=1)
                            landing_duration = len(landing_coords) / fps
                            landing_speed = np.mean(landing_dists) if landing_duration > 0 else np.nan
                        else:
                            landing_duration, landing_speed = np.nan, np.nan
                        #print("Landing Metrics - Duration:", landing_duration, " Speed:", landing_speed)

                # Extracting takeoff metrics (if any)
                takeoff_duration, takeoff_speed = np.nan, np.nan
                if resting_indices[-1] < len(state) - 1:
                    # Find the start of the first flight segment after resting
                    takeoff_start_idx = resting_indices[-1] + 1

                    takeoff_end_idx = takeoff_start_idx

                    if takeoff_end_idx < len(state):
                        while takeoff_end_idx < len(state) and state[takeoff_end_idx] == 1:
                            takeoff_end_idx += 1
                        takeoff_coords = coords[takeoff_start_idx:takeoff_end_idx]
                        if len(takeoff_coords) > 1:  # To prevent issues with 1D arrays
                            takeoff_dists = np.linalg.norm(np.diff(takeoff_coords, axis=0), axis=1)
                            takeoff_duration = len(takeoff_coords) / fps
                            takeoff_speed = np.mean(takeoff_dists) if takeoff_duration > 0 else np.nan
                        else:
                            takeoff_duration, takeoff_speed = np.nan, np.nan
                        #print("Takeoff Metrics - Duration:", takeoff_duration, " Speed:", takeoff_speed)

                # Resting duration
                resting_duration = len(resting_indices) / fps

                metrics.append([zone, id, [landing_duration, landing_speed], resting_duration, [takeoff_duration, takeoff_speed]])
                #print(f"ID: {id}, Zone: {zone}, Metrics: {metrics[-1]}")

        metrics_df = pd.DataFrame(metrics, columns=['zone', 'mosquito_id', 'landing', 'resting_time', 'takeoff'])
        self.mosquito_tracks.flight_metrics_around_resting = metrics_df
        print("Flight metrics around resting points have been extracted and saved.")

# ########################## Assemble backward and forward tracks ###################
#    def assemble_forward_backward_tracks(self,step_to_force_analyze):








#     def assemble_forward_backward_tracks(self):

#         #Parameters
#         max_dist_rest = 3
        
#         # Load necessary data forward
#         direction = "forward"
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             matching_matrix = pickle.load(f)

#         # Load necessary data backward
#         direction = "backward"
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             back_resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             back_moving_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             back_matching_matrix = pickle.load(f)

#         # Get the list of IDs resting
#         list_of_ID_resting_forward = get_ids_from_dict(resting_object_traj)
#         list_of_ID_resting_backward = get_ids_from_dict(back_resting_object_traj)

#         n_frames = self.total_number_frames

#         numb_id_resting_forward = len(list_of_ID_resting_forward)
#         numb_id_resting_backward = len(list_of_ID_resting_backward)

#         #print(numb_id_resting_forward)
#         #print(numb_id_resting_backward)

#         # Square zeros matrix to store link between IDs (resting and moving) and (moving moving)
#         matching_matrix_rest = np.zeros((numb_id_resting_forward+numb_id_resting_backward,numb_id_resting_forward+numb_id_resting_backward))

#         for i,rest_id_forw in enumerate(list_of_ID_resting_forward): # For all ID of forward tracking
#             progress_bar(i, numb_id_resting_forward , bar_length=20)
#             time_forw = resting_object_traj["time_ID_"+str(rest_id_forw)]
#             pos_forw = resting_object_traj["ID_"+str(rest_id_forw)]

#             for j,rest_id_back in enumerate(list_of_ID_resting_backward):
#                 time_back = back_resting_object_traj["time_ID_"+str(rest_id_back)]
#                 pos_back = back_resting_object_traj["ID_"+str(rest_id_back)]

#                 time_back.reverse()
#                 pos_back.reverse()

#                 time_back = [n_frames-k for k in time_back]

#                 start_f = time_forw[0]
#                 end_f = time_forw[-1]

#                 start_b = time_back[0]
#                 end_b = time_back[-1]

#                 start = np.min([start_f,start_b])
#                 end = np.max([end_f,end_b])

#                 if end-start < (end_f-start_f)+(end_b-start_b):
#                     start_search = np.max([start_f,start_b])
#                     end_search = np.min([end_f,end_b])

#                     start_time_back = time_back.index(start_search)
#                     start_time_forw = time_forw.index(start_search)

#                     for s,t in enumerate(np.arange(start_search,end_search,1)):
#                         D_traj = dist.cdist(np.array(pos_forw[start_time_forw+s]).reshape(1,-1),np.array(pos_back[start_time_back+s]).reshape(1,-1))
#                         if D_traj[0] < max_dist_rest:
#                             matching_matrix_rest[i][j+numb_id_resting_forward] +=1

#         with open(self.folder_intermediate+"/matching_matrix_rest_"+self.video_name+".pkl", 'wb') as f:
#             pickle.dump(matching_matrix_rest,f)

#         matching_matrix_rest = np.where(matching_matrix_rest>40,1,0)
#         matched_ids = graph.connected_components(matching_matrix_rest)
#         print(matched_ids)



########################## Assemble resting and moving obj tracks ##############
#     def assemble_resting_moving_tracks(self,direction):
#         print("Start assembling resting and moving tracks "+direction)
#         # Parameters
#         max_distance_takeoff = 60
#         time_window_takeoff = 30
#         max_distance_landing = 60
#         time_window_landing = 30
#         max_distance_moving_cross = 20

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)

#         numb_id_resting = len(list_of_ID_still)
#         numb_id_moving = len(list_of_ID_moving)

#         # Square zeros matrix to store link between IDs (resting and moving) and (moving moving)
#         matching_matrix = np.zeros((numb_id_resting+numb_id_moving,numb_id_resting+numb_id_moving))

#         # Find all the intersection between resting and moving tracks
#         for i,rest_id in enumerate(list_of_ID_still):
#             progress_bar(i, numb_id_resting, bar_length=20)
#             pos_v = resting_object_traj["ID_"+str(rest_id)]
#             time_v = resting_object_traj["time_ID_"+str(rest_id)]

#             if time_v[-1] == self.total_number_frames-1:
#                 print("resting id "+str(rest_id)+" does not move until end of vid")
#             else:
#                 t_landing_tar = time_v[0]
#                 t_takeoff_tar = time_v[-self.maxobjdisappear_resting]
#                 for j,mov_id in enumerate(list_of_ID_moving):

#                     mov_pos = moving_object_traj["ID_"+str(mov_id)]
#                     mov_time = moving_object_traj["time_ID_"+str(mov_id)]

#                     # Look for landing
#                     for t_landing in np.arange(t_landing_tar-int(time_window_landing/2), t_landing_tar+int(time_window_landing/2), 1):
#                         if t_landing in mov_time:
#                             t_landing_mov = mov_time.index(t_landing)
#                             D_landing = dist.cdist(np.array(pos_v[0]).reshape(1,-1), np.array(mov_pos[t_landing_mov]).reshape(1,-1))
#                             if D_landing[0].min(axis=0) < max_distance_landing:
#                                 #print("resting id "+str(rest_id)+" lands with :"+str(mov_id))
#                                 matching_matrix[i][j+numb_id_resting] += 1 # Save the link for landing

#                     # Look for take-off match
#                     for t_takeoff in np.arange(t_takeoff_tar-int(time_window_takeoff/2), t_takeoff_tar+int(time_window_takeoff/2), 1):
#                         if t_takeoff in mov_time:
#                             t_takeoff_mov = mov_time.index(t_takeoff)
#                             D_takeoff = dist.cdist(np.array(pos_v[-self.maxobjdisappear_resting]).reshape(1,-1), np.array(mov_pos[t_takeoff_mov]).reshape(1,-1))
#                             if D_takeoff[0].min(axis=0) < max_distance_takeoff:
#                                 #print("resting id "+str(rest_id)+" takes-off with :"+str(mov_id))
#                                 matching_matrix[i][j+numb_id_resting] += 1 

#         # Find all the intersection between moving tracks
#         # for j_1,mov_id_1 in enumerate(list_of_ID_moving):
#         #     progress_bar(j_1, numb_id_moving, bar_length=20)
#         #     mov_pos_1 = moving_object_traj["ID_"+str(mov_id_1)]
#         #     mov_time_1 = moving_object_traj["time_ID_"+str(mov_id_1)]
#         #     for k,t in enumerate(mov_time_1):
#         #         for j_2,mov_id_2 in enumerate(list_of_ID_moving):
#         #             mov_pos_2 = moving_object_traj["ID_"+str(mov_id_2)]
#         #             mov_time_2 = moving_object_traj["time_ID_"+str(mov_id_2)]
#         #             if np.abs(j_2-j_1)>0:
#         #                 if t in mov_time_2:
#         #                     t_mov_2 = mov_time_2.index(t)
#         #                     D_mov = dist.cdist(np.array(mov_pos_1[k]).reshape(1,-1), np.array(mov_pos_2[t_mov_2]).reshape(1,-1))
#         #                     if D_mov[0].min(axis=0) < max_distance_moving_cross:
#         #                         matching_matrix[j_1+numb_id_resting][j_2+numb_id_resting] += 1
#         #                         print("moving id "+str(mov_id_1)+" cross with moving id :"+str(mov_id_2))

#         #matching_matrix = np.where(matching_matrix>0,1,0)
#         #matched_ids = graph.connected_components(matching_matrix)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'wb') as f:
#             pickle.dump(matching_matrix, f)
#         #print(matched_ids)
# ########################## Assemble resting and moving obj tracks ##############
#     def assemble_resting_moving_tracks_V2(self,direction):
#         print("Start assembling resting and moving tracks "+direction)
#         # Parameters
#         max_distance_takeoff = 60
#         time_window_takeoff = 30
#         max_distance_landing = 60
#         time_window_landing = 30
#         max_distance_moving_cross = 20

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)

#         numb_id_resting = len(list_of_ID_still)
#         numb_id_moving = len(list_of_ID_moving)

#         rest_ids_boots = np.empty((len(list_of_ID_still),2,))
#         rest_ids_boots[:] = np.nan
#         mov_ids_boots = np.empty((len(list_of_ID_moving),2,))
#         mov_ids_boots[:] = np.nan

#         # Find all the intersection between resting and moving tracks
#         for i,rest_id in enumerate(list_of_ID_still):
#             progress_bar(i, numb_id_resting, bar_length=20)
#             pos_v = resting_object_traj["ID_"+str(rest_id)]
#             time_v = resting_object_traj["time_ID_"+str(rest_id)]

#             if time_v[-1] == self.total_number_frames-1:
#                 print("resting id "+str(rest_id)+" does not move until end of vid")
#             else: # If track disappears before the end of the video
#                 t_landing_tar = time_v[0]
#                 t_takeoff_tar = time_v[-self.maxobjdisappear_resting]

#                 # Initialize 
#                 min_dist_takeoff = max_distance_takeoff
#                 min_dist_landing = max_distance_landing

#                 mov_id_takeoff_tar = np.nan
#                 mov_id_landing_tar = np.nan

#                 j_mov_takeoff = np.nan
#                 j_mov_landing = np.nan

#                 # Loop over all the moving IDs
#                 for j,mov_id in enumerate(list_of_ID_moving):

#                     mov_pos = moving_object_traj["ID_"+str(mov_id)]
#                     mov_time = moving_object_traj["time_ID_"+str(mov_id)]

#                     # Look for landing
#                     if np.isnan(rest_ids_boots[i][0]) == 1 and np.isnan(mov_ids_boots[j][1]) == 1: # if ids not already matched 
#                         for t_landing in np.arange(t_landing_tar-int(time_window_landing/2), t_landing_tar+int(time_window_landing/2), 1):
#                             if t_landing in mov_time:
#                                 t_landing_mov = mov_time.index(t_landing)
#                                 D_landing = dist.cdist(np.array(pos_v[0]).reshape(1,-1), np.array(mov_pos[t_landing_mov]).reshape(1,-1))
#                                 if D_landing[0].min(axis=0) < max_distance_landing:
#                                     #print("resting id "+str(rest_id)+" lands with :"+str(mov_id))
#                                     #matching_matrix[i][j+numb_id_resting] += 1 # Save the link for landing
#                                     if D_landing[0].min(axis=0)<min_dist_landing: #and np.abs(t_landing-t_landing_tar)<min_time_diff_landing: # If better landing match
#                                         min_dist_landing = D_landing[0].min(axis=0)
#                                         #min_time_diff_landing = np.abs(t_landing-t_landing_tar)
#                                         mov_id_landing_tar = mov_id
#                                         j_mov_landing = j

#                     # Look for take-off match
#                     if np.isnan(rest_ids_boots[i][1]) == 1 and np.isnan(mov_ids_boots[j][0]) == 1: # if ids not already matched 
#                         for t_takeoff in np.arange(t_takeoff_tar-int(time_window_takeoff/2), t_takeoff_tar+int(time_window_takeoff/2), 1):
#                             if t_takeoff in mov_time:
#                                 t_takeoff_mov = mov_time.index(t_takeoff)
#                                 D_takeoff = dist.cdist(np.array(pos_v[-self.maxobjdisappear_resting]).reshape(1,-1), np.array(mov_pos[t_takeoff_mov]).reshape(1,-1))
#                                 if D_takeoff[0].min(axis=0) < max_distance_takeoff:
#                                         if D_takeoff[0].min(axis=0)<min_dist_takeoff: #and np.abs(t_takeoff-t_takeoff_tar)<min_time_diff_takeoff: # If better landing match
#                                             min_dist_takeoff = D_takeoff[0].min(axis=0)
#                                             #min_time_diff_takeoff = np.abs(t_takeoff-t_takeoff_tar)
#                                             mov_id_takeoff_tar = mov_id
#                                             j_mov_takeoff = j
#                                 #print("resting id "+str(rest_id)+" takes-off with :"+str(mov_id))
#                                 #matching_matrix[i][j+numb_id_resting] += 1 


#                 if np.isnan(mov_id_takeoff_tar)==0: # If a taking-off match was found
#                     #print("resting id "+str(rest_id)+" takesoff with :"+str(mov_id_takeoff_tar))
#                     #print(min_dist_takeoff)
#                     rest_ids_boots[i][1] = mov_id_takeoff_tar
#                     mov_ids_boots[j_mov_takeoff][1] = rest_id

#                 if np.isnan(mov_id_landing_tar)==0: # If a landing match was found
#                     #print("resting id "+str(rest_id)+" lands with :"+str(mov_id_landing_tar))
#                     #print(min_dist_landing)
#                     rest_ids_boots[i][0] = mov_id_landing_tar
#                     mov_ids_boots[j_mov_landing][1] = rest_id
#         print(rest_ids_boots)
#         print(mov_ids_boots)
#         # with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'wb') as f:
#             #     pickle.dump(matching_matrix, f)
#         #print(matched_ids)
#         # 

# ########################## Assemble resting and moving obj tracks ##############
#     def assemble_resting_moving_tracks_V3(self,direction):
#         print("Start assembling resting and moving tracks "+direction)

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)

#         # Parameters
#         limit_max_distance_search = 100
#         time_window_search = 30

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)        # 

#         rest_ids_boots = np.empty((len(list_of_ID_still),3,))
#         rest_ids_boots[:] = np.nan
#         for k in np.arange(len(list_of_ID_still)):
#             rest_ids_boots[k][0] = list_of_ID_still[k]
        
#         mov_ids_boots = np.empty((len(list_of_ID_moving),3,))
#         mov_ids_boots[:] = np.nan
#         for k in np.arange(len(list_of_ID_moving)):
#             mov_ids_boots[k][0] = list_of_ID_moving[k]
        
#         # Initialize
#         updated_moving_traj = moving_object_traj
#         updated_resting_traj = resting_object_traj
#         current_resting_traj = resting_object_traj
#         current_moving_traj = moving_object_traj

#         max_distance_search = 2 # Initial search distance
#         while max_distance_search < limit_max_distance_search:

#             # Loop over all the resting IDs
#             for i,rest_id in enumerate(list_of_ID_still):
#                 print(i)
#                 updated_moving_traj,updated_resting_traj,rest_ids_boots,mov_ids_boots = self.find_match_with_moving(i,rest_id,current_resting_traj,current_moving_traj,time_window_search,max_distance_search,rest_ids_boots,mov_ids_boots,updated_resting_traj,updated_moving_traj)
            
#             # Udpate the traj for the next round
#             current_moving_traj = updated_moving_traj
#             current_resting_traj = updated_resting_traj

#             max_distance_search += 2
#             print(rest_ids_boots)

# #####################################

    
# #####################################
#     def find_match_with_moving(self,i,rest_id,current_resting_traj,current_moving_traj,time_window_search,max_distance_search,rest_ids_boots,mov_ids_boots,updated_resting_traj,updated_moving_traj):
#         list_of_ID_moving = get_ids_from_dict(current_moving_traj) 
        
#         pos_v = current_resting_traj["ID_"+str(rest_id)]
#         time_v = current_resting_traj["time_ID_"+str(rest_id)]

#         if time_v[-1] == self.total_number_frames-1:
#             print("resting id "+str(rest_id)+" does not move until end of vid")
#         else: # If track disappears before the end of the video
#             t_landing_tar = time_v[0]
#             t_takeoff_tar = time_v[-self.maxobjdisappear_resting]

#             # Initialize 
#             min_time_diff_takeoff = int(time_window_search/2)
#             min_time_diff_landing = int(time_window_search/2)

#             mov_id_takeoff_tar = np.nan
#             mov_id_landing_tar = np.nan

#             j_mov_takeoff = np.nan
#             j_mov_landing = np.nan

#             # Loop over all the moving IDs
#             for j,mov_id in enumerate(list_of_ID_moving):

#                 mov_pos = current_moving_traj["ID_"+str(mov_id)]
#                 mov_time = current_moving_traj["time_ID_"+str(mov_id)]

#                 # Look for landing
#                 print(np.size(mov_ids_boots))
#                 if np.isnan(rest_ids_boots[i][1]) == 1 and np.isnan(mov_ids_boots[j][2]) == 1: # if ids not already matched 
#                     for t_landing in np.arange(t_landing_tar-int(time_window_search/2), t_landing_tar+int(time_window_search/2), 1):
#                         if t_landing in mov_time:
#                             t_landing_mov = mov_time.index(t_landing)
#                             D_landing = dist.cdist(np.array(pos_v[0]).reshape(1,-1), np.array(mov_pos[t_landing_mov]).reshape(1,-1))
#                             if D_landing[0].min(axis=0) < max_distance_search:
#                                 #print("resting id "+str(rest_id)+" lands with :"+str(mov_id))
#                                 #matching_matrix[i][j+numb_id_resting] += 1 # Save the link for landing
#                                 if np.abs(t_landing-t_landing_tar)<min_time_diff_landing: # If better landing match
#                                     #min_dist_landing = D_landing[0].min(axis=0)
#                                     min_time_diff_landing = np.abs(t_landing-t_landing_tar)
#                                     mov_id_landing_tar = mov_id
#                                     j_mov_landing = j
#                                     t_landing_opt = t_landing_mov

#                 # Look for take-off match
#                 if np.isnan(rest_ids_boots[i][2]) == 1 and np.isnan(mov_ids_boots[j][1]) == 1: # if ids not already matched 
#                     for t_takeoff in np.arange(t_takeoff_tar-int(time_window_search/2), t_takeoff_tar+int(time_window_search/2), 1):
#                         if t_takeoff in mov_time:
#                             t_takeoff_mov = mov_time.index(t_takeoff)
#                             D_takeoff = dist.cdist(np.array(pos_v[-self.maxobjdisappear_resting]).reshape(1,-1), np.array(mov_pos[t_takeoff_mov]).reshape(1,-1))
#                             if D_takeoff[0].min(axis=0) < max_distance_search:
#                                     if np.abs(t_takeoff-t_takeoff_tar)<min_time_diff_takeoff: # If better landing match
#                                         #min_dist_takeoff = D_takeoff[0].min(axis=0)
#                                         min_time_diff_takeoff = np.abs(t_takeoff-t_takeoff_tar)
#                                         mov_id_takeoff_tar = mov_id
#                                         j_mov_takeoff = j
#                                         t_takeoff_opt = t_takeoff_mov

#                 # Update the trajectories
                
#                 if np.isnan(mov_id_landing_tar)==0: # If landing match found
                    
#                     rest_ids_boots[i][1] = mov_id_landing_tar # Mark the ID as landed


#                     mov_pos = current_moving_traj["ID_"+str(mov_id_landing_tar)]
#                     mov_time = current_moving_traj["time_ID_"+str(mov_id_landing_tar)]

#                     # Remove the part of the flying traj after landing
#                     updated_moving_traj["ID_"+str(mov_id_landing_tar)] = mov_pos[0:t_landing_opt]
#                     updated_moving_traj["time_ID_"+str(mov_id_landing_tar)] = mov_time[0:t_landing_opt]
                    
#                     # Transfer this deleted traj as a new flying trajectory
#                     list_of_ID_moving = get_ids_from_dict(updated_moving_traj)  
#                     new_mov_id = np.max(list_of_ID_moving)+1 # define an ID that does not exist

#                     updated_moving_traj["ID_"+str(new_mov_id)] = mov_pos[t_landing_opt:-1]
#                     updated_moving_traj["time_ID_"+str(new_mov_id)] = mov_time[t_landing_opt:-1]

#                     index_vec = [mov_ids_boots[i][0] for i in np.arange(len(mov_ids_boots))]
#                     index_of_moving_id =  index_vec.index(index_vec == mov_id_landing_tar)
#                     np.append(mov_ids_boots,[[new_mov_id,np.nan,mov_ids_boots[index_of_moving_id][2]]],axis=0)

#                 if np.isnan(mov_id_takeoff_tar)==0: # If taking-off match found

#                     rest_ids_boots[i][2] = mov_id_takeoff_tar # Mark the ID as taken-off
#                     updated_resting_traj["ID_"+str(rest_id)] = updated_resting_traj["ID_"+str(rest_id)][0:-self.maxobjdisappear_resting]
#                     updated_resting_traj["time_ID_"+str(rest_id)] = updated_resting_traj["time_ID_"+str(rest_id)][0:-self.maxobjdisappear_resting]

#                     # Remove the part of the flying traj before taking-off
#                     mov_pos = current_moving_traj["ID_"+str(mov_id_takeoff_tar)]
#                     mov_time = current_moving_traj["time_ID_"+str(mov_id_takeoff_tar)]

#                     # Remove the part of the flying traj before taking off
#                     #print(updated_moving_traj.keys())
#                     updated_moving_traj["ID_"+str(mov_id_takeoff_tar)] = mov_pos[t_takeoff_opt:-1]
#                     updated_moving_traj["time_ID_"+str(mov_id_takeoff_tar)] = mov_time[t_takeoff_opt:-1]
                    
#                     # Transfer this deleted traj as a new flying trajectory
#                     list_of_ID_moving = get_ids_from_dict(updated_moving_traj)  
#                     new_mov_id = np.max(list_of_ID_moving)+1 # define an ID that does not exist

#                     updated_moving_traj["ID_"+str(new_mov_id)] = mov_pos[0:t_takeoff_opt]
#                     updated_moving_traj["time_ID_"+str(new_mov_id)] = mov_time[0:t_takeoff_opt]

#                     index_vec = [mov_ids_boots[i][0] for i in np.arange(len(mov_ids_boots))]
#                     index_of_moving_id =  index_vec.index(mov_id_takeoff_tar)
#                     #print(index_of_moving_id)
#                     #print(mov_ids_boots[-1])
#                     np.append(mov_ids_boots,[[new_mov_id,mov_ids_boots[index_of_moving_id][1],np.nan]],axis=0)


#         return updated_moving_traj,updated_resting_traj,rest_ids_boots,mov_ids_boots
# ########################## Display video with assembled tracks ###################
#     def display_assembled_resting_moving_tracks(self,direction,starting_frame,time_btw_frames):

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             matching_matrix = pickle.load(f)


#         matching_matrix = np.where(matching_matrix>0,1,0)
#         matched_ids = graph.connected_components(matching_matrix)

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)

#         numb_id_resting = len(list_of_ID_still)
#         numb_id_moving = len(list_of_ID_moving)


#         # Initialiaze video
#         cap = cv2.VideoCapture(self.video_path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
#         n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
#         f_i = starting_frame

#         moving_objects_ID = list(moving_object_traj.keys())
#         resting_objects_ID = list(resting_object_traj.keys())

#         #print(list_of_ID_still)
#         #print(resting_objects_ID)

#         while True:
#             suc,frame = cap.read()
#             time.sleep(time_btw_frames)

#             if suc == True:
#                 frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 img = frame.copy()

#                 if f_i > 1:
#                     if direction =="forward":
#                         frame_idx = f_i
#                     else:
#                         frame_idx = n_frame-f_i


#                     # Show resting tracks
#                     for k,id in enumerate(resting_objects_ID):
#                         parse_ID = id.partition("ID_")[2]
#                         if "time_ID_"+str(parse_ID) in resting_objects_ID:
#                             if frame_idx in resting_object_traj["time_ID_"+str(parse_ID)]:
#                                 t_frame = resting_object_traj["time_ID_"+str(parse_ID)].index(frame_idx)
#                                 if t_frame < len(resting_object_traj["ID_"+str(parse_ID)]):
#                                     index_still = list_of_ID_still.index(int(parse_ID))
#                                     text = "ID {}".format(matched_ids[1][index_still])
#                                     centroid = resting_object_traj["ID_"+str(parse_ID)][t_frame]
#                                     cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#                                     cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, (0, 0, 255), 1)

#                     # Show moving tracks
#                     for k,id in enumerate(moving_objects_ID):
#                         parse_ID = id.partition("ID_")[2]
#                         if "time_ID_"+str(parse_ID) in moving_objects_ID:
#                             if frame_idx+1 in moving_object_traj["time_ID_"+str(parse_ID)]:
#                                 t_frame = moving_object_traj["time_ID_"+str(parse_ID)].index(frame_idx+1)
#                                 if t_frame < len(moving_object_traj["ID_"+str(parse_ID)]):
#                                     index_mov = list_of_ID_moving.index(int(parse_ID))
#                                     text = "ID {}".format(matched_ids[1][index_mov+numb_id_resting])
#                                     centroid = moving_object_traj["ID_"+str(parse_ID)][t_frame]
#                                     cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                                     cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 4, (0, 255, 0), 1)

#                 cv2.imshow("frame",img)
#                 f_i += 1

#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break

#                 if f_i>n_frame-2:
#                     break
#         cap.release()
#         cv2.destroyAllWindows()





# ########################## Check moving object traj looks like a mosquito flight (maybe remove?) ####################
#     def looks_like_flight(self,moving_object_single_traj):

#         max_vel = 40
#         min_vel = 2
#         min_len = 40

#         velocity = []
#         for t,centroid in enumerate(moving_object_single_traj):
#             if t>0:
#                 velocity.append(dist.cdist(np.array(moving_object_single_traj[t-1]).reshape(1,-1),np.array(moving_object_single_traj[t]).reshape(1,-1)))

#         average_speed = np.mean(velocity)
#         unicity = len(np.unique(velocity))
#         #print(average_speed)
#         return average_speed < max_vel and average_speed > min_vel
