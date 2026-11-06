"""Shared helpers for yolo_export: data roots and the output guard, the behaviour/label
definitions (written to docs/definitions.json and read again by the Pi app), cage geometry and
zone maps, tracking-file loading, segment names, and table I/O.

Everything here is read-only on the data drives. Output paths go through check_output_dir(), which
refuses any folder inside a Buzzwatch data root (its Analysis/ and Recording/ trees included).
"""
import copy
import glob
import json
import os
import re
import string
import sys
import zlib

import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from yolo_export import GENERATOR_VERSION  # noqa: E402

OUT_DIRNAME = "BuzzWatch_LiveDetector"
TRACK_PREFIX = "forward_mosq_tracks_"
NOMINAL_FPS = 25.0
ZONE_DS = 2            # zone distance maps are built at 1/ZONE_DS resolution (as fp_extract.DS)

# Experiment sub-folders never worth walking into while scanning for experiment_*.json.
HEAVY_SUBDIRS = ("final_tracking_data", "temp_data", "individual_images", "images_mortality",
                 "tracking_resting", "tracking_moving", "log_analysis", "test_tracking", "plots",
                 "exports", "day_cache", "glmm_results", "speaker_side_plots", "activity_plots")

# --------------------------------------------------------------------------- definitions
# Distances are in cage widths (cw = mean of the top and bottom edges of cage_border_points, as
# fp_config.load_geometry), so one set of numbers serves every camera. Sources are noted per key.
DEFINITIONS = {
    "schema_version": 1,
    "generator_version": GENERATOR_VERSION,
    "units": {
        "cw": "cage width in px: mean length of the top and bottom edges of cage_border_points",
        "frame": "video frame index of the source .mp4 (0-based)",
    },
    "alignment": {
        "flying_offset_frames": 2,
        "resting_offset_frames": 0,
        "rule": "video_frame = tracker_index - 2 for state 1 (flying) points, = tracker_index for "
                "state 0 (resting). The forward moving detector stores the blob of frame f-1 at "
                "index f+1 (single_video_analysis.py, segment loop); resting detections are stored "
                "at their own frame. Measured 2026-10-22: flying points sit 2.3 px (median) from a "
                "blob at +2 and 31-52 px at -1..+1. The few tracks recovered by the backward pass "
                "(merge_forward_backward_tracks) are at offset 0; frames QA retries them there.",
        "duplicates": "a track with both a resting point and a flying point on one video frame "
                      "(the 2 frames around a take-off) keeps the flying point",
        "gap_fill_frames": 2,
    },
    "classes": {"0": "resting", "1": "flying"},
    "class_rule": "flying (1) = motion 2; every other motion (resting, walking, hold) = resting (0)",
    "motion_states": {
        "0": "resting: tracker state 0, not displaced",
        "1": "walking: state 0 displaced >= walking_disp_cw over 24 frames, or state 1 slower than "
             "real flight but faster than a hold (the moving detector also picks up walkers)",
        "2": "flying: state 1 at >= real_flight_speed_cw_per_frame (or with no speed estimate)",
        "3": "hold: state 1 with ~zero speed (tracker holding a lost or landed mosquito)",
    },
    "behaviour": {
        "real_flight_speed_cw_per_frame": 0.005,   # BuzzSwarm FLY_PX_FRAME 3 px/frame at Flare scale
        "hold_speed_cw_per_frame": 0.0005,         # below this a state-1 point is a hold, not a walker
        "speed_half_window_frames": 2,             # smoothed speed = |p(t+2) - p(t-2)| / 4 (BuzzSwarm)
        "walking_disp_cw": 0.008,                  # resting-tracker point moving >= this over 24 frames
        "walking_half_window_frames": 12,
        "walk_bout_min_frames": 25,
        "flyer_min_points": 10,                    # fp_config MIN_FLYER_PTS
        "flyer_mean_step_cw": 0.005,               # fp_config FLYER_STEP 4 px/frame at 780 px
        "link_max_gap_frames": 25,                 # fp_links_extract MAXGAP
        "zone_radius_cw": 0.10,                    # fp_config R_FRACS[R_MAIN]
        "swarm_radius_cw": 0.16,                   # BuzzSwarm SWARM_RADIUS_PX 100 px at Flare scale
        "swarm_min_flyers": 3,
        "swarm_min_frames": 50,
        "fly_px_step_cw_per_s": 300.0 / 780.0,     # fp_config FLY_PX_SPEED (clock burst series)
        "site_radius_cw": 0.01,                    # ~1 body length around a take-off / landing site
        "site_window_frames": [5, 25],             # checked after a take-off / before a landing
        "site_max_occupied": 0.5,                  # verified when the site is occupied < 50% of it
        "px_frames": [-25, -8, 8, 25],             # pixel check frames around a take-off / landing
        "px_min_contrast": 15,                     # the occupied side must be at least this dark
        "px_min_change": 8,                        # ... and this much darker than the empty side
        "px_window_cw": 0.006,                     # site window half-size (~7 px at 1080p)
    },
    "events": {
        "take_off": "0->1 change in an assembled track, previous resting point <= 25 frames before; "
                    "the bout passes the flyer rule; x, y = take-off site",
        "landing": "1->0 change, next resting point <= 25 frames after; x, y = landing site",
        "zone_pass": "a flyer bout comes within zone_radius_cw of the zone polygon; frame = closest",
        "arrival": "a bout linked at both ends that starts > zone_radius_cw from the zone and lands "
                   "within it (fp_links_build arrival_prob)",
        "zone_landing": "a landing inside the zone polygon (fp_links_build landing_poly)",
        "walk_bout": ">= walk_bout_min_frames consecutive walking points of one track",
        "swarm": ">= swarm_min_flyers real flyers within swarm_radius_cw of the cage centroid for "
                 ">= swarm_min_frames frames",
        "verified": "take_off / landing / zone_landing / arrival are checked twice. site_ok (tracker "
                    "points, annotate): a real take-off empties its site within 5-25 frames and a "
                    "real landing site was empty 25-5 frames before (no resting point within "
                    "site_radius_cw on >= 50% of those frames). px_ok (pixels, verify): the "
                    "edge-robust darkness at the site (px_window_cw) is compared between the "
                    "occupied side (take-off: frame -8; landing: min of +8, +25) and the empty side "
                    "(take-off: max of +8, +25; landing: -25 -- mosquitoes hover near the site "
                    "before touching down, so -8 is often still occupied): occupied >= "
                    "px_min_contrast and occupied - empty >= px_min_change. verified = site_ok and "
                    "px_ok. Measured 2026-10-22 on a 1080p dusk segment: ~35% of site_ok take-offs "
                    "and ~28% of site_ok landings pass, vs ~1% of the ones site_ok rejects (the two "
                    "checks agree). Most rejected events are a resting mosquito linked by the "
                    "tracker's assembly to a flyer that passed by, or a phantom resting track; train "
                    "behaviour models on verified events.",
        "not_used": "the cosine 'approach index' failed validation (DEVLOG 2026-10-20) and is left out",
    },
    "boxes": {
        "default_box_cw": {"resting": 0.015, "flying": 0.018},
        "size_min_contrast": [45, 30],
        "size_min_samples": 30,
        "dup_merge_cw": 0.0025,                    # same class closer than this = one mosquito tracked twice
        "shared_blob": "labels whose dark component holds >1 label keep the tracker's own position "
                       "(x_raw, y_raw); snapping them all to the component centre stacked touching "
                       "mosquitoes into identical boxes (fixed 2026-10-23)",
        "rule": "one fixed box per class and experiment, centred on the label's (snapped) point. Its "
                "size is the median dark-component size of supported, single-label detections with "
                "contrast >= 45 (else >= 30; else default_box_cw). Component boxes themselves are "
                "not used: at 1080p they grow from 4 px (contrast <= 20) to 17 px (contrast 45-80) "
                "with the threshold, not with the mosquito.",
    },
    "qa": {
        "snap_radius_cw": 0.006,
        "rest_thr_floor": 12.0,                    # OVERHAUL resting_threshold_floor
        "rest_thr_k": 6.0,                         # OVERHAUL resting_adaptive_k
        "rest_std_scale": 4.0,                     # single_video_analysis._REST_STD_SCALE
        "flat_thr_fallback": 12.0,
        "contrast_window_px": 7,
        "robust_erode_px": 3,
        "edge_robust_min_contrast": 6,
        "edge_note": "edge-robust contrast = frame darker than the 3x3-min-filtered reference. A panel "
                     "edge shifted by a pixel looks dark against the plain reference but not against "
                     "the min-filtered one. Resting-tracker labels need >= edge_robust_min_contrast to "
                     "count as supported (2026-10-22 smoke test: removed 31 of 38 edge-stack false "
                     "labels, 2-3% of resting labels in clean segments).",
        "motion_note": "motion contrast (mcontrast) = max over the contrast window of (frame t-1 or "
                       "t-2 minus frame t): how much darker the label's spot became at t. At 1080p the "
                       "noise is ~4-5 grey levels (99th pct); real flyers ~50 (median), resting ~3.",
        "population_max": 32,
        "population_note": "Most experiments hold about 30 live mosquitoes, 1-2 more at most, and fewer as "
                           "they die; dead ones in the background are not counted (the user, "
                           "2026-10-28). Kept labels per frame above population_max mean duplicates or "
                           "phantoms. The camera may not see the whole cage, so fewer is normal.",
        "unexplained_min_contrast": 20,
        "unexplained_min_robust": 12,
        "blob_area_cw2": [1.5e-5, 4.0e-4],
        "max_blob_cw": 0.06,
    },
    "sampling": {"frames_per_segment_target": 40, "frames_per_segment_aux": 6,
                 "min_spacing_frames": 50, "edge_frames": 10, "dropout_guard_frames": 25},
    "tiles": {"size": 640, "min_visible_frac": 0.6, "roi_margin_px": 8},
    "policy": {"max_uncertain_frac": 0.10, "max_unexplained": 1, "max_empty_frac": 0.10,
               "keep_unsupported_state0_min_contrast": 10,
               "flying_min_motion": 10, "flying_strong_motion": 25, "flying_near_rest_cw": 0.017,
               "rest_sat_radius_cw": 0.018, "rest_sat_merge_box": 0.6, "rest_sat_min_area": 0.5,
               "rest_sat_contrast_frac": 0.33,
               "junk_filter": True, "junk_grey_scale": 1.4, "junk_grey_ring_px": 4,
               "junk_filter_note": "Generator 1.3.0 (2026-11-06). A gold set of 1,000 test-day crops judged "
                                   "by eye showed the kept resting labels are only ~58% real (junk, shadows, "
                                   "parts of mosquitoes); confidence of a model trained on them does not "
                                   "separate the two. yolo_export/junk_filter.py scores every kept resting "
                                   "label from the tracker's own meta (contrasts, support, blob size in cage "
                                   "widths, track span); below its threshold (85% of real labels kept, ~71% "
                                   "of false removed in grouped CV) the label is dropped (junk) and its spot "
                                   "is greyed out in the tile (box = junk_grey_scale x the resting box, filled "
                                   "with the median of a junk_grey_ring_px ring around it) so the model is told "
                                   "neither 'mosquito' nor 'background' there. Spots touching a kept label's "
                                   "box are not greyed.",
               "rest_sat_note": "One mosquito tracked twice: the tracker often keeps a second, long-lived "
                                "resting track beside a resting mosquito, on its legs, a mesh hole or "
                                "empty wall (2026-10-28: about half of the resting labels had another "
                                "within 20 px; the two tracks coexist for the shorter one's whole life). "
                                "Among kept resting labels of one frame, strongest first (supported, "
                                "then contrast), a label within rest_sat_radius_cw (~21 px at 1080p) of a "
                                "stronger one is dropped (rest_sat) when it is closer than "
                                "rest_sat_merge_box x the resting box, is unsupported, sits on a blob "
                                "smaller than rest_sat_min_area x a single mosquito's blob (the "
                                "experiment median), has under rest_sat_contrast_frac x its stronger "
                                "neighbour's contrast, or shares a blob that has no room left "
                                "(round(blob area / single area) stronger labels already on it). Two real "
                                "mosquitoes side by side keep both labels. It removed 23% (Aedes) and "
                                "30% (Tbambusa) of kept resting labels.",
               "segment_max_label_ratio": 2.0, "segment_min_support": 0.75, "segment_max_edge_frac": 0.30,
               "flying_note": "A flying label also needs motion of its own (qa.motion_note): mcontrast "
                              ">= flying_strong_motion, or >= flying_min_motion with no resting label "
                              "within flying_near_rest_cw (~20 px at 1080p). Otherwise it is dropped "
                              "(static_fly); a mosquito there keeps its own resting label. A dropped "
                              "one on a dark blob with no resting label within that radius "
                              "(orphan_fly: maybe a sitting mosquito the tracker calls flying) counts "
                              "as an unexplained blob in its tile. Found "
                              "2026-10-26: at night most tracker 'flying' points were faint moving "
                              "traces beside resting mosquitoes (legs / antennae), passing pixel "
                              "support on the resting mosquito's blob. On the test day this rule kept "
                              "98-100% of the flyers a trained model found and dropped 84-94% of the "
                              "night 'flyers' it missed.",
               "note": "Kept labels = pixel-supported labels, plus unsupported resting-tracker "
                       "points (state 0) with contrast >= keep_unsupported_state0_min_contrast "
                       "(faint real mosquitoes) that are not edge suspects (qa.edge_note; edge "
                       "suspects are dropped). Unsupported moving-detector points (state 1: "
                       "flying, slow movers, holds) are dropped: frozen holds and misaligned points "
                       "live there, and a real mosquito they miss shows up as an unexplained blob. "
                       "A tile is used when its uncertain labels (kept but unsupported) are <= "
                       "max(1, max_uncertain_frac x kept) and it has <= max_unexplained "
                       "unexplained blobs. A whole segment is left out when its labels per frame "
                       "exceed segment_max_label_ratio x the experiment median (tracker noise), when "
                       "more than segment_max_edge_frac of its labels are edge suspects, or when the "
                       "support of its other labels (holds excluded) is below segment_min_support."},
    "stack": {
        "png_channels_cv2_bgr": ["t", "t-1", "t-2"],
        "model_input_rgb": ["t-2", "t-1", "t"],
        "note": "cv2.imread gives B=t, G=t-1, R=t-2; Ultralytics flips BGR->RGB, so the network sees "
                "(t-2, t-1, t). Build the Pi input the same way: RGB = (gray[t-2], gray[t-1], gray[t]). "
                "The 'single' variant stores gray[t] only (read back as 3 equal channels).",
        "gray": "cv2.cvtColor(decoded BGR frame, COLOR_BGR2GRAY)",
    },
    "clock": {"nominal_fps": NOMINAL_FPS, "edge_s": 60.0, "burst_bin_s": 5.0,
              "note": "real time = S0 + secs - a_session - b * secs / 86400, secs = (k * NF + frame) "
                      "/ 25 (fp_clock.py). Unfitted: a = 0, b from the rig fps in selection.yaml."},
    "light": {"dawn_ramp_minutes": 30, "dusk_ramp_minutes": 30, "default_night": [17, 5]},
}


def definitions():
    return copy.deepcopy(DEFINITIONS)


def log_default(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- roots and output
def configured_data_root():
    """The app's data root: config.json's data_root_override, else path_utils auto-detection."""
    import path_utils
    override = None
    try:
        with open(os.path.join(APP_DIR, "config.json"), "r") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            override = cfg.get("data_root_override")
    except (OSError, ValueError):
        pass
    root = path_utils.resolve_data_root(override)
    return root if root and os.path.isdir(root) else None


def discover_data_roots():
    """Every mounted Buzzwatch data root that has an Analysis/ tree: the configured one first,
    then any other drive (Windows) or volume (macOS) carrying a Buzzwatch folder."""
    cands = [configured_data_root()]
    if os.name == "nt":
        cands.extend("%s:\\Buzzwatch" % letter for letter in string.ascii_uppercase)
    else:
        cands.extend(sorted(glob.glob("/Volumes/*/Buzzwatch")) + sorted(glob.glob("/Volumes/*/BuzzWatch")))
    roots, seen = [], set()
    for c in cands:
        if not c or not os.path.isdir(os.path.join(c, "Analysis")):
            continue
        key = os.path.normcase(os.path.realpath(c))
        if key not in seen:
            seen.add(key)
            roots.append(os.path.normpath(c))
    return roots


def default_out_dir():
    root = configured_data_root()
    if not root:
        return None
    return os.path.join(os.path.dirname(os.path.normpath(root)), OUT_DIRNAME)


_DATA_TREE_RE = re.compile(r"(^|/)buzzwatch(/|$)", re.I)


def check_output_dir(out_dir, roots=()):
    """Refuse an output folder inside a Buzzwatch data root (or any folder named Buzzwatch)."""
    if not out_dir:
        raise ValueError("No output folder given and no data root found to derive one; pass --out.")
    o = os.path.normcase(os.path.abspath(out_dir))
    if _DATA_TREE_RE.search(o.replace("\\", "/")):
        raise ValueError("Refusing to write inside a Buzzwatch data tree: %s" % out_dir)
    for r in roots or ():
        p = os.path.normcase(os.path.abspath(r))
        if o == p or o.startswith(p.rstrip("\\/") + os.sep):
            raise ValueError("Refusing to write inside the data root %s: %s" % (r, out_dir))
    return os.path.abspath(out_dir)


# --------------------------------------------------------------------------- small utilities
def seed_for(name):
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def parse_windows(text):
    """'40-50' or '40-50,10-15' -> [(40.0, 50.0), (10.0, 15.0)] (minutes of the hour)."""
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-", 1)
        out.append((float(a), float(b)))
    return out


# {cage}_{YYYYMMDD}_{HHMMSS}[-{attempt}]_{NNNNN}. The Pi recorder adds '-{attempt}' when a session
# name already exists (its clock went backwards); see src/buzzphono/io/naming.py (STEM_RE).
_STEM_RE = re.compile(r"^(?P<cage>.+)_(?P<date>\d{8})_(?P<time>\d{6})(?:-(?P<attempt>\d+))?_(?P<seg>\d+)$")


def parse_stem(stem):
    """{'cage', 'session_date', 'session_time', 'attempt', 'segment_int'} for a segment stem, or None."""
    base = os.path.basename(stem)
    if base.startswith(TRACK_PREFIX):
        base = base[len(TRACK_PREFIX):]
    m = _STEM_RE.match(base)
    if not m:
        return None
    return {"cage": m.group("cage"), "session_date": m.group("date"), "session_time": m.group("time"),
            "attempt": int(m.group("attempt")) if m.group("attempt") else None,
            "segment_int": int(m.group("seg"))}


def session_key(stem):
    """('YYYYMMDD_HHMMSS[-attempt]', segment index), or (None, None) for a name that does not parse.
    The attempt stays in the key so a repeated session name is its own session; the first 15
    characters are always the session's start time (clock._s0)."""
    p = parse_stem(stem)
    if not p:
        return None, None
    key = "%s_%s" % (p["session_date"], p["session_time"])
    if p["attempt"] is not None:
        key += "-%d" % p["attempt"]
    return key, p["segment_int"]


# Everything a BuzzWatch tracking file holds (as src/buzzphono/io/safe_pickle.py allows): the tracker
# class, standard containers, datetime and numpy. pandas/pyarrow objects (only the final files'
# population_variables table, which yolo_export never reads) load as inert placeholders.
_PICKLE_ALLOWED = {
    ("buzzwatch_data_analysis.mosquito_obj_tracker", "mosquito_obj_tracker"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("builtins", "slice"), ("builtins", "bytearray"), ("builtins", "set"), ("builtins", "frozenset"),
    ("builtins", "complex"),
    ("datetime", "datetime"), ("datetime", "date"), ("datetime", "time"), ("datetime", "timedelta"),
    ("datetime", "timezone"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("_codecs", "encode"),
}


class _PandasPlaceholder(object):
    """Inert stand-in for any pandas object inside a tracking pickle."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _PandasPlaceholder()


def _pandas_factory(*args, **kwargs):
    return _PandasPlaceholder()


def _tracking_unpickler(fh):
    import pickle

    class _Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in _PICKLE_ALLOWED:
                try:
                    return super(_Restricted, self).find_class(module, name)
                except ImportError:  # a numpy-2 pickle read with numpy 1
                    if module.startswith("numpy._core"):
                        return super(_Restricted, self).find_class(
                            module.replace("numpy._core", "numpy.core"), name)
                    raise
            if module == "numpy.dtypes":
                import numpy.dtypes as nd  # numpy >= 1.25 only
                if isinstance(getattr(nd, name, None), type):
                    return getattr(nd, name)
            if module.split(".")[0] in ("pandas", "pyarrow"):
                return _PandasPlaceholder if name[:1].isupper() else _pandas_factory
            raise pickle.UnpicklingError("unexpected object %s.%s: not a BuzzWatch tracking file" % (module, name))

    return _Restricted(fh)


def load_tracking(path):
    """The frozen tracker object, or None if the file is unreadable. Loaded with an allow-listed
    unpickler: a plain pickle.load would run whatever code a tampered file asks for, and the data
    roots are discovered on any mounted drive."""
    try:
        with open(path, "rb") as fh:
            return _tracking_unpickler(fh).load()
    except Exception as exc:  # noqa: BLE001
        log_default("  ! load failed %s: %s" % (path, exc))
        return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def table_path(path):
    """The file actually holding a table written by write_table (Parquet, or the CSV fallback)."""
    if os.path.isfile(path):
        return path
    alt = re.sub(r"\.parquet$", ".csv.gz", path)
    return alt if os.path.isfile(alt) else None


def write_table(df, path):
    """Parquet (pyarrow, imported lazily by pandas) with a gzipped-CSV fallback; atomic."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        alt = re.sub(r"\.parquet$", ".csv.gz", path)
        df.to_csv(alt + ".tmp", index=False, compression="gzip")
        os.replace(alt + ".tmp", alt)
        return alt
    df.to_parquet(path + ".tmp", index=False, compression="snappy")
    os.replace(path + ".tmp", path)
    return path


def read_table(path, columns=None):
    import pandas as pd
    p = table_path(path)
    if p is None:
        return None
    if p.endswith(".parquet"):
        return pd.read_parquet(p, columns=columns)
    df = pd.read_csv(p, compression="gzip")
    return df[columns] if columns else df


# --------------------------------------------------------------------------- geometry
def load_zones(folder):
    """{name: (n, 2) float polygon} from <folder>/custom_zones.json, in file order ({} if none)."""
    zj = read_json(os.path.join(folder, "custom_zones.json"), default=None)
    out = {}
    if isinstance(zj, dict):
        for name, z in zj.items():
            poly = z.get("polygon") if isinstance(z, dict) else None
            if poly and len(poly) >= 3:
                out[str(name)] = np.asarray(poly, dtype=float).reshape(-1, 2)
    return out


def load_settings(exp_dir):
    import yaml
    with open(os.path.join(exp_dir, "buzzwatch_track_settings.yml"), "r") as f:
        return yaml.safe_load(f) or {}


def load_geometry(exp_dir, zones_dir=None):
    """Cage polygon, cage width (cw, px), centroid, bbox and zones of one experiment."""
    from buzzwatch_data_analysis.flight_center_distance_analysis import calculate_cage_centroid
    settings = load_settings(exp_dir)
    pts = settings.get("cage_border_points") or []
    cage = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(cage) < 3:
        raise ValueError("no cage border drawn for %s" % exp_dir)
    if len(cage) == 4:
        width = 0.5 * (np.hypot(*(cage[1] - cage[0])) + np.hypot(*(cage[2] - cage[3])))
    else:
        width = float(cage[:, 0].max() - cage[:, 0].min())
    # Float points on purpose: with the YAML's ints the helper builds a platform-int array, which is
    # int32 on Windows (numpy 1.24), and its shoelace sums overflow for 1080p cages (a 1920x1080
    # cage came out at (281, -39) instead of (861, 541)). Floats give the intended centroid.
    centroid = np.asarray(calculate_cage_centroid(cage.tolist()), dtype=float)
    return {
        "cage": cage, "width": float(width), "centroid": centroid,
        "bbox": (float(cage[:, 0].min()), float(cage[:, 1].min()),
                 float(cage[:, 0].max()), float(cage[:, 1].max())),
        "zones": load_zones(zones_dir or exp_dir),
        "tracking": settings.get("tracking") or {},
    }


def zone_maps(zones, frame_w, frame_h):
    """Per-zone distance maps (px to the polygon, 0 inside) and an 'inside' label map (zone index
    + 1, 0 = none), both at 1/ZONE_DS resolution, as fp_extract.setup builds them."""
    import cv2
    gw, gh = int(frame_w) // ZONE_DS + 2, int(frame_h) // ZONE_DS + 2
    names = list(zones)
    inside = np.zeros((gh, gw), np.uint8)
    dists = []
    for i, n in enumerate(names):
        m = np.zeros((gh, gw), np.uint8)
        cv2.fillPoly(m, [np.round(zones[n] / ZONE_DS).astype(np.int32)], 255)
        inside[(m > 0) & (inside == 0)] = i + 1
        src = np.where(m > 0, 0, 255).astype(np.uint8)
        dists.append(cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32) * ZONE_DS)
    return {"names": names, "inside": inside, "dists": dists, "gw": gw, "gh": gh}


def map_lookup(arr, gw, gh, x, y):
    """Vectorised lookup of a 1/ZONE_DS map at full-resolution coordinates (finite x, y only)."""
    xi = np.clip((np.asarray(x) / ZONE_DS).astype(np.int64), 0, gw - 1)
    yi = np.clip((np.asarray(y) / ZONE_DS).astype(np.int64), 0, gh - 1)
    return arr[yi, xi]
