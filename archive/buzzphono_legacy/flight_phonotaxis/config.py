"""Constants, parameters, geometry and chunk listing for the flight-phonotaxis analysis.

Ported from the 2026-10 research script ``fp_config.py``. The hard-coded per-experiment ``EXPS``
dict is gone: everything is derived from the experiment folder the caller passes in
(``buzzwatch_track_settings.yml`` cage corners, ``custom_zones.json`` zones, the
``tracking_moving/`` file names, and the frames per chunk read from the tracking data itself).

Light module on purpose (numpy/json/re only): spawned extraction workers import it.
"""
import glob
import json
import os
import re
from collections import OrderedDict

import numpy as np

# ---- constants (identical to fp_config / fp_extract / fp_build) ----------------------------------
NF_DEFAULT = 30060     # frames per rpicam --segment chunk on the Penzance camera (Bangkok: 30000)
NMAX_MIN = 30200       # per-chunk array length used by the reference run (a few frames of slack)
BLOCK = 250            # frames per heat-map block (10 s)
GRID = 40              # heat-map bins per axis over the cage bounding box
REF_WIDTH = 780.0      # Aedes camera cage width in px; s = cage_width / REF_WIDTH
FLYER_STEP = 4.0       # thesis "real flyer" rule: mean step >= 4 px/frame (x s, x multiplier)
FLY_PX_SPEED = 300.0   # thesis activity metric: steps >= 300 px/s (x s)
MIN_FLYER_PTS = 10
MIN_MEDIAN_REF = 3.0   # reference value of the optional sustained-movement filter (median step >= 3 px/frame x s)
R_MAIN = 1            # index of the main neighbourhood radius in r_fracs()
COLONY = 30.0          # resting-count cap per frame (old resting metric)
FPS = 25.0             # nominal camera rate (recording clock)
FPS_EFF = 25.015       # effective rate used for coverage thresholds and per-minute rates
DS = 2                 # distance-map downsample
MAXGAP = 25            # frames (1 s): a state transition across a longer gap is not a trusted link

ROLES = ["speaker", "right", "top", "bottom"]
# 'right' is the CSV key of the wall opposite the speaker (kept so CSV columns stay comparable with
# the research run); the UI and plots always call it "Opposite wall".
ROLE_LABELS = {"speaker": "Speaker", "right": "Opposite wall", "top": "Top wall", "bottom": "Bottom wall"}

KEY_RE = re.compile(r'^moving_(.+)_(\d{8})_(\d{6})_(\d{5})\.pkl$')

OUT_SUBDIR = os.path.join("plots", "flight_phonotaxis")
FLIGHT_ZONES_NAME = "custom_zones_flight.json"


class FlightPhonotaxisError(ValueError):
    """A problem the user can fix (missing zones, no tracking data, bad parameter). Shown verbatim."""


class Cancelled(Exception):
    """Raised when the caller's cancel Event is set."""


def role_label(zone):
    """Human label for a role key or a 'speaker - <role>' contrast."""
    if zone in ROLE_LABELS:
        return ROLE_LABELS[zone]
    if zone == "speaker - mean(controls)":
        return "Speaker - mean of the 3 other walls"
    if zone.startswith("speaker - "):
        return "Speaker - " + ROLE_LABELS.get(zone[len("speaker - "):], zone[len("speaker - "):])
    if zone == "all":
        return "whole cage"
    return zone


# ---- parameters -----------------------------------------------------------------------------------
class FlightParams(object):
    """User-facing analysis parameters. The defaults reproduce the research run exactly."""

    def __init__(self, on=(40, 50), off=(28, 38), light_hours=(5, 17), flyer_mult=1.0, radius=0.1,
                 onset_minutes=2, sham=(10, 20), early=(10, 30), start=None, end=None, min_median=0.0):
        self.on = (int(on[0]), int(on[1]))
        self.off = (int(off[0]), int(off[1]))
        self.light_hours = (int(light_hours[0]), int(light_hours[1]))
        self.flyer_mult = float(flyer_mult)
        self.radius = float(radius)
        self.min_median = float(min_median)    # 0 = filter off (published Penzance / Aedes results)
        self.onset_minutes = int(onset_minutes)
        self.sham = (int(sham[0]), int(sham[1]))
        self.early = (int(early[0]), int(early[1]))
        self.start = start
        self.end = end
        self.validate()

    def validate(self):
        for name, (lo, hi) in (("ON window", self.on), ("OFF window", self.off),
                               ("sham window", self.sham), ("early baseline", self.early)):
            if not (0 <= lo < hi <= 60):
                raise FlightPhonotaxisError("%s must be minutes of the hour with start < end (0-60); got %d-%d."
                                            % (name, lo, hi))
        if self.on[1] - self.on[0] <= self.onset_minutes:
            raise FlightPhonotaxisError("ON window must be longer than %d minutes." % self.onset_minutes)
        if self.on[0] < 10 or self.on[0] + 12 > 60:
            raise FlightPhonotaxisError("ON start must be between minute 10 and 48 (the clock fit looks "
                                        "for the flight burst from 10 min before to 12 min after it).")
        ls, le = self.light_hours
        if not (0 <= ls < le <= 24) or le - ls < 3:
            raise FlightPhonotaxisError("Light hours must be start < end within 0-24 h and at least 3 h long; "
                                        "got %d-%d." % (ls, le))
        if not (0.05 <= self.flyer_mult <= 10):
            raise FlightPhonotaxisError("Flyer-threshold multiplier must be between 0.05 and 10.")
        if not (0.0 <= self.min_median <= 20.0):
            raise FlightPhonotaxisError("Minimum sustained movement must be between 0 (off) and 20 px/frame "
                                        "(x the camera scale); the reference value for dense cages is 3.")
        if not (0.01 <= self.radius <= 0.5):
            raise FlightPhonotaxisError("Neighbourhood radius must be between 0.01 and 0.5 cage widths.")

    # windows / pairs (fp_build.WINDOWS / PAIRS, generalised; identical strings at the defaults)
    def windows(self):
        on0, on1 = self.on
        k = on0 + self.onset_minutes
        return OrderedDict([("OFF", self.off), ("ON", self.on), ("ON_onset", (on0, k)), ("ON_sust", (k, on1)),
                            ("SHAM", self.sham), ("EARLY", self.early)])

    def pairs(self):
        on0, on1 = self.on
        k = on0 + self.onset_minutes
        return [("ON vs OFF", "ON", "OFF"),
                ("ON-onset (%d-%d) vs OFF" % (on0, k), "ON_onset", "OFF"),
                ("ON-sustained (%d-%d) vs OFF" % (k, on1), "ON_sust", "OFF"),
                ("sham: OFF %d-%d vs %d-%d" % (self.off + self.sham), "OFF", "SHAM"),
                ("ON vs early baseline (%d-%d)" % self.early, "ON", "EARLY")]

    # light phase (fp_build.light_of / PHASES, generalised)
    @property
    def light_label(self):
        return "light (%02d-%02dh)" % self.light_hours

    @property
    def dark_label(self):
        return "dark (%02d-%02dh)" % (self.light_hours[1], self.light_hours[0])

    def light_of(self, h):
        ls, le = self.light_hours
        return self.light_label if ls <= h < le else self.dark_label

    def phases(self):
        ls, le = self.light_hours
        night = set(range(24)) - set(range(ls, le))
        return [("dawn (%02d-%02dh)" % (ls, ls + 1), {ls}),
                ("day (%02d-%02dh)" % (ls + 1, le - 1), set(range(ls + 1, le - 1))),
                ("dusk (%02d-%02dh)" % (le - 1, le), {le - 1}),
                ("night (%02d-%02dh)" % (le, ls), night)]

    def phase_of(self, h):
        for name, hrs in self.phases():
            if h in hrs:
                return name
        return None

    def r_fracs(self):
        """neighbourhood radii in cage widths: half, main, double (0.05 / 0.1 / 0.2 at the default)."""
        return (self.radius / 2.0, self.radius, self.radius * 2.0)

    def to_dict(self):
        return dict(on=list(self.on), off=list(self.off), light_hours=list(self.light_hours),
                    flyer_mult=self.flyer_mult, min_median=self.min_median, radius=self.radius, onset_minutes=self.onset_minutes,
                    sham=list(self.sham), early=list(self.early),
                    start=None if self.start is None else str(self.start),
                    end=None if self.end is None else str(self.end))


def nmax_for(nf):
    return max(NMAX_MIN, int(nf) + 140)


def nblk_for(nmax):
    return nmax // BLOCK + 1


# ---- tracking files ---------------------------------------------------------------------------------
def load_pickle(path):
    """Load a tracking pickle with the BuzzPhono compat unpickler (numpy._core -> numpy.core)."""
    from buzzphono_legacy.plot_speaker_distance_phonotaxis import _CompatUnpickler
    with open(path, "rb") as fh:
        return _CompatUnpickler(fh).load()


def moving_path(exp, stem):
    return os.path.join(exp, "tracking_moving", "moving_" + stem + ".pkl")


def resting_path(exp, stem):
    return os.path.join(exp, "tracking_resting", "resting_" + stem + ".pkl")


def final_path(exp, stem):
    return os.path.join(exp, "final_tracking_data", "forward_mosq_tracks_" + stem)


def list_chunks(exp):
    """[(stem, session_key 'YYYYMMDD_HHMMSS', k)] for every moving pkl, sorted."""
    out = []
    for p in sorted(glob.glob(os.path.join(exp, "tracking_moving", "moving_*.pkl"))):
        m = KEY_RE.match(os.path.basename(p))
        if m:
            out.append((os.path.basename(p)[len("moving_"):-4], m.group(2) + "_" + m.group(3), int(m.group(4))))
    return out


def session_start(sess):
    import pandas as pd
    return pd.Timestamp("%s-%s-%s %s:%s:%s" % (sess[:4], sess[4:6], sess[6:8], sess[9:11], sess[11:13], sess[13:15]))


def detect_frames_per_chunk(exp, chunks, log=None):
    """Frames per recording chunk, read from the data (30060 on the Penzance rpicam, 30000 in Bangkok).

    Primary: len(mt.time_stamp) of a few final_tracking_data files from the middle of the run (full
    chunks all have the same length; only a session's last chunk is shorter), taking the largest.
    Fallback: the largest frame index in a few tracking_moving pickles, + 1."""
    if not chunks:
        raise FlightPhonotaxisError("No tracking_moving/moving_*.pkl files in %s." % exp)
    n = len(chunks)
    sample = sorted(set([chunks[n // 4][0], chunks[n // 2][0], chunks[(3 * n) // 4][0]]))
    lens = []
    for stem in sample:
        p = final_path(exp, stem)
        if not os.path.isfile(p):
            continue
        try:
            mt = load_pickle(p)
            lens.append(len(mt.time_stamp))
            del mt
        except Exception as exc:  # noqa: BLE001
            if log:
                log("frames-per-chunk: could not read %s (%s)" % (os.path.basename(p), exc))
    if lens and max(lens) > 1000:
        return int(max(lens))
    mx = -1
    for stem in sample:
        try:
            obj = load_pickle(moving_path(exp, stem))
        except Exception:  # noqa: BLE001
            continue
        for td in (obj.objects.values() if getattr(obj, "objects", None) else []):
            ts = [t for t in td.get("time_stamp", []) if t is not None and np.isfinite(t)]
            if ts:
                mx = max(mx, int(max(ts)))
        del obj
    if mx > 1000:
        return mx + 1
    raise FlightPhonotaxisError("Could not determine the frames per chunk from the tracking data in %s; "
                                "pass frames_per_chunk explicitly." % exp)


# ---- geometry ---------------------------------------------------------------------------------------
def resolve_zones_file(exp, zones_file=None):
    """Which zones file to use: an explicit one, else <exp>/custom_zones.json when it holds a speaker plus
    three other zones, else <exp>/plots/flight_phonotaxis/custom_zones_flight.json (traced for this analysis)."""
    if zones_file:
        if not os.path.isfile(zones_file):
            raise FlightPhonotaxisError("Zones file not found: %s" % zones_file)
        return zones_file
    root = os.path.join(exp, "custom_zones.json")
    alt = os.path.join(exp, OUT_SUBDIR, FLIGHT_ZONES_NAME)
    root_err = None
    if os.path.isfile(root):
        try:
            load_zones(root)
            return root
        except FlightPhonotaxisError as exc:
            root_err = exc
    if os.path.isfile(alt):
        return alt
    if root_err is not None:
        raise root_err
    raise FlightPhonotaxisError(
        "No zones defined for %s.\nFlight phonotaxis needs custom_zones.json in the experiment folder with a "
        "zone named 'speaker' and exactly three other zones (one on each other wall). Use 'Define speaker zone' "
        "and 'Add comparison zone' (three times), or choose a zones file." % os.path.basename(os.path.normpath(exp)))


def load_zones(zones_file):
    """(names, {name: Nx2 array}) with 'speaker' first and the three controls in file order."""
    try:
        with open(zones_file) as f:
            zj = json.load(f, object_pairs_hook=OrderedDict)
    except Exception as exc:  # noqa: BLE001
        raise FlightPhonotaxisError("Could not read zones file %s: %s" % (zones_file, exc))
    polys = OrderedDict()
    for name, v in zj.items():
        if isinstance(v, dict) and isinstance(v.get("polygon"), list) and len(v["polygon"]) >= 3:
            polys[name] = np.asarray(v["polygon"], float)
    if "speaker" not in polys:
        raise FlightPhonotaxisError("%s has no zone named 'speaker' (found: %s)." % (
            os.path.basename(zones_file), ", ".join(polys) or "no polygons"))
    ctrls = [n for n in polys if n != "speaker"]
    if len(ctrls) != 3:
        raise FlightPhonotaxisError(
            "%s must have the speaker zone plus exactly three control zones (one per other wall); it has %d "
            "control zone(s): %s." % (os.path.basename(zones_file), len(ctrls), ", ".join(ctrls) or "none"))
    names = ["speaker"] + ctrls
    return names, dict((n, polys[n]) for n in names)


def load_geometry(exp, zones_file):
    """cage corners (4x2, TL TR BR BL), zones {name: Nx2}, names, scale s, cage width, centre, bbox."""
    import yaml
    yml = os.path.join(exp, "buzzwatch_track_settings.yml")
    try:
        with open(yml) as f:
            y = yaml.safe_load(f)
        cage = np.asarray(y["cage_border_points"], float)
    except Exception as exc:  # noqa: BLE001
        raise FlightPhonotaxisError("Could not read the cage borders (cage_border_points) from %s: %s" % (yml, exc))
    if cage.shape != (4, 2):
        raise FlightPhonotaxisError("cage_border_points in %s must be 4 corners; got shape %s. Draw the cage "
                                    "borders first." % (yml, cage.shape))
    names, zones = load_zones(zones_file)
    top = np.hypot(*(cage[1] - cage[0]))
    bot = np.hypot(*(cage[2] - cage[3]))
    width = 0.5 * (top + bot)
    return dict(cage=cage, zones=zones, names=names, width=width, s=width / REF_WIDTH,
                centre=cage.mean(axis=0),
                bbox=(cage[:, 0].min(), cage[:, 0].max(), cage[:, 1].min(), cage[:, 1].max()),
                zones_file=zones_file)


_SETUP = {}


def setup(geom, r_fracs):
    """Distance maps, zone centroids, polygon paths and wall sectors (fp_extract.setup). Memoised per
    process; spawned workers call it once per experiment."""
    key = (geom["zones_file"], tuple(np.round(geom["cage"].ravel(), 3)),
           tuple((n, tuple(np.round(geom["zones"][n].ravel(), 3))) for n in geom["names"]), tuple(r_fracs))
    if key in _SETUP:
        return _SETUP[key]
    import cv2
    from matplotlib.path import Path as MplPath
    g = geom
    names = g["names"]
    allpts = np.vstack([g["cage"]] + [g["zones"][n] for n in names])
    gw, gh = int(allpts[:, 0].max() + 200) // DS + 2, int(allpts[:, 1].max() + 200) // DS + 2
    dms, cents, paths = [], [], []
    for n in names:
        poly = g["zones"][n]
        m = np.zeros((gh, gw), np.uint8)
        cv2.fillPoly(m, [np.round(poly / DS).astype(np.int32)], 255)
        src = np.where(m > 0, 0, 255).astype(np.uint8)
        dms.append(cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32) * DS)
        yy, xx = np.nonzero(m)
        if len(xx) == 0:
            raise FlightPhonotaxisError("Zone '%s' is empty (no pixels inside the polygon)." % n)
        cents.append(np.array([xx.mean() * DS, yy.mean() * DS]))
        paths.append(MplPath(poly))
    cents = np.array(cents)
    # sectors: triangle (centre, corner i, corner i+1) = wall i; wall -> nearest zone centroid
    cage, ctr = g["cage"], g["centre"]
    cang = np.arctan2(cage[:, 1] - ctr[1], cage[:, 0] - ctr[0])
    sign = 1.0
    rel = (cang - cang[0]) % (2 * np.pi)
    if not (rel[1] < rel[2] < rel[3]):
        sign = -1.0
        rel = (-(cang - cang[0])) % (2 * np.pi)
    if not (rel[1] < rel[2] < rel[3]):
        raise FlightPhonotaxisError("The four cage corners are not in order around the cage; redraw the cage borders.")
    mids = 0.5 * (cage + np.roll(cage, -1, axis=0))
    sec2zone = np.array([int(np.argmin(np.hypot(*(cents - m).T))) for m in mids])
    if len(set(sec2zone.tolist())) != 4:
        raise FlightPhonotaxisError(
            "Each of the four zones must sit on a different cage wall (wall -> nearest zone came out as %s for "
            "zones %s). Move the zones so there is one per wall." % (sec2zone.tolist(), names))
    s = g["s"]
    G = dict(g=g, names=names, dms=dms, cents=cents, paths=paths, gw=gw, gh=gh, rel=rel, sign=sign, cang0=cang[0],
             ctr=ctr, sec2zone=sec2zone, s=s, Rpx=[f * g["width"] for f in r_fracs])
    _SETUP[key] = G
    return G


def role_order(geom):
    """indices into geom['names'] giving [speaker, opposite, top, bottom] by geometry (fp_build.role_order).
    The opposite wall is the control farthest from the speaker, so the speaker can be on any wall."""
    names = geom["names"]
    cen = dict((n, geom["zones"][n].mean(axis=0)) for n in names)
    spk = names[0]
    ctrls = names[1:]
    right = max(ctrls, key=lambda n: np.hypot(*(cen[n] - cen[spk])))
    rest = [n for n in ctrls if n != right]
    top = min(rest, key=lambda n: cen[n][1])
    bottom = [n for n in rest if n != top][0]
    return [names.index(n) for n in (spk, right, top, bottom)], [spk, right, top, bottom]
