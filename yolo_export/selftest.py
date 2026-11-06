"""selftest: headless checks of the exporter logic (synthetic data) and of a written dataset.

    python -m yolo_export.selftest                 # synthetic checks only
    python -m yolo_export.selftest <out_dir>       # + YOLO format checks of an output folder

Synthetic checks: tracker -> video frame alignment (+2 for flying, 0 for resting, the take-off
duplicate rule), motion classes, take-off / landing frames, the flying-label motion rule, the
resting-satellite rule, tile <-> full-frame label round trip, split determinism, and the output-folder guard. Exits non-zero on the
first failure.
"""
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo_export import build as BU  # noqa: E402
from yolo_export import common as C  # noqa: E402
from yolo_export import tracks as TR  # noqa: E402


def _defs():
    """The definitions with the junk filter off: these checks are of the box rules on synthetic
    labels, which lack the tracker meta the trained junk-filter model scores."""
    defs = C.definitions()
    defs["policy"]["junk_filter"] = False
    return defs


class _MT(object):
    def __init__(self, objects, n):
        self.objects = objects
        self.time_stamp = list(range(n))


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok  " + msg)


def test_alignment():
    # One mosquito: resting at (100, 100) for tracker idx 0..49, flies +10 px/frame for idx 50..79,
    # rests again at the end position for idx 80..99 (the tracker stores flying points 2 late).
    coords, state = [], []
    for i in range(100):
        if i < 50:
            coords.append((100.0, 100.0)); state.append(0)
        elif i < 80:
            coords.append((100.0 + 10 * (i - 49), 100.0)); state.append(1)
        else:
            coords.append((400.0, 100.0)); state.append(0)
    mt = _MT(OrderedDict([(7, {"coordinates": coords, "state": state, "start": 0})]), 100)
    defs = _defs()
    raw = TR.raw_points(mt)
    P = TR.aligned_points(raw, defs, 1000.0, np.array([500.0, 500.0]), None)
    vf = dict(zip(P["vf"].tolist(), P["state"].tolist()))
    _check(vf[48] == 1 and vf[49] == 1, "take-off: frames 48/49 keep the flying point (resting duplicate dropped)")
    _check(P["vf"][P["state"] == 1].min() == 48 and P["vf"][P["state"] == 1].max() == 77,
           "flying points re-indexed to video frame = tracker index - 2")
    _check(78 not in vf and 79 not in vf and vf[80] == 0, "landing: 2-frame gap before the resting point at its own frame")
    _check(bool(P["real"][P["state"] == 1][2:-2].all()), "10 px/frame at cw 1000 is real flight (>= 0.005 cw)")
    act = TR.activity(P, 100, defs, 1000.0, 0)
    ev = TR.events(raw, P, act, defs, 1000.0, None)
    take = ev[ev.event == "take_off"]
    land = ev[ev.event == "landing"]
    _check(len(take) == 1 and int(take.video_frame.iloc[0]) == 48 and take.x.iloc[0] == 100.0,
           "take-off event at video frame 48 at the take-off site")
    _check(len(land) == 1 and int(land.video_frame.iloc[0]) == 80 and land.x.iloc[0] == 400.0,
           "landing event at video frame 80 at the landing site")


def test_shared_blob():
    import pandas as pd
    defs = _defs()
    # a, b: two touching mosquitoes in one blob (snapped to its centre); c: a second track 1 px from b
    L = pd.DataFrame({"frame_id": "f", "cls": 0, "state": 0, "motion": 0, "supported": True, "edge_suspect": False,
                      "contrast": [40, 30, 20], "comp_nlabels": 3, "comp_w": 30, "comp_h": 15,
                      "x": [110.0, 110.0, 110.0], "y": [50.0, 50.0, 50.0],
                      "x_raw": [100.0, 120.0, 121.0], "y_raw": [50.0, 50.0, 50.0]})
    L = BU.label_boxes(L, {0: (16.0, 16.0), 1: (20.0, 20.0)}, defs, 1000.0)
    _check(list(L.bx) == [100.0, 120.0, 121.0], "labels sharing a blob keep their tracker positions")
    _check(list(L.keep) == [True, True, False] and bool(L.merged.iloc[2]),
           "same class within dup_merge_cw -> one label (the higher contrast one kept)")


def test_flying_motion():
    import pandas as pd
    defs = _defs()
    pol = defs["policy"]
    weak = (pol["flying_min_motion"] + pol["flying_strong_motion"]) // 2
    # r: resting mosquito; a: weak mover beside it (a leg); b: strong mover beside it (a take-off);
    # c: weak mover far from any resting label; d: no motion at all; e: weak mover in another frame
    L = pd.DataFrame({"frame_id": ["f", "f", "f", "f", "f", "g"], "cls": [0, 1, 1, 1, 1, 1],
                      "state": [0, 1, 1, 1, 1, 1], "motion": [0, 2, 2, 2, 2, 2], "supported": True,
                      "edge_suspect": False, "contrast": 40, "comp_nlabels": 1, "comp_w": 15, "comp_h": 15,
                      "x": [100.0, 110.0, 114.0, 400.0, 600.0, 110.0], "y": 100.0,
                      "mcontrast": [3, weak, pol["flying_strong_motion"], weak, pol["flying_min_motion"] - 1, weak]})
    L = BU.label_boxes(L, {0: (16.0, 16.0), 1: (20.0, 20.0)}, defs, 1000.0)
    _check(list(L.keep) == [True, False, True, True, False, True],
           "flying labels: weak motion beside a resting mosquito or no motion -> dropped; strong motion, "
           "or weak motion away from resting mosquitoes (same frame only) -> kept")
    _check(list(L.static_fly) == [False, True, False, False, True, False], "dropped flying labels flagged static_fly")
    _check(list(L.orphan_fly) == [False, False, False, False, True, False],
           "a dropped flyer on its own blob, away from resting labels, is an orphan (unexplained blob)")
    F = pd.DataFrame({"frame_id": ["f"], "stem": ["s"], "width": [1920], "height": [1080], "ref_kind": ["rest"]})
    t = BU.plan_tiles(F, L[L.frame_id == "f"], None, defs, (0.0, 0.0, 640.0, 640.0))
    _check(len(t) == 1 and t[0]["n_unexplained"] == 1 and t[0]["n_kept"] == 3, "plan_tiles counts the orphan as unexplained")
    cols = ["mcontrast", "keep", "static_fly", "orphan_fly"]
    L2 = BU.label_boxes(L.drop(columns=cols), {0: (16.0, 16.0), 1: (20.0, 20.0)}, defs, 1000.0)
    _check(bool(L2.keep.all()), "labels without mcontrast (synthetic) are not filtered")
    L3 = BU.label_boxes(L.drop(columns=cols[1:]), {0: (16.0, 16.0), 1: (20.0, 20.0)}, defs, None)
    _check(list(L3.keep) == [True, True, True, True, False, True] and not L3.orphan_fly.any(),
           "without cw only the motion thresholds apply")


def test_resting_satellites():
    import pandas as pd
    defs = _defs()
    # cw 1000: radius 18 px, merge 0.6 x 16 px box = 9.6 px; too few clean labels -> single blob
    # area 0.6 x 16 x 16 = 153.6 px^2. a: mosquito, b: its leg blob; c, d: two real neighbours;
    # e, f: one blob tracked twice 8 px apart; g, h, i: a touching pair sharing a 2-mosquito blob plus
    # a third label on it; k, j: mosquito + unsupported ghost; m, l: mosquito + a much fainter
    # neighbour; n: an isolated weak label; z: a flyer beside a
    names = list("abcdefghikjmlnz")
    xs = [100, 114, 200, 216, 300, 308, 400, 412, 406, 500, 512, 600, 615, 800, 110]
    ys = [100] * 8 + [112] + [100] * 5 + [110]
    con = [50, 20, 45, 40, 50, 30, 50, 45, 40, 50, 15, 90, 20, 15, 60]
    area = [150, 30, 150, 140, 160, 160, 338, 338, 338, 150, 0, 150, 150, 20, 150]
    comp = [1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 0, 8, 9, 10, 11]
    sup = [c != 0 for c in comp]
    nl = [{5: 2, 6: 3}.get(c, 1 if c else 0) for c in comp]
    cls = [0] * 14 + [1]
    L = pd.DataFrame({"frame_id": "f", "cls": cls, "state": cls, "motion": [2 * c for c in cls],
                      "supported": sup, "edge_suspect": False, "contrast": con, "comp_nlabels": nl,
                      "comp_area": area, "comp_x": [10 * c for c in comp], "comp_y": 90, "comp_w": 16,
                      "comp_h": 16, "x": xs, "y": ys, "x_raw": xs, "y_raw": ys})
    med = {0: (16.0, 16.0), 1: (20.0, 20.0)}
    out = BU.label_boxes(L, med, defs, 1000.0)
    dropped = "".join(n for n, s in zip(names, out.rest_sat) if s)
    _check(dropped == "bfijl",
           "resting satellites: leg blob, same blob 8 px apart, third label on a 2-mosquito blob, "
           "unsupported ghost and a much fainter neighbour dropped (%s)" % dropped)
    _check(bool(out.keep[[n in "acdeghkmnz" for n in names]].all()),
           "real neighbours, a touching pair, an isolated weak label and a flyer are kept")
    L2 = pd.concat([L, L.assign(frame_id="g", x=L.x + 1000, x_raw=L.x_raw + 1000)], ignore_index=True)
    out2 = BU.label_boxes(L2, med, defs, 1000.0)
    _check(list(out2.rest_sat[:15]) == list(out2.rest_sat[15:]) == list(out.rest_sat),
           "satellites are decided per frame")
    _check(not BU.label_boxes(L, med, defs, None).rest_sat.any(), "without cw no satellites are dropped")


def test_tiles():
    import pandas as pd
    defs = _defs()
    rng = np.random.default_rng(0)
    n = 40
    L = pd.DataFrame({"frame_id": "f", "x": rng.uniform(300, 1450, n), "y": rng.uniform(10, 1070, n),
                      "cls": rng.integers(0, 2, n), "state": 0, "motion": 0, "supported": True,
                      "contrast": 50, "edge_suspect": False, "comp_w": 15, "comp_h": 15, "comp_nlabels": 1})
    L = BU.label_boxes(L, {0: (16.0, 16.0), 1: (20.0, 20.0)}, defs)
    F = pd.DataFrame({"frame_id": ["f"], "stem": ["s"], "width": [1920], "height": [1080], "ref_kind": ["rest"]})
    tiles = BU.plan_tiles(F, L, None, defs, (266.0, 0.0, 1470.0, 1080.0))
    _check(len(tiles) == 4, "1080p cage ROI -> 2 x 2 tiles of 640")
    worst, seen, wrong_cls = 0.0, set(), 0
    for t in tiles:
        for ln in t["lines"].splitlines():
            c, cx, cy, w, h = ln.split()
            x, y = t["tx"] + float(cx) * 640, t["ty"] + float(cy) * 640
            ww, hh = float(w) * 640, float(h) * 640
            if ww < 15.99 or hh < 15.99:
                continue                          # clipped at a tile edge
            d = np.hypot(L.bx.values - x, L.by.values - y)
            j = int(np.argmin(d))
            worst = max(worst, float(d[j]))
            seen.add(j)
            wrong_cls += int(int(c) != int(L.cls.values[j]))
    _check(worst <= 0.5, "tile -> full-frame label round trip within 0.5 px (worst %.4f)" % worst)
    _check(wrong_cls == 0, "classes survive tiling")
    _check(len(seen) == n, "every label appears whole in at least one tile")


def test_splits_and_guard():
    row = {"role": "target", "test_experiment": False}
    a = [BU.split_of("stem_%05d" % i, row, [], "2026-01-01") for i in range(2000)]
    b = [BU.split_of("stem_%05d" % i, row, [], "2026-01-01") for i in range(2000)]
    _check(a == b, "splits are deterministic")
    frac = a.count("val") / 2000.0
    _check(0.07 < frac < 0.13, "about 10%% of segments go to val (%.1f%%)" % (100 * frac))
    _check(BU.split_of("x", row, ["2026-01-01"], "2026-01-01") == "test", "held-out day -> test")
    _check(BU.split_of("x", {"role": "aux"}, [], "d") == "train", "auxiliary rigs -> train only")
    for bad in (r"G:\Buzzwatch\Analysis\x", r"G:\Buzzwatch", "/Volumes/Mosquito2/Buzzwatch/Recording/y"):
        try:
            C.check_output_dir(bad, [])
            raise AssertionError("guard accepted %s" % bad)
        except ValueError:
            pass
    _check(True, "output guard refuses Buzzwatch data trees")
    C.check_output_dir(os.path.join(os.path.expanduser("~"), "BuzzWatch_LiveDetector_test"), [])
    _check(True, "output guard accepts a folder outside the data trees")


def check_dataset(out_dir):
    """YOLO format checks of a written dataset: pairs, 5 normalised columns, classes."""
    for variant in ("stack", "single"):
        root = os.path.join(out_dir, "yolo", variant)
        if not os.path.isdir(root):
            continue
        _check(os.path.isfile(os.path.join(root, "data.yaml")), "%s: data.yaml present" % variant)
        n_img = n_obj = 0
        for sp in ("train", "val", "test"):
            imgs = sorted(f[:-4] for f in os.listdir(os.path.join(root, "images", sp)))
            labs = sorted(f[:-4] for f in os.listdir(os.path.join(root, "labels", sp)))
            _check(imgs == labs, "%s/%s: every image has a label file (%d)" % (variant, sp, len(imgs)))
            for name in labs:
                with open(os.path.join(root, "labels", sp, name + ".txt")) as f:
                    for ln in f:
                        v = ln.split()
                        if len(v) != 5 or v[0] not in ("0", "1") or not all(0.0 <= float(t) <= 1.0 for t in v[1:]):
                            raise AssertionError("bad label line in %s/%s: %r" % (sp, name, ln))
                        n_obj += 1
            n_img += len(imgs)
        _check(n_img > 0, "%s: %d tiles, %d objects, all lines valid" % (variant, n_img, n_obj))
    import cv2
    st = os.path.join(out_dir, "yolo", "stack", "images", "test")
    if os.path.isdir(st) and os.listdir(st):
        im = cv2.imread(os.path.join(st, sorted(os.listdir(st))[0]), cv2.IMREAD_UNCHANGED)
        _check(im.shape == (640, 640, 3), "stack tiles are 640 x 640 x 3")


def main(argv):
    print("yolo_export selftest")
    test_alignment()
    test_shared_blob()
    test_flying_motion()
    test_resting_satellites()
    test_tiles()
    test_splits_and_guard()
    if len(argv) > 1:
        check_dataset(argv[1])
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
