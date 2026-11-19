#!/usr/bin/env python3
"""Does the moving tracker miss real flight in the zones? An independent check on raw video.

1. Background = median of every 10th frame of the clip; any spot darker than the background by
   >= --thresh grey levels (4-300 px) is a candidate mosquito.
2. Compares candidate spots with the tracker's flying detections (within 15 px, same frame), per
   area: centre, each zone, and other walls at the same distance from the centre. Also prints the
   background brightness and how much tracked flyers darken it there (contrast).
3. Takes a random sample of the zone spots the moving tracker did NOT record and sorts them:
   stationary (still dark 6 frames before and after) or moving, and whether the resting tracker
   recorded them.
4. Saves two PNG strips (5 zoomed frames, -6..+6, per spot) to look at by eye: a random sample of
   untracked zone spots, and the moving ones neither tracker recorded.

Usage (buzzwatch_env):
    python tools/feeder_check/untracked_spots.py <experiment_dir> <video_dir> <segment_name>
        --start 300 --dur 60 --outdir <folder> [--tag dawn] [--thresh 8] [--sample 400]
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

MATCH_PX = 15
OFFS = (-6, -3, 0, 3, 6)


def tracker_points(mt, f0, n, state, offset):
    out = {}
    for o in mt.objects.values():
        s = int(o.get("start", 0))
        cc = o.get("coordinates", []) or []
        st = o.get("state", []) or []
        for k in range(min(len(cc), len(st))):
            f = s + k + offset - f0
            if st[k] == state and 0 <= f < n and c.m.valid_point(cc[k]):
                out.setdefault(f, []).append(cc[k])
    return out


def strip(cap, f_abs, x, y, label, C=40, Z=3):
    import cv2
    tiles = []
    for o in OFFS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_abs + o)
        ok, im = cap.read()
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        x0, y0 = max(0, x - C), max(0, y - C)
        crop = g[y0:y0 + 2 * C, x0:x0 + 2 * C]
        crop = cv2.copyMakeBorder(crop, 0, 2 * C - crop.shape[0], 0, 2 * C - crop.shape[1],
                                  cv2.BORDER_CONSTANT, value=0)
        lo, hi = np.percentile(crop, 1), np.percentile(crop, 99.5)  # local stretch: faint flyers visible
        crop = np.clip((crop.astype(np.float32) - lo) / max(1.0, hi - lo) * 255, 0, 255).astype(np.uint8)
        t = cv2.cvtColor(cv2.resize(crop, (2 * C * Z, 2 * C * Z), interpolation=cv2.INTER_NEAREST),
                         cv2.COLOR_GRAY2BGR)
        if o == 0:
            cv2.rectangle(t, ((x - x0 - 8) * Z, (y - y0 - 8) * Z), ((x - x0 + 8) * Z, (y - y0 + 8) * Z),
                          (0, 200, 255), 1)
        cv2.putText(t, "%+d" % o, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles += [t, np.full((2 * C * Z, 4, 3), 255, np.uint8)]
    row = np.hstack(tiles)
    lab = np.zeros((18, row.shape[1], 3), np.uint8)
    cv2.putText(lab, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return [lab, row]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("video_dir")
    ap.add_argument("segment")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=60)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="clip")
    ap.add_argument("--thresh", type=float, default=8, help="grey levels darker than background")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--ring", type=float, nargs=2, default=c.DEFAULT_RING)
    ap.add_argument("--offset", type=int, default=c.FRAME_OFFSET)
    a = ap.parse_args()

    import cv2
    from matplotlib.path import Path

    exp = a.experiment
    geo, zones = c.load_geometry(exp)
    os.makedirs(a.outdir, exist_ok=True)
    mt = c.m.load_tracking_file(c.segment_path(exp, a.segment))
    cap = cv2.VideoCapture(os.path.join(a.video_dir, a.segment + ".mp4"))
    if not cap.isOpened():
        sys.exit("cannot open %s.mp4 in %s" % (a.segment, a.video_dir))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0, n = int(a.start * fps), int(a.dur * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fr = np.empty((n, H, W), np.uint8)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            n = i
            fr = fr[:n]
            break
        fr[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    bg = np.median(fr[::10], axis=0).astype(np.float32)

    yy, xx = np.mgrid[0:H, 0:W]
    P = np.c_[xx.ravel(), yy.ravel()]
    inside = Path(geo["polygon"]).contains_points(P).reshape(H, W)
    zm = {k: Path(v).contains_points(P).reshape(H, W) for k, v in zones.items()}
    anyz = np.any(list(zm.values()), 0)
    cx, cy = geo["centroid"]
    r = np.hypot(xx - cx, yy - cy)
    areas = {"centre": inside & (r < 150)}
    areas.update(zm)
    areas["same-radius walls"] = inside & ~anyz & (r > a.ring[0]) & (r < a.ring[1])

    fly = tracker_points(mt, f0, n, 1, a.offset)
    rest = tracker_points(mt, f0, n, 0, a.offset)

    print("== %s: %s, %d frames from %.0f s" % (a.tag, a.segment, n, a.start))
    darken = {k: [] for k in areas}
    tk = {k: 0 for k in areas}
    for f, pts in fly.items():
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if not (3 <= xi < W - 3 and 3 <= yi < H - 3):
                continue
            d = float((fr[f, yi - 3:yi + 4, xi - 3:xi + 4].astype(np.float32) - bg[yi - 3:yi + 4, xi - 3:xi + 4]).min())
            for k, msk in areas.items():
                if msk[yi, xi]:
                    darken[k].append(d)
                    tk[k] += 1
    print("  %-18s %10s %18s" % ("area", "background", "flyer darkening"))
    for k, msk in areas.items():
        print("  %-18s %10.0f %18.1f  (n=%d)" % (k, np.median(bg[msk]), np.median(darken[k]) if darken[k] else np.nan,
                                                  len(darken[k])))

    cnt = {k: 0 for k in areas}
    unmatched = {k: 0 for k in areas}
    samples = []
    for f in range(n):
        d = cv2.GaussianBlur(np.clip(bg - fr[f].astype(np.float32), 0, None), (3, 3), 0)
        nl, _lab, stats, cent = cv2.connectedComponentsWithStats((d >= a.thresh).astype(np.uint8), 8)
        tp = np.asarray(fly.get(f, []), float).reshape(-1, 2)
        for j in range(1, nl):
            if not (4 <= stats[j, cv2.CC_STAT_AREA] <= 300):
                continue
            x, y = cent[j]
            xi, yi = int(x), int(y)
            near = len(tp) and np.min(np.hypot(tp[:, 0] - x, tp[:, 1] - y)) < MATCH_PX
            for k, msk in areas.items():
                if msk[yi, xi]:
                    cnt[k] += 1
                    if not near:
                        unmatched[k] += 1
                        if k in zones:
                            samples.append((f, xi, yi, k))
    mins = n / fps / 60.0
    print("  per 1000 px^2 per minute: %-10s %-12s %s" % ("tracker", "dark spots", "spots with no flying detection"))
    for k, msk in areas.items():
        A = msk.sum() / 1000.0 * mins
        print("  %-18s %8.1f %12.1f %10.1f%%" % (k, tk[k] / A, cnt[k] / A, 100.0 * unmatched[k] / max(1, cnt[k])))

    def dark_at(f, x, y):
        if not (0 <= f < n) or not (2 <= x < W - 2 and 2 <= y < H - 2):
            return 0.0
        return float((fr[f, y - 2:y + 3, x - 2:x + 3].astype(np.float32) - bg[y - 2:y + 3, x - 2:x + 3]).min())

    random.seed(0)
    samp = random.sample(samples, min(a.sample, len(samples)))
    kinds = {}
    moving = []
    for f, x, y, k in samp:
        still = dark_at(f - 6, x, y) <= -a.thresh and dark_at(f + 6, x, y) <= -a.thresh
        rp = np.asarray(rest.get(f, []), float).reshape(-1, 2)
        rmatch = len(rp) and np.min(np.hypot(rp[:, 0] - x, rp[:, 1] - y)) < MATCH_PX
        key = (k, "stationary" if still else "moving", "resting tracker" if rmatch else "neither tracker")
        kinds[key] = kinds.get(key, 0) + 1
        if not still and not rmatch:
            moving.append((f, x, y, k))
    print("  random %d of %d zone spots without a flying detection:" % (len(samp), len(samples)))
    for key in sorted(kinds):
        print("    %-10s %-10s %-16s %d" % (key + (kinds[key],)))

    for name, pool in [("untracked_%s.png" % a.tag, samples), ("untracked_%s_moving.png" % a.tag, moving)]:
        picked = []
        for z in zones:
            cand = sorted(s for s in pool if s[3] == z)
            spread, last = [], -999
            for s in cand:
                if s[0] - last >= int(2 * fps):
                    spread.append(s)
                    last = s[0]
            random.shuffle(spread)
            picked += spread[:6]
        if not picked:
            continue
        rows = []
        for f, x, y, k in picked:
            rows += strip(cap, f0 + f, x, y, "%s  frame %d  (x=%d, y=%d)  centre frame boxed" % (k, f0 + f, x, y))
        out = os.path.join(a.outdir, name)
        cv2.imwrite(out, np.vstack(rows))
        print("  saved", out)
    cap.release()


if __name__ == "__main__":
    main()
