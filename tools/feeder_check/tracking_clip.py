#!/usr/bin/env python3
"""Labelled tracking clip: every tracker detection coloured by what BuzzSwarm does with it.

  blue    kept as real flight (what the BuzzSwarm heatmap plots)
  orange  removed: too slow (smoothed speed below the experiment's real-flight
          threshold: 3 px/frame = 75 px/s at 25 fps at the default 100 px swarm radius, scaled with it)
  purple  removed: short or jittery piece (step split / < FLIGHT_MIN_FRAMES / average-speed bounds)
  grey    resting tracker (state 0)
Zones from custom_zones.json are outlined; the corner box shows the timestamp and per-frame counts.
The density-artifact filter is not applied (it drops well under 1% of points).

Usage (buzzwatch_env):
    python tools/feeder_check/tracking_clip.py <experiment_dir> <video_dir> <segment_name>
        --start 300 --dur 60 -o clip.mp4 [--still frame.png]
<segment_name> is the video basename without .mp4, e.g. CageX_Light_test_20260704_120009_00051.
The clip is re-encoded to H.264 with ffmpeg when it is found (repo h264_converter/ffmpeg or PATH).
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

REAL, SLOW, PIECE, REST = 0, 1, 2, 3
COL = {REAL: (232, 160, 40), SLOW: (40, 140, 245), PIECE: (200, 60, 230), REST: (170, 170, 170)}  # BGR
NAME = {REAL: "kept: real flight (heatmap)", SLOW: "removed: too slow",
        PIECE: "removed: short/jittery piece", REST: "resting tracker"}


def classify(mt, f0, f1, offset, fly_px=None):
    """{video frame: [(x, y, category, track key)]} for frames [f0 - 15, f1)."""
    m, ca = c.m, c.ca
    fly_px = ca.FLY_PX_FRAME if fly_px is None else fly_px
    ts = mt.time_stamp
    per_frame = {}

    def add(fr, x, y, cat, key):
        if f0 - 15 <= fr < f1:
            per_frame.setdefault(fr, []).append((x, y, cat, key))

    for oid, obj in mt.objects.items():
        coords = obj.get("coordinates", []) or []
        state = obj.get("state", []) or []
        s = int(obj.get("start", 0))
        n = min(len(coords), len(state))
        if n == 0 or s + n + offset < f0 - 15 or s + offset > f1:
            continue
        i = 0
        while i < n:
            if state[i] != 1:
                if m.valid_point(coords[i]):
                    add(s + i + offset, coords[i][0], coords[i][1], REST, None)
                i += 1
                continue
            j = i
            while j < n and state[j] == 1:
                j += 1
            # as extract_raw_flights: invalid points and jumps > MAX_DISPLACEMENT_PX split the run
            runs, cur = [], []
            for k in range(i, j):
                pt = coords[k]
                if not m.valid_point(pt) or not (0 <= s + k < len(ts)):
                    if len(cur) >= 2:
                        runs.append(cur)
                    cur = []
                    continue
                prev = coords[cur[-1]] if cur else None
                if prev is not None and math.hypot(pt[0] - prev[0], pt[1] - prev[1]) > m.MAX_DISPLACEMENT_PX:
                    if len(cur) >= 2:
                        runs.append(cur)
                    cur = []
                cur.append(k)
            if len(cur) >= 2:
                runs.append(cur)
            claimed = set()
            for run in runs:
                cr = [coords[k] for k in run]
                if m.USE_STEP_SPLIT:
                    pieces = m._zip_split_step(cr, run, run, m.STEP_SPLIT_MIN, m.STEP_SPLIT_MAX)
                else:
                    pieces = [(cr, run, run)]
                for pc, pk, _z in pieces:
                    key = (oid, pk[0])
                    ok = (len(pc) >= ca.FLIGHT_MIN_FRAMES and
                          m.SPEED_FILTER_MIN <= m.average_step_speed(pc) <= m.SPEED_FILTER_MAX)
                    arr = np.asarray(pc, float)
                    for q, k in enumerate(pk):
                        claimed.add(k)
                        if not ok:
                            cat = PIECE
                        else:
                            sp = 0.0
                            if 2 <= q < len(pk) - 2:  # same smoothing as clock_analysis.smoothed_motion
                                sp = math.hypot(arr[q + 2, 0] - arr[q - 2, 0], arr[q + 2, 1] - arr[q - 2, 1]) / 4.0
                            cat = REAL if sp >= fly_px else SLOW
                        add(s + k + offset, arr[q, 0], arr[q, 1], cat, key)
            for k in range(i, j):
                if k not in claimed and m.valid_point(coords[k]):
                    add(s + k + offset, coords[k][0], coords[k][1], PIECE, None)
            i = j
    return per_frame


def find_ffmpeg():
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    sub = "win" if os.name == "nt" else "mac"
    local = os.path.join(c.REPO, "h264_converter", "ffmpeg", "bin", sub, exe)
    return local if os.path.isfile(local) else shutil.which("ffmpeg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("video_dir")
    ap.add_argument("segment")
    ap.add_argument("--start", type=float, default=0, help="seconds into the segment")
    ap.add_argument("--dur", type=float, default=60, help="clip length in seconds")
    ap.add_argument("--offset", type=int, default=c.FRAME_OFFSET)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--still", default=None, help="also save the frame 10 s into the clip as a PNG")
    a = ap.parse_args()

    import cv2
    from matplotlib.path import Path

    exp = a.experiment
    geo, zones = c.load_geometry(exp)
    mt = c.m.load_tracking_file(c.segment_path(exp, a.segment))
    ts = mt.time_stamp
    cap = cv2.VideoCapture(os.path.join(a.video_dir, a.segment + ".mp4"))
    if not cap.isOpened():
        sys.exit("cannot open %s.mp4 in %s" % (a.segment, a.video_dir))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video %.2f fps, %d frames, %dx%d | %d tracking timestamps" % (fps, nvid, W, H, len(ts)))
    f0 = int(a.start * fps)
    f1 = min(nvid, f0 + int(a.dur * fps))
    fly_px = c.fly_px_frame(exp)
    NAME[SLOW] = "removed: too slow (<%.1f px/frame)" % fly_px
    per_frame = classify(mt, f0, f1, a.offset, fly_px)

    zp = {k: np.asarray(v, np.int32) for k, v in zones.items()}
    zpaths = [Path(v) for v in zones.values()]
    tmp = a.out + ".tmp.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    trail = {}
    tot = np.zeros(4, int)
    tot_zone = np.zeros(4, int)
    for fr in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        for k, p in zp.items():
            cv2.polylines(frame, [p], True, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, k, (int(p[:, 0].min()), int(p[:, 1].min()) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 255), 1, cv2.LINE_AA)
        cnt = np.zeros(4, int)
        pts = per_frame.get(fr, [])
        for x, y, cat, key in pts:
            if key is not None:
                trail.setdefault(key, []).append((fr, int(x), int(y), cat))
        for key in list(trail):
            tr = [t for t in trail[key] if fr - t[0] < 12]
            if not tr:
                del trail[key]
                continue
            trail[key] = tr
            for (_fa, xa, ya, _ca), (_fb, xb, yb, cb) in zip(tr[:-1], tr[1:]):
                cv2.line(frame, (xa, ya), (xb, yb), COL[cb], 1, cv2.LINE_AA)
        for x, y, cat, _key in pts:
            cnt[cat] += 1
            cv2.circle(frame, (int(x), int(y)), 3 if cat == REST else 6, COL[cat], 1 if cat == REST else 2, cv2.LINE_AA)
            if cat != REST:
                tot[cat] += 1
                if any(p.contains_point((x, y)) for p in zpaths):
                    tot_zone[cat] += 1
        cv2.rectangle(frame, (8, 8), (330, 128), (25, 25, 25), -1)
        t = ts[min(len(ts) - 1, max(0, fr))]
        cv2.putText(frame, "%s  frame %d" % (t.strftime("%Y-%m-%d %H:%M:%S"), fr), (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for r, cat in enumerate([REAL, SLOW, PIECE, REST]):
            y = 50 + 20 * r
            cv2.circle(frame, (24, y - 4), 6, COL[cat], 2, cv2.LINE_AA)
            cv2.putText(frame, "%s: %d" % (NAME[cat], cnt[cat]), (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        wr.write(frame)
        if a.still and fr == f0 + int(fps * 10):
            cv2.imwrite(a.still, frame)
    wr.release()
    cap.release()
    names = ["kept real flight", "removed slow", "removed piece"]
    print("moving detections in clip:", {names[i]: int(tot[i]) for i in range(3)})
    print("  inside the zones:      ", {names[i]: int(tot_zone[i]) for i in range(3)})
    ff = find_ffmpeg()
    done = False
    if ff:
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", "-preset", "fast", a.out])
        done = r.returncode == 0
    if done:
        os.remove(tmp)
    else:
        os.replace(tmp, a.out)
        print("H.264 re-encode unavailable (no ffmpeg or no libx264): kept the MPEG-4 (mp4v) encoding")
    print("saved", a.out)


if __name__ == "__main__":
    main()
