"""clips: short annotated video clips -- portable "videos with data attached".

Per experiment, ~n 10-s windows stratified by stimulus ON/OFF x light phase x activity tercile
(at most 2 per segment, never within clock edge_s of a stimulus window edge). Each clip is cut
frame-accurately from the original .mp4 with the bundled ffmpeg (libx264 crf 16) and written as

    clips/<rig>/<alias>/<stem>_<start>.mp4
    ...gt.txt        MOTChallenge: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility
                     (frame 1-based within the clip; class 1 resting, 2 flying)
    ...labels.csv    per detection: clip_frame, video_frame, track_id, x, y, state, motion, cls, ...
    ...events.csv    the segment's events inside the clip (take-off, landing, zone events, swarm)
    ...json          source video, start frame, corrected time, stimulus, light phase, boxes
"""
import os
import subprocess

import numpy as np

from yolo_export import annotate as AN
from yolo_export import clock as CK
from yolo_export import common as C
from yolo_export import inventory as I

CLIP_FRAMES = 250


def _box_sizes(out_dir, alias, cw, defs):
    rep = C.read_json(os.path.join(out_dir, "meta", "build_report.json")) or {}
    med = ((rep.get("experiments") or {}).get(alias) or {}).get("median_box_px")
    if med:
        return {0: tuple(med["resting"]), 1: tuple(med["flying"])}
    d = defs["boxes"]["default_box_cw"]
    return {0: (d["resting"] * cw,) * 2, 1: (d["flying"] * cw,) * 2}


def plan_clips(row, out_dir, clock, n_clips, defs):
    """Chosen clip windows: list of dict(stem, start, act, stim_on, light_phase, day_tag, time)."""
    import pandas as pd
    rig, alias = row["rig_id"], row["alias"]
    windows = C.parse_windows(row["stim_windows"])
    sched = CK.schedule_of(row)
    seg = AN.seg_dir(out_dir, rig, alias)
    rows = []
    for stem, info in AN.segment_infos(out_dir, rig, alias, I.segment_set(row)):
        T = int(info["n_frames"])
        starts = np.arange(25, T - CLIP_FRAMES - 25, CLIP_FRAMES)
        if not len(starts):
            continue
        z = np.load(os.path.join(seg, stem + ".activity.npz"))
        fly = z["n_fly"].astype(float)
        act = np.array([fly[s:s + CLIP_FRAMES].mean() for s in starts])
        mid = CK.frame_times(clock, info["session"], info["k"], starts + CLIP_FRAMES // 2)
        on, edge, phase, tag = CK.flags(mid, windows, sched, defs)
        for s, a, o, e, p, d, t in zip(starts, act, on, edge, phase, tag, mid):
            if not e:
                rows.append({"stem": stem, "start": int(s), "act": float(a), "stim_on": bool(o),
                             "light_phase": str(p), "day_tag": str(d), "time": t})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["tercile"] = pd.qcut(df.act.rank(method="first"), 3, labels=False)
    keys = ["stim_on", "light_phase", "tercile"] if windows else ["light_phase", "tercile"]
    rng = np.random.default_rng(C.seed_for(alias))
    buckets = [g.sample(frac=1.0, random_state=int(rng.integers(1 << 30))) for _, g in df.groupby(keys)]
    chosen, per_seg, i = [], {}, 0
    while len(chosen) < n_clips and any(i < len(b) for b in buckets):
        for b in buckets:
            if i < len(b) and len(chosen) < n_clips:
                r = b.iloc[i]
                if per_seg.get(r.stem, 0) < 2:
                    chosen.append(r.to_dict())
                    per_seg[r.stem] = per_seg.get(r.stem, 0) + 1
        i += 1
    return chosen


def cut_clip(job):
    """Worker: one clip (video + MOT gt + labels + events + json). Returns (name, status, message)."""
    import pandas as pd
    from h264_converter.convert_gui import find_tool
    name = "%s_%05d" % (job["stem"], job["start"])
    base = os.path.join(job["dir"], name)
    if os.path.isfile(base + ".json") and not job.get("force"):
        return name, "skip", ""
    try:
        os.makedirs(job["dir"], exist_ok=True)
        s0, n = int(job["start"]), CLIP_FRAMES
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
               "-ss", "%.3f" % (s0 / C.NOMINAL_FPS), "-i", job["video"], "-frames:v", str(n), "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", base + ".mp4"]
        flags = 0x08000000 if os.name == "nt" else 0
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
        if r.returncode != 0:
            return name, "error", "ffmpeg: %s" % r.stderr.strip()[-300:]
        A = C.read_table(job["annotations"])
        A = A[(A.video_frame >= s0) & (A.video_frame < s0 + n)].copy()
        A.insert(0, "clip_frame", A.video_frame - s0 + 1)
        box = job["boxes"]
        bw = np.where(A.cls.values == 1, box[1][0], box[0][0])
        bh = np.where(A.cls.values == 1, box[1][1], box[0][1])
        gt = pd.DataFrame({"frame": A.clip_frame.values, "id": A.track_id.values,
                           "bb_left": np.round(A.x.values - bw / 2, 2), "bb_top": np.round(A.y.values - bh / 2, 2),
                           "bb_width": np.round(bw, 2), "bb_height": np.round(bh, 2), "conf": 1,
                           "class": A.cls.values.astype(int) + 1, "visibility": 1})
        gt.sort_values(["frame", "id"]).to_csv(base + ".gt.txt", index=False, header=False)
        A.drop(columns=["tracker_index"]).to_csv(base + ".labels.csv", index=False)
        ev = C.read_table(job["events"])
        if ev is not None and len(ev):
            ev = ev[(ev.video_frame >= s0) & (ev.video_frame < s0 + n)].copy()
            ev.insert(0, "clip_frame", ev.video_frame - s0 + 1)
            ev.to_csv(base + ".events.csv", index=False)
        else:
            pd.DataFrame(columns=["clip_frame"]).to_csv(base + ".events.csv", index=False)
        C.write_json(base + ".json", {
            "generator_version": C.GENERATOR_VERSION, "alias": job["alias"], "rig": job["rig"],
            "stem": job["stem"], "source_video": job["video"], "start_frame": s0, "n_frames": n,
            "fps": C.NOMINAL_FPS, "start_time_corrected": str(job["time_start"]),
            "stim_on": job["stim_on"], "light_phase": job["light_phase"], "day_tag": job["day_tag"],
            "mean_flyers": job["act"], "box_px": {"resting": list(box[0]), "flying": list(box[1])},
            "zone_names": job["zone_names"],
            "mot_columns": "frame,id,bb_left,bb_top,bb_width,bb_height,conf,class(1 resting, 2 flying),visibility",
            "note": "clip frame f (1-based) = source video frame start_frame + f - 1; positions are the "
                    "aligned tracker points (docs/definitions.json, alignment); boxes are the "
                    "experiment's class-median sizes"})
        return name, "ok", ""
    except Exception as exc:  # noqa: BLE001
        return name, "error", "%s: %s" % (type(exc).__name__, exc)


def clip_jobs(row, out_dir, n_clips, force=False):
    defs = C.definitions()
    clock = CK.load_clock(out_dir, row["alias"])
    if not clock:
        raise RuntimeError("no clock for %s (run clock first)" % row["alias"])
    rig, alias = row["rig_id"], row["alias"]
    seg = AN.seg_dir(out_dir, rig, alias)
    infos = dict(AN.segment_infos(out_dir, rig, alias, I.segment_set(row)))
    any_info = next(iter(infos.values()))
    boxes = _box_sizes(out_dir, alias, float(any_info["cage_width_px"]), defs)
    jobs = []
    for c in plan_clips(row, out_dir, clock, n_clips, defs):
        info = infos[c["stem"]]
        t0 = CK.frame_times(clock, info["session"], info["k"], [c["start"]])[0]
        jobs.append({"alias": alias, "rig": rig, "stem": c["stem"], "start": c["start"],
                     "dir": os.path.join(out_dir, "clips", rig, alias), "video": info["video"],
                     "annotations": os.path.join(seg, c["stem"] + ".parquet"),
                     "events": os.path.join(seg, c["stem"] + ".events.parquet"),
                     "boxes": boxes, "time_start": t0, "stim_on": c["stim_on"],
                     "light_phase": c["light_phase"], "day_tag": c["day_tag"], "act": c["act"],
                     "zone_names": info.get("zone_names") or [], "force": force})
    return jobs
