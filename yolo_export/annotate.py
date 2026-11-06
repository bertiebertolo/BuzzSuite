"""annotate: the "every video with data attached" layer -- no video decoding.

For every tracked segment of the chosen experiments, writes under
annotations/<rig>/<alias>/:
    <stem>.parquet           one row per aligned detection (video frame, track, x, y, state, motion,
                             YOLO class, real flight, smoothed speed, zone, tracker index)
    <stem>.events.parquet    take-off, landing, zone pass / arrival / landing, walking bouts, swarm
    <stem>.activity.npz      per-frame counts (resting/walking/flying/hold, real flyers, swarm),
                             flight pixels (clock fit) and resting per zone
    <stem>.json              written last: source .mp4 / .pkl, frame count, geometry, alignment
The per-experiment events/ and timeseries/ tables (with corrected clock time) come from clock.py.
"""
import gc
import os

import numpy as np

from yolo_export import common as C
from yolo_export import tracks as TR


def seg_dir(out_dir, rig, alias):
    return os.path.join(out_dir, "annotations", rig, alias)


def job_geometry(row):
    """Geometry dict passed to workers (plain lists so it pickles cheaply)."""
    geo = C.load_geometry(row["exp_dir"], zones_dir=row.get("zones_dir") or None)
    return {"cw": geo["width"], "centroid": geo["centroid"].tolist(), "cage": geo["cage"].tolist(),
            "bbox": list(geo["bbox"]), "zones": {k: v.tolist() for k, v in geo["zones"].items()},
            "frame_w": int(row["width"]), "frame_h": int(row["height"])}


def load_points(job, defs, want_events=True):
    """Load one segment and derive everything: (raw, P, activity, events_df or None, T, info)."""
    pkl = os.path.join(job["exp_dir"], "final_tracking_data", C.TRACK_PREFIX + job["stem"])
    mt = C.load_tracking(pkl)
    if mt is None or not getattr(mt, "time_stamp", None):
        raise RuntimeError("tracking file unreadable or empty: %s" % pkl)
    T = len(mt.time_stamp)
    ts0 = mt.time_stamp[0]
    raw = TR.raw_points(mt)
    del mt
    gc.collect()
    g = job["geo"]
    zones = {k: np.asarray(v, float) for k, v in g["zones"].items()}
    zm = C.zone_maps(zones, g["frame_w"], g["frame_h"]) if zones else None
    P = TR.aligned_points(raw, defs, g["cw"], np.asarray(g["centroid"]), zm)
    act = TR.activity(P, T, defs, g["cw"], len(zones))
    ev = TR.events(raw, P, act, defs, g["cw"], zm) if want_events else None
    info = {"n_frames": T, "tracker_t0": ts0, "n_tracks": raw["n_tracks"], "n_bad_tracks": raw["n_bad"],
            "n_raw_points": int(len(raw["tid"])), "n_points": int(len(P["tid"])),
            "zone_names": list(zones), "pkl": pkl, "pkl_mtime": os.path.getmtime(pkl)}
    return raw, P, act, ev, T, info


def annotate_segment(job):
    """Worker. Returns (stem, status, message)."""
    import pandas as pd
    out = seg_dir(job["out_dir"], job["rig"], job["alias"])
    base = os.path.join(out, job["stem"])
    if os.path.isfile(base + ".json") and not job.get("force"):
        return job["stem"], "skip", ""
    try:
        defs = C.definitions()
        raw, P, act, ev, T, info = load_points(job, defs)
        df = pd.DataFrame({
            "video_frame": P["vf"].astype(np.int32), "track_id": P["tid"].astype(np.int32),
            "x": P["x"].astype(np.float32), "y": P["y"].astype(np.float32),
            "state": P["state"].astype(np.int8), "motion": P["motion"], "cls": P["cls"],
            "real_flight": P["real"], "speed_px": P["speed"].astype(np.float32),
            "zone": P["zone"], "tracker_index": P["idx"].astype(np.int32)})
        C.write_table(df, base + ".parquet")
        C.write_table(ev, base + ".events.parquet")
        np.savez_compressed(base + ".activity.npz", **act)
        sess, k = C.session_key(job["stem"])
        g = job["geo"]
        C.write_json(base + ".json", dict(info, **{
            "generator_version": C.GENERATOR_VERSION, "stem": job["stem"], "alias": job["alias"],
            "rig": job["rig"], "session": sess, "k": k,
            "video": os.path.join(job["video_dir"], job["stem"] + ".mp4"),
            "fps_nominal": C.NOMINAL_FPS, "width": g["frame_w"], "height": g["frame_h"],
            "cage_width_px": g["cw"], "cage_centroid": g["centroid"], "cage": g["cage"],
            "swarm_radius_px": defs["behaviour"]["swarm_radius_cw"] * g["cw"],
            "zones_source": job.get("zones_source", ""),
            "alignment": defs["alignment"]["rule"],
            "columns": {
                "video_frame": "frame of the source .mp4 (0-based), alignment already applied",
                "track_id": "BuzzWatch assembled track id (one per individual within the segment)",
                "x, y": "pixel position (tracker blob centre)",
                "state": "tracker state: 1 flying/moving, 0 resting",
                "motion": "0 resting, 1 walking, 2 flying, 3 hold (see docs/definitions.json)",
                "cls": "YOLO class: 1 flying (motion 2), 0 resting",
                "real_flight": "state 1 at >= real_flight_speed_cw_per_frame",
                "speed_px": "smoothed speed |p(t+2)-p(t-2)|/4 in px/frame (+-1 fallback)",
                "zone": "index into zone_names of the custom zone containing the point, -1 none",
                "tracker_index": "index into the tracker's own frame axis (before alignment)"}}))
        del raw, P, act, ev, df
        gc.collect()
        return job["stem"], "ok", ""
    except Exception as exc:  # noqa: BLE001
        return job["stem"], "error", "%s: %s" % (type(exc).__name__, exc)


def segment_infos(out_dir, rig, alias, stems=None):
    """[(stem, info dict)] of annotated segments, sorted by stem; only `stems` when given (the
    experiment's current segments), so files left from an earlier selection are ignored."""
    d = seg_dir(out_dir, rig, alias)
    if not os.path.isdir(d):
        return []
    out = []
    for n in sorted(os.listdir(d)):
        if n.endswith(".json") and (stems is None or n[:-5] in stems):
            info = C.read_json(os.path.join(d, n))
            if info:
                if info.get("session") is None or info.get("k") is None:
                    # written before '-<attempt>' stems were parsed: take them from the name now
                    info["session"], info["k"] = C.session_key(n[:-5])
                if info["session"] is not None:  # an unparseable name has no time: leave it out
                    out.append((n[:-5], info))
    return out
