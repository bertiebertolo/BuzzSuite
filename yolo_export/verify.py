"""verify: pixel check of take-offs and landings (DEFINITIONS["events"]["verified"]).

For every take_off, landing, zone_landing and arrival of a segment, read the frames at
behaviour.px_frames (-25, -8, +8, +25) around it and measure the edge-robust darkness (frame darker
than the 3x3-min-filtered resting reference) within px_window_cw of the site. The occupied side must
be dark (>= px_min_contrast) and clearly darker than the empty side (>= px_min_change):

    take-off  occupied = frame -8              empty = max(+8, +25)
    landing   occupied = min(+8, +25)          empty = -25  (zone_landing / arrival too; mosquitoes
                                                hover near the site before touching down)

Only events that passed the tracker-point check (site_ok) are read; the others get px_ok False
and NaN pixel values. Writes px_m25 / px_m8 / px_p8 / px_p25 / px_ok back into
annotations/.../<stem>.events.parquet and sets verified = site_ok and px_ok. A segment is read once, front to back (frames are grabbed, only
the needed ones decoded), ~20-60 s per 1080p segment.
"""
import os

import numpy as np

from yolo_export import annotate as AN
from yolo_export import common as C
from yolo_export import frames as FR

KINDS = ("take_off", "landing", "zone_landing", "arrival")


def verify_segment(job):
    """Worker. Returns (stem, status, message)."""
    import cv2
    stem = job["stem"]
    d = AN.seg_dir(job["out_dir"], job["rig"], job["alias"])
    ep = os.path.join(d, stem + ".events.parquet")
    info = C.read_json(os.path.join(d, stem + ".json"))
    if info is None:
        return stem, "skip", "not annotated"
    try:
        ev = C.read_table(ep)
        if ev is None:
            return stem, "skip", "no events table"
        if "px_ok" in ev.columns and not job.get("force"):
            return stem, "skip", ""
        defs = C.definitions()
        b, q = defs["behaviour"], defs["qa"]
        offs = [int(o) for o in b["px_frames"]]
        half = max(2, int(round(b["px_window_cw"] * float(info["cage_width_px"]))))
        T = int(info["n_frames"])
        site_ok = ev["site_ok"].values.astype(bool) if "site_ok" in ev.columns else np.ones(len(ev), bool)
        # pixels are read only where the tracker-point check passed: ~1% of the events it rejects
        # pass the pixel check, and skipping them saves about a third of the decoding
        chk = ev.event.isin(KINDS).values & site_ok
        need = {}
        for i in np.nonzero(chk)[0]:
            f = int(ev.video_frame.values[i])
            for j, o in enumerate(offs):
                if 0 <= f + o < T:
                    need.setdefault(f + o, []).append((i, j))
        vals = np.full((len(ev), len(offs)), np.nan)
        ref, _thr, _kind = FR.load_reference(job["exp_dir"], stem, {}, defs)
        if ref is not None and need:
            er = cv2.erode(ref, np.ones((int(q["robust_erode_px"]),) * 2, np.uint8))
            xs, ys = ev.x.values, ev.y.values
            reader = FR.Reader(info["video"])
            try:
                for v in sorted(need):
                    g = reader.read_gray(v, 1)
                    if g is None:
                        continue
                    rob = cv2.subtract(er, g[0])
                    for i, j in need[v]:
                        vals[i, j] = FR.window_max(rob, xs[i], ys[i], half)
            finally:
                reader.close()
        m25, m8, p8, p25 = (vals[:, j] for j in range(4))
        is_take = ev.event.values == "take_off"
        occ = np.where(is_take, m8, np.fmin(p8, p25))
        emp = np.where(is_take, np.fmax(p8, p25), m25)
        with np.errstate(invalid="ignore"):
            ok = (occ >= float(b["px_min_contrast"])) & (occ - emp >= float(b["px_min_change"]))
        ev["px_m25"], ev["px_m8"], ev["px_p8"], ev["px_p25"] = m25, m8, p8, p25
        kinds = ev.event.isin(KINDS).values
        ev["px_ok"] = np.where(chk, ok, ~kinds).astype(bool)
        ev["verified"] = site_ok & ev["px_ok"].values
        C.write_table(ev, ep)
        return stem, "ok", "%d checked, %d verified" % (int(chk.sum()), int((ev["verified"].values & kinds).sum()))
    except Exception as exc:  # noqa: BLE001
        return stem, "error", "%s: %s" % (type(exc).__name__, exc)
