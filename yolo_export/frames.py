"""frames: sampled full frames (3-frame stack) with per-label pixel QA.

Per segment: choose K frames (half uniform, half weighted by the number of flyers; >= 2 s apart;
away from the segment edges and from tracking dropouts), decode frames t-2, t-1, t by exact seek,
save one PNG (cv2 channels B,G,R = t, t-1, t-2), and check every label against the frame's pixels:

  dark map  = <video>_rest.png - frame, thresholded at max(floor, k * std) from <video>_reststd.png
              (the tracker's own WS1/WS5 resting rule, cached in images_mortality/); fallback: the
              median background <video>.png with a flat threshold
  snap      = nearest dark component within snap_radius_cw; a flying label that finds none retries
              at tracker offset 0 (tracks recovered by the backward pass)
  unexplained blobs = mosquito-sized, contrast >= unexplained_min_contrast, inside the cage, and no
              label nearby -- a likely missed mosquito, used by build to reject tiles
  motion    = mcontrast: how much darker the label's spot is at t than at t-1 / t-2 (qa.motion_note);
              build drops flying labels without motion of their own (policy.flying_note)

Writes meta/{frames,labels,blobs}/<alias>/<stem>.parquet and frames/<rig>/<alias>/<stem>_<v>.png.
"""
import os

import numpy as np

from yolo_export import annotate as AN
from yolo_export import common as C
from yolo_export import tracks as TR

LABEL_COLUMNS = ["frame_id", "track_id", "state", "motion", "cls", "real_flight", "speed_px", "zone",
                 "dv", "offset", "x_raw", "y_raw", "supported", "support_dist", "x", "y", "contrast",
                 "rcontrast", "mcontrast", "edge_suspect", "comp_x", "comp_y", "comp_w", "comp_h", "comp_area",
                 "comp_nlabels"]
BLOB_COLUMNS = ["frame_id", "x", "y", "w", "h", "area", "contrast"]
FRAME_COLUMNS = ["frame_id", "alias", "rig", "stem", "session", "k", "video_frame", "n_frames", "png",
                 "width", "height", "cw", "n_labels", "n_flying", "n_resting", "n_supported",
                 "n_unexplained", "sample_kind", "ref_kind"]


def meta_paths(out_dir, alias, stem):
    return {k: os.path.join(out_dir, "meta", k, alias, stem + ".parquet") for k in ("frames", "labels", "blobs")}


def frame_id(alias, stem, v):
    return "%s__%s__%05d" % (alias, stem, v)


def png_rel(rig, alias, stem, v):
    return os.path.join("frames", rig, alias, "%s_%05d.png" % (stem, v))


# --------------------------------------------------------------------------- sampling
def choose_frames(P, T, K, defs, seed):
    """K frame indices (sorted) and {frame: 'uniform' | 'active'}."""
    s = defs["sampling"]
    edge = max(int(s["edge_frames"]), 2)
    vf = P["vf"]
    inr = (vf >= 0) & (vf < T)
    n_all = np.bincount(vf[inr], minlength=T)[:T]
    n_fly = np.bincount(vf[inr & (P["cls"] == 1)], minlength=T)[:T]
    ok = np.zeros(T, bool)
    ok[edge:max(edge, T - edge - 1)] = True
    if T and np.median(n_all) > 3:
        drop = n_all == 0
        if drop.any():
            g = int(s["dropout_guard_frames"])
            ok &= ~(np.convolve(drop.astype(np.int32), np.ones(2 * g + 1, np.int32), "same") > 0)
    cand = np.nonzero(ok)[0]
    if K <= 0 or not len(cand):
        return [], {}
    rng = np.random.default_rng(seed)
    spacing = int(s["min_spacing_frames"])
    chosen, kind = [], {}

    def take(order, n, label):
        got = 0
        for v in order:
            if got >= n:
                break
            v = int(v)
            if all(abs(v - c) >= spacing for c in chosen):
                chosen.append(v)
                kind[v] = label
                got += 1

    take(rng.permutation(cand)[:20 * K], K // 2, "uniform")
    w = n_fly[cand].astype(float) + 0.1
    order = rng.choice(cand, size=min(len(cand), 20 * K), replace=False, p=w / w.sum())
    take(order, K - len(chosen), "active")
    return sorted(chosen), kind


# --------------------------------------------------------------------------- pixels
class Reader:
    """Exact frame access: seek far, step forward when the target is close (both pixel-exact)."""

    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        self.pos = 0
        if not self.cap.isOpened():
            raise RuntimeError("cannot open video %s" % path)

    def read_gray(self, start, n):
        cv2 = self.cv2
        if not (0 <= start - self.pos <= 150):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
            self.pos = int(start)
        while self.pos < start:
            if not self.cap.grab():
                return None
            self.pos += 1
        out = []
        for _ in range(n):
            ok, fr = self.cap.read()
            if not ok:
                return None
            self.pos += 1
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        return out

    def close(self):
        self.cap.release()


def load_reference(exp_dir, stem, tracking, defs):
    """(reference, threshold map float32, kind) for the dark-object map, or (None, None, 'none')."""
    import cv2
    q = defs["qa"]
    d = os.path.join(exp_dir, "images_mortality")
    rest = cv2.imread(os.path.join(d, stem + "_rest.png"), cv2.IMREAD_GRAYSCALE)
    if rest is not None:
        std = cv2.imread(os.path.join(d, stem + "_reststd.png"), cv2.IMREAD_GRAYSCALE)
        k = float(tracking.get("resting_adaptive_k", q["rest_thr_k"]))
        floor = float(tracking.get("resting_threshold_floor", q["rest_thr_floor"]))
        if std is not None and std.shape == rest.shape:
            thr = np.maximum(floor, k * std.astype(np.float32) / q["rest_std_scale"]).astype(np.float32)
        else:
            thr = np.full(rest.shape, floor, np.float32)
        return rest, thr, "rest"
    med = cv2.imread(os.path.join(d, stem + ".png"), cv2.IMREAD_GRAYSCALE)
    if med is not None:
        return med, np.full(med.shape, q["flat_thr_fallback"], np.float32), "median"
    return None, None, "none"


def components(gray, ref, thr, erode_px=3):
    """Dark components of one frame: (dark, robust dark, label image, stats, centroids, per-label max
    of dark, per-label max of robust dark). Robust dark = frame darker than the min-filtered
    reference, which a pixel-shifted structure edge is not."""
    import cv2
    dark = cv2.subtract(ref, gray)
    rob = cv2.subtract(cv2.erode(ref, np.ones((erode_px, erode_px), np.uint8)), gray)
    mask = (dark.astype(np.float32) > thr).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cmax = np.zeros(n, np.float32)
    rmax = np.zeros(n, np.float32)
    ys, xs = np.nonzero(lab)
    if len(ys):
        np.maximum.at(cmax, lab[ys, xs], dark[ys, xs].astype(np.float32))
        np.maximum.at(rmax, lab[ys, xs], rob[ys, xs].astype(np.float32))
    return dark, rob, lab, stats, cents, cmax, rmax


def snap(lab, x, y, r):
    """(component id, distance px) of the nearest labelled pixel within r of (x, y); (0, inf) if none."""
    h, w = lab.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if x0 >= x1 or y0 >= y1:
        return 0, np.inf
    win = lab[y0:y1, x0:x1]
    ny, nx = np.nonzero(win)
    if not len(ny):
        return 0, np.inf
    d2 = (nx + x0 - x) ** 2 + (ny + y0 - y) ** 2
    j = int(np.argmin(d2))
    return int(win[ny[j], nx[j]]), float(np.sqrt(d2[j]))


def window_max(img, x, y, half):
    h, w = img.shape
    xi, yi = int(round(x)), int(round(y))
    win = img[max(0, yi - half):min(h, yi + half + 1), max(0, xi - half):min(w, xi + half + 1)]
    return int(win.max()) if win.size else 0


# --------------------------------------------------------------------------- worker
def frames_segment(job):
    """Worker. Returns (stem, status, message)."""
    import cv2
    import pandas as pd
    paths = meta_paths(job["out_dir"], job["alias"], job["stem"])
    if os.path.isfile(paths["frames"]) and not job.get("force"):
        return job["stem"], "skip", ""
    reader = None
    try:
        defs = C.definitions()
        q = defs["qa"]
        g = job["geo"]
        cw = float(g["cw"])
        raw, P, _act, _ev, T, info = AN.load_points(job, defs, want_events=False)
        frames, kind = choose_frames(P, T, int(job["k_frames"]), defs, C.seed_for(job["alias"] + "/" + job["stem"]))
        ref, thr, ref_kind = load_reference(job["exp_dir"], job["stem"], job.get("tracking") or {}, defs)
        raw_keys = TR.keys_of(raw["tid"], raw["idx"])          # raw arrays are sorted by (tid, idx)
        order_vf = np.argsort(P["vf"], kind="stable")
        vf_sorted = P["vf"][order_vf]
        gap = int(defs["alignment"]["gap_fill_frames"])
        off = int(defs["alignment"]["flying_offset_frames"])
        r_snap = max(3, int(np.ceil(q["snap_radius_cw"] * cw)))
        box_rest = defs["boxes"]["default_box_cw"]["resting"] * cw
        r_unexp = max(2.0 * r_snap, box_rest)
        a_min, a_max = (v * cw * cw for v in q["blob_area_cw2"])
        max_blob = q["max_blob_cw"] * cw
        half_c = int(q["contrast_window_px"]) // 2
        cage = np.asarray(g["cage"], np.float32).reshape(-1, 1, 2)
        sess, k = C.session_key(job["stem"])
        reader = Reader(os.path.join(job["video_dir"], job["stem"] + ".mp4"))
        F, L, B = [], [], []
        for v in frames:
            grays = reader.read_gray(v - 2, 3)
            if grays is None:
                continue
            g2, g1, g0 = grays
            fid = frame_id(job["alias"], job["stem"], v)
            rel = png_rel(job["rig"], job["alias"], job["stem"], v)
            out_png = os.path.join(job["out_dir"], rel)
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            cv2.imwrite(out_png, cv2.merge([g0, g1, g2]), [cv2.IMWRITE_PNG_COMPRESSION, 3])

            # labels on frame v: the track's point on v, else its nearest within +-gap frames
            lo = np.searchsorted(vf_sorted, v - gap)
            hi = np.searchsorted(vf_sorted, v + gap, side="right")
            cand = order_vf[lo:hi]
            if len(cand):
                d = P["vf"][cand] - v
                o = np.lexsort((np.abs(d) * 2 + (d < 0), P["tid"][cand]))
                c2 = cand[o]
                first = np.ones(len(c2), bool)
                first[1:] = P["tid"][c2][1:] != P["tid"][c2][:-1]
                sel = c2[first]
            else:
                sel = np.zeros(0, np.int64)

            if ref is not None:
                dark, rob, lab, stats, cents, cmax, rmax = components(g0, ref, thr, int(q["robust_erode_px"]))
            mot = np.maximum(cv2.subtract(g1, g0), cv2.subtract(g2, g0))     # darker at t than before
            rows = []
            for j in sel:
                x_raw, y_raw = float(P["x"][j]), float(P["y"][j])
                row = {"frame_id": fid, "track_id": int(P["tid"][j]), "state": int(P["state"][j]),
                       "motion": int(P["motion"][j]), "cls": int(P["cls"][j]),
                       "real_flight": bool(P["real"][j]), "speed_px": float(P["speed"][j]),
                       "zone": int(P["zone"][j]), "dv": int(P["vf"][j] - v),
                       "offset": off if P["state"][j] == 1 else 0, "x_raw": x_raw, "y_raw": y_raw,
                       "supported": False, "support_dist": np.nan, "x": x_raw, "y": y_raw,
                       "contrast": 0, "rcontrast": 0, "edge_suspect": False, "comp_x": -1, "comp_y": -1,
                       "comp_w": 0, "comp_h": 0, "comp_area": 0, "comp_nlabels": 0, "_cid": 0}
                if ref is not None:
                    cid, dist = snap(lab, x_raw, y_raw, r_snap)
                    if (cid == 0 or dist > r_snap) and P["state"][j] == 1:
                        p = TR.lookup(raw_keys, TR.keys_of([P["tid"][j]], [v]))[0]
                        if p >= 0 and raw["state"][p] == 1:
                            cid2, dist2 = snap(lab, raw["x"][p], raw["y"][p], r_snap)
                            if cid2 and dist2 <= r_snap:
                                cid, dist = cid2, dist2
                                row.update(offset=0, x_raw=float(raw["x"][p]), y_raw=float(raw["y"][p]))
                    if cid and dist <= r_snap:
                        sx, sy, sw, sh, sa = (int(t) for t in stats[cid])
                        row.update(supported=True, support_dist=dist, comp_x=sx, comp_y=sy, comp_w=sw,
                                   comp_h=sh, comp_area=sa, _cid=cid)
                        if max(sw, sh) <= 2.5 * box_rest:
                            row["x"], row["y"] = float(cents[cid][0]), float(cents[cid][1])
                        else:
                            row["x"], row["y"] = row["x_raw"], row["y_raw"]
                    else:
                        row["x"], row["y"] = row["x_raw"], row["y_raw"]
                    row["contrast"] = window_max(dark, row["x"], row["y"], half_c)
                    row["rcontrast"] = window_max(rob, row["x"], row["y"], half_c)
                    if row["supported"] and row["state"] == 0 and row["rcontrast"] < q["edge_robust_min_contrast"]:
                        row.update(supported=False, edge_suspect=True, _cid=0)
                rows.append(row)
            if rows:
                cids = np.array([r["_cid"] for r in rows])
                for r in rows:
                    r["comp_nlabels"] = int((cids == r["_cid"]).sum()) if r["_cid"] else 0
                    if r["comp_nlabels"] > 1:       # touching mosquitoes share a blob: keep each tracker position
                        r["x"], r["y"] = r["x_raw"], r["y_raw"]
                    r["mcontrast"] = window_max(mot, r["x"], r["y"], half_c)
            n_unexp = 0
            if ref is not None and stats.shape[0] > 1:
                pts = np.array([(r["x"], r["y"]) for r in rows] + [(r["x_raw"], r["y_raw"]) for r in rows],
                               float).reshape(-1, 2)
                used = set(r["_cid"] for r in rows if r["_cid"])
                ids = np.arange(1, stats.shape[0])
                area = stats[ids, cv2.CC_STAT_AREA]
                size = np.maximum(stats[ids, cv2.CC_STAT_WIDTH], stats[ids, cv2.CC_STAT_HEIGHT])
                okb = (area >= a_min) & (area <= a_max) & (size <= max_blob) & \
                      (cmax[ids] >= q["unexplained_min_contrast"]) & (rmax[ids] >= q["unexplained_min_robust"])
                for i in ids[okb]:
                    if i in used:
                        continue
                    cx, cy = float(cents[i][0]), float(cents[i][1])
                    if len(pts) and np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) <= r_unexp:
                        continue
                    if cv2.pointPolygonTest(cage, (cx, cy), False) < 0:
                        continue
                    B.append({"frame_id": fid, "x": cx, "y": cy, "w": int(stats[i][2]),
                              "h": int(stats[i][3]), "area": int(stats[i][4]), "contrast": float(cmax[i])})
                    n_unexp += 1
            for r in rows:
                r.pop("_cid", None)
            L.extend(rows)
            F.append({"frame_id": fid, "alias": job["alias"], "rig": job["rig"], "stem": job["stem"],
                      "session": sess, "k": k, "video_frame": v, "n_frames": T, "png": rel,
                      "width": int(g0.shape[1]), "height": int(g0.shape[0]), "cw": cw,
                      "n_labels": len(rows), "n_flying": sum(r["cls"] == 1 for r in rows),
                      "n_resting": sum(r["cls"] == 0 for r in rows),
                      "n_supported": sum(bool(r["supported"]) for r in rows), "n_unexplained": n_unexp,
                      "sample_kind": kind.get(v, ""), "ref_kind": ref_kind})
        C.write_table(pd.DataFrame(L, columns=LABEL_COLUMNS), paths["labels"])
        C.write_table(pd.DataFrame(B, columns=BLOB_COLUMNS), paths["blobs"])
        C.write_table(pd.DataFrame(F, columns=FRAME_COLUMNS), paths["frames"])   # last = done marker
        return job["stem"], "ok", "%d frames, ref=%s" % (len(F), ref_kind)
    except Exception as exc:  # noqa: BLE001
        import traceback
        return job["stem"], "error", "%s: %s | %s" % (type(exc).__name__, exc,
                                                      traceback.format_exc().splitlines()[-3:])
    finally:
        if reader is not None:
            reader.close()


def load_meta(out_dir, alias, kind, stems=None):
    """Concatenate one experiment's meta tables of one kind ('frames', 'labels', 'blobs'); only the
    segments in `stems` when given (files left from an earlier selection are ignored)."""
    import pandas as pd
    d = os.path.join(out_dir, "meta", kind, alias)
    if not os.path.isdir(d):
        return None
    parts = []
    for n in sorted(os.listdir(d)):
        stem = n[:-len(".parquet")] if n.endswith(".parquet") else n[:-len(".csv.gz")]
        if stems is not None and stem not in stems:
            continue
        if n.endswith(".parquet") or n.endswith(".csv.gz"):
            t = C.read_table(os.path.join(d, n))
            if t is not None and len(t):
                parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else None
