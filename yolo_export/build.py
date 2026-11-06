"""build: YOLO tile datasets from the frames meta (fast; re-run with another policy at will).

Steps: attach corrected time / stimulus / light phase to every sampled frame; split by whole
segment (test: held-out days + test experiments; val: 10% by hash; auxiliary rigs: train only);
work out each experiment's class-median boxes; apply the label policy; cut 640 tiles over the cage
bounding box; accept tiles whose labels are well supported and that hold no unexplained blob; write

    yolo/stack/  images/{train,val,test}/*.png (3 channels: t, t-1, t-2) + labels/... + data.yaml
    yolo/single/ the same tiles, frame t only (1-channel PNG, read back as 3 equal channels)
    calib/{stack,single}/  calibration tiles for the Hailo INT8 quantisation
    meta/frames_all.parquet, meta/tiles.parquet, splits.csv, build_report.json
"""
import math
import os
import shutil

import numpy as np

from yolo_export import clock as CK
from yolo_export import common as C
from yolo_export import frames as FR
from yolo_export import inventory as I
from yolo_export import junk_filter as JF

VARIANTS = ("stack", "single")


# --------------------------------------------------------------------------- clock / splits
def _clock_or_fallback(out_dir, row, selection):
    ck = CK.load_clock(out_dir, row["alias"])
    if ck:
        return ck
    fps_fb = float((selection.get("rig_fps") or {}).get(row["rig_id"], C.NOMINAL_FPS))
    return {"a": {}, "b": 86400.0 * (1.0 - C.NOMINAL_FPS / fps_fb), "nf": 30060, "fitted": False}


def default_test_days(row, clock):
    """The last calendar day with >= 60 tracked segments (by segment mid-time)."""
    counts = {}
    nf = int(clock.get("nf", 30060))
    for stem in I.segments_of(row):
        sess, k = C.session_key(stem)
        if sess is None:
            continue
        t = CK.frame_times(clock, sess, k, [nf // 2])[0]
        d = str(t.astype("datetime64[D]"))
        counts[d] = counts.get(d, 0) + 1
    full = sorted(d for d, n in counts.items() if n >= 60)
    return [full[-1]] if full else []


def split_of(stem, row, test_days, seg_day):
    if row.get("role") == "aux":
        return "train"
    if row.get("test_experiment"):
        return "test"
    if seg_day in test_days:
        return "test"
    return "val" if C.seed_for(stem) % 10 == 0 else "train"


# --------------------------------------------------------------------------- boxes / policy
def class_medians(L, cw, defs):
    """{cls: (w, h)} px: the fixed box of each class for one experiment (DEFINITIONS["boxes"])."""
    bx = defs["boxes"]
    out = {}
    for c, name in ((0, "resting"), (1, "flying")):
        out[c] = None
        if L is not None and len(L):
            clean = L[(L.cls == c) & L.supported & (L.comp_nlabels == 1)]
            if c == 1 and "mcontrast" in clean:       # flying boxes from clearly moving flyers only
                clean = clean[clean.mcontrast >= defs["policy"]["flying_strong_motion"]]
            for cmin in bx["size_min_contrast"]:
                t = clean[clean.contrast >= cmin]
                if len(t) >= bx["size_min_samples"]:
                    out[c] = (float(np.median(t.comp_w)), float(np.median(t.comp_h)))
                    break
        if out[c] is None:
            d = bx["default_box_cw"][name] * cw
            out[c] = (d, d)
    return out


def label_boxes(L, med, defs, cw=None):
    """Adds keep, uncertain, static_fly, orphan_fly, merged, rest_sat, junk, bx, by, bw, bh, box_src to
    the labels table (full-frame px).

    Labels sharing one dark component keep their tracker positions (several touching mosquitoes).
    Flying labels without motion of their own are dropped (static_fly, policy.flying_note); with cw
    given, those on a dark blob with no resting label nearby are flagged orphan_fly. With cw given,
    same-class labels closer than boxes.dup_merge_cw are merged into one, and resting labels without
    a body of their own beside a stronger one are dropped (rest_sat, policy.rest_sat_note), and
    kept resting labels the junk filter scores as false are dropped (junk, policy.junk_filter_note).
    """
    pol = defs["policy"]
    L = L.copy()
    sup = L.supported.values.astype(bool)
    st0 = L.state.values == 0
    edge = L.edge_suspect.values.astype(bool) if "edge_suspect" in L else np.zeros(len(L), bool)
    keep = sup | (st0 & ~edge & (L.contrast.values >= pol["keep_unsupported_state0_min_contrast"]))
    L["bx"], L["by"] = L.x.values.astype(float), L.y.values.astype(float)
    if "comp_nlabels" in L and "x_raw" in L:
        shared = L.comp_nlabels.values > 1
        L.loc[shared, "bx"] = L.x_raw.values[shared].astype(float)
        L.loc[shared, "by"] = L.y_raw.values[shared].astype(float)
    resting = keep & (L.cls.values == 0)
    static = keep & ~flying_moves(L, resting, pol, cw)
    keep &= ~static
    L["static_fly"] = static
    orphan = np.zeros(len(L), bool)              # dropped, but on a dark blob of its own: plan_tiles
    if cw and static.any():                      # counts it as an unexplained blob
        st = np.flatnonzero(static & sup)
        orphan[st] = ~near_labels(L, st, resting, pol["flying_near_rest_cw"] * float(cw))
    L["orphan_fly"] = orphan
    merged = np.zeros(len(L), bool)
    sat = np.zeros(len(L), bool)
    if cw:
        merged = _same_mosquito(L, keep, sup, defs["boxes"]["dup_merge_cw"] * float(cw))
        keep &= ~merged
        sat = resting_satellites(L, keep, sup, defs, cw, med)
        keep &= ~sat
    junk = np.zeros(len(L), bool)
    if cw and pol.get("junk_filter"):
        JF.require()                             # never a silent no-op (build() checks this up front too)
        p = JF.score(L, cw)
        junk = keep & (L.cls.values == 0) & (p < JF.threshold() if np.isfinite(p).any() else False)
        keep &= ~junk
    L["junk"] = junk
    L["keep"] = keep
    L["uncertain"] = keep & ~sup
    L["merged"] = merged
    L["rest_sat"] = sat
    L["bw"] = L.cls.map({0: med[0][0], 1: med[1][0]}).astype(float).values
    L["bh"] = L.cls.map({0: med[0][1], 1: med[1][1]}).astype(float).values
    L["box_src"] = "fixed"
    return L


def flying_moves(L, resting, pol, cw=None):
    """Bool mask: False for flying labels without motion of their own (policy.flying_note), True
    for every other label. `resting` marks the resting labels a weak mover must stay clear of; with
    no cw only the motion thresholds apply. Labels meta without mcontrast: all True."""
    ok = np.ones(len(L), bool)
    if "mcontrast" not in L or not len(L):
        return ok
    m = L.mcontrast.values.astype(float)
    fly = L.cls.values == 1
    ok[fly & (m < pol["flying_min_motion"])] = False
    weak = np.flatnonzero(fly & (m >= pol["flying_min_motion"]) & (m < pol["flying_strong_motion"]))
    if len(weak) and cw:
        ok[weak[near_labels(L, weak, resting, pol["flying_near_rest_cw"] * float(cw))]] = False
    return ok


def near_labels(L, sel, other, r):
    """Bool per position in `sel`: a label marked in `other` lies within r px in the same frame."""
    out = np.zeros(len(sel), bool)
    o_pos = np.flatnonzero(other)
    if not len(sel) or not len(o_pos):
        return out
    o_by_f = L.iloc[o_pos].groupby("frame_id").indices
    x, y = L.bx.values, L.by.values
    for fid, idx in L.iloc[sel].groupby("frame_id").indices.items():
        o = o_by_f.get(fid)
        if o is None:
            continue
        s, oo = sel[idx], o_pos[o]
        d2 = (x[s, None] - x[None, oo]) ** 2 + (y[s, None] - y[None, oo]) ** 2
        out[idx] = (d2 <= r * r).any(axis=1)
    return out


def _same_mosquito(L, keep, sup, r):
    """Bool mask of kept labels to drop: within r px of a better kept label of the same class and frame.

    Better = supported first, then higher contrast; the best of each close group survives.
    """
    drop = np.zeros(len(L), bool)
    cand = np.flatnonzero(keep)
    if not len(cand) or r <= 0:
        return drop
    sub = L.iloc[cand]
    order = np.lexsort((-sub.contrast.values, ~sup[cand]))      # supported, then contrast, best first
    cand = cand[order]
    sub = L.iloc[cand]
    fx, fy = sub.bx.values, sub.by.values
    for (_fid, _cls), idx in sub.groupby(["frame_id", "cls"], sort=False).indices.items():
        if len(idx) < 2:
            continue
        x, y = fx[idx], fy[idx]
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):                                # idx keeps the best-first order
            if alive[i]:
                close = (x[i + 1:] - x[i]) ** 2 + (y[i + 1:] - y[i]) ** 2 < r * r
                alive[i + 1:] &= ~close
        drop[cand[idx[~alive]]] = True
    return drop


def resting_area(L, defs, med):
    """px^2: the dark-blob area of one resting mosquito -- the median of supported single-label
    resting detections (contrast tiers as in class_medians); 0.6 x the resting box when too few."""
    bx = defs["boxes"]
    if "comp_area" in L:
        clean = L[(L.cls == 0) & L.supported & (L.comp_nlabels == 1)]
        for cmin in bx["size_min_contrast"]:
            t = clean[clean.contrast >= cmin]
            if len(t) >= bx["size_min_samples"]:
                return float(np.median(t.comp_area))
    return 0.6 * med[0][0] * med[0][1]


def resting_satellites(L, keep, sup, defs, cw, med):
    """Bool mask of kept resting labels to drop: one mosquito tracked twice (policy.rest_sat_note).

    Strongest first (supported, then contrast); a label within rest_sat_radius_cw of a stronger kept
    resting label is dropped when it is closer than rest_sat_merge_box x the resting box, is
    unsupported, sits on a blob under rest_sat_min_area x a single mosquito's, has under
    rest_sat_contrast_frac x that neighbour's contrast, or shares a blob already holding as many
    stronger labels as the blob has room for. Labels meta without blob columns (synthetic): no drops.

    Blob area grows with contrast, so the area test works mostly as "faint" next to a stronger label:
    at contrast <= 25 the dropped blobs are as big as isolated faint labels, at >= 30 they are
    0.1-0.4 of them (2026-10-28). Beside a mosquito a faint small blob is most often its own legs,
    so the test stays tied to one mosquito's blob. The flying rule in label_boxes runs before this
    step and sees these labels as resting.
    """
    pol = defs["policy"]
    drop = np.zeros(len(L), bool)
    if not cw or not all(c in L for c in ("comp_area", "comp_x", "comp_y", "comp_w", "comp_h")):
        return drop
    cand = np.flatnonzero(keep & (L.cls.values == 0))
    if len(cand) < 2:
        return drop
    a1 = resting_area(L, defs, med)
    r2 = (pol["rest_sat_radius_cw"] * float(cw)) ** 2
    m2 = (pol["rest_sat_merge_box"] * 0.5 * (med[0][0] + med[0][1])) ** 2
    sub = L.iloc[cand]
    order = np.lexsort((-sub.contrast.values, ~sup[cand]))      # supported, then contrast, best first
    cand = cand[order]
    sub = L.iloc[cand]
    x, y = sub.bx.values.astype(float), sub.by.values.astype(float)
    area = np.nan_to_num(sub.comp_area.values.astype(float))
    con = np.nan_to_num(sub.contrast.values.astype(float))
    comp = ((sub.comp_x.values.astype(np.int64) * 4096 + sub.comp_y.values) * 4096
            + sub.comp_w.values) * 4096 + sub.comp_h.values     # one dark component of one frame
    s = sup[cand]
    for idx in sub.groupby("frame_id", sort=False).indices.values():
        if len(idx) < 2:
            continue
        d2 = (x[idx, None] - x[None, idx]) ** 2 + (y[idx, None] - y[None, idx]) ** 2
        alive = np.ones(len(idx), bool)
        for k in range(1, len(idx)):                          # idx keeps the best-first order
            near = alive[:k] & (d2[k, :k] < r2)
            if not near.any():
                continue
            nb, j = idx[:k][near], idx[k]
            same = comp[nb] == comp[j]
            if (d2[k, :k][near].min() < m2 or not s[j] or area[j] < pol["rest_sat_min_area"] * a1
                    or con[j] < pol["rest_sat_contrast_frac"] * con[nb].max()
                    or (same.any() and same.sum() >= max(1, int(round(area[j] / a1))))):
                alive[k] = False
        drop[cand[idx[~alive]]] = True
    return drop


def segment_gate(F, L, defs):
    """{stem: reason} for segments left out: tracker-noise label counts or poor pixel support."""
    pol = defs["policy"]
    if L is None or not len(L):
        return {}
    per_frame = F.groupby("stem").n_labels.mean()
    LS = L.merge(F[["frame_id", "stem"]], on="frame_id")
    edge = LS.edge_suspect.astype(bool) if "edge_suspect" in LS else pd_false(LS)
    edge_frac = edge.groupby(LS.stem).mean()
    core = LS[~edge & (LS.motion != 3)]
    support = core.groupby("stem").supported.mean()
    med = float(per_frame.median()) if len(per_frame) else 0.0
    out = {}
    for stem, n in per_frame.items():
        if med > 0 and n > pol["segment_max_label_ratio"] * med:
            out[stem] = "segment excluded: %.0f labels/frame vs experiment median %.0f" % (n, med)
        elif edge_frac.get(stem, 0.0) > pol["segment_max_edge_frac"]:
            out[stem] = "segment excluded: %.0f%% of labels are edge suspects" % (100 * edge_frac[stem])
        elif stem in support and support[stem] < pol["segment_min_support"]:
            out[stem] = "segment excluded: label support %.0f%%" % (100 * support[stem])
    return out


def pd_false(df):
    import pandas as pd
    return pd.Series(False, index=df.index)


def tile_origins(lo, hi, size, limit, min_overlap):
    """Tile start positions covering [lo, hi) inside [0, limit) with >= min_overlap between tiles."""
    lo, hi = max(0.0, lo), min(float(limit), hi)
    span = hi - lo
    if limit <= size:
        return [0]
    if span <= size:
        return [int(round(min(max(lo - (size - span) / 2.0, 0), limit - size)))]
    n = int(math.ceil((span - min_overlap) / float(size - min_overlap)))
    n = max(n, 2)
    xs = np.linspace(lo, hi - size, n)
    return [int(round(min(max(x, 0), limit - size))) for x in xs]


# --------------------------------------------------------------------------- tile planning
def plan_tiles(F, L, B, defs, roi, excluded=None):
    """Decide every tile of one experiment's frames (no image I/O). Returns a list of tile dicts."""
    tsize = int(defs["tiles"]["size"])
    minvis = float(defs["tiles"]["min_visible_frac"])
    pol = defs["policy"]
    excluded = excluded or {}
    by_f = {k: g for k, g in L.groupby("frame_id")} if L is not None and len(L) else {}
    bl_f = {k: g for k, g in B.groupby("frame_id")} if B is not None and len(B) else {}
    tiles = []
    for fr in F.itertuples(index=False):
        x0r, y0r, x1r, y1r = roi
        xs = tile_origins(x0r, x1r, tsize, fr.width, 64)
        ys = tile_origins(y0r, y1r, tsize, fr.height, 64)
        lab = by_f.get(fr.frame_id)
        blb = bl_f.get(fr.frame_id)
        if lab is not None:
            kx0 = (lab.bx - lab.bw / 2).values
            kx1 = (lab.bx + lab.bw / 2).values
            ky0 = (lab.by - lab.bh / 2).values
            ky1 = (lab.by + lab.bh / 2).values
            area = (kx1 - kx0) * (ky1 - ky0)
            keepm = lab.keep.values.astype(bool)
            uncm = lab.uncertain.values.astype(bool)
            px, py = lab.x.values, lab.y.values
            supm = lab.supported.values.astype(bool)
            orph = lab.orphan_fly.values.astype(bool) if "orphan_fly" in lab else np.zeros(len(lab), bool)
            cls = lab.cls.values
            junkm = lab.junk.values.astype(bool) if "junk" in lab else np.zeros(len(lab), bool)
            gh = pol["junk_grey_scale"] * 0.5 * np.maximum(lab.bw.values, lab.bh.values)
            gx0, gx1, gy0, gy1 = px - gh, px + gh, py - gh, py + gh
        for ri, ty in enumerate(ys):
            for ci, tx in enumerate(xs):
                rec = {"tile_id": "%s__r%dc%d" % (fr.frame_id, ri, ci), "frame_id": fr.frame_id,
                       "tx": tx, "ty": ty, "n_labels": 0, "n_flying": 0, "n_resting": 0, "n_points": 0,
                       "n_supported": 0, "n_kept": 0, "n_uncertain": 0, "n_unexplained": 0, "lines": "",
                       "n_grey": 0, "grey": ""}
                if lab is not None:
                    inside = (px >= tx) & (px < tx + tsize) & (py >= ty) & (py < ty + tsize)
                    rec["n_points"] = int(inside.sum())
                    rec["n_supported"] = int((inside & supm).sum())
                    rec["n_kept"] = int((inside & keepm).sum())
                    rec["n_uncertain"] = int((inside & uncm).sum())
                    rec["n_unexplained"] = int((inside & orph).sum())      # dropped flyers on a blob
                    ix0, ix1 = np.maximum(kx0, tx), np.minimum(kx1, tx + tsize)
                    iy0, iy1 = np.maximum(ky0, ty), np.minimum(ky1, ty + tsize)
                    vis = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None) / np.maximum(area, 1e-6)
                    use = keepm & (vis >= minvis)
                    if use.any():
                        cx = ((ix0 + ix1) / 2.0 - tx) / tsize
                        cy = ((iy0 + iy1) / 2.0 - ty) / tsize
                        ww = (ix1 - ix0) / tsize
                        hh = (iy1 - iy0) / tsize
                        rec["lines"] = "".join("%d %.6f %.6f %.6f %.6f\n" % (c, a, b, w, h) for c, a, b, w, h in
                                               zip(cls[use], cx[use], cy[use], ww[use], hh[use]))
                        rec["n_labels"] = int(use.sum())
                        rec["n_flying"] = int((cls[use] == 1).sum())
                        rec["n_resting"] = int((cls[use] == 0).sum())
                    gsel = np.flatnonzero(junkm & inside)
                    kk = np.flatnonzero(keepm)
                    grey = []
                    for j in gsel:                       # grey out junk spots that touch no kept label
                        if not ((gx0[j] < kx1[kk]) & (gx1[j] > kx0[kk]) & (gy0[j] < ky1[kk]) & (gy1[j] > ky0[kk])).any():
                            grey.append("%d,%d,%d,%d" % (gx0[j] - tx, gy0[j] - ty, math.ceil(gx1[j] - tx), math.ceil(gy1[j] - ty)))
                    rec["grey"] = ";".join(grey)
                    rec["n_grey"] = len(grey)
                if blb is not None:
                    bx, byy = blb.x.values, blb.y.values
                    rec["n_unexplained"] += int(((bx >= tx) & (bx < tx + tsize) & (byy >= ty) & (byy < ty + tsize)).sum())
                if fr.stem in excluded:
                    rec["decision"], rec["reason"] = "reject", excluded[fr.stem]
                elif fr.ref_kind == "none":
                    rec["decision"], rec["reason"] = "reject", "no reference image for QA"
                elif rec["n_kept"] == 0 and rec["n_unexplained"] == 0:
                    rec["decision"], rec["reason"] = "empty", ""
                elif rec["n_uncertain"] > max(1, int(pol["max_uncertain_frac"] * rec["n_kept"])):
                    rec["decision"], rec["reason"] = "reject", "%d of %d labels uncertain" % (rec["n_uncertain"], rec["n_kept"])
                elif rec["n_unexplained"] > pol["max_unexplained"]:
                    rec["decision"], rec["reason"] = "reject", "%d unexplained blobs" % rec["n_unexplained"]
                else:
                    rec["decision"], rec["reason"] = "accept", ""
                tiles.append(rec)
    return tiles


# --------------------------------------------------------------------------- tile writing (workers)
def write_tiles(job):
    """Worker: crop and write the selected tiles of a group of frames. Returns (n_written, errors)."""
    import cv2
    n, errs = 0, []
    tsize = job["tsize"]
    for fr in job["frames"]:
        img = cv2.imread(os.path.join(job["out_dir"], fr["png"]), cv2.IMREAD_UNCHANGED)
        if img is None:
            errs.append("unreadable %s" % fr["png"])
            continue
        for t in fr["tiles"]:
            crop = img[t["ty"]:t["ty"] + tsize, t["tx"]:t["tx"] + tsize]
            if t.get("grey"):
                crop = grey_out(crop, t["grey"], job["ring"])
            for variant in VARIANTS:
                root = os.path.join(job["tmp"][variant])
                ip = os.path.join(root, "images", t["split"], t["tile_id"] + ".png")
                lp = os.path.join(root, "labels", t["split"], t["tile_id"] + ".txt")
                data = crop if variant == "stack" else crop[:, :, 0]
                cv2.imwrite(ip, data, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                with open(lp, "w") as f:
                    f.write(t["lines"])
            n += 1
    return n, errs


def grey_out(crop, boxes, ring):
    """Copy of `crop` with each 'x0,y0,x1,y1' box (tile px) filled by the per-channel median of a `ring`-px
    band around it: the junk label's spot becomes plain background."""
    out = crop.copy()
    h, w = out.shape[:2]
    for b in boxes.split(";"):
        x0, y0, x1, y1 = [int(v) for v in b.split(",")]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        X0, Y0, X1, Y1 = max(x0 - ring, 0), max(y0 - ring, 0), min(x1 + ring, w), min(y1 + ring, h)
        band = np.ones((Y1 - Y0, X1 - X0), bool)
        band[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        if band.any():
            out[y0:y1, x0:x1] = np.median(out[Y0:Y1, X0:X1][band], axis=0).astype(out.dtype)
    return out


def _data_yaml(path, root, names, variant):
    chan = ("3 channels from cv2: B=t, G=t-1, R=t-2 (Ultralytics feeds RGB = t-2, t-1, t)"
            if variant.startswith("stack") else "frame t only (gray, read back as 3 equal channels)")
    lines = ["# BuzzWatch live detector - YOLO tile dataset (%s). See ../../README.md and" % variant,
             "# ../../docs/definitions.json. Images: %s." % chan,
             "# Train with hsv_h=0 hsv_s=0 (channels are gray / time, not colour).",
             "path: %s" % root.replace("\\", "/"), "train: images/train", "val: images/val", "test: images/test",
             "names:"]
    lines += ["  %d: %s" % (k, v) for k, v in sorted(names.items())]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _swap_dir(tmp, final):
    if os.path.isdir(final):
        shutil.rmtree(final)
    os.replace(tmp, final)


def build(out_dir, rows, selection, workers=None, single_class=False, log=C.log_default, pool_map=None):
    """Build the tile datasets for every chosen experiment with frames meta. Returns the report."""
    import pandas as pd
    defs = C.definitions()
    tsize = int(defs["tiles"]["size"])
    margin = float(defs["tiles"]["roi_margin_px"])
    junk_on = bool(defs["policy"].get("junk_filter"))
    if junk_on:
        JF.require()  # before anything is written
    all_frames, all_tiles = [], []
    report = {"experiments": {}, "junk_filter": {"applied": junk_on,
                                                 "threshold": JF.threshold() if junk_on else None}}
    for row in rows:
        alias = row["alias"]
        stems = I.segment_set(row)
        F = FR.load_meta(out_dir, alias, "frames", stems)
        if F is None or not len(F):
            continue
        if F.session.isna().any() or F.k.isna().any():  # meta written before '-<attempt>' stems parsed
            keys = [C.session_key(s) for s in F.stem]
            F = F.assign(session=[a for a, _ in keys], k=[b for _, b in keys])
            F = F[F.session.notna()].reset_index(drop=True)
            if not len(F):
                continue
        L = FR.load_meta(out_dir, alias, "labels", stems)
        if L is not None and len(L) and ("mcontrast" not in L or L.mcontrast.isna().any()):
            raise RuntimeError("%s: some labels meta has no motion contrast (made before generator 1.1), "
                               "which the flying-label rule needs -- run 'yolo frames --force' first" % alias)
        B = FR.load_meta(out_dir, alias, "blobs", stems)
        clock = _clock_or_fallback(out_dir, row, selection)
        sched = CK.schedule_of(row)
        windows = C.parse_windows(row["stim_windows"])
        times = np.array([CK.frame_times(clock, s, k, [v])[0] for s, k, v in
                          zip(F.session, F.k, F.video_frame)], dtype="datetime64[ms]")
        on, edge, phase, tag = CK.flags(times, windows, sched, defs)
        F = F.assign(time=times, stim_on=on, stim_edge=edge, light_phase=phase, day_tag=tag,
                     role=row.get("role", ""))
        test_days = (selection.get("test_days") or {}).get(alias) or default_test_days(row, clock)
        seg_mid = {}
        for stem, sess, k, n in F[["stem", "session", "k", "n_frames"]].drop_duplicates("stem").itertuples(index=False):
            seg_mid[stem] = str(CK.frame_times(clock, sess, k, [int(n) // 2])[0].astype("datetime64[D]"))
        F["split"] = [split_of(s, row, test_days, seg_mid[s]) for s in F.stem]
        cw = float(F.cw.iloc[0])
        med = class_medians(L, cw, defs)
        if L is not None and len(L):
            L = label_boxes(L, med, defs, cw)
        excluded = segment_gate(F, L, defs)
        for stem, why in sorted(excluded.items()):
            log("  %s: %s -- %s" % (alias, stem, why))
        geo = C.load_geometry(row["exp_dir"])
        x0, y0, x1, y1 = geo["bbox"]
        roi = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        tiles = plan_tiles(F, L, B, defs, roi, excluded)
        split_by_frame = dict(zip(F.frame_id, F.split))
        phase_by_frame = dict(zip(F.frame_id, F.light_phase))
        for t in tiles:
            t.update(alias=alias, rig=row["rig_id"], split=split_by_frame[t["frame_id"]],
                     light_phase=phase_by_frame[t["frame_id"]])
        all_tiles.extend(tiles)
        all_frames.append(F)
        report["experiments"][alias] = {
            "frames": int(len(F)), "test_days": test_days, "clock_fitted": bool(clock.get("fitted")), "cw": cw,
            "labels_merged": int(L.merged.sum()) if L is not None and len(L) else 0,
            "labels_static_flying": int(L.static_fly.sum()) if L is not None and len(L) else 0,
            "labels_orphan_flying": int(L.orphan_fly.sum()) if L is not None and len(L) else 0,
            "labels_resting_satellites": int(L.rest_sat.sum()) if L is not None and len(L) else 0,
            "labels_junk": int(L.junk.sum()) if L is not None and len(L) else 0,
            "median_box_px": {"resting": [round(v, 1) for v in med[0]], "flying": [round(v, 1) for v in med[1]]},
            "labels": int(len(L)) if L is not None else 0,
            "labels_kept": int(L.keep.sum()) if L is not None and len(L) else 0,
            "label_support": float(L.supported.mean()) if L is not None and len(L) else None,
            "segments_excluded": excluded,
            "splits": F.split.value_counts().to_dict()}
    if not all_frames:
        raise RuntimeError("no frames meta found -- run the 'frames' action first")
    T = pd.DataFrame(all_tiles)
    # keep at most max_empty_frac empty tiles per split
    T["selected"] = T.decision == "accept"
    rng = np.random.default_rng(12345)
    for sp, g in T.groupby("split"):
        n_acc = int((g.decision == "accept").sum())
        emp = g.index[g.decision == "empty"].values
        n_keep = min(len(emp), int(defs["policy"]["max_empty_frac"] * n_acc))
        if n_keep:
            T.loc[rng.choice(emp, size=n_keep, replace=False), "selected"] = True

    # write tiles into temporary dataset folders, then swap them in
    tmp = {v: os.path.join(out_dir, "yolo", v + ".tmp") for v in VARIANTS}
    for v in VARIANTS:
        if os.path.isdir(tmp[v]):
            shutil.rmtree(tmp[v])
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(tmp[v], "images", sp))
            os.makedirs(os.path.join(tmp[v], "labels", sp))
    Fall = pd.concat(all_frames, ignore_index=True)
    sel = T[T.selected]
    png_of = dict(zip(Fall.frame_id, Fall.png))
    stem_of = dict(zip(Fall.frame_id, Fall.stem))
    groups = {}
    for t in sel.to_dict(orient="records"):
        groups.setdefault(stem_of[t["frame_id"]], {}).setdefault(t["frame_id"], []).append(
            {k: t[k] for k in ("tile_id", "tx", "ty", "split", "lines", "grey")})
    jobs = [{"out_dir": out_dir, "tmp": tmp, "tsize": tsize, "ring": int(defs["policy"]["junk_grey_ring_px"]),
             "frames": [{"png": png_of[fid], "tiles": tl} for fid, tl in fr.items()]}
            for fr in groups.values()]
    log("  writing %d tiles (%d accepted, %d empty kept, %d rejected) from %d frames ..."
        % (len(sel), int((T.decision == "accept").sum()), int((sel.decision == "empty").sum()),
           int((T.decision == "reject").sum()), len(Fall)))
    results = (pool_map or map)(write_tiles, jobs)
    n_written, errs = 0, []
    for n, e in results:
        n_written += n
        errs.extend(e)
    for v in VARIANTS:
        _swap_dir(tmp[v], os.path.join(out_dir, "yolo", v))
        _data_yaml(os.path.join(out_dir, "yolo", v, "data.yaml"), os.path.join(out_dir, "yolo", v),
                   {int(k): n for k, n in defs["classes"].items()}, v)
    if single_class:
        _single_class_copies(out_dir, log)

    # Only tiles that were actually written (write_tiles skips a frame whose PNG is unreadable and
    # reports it in errs) can be calibration tiles or count as written in the tables below.
    img_dir = os.path.join(out_dir, "yolo", VARIANTS[0], "images")
    T["written"] = [bool(s) and os.path.isfile(os.path.join(img_dir, sp, tid + ".png"))
                    for s, sp, tid in zip(T.selected, T.split, T.tile_id)]
    if errs:
        log("  ! %d frame(s) could not be read; their tiles were left out (first: %s)" % (len(errs), errs[0]))
    sel = T[T.written]

    # calibration tiles: stratified over experiment x light phase, labelled train tiles first
    calib = _pick_calib(sel, 1024)
    for v in VARIANTS:
        d = os.path.join(out_dir, "calib", v)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for t in calib.itertuples(index=False):
            shutil.copy2(os.path.join(out_dir, "yolo", v, "images", t.split, t.tile_id + ".png"), d)
    C.write_table(Fall, os.path.join(out_dir, "meta", "frames_all.parquet"))
    C.write_table(T.drop(columns=["lines"]), os.path.join(out_dir, "meta", "tiles.parquet"))
    Fall[["alias", "stem", "split"]].drop_duplicates().to_csv(os.path.join(out_dir, "splits.csv"), index=False)
    report.update({
        "tiles_total": int(len(T)), "tiles_written": int(n_written), "errors": errs[:20],
        "decisions": T.decision.value_counts().to_dict(),
        "written_by_split": sel.split.value_counts().to_dict(),
        "objects_by_split": {sp: {"resting": int(g.n_resting.sum()), "flying": int(g.n_flying.sum())}
                             for sp, g in sel.groupby("split")},
        "reject_reasons": T[T.decision == "reject"].reason.str.replace(r"\d+", "N", regex=True)
                          .value_counts().head(10).to_dict(),
        "calibration_tiles": int(len(calib))})
    C.write_json(os.path.join(out_dir, "meta", "build_report.json"), report)
    log("  build: %s" % {k: report[k] for k in ("tiles_written", "decisions", "written_by_split", "objects_by_split")})
    return report


def _pick_calib(sel, n):
    tr = sel[(sel.split == "train") & (sel.n_labels > 0)]
    if not len(tr):
        tr = sel[sel.n_labels > 0]
    if not len(tr):
        return tr
    rng = np.random.default_rng(7)
    buckets = [g.sample(frac=1.0, random_state=int(rng.integers(1 << 30))) for _, g in tr.groupby(["alias", "light_phase"])]
    out, i = [], 0
    while len(out) < min(n, len(tr)):
        for b in buckets:
            if i < len(b):
                out.append(b.iloc[i])
                if len(out) >= n:
                    break
        i += 1
    import pandas as pd
    return pd.DataFrame(out)


def _single_class_copies(out_dir, log):
    """yolo/<variant>_1class: hard-linked images, labels with every class set to 0 (YOLito-style)."""
    for v in VARIANTS:
        src = os.path.join(out_dir, "yolo", v)
        dst = os.path.join(out_dir, "yolo", v + "_1class")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(dst, "images", sp))
            os.makedirs(os.path.join(dst, "labels", sp))
            for n in os.listdir(os.path.join(src, "images", sp)):
                s = os.path.join(src, "images", sp, n)
                d = os.path.join(dst, "images", sp, n)
                try:
                    os.link(s, d)
                except OSError:
                    shutil.copy2(s, d)
            for n in os.listdir(os.path.join(src, "labels", sp)):
                with open(os.path.join(src, "labels", sp, n)) as f:
                    lines = ["0" + ln[ln.index(" "):] for ln in f if ln.strip()]
                with open(os.path.join(dst, "labels", sp, n), "w") as f:
                    f.write("".join(lines))
        _data_yaml(os.path.join(dst, "data.yaml"), dst, {0: "mosquito"}, v + "_1class")
    log("  single-class copies written (yolo/*_1class)")
