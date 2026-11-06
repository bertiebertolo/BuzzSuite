"""qa: the label-quality audit (qa/report.md + qa/sheets/*.png).

  alignment   decode the busiest 100 frames of 3 segments per experiment and measure, for tracker
              offsets -3..+4, how far flying / resting points sit from the nearest dark blob; the
              flying minimum must be at +2 (DEFINITIONS["alignment"])
  statistics  label support, offset-0 rescues, gap fills, box sources, tile decisions, classes
  sheets      accepted / rejected tiles with boxes (red flying, blue resting), dropped labels (x),
              unexplained blobs (yellow); a motion composite of accepted stack tiles
  strips      5-frame strips (-25, -8, 0, +8, +25 frames) around verified take-offs, landings, walks
  clips       frame accuracy of the cut clips (clip frame 1 vs source frames start-1..start+1)
"""
import os

import numpy as np

from yolo_export import annotate as AN
from yolo_export import build as BU
from yolo_export import common as C
from yolo_export import frames as FR
from yolo_export import inventory as I
from yolo_export import tracks as TR

RED, BLUE, YELLOW, GREEN, WHITE = (0, 0, 255), (255, 0, 0), (0, 255, 255), (0, 200, 0), (255, 255, 255)


# --------------------------------------------------------------------------- alignment
def alignment_check(row, out_dir, n_segments=3, n_frames=100, offsets=range(-3, 5)):
    defs = C.definitions()
    rep = C.read_json(os.path.join(out_dir, "meta", "build_report.json")) or {}
    excluded = set(((rep.get("experiments") or {}).get(row["alias"]) or {}).get("segments_excluded") or {})
    infos = [(s, i) for s, i in AN.segment_infos(out_dir, row["rig_id"], row["alias"], I.segment_set(row))
             if s not in excluded]
    if not infos:
        return None
    pick = [infos[int(i)] for i in np.linspace(0, len(infos) - 1, min(n_segments, len(infos)))]
    res = {d: {0: [], 1: []} for d in offsets}
    for stem, info in pick:
        z = np.load(os.path.join(AN.seg_dir(out_dir, row["rig_id"], row["alias"]), stem + ".activity.npz"))
        fly = z["n_fly"].astype(float)
        if len(fly) < n_frames + 20:
            continue
        a = int(np.argmax(np.convolve(fly, np.ones(n_frames), "valid")[10:-10])) + 10
        ref, thr, kind = FR.load_reference(row["exp_dir"], stem, {}, defs)
        if ref is None:
            continue
        job = {"exp_dir": row["exp_dir"], "stem": stem, "geo": AN.job_geometry(row)}
        raw, _P, _act, _ev, _T, _info = AN.load_points(job, defs, want_events=False)
        keys = TR.keys_of(raw["tid"], raw["idx"])
        order = np.argsort(raw["idx"], kind="stable")
        idx_sorted = raw["idx"][order]
        reader = FR.Reader(info["video"])
        try:
            for v in range(a, a + n_frames):
                g = reader.read_gray(v, 1)
                if g is None:
                    break
                stats, cents = FR.components(g[0], ref, thr)[3:5]
                if stats.shape[0] < 2:
                    continue
                cc = cents[1:]
                for d in offsets:
                    lo, hi = np.searchsorted(idx_sorted, v + d), np.searchsorted(idx_sorted, v + d, side="right")
                    sel = order[lo:hi]
                    for s in (0, 1):
                        m = sel[raw["state"][sel] == s]
                        if len(m):
                            dist = np.min(np.hypot(cc[None, :, 0] - raw["x"][m][:, None],
                                                   cc[None, :, 1] - raw["y"][m][:, None]), axis=1)
                            res[d][s].extend(dist.tolist())
        finally:
            reader.close()
        del raw, keys
    out = []
    for d in offsets:
        f, r = np.array(res[d][1]), np.array(res[d][0])
        out.append({"offset": d, "flying_n": len(f), "flying_median_px": float(np.median(f)) if len(f) else np.nan,
                    "flying_within_5px": float((f < 5).mean()) if len(f) else np.nan,
                    "resting_n": len(r), "resting_median_px": float(np.median(r)) if len(r) else np.nan})
    return out


# --------------------------------------------------------------------------- drawing helpers
def _labels_with_boxes(out_dir, rows):
    import pandas as pd
    rep = C.read_json(os.path.join(out_dir, "meta", "build_report.json")) or {}
    defs = C.definitions()
    parts = []
    for row in rows:
        alias = row["alias"]
        L = FR.load_meta(out_dir, alias, "labels", I.segment_set(row))
        if L is None or not len(L):
            continue
        er = (rep.get("experiments") or {}).get(alias) or {}
        cw = er.get("cw")
        if not cw:                                   # no build report yet: the cage width from frames meta
            F = FR.load_meta(out_dir, alias, "frames", I.segment_set(row))
            cw = float(F.cw.iloc[0]) if F is not None and len(F) else None
        med = er.get("median_box_px")
        med = {0: tuple(med["resting"]), 1: tuple(med["flying"])} if med else BU.class_medians(L, cw or 1000.0, defs)
        parts.append(BU.label_boxes(L, med, defs, cw))
    return pd.concat(parts, ignore_index=True) if parts else None


def _draw_tile(img_bgr, t, labels, blobs, tsize):
    import cv2
    tx, ty = int(t["tx"]), int(t["ty"])
    if labels is not None:
        for r in labels.itertuples(index=False):
            if not (tx <= r.x < tx + tsize and ty <= r.y < ty + tsize):
                continue
            col = RED if r.cls == 1 else BLUE
            if r.keep and r.uncertain:
                col = (255, 0, 255)
            if r.keep:
                p0 = (int(r.bx - r.bw / 2 - tx), int(r.by - r.bh / 2 - ty))
                p1 = (int(r.bx + r.bw / 2 - tx), int(r.by + r.bh / 2 - ty))
                cv2.rectangle(img_bgr, p0, p1, col, 1)
            else:
                c = (int(r.x - tx), int(r.y - ty))
                cv2.drawMarker(img_bgr, c, col, cv2.MARKER_TILTED_CROSS, 10, 1)
    if blobs is not None:
        for b in blobs.itertuples(index=False):
            if tx <= b.x < tx + tsize and ty <= b.y < ty + tsize:
                cv2.circle(img_bgr, (int(b.x - tx), int(b.y - ty)), 12, YELLOW, 1)
    return img_bgr


def _sheet(tiles_imgs, captions, cols, cell):
    import cv2
    rows = int(np.ceil(len(tiles_imgs) / float(cols)))
    sheet = np.zeros((rows * (cell + 18), cols * cell, 3), np.uint8)
    for i, (im, cap) in enumerate(zip(tiles_imgs, captions)):
        r, c = divmod(i, cols)
        im = cv2.resize(im, (cell, cell), interpolation=cv2.INTER_AREA)
        sheet[r * (cell + 18):r * (cell + 18) + cell, c * cell:(c + 1) * cell] = im
        cv2.putText(sheet, cap[:48], (c * cell + 3, r * (cell + 18) + cell + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, WHITE, 1, cv2.LINE_AA)
    return sheet


def contact_sheets(out_dir, rows, n_accept=48, n_reject=24, cell=320):
    import cv2
    tiles = C.read_table(os.path.join(out_dir, "meta", "tiles.parquet"))
    frames = C.read_table(os.path.join(out_dir, "meta", "frames_all.parquet"))
    if tiles is None or frames is None:
        return {}
    aliases = [r["alias"] for r in rows]
    tiles = tiles[tiles.alias.isin(aliases)]
    png_of = dict(zip(frames.frame_id, frames.png))
    L = _labels_with_boxes(out_dir, rows)
    Lf = {k: g for k, g in L.groupby("frame_id")} if L is not None else {}
    import pandas as pd
    Bs = [FR.load_meta(out_dir, r["alias"], "blobs", I.segment_set(r)) for r in rows]
    Bs = [b for b in Bs if b is not None and len(b)]
    Bf = {k: g for k, g in pd.concat(Bs).groupby("frame_id")} if Bs else {}
    tsize = int(C.definitions()["tiles"]["size"])
    out = {}
    rng = np.random.default_rng(3)
    for kind, sel, n in (("accepted", tiles[tiles.decision == "accept"], n_accept),
                         ("rejected", tiles[tiles.decision == "reject"], n_reject)):
        if not len(sel):
            continue
        pick = sel.iloc[rng.choice(len(sel), size=min(n, len(sel)), replace=False)]
        gray_imgs, motion_imgs, caps = [], [], []
        for t in pick.to_dict(orient="records"):
            img = cv2.imread(os.path.join(out_dir, png_of[t["frame_id"]]), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            crop = img[t["ty"]:t["ty"] + tsize, t["tx"]:t["tx"] + tsize]
            g = cv2.cvtColor(crop[:, :, 0], cv2.COLOR_GRAY2BGR)
            gray_imgs.append(_draw_tile(g, t, Lf.get(t["frame_id"]), Bf.get(t["frame_id"]), tsize))
            motion_imgs.append(_draw_tile(crop.copy(), t, Lf.get(t["frame_id"]), Bf.get(t["frame_id"]), tsize))
            parts = t["tile_id"].split("__")
            caps.append("f%s %s %s %s" % (parts[-2], parts[-1], t["light_phase"],
                                          t.get("reason") or "fly %d rest %d" % (t["n_flying"], t["n_resting"])))
        if gray_imgs:
            p = os.path.join(out_dir, "qa", "sheets", "%s_tiles.png" % kind)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, _sheet(gray_imgs, caps, 8, cell))
            out[kind] = p
            if kind == "accepted":
                p2 = os.path.join(out_dir, "qa", "sheets", "accepted_tiles_motion.png")
                cv2.imwrite(p2, _sheet(motion_imgs, caps, 8, cell))
                out["accepted_motion"] = p2
    return out


def behaviour_strips(out_dir, rows, n_each=10, half=48, steps=(-25, -8, 0, 8, 25)):
    """5-frame crops (-25, -8, 0, +8, +25) around verified take-offs and landings and walking bouts."""
    import cv2
    out = {}
    rng = np.random.default_rng(11)
    for kind in ("take_off", "landing", "walk_bout"):
        strips = []
        for row in rows:
            seg = AN.seg_dir(out_dir, row["rig_id"], row["alias"])
            infos = dict(AN.segment_infos(out_dir, row["rig_id"], row["alias"], I.segment_set(row)))
            evs = []
            for stem in list(infos)[:6]:
                e = C.read_table(os.path.join(seg, stem + ".events.parquet"))
                if e is not None and len(e):
                    e = e[e.event == kind]
                    if kind != "walk_bout" and "verified" in e.columns:
                        e = e[e.verified.astype(bool)]
                    evs += [(stem, r) for r in e.to_dict(orient="records")]
            if not evs:
                continue
            for i in rng.choice(len(evs), size=min(max(1, n_each // max(1, len(rows))), len(evs)), replace=False):
                stem, ev = evs[int(i)]
                A = C.read_table(os.path.join(seg, stem + ".parquet"))
                tr = A[A.track_id == ev["track_id"]]
                reader = FR.Reader(infos[stem]["video"])
                crops = []
                try:
                    for s in steps:
                        v = int(ev["video_frame"]) + s
                        g = reader.read_gray(max(v, 0), 1)
                        if g is None:
                            break
                        im = cv2.cvtColor(g[0], cv2.COLOR_GRAY2BGR)
                        x0, y0 = int(ev["x"]) - half, int(ev["y"]) - half
                        pts = tr[tr.video_frame == v]
                        for p in pts.itertuples(index=False):
                            cv2.circle(im, (int(p.x), int(p.y)), 6, RED if p.cls == 1 else BLUE, 1)
                        pad = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT)
                        crop = pad[y0 + half:y0 + 3 * half, x0 + half:x0 + 3 * half]
                        crop = cv2.resize(crop, (4 * half, 4 * half), interpolation=cv2.INTER_NEAREST)
                        cv2.putText(crop, "%+d" % s, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
                        crops.append(crop)
                finally:
                    reader.close()
                if len(crops) == len(steps):
                    strip = np.hstack(crops)
                    cv2.putText(strip, "%s %s f%d %s" % (kind, row["alias"][:30], ev["video_frame"], ev.get("zone") or ""),
                                (3, strip.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
                    strips.append(strip)
        if strips:
            p = os.path.join(out_dir, "qa", "sheets", "strips_%s.png" % kind)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, np.vstack(strips))
            out[kind] = p
    return out


def clip_accuracy(out_dir, rows, n=3):
    import cv2
    res = []
    for row in rows:
        d = os.path.join(out_dir, "clips", row["rig_id"], row["alias"])
        if not os.path.isdir(d):
            continue
        for js in sorted(n for n in os.listdir(d) if n.endswith(".json"))[:n]:
            info = C.read_json(os.path.join(d, js))
            cap = cv2.VideoCapture(os.path.join(d, js[:-5] + ".mp4"))
            ok, fr = cap.read()
            cap.release()
            if not ok:
                continue
            c0 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            reader = FR.Reader(info["source_video"])
            try:
                src = reader.read_gray(max(info["start_frame"] - 1, 0), 3)
            finally:
                reader.close()
            if src is None:
                continue
            diffs = [float(np.abs(c0 - s.astype(np.float32)).mean()) for s in src]
            res.append({"clip": js[:-5], "mad_prev_same_next": [round(x, 2) for x in diffs],
                        "frame_accurate": int(np.argmin(diffs)) == 1})
    return res


# --------------------------------------------------------------------------- report
def report(out_dir, rows, log=C.log_default, do_alignment=True):
    import pandas as pd
    lines = ["# Label QA report", "",
             "Generated by `buzzsuite_cli.py yolo qa` (yolo_export %s). Definitions: `docs/definitions.json`." % C.GENERATOR_VERSION, ""]
    aliases = [r["alias"] for r in rows]
    if do_alignment:
        lines += ["## Alignment re-check", "",
                  "Median distance (px) from tracked points at tracker index = frame + offset to the nearest dark "
                  "blob, on the busiest 100 frames of 3 segments per experiment. Flying points must be closest at "
                  "**+2**; resting points are stationary, so offset doesn't matter for them.", ""]
        for row in rows:
            tab = alignment_check(row, out_dir)
            if not tab:
                continue
            best = min(tab, key=lambda r: r["flying_median_px"] if r["flying_n"] else 1e9)
            lines += ["**%s** — flying minimum at offset %+d (%s)" % (row["alias"], best["offset"],
                      "OK" if best["offset"] == 2 else "CHECK"), "",
                      "| offset | flying n | flying median px | flying < 5 px | resting median px |",
                      "|---:|---:|---:|---:|---:|"]
            for r in tab:
                lines.append("| %+d | %d | %.2f | %.2f | %.2f |" % (r["offset"], r["flying_n"], r["flying_median_px"],
                                                                   r["flying_within_5px"], r["resting_median_px"]))
            lines.append("")
            log("  alignment %-46s flying minimum at %+d" % (row["alias"], best["offset"]))
    L = _labels_with_boxes(out_dir, rows)
    if L is not None:
        lines += ["## Labels", "",
                  "| | n | supported | median support px | kept | kept but uncertain |", "|---|---:|---:|---:|---:|---:|"]
        names = {0: "resting", 1: "walking", 2: "flying", 3: "hold"}
        for m, g in L.groupby("motion"):
            lines.append("| motion %s | %d | %.1f%% | %.2f | %.1f%% | %.1f%% |" % (
                names.get(m, m), len(g), 100 * g.supported.mean(), g.support_dist.median(),
                100 * g.keep.mean(), 100 * g.uncertain.mean()))
        st1 = L[L.state == 1]
        if "mcontrast" in L and L.mcontrast.notna().all():
            fly = L[(L.cls == 1) & (L.keep | L.static_fly)]
            lines += ["", "- flying labels dropped for no motion of their own (policy.flying_note): %d of %d (%.1f%%); "
                      "%d of them sit on a dark blob with no resting label nearby (counted as unexplained blobs)"
                      % (fly.static_fly.sum(), len(fly), 100.0 * fly.static_fly.mean() if len(fly) else 0,
                         fly.orphan_fly.sum())]
        else:
            lines += ["", "- labels meta predates generator 1.1 (no motion contrast): run `yolo frames --force`"]
        rs = L[(L.cls == 0) & (L.keep | L.rest_sat)]
        lines.append("- resting labels dropped as one mosquito tracked twice (policy.rest_sat_note): %d of %d (%.1f%%)"
                     % (rs.rest_sat.sum(), len(rs), 100.0 * rs.rest_sat.mean() if len(rs) else 0))
        cap = C.definitions()["qa"]["population_max"]
        per = L.groupby("frame_id").keep.sum()          # frames with labels but none kept count as 0
        for alias, n in per.groupby(per.index.str.split("__").str[0]):
            lines.append("- %s: kept labels per frame, median %.0f (p90 %.0f); %.0f%% of frames above %d "
                         "(qa.population_note)" % (alias, n.median(), n.quantile(0.9), 100.0 * (n > cap).mean(), cap))
        lines += ["", "- flying-detector points rescued at tracker offset 0 (backward-pass tracks): %d of %d (%.2f%%)"
                  % ((st1.offset == 0).sum(), len(st1), 100.0 * (st1.offset == 0).mean() if len(st1) else 0),
                  "- labels filled from a neighbouring frame (gap <= 2): %.2f%%" % (100.0 * (L.dv != 0).mean()),
                  "- kept box size px (median w x h): resting %.1f x %.1f, flying %.1f x %.1f" % (
                      L[(L.cls == 0) & L.keep].bw.median(), L[(L.cls == 0) & L.keep].bh.median(),
                      L[(L.cls == 1) & L.keep].bw.median(), L[(L.cls == 1) & L.keep].bh.median()), ""]
    ev_rows = []
    for row in rows:
        ev = C.read_table(os.path.join(out_dir, "events", row["rig_id"], row["alias"] + ".parquet"))
        if ev is None or not len(ev):
            continue
        for kind, g in ev.groupby("event"):
            v = g.verified.astype(bool) if "verified" in g.columns else None
            on = g.stim_on.astype(bool) if "stim_on" in g.columns else None
            ev_rows.append("| %s | %s | %d | %s | %s |" % (
                row["alias"], kind, len(g),
                "%.0f%%" % (100 * v.mean()) if v is not None and kind in ("take_off", "landing", "zone_landing", "arrival") else "-",
                "%.0f%%" % (100 * on.mean()) if on is not None else "-"))
    if ev_rows:
        lines += ["## Events", "",
                  "Per experiment (annotated segments). Verified = site empties / fills in the tracker points "
                  "*and* in the pixels (docs/definitions.json, events.verified); train behaviour models on "
                  "verified events.", "",
                  "| experiment | event | n | verified | during stimulus |", "|---|---|---:|---:|---:|"] + ev_rows + [""]
    T = C.read_table(os.path.join(out_dir, "meta", "tiles.parquet"))
    rep = C.read_json(os.path.join(out_dir, "meta", "build_report.json")) or {}
    if T is not None:
        T = T[T.alias.isin(aliases)]
        lines += ["## Tiles", "", "| experiment | frames | tiles | accepted | empty kept | rejected | acceptance |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for a, g in T.groupby("alias"):
            nonempty = g[g.decision != "empty"]
            lines.append("| %s | %d | %d | %d | %d | %d | %.0f%% |" % (
                a, g.frame_id.nunique(), len(g), (g.decision == "accept").sum(), (g.selected & (g.decision == "empty")).sum(),
                (g.decision == "reject").sum(), 100.0 * (nonempty.decision == "accept").mean() if len(nonempty) else 0))
        lines += ["", "Objects written per split: `%s`" % rep.get("objects_by_split"),
                  "", "Main reject reasons: `%s`" % rep.get("reject_reasons"), ""]
    sheets = contact_sheets(out_dir, rows)
    strips = behaviour_strips(out_dir, rows)
    if sheets or strips:
        lines += ["## Sheets", "", "Boxes: red = flying, blue = resting, magenta = kept but not pixel-supported; "
                  "x = label dropped by the policy; yellow circle = "
                  "unexplained blob (likely missed mosquito). The motion sheet shows stack tiles as stored "
                  "(blue = t, green = t-1, red = t-2), so moving mosquitoes leave coloured trails.", ""]
        for k, p in list(sheets.items()) + list(strips.items()):
            lines.append("- %s: `%s`" % (k, os.path.relpath(p, out_dir)))
        lines.append("")
    ca = clip_accuracy(out_dir, rows)
    if ca:
        lines += ["## Clip frame accuracy", "",
                  "Mean abs difference of clip frame 1 vs source frames start-1, start, start+1 (lowest must be the middle).", ""]
        lines += ["- %s: %s %s" % (c["clip"], c["mad_prev_same_next"], "OK" if c["frame_accurate"] else "CHECK") for c in ca]
        lines.append("")
    p = os.path.join(out_dir, "qa", "report.md")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("  QA report: %s" % p)
    return p
