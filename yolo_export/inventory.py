"""inventory: every experiment with tracking, its video folder, rig id and whether it is used.

Video folders are resolved in this order, keeping the first that shares segments with
final_tracking_data/: the JSON's folder_videos, the JSON's rebuilt Recording path, the same relative
path under this drive's Recording/ tree, and finally any Recording/ folder on any scanned root with
the most matching segment names (stale E:\\ and /Volumes JSONs, and Lightctrl analyses whose videos
sit under Recording/Phonotaxis/).

One analysis is kept per recording (e.g. AedesAegyptiLiv475M_6_4-5 / _overhaul / _overhaul_split):
the adopted overhaul + blob-split settings first, then the most usable segments, then the newest
tracking. inventory/selection.yaml (written with defaults on first run) overrides any of it.
"""
import json
import os

from yolo_export import common as C

SELECTION_FILE = "selection.yaml"
STIM_ASSAYS = ("phonotaxis", "100hz", "test_100hz", "repulsion")

SELECTION_TEMPLATE = """\
# yolo_export selection overrides. Edit, then re-run `buzzsuite_cli.py yolo inventory`.
# Aliases are the experiment aliases in inventory.csv.

# Rig the live Pi watches. Its experiments are exported by default; the rest are 'aux' (--aux).
target_rig: Penzance_1920x1080

# Force experiments in or out (aliases).
include: []
exclude: []

# Stimulus windows (minutes of the hour) per alias. Default: 40-50 for Phonotaxis / 100Hz /
# Test_100Hz / Repulsion assays, none otherwise. Example:  Penzance_Aedes_M_20260911_20260916: "40-50"
stim_windows: {}

# Borrow another experiment's custom_zones.json (alias: other_alias). Without an entry, an
# experiment with no zones borrows them from a chosen experiment on the same rig whose cage
# corners are within 15 px, and that is flagged in inventory.csv (zones_source).
zones_from: {}

# Recording-clock rate per rig when the minute-40 burst fit is not possible (fp_clock fallback).
# Measured 2026-10-20 on the Penzance rigs: 25.0147-25.0155 fps.
rig_fps:
  Penzance_1920x1080: 25.015
  Penzance_1280x720: 25.015

# Experiments whose segments all go to the test split (default: assays starting with "Test").
test_experiments: []
# Held-out test days per alias (YYYY-MM-DD). Default: the last calendar day with >= 60 segments.
test_days: {}
"""


def _selection(out_dir):
    import yaml
    path = os.path.join(out_dir, "inventory", SELECTION_FILE)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(SELECTION_TEMPLATE)
    with open(path, "r", encoding="utf-8") as f:
        sel = yaml.safe_load(f) or {}
    defaults = yaml.safe_load(SELECTION_TEMPLATE)
    for k, v in defaults.items():
        if sel.get(k) is None:
            sel[k] = v
    return sel


def load_selection(out_dir):
    return _selection(out_dir)


def _list_mp4_stems(folder):
    try:
        names = os.listdir(folder)
    except OSError:
        return set()
    return set(os.path.splitext(n)[0] for n in names
               if n.lower().endswith(".mp4") and not n.startswith("._"))


def _tracked_stems(exp_dir):
    d = os.path.join(exp_dir, "final_tracking_data")
    try:
        names = os.listdir(d)
    except OSError:
        return set(), 0.0
    stems, newest = set(), 0.0
    for n in names:
        if n.startswith(C.TRACK_PREFIX) and not n.startswith("._"):
            stems.add(n[len(C.TRACK_PREFIX):])
    if stems:
        try:
            newest = max(os.path.getmtime(os.path.join(d, C.TRACK_PREFIX + s)) for s in list(stems)[:50])
        except OSError:
            newest = 0.0
    return stems, newest


def _recording_index(roots, log):
    """[(folder, set(mp4 stems))] for every folder holding .mp4 files under each root's Recording/."""
    out = []
    for root in roots:
        rec = os.path.join(root, "Recording")
        for dp, dn, fn in os.walk(rec):
            dn[:] = [d for d in dn if not d.startswith(".")]
            stems = set(os.path.splitext(n)[0] for n in fn
                        if n.lower().endswith(".mp4") and not n.startswith("._"))
            if stems:
                out.append((dp, stems))
    log("  recording folders with .mp4: %d" % len(out))
    return out


def _experiment_dirs(root):
    """(exp_dir, json_path or None) for every experiment folder under root/Analysis: folders with
    an experiment_*.json, and re-tracking folders that only hold settings + final_tracking_data
    (e.g. AedesAegyptiLiv475M_6_4-5_overhaul_split, made by the overhaul batch scripts)."""
    found = []
    for dp, dn, fn in os.walk(os.path.join(root, "Analysis")):
        has_tracks = "final_tracking_data" in dn
        dn[:] = [d for d in dn if d not in C.HEAVY_SUBDIRS and not d.startswith(".")]
        js = sorted(n for n in fn if n.startswith("experiment_") and n.endswith(".json")
                    and not n.startswith("._"))
        if js:
            plain = [n for n in js if " copy" not in n.lower()]
            found.append((dp, os.path.join(dp, (plain or js)[0])))
        elif has_tracks and "buzzwatch_track_settings.yml" in fn:
            found.append((dp, None))
    return found


def _probe(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS) or 0.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


def _row(root, exp_dir, json_path, rec_index):
    meta = (C.read_json(json_path, default={}) or {}) if json_path else {}
    rel_parts = os.path.relpath(exp_dir, os.path.join(root, "Analysis")).split(os.sep)
    assay = meta.get("experiment_name") or (rel_parts[0] if rel_parts else "")
    location = meta.get("cage_name") or (rel_parts[1] if len(rel_parts) > 1 else "")
    batch = meta.get("batch_name") or os.path.basename(exp_dir)
    if json_path:
        alias = meta.get("experiment_alias") or os.path.basename(json_path)[len("experiment_"):-len(".json")]
        if json_path and meta.get("batch_name") and os.path.basename(exp_dir) != meta["batch_name"]:
            alias = "%s_%s" % (location, os.path.basename(exp_dir))   # a copied JSON in a re-run folder
    else:
        alias = "%s_%s" % (location, os.path.basename(exp_dir))
        json_path = ""
    tracked, newest = _tracked_stems(exp_dir)
    row = {"alias": alias, "exp_dir": exp_dir, "root": root, "assay": assay, "location": location,
           "batch": batch, "json": json_path, "n_tracked": len(tracked), "newest_pkl_mtime": newest,
           "video_dir": "", "video_source": "", "n_videos": 0, "n_both": 0}

    cands = []
    if meta.get("folder_videos"):
        cands.append(("json", meta["folder_videos"]))
    if meta.get("root_folder_videos") and meta.get("experiment_name") and meta.get("cage_name") and meta.get("batch_name"):
        cands.append(("json_rebuilt", os.path.join(meta["root_folder_videos"], meta["experiment_name"],
                                                   meta["cage_name"], meta["batch_name"])))
    cands.append(("same_root", os.path.join(root, "Recording", *rel_parts)))
    if meta.get("experiment_name") and meta.get("cage_name") and meta.get("batch_name"):
        cands.append(("same_root_meta", os.path.join(root, "Recording", meta["experiment_name"],
                                                     meta["cage_name"], meta["batch_name"])))
    best = None
    for src, folder in cands:
        if folder and os.path.isdir(folder):
            stems = _list_mp4_stems(folder)
            n = len(stems & tracked)
            if n and (best is None or n > best[2]):
                best = (src, folder, n, len(stems))
                break
    if best is None and tracked:
        for folder, stems in rec_index:
            n = len(stems & tracked)
            if n and (best is None or n > best[2]):
                best = ("name_match", folder, n, len(stems))
    if best:
        row.update(video_source=best[0], video_dir=os.path.normpath(best[1]), n_both=best[2], n_videos=best[3])
    return row, tracked


def _cage_close(a, b, tol):
    try:
        return len(a) == len(b) and float(abs(a - b).max()) <= tol
    except Exception:
        return False


def build_inventory(out_dir, roots, log=C.log_default):
    """Scan, resolve, choose; write inventory/inventory.csv + inventory.json. Returns the rows."""
    import numpy as np
    import pandas as pd
    sel = _selection(out_dir)
    log("Scanning %d data root(s): %s" % (len(roots), ", ".join(roots)))
    rec_index = _recording_index(roots, log)
    rows, tracked_by = [], {}
    for root in roots:
        for exp_dir, json_path in _experiment_dirs(root):
            r, tracked = _row(root, exp_dir, json_path, rec_index)
            rows.append(r)
            tracked_by[exp_dir] = tracked
    log("  experiment folders: %d" % len(rows))

    for r in rows:
        r.update(width=0, height=0, fps=0.0, rig_id="", cage_width_px=float("nan"), cage_points="",
                 has_zones=False, zone_names="", has_schedule=False, overhaul=False, split_blobs=False,
                 backward_pass=False, chosen=False, role="", reason="", multi_cage=False,
                 stim_windows="", zones_source="", zones_dir="", test_experiment=False)
        try:
            settings = C.load_settings(r["exp_dir"])
            pts = settings.get("cage_border_points") or []
            tr = settings.get("tracking") or {}
            r.update(overhaul=bool(tr), split_blobs=bool(tr.get("split_blobs")),
                     backward_pass=bool(tr.get("use_backward_pass")),
                     cage_points=json.dumps(pts))
            if len(pts) >= 3:
                g = C.load_geometry(r["exp_dir"])
                r["cage_width_px"] = round(g["width"], 1)
                r["has_zones"] = bool(g["zones"])
                r["zone_names"] = ",".join(g["zones"])
        except Exception as exc:  # noqa: BLE001
            r["reason"] = "settings unreadable: %s" % exc
        r["has_schedule"] = os.path.isfile(os.path.join(r["exp_dir"], "ld_dd_schedule.json"))
        if r["video_dir"] and r["n_both"]:
            stem = sorted(tracked_by[r["exp_dir"]] & _list_mp4_stems(r["video_dir"]))[0]
            pr = _probe(os.path.join(r["video_dir"], stem + ".mp4"))
            if pr:
                r.update(width=pr[0], height=pr[1], fps=round(pr[2], 3))
                r["rig_id"] = "%s_%dx%d" % (r["location"], pr[0], pr[1])

    # ---- one analysis per recording (or flag several cages in one video)
    by_video = {}
    for r in rows:
        if r["video_dir"] and r["n_both"] and r["cage_points"] and not r["reason"]:
            by_video.setdefault(os.path.normcase(r["video_dir"]), []).append(r)
        elif not r["reason"]:
            r["reason"] = "no tracked segments with video" if r["n_tracked"] else "not tracked"
    for group in by_video.values():
        cages = [np.asarray(json.loads(r["cage_points"]), float) for r in group]
        tol = 0.25 * max(float(np.nanmax([r["cage_width_px"] for r in group]) or 1.0), 1.0)
        distinct = []
        for i, r in enumerate(group):
            if not any(_cage_close(cages[i], cages[j], tol) for j in distinct):
                distinct.append(i)
        if len(distinct) > 1:
            for r in group:
                r["multi_cage"] = True
                r["reason"] = "video holds %d cages: merged labels are not in v1" % len(distinct)
            continue
        group.sort(key=lambda r: (r["split_blobs"] and r["backward_pass"], r["n_both"], r["newest_pkl_mtime"]),
                   reverse=True)
        group[0]["chosen"] = True
        for r in group[1:]:
            r["reason"] = "duplicate analysis of the same videos as %s" % group[0]["alias"]

    # ---- overrides, roles, stimulus, zones, test experiments
    inc, exc = set(sel.get("include") or []), set(sel.get("exclude") or [])
    for r in rows:
        if r["alias"] in exc:
            r["chosen"], r["reason"] = False, "excluded in selection.yaml"
        elif r["alias"] in inc and r["n_both"] and r["rig_id"]:
            r["chosen"], r["reason"] = True, "included in selection.yaml"
        if r["chosen"]:
            r["role"] = "target" if r["rig_id"] == sel.get("target_rig") else "aux"
        stim = (sel.get("stim_windows") or {}).get(r["alias"])
        if stim is None:
            stim = "40-50" if str(r["assay"]).lower() in STIM_ASSAYS else ""
        r["stim_windows"] = stim
        tests = set(sel.get("test_experiments") or [])
        r["test_experiment"] = (r["alias"] in tests) or (not tests and str(r["assay"]).lower().startswith("test"))
    # one owner per recording segment: the same .mp4 (name + size) can sit in several folders
    # (e.g. Test_100Hz/Aedes_1 and Aedes_2 are copies of segments of the main 100Hz session); the
    # biggest experiment keeps it, so no video can land in two splits
    for r in rows:
        r["dup_segments"], r["n_unique"] = "[]", r["n_both"] if r["chosen"] else 0
    claimed = {}
    for r in sorted([r for r in rows if r["chosen"]],
                    key=lambda r: (r["n_both"], r["split_blobs"] and r["backward_pass"], r["newest_pkl_mtime"]),
                    reverse=True):
        dups = []
        for s in sorted(tracked_by[r["exp_dir"]] & _list_mp4_stems(r["video_dir"])):
            try:
                key = (s, os.path.getsize(os.path.join(r["video_dir"], s + ".mp4")))
            except OSError:
                key = (s, -1)
            if key in claimed:
                dups.append(s)
            else:
                claimed[key] = r["alias"]
        r["dup_segments"], r["n_unique"] = json.dumps(dups), r["n_both"] - len(dups)
        if dups and not r["n_unique"]:
            owners = sorted(set(claimed[(s, os.path.getsize(os.path.join(r["video_dir"], s + ".mp4")))] for s in dups[:20]))
            r["chosen"], r["role"] = False, ""
            r["reason"] = "every segment duplicates %s" % ", ".join(owners)

    chosen = [r for r in rows if r["chosen"]]
    zones_from = sel.get("zones_from") or {}
    for r in chosen:
        if r["has_zones"]:
            r["zones_source"], r["zones_dir"] = "own", r["exp_dir"]
            continue
        donor, gap = None, float("nan")
        mine = np.asarray(json.loads(r["cage_points"]), float)
        if r["alias"] in zones_from:
            donor = next((d for d in rows if d["alias"] == zones_from[r["alias"]] and d["has_zones"]), None)
        else:
            for d in chosen:
                if d is not r and d["has_zones"] and d["rig_id"] == r["rig_id"] and \
                        _cage_close(mine, np.asarray(json.loads(d["cage_points"]), float),
                                    0.03 * r["cage_width_px"]):
                    donor = d
                    break
        if donor is not None:
            theirs = np.asarray(json.loads(donor["cage_points"]), float)
            if len(theirs) == len(mine):
                gap = float(abs(theirs - mine).max())
            r["zones_source"] = "borrowed:%s (cage corners within %.0f px)" % (donor["alias"], gap)
            r["zones_dir"] = donor["exp_dir"]

    os.makedirs(os.path.join(out_dir, "inventory"), exist_ok=True)
    df = pd.DataFrame(rows)
    order = ["alias", "chosen", "role", "rig_id", "assay", "location", "batch", "n_both", "n_unique", "n_tracked",
             "n_videos", "width", "height", "fps", "cage_width_px", "has_zones", "zone_names",
             "zones_source", "stim_windows", "has_schedule", "test_experiment", "overhaul",
             "split_blobs", "backward_pass", "multi_cage", "reason", "exp_dir", "video_dir",
             "video_source", "zones_dir", "json", "root", "cage_points", "newest_pkl_mtime", "dup_segments"]
    df = df[order].sort_values(["chosen", "role", "rig_id", "alias"], ascending=[False, False, True, True])
    df.to_csv(os.path.join(out_dir, "inventory", "inventory.csv"), index=False)
    C.write_json(os.path.join(out_dir, "inventory", "inventory.json"),
                 {"generator_version": C.GENERATOR_VERSION, "roots": roots, "selection": sel,
                  "rows": df.to_dict(orient="records")})
    n_t = sum(1 for r in chosen if r["role"] == "target")
    log("  chosen: %d experiments (%d on the target rig %s, %d auxiliary)"
        % (len(chosen), n_t, sel.get("target_rig"), len(chosen) - n_t))
    for r in sorted(chosen, key=lambda r: (r["role"] != "target", r["rig_id"], r["alias"])):
        log("    %-6s %-22s %-48s %4d segments%s  zones=%s stim=%s%s"
            % (r["role"], r["rig_id"], r["alias"], r["n_unique"],
               " (+%d duplicates skipped)" % (r["n_both"] - r["n_unique"]) if r["n_both"] != r["n_unique"] else "",
               r["zones_source"] or "-", r["stim_windows"] or "-", "  [test]" if r["test_experiment"] else ""))
    for r in rows:
        if not r["chosen"] and str(r.get("reason", "")).startswith("every segment duplicates"):
            log("    skip   %-22s %-48s %s" % (r["rig_id"], r["alias"], r["reason"]))
    return df.to_dict(orient="records")


def load_inventory(out_dir):
    data = C.read_json(os.path.join(out_dir, "inventory", "inventory.json"))
    return data["rows"] if data else None


def chosen_rows(rows, rig=None, aux=False, experiments=None):
    """Chosen inventory rows for one run: explicit aliases, else a rig, else target (or aux) role."""
    out = [r for r in rows if r.get("chosen")]
    if experiments:
        want = set(experiments)
        return [r for r in out if r["alias"] in want]
    if rig:
        return [r for r in out if r["rig_id"] == rig]
    return [r for r in out if r["role"] == ("aux" if aux else "target")]


def segment_set(row):
    """The experiment's current segments as a set (for filtering outputs of earlier runs)."""
    return set(segments_of(row))


def segments_of(row):
    """Sorted segment stems with both tracking and a video, minus duplicates owned elsewhere."""
    tracked, _ = _tracked_stems(row["exp_dir"])
    dups = set(json.loads(row.get("dup_segments") or "[]"))
    return sorted((tracked & _list_mp4_stems(row["video_dir"])) - dups)
