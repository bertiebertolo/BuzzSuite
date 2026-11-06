"""docs: README.md, docs/ROADMAP.md and docs/definitions.json in the output folder.

The README is regenerated on every run with the current counts, so the folder always explains
itself: what is there, how it was made, what the findings were, and how to run each stage.
"""
import glob
import os
import shutil

from yolo_export import common as C
from yolo_export import inventory as I

PLAN_FILE = os.path.join(os.path.expanduser("~"), ".claude", "plans", "i-have-my-raspberry-jiggly-honey.md")


def _count(pattern):
    return len(glob.glob(pattern))


def _status(out_dir):
    inv = C.read_json(os.path.join(out_dir, "inventory", "inventory.json")) or {}
    rows = [r for r in inv.get("rows", []) if r.get("chosen")]
    lines = ["| experiment | role | rig | segments | annotated | sampled frames | clips |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["role"] != "target", r["rig_id"], r["alias"])):
        segs = I.segment_set(r)
        a = sum(os.path.basename(p)[:-5] in segs for p in glob.glob(os.path.join(out_dir, "annotations", r["rig_id"], r["alias"], "*.json")))
        f = sum(os.path.basename(p)[:-10] in segs for p in glob.glob(os.path.join(out_dir, "frames", r["rig_id"], r["alias"], "*.png")))
        c = sum(os.path.basename(p)[:-10] in segs for p in glob.glob(os.path.join(out_dir, "clips", r["rig_id"], r["alias"], "*.mp4")))
        lines.append("| %s | %s | %s | %d | %d | %d | %d |" % (r["alias"], r["role"], r["rig_id"], r.get("n_unique", r["n_both"]), a, f, c))
    rep = C.read_json(os.path.join(out_dir, "meta", "build_report.json")) or {}
    if rep:
        lines += ["", "YOLO tiles written: **%s** (by split: `%s`; objects: `%s`)."
                  % (rep.get("tiles_written"), rep.get("written_by_split"), rep.get("objects_by_split"))]
    return "\n".join(lines)


README = """# BuzzWatch live detector — training data from BuzzWatch tracking

Auto-labelled training data, aligned annotations, behaviour events and clips for the **live
detection Pi** (Raspberry Pi 5 + AI HAT+ 2 / Hailo-10H). Every label comes from the frozen BuzzWatch
tracking output, checked against the video's own pixels — no manual annotation (AI_loop Handoff E).

Made by `buzzsuite_cli.py yolo ...` (package `yolo_export`, BuzzSuite). Everything here is
regenerable; the Buzzwatch data drives are only read. Last updated: {now}.

## What is here

{status}

| folder | contents |
|---|---|
| `inventory/` | `inventory.csv` (every experiment: rig, videos, segments, zones, stimulus, chosen or why not) and `selection.yaml` (your overrides) |
| `annotations/<rig>/<exp>/` | **every tracked segment with data attached**: `<segment>.parquet` (one row per detection, already aligned to the video frame), `.events.parquet`, `.activity.npz`, `.json` (points at the original `.mp4`) |
| `events/`, `timeseries/` | per experiment: all behaviour events (take-off, landing, zone pass / arrival / landing, walking bouts, swarm episodes; take-offs and landings carry `site_ok`, `px_ok` and **`verified`**), and per-second activity (resting / walking / flying / swarm counts), each with corrected clock time, `stim_on`, `stim_edge`, `light_phase` |
| `clock/` | the fitted recording clock per experiment (minute-40 burst fit, as fp_clock.py) |
| `frames/` | sampled full frames as 3-channel PNGs (B = frame t, G = t-1, R = t-2) |
| `meta/` | per-label pixel QA, per-frame info, tile decisions, `build_report.json` |
| `yolo/stack/`, `yolo/single/` | **the YOLO datasets** (640 tiles at native scale; classes 0 resting, 1 flying), each with `data.yaml` |
| `calib/` | 1,024 tiles for the Hailo INT8 calibration (Stage 3) |
| `clips/` | 10-s clips + `gt.txt` (MOTChallenge) + `labels.csv` + `events.csv` + `.json` |
| `qa/` | `report.md` (alignment check, label and tile statistics) and contact sheets / behaviour strips |
| `docs/` | `ROADMAP.md` (the full plan, Stages 0–5, as approved on 2026-10-22; where label rules differ, `definitions.json` and this README are current) and `definitions.json` (every threshold; the Pi app reads the same file) |

## Findings the labels rest on (measured 2026-10-22)

- **The tracker stores only blob centres**, so boxes come from the frame's own dark components
  (tracker's own WS1/WS5 reference + threshold from `images_mortality/`), else a class-median box.
- **Flying points are 2 frames late** in the tracker's frame index (the forward moving detector
  stores frame f-1's blob at index f+1); resting points are on time. Annotations are already
  aligned. On the 1080p rig flying points sit 2.3 px (median) from a blob at +2 and 31–52 px at
  -1..+1. The ~1 % of tracks recovered by the backward pass are at offset 0; frame QA rescues them.
- **Label support at 1080p**: ~98 % of resting points and ~80 % of real-flight points sit on a dark
  blob (median 0.5 px). Tracker "holds" (state 1, zero speed) are mostly empty and are dropped.
- **Flying labels need motion of their own** (found 2026-10-26, after the first training run): at
  night most tracker "flying" points were faint moving traces beside resting mosquitoes (legs /
  antennae), and passed pixel support on the resting mosquito's blob. A flying label is kept only
  if its spot is clearly darker at t than at t-1 / t-2 (`policy.flying_note` in definitions.json).
- **One resting mosquito, two labels** (found 2026-10-28 from the trained model's "stacked" boxes):
  the tracker often keeps a second long-lived resting track beside a resting mosquito, on its legs,
  a mesh hole or empty wall. Labels per frame ran 40-60 against about 30 mosquitoes per cage. A
  resting label next to a stronger one is dropped unless it has a body of its own
  (`policy.rest_sat_note`): 23 % (Aedes) and 30 % (Tbambusa) of kept resting labels.
- **Edge noise**: the resting tracker leaves stacks of false points along dark panel edges. They
  are dark only against the plain reference, not against a 3x3-min-filtered one, so labels below
  an edge-robust contrast of 6 are dropped (31 of 38 edge-stack labels; 2-3 % of clean labels).
- **Boxes**: one fixed box per class and experiment (about 16-20 px resting, 20-29 px flying at
  1080p), from high-contrast detections; thresholded blob sizes follow contrast, not the mosquito.
- **Take-offs and landings need checking**: about 1/3 of tracker take-offs leave a mosquito still
  sitting on the site (the assembly linked a passing flyer). Each event is checked in the tracker
  points (`site_ok`) and in the pixels (`px_ok`); about 30 % of take-offs and 20-25 % of landings
  pass both (`verified`). The two checks agree: ~1 % of `site_ok`-rejected events pass `px_ok`.
- **Segment gate**: segments whose labels per frame exceed 2x the experiment median (e.g. the
  first, one-segment Tbambusa session at 14:07 on 10-07: 250 vs 70) or with poor support are left
  out of the YOLO tiles.
- **The clock drifts** (~25.015 fps camera, 30,060-frame chunks): stimulus labels use the corrected
  clock; frames within 60 s of a window edge are flagged `stim_edge`.
- **Bug found, not fixed (frozen code)**: `calculate_cage_centroid` with the YAML's integer corners
  overflows int32 on Windows for 1920x1080 cages (gives (281, -39) instead of (861, 541)). The
  exporter passes floats; BuzzSwarm's `cage_geometry` on a 1080p rig would be affected.

## How to run (from `{app}`, with buzzwatch_env's python)

```
python buzzsuite_cli.py yolo inventory                       # scan drives, write inventory/ (seconds)
python buzzsuite_cli.py yolo all --workers 8                 # target rig: annotate, verify, clock, frames, build, clips, qa
python buzzsuite_cli.py yolo all -e <alias> ... --limit 5    # a pilot on a few segments per experiment
python buzzsuite_cli.py yolo build                           # re-tile with another policy (fast, no decoding)
python buzzsuite_cli.py yolo all --aux --workers 8           # the other rigs (fewer frames per segment)
```
Every action skips work already done (use `--force` to redo). Output defaults to this folder.

## Stage 0 — benchmark the AI HAT+ 2 (on the Pi, ~15 min)

On the Pi (Raspberry Pi OS Trixie with `hailo-h10-all` installed):
```
hailortcli fw-control identify            # confirms the Hailo-10H is seen
hailortcli benchmark <model>.hef          # FPS of a compiled model, e.g. yolov8n / yolov8s / yolo11s at 640
```
Stock Hailo-10H HEFs come from the Hailo Model Zoo (github.com/hailo-ai/hailo_model_zoo, Hailo-10H
model list) or the hailo-apps resources. We need **>= 100 tiles/s** (4 tiles x 25 fps) with headroom;
the result fixes the model size for Stage 2.

## Stage 2 preview — training (separate env, not set up yet)

```
yolo detect train data={out}/yolo/stack/data.yaml model=yolo11s.pt imgsz=640 \\
     hsv_h=0 hsv_s=0 hsv_v=0.2 mosaic=1.0 fliplr=0.5 flipud=0.5 epochs=100 batch=32
```
Train `stack` and `single`, compare on the test split (held-out days + Test_100Hz) and against
YOLito zero-shot (`G:\\YOLito\\weights\\best.pt`, single class: use `yolo/*_1class`). See `docs/ROADMAP.md`.
"""


def write_docs(out_dir, log=C.log_default):
    import datetime as _dt
    os.makedirs(os.path.join(out_dir, "docs"), exist_ok=True)
    C.write_json(os.path.join(out_dir, "docs", "definitions.json"), C.definitions())
    if os.path.isfile(PLAN_FILE):
        shutil.copy2(PLAN_FILE, os.path.join(out_dir, "docs", "ROADMAP.md"))
    text = (README.replace("{now}", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("{status}", _status(out_dir)).replace("{app}", C.APP_DIR)
            .replace("{out}", os.path.abspath(out_dir).replace("\\", "/")))
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)
    log("  docs: README.md, docs/ROADMAP.md, docs/definitions.json")
