# BuzzSuite DEVLOG (fresh start 2026-11-06)

The log up to 2026-09-25 is in `G:\BuzzSuite\DEVLOG.md`; entries from 2026-09-25 to 2026-11-05 were on
`D:\BuzzSuite\DEVLOG.md`, which is lost (the Claude Code transcripts under `~\.claude\projects\d--BuzzSuite` still
hold the work itself).

## 2026-11-06 — D: gone; main moved to G:\BuzzSuite-github; yolo_export rebuilt

- `yolo_export` existed only on D:. Rebuilt by replaying, in time order, every Write/Edit/MultiEdit tool call and the
  six shell edits recorded in the Claude Code transcripts (145 events, 0 failed after normalising CRLF line endings,
  which the shell edits had introduced). The one shell edit that errored at the time (10-22 12:47) is left out.
- Verification: `label_boxes()` on the existing meta reproduces `build_report.json` exactly for all three target
  experiments (labels, kept, merged, static flying, orphan flying, resting satellites). A 2-segment smoke run of
  annotate / verify / frames on a 720p June experiment passed.
- Live detector: started the aux export of the six Penzance 720p experiments (June phonotaxis x3 overhaul_split,
  Light_Test M 07-04, Repulsion F 250 Hz 07-11, piezo LD 4/23-25), 6 frames per segment, then a full re-tile.
  1.2.0 tables backed up to `G:\BuzzWatch_LiveDetector\meta\backup_1.2.0`.

## 2026-11-06 — Live detector: gold set (1,000 crops judged by eye)

`training/gold_set.py`, `label.html`, `gold_analysis.py` -> `gold_set/gold_report.md`. 800 v12 resting boxes
(80 per confidence bin per test experiment) + 200 tracker labels v12 missed; blind, random order.
- v12 resting precision by eye: Aedes 0.65, Tbambusa 0.45, and flat across confidence (0.75+ boxes are no
  better than 0.25-0.35). A confidence cut or a top-30 rule cannot fix the count.
- Real resting found per frame: Aedes 25.1 (+3.1 missed) ~ 28; Tbambusa 17.8 (+3.3) ~ 21: recall vs eye ~85-90%.
  False boxes per frame: 13.6 / 22.1, mostly junk, then shadows, few parts.
- Boxes matching a tracker label are only 76% / 54% real; tracker labels v12 missed are only 44% / 30% real.
  The tracker labels carry many phantoms and v12 learned them: the fix is cleaner training labels.

## 2026-11-06 — yolo_export 1.3.0: junk filter + grey-out; v13 started

- `yolo_export/junk_filter.py`: HistGradientBoosting (depth 2, 150 iters) on rig-independent tracker meta
  (supported, contrast, rcontrast, mcontrast, edge_suspect, comp w/h/area in cage widths, comp_nlabels, track
  span as a fraction of the segment's sampled frames), trained on 679 gold items that sit on a tracker resting
  label (394 real, 285 false). Grouped-CV AUC 0.86; threshold 0.442 keeps 85% of real labels, removes 71% of
  false. Model + threshold: `junk_filter_model.pkl`, `junk_filter.json` (trainer:
  `G:\BuzzWatch_LiveDetector\training\gold_train_filter.py`). Contrast alone: AUC ~0.80; position/zone left out.
- `build.py`: kept resting labels scored below the threshold are dropped (`junk`), and their spot is greyed out in
  the tile (1.4 x resting box, filled with the median of a 4-px ring) unless it touches a kept label's box.
  On the existing meta it drops 27% (Aedes M), 48% (Tbambusa) and 41% (Aedes_2) of kept resting labels, in line
  with the gold set's false shares. Policy keys: junk_filter, junk_grey_scale, junk_grey_ring_px.
- Aux export: the piezo LD4_23-25 experiment (960x720) is left out: loading its .pkl files crashes py3.8 with
  an access violation, which silently hung the worker pool twice (pool_run waits forever on a dead worker).
- `training/run_v13.cmd`: aux export of the five 720p experiments -> build 1.3.0 -> mirror stack to
  C:\yolo_data_v13 (C:\yolo_data keeps v12's set) -> train yolov8s_stack_v13 (v12's args) -> test val of v13
  and of v12 on the 1.3.0 labels. Judge v13 mainly on the gold set (raw frames), since test tiles are greyed too.

## 2026-11-06 — fixes for the cloud review of PR #3 (10 findings), not committed yet

- Phonotaxis tab: the run folder is claimed (`.buzzphono_run` marker, `buzzphono_bridge.claim_run_dir`) before
  process.log goes in, so BuzzPhono accepts it (before, every Analyse failed). A run whose process is gone while
  status.json still says "running" (killed after Cancel, crash, OOM) now ends the wait and re-enables Analyse.
- BuzzPhono app: `start_job` runs `pipeline.preflight` before claiming the folder (refusals show on the page,
  nothing is written); `common.job_status` reports a dead run as failed; `stop_job` records "cancelled" after a
  hard kill (`jobs.mark_ended`).
- BuzzPhono pipeline: the light schedule resolved from the Pi config.ini (light.source auto) is used by every
  analysis and figure (ZT profile, flight, sensitivity), recorded per experiment in the manifest (`light`),
  used by `redraw`, and a warning says when it differs from the settings.
- `buzzsuite_cli.py yolo <action>` exists (calls `yolo_export.runner.run`).
- yolo_export: `junk_filter_model.pkl` is un-ignored (.gitignore) so it ships; a missing or unloadable model stops
  the build with a clear message (it loads only with scikit-learn 1.3.2 = buzzwatch_env, not buzzsuite_env's
  1.9); build_report.json records `junk_filter.applied`. Calibration tiles come only from tiles actually written
  (tiles.parquet gains `written`). Experiments with no cage border get reason "no cage border drawn"; one bad
  experiment no longer stops job building. Stems with the Pi's `-<attempt>` suffix parse (the attempt stays in
  the session key); older annotate/frames meta with null session/k is repaired from the name. Tracking pickles
  load through an allow-listed unpickler (`common.load_tracking`); identical data to the old loader on 9 real
  files. selftest runs its box-rule checks with the junk filter off (it failed before on synthetic labels).
- Checked: BuzzPhono ruff/format/mypy clean, 957 CI tests pass; tab run headless against real BuzzPhono on an L2
  synthetic experiment (done, and recovery after a hard kill); yolo selftest passes in buzzwatch_env.
- Still open (known, see 1.3.0 entry): numpy-2-era pickles (e.g. piezo LD4_23-25) crash py3.8 / numpy 1.24
  with an access violation in plain pickle.load too, which can hang pool_run on a dead worker.
