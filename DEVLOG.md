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

## 2026-11-08 — v13 training failed; two fixes (missed review fixes, numpy 2.4)

- run_v13.cmd: aux export, the 1.3.0 build (201 min) and the copy to C:\yolo_data_v13 finished; training failed at
  the end of epoch 1 (`np.trapz` is gone in numpy 2.4.6, installed in yolo_env on 10-29 20:43, after v12; Ultralytics
  8.3.40 still calls it). Epoch 1 also took 72 min, not ~5: data.yaml carried `path: G:/...`, so it trained off USB.
- Cause of the path: the transcript replay had missed `apply_review_fixes.py` (10-22 12:48, 15 edits written to a
  scratchpad script and run from the shell): relocatable data.yaml, pool terminate on error, test_days
  normalisation and one-split-per-stem check, cage-polygon requirement, raw sort-order guard, numpy-bool json,
  clock schedule warning. Replayed it (0 later edits fail), patched it into G:\BuzzSuite-github\yolo_export
  (one runner hunk skipped: an equivalent job_geometry guard was already there). Label rules re-checked with the
  junk filter off: identical to the 1.2.0 build_report. Copy before the patch: G:\BuzzWatch_LiveDetector\code\.
  Fixed the three data.yaml files already written. Still open: pool_run hangs when a worker dies hard.
- training/train_v13.py + train_v13.cmd: train (v12's args) via the Python API with np.trapz aliased, test-val of v13
  and of v12 on the 1.3.0 labels, gold_eval_model.py, eval_centres. The environment itself is unchanged.

## 2026-11-08 — v13 results (1.3.0 junk filter + 5 Penzance 720p aux experiments)

Trained 60 epochs (08:20-16:18, ~8 min/epoch from C:). Gold set (raw test frames, conf 0.25), v12 -> v13:
- resting boxes per frame: Aedes 39.1 -> 29.6, Tbambusa 40.4 -> 20.5 (gold-set estimate of real resting: ~28 / ~21).
- junk boxed 0.62 -> 0.21 / 0.75 -> 0.13; shadows 0.83 -> 0.26 / 0.82 -> 0.36; parts 0.40 -> 0.20 / 0.52 -> 0.07.
- real v12 boxes still boxed 0.97 -> 0.86 / 0.97 -> 0.70; real mosquitoes v12 missed now boxed 0.14 -> 0.42 / 0.03 -> 0.23.
Test split on the 1.3.0 labels (eval_centres, 8 px): resting F1 0.659 -> 0.769, flying 0.832 -> 0.836, resting count
bias +8.9 -> +1.8 per frame, MAE 9.4 -> 3.6, r 0.33 -> 0.73; night flying F1 0.571 -> 0.583.
Demo (demo/stack_v13_demo.mp4) resting per frame v12 -> v13: 22.6 -> 18.0, 25.3 -> 13.3, 36.2 -> 30.1, 37.8 -> 18.6.
Open: Tbambusa real-box retention 0.70 (the filter's 15% real loss plus grey-out); HEF compile + chip check for v13.

## 2026-11-10 — gold round 2; v14 started (resting relabelled by v13 + tracker agreement)

Round 2 (`training/gold_set2`, 500 crops judged 11-10): v13-only boxes 91% / 84% real (Aedes / Tbambusa) vs
v12-only 32% / 36%. Junk filter on 720p aux cages: kept 80% real, dropped 26% real (generalises). Near its threshold
labels are 40-55% real. Retraining the filter on rounds 1+2 (1,131 labels): grouped-CV AUC 0.821 -> 0.829, removes
60 -> 63% of false at 85% real kept: tracker meta is near its ceiling; the retrained model is NOT switched on.
v14 (`training/pseudo_v14.py`, `run_v14.cmd`): train/val tiles re-cut from raw frames; resting = v13 >= 0.40, or
v13 0.25-0.40 agreeing (8 px) with a 1.3.0 tracker label; one-source spots greyed; flying = tracker. A v13-only
first version taught real 720p mosquitoes v13 missed as background (seen on a tile check) and was replaced.
Test split = v13's (1.3.0). Scoring: gold rounds 1 and 2 (gold_eval_model.py, gold2_eval_model.py) + eval_centres.
Launch note: a window started with Start-Process from a tool call got the tool's Ctrl-C when the call timed out;
launch long jobs with Win32_Process Create (detached).
- 12:08: held out a whole experiment for an unseen test: Penzance_Aedes_aegypti_F_250hz_20260711_20260714 (females,
  repulsion, 720p; 5,863 tiles) removed from v14's training list (`v14_tiles.csv`; full list `v14_tiles_all.csv`); its
  3,907 tiles already relabelled were moved to C:\yolo_data_v14_heldout_moved. `holdout_eval.py` scores models on its
  1,350 complete frames vs the 1.3.0 tracker labels (5.3 px at 720p). Baseline v12: resting 44.6 / frame vs 16.4
  reference (F1 0.374), flying F1 0.854. v13 trained on it, so only v12 vs v14 is a fair comparison there.

## 2026-11-11 -- v14 results (live detector, G:\BuzzWatch_LiveDetector\training)
The first relaunch (11-10 12:09) died in the relabel at 12:14 with ^C (cause not found); relaunched 11-11 00:26 by
`launch_v14_when_gpu_free.ps1` (waits until < 3 GB GPU used). Relabel 39,945 tiles (13 min), train 60 epochs
00:40-08:55, all scoring done 09:32.
- Gold round 1 (real kept / junk boxed / resting per frame), v13 -> v14: Aedes 0.86 -> 0.90 / 0.21 -> 0.18 / 29.6 -> 30.6;
  Tbambusa 0.70 -> 0.77 / 0.13 -> 0.13 / 20.5 -> 22.7 (true about 28 / 21). Shadow boxed 0.26 -> 0.24, 0.36 -> 0.25.
- Gold round 2: v14 boxes 56-57% of the real v12-only items with 3-8% of the false ones; it keeps 75% / 83% of the
  real v13-only items (v13 100% by construction).
- Test split (eval_centres, conf 0.21, 8 px): resting F1 0.760 (v13 0.769), flying 0.834, night flying 0.570;
  resting count bias +1.4 / frame (v13 +1.8), MAE 3.6.
- Unseen F 250 Hz run (vs 1.3.0 tracker labels, 16.4 resting / frame): v12 44.6 / F1 0.374 -> v14 26.9 / F1 0.522;
  flying 0.854 -> 0.859. (v13 trained on it: 14.0 / 0.565.) The reference keeps only ~80% of real 720p resting
  mosquitoes (round 2), so part of v14's excess may be real; a small gold set on this run would settle it.
- Demo `demo/stack_v14_demo.mp4`: user: "much better", but resting boxes flash. Added `demo/stabilize.py` (resting
  tracks: shown after 5 detections >= 0.30, kept alive down to 0.20, held 6 frames; EMA box; flying untouched), tuned
  over 72 settings on cached v14 detections. Demo clips: resting boxes vanishing per second 7.8/25.8/20.4/36.1 ->
  0.7/1.3/1.8/2.9, mean count vs raw -0.3..+0.8 / frame. `demo/stack_v14_smooth_demo.mp4`. Not yet in the Pi app.
- 11-12: v14 compiled for the Hailo-10H with v12b's options: `hailo/stack_v14/yolov8s_stack_v14.hef` (README there).
  Emulator on 1,195 1.3.0 test tiles: F1 any/resting/flying 0.831/0.792/0.861 (PyTorch) -> 0.826/0.790/0.847 (drop
  0.005/0.002/0.014; v12b 0.005/0.003/0.012); passes. On-chip kit in the same folder (pi_parity.py + the 1,195 tiles);
  chip check not run yet.
- 11-16: PC -> Pi over Tailscale (`ssh buzzpi`, user penzance, 100.110.112.23, key login). v14 HEF on the chip
  (1,195 tiles): F1 any/resting/flying 0.823/0.790/0.841 vs PyTorch 0.831/0.792/0.861 (drop 0.008/0.002/0.020; v12b
  chip drop 0.005/0.002/0.016); chip-emulator agreement 0.981; 37.3 ms per sequential batch of 4 (= v12b). Passes,
  flying right at the 0.02 line. No live-detector app on the Pi yet (only ~/hailo_bench); PiPeltier there is the
  separate temperature controller.

## 2026-11-19 — BuzzSwarm swarm-zone radius; BuzzPhono wall control zones; feeder-zone checks

- `517a3b7` BuzzSwarm: a "Swarm zone…" button (`swarm_zone_dialog.py`) sets the swarm-zone radius on a camera frame
  of the cage (slider, number box or drag; shows the share of cage area). Saved per experiment in
  `plots/buzzswarm/swarm_zone.json` (`clock_analysis.save_swarm_radius` / `load_swarm_radius`); `run()` uses the
  experiment dict's `swarm_radius_px`, else the saved radius, else 100 px. Day caches do not depend on it. The
  Swarming tab's controls and plots are split by a draggable divider ("Continuous" no longer cut off).
  Headless check, 15/15: save/load and bad values rejected; `run()` order dict > saved > default; on
  Penzance_Aedes_aegypti_M_20260704_20260707 (2026-07-05 day cache) radius 100 reproduces the saved windows.csv
  `swarm_mean` exactly and 200 gives larger values; dialog drag/save/reopen; tab builds with stub managers.
- `a2c9603` BuzzPhono: Calibrate starts with the speaker zone; the control zones are copies of it on the other three
  walls (`calibration.wall_controls`, cage corners from `buzzwatch_track_settings.yml` via `cage_from_settings`),
  tests in `tests/buzzphono/unit/test_wall_controls.py`. buzzphono_env's editable install points at
  `G:\BuzzSuite-github` again (it was on the `swarm-plots` worktree).
- Branch `feeder-tracking-check` (not merged): `tools/feeder_check/`, read-only diagnostics for zones that look empty
  in the BuzzSwarm flight heatmap (before/after-filter heatmap, resting vs flying heatmap with hourly zone density,
  labelled tracking clip, independent dark-spot check on raw video). Findings on the Penzance Ae. aegypti M
  experiment (Jul 5-7): the heatmap's linear colour scale hides everything under ~10% of the centre; over the dark
  wall panels a flyer darkens the image by ~10-30 grey levels vs ~70 at the centre; the filters remove most wall
  detections everywhere (walking/landing), not only at the feeders. Untracked spots in the feeder zones were mostly
  mosquitoes sitting there and recorded by the resting tracker; missed flight ~3-5%. Resting density on the feeder
  panels is ~1.4-1.5x the other walls with lights on, lower at night. Track coordinate k = video frame start+k-2
  (as in VideoManager.overlay_tracking; checked on video).
- Open: that experiment's `buzzwatch_track_settings.yml` has an empty `cage_border_points` (its geometry.json is
  older), so a BuzzSwarm rerun or the swarm-zone dialog there stops with "no cage border drawn" until the border is
  redrawn. Its experiment JSON still points at the old data drive's Light_Test folder.
- 11-24: live kit `hailo/stack_v14_live/` sent to the Pi (`~/hailo_bench/stack_v14_live`, md5sum -c all 28 OK) for the
  Pi side's live run with the Camera Module 3 NoIR (1536x864, 25 fps): HEF, README (contract, camera/scale notes),
  `live_preprocess.py` (exporter tiling verbatim; check reproduces the 1080p training tile origins), stabilize.py,
  exporter copies, pi_check (8 tiles + emulator output), clip_check (10 s 1080p clip + PyTorch-through-live-code boxes,
  32.9 / frame = the demo). Training cage widths: ~1,165 px (1080p), ~770 px (720p); all 25 fps.

## 2026-11-25 — v1.0.0: hang, border, radius and install fixes; feeder tools merged; clean-up

- **yolo_export:** `safe_pool.SafePool` runs one job per spawn worker over its own pipe; a worker that dies
  (e.g. 0xC0000005 on a numpy-2-era pickle) or passes `--task-timeout` (default 1800 s) fails only its segment
  (logged with experiment + file) and is replaced. The selftest has a crash/truncated/hang case (~6 s; the old
  pool was still stuck after 30 s).
- **BuzzSwarm:** a missing/invalid cage border raises `CageBorderMissing` in a preflight before any day loads; the
  Run button and swarm-zone picker offer to open Setup → Draw Cage Borders; `swarm --which clock` exits with the
  same message. Never guessed. `px_settings(radius)` scales central/core radii, shape rings, heatmap cells and
  `fly_px_frame` with the swarm radius (s = r/100); outputs at 100 px are byte-identical; `geometry.json` records
  the values. `NEAR_BAND_PX`, r50 and `prop_within_100/150` stay unscaled on purpose.
- **App:** opening a second experiment kept the first one's settings (`em.settings` was reused); fixed.
- **BuzzPhono tests:** the "worker process stopped abruptly" failures were test pollution: AppTest left a page as
  `sys.modules['__main__']`, spawn workers re-ran it, and 2_Calibrate.py raises outside Streamlit. conftest puts
  `__main__` back after each test; `test_main_module_restored.py` guards it. Production unaffected.
- **Install:** a lab PC had only `buzzsuite_env`; Open the BuzzPhono app fell back to `conda run -n
  buzzphono_env` → "EnvironmentLocationNotFound". The bridge now asks `conda env list --json`, uses the env's
  python wherever it is, else says it is not installed and how to install it. `setup_env.bat` creates/checks
  `buzzphono_env` too. README: Windows short way + Troubleshooting.
- **tools/feeder_check** merged (rebased on main); uses the experiment's scaled `fly_px_frame`; excluded from ruff
  (Python 3.8 app-side code).
- **Clean-up:** deleted branches yolo-export, review/base, review/full (closes PR #3 unmerged), swarm-plots,
  fix/review-pr3-findings and the two Copilot branches; removed the radius and swarm-plots worktrees; the old
  BuzzSuite copy's legacy `buzzphono/` folder renamed `buzzphono_OLD`.
- **Tests:** `tests/buzzsuite/` (stdlib runner, `python tests/buzzsuite/run_all.py`, no pytest needed in
  buzzsuite_env). BuzzPhono 1.0.0. Tagged v1.0.0.
