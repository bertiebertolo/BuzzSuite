# BuzzSuite — Claude Code Build Plan

## Efficiency guide for implementing agents (read first)

High-signal working rules for this codebase. Follow these to stay fast and avoid known traps.

1. **Query the graphify graph first; don't bulk-read.** A `graphify-out/` knowledge graph of this
   repo exists — query it (query/path/explain) to locate code, then `Read` only the exact line
   ranges it points at. The tab managers are 2000+ lines each; never read them whole. Rebuild the
   graph after large edits if you rely on it again.
2. **Verify headlessly with the conda python.** Use `/opt/anaconda3/envs/buzzwatch_env/bin/python`
   (system `python3` crashes on Tk imports). Test Tk code with a hidden root
   (`tk.Tk(); root.withdraw()`), stub the manager/experiment objects, build the tab into a
   `ttk.Frame`, assert no exception. Never launch the live GUI or write to the data drive in tests.
3. **Reuse-first via `build_*_section(parent)`.** The UI patterns you need already exist in
   `buzzphono_tab_manager.py` (`_run_image_step`, `_draw_borders_and_refresh`, `_overlay_zones`,
   `_display_pil`, `_show_image`). Compose tabs from section-builders; do not move big method bodies
   between files (they are tightly coupled to their `self.<widget>` handles).
4. **Frozen — never touch:** tracking numerics, the `.pkl` format, the `buzzwatch_data_analysis`
   package name (baked into pickled objects + the `_compat_shim`), `buzzwatch_track_settings.yml`,
   and the `buzzwatch_env` conda env. The app shell/UI is fair game.
5. **Python 3.8 only** (no walrus/match/3.9+). No new dependencies.
6. **Work incrementally:** one tab at a time — build → headless-verify → update `DEVLOG.md` → pause
   for the user's go-ahead. Don't re-read files you just edited (the harness tracks state).
7. **Data drive is read-only and can unmount.** Resolve paths via `path_utils`; never hardcode. If a
   command reports the cwd was deleted, `/Volumes/Mosquito2` dropped — wait for it to remount.
8. **Entry point is `run_gui.py`** (renamed from `buzzwatch_analysis_app.py`).

> **Start here if you are a fresh agent (including on macOS).** Read `DEVLOG.md` first —
> it says exactly what has been done and what to do next. Read `BUILD_PLAN.md` for the full
> rationale, corrections, and cross-platform/modernization decisions behind this spec. This
> file is the canonical build spec; it has been corrected against what is actually on disk
> (see "Corrections applied" below) — do not trust the original uncorrected assumptions if
> you find them anywhere else.
>
> **Working agreement: pause after each step below and wait for the user's explicit
> go-ahead before starting the next one.** After finishing a step's code changes, run its
> verification subagent, then update `DEVLOG.md` with what changed and how to resume, then
> stop and wait.

## Corrections applied (do not repeat these mistakes)

1. **BuzzSwarm's r50/Sholl engine is `fru2_zt_normalized_analysis.py`**, not
   `batch_centroid_analysis.py`. The latter only exists inside the embedded
   `BuzzWatch_Fru_2\BuzzWatch_app_v2_copy\` copy and computes per-trajectory median
   distance + threshold proportions — it has no Sholl CDF or r50 logic at all. The real
   engine (r50 at lines ~621-653/958, Sholl at ~1186-1246) is imported by all three keeper
   scripts (`fru2_sholl_csv.py`, `fru2_pooled_analysis.py`, `fru2_per_trajectory_plots.py`
   each do `import fru2_zt_normalized_analysis as m`).
2. **BuzzPhono uses `custom_zones.json` (speaker polygon)**, not the older
   `speaker_side_data` 4-quadrant metric. `plot_speaker_side_phonotaxis.py` and
   `plot_speaker_combined.py` are excluded (the latter is also a one-off hardcoded to 3
   specific cage names from a past study).
3. **This build targets Windows and macOS.** The original source already models this
   (dual root lists like `_ANALYSIS_ROOTS = ["/Volumes/Mosquito2/...", r"E:\Buzzwatch\..."]`
   in `plot_speaker_distance_phonotaxis.py`) — extend that pattern everywhere via
   `path_utils.py`, don't add new Windows-only assumptions.
4. **Modernization is scoped to the app shell, not the tracking numerics.** The tracker's
   math and `.pkl` format must stay byte-compatible with what's already been validated;
   architecture, concurrency, UX, and an opt-in portable GPU path (OpenCV UMat/OpenCL, CPU
   fallback always available) are fair game.

---

## What you are building

**BuzzSuite** is a unified Python/Tkinter desktop application that takes a user from raw
Raspberry Pi video all the way to finished behavioural analysis plots, without leaving the
app or touching the command line. It combines three mosquito behavioural assay pipelines —
activity tracking (BuzzWatch), spatial aggregation (BuzzSwarm), and phonotaxis (BuzzPhono)
— into one app with a shared tracking back-end, a single experiment format, and a
module-selector UI. The goal is the **minimum viable, least-bloated version** of all three
pipelines in one place, running the same on Windows and macOS.

### End-to-end user workflow (the primary design target)

```
1. Select raw video folder (.h264 or .mp4 files from the Raspberry Pi)
2. Select which dates / recording sessions to include
3. Convert .h264 → .mp4 in-app (H264toMP4 utility, lossless remux)
4. App auto-creates the experiment folder structure under the resolved data root
   (Analysis\ tree) and writes the experiment_{alias}.json — user confirms/edits name and alias
5. User chooses one or more analyses: Activity / BuzzSwarm / BuzzPhono
6. App runs the chosen analyses (tracking + aggregation/phonotaxis as selected)
7. Output PDFs appear in the experiment's plots\ folder; user can open them in-app
```

**Multiple experiments in parallel**: the app must support opening and running multiple
experiments simultaneously. Each experiment runs in its own background thread wrapping its
own process pool. The UI shows a dashboard of active experiments with individual progress
bars. The user can start a new experiment without waiting for running ones to finish.

---

## Codebase map (original source — read before touching anything)

All source material was under `E:\BuzzSuiteDev\` on the machine this build started on. Once
Step 1 is complete, **BuzzSuite/ is self-contained** and these original folders are no
longer needed at runtime — this map is kept for provenance only.

| Folder | What it contains | Relevance |
|---|---|---|
| `BuzzWatch_app_v2/` | Most up-to-date tracking engine + phonotaxis scripts + the full Tkinter app | **Primary source** — tracker core, GUI skeleton, all phonotaxis (custom-zones) code |
| `BuzzWatch_Fru_2/` | Aggregation/swarming analysis scripts (`fru2_*.py`) | Source for BuzzSwarm; real engine is `fru2_zt_normalized_analysis.py` (see Corrections); also has `_compat_shim/` |
| `BuzzWatch_app/` | Older activity-only version of the app | Reference only — not used |
| `H264toMP4_App/` | Standalone `.h264 → .mp4` converter (`convert_gui.py` + bundled Windows ffmpeg) | Included as a utility tab; macOS needs its own ffmpeg binary or PATH fallback |
| `BuzzSuite/` | **The build target** | Everything goes here |

### The three analysis modules

**BuzzWatch_V3 (activity tracking)** — `BuzzSuite/buzzwatch_data_analysis/`
- Entry: `single_video_analysis.py` — processes one 20-min video segment
- Trackers: `moving_obj_tracker.py` (flying), `resting_obj_tracker.py` (resting), `mosquito_obj_tracker.py` (data structure)
- The overhaul (V3) is feature-flagged in `tracking_overhaul.py` and `single_video_analysis.py`:
  - `assignment="hungarian"` + `motion_model="constant_velocity"` + `use_backward_pass=True` → motion-aware flying tracker
  - `split_blobs=True` → distance-transform blob splitting for crowded frames
- `tracking_overhaul.py::OVERHAUL_TRACKING` is the **adopted, correct config** and, per Step 2, becomes the only tracking path (no old-vs-new toggle).

**BuzzSwarm (aggregation)** — `BuzzSuite/buzzswarm/`
- **Core engine: `fru2_zt_normalized_analysis.py`** — computes Sholl-style CDFs of flying
  trajectory points vs cage centre; key metric is **r50** (median radial distance). All
  other buzzswarm scripts import it as `import fru2_zt_normalized_analysis as m`.
- Entry scripts (thin, callable-function based): `fru2_sholl_csv.py`, `fru2_pooled_analysis.py`, `fru2_per_trajectory_plots.py`
- Depends on `calculate_cage_centroid` from `buzzwatch_data_analysis/flight_center_distance_analysis.py`
- `_compat_shim/` patches pickle loading for cross-version `.pkl` files — kept as a fallback if the real `buzzwatch_data_analysis` package is ever unavailable on `sys.path`

**BuzzPhono (phonotaxis)** — `BuzzSuite/buzzphono/`
- Key scripts: `plot_speaker_distance_phonotaxis.py` (primary), `plot_flying_phonotaxis.py`, `plot_phonotaxis_per_day.py`, `plot_activity_phonotaxis.py`
- Reads `custom_zones.json` for the speaker polygon (`{"speaker": {"polygon": [[x,y], ...]}}`); computes speaker-zone resting fraction — stimulus OFF window is minutes **28-40** of each hour (`OFF_START_MIN`/`OFF_END_MIN` = 28/38 in source, docstrings sometimes say 30-40 — trust the constants), ON window is minutes **40-50**
- Outputs: time-series PDFs, paired boxplots, per-day PDFs, 2D occupancy heatmaps overlaid with the speaker polygon

### H264toMP4

A small standalone Tkinter app (`h264_converter/convert_gui.py`) that batch-converts raw
`.h264` Raspberry Pi recordings to `.mp4` via a lossless FFmpeg remux. It handles the common
case where only the first segment in a multi-segment recording carries SPS/PPS headers,
borrowing headers from a valid segment and prepending them to headerless ones. ffmpeg/ffprobe
are resolved via `find_tool()`: app-relative `ffmpeg/bin/{win,mac}/` first, then PATH.

---

## Data folder structure (resolved via `path_utils.resolve_data_root()`)

All raw recordings and analysis outputs are stored under a single root, separated into
`Recording\` and `Analysis\` trees that share the same experiment hierarchy:
`{AssayType}\{Location}\{ExperimentName}\`. On the machine this was built on, that root is
`E:\Buzzwatch\`; on macOS it may be mounted at `/Volumes/Mosquito2/Buzzwatch` or similar —
**never hardcode the drive letter/mount point**, resolve it.

```
{DataRoot}\
│   buzzwatch_track_settings.yml          ← global default tracking config
│
├── Recording\                             ← raw video files (.mp4 / .h264)
│   ├── Nosound\{Location}\{CageName}\
│   ├── Phonotaxis\{Location}\{CageName}\
│   └── Lightctrl\{Location}\{CageName}\
│
└── Analysis\                              ← all analysis outputs
    ├── Phonotaxis\{Location}\{Experiment}\
    └── Lightctrl\{Location}\{Experiment}\
```

Each experiment folder under `Analysis\` contains:

```
{Experiment}\
│   buzzwatch_track_settings.yml           ← per-experiment tracking config
│   experiment_{alias}.json                ← experiment definition (paths, cage, batch)
│   background_with_borders.png            ← reference background frame with cage borders
│   custom_zones.json                      ← BuzzPhono speaker-zone polygon (user-defined)
│   analyzed_data.pkl                      ← aggregated tracking output
│   analyzed_data_filtered.pkl             ← filtered version
│   analyzed_data_speed_filtered.pkl       ← speed-filtered version
│
├── final_tracking_data\                   ← forward-pass tracking results, one file per video segment
│       forward_mosq_tracks_{Cage}_{YYYYMMDD}_{HHMMSS}_{NNNNN}
│
├── temp_data\                             ← intermediate per-segment outputs (+ bg_cache_{session_key}.pkl)
├── individual_images\                     ← per-segment reference frame PNGs
├── images_mortality\                      ← per-segment mortality-check PNGs (same naming)
├── tracking_resting\                      ← resting tracker outputs
├── tracking_moving\                       ← moving (flying) tracker outputs
├── log_analysis\                          ← processing logs
├── test_tracking\                         ← test/debug tracking runs
│
└── plots\
    ├── activity_plots\
    ├── custom_zones\                       ← BuzzPhono outputs (proportion, heatmaps, csv)
    └── sholl_cache.pkl                     ← BuzzSwarm CDF cache (Step 5)
```

Segment filenames follow the pattern `{CageName}_{YYYYMMDD}_{HHMMSS}_{NNNNN}` where
`YYYYMMDD` and `HHMMSS` are the recording session start date/time and `NNNNN` is a
zero-padded 5-digit segment index (each segment is 1200 s = 20 min by default).

### Tech stack

- Python 3.8, conda env `buzzwatch_env` (works the same on Windows and macOS)
- GUI: `tkinter` + `ttk` + `tkcalendar`
- Vision: `opencv-python` (`cv2`) — optionally via UMat/T-API (OpenCL) for GPU accel, CPU fallback always present
- Analysis: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`
- Plotting: `matplotlib` + `seaborn` embedded via `FigureCanvasTkAgg`
- Parallel processing: `multiprocessing.Pool` (spawn-safe on both OSes)
- Config: YAML (`pyyaml`), JSON

---

## Build steps

Work through these steps **in order**. After each step: (1) call the `Task` tool to spawn a
verification subagent (prompt given per step below) and resolve any blocking issues it
reports; (2) update `DEVLOG.md` with what changed and how to resume; (3) **stop and wait for
the user's go-ahead** before starting the next step.

---

### Step 1 — Copy and clean up into BuzzSuite

**Goal:** A clean, organised copy of all needed source files in `BuzzSuite/` with no dead
weight, laid out so it's self-contained and portable.

**What to copy (from `BuzzWatch_app_v2/` unless noted):**

Core tracking engine → `BuzzSuite/buzzwatch_data_analysis/` (package, `__init__.py`)
- `single_video_analysis.py`, `moving_obj_tracker.py`, `resting_obj_tracker.py`, `mosquito_obj_tracker.py`, `misc_functions.py`, `experiment_analysis.py`, `flight_center_distance_analysis.py`

App skeleton → `BuzzSuite/` (flat, top level)
- `buzzwatch_analysis_app.py`, `config_manager.py`, `experiment_manager.py`, `state_manager.py`, `logger.py`, `common_imports.py`, `ui_manager.py`, `video_manager.py`, `image_manager.py`, `plot_manager.py`, `tracking_overhaul.py`

Tab managers → `BuzzSuite/`
- `video_tab_manager.py`, `analysis_tab_manager.py`, `batch_processing_tab_manager.py`, `comparison_tab_manager.py`, `statistical_analysis_manager.py`, `glmm_analysis_manager.py`

BuzzSwarm scripts (from `BuzzWatch_Fru_2/`) → `BuzzSuite/buzzswarm/` (package, `__init__.py`)
- **`fru2_zt_normalized_analysis.py` (the real r50/Sholl engine — do not substitute `batch_centroid_analysis.py`, see Corrections)**
- `fru2_sholl_csv.py`, `fru2_pooled_analysis.py`, `fru2_per_trajectory_plots.py`
- `_compat_shim/buzzwatch_data_analysis/` → copy into `BuzzSuite/buzzswarm/_compat_shim/buzzwatch_data_analysis/`
- Edit `fru2_zt_normalized_analysis.py`'s `sys.path`/import block so it points at BuzzSuite's own `buzzwatch_data_analysis/` package (with `_compat_shim` as lower-priority fallback), not `E:\BuzzWatch_app_v2`

BuzzPhono scripts (from `BuzzWatch_app_v2/`) → `BuzzSuite/buzzphono/` (package, `__init__.py`)
- `plot_speaker_distance_phonotaxis.py` (primary, custom-zones based)
- `plot_flying_phonotaxis.py`, `plot_phonotaxis_per_day.py`, `plot_activity_phonotaxis.py`, `behavioral_timeseries_analysis.py`

H264 converter → `BuzzSuite/h264_converter/`
- `convert_gui.py` from `H264toMP4_App/`
- `ffmpeg/bin/win/ffmpeg.exe` + `ffprobe.exe` (Windows binaries, bundled)
- `ffmpeg/bin/mac/` (empty placeholder — populate with mac binaries when building/running on macOS, or rely on PATH fallback in `find_tool()`)

Shared utilities → `BuzzSuite/`
- `path_utils.py` (new) — `resolve_first_existing()`, `resolve_data_root()`, `get_package_root()`

**What NOT to copy (explicit exclusions):**
- Any file starting with `._` (macOS resource fork metadata)
- `proto_*.py`, `tmp_*.py`, `original_*.py`, `check_*.py`, `compare_*.py`, `run_*.py`, `draw_cage_*.py`, `draw_speaker_*.py`
- `batch_centroid_analysis.py` (wrong file — see Corrections; only exists in the embedded `BuzzWatch_app_v2_copy/` anyway)
- `plot_speaker_side_phonotaxis.py`, `plot_speaker_proportion_boxplots.py`, `plot_speaker_combined.py` (old speaker-side metric / cage-specific one-off — see Corrections)
- `fru2_regen_*.py`, `fru2_validation_*.py`, `fru2_zt_extra_windows.py`, `fru2_male_*.py`, `fru2_female_*.py`, `fru2_per_trajectory_30min_all.py`, `fru2_sholl_r1_100_plot.py` (one-offs — note `fru2_zt_normalized_analysis.py` itself is NOT a one-off despite the `zt` in the name, it's the core engine, keep it)
- Any `__pycache__/`, `.git/`, `plots/`, `glmm_results/` directories
- `BuzzWatch_app_v2/BuzzWatch_app/` and `BuzzWatch_Fru_2/BuzzWatch_app_v2_copy/` embedded subdirectories
- `run_*.py` batch scripts that are experiment-specific
- `fill_missing_resting_pkls.py`, `generate_resting_zones_csv.py`, `export_zt_clips.py`

**Also create:**
- `BuzzSuite/environment.yml` — copy from `BuzzWatch_app_v2/`
- `BuzzSuite/buzzwatch_track_settings.yml` — copy the global default from the data root
- `BuzzSuite/BUILD_PLAN.md`, `BuzzSuite/DEVLOG.md` — handoff artifacts (see top of this file)

**Verification — spawn a subagent with this prompt:**
```
You are a code reviewer. Your job is to verify Step 1 of a BuzzSuite build.

Check the following in E:\BuzzSuiteDev\BuzzSuite\ (adjust root if running elsewhere):
1. All required files listed in Step 1 are present (spot-check at least 10 key ones)
2. No macOS ._* metadata files exist anywhere in the tree
3. No proto_*.py, tmp_*.py, check_*.py files exist in the main app folders
4. batch_centroid_analysis.py, plot_speaker_side_phonotaxis.py, plot_speaker_combined.py are ABSENT (these are explicitly excluded — see Corrections in CLAUDE.md)
5. fru2_zt_normalized_analysis.py IS present in buzzswarm/ and its import block near the top does NOT reference E:\BuzzWatch_app_v2 (it should point at BuzzSuite's local buzzwatch_data_analysis package instead)
6. The buzzwatch_data_analysis/ sub-package has an __init__.py (create an empty one if missing)
7. The buzzswarm/ and buzzphono/ directories exist and contain the expected files, each with __init__.py
8. The h264_converter/ directory exists with convert_gui.py and Windows ffmpeg binaries under ffmpeg/bin/win/
9. No __pycache__ directories were copied
10. path_utils.py, BUILD_PLAN.md, and DEVLOG.md exist at BuzzSuite/ root

Report: list any missing files, any prohibited files present, and any structural issues. Keep it under 300 words.
```

---

### Step 2 — Integrate all three analyses into one app

**Goal:** A single running Tkinter app with a module-selector that lets the user choose
Activity / BuzzSwarm / BuzzPhono (or all three), then routes to the correct tab set.

**Architecture:**

Extend `buzzwatch_analysis_app.py` with a top-level **AssaySelector** frame that appears on
startup (before any experiment is loaded) with three toggleable buttons:
- **Activity** (existing functionality)
- **Swarming** (BuzzSwarm — new)
- **Phonotaxis** (BuzzPhono — new)

The existing 4-tab notebook (Video Inspection, Preliminary Analysis, Batch Processing,
Compare Experiments — built in `ui_manager.py::init_tabs`) becomes the **Activity** module.
Add two new tab sets, wired into `UIManager` the same way the existing four are
(`setup_managers()` then `init_tabs()`):

**BuzzSwarm tab** (new `buzzswarm_tab_manager.py`):
- Input: select an existing experiment folder (reads `final_tracking_data/` tracking PKLs)
- Parameters: min flight frames, speed filter min/max, window minutes, Sholl radius step
- Run button: calls into `fru2_zt_normalized_analysis.py` (the engine) via the refactored
  callable functions in `fru2_sholl_csv.py` / `fru2_pooled_analysis.py` /
  `fru2_per_trajectory_plots.py` — **not** `batch_centroid_analysis.py` (wrong file, see
  Corrections)
- Output panel: shows the r50 CDF plot inline via `FigureCanvasTkAgg`
- Export: save plots to `plots/` under the experiment folder

**BuzzPhono tab** (new `buzzphono_tab_manager.py`):
- Input: select an experiment folder + select `custom_zones.json` (or define speaker zone interactively — Step 4)
- Parameters: stimulus ON window (default min 40–50), OFF window (default min 28–38), time range
- Run button: calls the phonotaxis analysis functions (refactor `plot_speaker_distance_phonotaxis.py` into callable functions)
- Output panel: shows speaker-zone resting fraction plot inline
- Export: save all phonotaxis PDFs to `plots/custom_zones/`

**H264 Converter tab** (new `h264_tab_manager.py`):
- Embed `convert_gui.py` logic as a tab (input folder browse, output folder, progress bar, convert button)
- No separate window — run inline

**Key rules for this step:**
- Do NOT rewrite the tracking engine numerics or change the `.pkl` format — only wire it up.
  The app **shell/architecture** (routing, services, path layer) may be modernized per
  `BUILD_PLAN.md`'s Modernization scope.
- Refactor `plot_speaker_distance_phonotaxis.py` and the `fru2_*.py` BuzzSwarm scripts so
  their core logic is importable functions (not `if __name__ == "__main__":` scripts). The
  scripts can remain as CLI entry points but the logic must be callable from the GUI. While
  doing this, replace their hardcoded root-path globals (`_ANALYSIS_ROOTS`, `DRIVE_ROOTS`,
  `BASE_PATH`/`OUTPUT_BASE` in `fru2_zt_normalized_analysis.py`) with the user-selected
  experiment path passed in as a parameter, or route through `path_utils.py`.
- The BuzzWatch_V3 overhaul config (`OVERHAUL_TRACKING` from `tracking_overhaul.py`) must be
  the **default and only tracking config** — remove the toggle checkbox in
  `batch_processing_tab_manager.py` (`use_overhaul_var`, `_on_overhaul_toggle`) that switches
  between old and new. There is no reason to run the old tracker.
- Shared state: all three modules use the same `ExperimentManager` and `ConfigManager`

**Verification — spawn a subagent with this prompt:**
```
You are a code reviewer. Your job is to verify Step 2 of a BuzzSuite build.

The target app is in E:\BuzzSuiteDev\BuzzSuite\.

1. Check that buzzwatch_analysis_app.py launches without import errors:
   cd E:\BuzzSuiteDev\BuzzSuite && python -c "import ast; ast.parse(open('buzzwatch_analysis_app.py').read()); print('OK')"
   Do the same for all new *_tab_manager.py files.
2. Check that buzzswarm_tab_manager.py and buzzphono_tab_manager.py exist and are syntactically valid Python.
3. Check that h264_tab_manager.py exists and imports convert_gui logic without spawning a window.
4. Confirm buzzswarm_tab_manager.py calls into fru2_zt_normalized_analysis.py / fru2_sholl_csv.py / fru2_pooled_analysis.py / fru2_per_trajectory_plots.py — NOT batch_centroid_analysis.py (that file should not exist in this codebase at all).
5. Check that the fru2_*.py BuzzSwarm scripts and plot_speaker_distance_phonotaxis.py now have importable functions (not just top-level script code).
6. Check that OVERHAUL_TRACKING is applied as default in the batch processing flow and there is no "enable overhaul" checkbox remaining (grep for use_overhaul_var, _on_overhaul_toggle — should be gone).
7. Grep for any remaining hardcoded experiment paths (e.g. /Volumes/Mosquito, C:\Users\alber, E:\BuzzWatch_app_v2, F:\BuzzWatch_Fru_2) — these must all come from config, path_utils.py, or user selection.

Report: syntax errors, missing integrations, hardcoded paths, structural problems. Under 400 words.
```

---

### Step 3 — Optimise the tracking pipeline

**Goal:** Make batch processing faster without changing results.

**Specific targets:**

1. **Multiprocessing**: The batch processor in `batch_processing_tab_manager.py` already
   uses `multiprocessing.Pool`. Verify it is passing work correctly and not serialising
   unnecessarily. The pool should process video segments in parallel (one worker per
   segment), capped at `min(cpu_count()-1, n_segments)`.
2. **Background caching**: `single_video_analysis.py` computes a background reference per
   segment. If the same video session has many segments, cache the bright-percentile
   background once per session (keyed by `{CageName}_{YYYYMMDD}_{HHMMSS}`) and reuse it
   across segments in that session. Write cache to `temp_data/bg_cache_{session_key}.pkl`.
3. **Unnecessary recomputation in `analysis_tab_manager.py`**: There is a known bug where
   single-video analysis is triggered twice per run. Find and fix it — look for duplicate
   calls to the analysis function in the tab manager's run button handler.
4. **Blob split threshold**: `OVERHAUL_SINGLE_BLOB_AREA = 150` is the single-mosquito
   footprint. Verify this is read from settings and not recalculated per frame.
5. **Portable opt-in GPU** (per `BUILD_PLAN.md` Modernization scope): route the per-frame
   vision ops (threshold, morphology, distance transform) through OpenCV **UMat/T-API**
   when `cv2.ocl.haveOpenCL()` is true, behind a settings flag, with an unconditional CPU
   fallback path. Add a parity check (UMat output vs CPU output, within tolerance) so
   numerics stay unchanged. Vectorise any remaining per-frame Python loops with NumPy where
   cheap to do so.
6. **Do not** optimise the plotting code in this step — that is Step 5.

**Verification — spawn a subagent with this prompt:**
```
You are a code reviewer. Your job is to verify Step 3 of a BuzzSuite build.

1. Open E:\BuzzSuiteDev\BuzzSuite\batch_processing_tab_manager.py. Confirm that multiprocessing.Pool is used and the worker count is capped at cpu_count()-1 or fewer.
2. Open single_video_analysis.py. Check whether a background cache mechanism has been added (look for bg_cache or session_key or similar). Report what you find.
3. Check analysis_tab_manager.py for the duplicate-call bug: grep for calls to the single-video analysis function and count how many times the run handler invokes it. Report the count and line numbers.
4. Confirm OVERHAUL_SINGLE_BLOB_AREA is not recalculated per frame — it should be read once at setup.
5. Check for a GPU opt-in path: grep for cv2.ocl.haveOpenCL, UMat, or a GPU settings flag, and confirm there is an unconditional CPU fallback (not a hard dependency on OpenCL being available).
6. Run a syntax check on all modified files.

Report findings and any new bugs introduced. Under 300 words.
```

---

### Step 4 — Make it work end-to-end as one app

**Goal:** A user can go from raw `.h264` files to finished analysis plots entirely within
the app. Multiple experiments can run simultaneously.

**What to implement:**

1. **Intake wizard** (`intake_wizard.py` — new): A step-by-step panel (embedded in the main
   notebook as an "Intake" tab, always visible, not a separate window):
   - **Step A — Source video**: Browse button to select the raw recording folder. The app
     scans for `.h264` and `.mp4` files and lists them with their dates/timestamps parsed
     from the filename convention `{CageName}_{YYYYMMDD}_{HHMMSS}_{NNNNN}`.
   - **Step B — Date selection**: A calendar or date-range picker (`tkcalendar`) showing
     which dates have recordings. User ticks which days to include.
   - **Step C — Convert**: If `.h264` files are selected, show a "Convert to MP4" button
     that calls `h264_converter/convert_gui.py` logic inline (no separate window). A
     progress bar shows conversion progress. Already-converted files are skipped
     automatically.
   - **Step D — Experiment setup**: Auto-populate experiment name from cage name + date
     range. Let user set: assay type (Phonotaxis / Lightctrl / Nosound), location name,
     experiment alias. The app constructs the `Analysis\{AssayType}\{Location}\{Experiment}\`
     path (relative to the resolved data root) and creates all required subfolders. Writes
     `experiment_{alias}.json`.
   - **Step E — Analysis selection**: Three checkboxes: "BuzzWatch (Activity)", "BuzzSwarm
     (Aggregation)", "BuzzPhono (Phonotaxis)". If BuzzPhono is selected, show a "Define
     speaker zone" button (see point 3 below). A "Run" button starts the selected analyses.
2. **Multi-experiment dashboard** (`experiment_dashboard.py` — new): A persistent panel
   showing all experiments started this session. Each row: experiment alias, assay type,
   current step, progress bar, Cancel button. Each experiment runs in its own
   `threading.Thread` wrapping its own `multiprocessing.Pool` — fully independent.
3. **Custom zones UI**: BuzzPhono requires `custom_zones.json` (speaker polygon). Add a
   "Define Zones" button that opens a canvas overlay on `background_with_borders.png` and
   lets the user click to place polygon vertices, closing on double-click or Enter. Saves as
   `{"speaker": {"polygon": [[x, y], ...]}}` in `custom_zones.json` — matches
   `_load_speaker_zone()`'s expected structure. Polygon only, no curves.
4. **Unified experiment JSON**: `ExperimentManager` must create/update
   `experiment_{alias}.json` correctly regardless of which module is active. All three
   modules read the same JSON and folder structure.
5. **Module routing on experiment load**: When loading an existing experiment JSON, detect
   which analyses have already run (content in `final_tracking_data/`, `tracking_moving/`,
   `tracking_resting/`, `plots/`) and pre-tick the corresponding checkboxes. Don't re-run
   completed tracking unless the user explicitly requests it.
6. **Progress and cancellation**: Every long-running operation must update the dashboard
   progress bar and be cancellable via `threading.Event`. On cancel, terminate the pool and
   delete partial output files.
7. **Output routing**: All plots save to the correct subfolder under the experiment's
   `plots/` directory. No hardcoded output paths anywhere.
8. **Single entry point**: `python buzzwatch_analysis_app.py` launches the app from
   `BuzzSuite/` on both Windows and macOS.

**Verification — spawn a subagent with this prompt:**
```
You are a code reviewer. Your job is to verify Step 4 of a BuzzSuite build.

1. Check that intake_wizard.py exists in E:\BuzzSuiteDev\BuzzSuite\ and contains at least Steps A-E (source video, date selection, convert, experiment setup, analysis selection). Check for syntax errors.
2. Check that experiment_dashboard.py exists and implements a multi-experiment tracking panel with per-experiment progress. Check for syntax errors.
3. Open experiment_manager.py. Confirm it creates all required subfolders (final_tracking_data, temp_data, individual_images, images_mortality, tracking_resting, tracking_moving, log_analysis, plots) and writes experiment_{alias}.json.
4. Check buzzphono_tab_manager.py for a polygon-drawing zone editor — confirm it saves to custom_zones.json with a {"speaker": {"polygon": [...]}} shape.
5. Grep BuzzSuite/ for hardcoded paths containing "Mosquito", "Volumes", "alber", "BuzzSuiteDev", "E:\Buzzwatch" — none should appear in business logic (path_utils.py itself listing candidate roots is fine).
6. Confirm threading.Event or equivalent cancellation is present in at least the batch processing and BuzzSwarm analysis flows.
7. Confirm python buzzwatch_analysis_app.py parses without syntax errors.

Report any blockers. Under 500 words.
```

---

### Step 5 — Optimise again (post-integration)

**Goal:** Fix performance issues introduced by integration, and optimise the plotting
pipeline.

**Targets:**

1. **Lazy imports**: `common_imports.py` likely imports heavy packages (cv2, matplotlib,
   sklearn) at startup. Move heavy imports to be lazy — only import inside the function
   that needs them. This reduces cold-start time.
2. **Plot generation**: Phonotaxis plots regenerate all figures on every run even if the
   data hasn't changed. Add a `force_replot=False` parameter to each plot function; skip
   regeneration if the output PDF already exists and the tracking PKL modification time is
   older than the PDF.
3. **BuzzSwarm CDF computation**: the BuzzSwarm engine recomputes CDFs from raw PKLs every
   time. Cache the per-experiment CDF result as `plots/sholl_cache.pkl`. Invalidate the
   cache if any `final_tracking_data/` file is newer than the cache.
4. **Memory**: After batch processing completes, explicitly `del` large arrays and call
   `gc.collect()`. Do not hold all tracking PKLs in memory simultaneously during BuzzSwarm
   analysis — process and discard one at a time.
5. **UI responsiveness**: All analysis must run in a background thread (not the main
   Tkinter thread). If any analysis function is called directly from a button callback
   without threading, fix it.

**Verification — spawn a subagent with this prompt:**
```
You are a code reviewer. Your job is to verify Step 5 of a BuzzSuite build.

1. Open common_imports.py. List every top-level import. Flag any that are heavy (cv2, matplotlib, sklearn, scipy, statsmodels, umap) — these should be inside functions, not top-level.
2. Check plot_speaker_distance_phonotaxis.py (or its refactored equivalent in buzzphono/) for a force_replot or equivalent skip-if-exists mechanism.
3. Check the BuzzSwarm engine/scripts for a CDF cache mechanism (sholl_cache.pkl or similar).
4. Grep BuzzSuite/ for any analysis function called directly from a button callback (look for .configure(command=...) or .bind(...) that calls a function taking more than 1 second — check if it is wrapped in threading.Thread).
5. Check for gc.collect() or explicit del of large arrays after batch processing.

Report issues. Under 300 words.
```

---

### Step 6 — Remove bloat and dead code

**Goal:** Every file and function in `BuzzSuite/` must be used by the running app. Delete
the rest.

**Process:**

1. Run a static import trace: starting from `buzzwatch_analysis_app.py`, follow every
   import and identify which modules are imported. Any `.py` file in `BuzzSuite/` not
   reachable from this trace is a candidate for deletion.
2. Within each file, identify unused functions: functions defined but never called from
   within the app (`grep -rn "function_name"` across the whole `BuzzSuite/` tree). Remove
   them or move them to a `utils/archive/` folder if genuinely uncertain.
3. Remove redundant wrappers: if a function in one module simply calls another function in
   a different module with no transformation, collapse it.
4. Remove known dead code patterns: commented-out import blocks, `# print(...)` debug
   lines, unreachable `if False:` blocks, old non-overhaul tracker paths (any branch
   guarded by `if assignment == "greedy":` — but only after confirming "greedy" is truly
   never used, since the overhaul is now the only config).
5. Clean up `comparison_tab_manager.py` and `statistical_analysis_manager.py`: look for
   duplicate plot functions that produce nearly identical outputs and consolidate.
6. **Do not remove**: any function related to pickle loading/saving, the compat shim, cage
   border handling, or `calculate_cage_centroid` — these are load-bearing even if they look
   standalone.

**Verification — spawn a subagent with this prompt:**
```
You are a code reviewer. Your job is to verify Step 6 of a BuzzSuite build.

1. Check that buzzwatch_analysis_app.py still parses cleanly after removals.
2. Pick 5 random .py files in BuzzSuite/ and check each one for: commented-out import blocks, # print debug lines, and unreachable if False: blocks. Report what you find.
3. Check that mosquito_obj_tracker.py, moving_obj_tracker.py, resting_obj_tracker.py, single_video_analysis.py, and flight_center_distance_analysis.py are all still present and unmodified in their core logic (these must not have had functions removed).
4. Count the total number of .py files in BuzzSuite/ (excluding __pycache__). Report the count.
5. Confirm that no file in BuzzSuite/ imports from BuzzWatch_app_v2/ or BuzzWatch_Fru_2/ directly (all imports must be local to BuzzSuite/).

Report issues. Under 300 words.
```

---

### Step 7 — Make it user-friendly

**Goal:** A non-expert user can run a complete analysis without reading any documentation.

**Specific changes:**

1. **Startup screen**: Replace the blank window on first launch with a welcome panel
   showing: (a) a "New experiment" button that launches the intake wizard, (b) a "Load
   recent experiment" list populated from `config.json` (last 10 experiments), (c) one-line
   descriptions of the three analysis types.
2. **Inline help**: Every settings parameter must have a tooltip (`ToolTip` widget on
   hover) with a plain-English description of what it does and typical values.
3. **Status bar**: Add a persistent status bar showing: current experiment name, module
   active, last action completed, and a spinner when processing.
4. **Error handling**: Wrap all file I/O and analysis calls in try/except with
   user-visible error dialogs (`tkinter.messagebox.showerror`) instead of silent failures or
   terminal tracebacks. Common cases: experiment folder not found, empty
   `final_tracking_data/`, missing `custom_zones.json` when running BuzzPhono, corrupt
   `.pkl`, ffmpeg not found.
5. **Progress dialogs**: Replace raw log text for long operations with a modal progress
   dialog showing percentage complete and estimated time remaining.
6. **Consistent naming in UI**: All UI labels use the public names: "BuzzWatch (Activity)",
   "BuzzSwarm (Aggregation)", "BuzzPhono (Phonotaxis)". No internal code names visible to
   users.
7. **Settings persistence**: When the user changes any analysis parameter, save it back to
   `config.json`.

**Verification — spawn a subagent with this prompt:**
```
You are a code reviewer and UX auditor. Your job is to verify Step 7 of a BuzzSuite build.

1. Open buzzwatch_analysis_app.py and its UI manager. Confirm a startup/welcome panel or module-selector exists (not just a blank notebook).
2. Search BuzzSuite/ for "ToolTip" or "tooltip" — confirm at least 5 parameter widgets have tooltips wired up.
3. Search for tkinter.messagebox — confirm it is used in at least 3 different error-handling locations.
4. Search for "BuzzWatch_V3", "BuzzPhono", "BuzzSwarm" in UI label strings (these should not appear as raw internal names — they should be user-facing names like "BuzzWatch (Activity)").
5. Check config_manager.py — confirm it saves analysis parameters back to config.json (look for a save/write call triggered from the tab managers).
6. Confirm python -c "import ast; ast.parse(open('buzzwatch_analysis_app.py').read())" succeeds (proxy for syntax correctness).

Report UX gaps and any code issues. Under 400 words.
```

---

## General rules for all steps

- **Never modify** the data drive (e.g. `E:\Buzzwatch\` / `/Volumes/Mosquito2/Buzzwatch`) —
  it is read-only
- **Never commit** to the `BuzzWatch_app_v2/` or `BuzzWatch_Fru_2/` git repos — work only in
  `BuzzSuite/`
- **Python 3.8 compatibility** — do not use walrus operator (`:=`), `match` statements, or
  any 3.9+ features
- **One `__init__.py`** per sub-package (`buzzwatch_data_analysis/`, `buzzswarm/`,
  `buzzphono/`)
- **No new dependencies** — use only packages already in `environment.yml` / the existing
  conda env
- **Preserve all `.pkl` compatibility** — the compat shim in `buzzswarm/_compat_shim/` must
  remain; never change the data serialisation format
- **The BuzzWatch_V3 overhaul config is the only tracking config** —
  `OVERHAUL_TRACKING` from `tracking_overhaul.py` is always applied; do not keep any UI
  toggle for old vs new tracker
- **Cross-platform**: no hardcoded drive letters or `/Volumes/...` paths in business logic
  — route through `path_utils.py`; ffmpeg resolution must work on both Windows and macOS;
  keep all Tk work on the main thread
- **Tracking numerics and `.pkl` format are frozen** — modernize the app shell, concurrency,
  and (opt-in, CPU-fallback) GPU path, but never change what the tracker computes
- **Update `DEVLOG.md` after every step** with what changed and how to resume — this is the
  durable handoff if work continues on a different machine with a fresh agent
- **Pause after each step** and wait for the user's explicit go-ahead before starting the
  next one
- After completing all 7 steps, run a final end-to-end test: load an existing experiment
  from the data root's `Analysis\Phonotaxis\Penzance\AedesAegyptiLiv475M_6_4-5_overhaul\`,
  run batch tracking (or skip if already tracked), run BuzzPhono analysis, run BuzzSwarm
  analysis, confirm output PDFs appear in the experiment's `plots/` folder

---

## Future work (planned — NOT yet implemented)

> Captured specs for later work. **None of this is wired into the running app yet.** It is
> documented here so a future agent can implement it without re-deriving the data model. When you
> pick one up, follow the same working agreement as the build steps (implement → verify with a
> sub-agent → update `DEVLOG.md`).

### Human- and AI-readable tracking-data export

**Goal.** Let a user (and downstream ML) read the tracking results **outside** the app, without
unpickling BuzzSuite's classes. Export each experiment's per-segment tracking into a
self-describing dataset that is (a) directly usable as **training data for deeper
behaviour-detection models** and (b) human-inspectable in Excel/pandas. Primary bulk format is
**Parquet** (typed, columnar, compressed); a small CSV summary + a JSON manifest are the
human-readable companion layer.

**Format decision (settled — do not re-litigate).** The bulk per-detection export is **Parquet**,
not CSV. Full per-detection output is tens of millions of rows / multiple GB per experiment; CSV has
no schema/types (timestamps become strings, `track_id` degrades to float), no compression, and no
columnar/partitioned reads — an ML data loader pays for all of it. Parquet is typed, compressed,
columnar, Hive-partitionable for selective/incremental reads, stores nested per-track sequence views
natively, and is consumed directly by pandas/polars/DuckDB/PyTorch & TF data loaders/HuggingFace
`datasets`. This adds **exactly one dependency, `pyarrow`**, accepted here because this exporter is
*separately scoped*: it is read-only w.r.t. the tracker, opt-in, and touches none of the frozen
tracking/analysis numerics or `.pkl` format that the project-wide "no new dependencies" rule exists
to protect. (DuckDB was considered and rejected as unnecessary: pyarrow-written Parquet is already
queryable by DuckDB/polars/pandas without adding DuckDB itself.)

**Read-only w.r.t. the tracker.** This is an *export*, never a re-computation: load the frozen
`.pkl`s and emit tables. Do **not** change the tracker, the `.pkl` format, or any numerics. Load via
`buzzswarm/fru2_zt_normalized_analysis.py::load_tracking_file` (or an equivalent CompatUnpickler) so
cross-version / `numpy._core` pickles still open.

**Source data model (already on disk — this is exactly the schema to flatten).**
Each `final_tracking_data/forward_mosq_tracks_{Cage}_{YYYYMMDD}_{HHMMSS}_{NNNNN}` file is a pickled
tracker object `mt` where:
- `mt.time_stamp` — list of per-frame `datetime`s, indexed by **absolute frame index**.
- `mt.objects` — `OrderedDict` mapping `track_id -> { ... }`:
  - `'coordinates'`: list of `(x, y)` pixel positions, one per frame the track exists;
  - `'state'`: list of per-frame ints — **`1` = flying/moving, `0` = resting**
    (see `single_video_analysis.py` where `state[frame] = 1 # moving` / `= 0 # resting`);
  - `'start'`: absolute frame index where this track's `coordinates`/`state` begin;
  - (plus `'end'`, `'start_boot'`, `'end_boot'` bookkeeping fields).
  - Align a track's k-th point to wall-clock time via `mt.time_stamp[start + k]`.
  (This is precisely what `fru2_zt_normalized_analysis.extract_raw_flights(mt)` walks — reuse it as
  the reference reader; flights are the `state == 1` runs.)

**Output A — per-detection dataset (one row per detection): the ML training substrate.** Parquet,
Hive-partitioned as `detections/session_date=YYYYMMDD/segment=NNNNN/part-0.parquet`. Columns with
their Arrow types: `segment` (int32), `cage` (dictionary/string), `frame_index` (int64),
`timestamp` (timestamp[us], from `mt.time_stamp`), `track_id` (int64), `x`/`y` (float32, raw pixel
coords — do **not** bake in cage-centre distance; a consumer can join `calculate_cage_centroid` if
wanted), `state` (int8), `state_label` (dictionary/string), plus cheap derived features
`dx, dy, step_px, speed_px_per_frame` (float32) and `is_flying` (bool). `session_date`/`session_time`
are the partition keys. Compression snappy by default (zstd optional). Partitioning lets a later run
export only new dates/segments whose source `.pkl` is newer than the manifest record — incremental,
not a full re-export.

**Output A′ — nested per-track sequence view: `tracks.parquet`.** The same records grouped into
per-track sequences (`coordinates`, `state`, `timestamp` as Arrow *list* columns, keyed by
`track_id` + `segment`) — the shape sequence/behaviour models want directly, which Parquet stores
natively and CSV cannot.

**Output B — per-track summary (one row per trajectory): the human-facing overview.** Stays **CSV**
(`track_summary.csv`) — a few thousand rows, opens fine in Excel. Columns: `track_id, segment,
start_frame, end_frame, n_frames, n_flying_frames, n_resting_frames, duration_s, path_length_px,
mean_speed_px_s, xmin, xmax, ymin, ymax`. Optionally also emit `sample_segment.csv` (one segment's
detections) purely for eyeballing the long-table shape.

**Output C — manifest** (`exports/manifest.json`): experiment alias, cage, source `.pkl` list +
mtimes, a column dictionary (name → dtype + units + the meaning of `state`), the partitioning scheme,
fps if known, the compression codec, the `pyarrow` version used, and a schema/generator version — so
the export is self-describing to a human or an LLM with no access to this codebase.

**Where it plugs in (mirror the existing BuzzSwarm entry-script pattern).**
- New top-level module `tracking_export.py` with a pure, importable
  `export_experiment_tracks(experiment_dir, out_dir=None, combined=True) -> out_dir` — no path
  globals; output defaults to `{experiment_dir}/exports/` (routed via the passed dir / `path_utils`,
  same tree the app already writes plots into).
- Surface the *same* callable from three places: a button on the Activity / Preliminary-Analysis tab
  ("Export tracking data"), a `buzzsuite_cli.py` subcommand (`export`), and a Dashboard step —
  exactly like BuzzSwarm's `fru2_sholl_csv.run`.
- **Import `pyarrow` lazily inside the export function** (Step-5 lazy-import rule) so app cold-start
  and the rest of the app stay dependency-neutral. Always write `track_summary.csv` + `manifest.json`
  (stdlib) as the human layer; if `pyarrow` is somehow unavailable, fall back to a CSV per-detection
  dump and record that degradation in the manifest.
- Stream one `.pkl` at a time and `del` + `gc.collect()` between (the Step-5 memory rule); large
  experiments are 100+ segments.

**Constraints for whoever builds it.** Python 3.8-clean. **Adds exactly one dependency, `pyarrow`,
declared in `environment.yml`** (conda-forge). Pin for Python 3.8: `pyarrow>=14,<17` — 16.1.0 is the
last release with Python-3.8 wheels; 17.0 requires 3.9+. No other new dependencies are added
(`fastparquet`, `duckdb`, and `polars` are **not** added). Reuse the compat unpickler
(`fru2_zt_normalized_analysis.load_tracking_file`). All output goes under the experiment's own
`exports/` folder — never a hardcoded location, never the raw `Recording\` tree.

---

## How to spawn a verification subagent (Claude Code syntax)

After each step, use the `Task` tool to spawn a verification agent:

```python
Task(
    description="Verify Step N of BuzzSuite build",
    prompt="""<paste the verification prompt from above>"""
)
```

Do not proceed to the next step until the subagent returns with no blocking issues. If it
reports issues, fix them before moving on. Then update `DEVLOG.md`, and wait for the user's
go-ahead before starting the next step.
