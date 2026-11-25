# BuzzSuite

A unified Python/Tkinter desktop app for mosquito behavioural-assay analysis. BuzzSuite
takes you from raw Raspberry Pi video all the way to finished behavioural plots — without
leaving the app or touching the command line.

It combines three analysis pipelines behind one module selector, one shared tracking
back-end, and one experiment format:

| Module | What it does |
|---|---|
| **BuzzWatch (Activity)** | Tracks flying/resting mosquito activity per cage from raw video (motion-aware Hungarian-assignment tracker with blob splitting for crowded frames). |
| **BuzzSwarm (Aggregation)** | Aggregation/swarming analysis by real clock time, one calendar day at a time: swarm size (mosquitoes in flight in the cage centre at the same moment), heatmaps and a centre profile of where they fly, speed, turning and nearest-neighbour distance, per clock window and through the day. The swarm zone's radius is set per experiment on a picture of the cage, and the other zone sizes follow it. The same analysis runs headless with `buzzsuite_cli.py swarm --which clock`; the older Sholl/r50 analyses remain available from the CLI. |
| **BuzzPhono (Phonotaxis)** | Phonotaxis analysis — does a sound draw mosquitoes to the speaker? Speaker-zone resting fraction during the tone vs. just before, every hour, with cluster-robust statistics, sham and control-zone checks, time courses, time of day, spatial maps, directed flight, flight phonotaxis, reliability and sensitivity. Its own package and browser app (`src/buzzphono/`, environment `buzzphono_env`); see the [BuzzPhono guide](docs/BUZZPHONO_GUIDE.md). |
| **yolo_export** | Training data for a live (on-camera) mosquito detector, built from BuzzWatch tracking and the original videos: YOLO tile datasets, calibration tiles, clips and QA. Command line only. |

A bundled **H264 → MP4 Converter** handles the raw-video intake step (lossless remux, with
automatic SPS/PPS header recovery for multi-segment Raspberry Pi recordings).

## The end-to-end workflow

```
1. Select raw video folder (.h264 or .mp4 files from the Raspberry Pi)
2. Select which dates / recording sessions to include
3. Convert .h264 → .mp4 in-app (lossless remux)
4. App creates the experiment folder structure and experiment_{alias}.json for you
5. Choose one or more analyses: Activity / BuzzSwarm / BuzzPhono
6. Run the chosen analyses — tracking, then aggregation/phonotaxis
7. Output PDFs/PNGs land in the experiment's plots/ folder, viewable in-app
```

Multiple experiments can run at once, each in its own background thread driving its own
process pool, with a dashboard showing per-experiment progress.

**A full click-by-click walkthrough of this workflow is in
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)** — start there if you're new to the app.
(A LaTeX source for a printable PDF version is also included, at
[`docs/user_guide.tex`](docs/user_guide.tex).)

## Install and launch

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda, and a copy of
this folder (`git clone https://github.com/bertiebertolo/BuzzSuite.git`, or download the release
zip). There are two environments: `buzzsuite_env` (Python 3.8) for the desktop app, which holds
BuzzWatch, BuzzSwarm and yolo_export, and `buzzphono_env` (Python 3.12) for BuzzPhono. Each is
installed once; run the commands below in the BuzzSuite folder (on Windows, in "Anaconda Prompt").
To update later, run `git pull` in the same folder.

> **On Windows, the short way:** install Anaconda, open "Anaconda Prompt", `cd` to the BuzzSuite
> folder and run `setup_env.bat`. It installs **both** environments and checks them (it takes a
> while). Then double-click `run_buzzsuite.bat`.
>
> **You need both environments.** With only `buzzsuite_env`, the app opens but BuzzPhono does not:
> see [Troubleshooting](#troubleshooting).

### BuzzSuite app: BuzzWatch (Activity) and BuzzSwarm (Aggregation)

**Install**

```bash
conda env create -f environment.yml          # creates buzzsuite_env
conda activate buzzsuite_env
python check_install.py                       # should end with "All checks passed"
```

`environment.yml` pins the exact versions tracking was validated with. On Windows, run
`setup_env.bat` from "Anaconda Prompt" instead: it creates the environment, swaps in the PyPI
numpy/matplotlib wheels (the conda-forge builds crash plotting on PCs with Memory Integrity +
CET), runs `check_install.py`, and then creates BuzzPhono's `buzzphono_env` too (below). It is
safe to run again: environments that already exist are only checked. Run
`python check_install.py` any time to re-check an install.

Or with plain pip (a `.venv` is picked up automatically by `run_buzzsuite.bat` if present):

```bash
python3.8 -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux
```

The H264→MP4 converter also needs `ffmpeg`/`ffprobe`. They aren't bundled in this repo (they're
~200 MB each — see [`h264_converter/ffmpeg/README.md`](h264_converter/ffmpeg/README.md)).
Install ffmpeg system-wide and make sure it's on `PATH`:

- **Windows:** [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) or `winget install ffmpeg`
- **macOS:** `brew install ffmpeg`

**Launch:** on Windows, double-click `run_buzzsuite.bat` (it uses a local `.venv` if present, else
`buzzsuite_env`, which it finds even when conda is not on `PATH`). Anywhere:

```bash
conda activate buzzsuite_env
python run_gui.py
```

### BuzzPhono (Phonotaxis)

**Install**

```bash
conda env create -f environment-buzzphono.yml   # creates buzzphono_env
conda activate buzzphono_env
buzzphono --version
```

This installs exact, tested versions of every package (`constraints-buzzphono.txt`) and the
`buzzphono` command (on Windows, `setup_env.bat` does this step for you). BuzzSuite never imports
BuzzPhono: it runs it in this environment, and if the environment is missing, BuzzSuite says so
and how to install it.

**Launch:** double-click `BuzzPhono.bat` (Windows) or `BuzzPhono.command` (macOS), click
**Open the BuzzPhono app** in BuzzSuite's **3 · Plotting → Speaker custom zone (Phonotaxis)** tab,
or run `buzzphono app`. The app opens in your web browser at <http://127.0.0.1:8501> and stays on
this computer. Streamlit needs no setup first: on a computer that has never run it, the app still
opens straight away (no sign-up or e-mail prompt). If `buzzphono_env` is somewhere unusual, set
`BUZZPHONO_PYTHON` to its `python` executable.

### yolo_export

**Install:** nothing extra; it runs in `buzzsuite_env`. Check it with
`python -m yolo_export.selftest` (ends with "all checks passed").

**Launch:** from the command line only, `python buzzsuite_cli.py yolo <action>` (see the quick
start below). Output goes to a `BuzzWatch_LiveDetector` folder next to the data root, never inside
a Buzzwatch data tree.

## Quick start

### BuzzWatch (Activity)

The full click-by-click version is in [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

1. Launch BuzzSuite and click **New experiment →** on the welcome screen.
2. **1 · Setup → Experiment & Video:** choose the raw video folder, tick the days, convert
   `.h264` to `.mp4`, name the experiment and click to create it.
3. **1 · Setup → Background & Cage Border:** **Extract Images from Video**, **Get Background from
   Images**, then **Draw Cage Borders** and click the four cage corners.
4. **2 · Analysis → Single experiment** (or **Multiple experiments** for several at once): track
   all untracked segments.
5. **3 · Plotting → Activity (in-cage):** add the experiment, tick the days, **Plot**, **Save PNG**.

### BuzzSwarm (Aggregation)

Needs a tracked experiment with a cage border (BuzzWatch steps 1-4).

1. **3 · Plotting → Swarming (Aggregation):** **Add experiment**.
2. **Swarm zone…** to set the swarm zone's radius on a picture of the cage (100 px suits the old
   720p camera; a higher-resolution camera needs more). The central and core zones, heatmap cells
   and the flight-speed threshold scale with it. Skip this to keep the 100 px default.
3. Tick the days, keep the default clock windows (or add your own), then **Run analysis**.
   **Redraw plots** changes the plots without re-reading the tracking. Plots and CSVs go to the
   experiment's `plots/buzzswarm/`.

Headless: `python buzzsuite_cli.py swarm --experiment <exp-dir> --which clock`.

### BuzzPhono (Phonotaxis)

Try it on simulated data first: `buzzphono demo buzzphono_demo --open` (a few minutes).

For your own experiment, open the app and work through its pages: **Load** (add the experiment's
analysis folder) → **Calibrate** (the speaker zone first; control zones follow on the other walls)
→ **Validate** → **Analyse** → **Results** → **Export**. Or from the command line:

```bash
buzzphono analyse "<experiment folder>" -a all --open
```

The [BuzzPhono guide](docs/BUZZPHONO_GUIDE.md) explains every analysis, the statistics and the
settings.

### yolo_export

```bash
python buzzsuite_cli.py yolo inventory            # find experiments, videos and rigs
#   check BuzzWatch_LiveDetector/inventory/inventory.csv; choices go in inventory/selection.yaml
python buzzsuite_cli.py yolo all --workers 8      # annotate -> verify -> clock -> frames -> build -> clips -> qa
```

Each action can also be run on its own (`--help` lists them). A segment whose worker crashes or
runs past `--task-timeout` (default 30 min) is skipped and logged with its file name; the run
carries on.

## Troubleshooting

| What you see | What to do |
|---|---|
| **"The BuzzPhono app stopped before its page could open"** and `EnvironmentLocationNotFound: Not a conda environment: …\envs\buzzphono_env` (versions before 1.0.0), or **"BuzzPhono's environment (buzzphono_env) is not installed on this computer"** | BuzzPhono's environment was never installed on this PC; only the app's was. In "Anaconda Prompt", in the BuzzSuite folder, run `conda env create -f environment-buzzphono.yml` (or run `setup_env.bat`, which does it). It takes a few minutes. Then click **Open the BuzzPhono app** again. |
| BuzzPhono is still not found, although `buzzphono_env` exists | It is somewhere unusual. Set the `BUZZPHONO_PYTHON` environment variable to its `python.exe` (for example `…\envs\buzzphono_env\python.exe`). |
| The BuzzPhono app stops before its page opens, with another message | The message shows the last lines of `.buzzphono_app.log` in your home folder. For the full error, run `conda activate buzzphono_env` and then `buzzphono app` in Anaconda Prompt. |
| Plotting crashes or Python closes when a plot is drawn (Windows) | The conda-forge numpy/matplotlib builds crash on PCs with Memory Integrity + CET. Run `setup_env.bat`, which installs the PyPI builds instead. |
| `run_buzzsuite.bat` says "buzzsuite_env was not found" | The app's environment is not installed. Run `setup_env.bat` from Anaconda Prompt in the BuzzSuite folder. |

## The data folder layout

BuzzSuite stores raw recordings and analysis outputs under a single data root, split into
parallel `Recording\` and `Analysis\` trees sharing the same
`{AssayType}\{Location}\{ExperimentName}\` hierarchy:

```
{DataRoot}\
├── Recording\{AssayType}\{Location}\{CageName}\      ← raw .h264 / .mp4
└── Analysis\{AssayType}\{Location}\{Experiment}\      ← tracking output + plots
    ├── experiment_{alias}.json
    ├── final_tracking_data\
    ├── tracking_moving\ / tracking_resting\
    └── plots\
```

The app resolves this root automatically (`path_utils.py`) — checking a `Buzzwatch/`
folder next to the app, then common mount points, then the `BUZZSUITE_DATA_ROOT`
environment variable. Set that variable, or use the in-app "Browse…" data-root picker, to
point BuzzSuite at your own data drive without editing any code.

## Command-line interface

For unattended/overnight batch runs, `buzzsuite_cli.py` wraps the same functions the GUI
buttons call — no separate code path, byte-identical results:

```bash
python buzzsuite_cli.py track   --experiment <dir-or-json> --workers 8   # batch-track segments
python buzzsuite_cli.py swarm   --experiment <exp-dir> --which clock     # clock-time swarm analysis + plots
python buzzsuite_cli.py swarm   --experiment <exp-dir> --which all       # older Sholl/r50 BuzzSwarm analyses
python buzzsuite_cli.py phono   --experiment <exp-dir>                   # BuzzPhono analysis (runs in buzzphono_env)
python buzzsuite_cli.py export  --experiment <exp-dir>                   # tracking → Parquet/CSV
python buzzsuite_cli.py convert --in-dir <raw-video-folder>              # .h264 → .mp4
python buzzsuite_cli.py yolo    all --workers 8                          # live-detector training data
```

Run `python buzzsuite_cli.py <subcommand> --help` for the full flag list.

To prepare and analyse every pending experiment under a data root in one go, run
`python prepare_experiments.py --root <data-root>` (backgrounds and cage borders), then
`python run_all_analysis.py --root <data-root>` (tracking, BuzzSwarm, BuzzPhono, export).
Add `--dry-run` to either to see the plan without changing anything.

## Project layout

```
run_gui.py                  ← GUI entry point
buzzsuite_app.py             ← main Tk application
buzzsuite_cli.py             ← headless CLI entry point
path_utils.py                ← cross-platform data-root resolution
ld_dd.py                     ← light schedule (night hours, LD/DD days) shared by the plot tabs
experiment_manager.py        ← experiment_{alias}.json + folder structure
batch_processing_tab_manager.py, *_tab_manager.py   ← per-tab UI + orchestration
buzzwatch_data_analysis/     ← the tracking engine (see note below)
buzzswarm/                   ← BuzzSwarm: clock-time swarm analysis (clock_analysis.py) and
                               its plots (clock_plots.py), plus the older Sholl/r50 scripts
src/buzzphono/               ← BuzzPhono phonotaxis package + app (own env; docs/BUZZPHONO_GUIDE.md)
buzzphono_bridge.py          ← runs BuzzPhono from BuzzSuite (CLI, run_all_analysis, Phonotaxis tab)
yolo_export/                 ← live-detector training data from BuzzWatch tracking
archive/                     ← superseded phonotaxis scripts, kept for provenance
h264_converter/               ← H264→MP4 conversion (ffmpeg wrapper)
tests/buzzphono/             ← BuzzPhono tests (pytest, buzzphono_env; run in CI)
tests/buzzsuite/             ← app, BuzzSwarm tests (python tests/buzzsuite/run_all.py, buzzsuite_env)
tools/                       ← standalone helpers (e.g. collecting the Raspberry Pi's logs)
docs/                        ← user guide, BuzzPhono guide
CHANGELOG.md                 ← what changed in each release
```

## Tests

```bash
# BuzzPhono (buzzphono_env): the CI selection; add -m slow for the long statistical runs
python -m pytest -m "not slow and not golden and not realdata and not browser"
# BuzzSuite app and BuzzSwarm (buzzsuite_env; no pytest needed)
python tests/buzzsuite/run_all.py
# yolo_export (buzzsuite_env)
python -m yolo_export.selftest
```

## A note on the `buzzwatch_data_analysis` package name

The core tracking engine's package is still named `buzzwatch_data_analysis`, and its main
class is still `buzzwatch_experiment_analysis` — these are **not** leftover naming, they're
load-bearing. Every existing `.pkl` tracking file stores this exact module/class path as
part of its serialized format (Python pickles by import path, not by value). Renaming the
package would make every already-tracked experiment unloadable. See
[`NOTICE`](NOTICE) for the attribution this reflects.

Everything else — UI labels, the app module, the conda environment, the CLI, and all
config/env-var names — uses the BuzzSuite name.

## License

MIT — see [`LICENSE`](LICENSE). The tracking engine (`buzzwatch_data_analysis/`) is derived
from the original BuzzWatch project; see [`NOTICE`](NOTICE) for attribution.
