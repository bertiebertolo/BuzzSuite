# BuzzSuite

A unified Python/Tkinter desktop app for mosquito behavioural-assay analysis. BuzzSuite
takes you from raw Raspberry Pi video all the way to finished behavioural plots — without
leaving the app or touching the command line.

It combines three analysis pipelines behind one module selector, one shared tracking
back-end, and one experiment format:

| Module | What it does |
|---|---|
| **Activity** | Tracks flying/resting mosquito activity per cage from raw video (motion-aware Hungarian-assignment tracker with blob splitting for crowded frames). |
| **BuzzSwarm** | Aggregation/swarming analysis — Sholl-style CDFs of flight positions vs. cage centre, keyed on the r50 (median radial distance) metric. |
| **BuzzPhono** | Phonotaxis analysis — speaker-zone resting fraction from a user-drawn polygon zone, comparing stimulus ON vs. OFF windows. |

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

## Install

Requires **Python 3.8**. Conda is the primary, tested path:

```bash
conda env create -f environment.yml
conda activate buzzsuite_env
```

Or with plain pip (a `.venv` is picked up automatically by `run_buzzsuite.bat` if present):

```bash
python3.8 -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux
```

### ffmpeg

The H264→MP4 converter needs `ffmpeg`/`ffprobe`. They aren't bundled in this repo (they're
~200 MB each — see [`h264_converter/ffmpeg/README.md`](h264_converter/ffmpeg/README.md)).
Easiest: install ffmpeg system-wide and make sure it's on `PATH`.

- **Windows:** [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) or `winget install ffmpeg`
- **macOS:** `brew install ffmpeg`

## Run

```bash
python run_gui.py
```

or, on Windows, double-click `run_buzzsuite.bat` (uses a local `.venv` if present, else the
`buzzsuite_env` conda environment, else system Python).

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
python buzzsuite_cli.py swarm   --experiment <exp-dir> --which all       # BuzzSwarm analysis
python buzzsuite_cli.py phono   --experiment <exp-dir>                   # BuzzPhono analysis
python buzzsuite_cli.py export  --experiment <exp-dir>                   # tracking → Parquet/CSV
python buzzsuite_cli.py convert --in-dir <raw-video-folder>              # .h264 → .mp4
```

Run `python buzzsuite_cli.py <subcommand> --help` for the full flag list.

## Project layout

```
run_gui.py                  ← GUI entry point
buzzsuite_app.py             ← main Tk application
buzzsuite_cli.py             ← headless CLI entry point
path_utils.py                ← cross-platform data-root resolution
experiment_manager.py        ← experiment_{alias}.json + folder structure
batch_processing_tab_manager.py, *_tab_manager.py   ← per-tab UI + orchestration
buzzwatch_data_analysis/     ← the tracking engine (see note below)
buzzswarm/                   ← BuzzSwarm (Sholl/r50) aggregation engine
buzzphono/                   ← BuzzPhono speaker-zone phonotaxis analysis
h264_converter/               ← H264→MP4 conversion (ffmpeg wrapper)
docs/                        ← user guide
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
