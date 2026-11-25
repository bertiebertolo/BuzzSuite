# Changelog

All notable changes to BuzzSuite. Dates are release dates; the [DEVLOG](DEVLOG.md) has the
day-by-day working notes.

## [1.0.0] — 2026-11-25

The first release. BuzzSuite takes a user from raw Raspberry Pi video to finished behavioural
plots for three assays, in one desktop app plus BuzzPhono's own browser app.

### What's in it

- **BuzzSuite app** (`run_gui.py`, `buzzsuite_env`, Python 3.8): welcome screen, intake wizard
  (raw video → `.mp4` → experiment folder and `experiment_{alias}.json`), background and cage
  border set-up, single and multi-experiment tracking with a dashboard, cancel and progress, status
  bar, tooltips, and settings saved between sessions. Windows and macOS.
- **BuzzWatch (Activity)**: the motion-aware tracker (Hungarian assignment, constant-velocity
  model, backward pass, blob splitting), background caching per session, an opt-in OpenCL path with
  a CPU fallback, and activity plots by day (individual, overlay, continuous; LD/DD tags). Tracking
  numerics and the `.pkl` format are unchanged from the validated BuzzWatch engine.
- **BuzzSwarm (Aggregation)**: swarm analysis by real clock time, one calendar day at a time —
  swarm size, central enrichment, core density, shape index, speed, turning, nearest-neighbour
  distance and "flying together" — per clock window and through the day, with heatmaps, a centre
  profile and per-day caches. The older Sholl / r50 analyses stay available from the CLI.
- **BuzzPhono (Phonotaxis)** 1.0.0 (`src/buzzphono`, `buzzphono_env`, Python 3.12): speaker-zone
  resting fraction with cluster-robust statistics, sham and control zones, attraction index, PSTH,
  time of day, spatial maps, directed flight, flight phonotaxis, reliability, sensitivity and
  comparisons; a corrected recording clock; Raspberry Pi config and playback-log checks; a
  calibration page (speaker zone first, control zones placed on the other walls); reports,
  manifests, a command line and a Streamlit app. See the [BuzzPhono guide](docs/BUZZPHONO_GUIDE.md).
- **yolo_export** 1.3.0: training data for the live detector from BuzzWatch tracking — inventory,
  aligned annotations, clock fit, frame stacks with pixel QA, YOLO tile datasets with a junk
  filter, Hailo calibration tiles, clips and QA.
- **Command line** (`buzzsuite_cli.py`: track, swarm, phono, phono-flight, export, convert, yolo) and
  batch scripts (`prepare_experiments.py`, `run_all_analysis.py`) that call the same functions as
  the app.

### Changes in the last release cycle (since 2026-11-19)

- **yolo_export:** a worker that crashes outright (e.g. an access violation on a numpy-2-era
  pickle under numpy 1.x) or runs past `--task-timeout` (default 30 min) now fails only its own
  segment, logged with the experiment and file name; the run no longer hangs. An error or Ctrl-C
  stops the workers at once. Review fixes the November rebuild had missed are restored:
  relocatable `data.yaml`, one split per segment, the cage-polygon requirement, a raw sort-order
  guard, single `test_days` values, numpy bools in JSON and a warning for an unreadable schedule.
- **BuzzSwarm:** the central and core zones, shape rings, heatmap cells and the real-flight speed
  threshold now scale with the swarm-zone radius chosen for the experiment (unchanged at the
  100 px default). `geometry.json` records the values used.
- **BuzzSwarm:** an experiment without a cage border no longer fails a run half way. Run analysis,
  the swarm-zone picker and `swarm --which clock` say which experiments need a border, run
  nothing, and offer to open the experiment in Setup to draw it. No border is ever guessed.
- **BuzzSwarm:** choose the swarm-zone radius on a picture of the cage; resizable controls panel.
- **BuzzPhono:** the app opens on a computer that has never run Streamlit (it used to wait
  invisibly for an e-mail address); the launcher reports why if the app stops before its page
  opens; an old `buzzphono/` folder in the current folder can no longer hide the package.
- **BuzzPhono / install:** on a PC where BuzzPhono's environment was never created, BuzzSuite now
  says so and how to install it (it used to fail with "EnvironmentLocationNotFound: Not a conda
  environment"), and finds `buzzphono_env` in any conda envs folder. `setup_env.bat` now creates
  `buzzphono_env` as well as `buzzsuite_env`. The README has a Troubleshooting section.
- **App:** a second experiment opened in one session now reads its own settings (cage border
  included) instead of keeping the first one's.
- **tools/feeder_check:** diagnostics for empty feeder zones in BuzzSwarm heatmaps (labelled
  tracking clips, before/after heatmaps, resting vs flying, untracked dark spots), using the same
  radius-scaled real-flight threshold as BuzzSwarm.
- **Tests:** `tests/buzzsuite` (app and BuzzSwarm; runs without pytest in `buzzsuite_env`), a
  crash/hang/timeout check in the yolo_export self-test, and a fix for BuzzPhono test runs in which
  every multi-worker ingest after the app tests failed with "A worker process stopped abruptly"
  (Streamlit's `AppTest` left a page as `__main__`, which spawn workers then re-ran).

[1.0.0]: https://github.com/bertiebertolo/BuzzSuite/releases/tag/v1.0.0
