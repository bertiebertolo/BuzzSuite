---
name: buzzphono
description: Run, check or extend BuzzPhono phonotaxis analyses (speaker-zone resting fraction ON vs OFF, PSTH, time of day, spatial maps, approach, flight phonotaxis, reliability, sensitivity, experiment comparison) on BuzzWatch tracking data. Use when the user asks to analyse phonotaxis, run or interpret BuzzPhono, calibrate speaker zones, check a Pi config or playback log, reproduce the thesis phonotaxis numbers, or change BuzzPhono code.
---

# BuzzPhono

BuzzPhono lives in `src/buzzphono/` and runs in its own conda environment, `buzzphono_env`
(Python 3.12). BuzzWatch's `buzzsuite_env` (Python 3.8) never imports it. BuzzSuite reaches it
through `buzzphono_bridge.py`, which `buzzsuite_cli.py phono` / `phono-flight`,
`run_all_analysis.py` and the Phonotaxis tab use.

Read `docs/BUZZPHONO_GUIDE.md` for what each analysis means and `src/buzzphono/CLAUDE.md` for code
conventions before changing code.

## Interpreter

Always call BuzzPhono's interpreter by full path; plain `python` may be another environment.

- Windows: `%USERPROFILE%\anaconda3\envs\buzzphono_env\python.exe -m buzzphono ...`
- macOS/Linux: `~/anaconda3/envs/buzzphono_env/bin/python -m buzzphono ...` (or `miniconda3`, `miniforge3`)
- `buzzphono_bridge.buzzphono_command()` resolves it the same way BuzzSuite does.

## Common tasks

| user wants | do |
|---|---|
| check an experiment | `python -m buzzphono validate <exp> [--pi-config config.ini] [--playback-log logs.zip]` |
| headline result | `python -m buzzphono analyse <exp> -a resting_fraction` |
| everything | `python -m buzzphono analyse <exp> -a all` |
| several cages / saved settings | write a YAML settings file (guide §9), `python -m buzzphono analyse -c file.yml` |
| reproduce the thesis numbers | `--clock thesis -a resting_fraction` (golden test: `pytest -m golden`) |
| draw or fix zones | the app's Calibrate page (`python -m buzzphono app`), or `python -m buzzphono calibrate <exp> --speaker "x,y ..." --control "name=x,y ..."` |
| compare finished runs | `python -m buzzphono compare <run> <run> -o <dir>` |
| demo / sanity check | `python -m buzzphono demo <dir>` (known planted effect, plus a null cage) |

**Long real-data runs** (many cages, `-a all`, first ingest from an HDD): give the user the exact
command to paste rather than starting it. Use below-normal priority and `--workers 4` for a hard
drive. For example, on Windows:
`cmd /c start /belownormal cmd /k "<python> -m buzzphono analyse -c settings.yml --workers 4"`.

## Reading results

- **Primary:** `experiments/<label>/resting_fraction/statistics.csv`, row
  `ON vs OFF (speaker zone)`, phase `all`. Report `mean_delta` with
  `wild_cluster_ci_low/high` and `wild_cluster_p_value`; per-hour Wilcoxon p-values overstate the
  evidence.
- **Before claiming an effect, check that:**
  - the sham row is near 0;
  - the control-zone row is near 0;
  - `speaker minus control zones` is positive;
  - the sensitivity table has no flags on small zone changes.
- Approach: only `bouts_per_flight_s` is validated. Radial velocity failed the Aedes positive
  control; never present it as evidence of approach.
- With fewer than 5 cages, say that between-cage conclusions are limited (`reliability/summary.csv`
  warns).

## Safety rules

- Tracking data and the data drives are read-only. Analysis writes go to `--out` or the default
  `<exp>/plots/buzzphono/`. Tests write only to temporary folders.
- Never change the tracker, the `.pkl` format or `buzzsuite_env`.
- Real-data numbers, cage names and personal paths stay out of anything pushed to GitHub. Golden
  fixtures and `tests/buzzphono/golden/local_paths.json` are local only.
- Checks after code changes: `ruff check src tests tools`, `mypy`, and
  `pytest -m "not slow and not golden and not realdata and not browser"`. Add `-m golden` on the
  analysis PC when the reading, clock or resting-fraction code changed.
