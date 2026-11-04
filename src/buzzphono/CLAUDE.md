# BuzzPhono — conventions for agents and developers

BuzzPhono is the phonotaxis package of BuzzSuite. It has its own environment (`buzzphono_env`,
Python 3.10+, built with 3.12) and never imports BuzzWatch, BuzzSuite or the frozen tracker. The
user guide is `docs/BUZZPHONO_GUIDE.md`; the statistical plan is `docs/STATS_PLAN.md` (frozen —
changes go in its *Changes* section with their evidence).

## Layout

```
cli.py            Typer commands only: parse, call the package, print, map errors to exit codes
pipeline.py       run(config) -> RunResult: the single entry point of the CLI and the app
config.py         pydantic settings (AnalysisConfig); plain-English errors
calibration.py    zones, cage, mm homography; buzzphono_calibration.json
io/               naming, discover, safe_pickle (allow-listed unpickler), ingest + cache, Pi config, playback log
timebase.py       ns time arithmetic, SessionFrames, ClockModel, CorrectedClock, ThesisClock
stimulus.py       burst-onset clock fit
frames.py         ExperimentClock + bin_experiment: per-frame counts in 30-s bins (the shared engine)
metrics/          one module per analysis; cycles.py = hour-cycle tables + paired_comparison
stats.py          every test and effect size (wild cluster bootstrap is primary)
flight_phono/     flight analyses (port of the 2026 package; legacy=True reproduces it exactly)
plotting/         style.py (palette, deterministic save), static.py (figures), figures.py (which, from what)
report.py, manifest.py, jobs.py (status.json / CANCEL / experiment lock), validate.py, compare.py
synthetic/        L1 (cycle-level) and L2 (agent, BuzzWatch-format files) simulators with known truth
app/              Streamlit pages (excluded from coverage; driven by AppTest and Playwright tests)
thesis.py         exact reproduction of the published thesis tables
```

## Rules

- **Core never imports typer, streamlit or plotly** (only `cli.py` and `app/` do).
- **Read-only inputs.** The only files written next to tracking data are the cache (default
  `<exp>/plots/buzzphono/cache`) and, on request, `buzzphono_calibration.json`. Tests write only to
  `tmp_path`; real-data tests write to a local cache and scratch.
- **Units in names.** Columns and labels carry units: `_px`, `_mm`, `_s`, `_frac`, `_per_s`,
  `_per_min`. Pixels become mm only through the calibration homography.
- **Time is int64 nanoseconds** (`timebase.to_ns`/`from_ns`); never float seconds across chunks.
- **Every skip is counted and reported** (warnings travel to the report and the manifest); nothing
  fails silently. Errors for people are `BuzzPhonoError` subclasses with a `hint`.
- **Determinism.** Seeds come from `stats.seed`; figures are saved with `plotting.style.save`
  (fixed SVG salt, no dates) so reruns give identical output hashes.
- **Cache formats are versioned per tier** (`io/cache.py: TIER_VERSIONS`); bump a tier's version
  when its arrays change and only that tier is rebuilt.
- **Spawn-safe.** Worker functions are top-level; BLAS threads are set to 1 in workers. Scripts
  that call BuzzPhono need `if __name__ == "__main__":`.
- Python ≥ 3.10 syntax, `from __future__ import annotations`, strict mypy, ruff (line length 110),
  British English in user-facing text.
- No real-data numbers, cage names or personal paths in public files. Golden fixtures and
  `tests/buzzphono/golden/local_paths.json` stay local.

## Checks before committing

```bash
ruff check src tests tools && ruff format --check src tests tools && mypy
pytest -m "not slow and not golden and not realdata and not browser"     # the CI set
pytest -m "golden"                                                        # on the analysis PC
pytest tests/buzzphono/e2e/test_browser.py                                # needs Edge or Chrome
```

## Adding an analysis

1. Put the computation in `metrics/<name>.py`, taking `Binned` / cycle tables / cache arrays and
   returning tidy DataFrames.
2. Toy test with hand-computed values, an L2 recovery test (planted effect found) and a null test.
3. Register it: `config.Analysis`, `pipeline.ANALYSES`, figures in `plotting/figures.py`, the app's
   `ANALYSES` list, the guide (section 4) and the report if it has a headline table.
