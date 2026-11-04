"""Flight-based phonotaxis analysis (BuzzPhono): does the tone change where and how mosquitoes fly?

Ported from the 2026-10 research scripts (fp_*.py) into importable, experiment-agnostic functions.
Entry point: ``run_flight_phonotaxis(experiment_dir, ...)`` (see pipeline.py). Modules:

  config    constants, FlightParams, zones / cage geometry, chunk listing, frames-per-chunk detection
  extract   per-chunk read-only pass over the tracking pickles -> .npz caches (spawn-safe workers)
  clock     recording-clock calibration on the tone-onset flight burst
  build     per-minute accumulation, hour-cycle metrics, paired ON-vs-OFF statistics
  figures   thread-safe matplotlib figures (saved PNGs and the GUI's inline view)
  report    key-results table, README, flyer-threshold sensitivity
  pipeline  run_flight_phonotaxis()

Not ported yet: the multi-experiment comparison (pooled cages, cross-species figures).
``build.pool_cycle_metrics(tables, params)`` is the hook for it.

Kept light: importing this package pulls in numpy only (heavy modules load inside the functions).
"""
from .config import Cancelled, FlightParams, FlightPhonotaxisError  # noqa: F401


def run_flight_phonotaxis(*args, **kwargs):
    from .pipeline import run_flight_phonotaxis as _run
    return _run(*args, **kwargs)


def default_workers():
    from .pipeline import default_workers as _dw
    return _dw()
