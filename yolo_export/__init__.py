"""yolo_export — training data for the live-detection Pi, built from BuzzWatch tracking.

Turns frozen tracking output (final_tracking_data/forward_mosq_tracks_*) plus the original
videos into:

  inventory    every experiment, its videos, rig id and whether it is used
  annotate     aligned per-detection labels + behaviour for EVERY tracked segment (no video
               decoding; each segment points at its original .mp4) + events + per-frame activity
  clock        corrected wall-clock time per session (fit on the minute-40 flight burst), so
               stimulus ON/OFF and light phase can be attached to frames and events
  frames       sampled full frames (3-frame stack t, t-1, t-2) with per-label pixel QA
  build        YOLO tile datasets (stack / single), splits, data.yaml, Hailo calibration tiles
  clips        short re-encoded clips with MOT ground truth, labels and events
  qa           alignment re-check, statistics, contact sheets and behaviour strips

Read-only on all data: every output goes under one folder outside the Buzzwatch data trees
(default <data-root parent>/BuzzWatch_LiveDetector). Tracking numerics and the .pkl format are
untouched. Entry point: runner.run(action, ...), also reached via `buzzsuite_cli.py yolo`.
"""

GENERATOR_VERSION = "1.3.0"
