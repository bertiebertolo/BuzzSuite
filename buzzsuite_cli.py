#!/usr/bin/env python
"""BuzzSuite CLI — unattended / headless / overnight batch runner.

Thin command-line wrapper around the *same* callables the GUI buttons invoke, so results are
byte-identical to running from the app (no separate code path, no numeric divergence):

    track  ->  batch_processing_tab_manager._run_batch_analysis  (multiprocessing.Pool)
    swarm  ->  buzzswarm.fru2_sholl_csv / fru2_pooled_analysis / fru2_per_trajectory_plots .run()
    phono  ->  buzzphono.plot_speaker_distance_phonotaxis.run_speaker_distance_analysis()

Intended for the common workflow: set up + initialize experiments in the GUI during the day,
then run the heavy tracking/analysis unattended (SSH, nohup, tmux, cron) without a display.
It does not need Tk to be usable (matplotlib is forced to the headless Agg backend).

Examples
--------
    python buzzsuite_cli.py track --experiment <dir-or-json> --workers 8
    python buzzsuite_cli.py track --experiment <json> --force            # re-track everything
    python buzzsuite_cli.py swarm --experiment <exp-dir> --which all
    python buzzsuite_cli.py phono --experiment <exp-dir>                 # dates auto-derived
    python buzzsuite_cli.py phono --experiment <exp-dir> --resting-only

Notes
-----
* Run it from the BuzzSuite/ directory (or anywhere — it adds its own dir to sys.path).
* Use the conda env python: /opt/anaconda3/envs/buzzsuite_env/bin/python (macOS system python3
  crashes Tk imports).
* `track` needs an experiment_*.json (paths + settings). `swarm`/`phono` accept either the JSON
  or the experiment directory (they read final_tracking_data/ etc. relative to it).
* Tracking precondition: a segment is only tracked if its Initialization background image
  images_mortality/<video>.png exists (the tracker silently skips segments without one). `track`
  warns about missing backgrounds up front. Generate them via the GUI's Setup tab
  ("Extract Images from Video" / "Get Background from Images" buttons) first.
"""
import argparse
import glob
import os
import re
import sys
import time

# Headless: no display needed; plt.show() becomes a no-op. Set before any matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

_TS_RE = re.compile(r'_(\d{8})_(\d{6})_')   # _{YYYYMMDD}_{HHMMSS}_ in segment filenames


def _log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------- shared helpers
def _resolve_experiment(path, need_json):
    """Accept an experiment_*.json OR an experiment directory containing one.

    Returns (json_path_or_None, exp_dir). If need_json and no JSON is found, exit with a message.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return path, os.path.dirname(path)
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "experiment_*.json")))
        json_path = cands[0] if cands else None
        if need_json and json_path is None:
            raise SystemExit("No experiment_*.json found in %s (required for 'track')." % path)
        return json_path, path
    raise SystemExit("Experiment path not found: %s" % path)


def _derive_range(exp_dir):
    """(start_iso, end_iso) spanning the tracked segments, mirroring the BuzzPhono tab auto-fill."""
    tdir = os.path.join(exp_dir, "final_tracking_data")
    if not os.path.isdir(tdir):
        raise SystemExit("No final_tracking_data/ in %s — nothing to derive a date range from." % exp_dir)
    stamps = []
    for name in os.listdir(tdir):
        m = _TS_RE.search(name)
        if m:
            d, t = m.group(1), m.group(2)
            stamps.append("%s-%s-%s %s:%s:%s" % (d[0:4], d[4:6], d[6:8], t[0:2], t[2:4], t[4:6]))
    if not stamps:
        raise SystemExit("No dated tracking segments in %s." % tdir)
    import pandas as pd
    stamps.sort()
    end = (pd.Timestamp(stamps[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    return stamps[0], end


# ----------------------------------------------------------------- track
def cmd_track(args):
    from multiprocessing import cpu_count
    from experiment_manager import ExperimentManager
    from batch_processing_tab_manager import _should_skip_video
    # The pool loop itself is shared with the GUI (Section 2 queue + Experiment Dashboard) — see
    # batch_runner.run_pool. The CLI keeps its own video selection / --workers / warn-don't-filter
    # background-image handling below, which is why it calls run_pool directly rather than the
    # GUI-facing run_tracking_job_for_folder wrapper.
    from batch_runner import run_pool, resolve_worker_count

    json_path, folder_analysis_hint = _resolve_experiment(args.experiment, need_json=True)

    if args.force:
        os.environ['BUZZSUITE_FORCE_REPROCESS'] = '1'   # honored by _should_skip_video

    em = ExperimentManager(_log)
    em.load_experiment_from_json(json_path)
    folder_videos = em.folder_videos
    folder_analysis = em.folder_analysis
    settings_file = em.settings_file
    settings_data = em.settings

    # Collect segment names: prefer the experiment's own list, else the .mp4 files on disk.
    video_names = []
    try:
        video_names = list(em.experiment.list_video_name)
    except Exception:
        video_names = []
    if not video_names and folder_videos and os.path.isdir(folder_videos):
        video_names = sorted(os.path.splitext(f)[0] for f in os.listdir(folder_videos)
                             if f.endswith('.mp4') and not f.startswith('.'))
    if args.videos:
        wanted = set(args.videos)
        video_names = [v for v in video_names if v in wanted]

    if not video_names:
        raise SystemExit("No videos found for this experiment (checked list_video_name and %s)." % folder_videos)

    pending = [v for v in video_names if not _should_skip_video(v, folder_analysis)]
    skipped = len(video_names) - len(pending)
    if not pending:
        _log("All %d segment(s) already tracked (nothing to do). Use --force to re-track." % len(video_names))
        return 0

    # Precondition check: the tracker silently skips any segment lacking its background image.
    mort = os.path.join(folder_analysis, "images_mortality")
    missing_bg = [v for v in pending if not os.path.isfile(os.path.join(mort, v + ".png"))]
    if missing_bg:
        _log("WARNING: %d of %d pending segment(s) have NO background image in images_mortality/ and "
             "will be silently skipped by the tracker. Generate them via the GUI's Preliminary "
             "Analysis tab ('Extract Images from Video' / 'Get Background from Images') first."
             % (len(missing_bg), len(pending)))
        _log("         e.g. missing: %s%s" % (", ".join(missing_bg[:3]),
             " ..." if len(missing_bg) > 3 else ""))

    worker_count = resolve_worker_count(args.workers, len(pending), em.config)
    cv2_threads = max(1, (cpu_count() or 1) // max(1, worker_count))

    _log("Experiment : %s" % folder_analysis)
    _log("Segments   : %d total, %d pending, %d already done" % (len(video_names), len(pending), skipped))
    _log("Workers    : %d process(es), %d OpenCV thread(s) each"
         % (worker_count, cv2_threads))

    counts = {'ok': 0, 'fail': 0}

    def on_result(res, i, total):
        if isinstance(res, dict) and res.get('ok'):
            counts['ok'] += 1
            _log("[%d/%d] OK   %s" % (i, total, res.get('video_name')))
        else:
            counts['fail'] += 1
            name = res.get('video_name', '?') if isinstance(res, dict) else '?'
            err = (res.get('error') if isinstance(res, dict) else str(res)) or 'unknown'
            _log("[%d/%d] FAIL %s : %s" % (i, total, name, str(err).splitlines()[0][:200]))

    t0 = time.time()
    run_pool(pending, settings_file, folder_videos, folder_analysis, json_path,
             settings_data, worker_count, on_result=on_result)

    _log("Tracking done: %d ok, %d failed, %d skipped  (%.0fs)"
         % (counts['ok'], counts['fail'], skipped, time.time() - t0))
    return 0 if counts['fail'] == 0 else 1


# ----------------------------------------------------------------- swarm
def cmd_swarm(args):
    _, exp_dir = _resolve_experiment(args.experiment, need_json=False)
    if not os.path.isdir(os.path.join(exp_dir, "final_tracking_data")):
        raise SystemExit("No final_tracking_data/ in %s — run tracking first." % exp_dir)

    from buzzswarm import fru2_zt_normalized_analysis as engine
    # Optional engine tunables (same overrides the GUI applies via setattr before a run).
    overrides = {
        'MIN_FLIGHT_FRAMES': args.min_flight_frames,
        'SPEED_FILTER_MIN': args.speed_min,
        'SPEED_FILTER_MAX': args.speed_max,
        'BIN_MINUTES': args.bin_minutes,
        'SHOLL_RADIUS_STEP_PX': args.sholl_step,
    }
    for attr, val in overrides.items():
        if val is not None:
            setattr(engine, attr, val)
            _log("  set %s = %s" % (attr, val))

    t0 = time.time()
    which = args.which
    ran = []
    if which in ('sholl', 'all'):
        from buzzswarm import fru2_sholl_csv
        out = fru2_sholl_csv.run(exp_dir)
        _log("sholl CSV       -> %s" % out); ran.append(out)
    if which in ('pooled', 'all'):
        from buzzswarm import fru2_pooled_analysis
        out = fru2_pooled_analysis.run(exp_dir)
        _log("pooled r50      -> %s" % out); ran.append(out)
    if which in ('per-traj', 'all'):
        from buzzswarm import fru2_per_trajectory_plots
        out = fru2_per_trajectory_plots.run(exp_dir)
        _log("per-trajectory  -> %s" % out); ran.append(out)

    _log("BuzzSwarm done: %d analysis(es)  (%.0fs)" % (len(ran), time.time() - t0))
    return 0


# ----------------------------------------------------------------- phono
def cmd_phono(args):
    _, exp_dir = _resolve_experiment(args.experiment, need_json=False)

    start, end = args.start, args.end
    if not start or not end:
        ds, de = _derive_range(exp_dir)
        start = start or ds
        end = end or de

    zones = args.zones or os.path.join(exp_dir, "custom_zones.json")
    if not os.path.isfile(zones):
        raise SystemExit("No custom_zones.json at %s — define the speaker zone first "
                         "(GUI BuzzPhono tab, or the Step-4 zone editor)." % zones)

    from buzzphono import plot_speaker_distance_phonotaxis as ph
    ph.STIM_START_MIN, ph.STIM_END_MIN = args.on_start, args.on_end
    ph.OFF_START_MIN, ph.OFF_END_MIN = args.off_start, args.off_end

    _log("Experiment : %s" % exp_dir)
    _log("Window     : %s -> %s   (ON %d-%d, OFF %d-%d, resting_only=%s)"
         % (start, end, args.on_start, args.on_end, args.off_start, args.off_end, args.resting_only))
    t0 = time.time()
    out = ph.run_speaker_distance_analysis(
        exp_dir, start=start, end=end, custom_zones_path=zones,
        resting_only=args.resting_only, force_replot=args.force)
    _log("BuzzPhono done -> %s  (%.0fs)" % (out, time.time() - t0))
    return 0


# ----------------------------------------------------------------- convert
class _MsgSink(object):
    """Adapts convert_worker's ("log"/"status"/"progress"/"done", payload) queue protocol to
    plain _log() calls -- the CLI process itself is the unattended background context, so there's
    no GUI event loop to feed and no need for a real queue.Queue + polling thread."""

    def __init__(self):
        self._last_progress = None

    def put(self, item):
        kind, payload = item
        if kind == "log":
            _log(payload)
        elif kind == "status":
            _log("  ... %s" % payload)
        elif kind == "progress":
            done, total = payload
            if total and (done == total or done % 25 == 0):
                _log("  progress: %d/%d" % (done, total))
        elif kind == "done":
            ok, skip, fail, err = payload
            if err:
                _log("  ERROR: %s" % err)
            else:
                _log("  Converted: %d  Skipped: %d  Failed: %d" % (ok, skip, fail))


def _list_h264(in_dir):
    return sorted(os.path.join(in_dir, f) for f in os.listdir(in_dir)
                  if f.lower().endswith(".h264") and not f.startswith("."))


def _convert_one_folder(in_dir, out_dir, assume_default_header):
    from h264_converter import convert_gui

    files = _list_h264(in_dir)
    if not files:
        _log("  No .h264 files in %s -- skipping." % in_dir)
        return True

    blob, donor = convert_gui.find_header_blob(files)
    if blob is None:
        if not assume_default_header:
            _log("  WARNING: no SPS/PPS header-bearing segment found in %s -- skipping "
                 "(pass --assume-default-header to use the built-in %s default instead)."
                 % (in_dir, convert_gui.DEFAULT_DESC))
            return False
        blob = convert_gui.DEFAULT_HEADER
        header_src = "built-in default (%s)" % convert_gui.DEFAULT_DESC
        fps = convert_gui.DEFAULT_FPS
    else:
        header_src = os.path.basename(donor)
        fps = convert_gui.detect_fps(donor)

    os.makedirs(out_dir, exist_ok=True)
    _log("Converting: %s -> %s  (%d files)" % (in_dir, out_dir, len(files)))
    convert_gui.convert_worker(in_dir, out_dir, False, blob, header_src, fps, _MsgSink())
    return True


def cmd_convert(args):
    from h264_converter import convert_gui
    if not convert_gui.ffmpeg_available():
        raise SystemExit("ffmpeg was not found (bundled or on PATH) -- cannot convert.")

    t0 = time.time()

    if args.all:
        if not args.raw_root or not os.path.isdir(args.raw_root):
            raise SystemExit("--raw-root is required with --all and must be an existing directory.")
        exclude = set(args.exclude or [])
        jobs = []
        for name in sorted(os.listdir(args.raw_root)):
            if name in exclude:
                continue
            in_dir = os.path.join(args.raw_root, name)
            if not os.path.isdir(in_dir):
                continue
            files = _list_h264(in_dir)
            if not files:
                continue
            if args.out_root:
                out_dir = os.path.join(args.out_root, name)
            else:
                out_dir = os.path.join(os.path.dirname(in_dir), os.path.basename(in_dir) + "mp4")
            jobs.append((name, in_dir, out_dir, len(files)))

        if not jobs:
            _log("No pending folders found under %s (after exclusions)." % args.raw_root)
            return 0

        _log("Job list (%d folder(s)):" % len(jobs))
        for name, in_dir, out_dir, n in jobs:
            _log("  %-45s %4d file(s) -> %s" % (name, n, out_dir))

        if args.dry_run:
            _log("Dry run -- no files converted.")
            return 0

        ok = skipped = 0
        for name, in_dir, out_dir, n in jobs:
            if _convert_one_folder(in_dir, out_dir, args.assume_default_header):
                ok += 1
            else:
                skipped += 1
        _log("Batch done: %d folder(s) processed, %d skipped  (%.0fs)" % (ok, skipped, time.time() - t0))
        return 0

    if not args.in_dir or not args.out_dir:
        raise SystemExit("Provide either --all --raw-root, or --in-dir and --out-dir.")
    if args.dry_run:
        files = _list_h264(args.in_dir)
        _log("Dry run: %s -> %s  (%d file(s))" % (args.in_dir, args.out_dir, len(files)))
        return 0
    _convert_one_folder(args.in_dir, args.out_dir, args.assume_default_header)
    _log("Done  (%.0fs)" % (time.time() - t0))
    return 0


# ----------------------------------------------------------------- export
def cmd_export(args):
    _, exp_dir = _resolve_experiment(args.experiment, need_json=False)
    if not os.path.isdir(os.path.join(exp_dir, "final_tracking_data")):
        raise SystemExit("No final_tracking_data/ in %s — run tracking first." % exp_dir)

    from tracking_export import export_experiment_tracks
    t0 = time.time()
    out = export_experiment_tracks(
        exp_dir, out_dir=args.out, combined=not args.no_tracks,
        compression=args.compression, log=_log)
    _log("Export done -> %s  (%.0fs)" % (out, time.time() - t0))
    return 0


# ----------------------------------------------------------------- argparse
def build_parser():
    p = argparse.ArgumentParser(
        prog="buzzsuite_cli.py",
        description="BuzzSuite headless batch runner (track / swarm / phono) — reuses the GUI callables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with the conda env python. See the module docstring for examples.")
    sub = p.add_subparsers(dest="command", metavar="{track,swarm,phono,export,convert}")
    sub.required = True

    # track
    t = sub.add_parser("track", help="Batch-track an experiment's video segments (multiprocessing).")
    t.add_argument("--experiment", "-e", required=True, help="experiment_*.json (or a dir containing one).")
    t.add_argument("--workers", "-w", type=int, default=None, help="Worker processes (default: cores-2, capped).")
    t.add_argument("--force", action="store_true", help="Re-track segments even if output already exists.")
    t.add_argument("--videos", nargs="+", metavar="NAME", help="Only track these segment names (space-separated).")
    t.set_defaults(func=cmd_track)

    # swarm
    s = sub.add_parser("swarm", help="Run BuzzSwarm (Sholl/r50 aggregation) on an experiment.")
    s.add_argument("--experiment", "-e", required=True, help="Experiment dir (or its experiment_*.json).")
    s.add_argument("--which", choices=["sholl", "pooled", "per-traj", "all"], default="all",
                   help="Which analysis to run (default: all).")
    s.add_argument("--min-flight-frames", type=int, default=None, help="Engine MIN_FLIGHT_FRAMES (default 50).")
    s.add_argument("--speed-min", type=float, default=None, help="Engine SPEED_FILTER_MIN px/frame (default 1.0).")
    s.add_argument("--speed-max", type=float, default=None, help="Engine SPEED_FILTER_MAX px/frame (default 40.0).")
    s.add_argument("--bin-minutes", type=int, default=None, help="Engine BIN_MINUTES (default 10).")
    s.add_argument("--sholl-step", type=int, default=None, help="Engine SHOLL_RADIUS_STEP_PX (default 5).")
    s.set_defaults(func=cmd_swarm)

    # phono
    ph = sub.add_parser("phono", help="Run BuzzPhono (speaker-distance phonotaxis) on an experiment.")
    ph.add_argument("--experiment", "-e", required=True, help="Experiment dir (or its experiment_*.json).")
    ph.add_argument("--start", default=None, help="ISO start (default: auto-derived from segments).")
    ph.add_argument("--end", default=None, help="ISO end (default: auto-derived from segments).")
    ph.add_argument("--zones", default=None, help="custom_zones.json path (default: <exp>/custom_zones.json).")
    ph.add_argument("--on-start", type=int, default=40, help="Stimulus ON window start minute (default 40).")
    ph.add_argument("--on-end", type=int, default=50, help="Stimulus ON window end minute (default 50).")
    ph.add_argument("--off-start", type=int, default=28, help="Stimulus OFF (baseline) start minute (default 28).")
    ph.add_argument("--off-end", type=int, default=38, help="Stimulus OFF (baseline) end minute (default 38).")
    ph.add_argument("--resting-only", action="store_true",
                    help="Only the resting speaker-proportion pass (skip trajectory/heatmap pass; faster).")
    ph.add_argument("--force", action="store_true", help="Regenerate even if outputs are newer than tracking data.")
    ph.set_defaults(func=cmd_phono)

    # export
    ex = sub.add_parser("export", help="Export frozen tracking .pkls to Parquet + CSV (human/ML readable).")
    ex.add_argument("--experiment", "-e", required=True, help="Experiment dir (or its experiment_*.json).")
    ex.add_argument("--out", default=None, help="Output dir (default: <exp>/exports/).")
    ex.add_argument("--compression", default="snappy", choices=["snappy", "zstd"],
                    help="Parquet compression codec (default: snappy).")
    ex.add_argument("--no-tracks", action="store_true",
                    help="Skip the nested tracks.parquet + sample_segment.csv (detections + summary only).")
    ex.set_defaults(func=cmd_export)

    # convert
    c = sub.add_parser("convert", help="Lossless .h264 -> .mp4 remux, one folder or a whole raw-video root.")
    c.add_argument("--in-dir", default=None, help="Single source folder of .h264 files.")
    c.add_argument("--out-dir", default=None, help="Destination for --in-dir mode.")
    c.add_argument("--all", action="store_true", help="Batch mode: scan --raw-root for pending folders.")
    c.add_argument("--raw-root", default=None, help="Root containing one subfolder per recording session (--all mode).")
    c.add_argument("--out-root", default=None,
                   help="Consolidated output root; each session goes to <out-root>/<session-name>/ "
                        "(--all mode; default: sibling '<folder>mp4' next to each source folder).")
    c.add_argument("--exclude", action="append", metavar="NAME",
                   help="Immediate subfolder name to skip in --all mode (repeatable).")
    c.add_argument("--assume-default-header", action="store_true",
                   help="If a folder has no SPS/PPS donor segment, use the built-in default header "
                        "instead of skipping it (unattended equivalent of the GUI's confirmation dialog).")
    c.add_argument("--dry-run", action="store_true", help="Print the job list/counts without converting anything.")
    c.set_defaults(func=cmd_convert)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
