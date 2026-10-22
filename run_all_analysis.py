#!/usr/bin/env python
"""Run every unattended analysis for each selected experiment (3 experiments at once by default,
same as run_flare_pipeline.py --concurrent-experiments 3 with auto-scaled workers):

    1. backgrounds   -- reference + background images for any segment still missing them
                        (safety net; normally already done by prepare_experiments.py)
    2. tracking      -- buzzsuite_cli cmd_track (multiprocessing, auto-scaled worker count)
    3. concatenation -- analyzed_data.pkl (same call the GUI makes after tracking)
    4. BuzzSwarm     -- Sholl CSV + pooled r50 + per-trajectory plots (buzzsuite_cli cmd_swarm)
    5. BuzzPhono     -- speaker-distance phonotaxis, ONLY if the experiment has custom_zones.json
    6. export        -- Parquet/CSV tracking export, only with --export

Pure orchestration over existing callables (run_flare_pipeline.py's worker auto-scaling + heartbeat
+ KNOWN_BAD_SEGMENTS, buzzsuite_cli's track/swarm/phono/export, prepare_experiments.py's selection
and status checks) -- no analysis logic lives here. Every step is idempotent (finished segments are
skipped, BuzzSwarm/BuzzPhono reuse their caches), so re-running after an interruption resumes.

Experiments are SKIPPED (with a reason in the summary) when:
  - the folder name contains INVESTIGATE (set aside on purpose),
  - no cage border has been drawn yet (run prepare_experiments.py first -- tracking without a
    border would analyse the wrong area),
  - there are no .mp4 videos / no Recording folder.
A failure in one step or one experiment never stops the rest.

Activity plots are not produced here: they are made interactively in the GUI's plot tab from the
analyzed_data.pkl this script writes.

Usage
-----
    python run_all_analysis.py --root F:\\Buzzwatch --dry-run       # show the plan only
    python run_all_analysis.py --root F:\\Buzzwatch                 # everything selected
    python run_all_analysis.py --root F:\\Buzzwatch --only Cakung_1 # name-substring filter
    python run_all_analysis.py F:\\Buzzwatch\\Analysis\\Flare\\Bangkok\\Lutzia_F_20260906_20260909
    python run_all_analysis.py --root F:\\Buzzwatch --skip-swarm --max-workers
"""
import argparse
import concurrent.futures
import os
import sys
import threading
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import prepare_experiments as prep
import run_flare_pipeline as rfp

_log = rfp._log  # timestamped console + log-file logger (file opened in main)

# With --concurrent-experiments > 1, BuzzSwarm/BuzzPhono run one at a time across experiments: both
# configure their engine through module-level globals, which concurrent threads would share. They
# are short next to tracking, so serialising them costs little.
_post_lock = threading.Lock()


def _step(results, name, fn, *args, **kwargs):
    """Run one step, record 'ok'/'FAILED'/a skip reason, never raise (SystemExit included -- the
    buzzsuite_cli commands use it for "nothing to do" style precondition messages)."""
    t0 = time.time()
    try:
        rc = fn(*args, **kwargs)
        ok = rc in (None, 0)
        results[name] = ("ok" if ok else "FAILED (exit %s)" % rc) + " %.0fs" % (time.time() - t0)
    except SystemExit as exc:
        results[name] = "FAILED: %s" % exc
        _log("  %s stopped: %s" % (name, exc))
    except Exception:
        results[name] = "FAILED (exception)"
        _log("  %s FAILED:\n%s" % (name, traceback.format_exc()))


def _has_files(folder):
    return os.path.isdir(folder) and any(not f.startswith(".") for f in os.listdir(folder))


def _skip_reason(st):
    if st["set_aside"]:
        return "set aside (INVESTIGATE)"
    if st["error"]:
        return st["error"]
    if not st["has_video_folder"]:
        return "no Recording folder"
    if st["videos"] == 0:
        return "no .mp4 videos"
    if not st["border"]:
        return "no cage border -- run prepare_experiments.py first"
    return None


def _process(i, n, st, args):
    import argparse as _ap
    from buzzsuite_cli import cmd_track, cmd_swarm, cmd_phono, cmd_export
    from batch_processing_tab_manager import concatenate_and_save_experiment_data, pool_concurrency_slot
    from experiment_manager import ExperimentManager

    jp, exp_dir = st["json"], st["dir"]
    results = {}
    _log("\n" + "=" * 100)
    _log("[%d/%d] %s" % (i, n, exp_dir))
    _log("=" * 100)

    # 1. backgrounds (only if something is missing)
    if st["need_images"] or st["need_bg"]:
        _log("[%d/%d] backgrounds missing for some segments -- extracting..." % (i, n))
        def _bg():
            # Own process: extract_average_background swaps the global sys.stdout, which would
            # garble other experiments' threads under --concurrent-experiments.
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
                _jp, ok, msg = ex.submit(prep._prep_one, jp, None).result()
            if not ok:
                _log(msg)
            return 0 if ok else 1
        _step(results, "backgrounds", rfp._run_with_heartbeat, "%s (backgrounds)" % exp_dir, _bg)

    # 2. tracking
    if not args.skip_tracking:
        em = ExperimentManager(lambda *a, **k: None)
        em.load_experiment_from_json(jp)
        from batch_processing_tab_manager import _should_skip_video
        videos = [v for v in em.experiment.list_video_name if v not in rfp.KNOWN_BAD_SEGMENTS]
        n_pending = len([v for v in videos if not _should_skip_video(v, exp_dir)])
        with pool_concurrency_slot() as concurrency:
            if args.workers is not None:
                workers = args.workers
            else:
                workers, reason = rfp._adaptive_worker_count(max(1, n_pending), max_workers=args.max_workers,
                                                             concurrency=concurrency)
                _log("[%d/%d] auto-selected %d worker(s): %s" % (i, n, workers, reason))
            ns = _ap.Namespace(experiment=jp, workers=workers, force=False, videos=videos)
            _step(results, "tracking", rfp._run_with_heartbeat, "%s (tracking)" % exp_dir, cmd_track, ns)

        # 3. concatenation -- always attempted: a few failed segments must not block the rest
        def _concat():
            em2 = ExperimentManager(lambda *a, **k: None)
            em2.load_experiment_from_json(jp)
            concatenate_and_save_experiment_data(exp_dir, config=em2.config, log_fn=_log)
        _step(results, "concatenation", _concat)

    tracked = _has_files(os.path.join(exp_dir, "final_tracking_data"))

    # 4. BuzzSwarm
    if args.skip_swarm:
        pass
    elif not tracked:
        results["swarm"] = "skipped (no tracking data)"
    else:
        ns = _ap.Namespace(experiment=exp_dir, which="all", min_flight_frames=None, speed_min=None,
                           speed_max=None, bin_minutes=None, sholl_step=None)
        with _post_lock:
            _step(results, "swarm", rfp._run_with_heartbeat, "%s (BuzzSwarm)" % exp_dir, cmd_swarm, ns)

    # 5. BuzzPhono (needs a speaker zone)
    if args.skip_phono:
        pass
    elif not os.path.isfile(os.path.join(exp_dir, "custom_zones.json")):
        results["phono"] = "skipped (no custom_zones.json)"
    elif not tracked:
        results["phono"] = "skipped (no tracking data)"
    else:
        ns = _ap.Namespace(experiment=exp_dir, start=None, end=None, zones=None, on_start=40, on_end=50,
                           off_start=28, off_end=38, resting_only=False, force=False)
        with _post_lock:
            _step(results, "phono", rfp._run_with_heartbeat, "%s (BuzzPhono)" % exp_dir, cmd_phono, ns)

    # 6. export (opt-in)
    if args.export:
        if tracked:
            ns = _ap.Namespace(experiment=exp_dir, out=None, compression="snappy", no_tracks=False)
            _step(results, "export", cmd_export, ns)
        else:
            results["export"] = "skipped (no tracking data)"
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*",
                   help="Experiment folders or experiment_*.json files. Omit to scan the data root.")
    p.add_argument("--root", default=None, help="Data root (e.g. F:\\Buzzwatch).")
    p.add_argument("--assay", default=None, help="Only this assay folder under Analysis\\ (e.g. Flare).")
    p.add_argument("--only", default=None, help="Only experiments whose folder name contains this.")
    p.add_argument("--skip", default=None, help="Skip experiments whose folder name contains this.")
    p.add_argument("--workers", "-w", type=int, default=None,
                   help="Pin the tracking worker count (default: auto from CPU + free RAM).")
    p.add_argument("--max-workers", action="store_true",
                   help="Auto worker count biased toward the CPU ceiling (see run_flare_pipeline.py).")
    p.add_argument("--concurrent-experiments", type=int, default=3,
                   help="Experiments processed at once (default 3, as in the 2026-09-17 overnight run "
                        "that finished clean). Auto worker counts divide across them via "
                        "pool_concurrency_slot(), so total workers stay within the one-experiment "
                        "budget. Use 1 for strictly one-at-a-time.")
    p.add_argument("--skip-tracking", action="store_true", help="Don't track/concatenate.")
    p.add_argument("--skip-swarm", action="store_true", help="Don't run BuzzSwarm.")
    p.add_argument("--skip-phono", action="store_true", help="Don't run BuzzPhono.")
    p.add_argument("--export", action="store_true", help="Also export Parquet/CSV tracking data.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = p.parse_args(argv)

    if not args.dry_run:
        os.makedirs(rfp.LOG_DIR, exist_ok=True)
        path = os.path.join(rfp.LOG_DIR, "run_all_analysis_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
        rfp._log_file = open(path, "a", encoding="utf-8")
        import buzzsuite_cli
        buzzsuite_cli._log = _log  # per-segment OK/FAIL lines land in the same log file
        _log("Logging to %s" % path)

    jsons = prep._select(args)
    if not jsons:
        return 1
    statuses = [prep._status(jp) for jp in jsons]
    todo, skipped = [], []
    for st in statuses:
        reason = _skip_reason(st)
        (skipped if reason else todo).append((st, reason))

    steps = [s for s, off in (("tracking+concat", args.skip_tracking), ("BuzzSwarm", args.skip_swarm),
                              ("BuzzPhono (if custom_zones.json)", args.skip_phono)) if not off]
    if args.export:
        steps.append("export")
    _log("Steps: %s" % ", ".join(steps))
    _log("%-60s %s" % ("Experiment", "plan"))
    for st, reason in todo + skipped:
        _log("%-60s %s" % (st["name"], "SKIP: " + reason if reason else
                           "run (%d videos%s)" % (st["videos"], ", backgrounds missing"
                                                  if st["need_images"] or st["need_bg"] else "")))
    if args.dry_run:
        _log("Dry run -- nothing done.")
        return 0

    def _safe(i, st):
        try:
            return st["name"], _process(i, len(todo), st, args)
        except Exception:
            _log("[%d/%d] UNEXPECTED FAILURE for %s:\n%s" % (i, len(todo), st["dir"], traceback.format_exc()))
            return st["name"], {"experiment": "FAILED (unexpected)"}

    t0 = time.time()
    n_conc = max(1, args.concurrent_experiments)
    _log("Processing up to %d experiment(s) at once%s." % (
        n_conc, "" if n_conc == 1 else " -- log lines from different experiments will interleave"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_conc) as executor:
        futures = [executor.submit(_safe, i, st) for i, (st, _) in enumerate(todo, 1)]
        summary = [f.result() for f in futures]

    _log("\n" + "=" * 100)
    _log("ALL DONE in %.1f h" % ((time.time() - t0) / 3600.0))
    any_fail = False
    for name, res in summary:
        _log("  %s" % name)
        for step, outcome in res.items():
            any_fail = any_fail or outcome.startswith("FAILED")
            _log("      %-14s %s" % (step, outcome))
    for st, reason in skipped:
        _log("  %s -- SKIPPED: %s" % (st["name"], reason))
    if any_fail:
        _log("\nSome steps failed -- see the log above. Re-run to retry; finished work is skipped.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
