#!/usr/bin/env python
"""Sequentially resume batch tracking for every experiment under the data root that still has
pending video segments.

Thin orchestration around buzzsuite_cli.py's own `track` command -- calls cmd_track() directly
(same code path as running `buzzsuite_cli.py track --experiment <one>` by hand) for each
experiment_*.json found under Analysis\\*\\*\\*\\, one at a time, and then runs
concatenate_and_save_experiment_data() afterward (the same call the GUI's tracking queue /
Experiment Dashboard make right after tracking finishes, so analyzed_data.pkl stays in sync --
the CLI's `track` command alone does not do this).

Already-fully-tracked experiments (0 pending) are skipped without spinning up a Pool. If one
experiment fails, the error is logged and the run moves on to the next one rather than aborting
the whole overnight/multi-day run.

Usage
-----
    python run_all_pending.py                 # process every experiment with pending segments
    python run_all_pending.py --dry-run        # just print the plan, track nothing
    python run_all_pending.py --workers 6      # override worker count for every experiment
    python run_all_pending.py --skip-concat    # tracking only, skip the post-experiment concat step
"""
import argparse
import glob
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

DATA_ROOT = os.environ.get("BUZZSUITE_DATA_ROOT", r"D:\Buzzwatch")
EXPERIMENT_GLOB = os.path.join(DATA_ROOT, "Analysis", "*", "*", "*", "experiment_*.json")


def _log(msg):
    print(msg, flush=True)


def _plan():
    """Return [(json_path, exp_dir, total, pending), ...] for every experiment found."""
    from experiment_manager import ExperimentManager
    from batch_processing_tab_manager import _should_skip_video

    jsons = sorted(glob.glob(EXPERIMENT_GLOB))
    plan = []
    for jp in jsons:
        try:
            em = ExperimentManager(lambda *a, **k: None)
            em.load_experiment_from_json(jp)
            video_names = list(em.experiment.list_video_name)
            pending = [v for v in video_names if not _should_skip_video(v, em.folder_analysis)]
            plan.append((jp, em.folder_analysis, len(video_names), len(pending)))
        except Exception as e:
            _log("SKIP (could not load): %s -- %s" % (jp, e))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Worker processes per experiment (default: cores-2, capped).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, track nothing.")
    parser.add_argument("--skip-concat", action="store_true",
                        help="Skip the post-experiment concatenate_and_save_experiment_data() step.")
    args = parser.parse_args()

    from buzzsuite_cli import cmd_track
    from batch_processing_tab_manager import concatenate_and_save_experiment_data
    from experiment_manager import ExperimentManager

    plan = _plan()
    if not plan:
        _log("No experiment_*.json found under %s" % EXPERIMENT_GLOB)
        return 1

    _log("Found %d experiment(s) under %s\n" % (len(plan), DATA_ROOT))
    _log("%-70s %8s %8s" % ("Experiment", "total", "pending"))
    for jp, exp_dir, total, pending in plan:
        marker = "" if pending > 0 else "  (already done)"
        _log("%-70s %8d %8d%s" % (os.path.relpath(exp_dir, DATA_ROOT), total, pending, marker))

    todo = [(jp, exp_dir, total, pending) for (jp, exp_dir, total, pending) in plan if pending > 0]
    _log("\n%d experiment(s) have pending work, %d segment(s) pending in total.\n"
         % (len(todo), sum(p for _, _, _, p in todo)))

    if args.dry_run:
        _log("Dry run -- nothing tracked.")
        return 0

    overall_t0 = time.time()
    results = []
    for i, (jp, exp_dir, total, pending) in enumerate(todo, 1):
        _log("\n" + "=" * 100)
        _log("[%d/%d] %s  (%d pending of %d)" % (i, len(todo), exp_dir, pending, total))
        _log("=" * 100)

        ns = argparse.Namespace(experiment=jp, workers=args.workers, force=False, videos=None)
        t0 = time.time()
        try:
            rc = cmd_track(ns)
        except Exception as e:
            _log("[%d/%d] TRACKING FAILED (exception): %s -- %s" % (i, len(todo), exp_dir, e))
            rc = 1

        if rc == 0 and not args.skip_concat:
            try:
                em = ExperimentManager(lambda *a, **k: None)
                em.load_experiment_from_json(jp)
                concatenate_and_save_experiment_data(exp_dir, config=em.config, log_fn=_log)
            except Exception as e:
                _log("[%d/%d] CONCATENATION FAILED: %s -- %s" % (i, len(todo), exp_dir, e))

        elapsed = time.time() - t0
        results.append((exp_dir, rc, elapsed))
        _log("[%d/%d] done in %.0fs, tracking exit code %s" % (i, len(todo), elapsed, rc))

    _log("\n" + "=" * 100)
    _log("ALL EXPERIMENTS DONE in %.0fs total" % (time.time() - overall_t0))
    for exp_dir, rc, elapsed in results:
        _log("  %-70s exit=%s  %.0fs" % (exp_dir, rc, elapsed))
    failures = [r for r in results if r[1] != 0]
    if failures:
        _log("\n%d experiment(s) had tracking failures -- re-run this script to retry (already-done "
             "segments are skipped automatically)." % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
