#!/usr/bin/env python
"""Sequentially finish the Activity pipeline (background images -> tracking -> concatenation)
for the Lutzia experiments under Analysis\\Flare\\, skipping BuzzSwarm/BuzzPhono (not requested).

This is run_flare_pipeline.py's exact plan/run logic, scoped down to just the folders whose
experiment name contains "Lutzia" (case-insensitive) instead of every Flare experiment:

    Analysis\\Flare\\Bangkok\\Lutzia_F_20260622_20260625   (has raw video)
    Analysis\\Flare\\Cakung\\Lutzia_F_20260622_20260625    (NO raw recording on disk -- auto-skipped)
    Analysis\\Flare\\Cakung\\Lutzia_M_20260622_20260625    (has raw video)

Uses the same maxtasksperchild=1 fix in batch_runner.py::run_pool (worker processes are recycled
after every segment, so per-worker memory can't accumulate across a long run) plus the same
images_mortality/ precondition handling as run_flare_pipeline.py -- re-running this script after an
interruption just picks up where it left off (both get_images_from_video/get_background_from_images
and the tracker itself skip already-done output).

Usage
-----
    python run_lutzia_pipeline.py                 # process the Lutzia experiments with pending work
    python run_lutzia_pipeline.py --dry-run        # just print the plan, do nothing
    python run_lutzia_pipeline.py --workers 6      # pin tracking worker count (disables auto-scaling)
    python run_lutzia_pipeline.py --max-workers    # auto-scale, but bias toward the CPU ceiling
    python run_lutzia_pipeline.py --assay Lightctrl --root D:\\Buzzwatch2   # a different assay/root

Worker auto-scaling
--------------------
When --workers is NOT given, the worker count for each experiment's tracking pool is chosen
FRESHLY FOR EVERY EXPERIMENT (not once for the whole run, and not a fixed config default) from two
budgets, and the smaller one wins:
  - CPU budget: cpu_count() - 1 (never oversubscribe the machine).
  - Memory budget: current free RAM (via psutil, if installed), minus a reserve left for the
    OS/GUI/other apps, divided by an estimated per-worker cost.
Because this is re-evaluated before every experiment, it automatically scales itself back down
later in the same run if free RAM drops for any reason -- a genuine self-adjusting safety net, not
a one-time guess at startup.

Two memory profiles:
  - Default: conservative. BUZZSUITE_WORKER_MEM_GB per worker (4.0GB), no RAM reserved beyond that
    padding.
  - --max-workers: biases toward the CPU ceiling instead. Reserves BUZZSUITE_MAX_WORKERS_RESERVE_GB
    (4.0GB) for the OS/GUI/other apps, then estimates BUZZSUITE_MAX_WORKERS_MEM_GB (3.0GB) per
    worker. Still fully self-adjusting: if free RAM is genuinely tight when an experiment starts,
    this comes in below the CPU ceiling on its own, it does not blindly force max workers regardless
    of memory.

2026-09-17 correction: both figures above were raised (from 2.0GB / 1.2GB) after a real overnight
run of run_flare_pipeline.py on this exact box picked 19-22 workers (well within what the OLD,
lower-padded budget allowed) and hit real trouble on a dense Bangkok/Japonicus_F session -- six
segments failed with genuine OpenCV OutOfMemoryError, one needed a 992MiB single array (16127x16127
int32) and failed too, and then ALL 22 concurrently-dispatched workers went silent for 90 minutes
straight (system RAM stayed LOW throughout, ruling out simple exhaustion -- see batch_runner.py's
_pool_workers_busy for the companion fix to how a stall like that is now detected) before the
stall-timeout killed the lot, discarding real in-progress work. Per-worker peak memory on a
genuinely dense session (15,000+ tracks) is evidently well above the ~1GB/worker figure measured
from a light 2-3 worker run -- these higher paddings reflect that.

Both modes also never exceed that experiment's own pending-segment count (no point starting idle
workers), and never exceed cmd_track's own resolve_worker_count() cap, which still applies
underneath this as a final safety net. Pass --workers N to bypass all of this and pin an exact
count (takes priority over --max-workers if both are given). Without psutil installed, both modes
fall back to the CPU budget only.

Logging
-------
Every log line is timestamped and written to BOTH the console and logs/run_lutzia_pipeline_
<started>.log (path printed at startup) -- this survives the terminal being closed, the machine
rebooting, or nobody watching it live, so a multi-day unattended run leaves a complete record of
exactly what happened and when. A background heartbeat line is also printed every 5 minutes while
a segment batch is tracking (with memory/CPU if psutil is available), since a single segment can
legitimately take tens of minutes with no other output -- that silence used to look identical to a
genuine hang.
"""
import argparse
import glob
import os
import sys
import threading
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from path_utils import resolve_data_root

try:
    import psutil
except ImportError:
    psutil = None

WORKER_MEM_GB = float(os.environ.get("BUZZSUITE_WORKER_MEM_GB", "4.0"))
# --max-workers still leaves this much RAM unclaimed for the OS/GUI/other apps. See the "2026-09-17
# correction" note in the module docstring for why both this and WORKER_MEM_GB above were raised --
# a real overnight run on this box hit genuine OOMs and a 90-minute mass-stall at 19-22 workers.
MAX_WORKERS_RESERVE_GB = float(os.environ.get("BUZZSUITE_MAX_WORKERS_RESERVE_GB", "4.0"))
MAX_WORKERS_MEM_GB = float(os.environ.get("BUZZSUITE_MAX_WORKERS_MEM_GB", "3.0"))

LOG_DIR = os.path.join(APP_DIR, "logs")
_log_file = None  # set by _setup_logging(); None until then, so early calls just print

# IMPORTANT -- 2026-09-11 correction: file size is NOT evidence of corruption. Segment size varies
# legitimately with recording length and with mosquito activity (a busy cage compresses to far more
# bytes than a quiet one). A prior version of this list excluded 16 segments based purely on being
# smaller than ~300MB "normal" -- a structural audit (walking MP4 top-level boxes, no decoding)
# proved all 16 were fine: e.g. the tail-of-session files here decode 100% of their frames. Only ONE
# of the originally-listed files is genuinely broken -- its `mdat` box declares more bytes than the
# file actually contains (an incomplete write), and OpenCV correctly reads only part of it. Kept
# pre-excluded here purely to avoid a known, permanently-unfixable FAIL line on every rerun; it does
# not affect tracking numerics or the .pkl format -- concatenate_and_save_experiment_data only
# concatenates whatever final_tracking_data/ files actually exist.
KNOWN_BAD_SEGMENTS = {
    "CagebimaculataF_20260622_120319_00009",  # Bangkok/Lutzia_F -- mdat overruns file by ~200MB
}


def _setup_logging():
    """Open logs/run_lutzia_pipeline_<started>.log for the rest of this process's _log() calls.
    Also redirects buzzsuite_cli's own module-level _log (cmd_track's per-segment OK/FAIL lines)
    through this same function, so EVERYTHING this run produces -- not just this script's own
    banners -- lands in one timestamped, persistent file."""
    global _log_file
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "run_lutzia_pipeline_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
    _log_file = open(path, "a", encoding="utf-8")
    import buzzsuite_cli
    buzzsuite_cli._log = _log
    return path


def _log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if _log_file is not None:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except Exception:
            pass  # never let a logging failure take down the run


def _heartbeat_snapshot():
    """Includes system-wide CPU% (averaged since the previous heartbeat, via psutil.cpu_percent's
    interval=None mode) alongside RAM -- added 2026-09-17 after a real incident where 22 concurrent
    workers went silent for 90 minutes with RAM staying low the whole time, and there was no way to
    tell from the log alone whether they were genuinely hung (near-0% CPU) or legitimately grinding
    through unusually dense sessions (high CPU) until the machine's Windows Event Log was checked
    after the fact. This makes that distinction visible live, in the log, going forward."""
    if psutil is None:
        return ""
    try:
        p = psutil.Process()
        rss_gb = p.memory_info().rss / 1e9
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        return (" | this process %.1fGB RSS, system RAM %.0f%% used, system CPU %.0f%% used"
                % (rss_gb, vm.percent, cpu_pct))
    except Exception:
        return ""


def _run_with_heartbeat(label, fn, *args, interval=300, **kwargs):
    """Run fn(*args, **kwargs), logging '[heartbeat] ...' every `interval` seconds from a
    background thread while it runs -- so a long segment (or a whole experiment's worth of them)
    doesn't look identical to a stuck/frozen process. Stops the heartbeat thread before returning
    or re-raising, either way."""
    stop = threading.Event()
    t0 = time.time()

    def _beat():
        while not stop.wait(interval):
            _log("[heartbeat] still working on %s (%.0fm elapsed)%s"
                 % (label, (time.time() - t0) / 60.0, _heartbeat_snapshot()))

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        return fn(*args, **kwargs)
    finally:
        stop.set()
        t.join(timeout=2)


def _adaptive_worker_count(n_pending, max_workers=False):
    """Pick a tracking worker count for one experiment from the CPU budget and the current
    free-memory budget, whichever is smaller -- see the module docstring's "Worker auto-scaling"
    section. Re-evaluated fresh for every experiment (not once for the whole run), so it
    automatically scales itself down again later if free RAM drops (another app opened, a previous
    experiment's workers haven't fully released memory yet, etc.) -- this is a genuine safety net,
    not a one-time guess.

    `max_workers=True` (--max-workers) still applies that same safety net, it just biases toward
    the CPU ceiling: it reserves less headroom (MAX_WORKERS_RESERVE_GB, default 4GB left for the
    OS/GUI/other apps) and estimates per-worker cost from the actually-measured figure
    (MAX_WORKERS_MEM_GB, default 1.2GB -- live tasklist during a real run showed ~1.0GB/worker)
    instead of the default mode's 2x-padded WORKER_MEM_GB. It does NOT skip the memory check --
    if free RAM is genuinely tight, --max-workers still comes in below the CPU ceiling.
    Returns (count, reason_str) for logging."""
    cpu_budget = max(1, (os.cpu_count() or 1) - 1)
    reserve_gb = MAX_WORKERS_RESERVE_GB if max_workers else 0.0
    per_worker_gb = MAX_WORKERS_MEM_GB if max_workers else WORKER_MEM_GB
    mode = "--max-workers" if max_workers else "default"

    if psutil is None:
        count = max(1, min(cpu_budget, n_pending))
        return count, "cpu budget %d (%s mode, psutil not installed, no memory budget)" % (cpu_budget, mode)
    try:
        available_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        count = max(1, min(cpu_budget, n_pending))
        return count, "cpu budget %d (%s mode, could not read available memory)" % (cpu_budget, mode)

    usable_gb = max(0.0, available_gb - reserve_gb)
    mem_budget = max(1, int(usable_gb // per_worker_gb))
    count = max(1, min(cpu_budget, mem_budget, n_pending))
    return count, ("cpu budget %d, memory budget %d (%.1fGB free - %.1fGB reserved = %.1fGB usable "
                   "/ %.1fGB per worker, %s mode), pending %d"
                   % (cpu_budget, mem_budget, available_gb, reserve_gb, usable_gb, per_worker_gb,
                      mode, n_pending))


def _plan(data_root, assay):
    """Return [(json_path, exp_dir, total, pending, n_missing_bg), ...] for every Lutzia experiment
    under Analysis\\<assay>\\ whose folder_videos actually exists on disk."""
    from experiment_manager import ExperimentManager
    from batch_processing_tab_manager import _should_skip_video

    experiment_glob = os.path.join(data_root, "Analysis", assay, "*", "*", "experiment_*.json")
    jsons = sorted(glob.glob(experiment_glob))
    jsons = [jp for jp in jsons if "lutzia" in os.path.basename(os.path.dirname(jp)).lower()]
    plan = []
    for jp in jsons:
        try:
            em = ExperimentManager(lambda *a, **k: None)
            em.load_experiment_from_json(jp)
        except Exception:
            _log("SKIP (could not load) %s:\n%s" % (jp, traceback.format_exc()))
            continue
        if not em.folder_videos or not os.path.isdir(em.folder_videos):
            _log("SKIP (no Recording folder at %s): %s" % (em.folder_videos, jp))
            continue
        video_names = list(em.experiment.list_video_name)
        pending = [v for v in video_names if not _should_skip_video(v, em.folder_analysis)]
        pending = [v for v in pending if v not in KNOWN_BAD_SEGMENTS]
        mort = os.path.join(em.folder_analysis, "images_mortality")
        have_bg = set()
        if os.path.isdir(mort):
            have_bg = set(os.path.splitext(f)[0] for f in os.listdir(mort))
        missing_bg = [v for v in pending if v not in have_bg]
        plan.append((jp, em.folder_analysis, video_names, len(video_names), len(pending), len(missing_bg)))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Pin an exact worker count per experiment (disables all auto-scaling).")
    parser.add_argument("--max-workers", action="store_true",
                        help="Bias auto-scaling toward the CPU ceiling (cpu_count()-1) instead of "
                             "the default conservative memory budget. Still self-adjusts down if "
                             "free RAM is actually tight -- see the module docstring. Ignored if "
                             "--workers is also given.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, do nothing.")
    parser.add_argument("--force-concat", action="store_true",
                        help="Force-rebuild analyzed_data.pkl for EVERY experiment in the plan, "
                             "including ones with zero pending segments (the normal loop never "
                             "touches those). Bypasses the mtime-based skip-if-fresh check, which "
                             "cannot detect a code change to concatenation itself. Use this once "
                             "after a fix like the population-data fallback (2026-09-16) so "
                             "already-tracked experiments pick it up too; not needed on ordinary runs.")
    parser.add_argument("--root", default=None,
                        help="Data root override (default: $BUZZSUITE_DATA_ROOT or the usual "
                             "auto-detected mount, via path_utils.resolve_data_root).")
    parser.add_argument("--assay", default="Flare",
                        help="Assay type folder under Analysis\\ (default: Flare).")
    args = parser.parse_args()

    log_path = _setup_logging()
    _log("Logging to %s (this file persists even if the console is closed or the machine reboots)"
         % log_path)
    if psutil is None:
        _log("(psutil not available -- heartbeat lines will show elapsed time only, no memory/CPU)")

    from buzzsuite_cli import cmd_track
    from batch_processing_tab_manager import concatenate_and_save_experiment_data
    from experiment_manager import ExperimentManager

    data_root = resolve_data_root(args.root or os.environ.get("BUZZSUITE_DATA_ROOT"))

    plan = _plan(data_root, args.assay)
    if not plan:
        experiment_glob = os.path.join(data_root, "Analysis", args.assay, "*", "*", "experiment_*.json")
        _log("No Lutzia experiment_*.json found under %s" % experiment_glob)
        return 1

    _log("Found %d Lutzia experiment(s)\n" % len(plan))
    _log("%-55s %8s %8s %10s" % ("Experiment", "total", "pending", "need_bg"))
    for jp, exp_dir, video_names, total, pending, missing_bg in plan:
        marker = "" if pending > 0 else "  (already done)"
        _log("%-55s %8d %8d %10d%s" % (os.path.relpath(exp_dir, data_root), total, pending, missing_bg, marker))

    todo = [(jp, exp_dir, video_names, total, pending, missing_bg)
            for (jp, exp_dir, video_names, total, pending, missing_bg) in plan if pending > 0]
    _log("\n%d experiment(s) have pending work, %d segment(s) pending in total (%d need background "
         "extraction first).\n"
         % (len(todo), sum(p for _, _, _, _, p, _ in todo), sum(b for _, _, _, _, _, b in todo)))

    if args.dry_run:
        _log("Dry run -- nothing done.")
        return 0

    overall_t0 = time.time()
    results = []
    for i, (jp, exp_dir, video_names, total, pending, missing_bg) in enumerate(todo, 1):
        _log("\n" + "=" * 100)
        _log("[%d/%d] %s  (%d pending of %d, %d need background images)"
             % (i, len(todo), exp_dir, pending, total, missing_bg))
        _log("=" * 100)
        t0 = time.time()

        # Step 1+2: background images. Idempotent (skips existing outputs), safe to always call.
        if missing_bg > 0:
            try:
                em = ExperimentManager(_log)
                em.load_experiment_from_json(jp)
                _log("[%d/%d] extracting reference images..." % (i, len(todo)))
                _run_with_heartbeat("%s (extracting images)" % exp_dir, em.get_images_from_video,
                                    force_rerun=0)
                _log("[%d/%d] computing background images..." % (i, len(todo)))
                _run_with_heartbeat("%s (computing backgrounds)" % exp_dir,
                                    em.get_background_from_images, force_rerun=0)
            except Exception:
                _log("[%d/%d] BACKGROUND EXTRACTION FAILED for %s (tracking will skip segments "
                     "still missing a background):\n%s"
                     % (i, len(todo), exp_dir, traceback.format_exc()))

        # Step 3: tracking (same CLI path as buzzsuite_cli.py track). Pass the full video list minus
        # KNOWN_BAD_SEGMENTS as an explicit --videos filter -- cmd_track still applies its own
        # already-tracked skip on top of this intersection (see buzzsuite_cli.cmd_track), so this
        # only ever removes the permanently-hung segment(s), it doesn't force-retrack anything.
        wanted_videos = [v for v in video_names if v not in KNOWN_BAD_SEGMENTS]
        if args.workers is not None:
            workers_for_this = args.workers
        else:
            workers_for_this, reason = _adaptive_worker_count(pending, max_workers=args.max_workers)
            _log("[%d/%d] auto-selected %d worker(s): %s" % (i, len(todo), workers_for_this, reason))
        ns = argparse.Namespace(experiment=jp, workers=workers_for_this, force=False, videos=wanted_videos)
        try:
            rc = _run_with_heartbeat("%s (tracking)" % exp_dir, cmd_track, ns)
        except Exception:
            _log("[%d/%d] TRACKING FAILED (exception) for %s:\n%s"
                 % (i, len(todo), exp_dir, traceback.format_exc()))
            rc = 1

        # Step 4: concatenation into analyzed_data.pkl (same call the GUI makes after tracking).
        # Always attempted, regardless of `rc` -- cmd_track returns nonzero the moment ANY segment
        # fails (see buzzsuite_cli.cmd_track), which is right for a genuine tracking-run failure but
        # wrong here: a handful of permanently-corrupt raw files (KNOWN_BAD_SEGMENTS, or a fresh one
        # not yet catalogued) failing should never block getting analyzed_data.pkl for the hundreds
        # of segments that DID track successfully. concatenate_and_save_experiment_data is safe to
        # call unconditionally -- it only ever reads whatever files already exist in
        # final_tracking_data/ and no-ops with a log line if that folder is empty.
        try:
            em = ExperimentManager(lambda *a, **k: None)
            em.load_experiment_from_json(jp)
            concatenate_and_save_experiment_data(exp_dir, config=em.config, log_fn=_log,
                                                  force=args.force_concat)
        except Exception:
            _log("[%d/%d] CONCATENATION FAILED for %s:\n%s"
                 % (i, len(todo), exp_dir, traceback.format_exc()))

        elapsed = time.time() - t0
        results.append((exp_dir, rc, elapsed))
        _log("[%d/%d] done in %.0fs, tracking exit code %s" % (i, len(todo), elapsed, rc))

    # --force-concat also covers experiments with zero pending segments -- the loop above only
    # ever visits `todo` (pending > 0), so a fully-tracked experiment would otherwise never get
    # re-concatenated even though its analyzed_data.pkl is just as affected by a concatenation
    # code change (e.g. the population-data fallback fix) as any experiment still being tracked.
    if args.force_concat:
        done_only = [row for row in plan if row[4] == 0]
        if done_only:
            _log("\n" + "=" * 100)
            _log("--force-concat: re-concatenating %d already-fully-tracked experiment(s) too"
                 % len(done_only))
            _log("=" * 100)
            for jp, exp_dir, video_names, total, pending, missing_bg in done_only:
                try:
                    em = ExperimentManager(lambda *a, **k: None)
                    em.load_experiment_from_json(jp)
                    concatenate_and_save_experiment_data(exp_dir, config=em.config, log_fn=_log,
                                                          force=True)
                except Exception:
                    _log("FORCE-CONCAT FAILED for %s:\n%s" % (exp_dir, traceback.format_exc()))

    _log("\n" + "=" * 100)
    _log("ALL LUTZIA EXPERIMENTS DONE in %.0fs total" % (time.time() - overall_t0))
    for exp_dir, rc, elapsed in results:
        _log("  %-70s exit=%s  %.0fs" % (exp_dir, rc, elapsed))
    failures = [r for r in results if r[1] != 0]
    if failures:
        _log("\n%d experiment(s) had tracking failures -- re-run this script to retry (already-done "
             "segments are skipped automatically)." % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
