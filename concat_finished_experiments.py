#!/usr/bin/env python
"""Concatenate every finished experiment's tracking into analyzed_data.pkl, then merge all of them
into one combined activity table -- a way to get plottable data out of BuzzSuite while the in-app
Plotting tab is broken.

Three passes over every experiment_*.json found under {DataRoot}\\Analysis\\*\\*\\*\\:

1. Status (read-only): raw segments in the experiment's folder_videos vs tracked segments in
   final_tracking_data/. Known-bad raw files (run_flare_pipeline.KNOWN_BAD_SEGMENTS -- structurally
   broken recordings that can never be tracked) are excluded, so an experiment is FINISHED when every
   other segment has tracking output. Also lists Recording\\*\\*\\*\\ folders that no
   experiment_*.json points at (raw data with no experiment set up yet).
2. Re-concatenate (writes analyzed_data.pkl): for each selected experiment whose analyzed_data.pkl
   is missing, older than its newest final_tracking_data/ file, or was pickled by a numpy 2
   environment (unloadable in buzzsuite_env's numpy 1.x), calls the same
   concatenate_and_save_experiment_data() the GUI calls right after tracking.
3. Combine (writes under BuzzSuite/plots/combined_activity/): loads each selected experiment's
   population_data one at a time, resamples it (default 1 min; total_pixels_moved summed, every
   other variable averaged -- same aggregation as activity_plot_manager), tags each row with
   assay/location/experiment/alias, and writes one long-format CSV + pickle, plus a per-experiment
   status CSV.

Usage
-----
    python concat_finished_experiments.py --dry-run          # status table only, writes nothing
    python concat_finished_experiments.py                    # re-concat stale ones, then combine
    python concat_finished_experiments.py --only Flare       # filter by path substring
    python concat_finished_experiments.py --include-partial  # also include partly-tracked experiments
    python concat_finished_experiments.py --force-concat     # re-concat every selected experiment
    python concat_finished_experiments.py --no-concat        # combine existing analyzed_data.pkl only
    python concat_finished_experiments.py --resample 5min    # coarser bins ("none" = raw 1 s rows)
"""
import argparse
import gc
import glob
import json
import os
import pickle
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

TRACK_PREFIX = "forward_mosq_tracks_"
# Summed (not averaged) when resampling -- mirrors activity_plot_manager._SUM_VARIABLES.
SUM_VARIABLES = ("total_pixels_moved",)


def _log(msg):
    print(msg, flush=True)


def _tracked_segments(folder_analysis):
    """{segment_name: mtime} for every tracking output in final_tracking_data/ (same filename filter
    concatenate_and_save_experiment_data uses)."""
    folder = os.path.join(folder_analysis, "final_tracking_data")
    out = {}
    if not os.path.isdir(folder):
        return out
    for entry in os.scandir(folder):
        name = entry.name
        if name.startswith(TRACK_PREFIX) and not name.endswith((".png", ".csv", ".txt")):
            try:
                out[name[len(TRACK_PREFIX):]] = entry.stat().st_mtime
            except OSError:
                continue
    return out


def _raw_segments(folder_videos):
    """Set of raw segment names (no .mp4) in folder_videos, or None if the folder is missing --
    same listing experiment_analysis uses to build list_video_name."""
    if not folder_videos or not os.path.isdir(folder_videos):
        return None
    return set(os.path.splitext(f)[0] for f in os.listdir(folder_videos)
               if f.endswith(".mp4") and not f.startswith("."))


def _written_by_numpy2(path):
    """True if this interpreter runs numpy 1.x and the pickle references numpy 2's 'numpy._core'
    module path. Such a file (written by some other, newer Python env) hard-fails to load here --
    SystemError in structseq.c, seen on Bangkok/Lutzia_F on 2026-09-11 -- and leaves the interpreter
    unusable for the next load, so it is re-concatenated instead of loaded."""
    import numpy
    if int(numpy.__version__.split(".")[0]) >= 2:
        return False
    needle = b"numpy._core"
    tail = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                return False
            if needle in tail + chunk:
                return True
            tail = chunk[-len(needle):]


def _pkl_state(exp_dir, newest_track):
    """('missing' | 'stale' | 'numpy2' | 'ok', mtime) for the experiment's analyzed_data.pkl."""
    pkl = os.path.join(exp_dir, "analyzed_data.pkl")
    if not os.path.isfile(pkl):
        return "missing", None
    mtime = os.path.getmtime(pkl)
    if newest_track is not None and newest_track > mtime:
        return "stale", mtime
    if _written_by_numpy2(pkl):
        return "numpy2", mtime
    return "ok", mtime


def _survey(json_path, analysis_root):
    """One status row for an experiment. Read-only."""
    # run_flare_pipeline dropped its size-based _truncated_segments() for an explicit list of
    # structurally-audited broken files (see its module docstring); mirror that here.
    from run_flare_pipeline import KNOWN_BAD_SEGMENTS

    with open(json_path, "r") as f:
        meta = json.load(f)
    exp_dir = os.path.dirname(os.path.abspath(json_path))
    parts = os.path.relpath(exp_dir, analysis_root).split(os.sep)
    folder_videos = meta.get("folder_videos")

    raw = _raw_segments(folder_videos)
    truncated = (raw & KNOWN_BAD_SEGMENTS) if raw else set()
    tracked = _tracked_segments(exp_dir)
    trackable = (raw - truncated) if raw is not None else set()
    pending = sorted(trackable - set(tracked))

    if not tracked:
        status = "NO VIDEO" if raw is None else ("EMPTY" if not trackable else "NOT STARTED")
    elif pending:
        status = "PARTIAL"
    else:
        status = "FINISHED"

    newest_track = max(tracked.values()) if tracked else None
    pkl_state, pkl_mtime = _pkl_state(exp_dir, newest_track)
    return {
        "assay": parts[0] if len(parts) > 0 else "",
        "location": parts[1] if len(parts) > 1 else "",
        "experiment": parts[2] if len(parts) > 2 else "",
        "alias": meta.get("experiment_alias", ""),
        "exp_dir": exp_dir,
        "json_path": json_path,
        "folder_videos": folder_videos or "",
        "n_raw": len(raw) if raw is not None else 0,
        "n_truncated": len(truncated),
        "n_tracked": len(tracked),
        "n_pending": len(pending),
        "first_pending": pending[0] if pending else "",
        "status": status,
        "pkl_state": pkl_state,
        "pkl_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(pkl_mtime)) if pkl_mtime else "",
    }


def _unregistered_recordings(data_root, rows):
    """Recording\\*\\*\\*\\ folders containing .mp4/.h264 that no experiment_*.json points at."""
    known = set(os.path.normcase(os.path.normpath(r["folder_videos"])) for r in rows if r["folder_videos"])
    out = []
    for d in sorted(glob.glob(os.path.join(data_root, "Recording", "*", "*", "*"))):
        if not os.path.isdir(d) or os.path.normcase(os.path.normpath(d)) in known:
            continue
        try:
            names = os.listdir(d)
        except OSError:
            continue
        n_mp4 = sum(1 for f in names if f.endswith(".mp4") and not f.startswith("."))
        n_h264 = sum(1 for f in names if f.endswith(".h264") and not f.startswith("."))
        if n_mp4 or n_h264:
            out.append((os.path.relpath(d, data_root), n_mp4, n_h264))
    return out


def _resample(pop, rule):
    """Numeric columns only; SUM_VARIABLES summed per bin (empty bins stay NaN, not 0), the rest
    averaged. rule=None keeps the native 1 s rows."""
    pop = pop.select_dtypes(include="number").sort_index()
    if rule is None:
        return pop
    r = pop.resample(rule)
    out = r.mean()
    sum_cols = [c for c in SUM_VARIABLES if c in pop.columns]
    if sum_cols:
        out[sum_cols] = r[sum_cols].sum(min_count=1)
    return out.dropna(how="all")


def _print_status(rows, data_root, unregistered):
    _log("%-58s %5s %5s %7s %7s  %-11s %s" % ("Experiment", "raw", "trunc", "tracked", "pending",
                                            "status", "analyzed_data.pkl"))
    for r in rows:
        pkl = r["pkl_state"] + (" (%s)" % r["pkl_mtime"] if r["pkl_mtime"] else "")
        _log("%-58s %5d %5d %7d %7d  %-11s %s" % (os.path.relpath(r["exp_dir"], data_root)[:58],
                                                 r["n_raw"], r["n_truncated"], r["n_tracked"],
                                                 r["n_pending"], r["status"], pkl))

    def count(s):
        return sum(1 for r in rows if r["status"] == s)

    partial = [r for r in rows if r["status"] in ("PARTIAL", "NOT STARTED")]
    _log("\n%d experiment(s): %d finished, %d partial, %d not started, %d with no recording folder, "
         "%d empty." % (len(rows), count("FINISHED"), count("PARTIAL"), count("NOT STARTED"),
                        count("NO VIDEO"), count("EMPTY")))
    _log("%d segment(s) still to track across %d experiment(s) (truncated raw files excluded)."
         % (sum(r["n_pending"] for r in partial), len(partial)))
    if unregistered:
        _log("\n%d Recording folder(s) with video but NO experiment set up:" % len(unregistered))
        for rel, n_mp4, n_h264 in unregistered:
            _log("  %-70s %5d mp4 %5d h264" % (rel, n_mp4, n_h264))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None,
                        help="Data root containing Recording/ and Analysis/ (default: auto-detected).")
    parser.add_argument("--only", default=None,
                        help="Only experiments whose Analysis-relative path contains this substring.")
    parser.add_argument("--dry-run", action="store_true", help="Print the status table, write nothing.")
    parser.add_argument("--include-partial", action="store_true",
                        help="Also concatenate/combine experiments that still have untracked segments.")
    parser.add_argument("--force-concat", action="store_true",
                        help="Re-concatenate every selected experiment even if analyzed_data.pkl looks current.")
    parser.add_argument("--no-concat", action="store_true",
                        help="Skip re-concatenation; combine whatever analyzed_data.pkl files exist.")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Worker processes for concatenation (default: cores-2).")
    parser.add_argument("--resample", default="1min",
                        help="Bin width for the combined table, e.g. 1min, 5min, 1h; 'none' = raw 1 s rows.")
    parser.add_argument("--out", default=None,
                        help="Output folder (default: BuzzSuite/plots/combined_activity/).")
    args = parser.parse_args()

    from path_utils import get_package_root, resolve_data_root

    data_root = args.data_root or os.environ.get("BUZZSUITE_DATA_ROOT") or resolve_data_root()
    analysis_root = os.path.join(data_root, "Analysis")
    jsons = sorted(glob.glob(os.path.join(analysis_root, "*", "*", "*", "experiment_*.json")))
    if args.only:
        jsons = [jp for jp in jsons if args.only.lower() in os.path.relpath(jp, analysis_root).lower()]
    if not jsons:
        _log("No experiment_*.json found under %s" % analysis_root)
        return 1

    rows, seen = [], set()
    for jp in jsons:
        exp_dir = os.path.normcase(os.path.dirname(os.path.abspath(jp)))
        if exp_dir in seen:
            _log("SKIP (second experiment_*.json in the same folder): %s" % jp)
            continue
        seen.add(exp_dir)
        try:
            rows.append(_survey(jp, analysis_root))
        except Exception as e:
            _log("SKIP (could not read): %s -- %s" % (jp, e))

    unregistered = [] if args.only else _unregistered_recordings(data_root, rows)
    _print_status(rows, data_root, unregistered)

    wanted = ("FINISHED", "PARTIAL") if args.include_partial else ("FINISHED",)
    selected = [r for r in rows if r["status"] in wanted]
    to_concat = selected if args.force_concat else [r for r in selected if r["pkl_state"] != "ok"]
    _log("\nSelected %d experiment(s) to combine; %d need (re-)concatenation first%s."
         % (len(selected), 0 if args.no_concat else len(to_concat),
            " (skipped: --no-concat)" if args.no_concat and to_concat else ""))
    if args.dry_run:
        _log("Dry run -- nothing written.")
        return 0
    if not selected:
        return 0

    # Pass 2: re-concatenate stale/missing analyzed_data.pkl.
    if not args.no_concat and to_concat:
        from batch_processing_tab_manager import concatenate_and_save_experiment_data
        config = {"batch_processing_workers": args.workers} if args.workers else None
        for i, r in enumerate(to_concat, 1):
            _log("\n[concat %d/%d] %s (%d segments, analyzed_data.pkl %s)"
                 % (i, len(to_concat), r["exp_dir"], r["n_tracked"], r["pkl_state"]))
            t0 = time.time()
            try:
                concatenate_and_save_experiment_data(r["exp_dir"], config=config, log_fn=_log)
            except Exception as e:
                _log("  CONCATENATION FAILED: %s" % e)
            newest = max(_tracked_segments(r["exp_dir"]).values() or [None])
            r["pkl_state"], mtime = _pkl_state(r["exp_dir"], newest)
            r["pkl_mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else ""
            _log("  done in %.0fs -> analyzed_data.pkl %s" % (time.time() - t0, r["pkl_state"]))
            gc.collect()

    # Pass 3: combine population_data across experiments, one file in memory at a time.
    import pandas as pd

    rule = None if args.resample.lower() == "none" else args.resample
    tag = args.resample.lower()
    frames = []
    for r in selected:
        pkl = os.path.join(r["exp_dir"], "analyzed_data.pkl")
        if not os.path.isfile(pkl):
            _log("SKIP combine (no analyzed_data.pkl): %s" % r["exp_dir"])
            continue
        if r["pkl_state"] == "numpy2":
            _log("SKIP combine (analyzed_data.pkl written by numpy 2, unreadable here -- re-run "
                 "without --no-concat to rebuild it): %s" % r["exp_dir"])
            continue
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            pop = data.get("population_data") if isinstance(data, dict) else None
            del data
            if pop is None or len(pop) == 0:
                _log("SKIP combine (empty population_data): %s" % pkl)
                continue
            if "total_pixels_moved" not in pop.columns:
                _log("WARNING: %s has no total_pixels_moved (concatenated before July 2026) -- "
                     "re-run with --force-concat --only %s to add it." % (pkl, r["experiment"]))
            df = _resample(pop, rule)
            del pop
        except Exception as e:
            _log("SKIP combine (could not load %s): %s" % (pkl, e))
            continue
        df.index.name = "timestamp"
        df = df.reset_index()
        for col in ("alias", "experiment", "location", "assay"):
            df.insert(0, col, r[col])
        frames.append(df)
        _log("combined %-58s %8d rows  %s -> %s"
             % (os.path.relpath(r["exp_dir"], data_root)[:58], len(df),
                df["timestamp"].min(), df["timestamp"].max()))
        gc.collect()

    out_dir = args.out or os.path.join(get_package_root(), "plots", "combined_activity")
    os.makedirs(out_dir, exist_ok=True)
    status_path = os.path.join(out_dir, "experiment_status.csv")
    pd.DataFrame(rows).drop(columns=["json_path"]).to_csv(status_path, index=False)
    _log("\nWrote %s" % status_path)
    if not frames:
        _log("Nothing to combine.")
        return 1

    combined = pd.concat(frames, ignore_index=True, sort=False)
    csv_path = os.path.join(out_dir, "combined_population_%s.csv" % tag)
    pkl_path = os.path.join(out_dir, "combined_population_%s.pkl" % tag)
    combined.to_pickle(pkl_path)
    combined.to_csv(csv_path, index=False)
    _log("Wrote %s  (%d rows x %d columns, %d experiments)"
         % (csv_path, len(combined), combined.shape[1], len(frames)))
    _log("Wrote %s  (pd.read_pickle)" % pkl_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
