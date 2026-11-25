"""runner: the actions behind `buzzsuite_cli.py yolo <action>`.

    inventory  scan the data roots, resolve videos, choose one analysis per recording
    annotate   every tracked segment -> aligned annotations, events, activity (no decoding)
    verify     pixel check of take-offs / landings (reads each segment once)
    clock      fit the recording clock per experiment; write events/ and timeseries/ tables
    frames     sample frames, decode the 3-frame stack, pixel-QA every label
    build      YOLO tile datasets + calibration tiles (fast; re-run with another policy)
    clips      short annotated clips (MOT ground truth, labels, events)
    qa         alignment re-check, statistics, contact sheets, behaviour strips
    docs       README.md, docs/ROADMAP.md, docs/definitions.json
    all        everything above, in order

Segment work runs in spawn worker processes (safe_pool.SafePool) that exit with the parent. A
segment whose worker crashes (e.g. an unreadable numpy-2-era pickle) or runs past the per-task
timeout is logged as an error naming its file and skipped; the run carries on.
"""
import datetime as _dt
import os
import time

import numpy as np

from yolo_export import common as C
from yolo_export.safe_pool import SafePool

ACTIONS = ("inventory", "annotate", "verify", "clock", "frames", "build", "clips", "qa", "docs", "all")
DEFAULT_TASK_TIMEOUT = 1800   # s per job (one segment, clip or tile group); a stuck worker is stopped after this


def _logger(out_dir, base_log):
    path = os.path.join(out_dir, "logs", "yolo_export_%s.log" % _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(msg):
        base_log(msg)
        with open(path, "a", encoding="utf-8") as f:
            f.write("%s  %s\n" % (_dt.datetime.now().strftime("%H:%M:%S"), msg))
    return log


def _init_worker():
    from buzzswarm.clock_analysis import _exit_with_parent
    _exit_with_parent()


def job_name(job):
    """How a job is named in results and the log: the segment stem, or the clip name."""
    if "start" in job and "stem" in job:
        return "%s_%05d" % (job["stem"], int(job["start"]))
    return job.get("stem") or job.get("name") or "?"


def job_file(job):
    """The input file a segment job reads first (its tracking pickle), for crash reports."""
    if job.get("exp_dir") and job.get("stem"):
        return os.path.join(job["exp_dir"], "final_tracking_data", C.TRACK_PREFIX + job["stem"])
    return ""


def _failed(job, message):
    """The result of a job whose worker died, hung or raised: an error naming the file, so the run
    carries on without it."""
    where = job_file(job)
    alias = job.get("alias")
    return (job_name(job), "error", "%s%s%s" % (message, "; experiment %s" % alias if alias else "",
                                                 "; file %s" % where if where else ""))


def pool_run(fn, jobs, workers, log, label, timeout=DEFAULT_TASK_TIMEOUT):
    """Run fn over jobs (each returns (name, status, message)); log progress and errors.

    Jobs always run in worker processes (SafePool), so a job whose worker dies outright (e.g. an
    access violation while unpickling) or runs past `timeout` seconds becomes one 'error' result
    naming its file, and the other jobs carry on."""
    if not jobs:
        log("  %s: nothing to do" % label)
        return []
    t0 = time.time()
    results = []
    pool = SafePool(min(workers, len(jobs)), initializer=_init_worker, maxtasksperchild=20, timeout=timeout)
    ok = False
    try:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, _failed), 1):
            results.append(r)
            if r[1] == "error":
                log("    ! %s: %s" % (r[0], r[2]))
            if i % 10 == 0 or i == len(jobs):
                log("    %s %d/%d  (%.0f s)" % (label, i, len(jobs), time.time() - t0))
        ok = True
    finally:
        if ok:
            pool.close()
        else:
            pool.terminate()          # Ctrl-C / error: stop now, don't wait for queued work
    n = {}
    for r in results:
        n[r[1]] = n.get(r[1], 0) + 1
    log("  %s done: %s" % (label, n))
    return results


def _tiles_failed(job, message):
    """build.write_tiles' result for a job whose worker died or hung: no tiles, one error."""
    first = job["frames"][0]["png"] if job.get("frames") else "?"
    return 0, ["tiles of %d frame(s) from %s not written: %s" % (len(job.get("frames") or []), first, message)]


def _pool_map(workers, timeout=DEFAULT_TASK_TIMEOUT):
    """A map() for build.write_tiles that uses a pool (returns (map_fn, closer))."""
    if workers <= 1:
        return map, (lambda ok=True: None)
    pool = SafePool(workers, initializer=_init_worker, maxtasksperchild=50, timeout=timeout)

    def close(ok=True):
        if ok:
            pool.close()
        else:
            pool.terminate()
    return (lambda fn, jobs: pool.imap_unordered(fn, jobs, _tiles_failed)), close


def _segment_jobs(rows, out_dir, limit, force, k_frames, log=C.log_default):
    from yolo_export import annotate as AN
    from yolo_export import inventory as I
    defs = C.definitions()
    jobs = []
    for row in rows:
        try:
            geo = AN.job_geometry(row)
        except Exception as exc:  # noqa: BLE001 -- one bad experiment must not block the others
            log("  ! %s skipped: %s" % (row["alias"], exc))
            continue
        stems = I.segments_of(row)
        if limit and len(stems) > limit:
            pick = np.unique(np.linspace(0, len(stems) - 1, limit).round().astype(int))
            stems = [stems[i] for i in pick]
        tracking = C.load_settings(row["exp_dir"]).get("tracking") or {}
        k = k_frames or defs["sampling"]["frames_per_segment_%s" % ("aux" if row.get("role") == "aux" else "target")]
        for s in stems:
            jobs.append({"alias": row["alias"], "rig": row["rig_id"], "exp_dir": row["exp_dir"],
                         "video_dir": row["video_dir"], "stem": s, "out_dir": out_dir, "geo": geo,
                         "k_frames": int(k), "tracking": tracking, "zones_source": row.get("zones_source", ""),
                         "force": force})
    return jobs


def run(action, out_dir=None, roots=None, rig=None, aux=False, experiments=None, limit=None, workers=None,
        frames_per_segment=None, n_clips=60, force=False, single_class=False, log=C.log_default,
        task_timeout=DEFAULT_TASK_TIMEOUT):
    from yolo_export import annotate as AN
    from yolo_export import build as BU
    from yolo_export import clips as CL
    from yolo_export import clock as CK
    from yolo_export import docs as DOC
    from yolo_export import frames as FR
    from yolo_export import inventory as I
    from yolo_export import qa as QA
    from yolo_export import verify as VF

    if action not in ACTIONS:
        raise ValueError("unknown action %s (choose from %s)" % (action, ", ".join(ACTIONS)))
    roots = roots or C.discover_data_roots()
    out_dir = C.check_output_dir(out_dir or C.default_out_dir(), roots)
    os.makedirs(out_dir, exist_ok=True)
    log = _logger(out_dir, log)
    workers = int(workers or min(8, max(1, (os.cpu_count() or 4) - 2)))
    log("yolo %s -> %s  (workers %d%s%s%s)" % (action, out_dir, workers, ", rig %s" % rig if rig else "",
                                              ", aux" if aux else "", ", limit %d" % limit if limit else ""))
    steps = ["inventory", "annotate", "verify", "clock", "frames", "build", "clips", "qa", "docs"] if action == "all" else [action]
    inv = None if "inventory" in steps else I.load_inventory(out_dir)
    if inv is None and "inventory" not in steps:
        steps = ["inventory"] + steps
    sel = I.load_selection(out_dir)
    t_all = time.time()
    for step in steps:
        t0 = time.time()
        log("== %s" % step)
        if step == "inventory":
            inv = I.build_inventory(out_dir, roots, log)
            sel = I.load_selection(out_dir)
            continue
        rows = I.chosen_rows(inv, rig=rig, aux=aux, experiments=experiments)
        if not rows:
            log("  no chosen experiments for this selection (see inventory/inventory.csv)")
            continue
        if step == "annotate":
            pool_run(AN.annotate_segment, _segment_jobs(rows, out_dir, limit, force, frames_per_segment, log),
                     workers, log, "annotate", timeout=task_timeout)
        elif step == "verify":
            pool_run(VF.verify_segment, _segment_jobs(rows, out_dir, limit, force, frames_per_segment, log),
                     workers, log, "verify", timeout=task_timeout)
            if "clock" not in steps:
                for row in rows:                      # refresh the per-experiment events tables
                    ck = CK.load_clock(out_dir, row["alias"])
                    if ck:
                        CK.build_tables(row, out_dir, ck, log)
        elif step == "clock":
            for row in rows:
                try:
                    ck = CK.fit_experiment(row, out_dir, sel, log)
                    CK.build_tables(row, out_dir, ck, log)
                except Exception as exc:  # noqa: BLE001
                    log("  ! clock %s: %s" % (row["alias"], exc))
        elif step == "frames":
            pool_run(FR.frames_segment, _segment_jobs(rows, out_dir, limit, force, frames_per_segment, log),
                     workers, log, "frames", timeout=task_timeout)
        elif step == "build":
            b_rows = I.chosen_rows(inv, rig=rig, experiments=experiments) if (rig or experiments) else \
                [r for r in inv if r.get("chosen")]
            pmap, close = _pool_map(workers, task_timeout)
            done = False
            try:
                BU.build(out_dir, b_rows, sel, workers=workers, single_class=single_class, log=log, pool_map=pmap)
                done = True
            finally:
                close(done)
        elif step == "clips":
            jobs = []
            for row in rows:
                try:
                    jobs += CL.clip_jobs(row, out_dir, n_clips, force)
                except Exception as exc:  # noqa: BLE001
                    log("  ! clips %s: %s" % (row["alias"], exc))
            pool_run(CL.cut_clip, jobs, workers, log, "clips", timeout=task_timeout)
        elif step == "qa":
            QA.report(out_dir, rows, log)
        elif step == "docs":
            DOC.write_docs(out_dir, log)
        log("   (%s: %.0f s)" % (step, time.time() - t0))
    if "docs" not in steps:
        DOC.write_docs(out_dir, log)
    log("yolo %s finished in %.1f min -> %s" % (action, (time.time() - t_all) / 60.0, out_dir))
    return out_dir
