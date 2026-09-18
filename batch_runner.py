"""Shared tracking-job engine (workflow redesign, Step A).

Extracted from the thread+Pool loop that used to live only inside
``experiment_dashboard.ExperimentDashboard._run_activity``. Two layers:

* ``run_pool`` — the low-level ``Pool.imap_unordered(_run_batch_analysis, …)`` loop. This is the
  ONE place the tracking pool is driven, shared by all three surfaces: the GUI's Section 2 queue,
  the Experiment Dashboard, and ``buzzsuite_cli.cmd_track``.
* ``run_tracking_job_for_folder`` — the GUI-facing wrapper: rebuilds an ExperimentManager from the
  folder's ``experiment_*.json``, resolves pending videos, enforces the ``images_mortality/``
  background-image precondition (filter + report), then drives ``run_pool``.

The CLI deliberately does NOT use the wrapper: it has its own semantics (a ``--workers`` override,
``--videos``/``--force`` selection, and it *warns* about missing background images rather than
filtering those segments out). It calls ``run_pool`` directly, so the duplicated pool loop is gone
without flattening those differences.

Frozen numerics: this module only orchestrates workers; it does not change what
``_run_batch_analysis`` computes or how ``.pkl`` files are written.
"""
import os
import time
from multiprocessing import Pool, TimeoutError as _MPTimeoutError, cpu_count

from batch_processing_tab_manager import (_run_batch_analysis, _should_skip_video,
                                          _default_batch_worker_count, pool_concurrency_slot,
                                          load_excluded_segments)

# How often run_pool wakes up while waiting for a result, to notice Cancel or a segment the user
# just skipped. The stall checks below still run at most once per _STALL_CHECK_SECONDS, as before.
_POLL_SECONDS = 5
_STALL_CHECK_SECONDS = 60

try:
    import psutil
except ImportError:
    psutil = None


def _pool_workers_busy(pool, min_percent=5.0, sample_seconds=1.0):
    """True if any live worker process in `pool` is actively burning CPU right now.

    Used only at the moment the stall-timeout is about to fire (see run_pool below), to tell a
    genuinely dead/hung worker (0% CPU -- the case this watchdog exists to catch) apart from a
    worker that's simply grinding through an unusually dense segment (see single_video_analysis's
    known O(N^2) track-matching cost on 15000+-track sessions -- slow, not stuck). Killing the
    whole pool on the latter throws away real, otherwise-successful work for every other segment
    still in flight, for no reason other than an unlucky coincidence of several dense segments
    landing in the same batch (more likely now that worker counts auto-scale above the old flat
    default -- see DEVLOG's 2026-09-17 mass-stall incident). Degrades to "assume dead" (returns
    False, preserving the original conservative behavior) if psutil is unavailable or Pool's
    internal worker-process list can't be introspected -- never allowed to raise.
    """
    if psutil is None:
        return False
    try:
        workers = list(getattr(pool, "_pool", []))
    except Exception:
        return False
    procs = []
    for w in workers:
        pid = getattr(w, "pid", None)
        if pid is None or not w.is_alive():
            continue
        try:
            procs.append(psutil.Process(pid))
        except Exception:
            continue
    if not procs:
        return False
    for p in procs:
        try:
            p.cpu_percent(interval=None)  # prime the delta counter, first call is meaningless
        except Exception:
            pass
    time.sleep(sample_seconds)
    for p in procs:
        try:
            if p.cpu_percent(interval=None) >= min_percent:
                return True
        except Exception:
            continue
    return False


def resolve_worker_count(requested, n_pending, config=None, concurrency=1):
    """Workers for a tracking run: never more than cpu_count()-1 (no oversubscription), divided by
    ``concurrency`` if other Pool-consuming operations (other dashboard jobs, concatenation, the
    speed filter) are running at the same time, and never more than there are videos to process.
    ``requested`` None -> the configured/adaptive default."""
    concurrency = max(1, int(concurrency))
    workers = requested or _default_batch_worker_count(config, concurrency=concurrency)
    upper_bound = max(1, ((cpu_count() or 1) - 1) // concurrency)
    return max(1, min(int(workers), upper_bound, max(1, n_pending)))


def run_pool(pending, settings_file, folder_videos, folder_analysis, experiment_json_path,
             settings_data, worker_count, on_result=None, on_pool=None, cancel=None,
             stall_timeout=5400):
    """The single ``Pool.imap_unordered(_run_batch_analysis, …)`` loop for the whole app.

    Builds the frozen 8-tuple argument list, runs a cancellable pool, and invokes
    ``on_result(result, idx, total)`` per completed segment. ``on_pool(pool)`` hands the Pool to the
    caller (and ``on_pool(None)`` on completion) so a Cancel button can ``terminate()`` it —
    ``imap_unordered`` dispatches every task up front, so setting ``cancel`` alone would not stop
    already-dispatched work. Returns ``(done, total)``.

    ``stall_timeout`` (seconds, default 5400 = 90min) guards against a worker that dies mid-task
    WITHOUT raising a catchable Python exception -- e.g. a hard OS-level crash from an extreme
    allocation attempt on an unusually dense segment (observed live: a 21,554x21,554 float64 matrix
    allocation; other segments in the same run hit `cv::OutOfMemoryError`/`MemoryError` and were
    caught fine by `_run_batch_analysis`'s try/except, but a hard process crash isn't a Python
    exception at all). Plain `multiprocessing.Pool` has no built-in way to detect this: a dead
    worker just leaves its task's result never delivered, and `imap_unordered`/`pool.join()` waits
    for it forever -- confirmed live: all worker processes at 0% CPU, zero new output anywhere for
    14+ hours, with only the last couple of a batch's tasks left outstanding. If no NEW result
    arrives within `stall_timeout` seconds while tasks remain outstanding, the pool is terminated
    and every never-returned segment is reported via `on_result` as a synthetic failure so the
    caller's accounting and the next rerun both see it as pending again (same as any other failure
    -- no special-casing needed downstream).

    Callbacks may fire off the main thread; Tk callers marshal via ``after()`` themselves.
    """
    total = len(pending)
    if not total:
        return 0, 0
    # Adaptive OpenCV thread budget per worker: divide cores among workers so total threads ~= cores.
    # Result-neutral (threads only).
    cv2_threads = max(1, (cpu_count() or 1) // max(1, worker_count))
    args = [(v, settings_file, folder_videos, folder_analysis, experiment_json_path,
             settings_data, True, cv2_threads) for v in pending]

    # maxtasksperchild=1: recycle each worker process after every single segment. Without this,
    # a worker process lives for the *entire* run and processes every segment dispatched to it
    # back-to-back in the same process -- any per-segment memory that isn't released (matplotlib/
    # cv2/numpy objects, reference cycles, anything not explicitly freed in single_video_analysis's
    # object graph) accumulates for as long as that worker keeps running, on top of whatever a
    # single dense segment needs at its own peak. This is why reducing worker_count alone only
    # delayed the crash rather than fixing it (observed live: 30 workers died almost immediately,
    # 21 workers got much further before dying, 3 workers still eventually died) -- fewer workers
    # means fewer segments accumulate per worker per unit of wall time, not zero. Recycling after
    # every task returns each worker's entire memory to the OS unconditionally between segments,
    # independent of whether the accumulation's exact source is ever found. The cost is one extra
    # process spawn (re-importing cv2/numpy/etc, a few seconds) per segment, negligible against
    # segments that already take many minutes each.
    pool = Pool(processes=worker_count, maxtasksperchild=1)
    if on_pool:
        on_pool(pool)
    done = 0
    stalled = False
    stall_reason = None
    cancelled = False
    completed_names = set()
    # Segments the user skipped while they were still outstanding. A Pool can't abandon one running
    # task, so once EVERY outstanding segment is a skipped one, the pool is stopped and those are
    # reported as skipped -- the run then finishes normally (and the caller builds the data).
    skipped_rest = []
    last_stall_check = 0.0
    # A worker still burning real CPU past `stall_timeout` is grinding through an unusually dense
    # segment (see single_video_analysis's documented O(N^2) track-matching cost), not dead --
    # _pool_workers_busy() below lets that keep running instead of being killed. But an actual
    # infinite loop also burns CPU (this is precisely the failure mode the `frame_idx` fix already
    # patched once for a different code path -- there is no guarantee every such bug is found), so
    # CPU activity alone cannot excuse a stall forever: `hard_stall_timeout` is the absolute ceiling
    # past which the pool is terminated regardless of what the workers appear to be doing.
    hard_stall_timeout = stall_timeout * 4
    try:
        it = pool.imap_unordered(_run_batch_analysis, args)
        last_progress = time.time()
        # Poll with a short per-call timeout (NOT stall_timeout itself) so `cancel` is checked
        # frequently even while waiting on a slow segment, rather than only every stall_timeout.
        while done < total:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            outstanding = [v for v in pending if v not in completed_names]
            excluded = load_excluded_segments(folder_analysis)
            if outstanding and excluded and all(v in excluded for v in outstanding):
                skipped_rest = outstanding
                break
            try:
                result = it.next(timeout=_POLL_SECONDS)
            except _MPTimeoutError:
                now = time.time()
                if now - last_stall_check < _STALL_CHECK_SECONDS:
                    continue
                last_stall_check = now
                elapsed = now - last_progress
                if elapsed > hard_stall_timeout:
                    stalled = True
                    stall_reason = (
                        "no result for %ds even though worker(s) may still have been active -- "
                        "this exceeds the absolute %ds ceiling (4x the normal stall timeout), so "
                        "the pool was terminated regardless" % (int(elapsed), hard_stall_timeout))
                    break
                if elapsed > stall_timeout and not _pool_workers_busy(pool):
                    stalled = True
                    stall_reason = (
                        "worker did not return within %ds and shows no CPU activity -- likely "
                        "died from an extreme memory allocation attempt without raising a "
                        "catchable exception; the pool was terminated" % stall_timeout)
                    break
                continue
            done += 1
            last_progress = time.time()
            if isinstance(result, dict):
                completed_names.add(result.get('video_name'))
            if on_result:
                on_result(result, done, total)
        if not stalled and not cancelled and not skipped_rest:
            pool.close()
            pool.join()
    finally:
        try:
            pool.terminate()
        except Exception:
            pass
        if on_pool:
            on_pool(None)
    if stalled:
        stuck = [v for v in pending if v not in completed_names]
        for v in stuck:
            done += 1
            if on_result:
                on_result({'ok': False, 'video_name': v,
                           'error': '%s. Re-run to retry.' % stall_reason},
                          done, total)
    for v in skipped_rest:
        done += 1
        if on_result:
            on_result({'ok': True, 'video_name': v, 'fps': None, 'skipped': True, 'excluded': True},
                      done, total)
    return done, total


def find_experiment_json(folder_analysis):
    for name in os.listdir(folder_analysis):
        if name.startswith('experiment_') and name.endswith('.json'):
            return os.path.join(folder_analysis, name)
    return None


def resolve_pending_videos(folder_videos, folder_analysis, video_names=None):
    """All .mp4 segment names to track, honoring an explicit ``video_names`` selection if given
    (used regardless of prior tracked state — an explicit user pick), else 'all pending'
    (everything not already tracked and not skipped by the user, mirrors
    run_tracking_all_untracked)."""
    if video_names is not None:
        return list(video_names)
    if not folder_videos or not os.path.isdir(folder_videos):
        return []
    all_videos = sorted(f[:-4] for f in os.listdir(folder_videos)
                        if f.lower().endswith('.mp4') and not f.startswith('.'))
    excluded = load_excluded_segments(folder_analysis)
    return [v for v in all_videos
            if not _should_skip_video(v, folder_analysis) and v not in excluded]


def run_tracking_job_for_folder(folder_analysis, config_manager, video_names=None,
                                on_step=None, on_video_done=None, on_pool=None,
                                on_pending=None, cancel=None, requested_workers=None):
    """Track ``video_names`` (or all pending, if None) in the experiment at ``folder_analysis``.

    Rebuilds a throwaway ``ExperimentManager`` from the folder's ``experiment_*.json`` (sharing
    ``config_manager`` so recents/last-experiment bookkeeping doesn't drift from the caller's own
    manager) — the same approach the Dashboard used before this extraction.

    ``requested_workers``, when given, overrides the auto-computed worker count (still capped by
    ``resolve_worker_count`` at ``cpu_count()-1`` and at the number of pending segments). Default
    ``None`` keeps the previous auto-only behavior. This parameter used to not exist at all: Section
    2's queue had a "Worker Processes" spinbox that was never actually read anywhere, so every
    tracking run silently used the auto default regardless of what the user set -- see DEVLOG for
    how that (on a 32-core/48GB machine, with some segments needing several GB each for their
    assembly step) let enough simultaneously-dense segments pile up to exhaust memory and crash the
    whole run, with no way for the user to dial concurrency down from the GUI.

    Returns a dict: {'total': int, 'done': int, 'skipped_no_background': [names]}.
    ``on_step(text)`` / ``on_video_done(result, idx, total)`` report progress; ``on_pending(names)``
    fires once, right before the pool starts, with the exact list of segments about to be tracked
    (lets a caller build a per-file progress display); ``on_pool(pool)`` lets the caller stash the
    Pool (or None once finished) for cooperative cancellation via ``cancel`` (a threading.Event
    checked between completed videos). All callbacks may fire off the main thread — callers marshal
    any Tk updates via ``after()`` themselves.
    """
    from experiment_manager import ExperimentManager

    def _step(text):
        if on_step:
            on_step(text)

    json_path = find_experiment_json(folder_analysis)
    em = ExperimentManager(lambda m: None, config_manager)
    if json_path:
        em.load_experiment_from_json(json_path)
    folder_videos = getattr(em, 'folder_videos', None)
    settings_file = getattr(em, 'settings_file', None)
    if not folder_videos or not os.path.isdir(folder_videos):
        _step("no video folder for this experiment")
        return {'total': 0, 'done': 0, 'skipped_no_background': []}

    pending = resolve_pending_videos(folder_videos, folder_analysis, video_names)
    if not pending:
        _step("nothing to track")
        return {'total': 0, 'done': 0, 'skipped_no_background': []}

    # Background-image precondition: the tracker SILENTLY skips any segment lacking its
    # Initialization background image images_mortality/<video>.png. Filter here (skip + report)
    # rather than blocking with a messagebox, since this path also runs off the main thread for
    # both the Dashboard and Section 2's multi-experiment queue.
    mort_dir = os.path.join(folder_analysis, "images_mortality")
    ready = [v for v in pending if os.path.isfile(os.path.join(mort_dir, v + ".png"))]
    missing = [v for v in pending if v not in ready]
    if missing:
        _step(f"{len(missing)} segment(s) missing background image — skipped")
    if not ready:
        _step("needs Initialization (no segments have a background image)")
        return {'total': 0, 'done': 0, 'skipped_no_background': missing}

    # Without this, the caller's status display has nothing to show from here until the FIRST
    # segment fully completes (on_video_done) -- which single_video_analysis's own per-segment
    # logs show can take 20-45+ minutes for the full segmentation/tracking/assembly pipeline. That
    # silent gap reads as "stuck", not "working" (see DEVLOG: repeated relaunches during exactly
    # this gap discarded real in-progress work).
    _step(f"tracking {len(ready)} segment(s)…")
    if on_pending:
        on_pending(ready)

    with pool_concurrency_slot() as concurrency:
        workers = resolve_worker_count(requested_workers, len(ready), getattr(em, 'config', None),
                                       concurrency=concurrency)
        done, total = run_pool(
            ready, settings_file, folder_videos, folder_analysis, json_path,
            getattr(em, 'settings', None), workers,
            on_result=on_video_done, on_pool=on_pool, cancel=cancel)

    return {'total': total, 'done': done, 'skipped_no_background': missing}
