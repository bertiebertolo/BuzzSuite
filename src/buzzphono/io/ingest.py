"""Convert tracking pickles into the canonical cache (see :mod:`buzzphono.io.cache`).

Each tier has a pure extractor (tracker object -> arrays + summary) and the orchestrator
:func:`ingest` runs them in worker processes, skipping chunks that are already current. Every
detection BuzzWatch wrote is kept: nothing is filtered here except entries the published analysis
also dropped (positions or frame numbers that are not finite numbers), and those are counted.

Resting tier (``resting/<chunk>.npz``)
    Resting detections are mostly static, so consecutive detections of the same object on
    consecutive frames at the same position are stored as runs: ``run_obj`` (index into
    ``obj_ids``), ``run_f0`` (first frame), ``run_len`` (frames), ``run_xy`` (x, y in pixels).
    Detections at non-integer frame numbers (none seen so far) go to ``exc_*`` arrays one by one.
Moving tier (``moving/<chunk>.npz``)
    ``det_obj``, ``det_frame`` (float), ``det_xy`` for every moving detection.
Tracks tier (``tracks/<chunk>.npz``)
    BuzzWatch's assembled per-mosquito tracks. Track ``t`` covers frames
    ``trk_start[t] .. trk_start[t] + trk_len[t] - 1``; its runs ``trk_run_off[t]:trk_run_off[t+1]``
    hold ``run_len``, ``run_state`` (1 flying, 0 resting, -1 unknown/gap) and ``run_xy`` (NaN in
    gaps). ``nf`` is the chunk's true frame count (the length of the tracker's per-frame list).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from buzzphono.errors import Cancelled, DataError
from buzzphono.io.cache import (
    EXTRACTOR_VERSION,
    CacheIndex,
    atomic_save_npz,
    cache_dir_for,
    source_identity,
)
from buzzphono.io.discover import ExperimentLayout
from buzzphono.io.naming import TrackingKind
from buzzphono.io.safe_pickle import TrackerObject, load_tracking_pickle

Arrays = dict[str, np.ndarray]
Summary = dict[str, Any]
ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


# ------------------------------------------------------------------------- per-object conversion
def _as_xy(coordinates: list[Any], n: int) -> np.ndarray:
    """(n, 2) float array from a list of (x, y) pairs; malformed entries become NaN."""
    try:
        xy = np.asarray(coordinates[:n], dtype=np.float64)
        if xy.shape == (n, 2):
            return xy
    except (TypeError, ValueError):
        pass
    out = np.full((n, 2), np.nan)
    for i, c in enumerate(coordinates[:n]):
        try:
            if len(c) == 2:
                out[i] = (float(c[0]), float(c[1]))
        except (TypeError, ValueError):
            continue
    return out


def _detections(obj: TrackerObject) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Summary]:
    """All (object index, frame, x, y) detections of a resting/moving tracker, legacy filters applied."""
    ids: list[int] = []
    frames: list[np.ndarray] = []
    xys: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    len_mismatch = nonfinite = 0
    for i, (key, rec) in enumerate(obj.objects.items()):
        ids.append(int(key))
        ts = rec.get("time_stamp") or []
        co = rec.get("coordinates") or []
        n = min(len(ts), len(co))
        len_mismatch += max(len(ts), len(co)) - n
        if n == 0:
            continue
        try:
            f = np.asarray(ts[:n], dtype=np.float64)
        except (TypeError, ValueError):
            f = np.array([_float_or_nan(v) for v in ts[:n]])
        xy = _as_xy(co, n)
        ok = np.isfinite(f) & np.isfinite(xy).all(axis=1)
        nonfinite += int(n - ok.sum())
        if ok.any():
            frames.append(f[ok])
            xys.append(xy[ok])
            owner.append(np.full(int(ok.sum()), i, dtype=np.int32))
    obj_ids = np.asarray(ids, dtype=np.int64)
    if frames:
        frame = np.concatenate(frames)
        xy_all = np.concatenate(xys)
        own = np.concatenate(owner)
    else:
        frame, xy_all, own = np.empty(0), np.empty((0, 2)), np.empty(0, dtype=np.int32)
    summary = {
        "n_objects": len(ids),
        "n_detections": int(frame.size),
        "n_nonfinite_dropped": nonfinite,
        "n_length_mismatch": len_mismatch,
        "max_frame": float(frame.max()) if frame.size else None,
    }
    return obj_ids, own, frame, xy_all, summary


def _float_or_nan(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def extract_resting(obj: TrackerObject) -> tuple[Arrays, Summary]:
    obj_ids, own, frame, xy, summary = _detections(obj)
    is_int = (frame == np.floor(frame)) & (np.abs(frame) < 2**53)
    fi = frame[is_int].astype(np.int64)
    oi, pi = own[is_int], xy[is_int]
    n = fi.size
    brk = np.ones(n, dtype=bool)
    if n > 1:
        brk[1:] = (
            (oi[1:] != oi[:-1])
            | (fi[1:] != fi[:-1] + 1)
            | (pi[1:, 0] != pi[:-1, 0])
            | (pi[1:, 1] != pi[:-1, 1])
        )
    starts = np.flatnonzero(brk)
    run_len = np.diff(np.append(starts, n)).astype(np.int32)
    exc = ~is_int
    arrays: Arrays = {
        "obj_ids": obj_ids,
        "run_obj": oi[starts].astype(np.int32),
        "run_f0": fi[starts],
        "run_len": run_len,
        "run_xy": pi[starts],
        "exc_obj": own[exc].astype(np.int32),
        "exc_frame": frame[exc],
        "exc_xy": xy[exc],
    }
    summary.update({"n_runs": int(starts.size), "n_noninteger_frames": int(exc.sum())})
    return arrays, summary


def extract_moving(obj: TrackerObject) -> tuple[Arrays, Summary]:
    obj_ids, own, frame, xy, summary = _detections(obj)
    return {"obj_ids": obj_ids, "det_obj": own, "det_frame": frame, "det_xy": xy}, summary


def _state_codes(state: list[Any], n: int) -> np.ndarray:
    try:
        st = np.asarray(state[:n], dtype=np.float64)
    except (TypeError, ValueError):
        st = np.array([_float_or_nan(v) for v in state[:n]])
    return np.where(st == 1, 1, np.where(st == 0, 0, -1)).astype(np.int8)


def extract_tracks(obj: TrackerObject) -> tuple[Arrays, Summary]:
    trk_id: list[int] = []
    trk_start: list[int] = []
    trk_len: list[int] = []
    run_off = [0]
    lens: list[np.ndarray] = []
    states: list[np.ndarray] = []
    xys: list[np.ndarray] = []
    mismatch = bad_start = 0
    for key, rec in obj.objects.items():
        co = rec.get("coordinates") or []
        st = rec.get("state")
        n = len(co) if st is None else min(len(co), len(st))
        mismatch += 0 if st is None else abs(len(co) - len(st))
        start = rec.get("start")
        try:
            start_i = int(start) if start is not None else None
        except (TypeError, ValueError):
            start_i = None
        if start_i is None:
            bad_start += 1
            continue
        if n == 0:
            continue
        xy = _as_xy(co, n)
        code = _state_codes(st, n) if st is not None else np.full(n, -1, dtype=np.int8)
        nanx, nany = np.isnan(xy[:, 0]), np.isnan(xy[:, 1])
        brk = np.ones(n, dtype=bool)
        if n > 1:
            same_x = (xy[1:, 0] == xy[:-1, 0]) | (nanx[1:] & nanx[:-1])
            same_y = (xy[1:, 1] == xy[:-1, 1]) | (nany[1:] & nany[:-1])
            brk[1:] = (code[1:] != code[:-1]) | ~same_x | ~same_y
        starts = np.flatnonzero(brk)
        trk_id.append(int(key))
        trk_start.append(start_i)
        trk_len.append(n)
        lens.append(np.diff(np.append(starts, n)).astype(np.int32))
        states.append(code[starts])
        xys.append(xy[starts])
        run_off.append(run_off[-1] + starts.size)
    nf = len(getattr(obj, "time_stamp", []) or [])
    settings = getattr(obj, "settings", None)
    arrays: Arrays = {
        "trk_id": np.asarray(trk_id, dtype=np.int64),
        "trk_start": np.asarray(trk_start, dtype=np.int64),
        "trk_len": np.asarray(trk_len, dtype=np.int64),
        "trk_run_off": np.asarray(run_off, dtype=np.int64),
        "run_len": np.concatenate(lens) if lens else np.empty(0, dtype=np.int32),
        "run_state": np.concatenate(states) if states else np.empty(0, dtype=np.int8),
        "run_xy": np.concatenate(xys) if xys else np.empty((0, 2)),
        "nf": np.asarray(nf, dtype=np.int64),
        "settings_json": np.asarray(json.dumps(settings, default=str, sort_keys=True)),
        "video_name": np.asarray(str(getattr(obj, "video_name", "") or "")),
    }
    summary = {
        "n_tracks": len(trk_id),
        "n_runs": int(arrays["run_len"].size),
        "n_state_length_mismatch": mismatch,
        "n_bad_start": bad_start,
        "nf": nf,
        "max_frame": int(max((s + n - 1 for s, n in zip(trk_start, trk_len, strict=True)), default=-1)),
    }
    return arrays, summary


EXTRACTORS: dict[TrackingKind, Callable[[TrackerObject], tuple[Arrays, Summary]]] = {
    "resting": extract_resting,
    "moving": extract_moving,
    "tracks": extract_tracks,
}


def ingest_file(kind: TrackingKind, source: str, target: str) -> Summary:
    """Worker task: read one tracking file and write its cache arrays. Never raises."""
    t0 = time.perf_counter()
    try:
        obj = load_tracking_pickle(source)
        arrays, summary = EXTRACTORS[kind](obj)
        atomic_save_npz(Path(target), arrays)
    except DataError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # a bug or an unforeseen file format: report, do not kill the run
        return {"status": "error", "error": f"unexpected {type(exc).__name__}: {exc}"}
    summary["status"] = "ok"
    summary["seconds"] = round(time.perf_counter() - t0, 3)
    return summary


# ------------------------------------------------------------------------- orchestration
@dataclass
class IngestReport:
    """What :func:`ingest` did."""

    cache_dir: Path
    built: int = 0
    reused: int = 0
    missing: dict[str, list[str]] = field(default_factory=dict)
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (chunk, tier, message)
    seconds: float = 0.0


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def default_workers() -> int:
    """Conservative default: tracking files usually sit on a spinning disk, where more workers only thrash."""
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def ingest(
    layout: ExperimentLayout,
    *,
    tiers: Iterable[TrackingKind] = ("resting", "moving"),
    cache_root: Path | None = None,
    workers: int | None = None,
    chunk_ids: set[str] | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> IngestReport:
    """Bring the experiment's cache up to date for the requested tiers and chunks.

    Chunks whose cache is current are reused. Unreadable files are reported, never fatal: the
    caller decides whether the remaining data is enough. Raises :class:`Cancelled` if
    ``cancel()`` becomes true; finished chunks stay in the cache.
    """
    t0 = time.perf_counter()
    cache_dir = cache_dir_for(layout.root, cache_root)
    index = CacheIndex.open(cache_dir, layout.root)
    report = IngestReport(cache_dir=cache_dir)
    todo: list[tuple[str, TrackingKind, Path, dict[str, Any]]] = []
    for chunk in layout.chunks:
        if chunk_ids is not None and chunk.chunk_id not in chunk_ids:
            continue
        for tier in tiers:
            src = chunk.files.get(tier)
            if src is None:
                report.missing.setdefault(tier, []).append(chunk.chunk_id)
                continue
            try:
                ident = source_identity(src)
            except OSError as exc:
                report.errors.append((chunk.chunk_id, tier, f"cannot read {src}: {exc}"))
                continue
            if index.is_current(chunk.chunk_id, tier, ident):
                report.reused += 1
                continue
            todo.append((chunk.chunk_id, tier, src, ident))

    def record(chunk_id: str, tier: TrackingKind, ident: dict[str, Any], summary: Summary) -> None:
        index.set_entry(chunk_id, tier, {"source": ident, "extractor": EXTRACTOR_VERSION, **summary})
        if summary.get("status") == "ok":
            report.built += 1
        else:
            report.errors.append((chunk_id, tier, str(summary.get("error"))))

    total = len(todo)
    n_workers = min(max(1, workers or default_workers()), 61, max(1, total))
    last_save = time.monotonic()
    try:
        if n_workers == 1 or total <= 1:
            for done, (chunk_id, tier, src, ident) in enumerate(todo, start=1):
                if cancel is not None and cancel():
                    raise Cancelled("Ingest cancelled; finished chunks are kept in the cache.")
                record(
                    chunk_id, tier, ident, ingest_file(tier, str(src), str(index.tier_path(tier, chunk_id)))
                )
                if progress:
                    progress(done / total, f"Read {tier} {chunk_id}")
        else:
            saved_env = {k: os.environ.get(k) for k in _THREAD_VARS}
            for k in _THREAD_VARS:
                os.environ[k] = "1"  # inherited by the workers: one BLAS thread each
            try:
                pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"))
            finally:
                for k, v in saved_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            with pool:
                pending: dict[Future[Summary], tuple[str, TrackingKind, dict[str, Any]]] = {}
                queue = list(reversed(todo))
                done = 0
                while queue or pending:
                    while queue and len(pending) < 2 * n_workers:
                        if cancel is not None and cancel():
                            queue.clear()
                            break
                        chunk_id, tier, src, ident = queue.pop()
                        fut = pool.submit(ingest_file, tier, str(src), str(index.tier_path(tier, chunk_id)))
                        pending[fut] = (chunk_id, tier, ident)
                    if not pending:
                        break
                    finished, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        chunk_id, tier, ident = pending.pop(fut)
                        record(chunk_id, tier, ident, fut.result())
                        done += 1
                        if progress:
                            progress(done / total, f"Read {tier} {chunk_id}")
                    if time.monotonic() - last_save > 5.0:
                        index.save()
                        last_save = time.monotonic()
                if cancel is not None and cancel():
                    raise Cancelled("Ingest cancelled; finished chunks are kept in the cache.")
    finally:
        index.save()
        report.seconds = round(time.perf_counter() - t0, 2)
    return report
