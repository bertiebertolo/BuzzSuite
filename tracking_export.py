#!/usr/bin/env python
"""tracking_export.py — export frozen tracking .pkl trajectories to Parquet + CSV.

Read-only w.r.t. the tracker. Loads each ``final_tracking_data/forward_mosq_tracks_*``
pickle via the BuzzSwarm compat unpickler and emits, under ``<experiment_dir>/exports/``:

    detections/session_date=YYYYMMDD/segment=NNNNN/part-0.parquet   Output A  (ML substrate)
    tracks.parquet                                                  Output A' (nested sequences)
    track_summary.csv                                               Output B  (human overview)
    sample_segment.csv                                              one segment, for eyeballing
    manifest.json                                                   Output C  (self-describing)

Parquet is the primary bulk format (typed, compressed, columnar, Hive-partitioned so an ML
loader can read one date/segment without scanning everything). ``pyarrow`` is imported
lazily inside the exporter, so importing this module and the rest of the app stay
dependency-neutral. The CSV summary + JSON manifest are always written with the stdlib and
double as the human-readable layer; if ``pyarrow`` is somehow unavailable, the per-detection
dump falls back to gzipped CSV and the manifest records the degradation.

Source data model (already on disk — see CLAUDE.md):
    mt.time_stamp                 list of per-frame datetimes, indexed by ABSOLUTE frame index
    mt.objects[track_id] = {
        'coordinates': [(x, y), ...],   one per frame the track exists
        'state':       [1|0, ...],      1 = flying/moving, 0 = resting
        'start':       abs frame index where coordinates/state begin
    }
Align a track's k-th point to wall-clock time via mt.time_stamp[start + k].

Entry point (mirrors BuzzSwarm's fru2_sholl_csv.run):
    export_experiment_tracks(experiment_dir, out_dir=None, combined=True) -> out_dir
"""
import csv
import datetime as _dt
import gc
import glob
import json
import math
import os
import re
import shutil
import sys

GENERATOR_VERSION = "1.0.0"     # bump when the output layout / columns change
SCHEMA_VERSION = 1

_PREFIX = "forward_mosq_tracks_"
# {Cage}_{YYYYMMDD}_{HHMMSS}_{NNNNN}; cage may itself contain underscores, so anchor the
# trailing date/time/segment groups and let the cage soak up everything before them.
_NAME_RE = re.compile(r"^(?P<cage>.+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<seg>\d+)$")

_NAN = float("nan")


# --------------------------------------------------------------------------- helpers
def _default_log(msg):
    print(msg, flush=True)


def _parse_segment_name(name):
    """Parse a forward_mosq_tracks_* basename into its parts, or return None.

    Any file extension (e.g. a stray '.pkl' or '.converted.pkl') is stripped first.
    """
    base = os.path.basename(name)
    if base.startswith(_PREFIX):
        base = base[len(_PREFIX):]
    # strip a trailing extension chain (".pkl", ".converted.pkl", ...) if present
    while True:
        stem, ext = os.path.splitext(base)
        if ext and not ext[1:].isdigit():   # don't strip the _00034 segment index
            base = stem
            continue
        break
    m = _NAME_RE.match(base)
    if not m:
        return None
    return {
        "cage": m.group("cage"),
        "session_date": m.group("date"),
        "session_time": m.group("time"),
        "segment_str": m.group("seg"),
        "segment_int": int(m.group("seg")),
    }


def _f(v):
    """Coerce to a finite float, else NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _NAN
    return f if math.isfinite(f) else _NAN


def _xy(pt):
    if pt is None:
        return _NAN, _NAN
    try:
        return _f(pt[0]), _f(pt[1])
    except (TypeError, IndexError, ValueError):
        return _NAN, _NAN


def _experiment_alias(experiment_dir):
    """Best-effort alias from experiment_{alias}.json filename; None if absent."""
    for jp in sorted(glob.glob(os.path.join(experiment_dir, "experiment_*.json"))):
        m = re.match(r"experiment_(.+)\.json$", os.path.basename(jp))
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------- per-segment build
def _build_segment(mt, meta):
    """Walk one tracker object into flat detection columns, nested per-track rows, and
    per-track summary rows. Returns (det_cols, track_rows, summary_rows, stats).

    det_cols is a dict of parallel lists (one entry per detection); this is the ML-facing
    long table for Output A. track_rows/summary_rows are lists of dicts.
    """
    time_stamp = getattr(mt, "time_stamp", None) or []
    n_ts = len(time_stamp)
    objects = getattr(mt, "objects", None) or {}

    det = {
        "frame_index": [], "timestamp": [], "track_id": [],
        "x": [], "y": [], "state": [], "state_label": [],
        "dx": [], "dy": [], "step_px": [], "speed_px_per_frame": [], "is_flying": [],
    }
    track_rows = []
    summary_rows = []
    n_flying = 0
    n_resting = 0

    for fallback_id, (fly_id, obj) in enumerate(objects.items()):
        coords = obj.get("coordinates", []) or []
        state = obj.get("state", []) or []
        n = min(len(coords), len(state))
        if n <= 0:
            continue
        abs_start = int(obj.get("start", 0))
        try:
            tid = int(fly_id)
        except (TypeError, ValueError):
            tid = fallback_id

        x_seq = []
        y_seq = []
        state_seq = []
        t_seq = []
        prev_x = _NAN
        prev_y = _NAN
        path_len = 0.0
        t_start = None
        t_end = None
        tf = 0    # flying frames in this track
        tr = 0    # resting frames in this track

        for k in range(n):
            abs_idx = abs_start + k
            ts = time_stamp[abs_idx] if 0 <= abs_idx < n_ts else None
            x, y = _xy(coords[k])
            st = int(state[k]) if state[k] in (0, 1) else 0
            flying = (st == 1)

            if math.isfinite(prev_x) and math.isfinite(prev_y) and math.isfinite(x) and math.isfinite(y):
                dx = x - prev_x
                dy = y - prev_y
                step = math.hypot(dx, dy)
                path_len += step
            else:
                dx = dy = step = _NAN

            det["frame_index"].append(abs_idx)
            det["timestamp"].append(ts)
            det["track_id"].append(tid)
            det["x"].append(x)
            det["y"].append(y)
            det["state"].append(st)
            det["state_label"].append("flying" if flying else "resting")
            det["dx"].append(dx)
            det["dy"].append(dy)
            det["step_px"].append(step)
            det["speed_px_per_frame"].append(step)   # consecutive frames -> gap of 1
            det["is_flying"].append(flying)

            x_seq.append(x)
            y_seq.append(y)
            state_seq.append(st)
            t_seq.append(ts)
            if flying:
                tf += 1
            else:
                tr += 1
            if ts is not None:
                if t_start is None:
                    t_start = ts
                t_end = ts
            prev_x, prev_y = x, y

        n_flying += tf
        n_resting += tr

        finite_x = [v for v in x_seq if math.isfinite(v)]
        finite_y = [v for v in y_seq if math.isfinite(v)]
        duration_s = _NAN
        if t_start is not None and t_end is not None:
            try:
                duration_s = (t_end - t_start).total_seconds()
            except (TypeError, AttributeError):
                duration_s = _NAN
        mean_speed = path_len / duration_s if (duration_s and duration_s > 0) else _NAN

        track_rows.append({
            "segment": meta["segment_int"], "cage": meta["cage"],
            "session_date": meta["session_date"], "session_time": meta["session_time"],
            "track_id": tid, "start_frame": abs_start, "end_frame": abs_start + n - 1,
            "n_frames": n, "n_flying_frames": tf, "n_resting_frames": tr,
            "x_seq": x_seq, "y_seq": y_seq, "state_seq": state_seq, "t_seq": t_seq,
        })
        summary_rows.append({
            "track_id": tid, "segment": meta["segment_str"],
            "start_frame": abs_start, "end_frame": abs_start + n - 1, "n_frames": n,
            "n_flying_frames": tf, "n_resting_frames": tr,
            "duration_s": round(duration_s, 3) if math.isfinite(duration_s) else "",
            "path_length_px": round(path_len, 3),
            "mean_speed_px_s": round(mean_speed, 4) if math.isfinite(mean_speed) else "",
            "xmin": round(min(finite_x), 2) if finite_x else "",
            "xmax": round(max(finite_x), 2) if finite_x else "",
            "ymin": round(min(finite_y), 2) if finite_y else "",
            "ymax": round(max(finite_y), 2) if finite_y else "",
        })

    stats = {
        "n_tracks": len(track_rows),
        "n_detections": len(det["frame_index"]),
        "n_flying": n_flying,
        "n_resting": n_resting,
    }
    return det, track_rows, summary_rows, stats


def _estimate_fps(mt):
    """Median frames-per-second from consecutive timestamps, or None."""
    ts = getattr(mt, "time_stamp", None) or []
    diffs = []
    for i in range(1, min(len(ts), 2000)):
        a, b = ts[i - 1], ts[i]
        if a is None or b is None:
            continue
        try:
            d = (b - a).total_seconds()
        except (TypeError, AttributeError):
            continue
        if d > 0:
            diffs.append(d)
    if not diffs:
        return None
    diffs.sort()
    med = diffs[len(diffs) // 2]
    return round(1.0 / med, 4) if med > 0 else None


# --------------------------------------------------------------------------- pyarrow layer
def _pyarrow():
    """Lazy import; returns (pa, pq) or (None, None) if unavailable."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except ImportError:
        return None, None


def _det_schema(pa):
    return pa.schema([
        ("cage", pa.string()),
        ("session_time", pa.string()),
        ("frame_index", pa.int64()),
        ("timestamp", pa.timestamp("us")),
        ("track_id", pa.int64()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("state", pa.int8()),
        ("state_label", pa.string()),
        ("dx", pa.float32()),
        ("dy", pa.float32()),
        ("step_px", pa.float32()),
        ("speed_px_per_frame", pa.float32()),
        ("is_flying", pa.bool_()),
    ])


def _tracks_schema(pa):
    return pa.schema([
        ("segment", pa.int32()),
        ("cage", pa.string()),
        ("session_date", pa.string()),
        ("session_time", pa.string()),
        ("track_id", pa.int64()),
        ("start_frame", pa.int64()),
        ("end_frame", pa.int64()),
        ("n_frames", pa.int64()),
        ("n_flying_frames", pa.int64()),
        ("n_resting_frames", pa.int64()),
        ("x_seq", pa.list_(pa.float32())),
        ("y_seq", pa.list_(pa.float32())),
        ("state_seq", pa.list_(pa.int8())),
        ("t_seq", pa.list_(pa.timestamp("us"))),
    ])


def _write_det_parquet(pa, pq, det, meta, det_root, compression):
    """Write one segment's detections into its Hive partition. session_date/segment live in
    the directory path (canonical Hive layout), everything else lives in the file."""
    n = len(det["frame_index"])
    table = pa.table({
        "cage": [meta["cage"]] * n,
        "session_time": [meta["session_time"]] * n,
        "frame_index": det["frame_index"],
        "timestamp": pa.array(det["timestamp"], type=pa.timestamp("us")),
        "track_id": det["track_id"],
        "x": pa.array(det["x"], type=pa.float32()),
        "y": pa.array(det["y"], type=pa.float32()),
        "state": pa.array(det["state"], type=pa.int8()),
        "state_label": det["state_label"],
        "dx": pa.array(det["dx"], type=pa.float32()),
        "dy": pa.array(det["dy"], type=pa.float32()),
        "step_px": pa.array(det["step_px"], type=pa.float32()),
        "speed_px_per_frame": pa.array(det["speed_px_per_frame"], type=pa.float32()),
        "is_flying": det["is_flying"],
    }, schema=_det_schema(pa))
    part_dir = os.path.join(det_root, "session_date=%s" % meta["session_date"],
                            "segment=%s" % meta["segment_str"])
    os.makedirs(part_dir, exist_ok=True)
    pq.write_table(table, os.path.join(part_dir, "part-0.parquet"), compression=compression)


def _tracks_table(pa, rows):
    schema = _tracks_schema(pa)
    return pa.table({
        "segment": [r["segment"] for r in rows],
        "cage": [r["cage"] for r in rows],
        "session_date": [r["session_date"] for r in rows],
        "session_time": [r["session_time"] for r in rows],
        "track_id": [r["track_id"] for r in rows],
        "start_frame": [r["start_frame"] for r in rows],
        "end_frame": [r["end_frame"] for r in rows],
        "n_frames": [r["n_frames"] for r in rows],
        "n_flying_frames": [r["n_flying_frames"] for r in rows],
        "n_resting_frames": [r["n_resting_frames"] for r in rows],
        "x_seq": pa.array([r["x_seq"] for r in rows], type=pa.list_(pa.float32())),
        "y_seq": pa.array([r["y_seq"] for r in rows], type=pa.list_(pa.float32())),
        "state_seq": pa.array([r["state_seq"] for r in rows], type=pa.list_(pa.int8())),
        "t_seq": pa.array([r["t_seq"] for r in rows], type=pa.list_(pa.timestamp("us"))),
    }, schema=schema)


# --------------------------------------------------------------------------- CSV fallback / companions
_SUMMARY_COLS = ["track_id", "segment", "start_frame", "end_frame", "n_frames",
                 "n_flying_frames", "n_resting_frames", "duration_s", "path_length_px",
                 "mean_speed_px_s", "xmin", "xmax", "ymin", "ymax"]

_SUMMARY_COL_DESC = {
    "track_id": "int; trajectory id within the segment",
    "segment": "int; 20-min segment index",
    "start_frame": "int; absolute frame index where this track begins",
    "end_frame": "int; absolute frame index where this track ends",
    "n_frames": "int; total frames in this track",
    "n_flying_frames": "int; frames with state == 1 (flying/moving)",
    "n_resting_frames": "int; frames with state == 0 (resting)",
    "duration_s": "float seconds; n_frames / fps",
    "path_length_px": "float px; sum of per-frame step distances",
    "mean_speed_px_s": "float px/s; path_length_px / duration_s",
    "xmin": "float px; min x over the track",
    "xmax": "float px; max x over the track",
    "ymin": "float px; min y over the track",
    "ymax": "float px; max y over the track",
}

_DET_CSV_COLS = ["frame_index", "timestamp", "track_id", "x", "y", "state", "state_label",
                 "dx", "dy", "step_px", "speed_px_per_frame", "is_flying"]


def _det_csv_rows(det):
    n = len(det["frame_index"])
    for i in range(n):
        ts = det["timestamp"][i]
        yield {
            "frame_index": det["frame_index"][i],
            "timestamp": ts.isoformat() if isinstance(ts, _dt.datetime) else "",
            "track_id": det["track_id"][i],
            "x": det["x"][i], "y": det["y"][i],
            "state": det["state"][i], "state_label": det["state_label"][i],
            "dx": det["dx"][i], "dy": det["dy"][i],
            "step_px": det["step_px"][i],
            "speed_px_per_frame": det["speed_px_per_frame"][i],
            "is_flying": det["is_flying"][i],
        }


# --------------------------------------------------------------------------- main entry
def export_experiment_tracks(experiment_dir, out_dir=None, combined=True,
                             compression="snappy", log=None,
                             progress_cb=None, cancel_event=None):
    """Export one experiment's frozen tracking pickles to Parquet + CSV.

    Parameters
    ----------
    experiment_dir : str
        Folder containing ``final_tracking_data/`` (an ``experiment_*.json`` alongside it is
        used for the manifest alias if present).
    out_dir : str or None
        Output directory; defaults to ``<experiment_dir>/exports/``.
    combined : bool
        When True, also emit the nested per-track ``tracks.parquet`` and ``sample_segment.csv``.
        The partitioned ``detections/`` dataset, ``track_summary.csv`` and ``manifest.json`` are
        always written.
    compression : str
        Parquet codec (``"snappy"`` default, ``"zstd"`` for smaller files).
    log : callable or None
        Line logger (defaults to print). progress_cb(done, total) and a threading.Event
        ``cancel_event`` are optional GUI hooks.

    Returns the output directory path.
    """
    log = log or _default_log
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    tdir = os.path.join(experiment_dir, "final_tracking_data")
    if not os.path.isdir(tdir):
        raise ValueError("No final_tracking_data/ in %s — run tracking first." % experiment_dir)

    if out_dir is None:
        out_dir = os.path.join(experiment_dir, "exports")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load the compat unpickler (shares the engine's numpy._core shim). Lazy: keeps app
    # cold-start dependency-neutral.
    try:
        from buzzswarm import fru2_zt_normalized_analysis as engine
    except ImportError:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        from buzzswarm import fru2_zt_normalized_analysis as engine
    load_tracking_file = engine.load_tracking_file

    pa, pq = _pyarrow()
    parquet_ok = pa is not None
    if not parquet_ok:
        log("  ! pyarrow not available — falling back to gzipped CSV per-detection dump.")

    # Enumerate source segments (sorted for deterministic, reproducible output).
    names = sorted(f for f in os.listdir(tdir) if f.startswith(_PREFIX) and not f.startswith("."))
    if not names:
        raise ValueError("No forward_mosq_tracks_* files in %s." % tdir)

    # Clean slate for the bulk datasets so a re-run can't leave stale partitions behind.
    det_root = os.path.join(out_dir, "detections")
    det_csv_root = os.path.join(out_dir, "detections_csv")
    tracks_path = os.path.join(out_dir, "tracks.parquet")
    for stale in (det_root, det_csv_root, tracks_path):
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)
        elif os.path.isfile(stale):
            os.remove(stale)

    tracks_writer = None
    summary_rows_all = []
    source_meta = []
    total = len(names)
    stats_total = {"segments": 0, "tracks": 0, "detections": 0, "flying": 0, "resting": 0,
                   "skipped": 0}
    cage_seen = None
    fps = None
    sample_written = False
    import gzip

    try:
        for idx, name in enumerate(names, start=1):
            if cancel_event is not None and cancel_event.is_set():
                log("  cancelled after %d/%d segment(s)." % (idx - 1, total))
                break

            meta = _parse_segment_name(name)
            if meta is None:
                log("  ! skip (unparseable name): %s" % name)
                stats_total["skipped"] += 1
                continue
            fpath = os.path.join(tdir, name)
            mt = load_tracking_file(fpath)
            if mt is None:
                log("  ! skip (load failed): %s" % name)
                stats_total["skipped"] += 1
                continue

            if cage_seen is None:
                cage_seen = meta["cage"]
            if fps is None:
                fps = _estimate_fps(mt)

            det, track_rows, summary_rows, stats = _build_segment(mt, meta)
            source_meta.append({"name": name, "mtime": os.path.getmtime(fpath),
                                "n_tracks": stats["n_tracks"], "n_detections": stats["n_detections"]})

            if stats["n_detections"] > 0:
                if parquet_ok:
                    _write_det_parquet(pa, pq, det, meta, det_root, compression)
                else:
                    part_dir = os.path.join(det_csv_root, "session_date=%s" % meta["session_date"],
                                            "segment=%s" % meta["segment_str"])
                    os.makedirs(part_dir, exist_ok=True)
                    with gzip.open(os.path.join(part_dir, "part-0.csv.gz"), "wt", newline="") as gz:
                        w = csv.DictWriter(gz, fieldnames=_DET_CSV_COLS)
                        w.writeheader()
                        for row in _det_csv_rows(det):
                            w.writerow(row)

                if combined and not sample_written:
                    sample_path = os.path.join(out_dir, "sample_segment.csv")
                    with open(sample_path, "w", newline="") as sf:
                        w = csv.DictWriter(sf, fieldnames=_DET_CSV_COLS)
                        w.writeheader()
                        for row in _det_csv_rows(det):
                            w.writerow(row)
                    sample_written = True

            if combined and parquet_ok and track_rows:
                if tracks_writer is None:
                    tracks_writer = pq.ParquetWriter(tracks_path, _tracks_schema(pa),
                                                     compression=compression)
                tracks_writer.write_table(_tracks_table(pa, track_rows))

            summary_rows_all.extend(summary_rows)
            stats_total["segments"] += 1
            stats_total["tracks"] += stats["n_tracks"]
            stats_total["detections"] += stats["n_detections"]
            stats_total["flying"] += stats["n_flying"]
            stats_total["resting"] += stats["n_resting"]

            if progress_cb is not None:
                try:
                    progress_cb(idx, total)
                except Exception:
                    pass
            if idx % 20 == 0 or idx == total:
                log("  [%d/%d] %s tracks, %s detections so far"
                    % (idx, total, stats_total["tracks"], stats_total["detections"]))

            del mt, det, track_rows, summary_rows
            gc.collect()
    finally:
        if tracks_writer is not None:
            tracks_writer.close()

    # Human-facing per-track summary (Output B) — always CSV, stdlib.
    summary_path = os.path.join(out_dir, "track_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        w = csv.DictWriter(sf, fieldnames=_SUMMARY_COLS)
        w.writeheader()
        for row in summary_rows_all:
            w.writerow(row)

    # Self-describing manifest (Output C).
    manifest = {
        "generator": "tracking_export",
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "experiment_dir": experiment_dir,
        "experiment_alias": _experiment_alias(experiment_dir),
        "cage": cage_seen,
        "primary_format": "parquet" if parquet_ok else "csv-gzip-fallback",
        "compression": compression if parquet_ok else "gzip",
        "pyarrow_version": (pa.__version__ if parquet_ok else None),
        "fps": fps,
        "partitioning": ("detections/session_date=YYYYMMDD/segment=NNNNN/part-0.parquet"
                         if parquet_ok else
                         "detections_csv/session_date=YYYYMMDD/segment=NNNNN/part-0.csv.gz"),
        "state_meaning": {"1": "flying/moving", "0": "resting"},
        "counts": stats_total,
        "outputs": {
            "detections": os.path.relpath(det_root if parquet_ok else det_csv_root, out_dir),
            "tracks": (os.path.basename(tracks_path)
                       if (combined and parquet_ok and os.path.isfile(tracks_path)) else None),
            "track_summary": os.path.basename(summary_path),
            "sample_segment": "sample_segment.csv" if sample_written else None,
        },
        "columns": {
            "detections": {
                "cage": "str; recording cage name",
                "session_date": "str YYYYMMDD (Hive partition key)",
                "session_time": "str HHMMSS; recording session start",
                "segment": "int; 20-min segment index (Hive partition key)",
                "frame_index": "int; ABSOLUTE frame index into mt.time_stamp",
                "timestamp": "wall-clock datetime (us) of this detection",
                "track_id": "int; trajectory id within the segment",
                "x": "float px; raw pixel x (no cage-centre offset)",
                "y": "float px; raw pixel y",
                "state": "int8; 1=flying/moving, 0=resting",
                "state_label": "str; 'flying' or 'resting'",
                "dx": "float px; x - prev x within the same track (NaN at track start)",
                "dy": "float px; y - prev y within the same track",
                "step_px": "float px; hypot(dx, dy)",
                "speed_px_per_frame": "float px/frame; = step_px (consecutive frames)",
                "is_flying": "bool; state == 1",
            },
            "tracks": {
                "note": "one row per trajectory; *_seq are per-frame Arrow list columns",
                "x_seq": "list<float32> px", "y_seq": "list<float32> px",
                "state_seq": "list<int8> 1=flying/0=resting",
                "t_seq": "list<timestamp[us]> wall-clock",
            },
            "track_summary": dict(_SUMMARY_COL_DESC),
        },
        "source_pkls": source_meta,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    log("Export complete: %d segment(s), %d track(s), %d detection(s) -> %s"
        % (stats_total["segments"], stats_total["tracks"], stats_total["detections"], out_dir))
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python tracking_export.py <experiment_dir> [out_dir]")
    _out = sys.argv[2] if len(sys.argv) > 2 else None
    export_experiment_tracks(sys.argv[1], out_dir=_out)
