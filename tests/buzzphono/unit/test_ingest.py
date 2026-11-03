from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from buzzphono.errors import Cancelled
from buzzphono.io.cache import load_npz
from buzzphono.io.discover import discover
from buzzphono.io.ingest import extract_moving, extract_resting, extract_tracks, ingest
from buzzphono.io.naming import ChunkName
from buzzphono.io.safe_pickle import TrackerObject
from buzzphono.synthetic.writers import write_tracking_file


def _tracker(objects: dict, **extra) -> TrackerObject:
    obj = TrackerObject()
    obj.objects = objects  # type: ignore[assignment]
    obj.__dict__.update(extra)
    return obj


def _expand(arrays: dict) -> list[tuple[int, int, float, float]]:
    """Back from runs to sorted (object index, frame, x, y) detections."""
    out = []
    for o, f0, n, (x, y) in zip(
        arrays["run_obj"], arrays["run_f0"], arrays["run_len"], arrays["run_xy"], strict=True
    ):
        out += [(int(o), int(f0 + k), float(x), float(y)) for k in range(int(n))]
    return sorted(out)


def test_resting_runs_round_trip_and_break_rules() -> None:
    objects = {
        7: {
            "time_stamp": [0, 1, 2, 4, 5, 5],
            "coordinates": [(1, 1), (1, 1), (2, 1), (2, 1), (2, 1), (2, 1)],
        },
        9: {"time_stamp": [3, 4], "coordinates": [(5, 5), (5, 5)]},
    }
    arrays, summary = extract_resting(_tracker(objects))
    # runs: [0,1]@(1,1); [2]@(2,1) (moved); [4,5]@(2,1) (gap); [5]@(2,1) (duplicate frame); [3,4]@(5,5)
    assert arrays["run_len"].tolist() == [2, 1, 2, 1, 2]
    assert _expand(arrays) == sorted(
        [
            (0, 0, 1, 1),
            (0, 1, 1, 1),
            (0, 2, 2, 1),
            (0, 4, 2, 1),
            (0, 5, 2, 1),
            (0, 5, 2, 1),
            (1, 3, 5, 5),
            (1, 4, 5, 5),
        ]
    )
    assert arrays["obj_ids"].tolist() == [7, 9]
    assert summary["n_detections"] == 8 and summary["n_runs"] == 5


def test_resting_drops_and_counts_what_the_published_code_dropped() -> None:
    objects = {
        0: {"time_stamp": [0, 1, 2, 3], "coordinates": [(1, 1), (np.nan, 1), (1, 1)]},  # length mismatch
        1: {"time_stamp": [float("nan"), 6.5, 7], "coordinates": [(3, 3), (3, 3), (3, 3)]},
        2: {"time_stamp": [], "coordinates": []},
    }
    arrays, summary = extract_resting(_tracker(objects))
    assert summary["n_length_mismatch"] == 1
    assert summary["n_nonfinite_dropped"] == 2
    assert summary["n_noninteger_frames"] == 1
    assert arrays["exc_frame"].tolist() == [6.5]
    assert _expand(arrays) == [(0, 0, 1, 1), (0, 2, 1, 1), (1, 7, 3, 3)]


def test_moving_detections_kept_in_order() -> None:
    objects = {3: {"time_stamp": [10, 11], "coordinates": [(1, 2), (3, 4)]}}
    arrays, summary = extract_moving(_tracker(objects))
    assert arrays["det_frame"].tolist() == [10.0, 11.0]
    assert arrays["det_xy"].tolist() == [[1, 2], [3, 4]]
    assert summary["n_detections"] == 2


def test_tracks_runs_with_gaps_and_states() -> None:
    nan = float("nan")
    objects = {
        np.int32(4): {
            "start": 100,
            "state": [0, 0, 1, nan, nan, 1],
            "coordinates": [(1, 1), (1, 1), (2, 2), (nan, nan), (nan, nan), (3, 3)],
        },
        np.int32(5): {"start": 7, "state": [1.0], "coordinates": [(9, 9)]},
    }
    arrays, summary = extract_tracks(
        _tracker(objects, time_stamp=[0] * 30060, settings={"fps": 25.0}, video_name="v")
    )
    assert arrays["trk_id"].tolist() == [4, 5]
    assert arrays["trk_start"].tolist() == [100, 7]
    assert arrays["trk_len"].tolist() == [6, 1]
    assert arrays["run_len"].tolist() == [2, 1, 2, 1, 1]
    assert arrays["run_state"].tolist() == [0, 1, -1, 1, 1]
    assert int(arrays["nf"]) == 30060 and summary["max_frame"] == 105
    assert json.loads(str(arrays["settings_json"])) == {"fps": 25.0}


def _experiment(tmp_path, n=3):
    for i in range(n):
        c = ChunkName("Cage", "20260604", "120009", None, i)
        objs = {j: {"time_stamp": list(range(10)), "coordinates": [(j, j)] * 10} for j in range(4)}
        write_tracking_file(tmp_path, "resting", c, objs)
        write_tracking_file(
            tmp_path, "moving", c, {0: {"time_stamp": [1, 2], "coordinates": [(1, 1), (2, 2)]}}
        )
    return discover(tmp_path)


@pytest.mark.parametrize("workers", [1, 2])
def test_ingest_builds_then_reuses(tmp_path, workers) -> None:
    exp = tmp_path / "exp"
    layout = _experiment(exp)
    cache = tmp_path / "cache"
    report = ingest(layout, cache_root=cache, workers=workers)
    assert report.built == 6 and report.reused == 0 and not report.errors
    arrays = load_npz(report.cache_dir / "resting" / "Cage_20260604_120009_00001.npz")
    assert arrays["run_len"].tolist() == [10, 10, 10, 10]
    again = ingest(layout, cache_root=cache, workers=workers)
    assert again.built == 0 and again.reused == 6


def test_ingest_rebuilds_a_changed_source_and_reports_corrupt_files(tmp_path) -> None:
    exp = tmp_path / "exp"
    layout = _experiment(exp)
    cache = tmp_path / "cache"
    ingest(layout, cache_root=cache, workers=1)
    bad = exp / "tracking_resting" / "resting_Cage_20260604_120009_00002.pkl"
    bad.write_bytes(b"\x80\x04 corrupt")
    st = bad.stat()
    os.utime(bad, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    report = ingest(discover(exp), cache_root=cache, workers=1)
    assert report.built == 0 and report.reused == 5
    assert [(c, t) for c, t, _ in report.errors] == [("Cage_20260604_120009_00002", "resting")]


def test_ingest_can_be_cancelled_and_keeps_finished_chunks(tmp_path) -> None:
    layout = _experiment(tmp_path / "exp", n=4)
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(Cancelled):
        ingest(layout, cache_root=tmp_path / "cache", workers=1, cancel=cancel)
    resumed = ingest(layout, cache_root=tmp_path / "cache", workers=1)
    assert resumed.reused == 2 and resumed.built == 6


def test_ingest_never_writes_into_the_experiment(tmp_path) -> None:
    exp = tmp_path / "exp"
    layout = _experiment(exp)
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in exp.rglob("*")}
    time.sleep(0.01)
    ingest(layout, cache_root=tmp_path / "cache", workers=2)
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in exp.rglob("*")}
    assert before == after
