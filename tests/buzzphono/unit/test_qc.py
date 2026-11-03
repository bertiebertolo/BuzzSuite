from __future__ import annotations

from datetime import datetime

import numpy as np

from buzzphono import qc
from buzzphono.io.naming import ChunkName
from buzzphono.timebase import SessionFrames, to_ns


def _chunks(session_time: str = "120009", n: int = 4, date: str = "20260604", attempt=None, skip=()):
    return [ChunkName("Cage", date, session_time, attempt, k) for k in range(n) if k not in skip]


def _sessions(chunks, nf=None, full=30060):
    by: dict[str, dict[int, int]] = {}
    for c in chunks:
        by.setdefault(c.session, {})[c.index] = (nf or {}).get(c.index, full)
    return {
        s: SessionFrames(
            s,
            to_ns(
                chunks[0].session_start()
                if chunks[0].session == s
                else next(c for c in chunks if c.session == s).session_start()
            ),
            full,
            v,
            "tracker frame list",
        )
        for s, v in by.items()
    }


def test_clean_layout_has_no_findings() -> None:
    ch = _chunks()
    assert qc.layout_findings(ch, _sessions(ch), now=datetime(2026, 7, 1)) == []


def test_missing_and_short_chunks() -> None:
    ch = _chunks(n=8, skip=(3, 4))
    out = qc.layout_findings(ch, _sessions(ch, nf={5: 20000}), now=datetime(2026, 7, 1))
    msgs = " ".join(f.message for f in out)
    assert "missing chunks 3-4" in msgs and "chunks 5 are shorter" in msgs


def test_implausible_stamp_attempt_and_overlap() -> None:
    a = _chunks(date="19700101", n=2)
    out = qc.layout_findings(a, _sessions(a), now=datetime(2026, 7, 1))
    assert any("implausible" in f.message for f in out)
    b = _chunks(attempt=2, n=2)
    assert any(
        "'-attempt'" in f.message for f in qc.layout_findings(b, _sessions(b), now=datetime(2026, 7, 1))
    )
    first = _chunks(n=4)
    second = [ChunkName("Cage", "20260604", "122000", None, k) for k in range(2)]
    both = first + second
    sess = _sessions(first) | _sessions(second)
    assert any("overlap" in f.message for f in qc.layout_findings(both, sess, now=datetime(2026, 7, 1)))


def test_daylight_saving_crossing_is_flagged() -> None:
    ch = [ChunkName("Cage", "20261024", "120009", None, k) for k in range(80)]
    out = qc.layout_findings(ch, _sessions(ch), now=datetime(2026, 12, 1))
    assert any("daylight-saving change of 2026-10-25" in f.message for f in out)


def test_duplicate_sources() -> None:
    idx = {
        "a": {"resting": {"source": {"size": 10, "phash": "x"}}},
        "b": {"resting": {"source": {"size": 10, "phash": "x"}}},
        "c": {"resting": {"source": {"size": 11, "phash": "x"}}},
    }
    (f,) = qc.duplicate_sources(idx)
    assert f.severity == "error" and f.count == 1 and "b = a" in f.details[0]


def test_jump_and_spike_counts() -> None:
    moving = {
        "det_obj": np.array([0, 0, 0, 0, 1, 1]),
        "det_frame": np.array([0.0, 1, 2, 3, 0, 1]),
        "det_xy": np.array([[0, 0], [5, 0], [400, 0], [10, 0], [0, 0], [3, 0]], dtype=float),
    }
    steps, jumps, spikes = qc.jump_counts(moving, max_step_px=240)
    assert (steps, jumps, spikes) == (4, 2, 1)


def test_fragmentation_from_tracks() -> None:
    tracks = {
        "trk_id": np.array([0, 1]),
        "trk_start": np.array([0, 100]),
        "trk_len": np.array([60, 10]),
        "trk_run_off": np.array([0, 3, 4]),
        "run_len": np.array([10, 50, 0 + 0, 10])[[0, 1, 2, 3]].clip(min=0),
        "run_state": np.array([0, 1, 0, 1], dtype=np.int8),
        "nf": np.asarray(30000),
    }
    tracks["run_len"] = np.array([10, 50, 0, 10])
    f = qc.fragmentation(tracks, colony=30, fps=25.0)
    assert f.tracks == 2 and f.flight_bouts == 2 and f.median_bout_s == (50 + 10) / 2 / 25
    assert f.share_flight_in_long_bouts == 50 / 60


def test_warmup_ratio() -> None:
    n = np.r_[np.full(3000, 10.0), np.full(27000, 20.0)]
    assert qc.warmup_ratio(n) == 0.5
    assert np.isnan(qc.warmup_ratio(np.zeros(10)))


def test_static_objects_persist_across_chunks() -> None:
    chunks = [np.array([[100.0, 100.0], [300.0, 50.0]])] * 8 + [np.array([[100.5, 99.5]])] * 2
    found = qc.persistent_static_objects(chunks, tolerance_px=2.0, min_chunks=6)
    assert len(found) == 2
    resting = {"run_len": np.array([30000, 100]), "run_xy": np.array([[1.0, 2.0], [3.0, 4.0]])}
    assert qc.static_runs(resting, 30060).tolist() == [[1.0, 2.0]]


def test_close_pairs() -> None:
    frames = [np.array([[0.0, 0.0], [3.0, 0.0], [50.0, 50.0]]), np.array([[10.0, 10.0]])]
    assert qc.close_pair_rate(frames, max_distance_px=5) == 2 / 4


def test_light_findings_catch_an_hour_offset() -> None:
    times = {
        datetime(2026, 6, 4, h, m): (150.0 if 6 <= h < 18 else 60.0) for h in range(24) for m in (10, 30, 50)
    }
    out = qc.light_findings(times, "05:00", "17:00")
    assert len(out) == 2 and all("about 1 h" in f.message for f in out)
    assert (
        qc.light_findings({datetime(2026, 6, 4, h): 60.0 for h in range(24)}, "05:00", "17:00")[0].severity
        == "info"
    )
