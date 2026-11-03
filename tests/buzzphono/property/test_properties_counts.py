"""Invariants of the counting core, checked on many generated inputs (Hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from buzzphono.io.ingest import extract_resting
from buzzphono.io.safe_pickle import TrackerObject
from buzzphono.metrics.resting_fraction import FrameCounts, RestingGrids, frame_counts, polygon_test
from buzzphono.stats import wilcoxon_scipy110
from buzzphono.timebase import NS_PER_S

ZONE = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float)

detections = st.lists(
    st.tuples(st.integers(0, 3), st.integers(0, 40), st.integers(0, 10), st.integers(0, 10)), max_size=120
)


def _tracker(rows) -> TrackerObject:
    objects: dict[int, dict[str, list]] = {}
    for obj, frame, x, y in rows:
        rec = objects.setdefault(obj, {"time_stamp": [], "coordinates": []})
        rec["time_stamp"].append(frame)
        rec["coordinates"].append((float(x), float(y)))
    t = TrackerObject()
    t.objects = objects  # type: ignore[assignment]
    return t


@given(detections)
def test_run_length_encoding_loses_nothing(rows) -> None:
    arrays, summary = extract_resting(_tracker(rows))
    assert int(arrays["run_len"].sum()) == len(rows) == summary["n_detections"]
    fc = frame_counts(arrays, polygon_test(ZONE))
    expected_total: dict[int, int] = {}
    expected_in: dict[int, int] = {}
    for _, frame, x, y in rows:
        expected_total[frame] = expected_total.get(frame, 0) + 1
        inside = bool(polygon_test(ZONE)(np.array([[x, y]], dtype=float))[0])
        expected_in[frame] = expected_in.get(frame, 0) + int(inside)
    assert dict(zip(fc.frames.astype(int).tolist(), fc.n_rest.tolist(), strict=True)) == expected_total
    assert dict(zip(fc.frames.astype(int).tolist(), fc.n_in.tolist(), strict=True)) == expected_in


@given(
    st.lists(st.tuples(st.integers(0, 7199), st.integers(0, 80), st.integers(0, 80)), max_size=200),
    st.integers(1, 40),
)
def test_grid_invariants(frames, colony) -> None:
    frames = [(f, n, min(i, n)) for f, n, i in frames]
    g = RestingGrids.empty(0, 2 * 3600 * NS_PER_S, colony=colony)
    if frames:
        f = np.array(frames, dtype=np.int64)
        counts = FrameCounts(f[:, 0].astype(float), f[:, 1], f[:, 2])
        g.add(counts, f[:, 0] * NS_PER_S, 0, 2 * 3600 * NS_PER_S)
    assert (g.in_zone <= g.total).all()
    assert (g.total <= colony * g.n_frames).all()
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(g.total > 0, g.in_zone / g.total, 0.0)
    assert ((frac >= 0) & (frac <= 1)).all()
    assert (g.frac_sum <= g.n_frames + 1e-9).all()


@given(
    st.lists(
        st.floats(-10, 10, allow_nan=False).filter(lambda v: v != 0), min_size=1, max_size=80, unique_by=abs
    )
)
def test_one_sided_tests_mirror_each_other(diffs) -> None:
    """Without ties in |x - y|, testing x > y is the same as testing y < x. (With ties, SciPy 1.10's
    exact mode truncated the rank sum, which breaks this symmetry; that quirk is reproduced on
    purpose for the published numbers and is tested in test_wilcoxon_legacy.py.)"""
    d = np.array(diffs)
    x, y = d, np.zeros_like(d)
    greater = wilcoxon_scipy110(x, y, "greater").p_value
    less = wilcoxon_scipy110(y, x, "less").p_value
    assert 0.0 <= greater <= 1.0
    assert abs(greater - less) < 1e-12
