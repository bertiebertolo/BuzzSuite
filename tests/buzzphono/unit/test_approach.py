"""Approach-bout detection and resting jitter on hand-made signals."""

from __future__ import annotations

import numpy as np

from buzzphono.metrics.approach import ApproachSettings, _bouts, resting_jitter_px, turning_points


def test_turning_points_with_hysteresis() -> None:
    d = np.array([50, 52, 51, 40, 30, 31, 29, 45, 60, 58, 20, 21], dtype=float)
    assert turning_points(d, 5.0) == [(1, 6), (8, 10)]
    assert turning_points(d, 50.0) == []
    assert turning_points(np.array([3.0]), 1.0) == []


def test_jitter_floor_and_estimate() -> None:
    still = {
        "run_obj": np.array([1, 1, 1, 1]),
        "run_f0": np.array([0, 5, 9, 12]),
        "run_len": np.array([5, 4, 3, 2]),
        "run_xy": np.array([[0, 0], [1, 0], [1, 1], [10, 1]], dtype=float),
    }
    assert resting_jitter_px(still) >= 2.0
    assert resting_jitter_px({k: v[:2] for k, v in still.items()}) == 2.0


def test_bouts_filter_by_drop_and_duration() -> None:
    t0 = 5 * 3600 * 10**9 + 41 * 60 * 10**9  # minute 41 of an hour: in the ON window
    t = t0 + np.arange(12) * 40_000_000  # 25 fps
    d = np.array([100, 100, 90, 70, 50, 30, 10, 0, 0, 20, 40, 60], dtype=float)
    windows = {"on": (40, 50)}
    starts = np.array([5 * 3600 * 10**9])
    rows = _bouts(d, t, "c", windows, starts, ["on"], 30.0, ApproachSettings(min_duration_s=0.1), 5.0, "px")
    (b,) = rows
    assert b["drop_px"] == 100 and b["entered_zone"] and b["window"] == "on"
    assert b["speed_px_per_s"] == 100 / b["duration_s"]
    assert (
        _bouts(d, t, "c", windows, starts, ["on"], 30.0, ApproachSettings(min_duration_s=5.0), 5.0, "px")
        == []
    )
    assert (
        _bouts(
            d, t, "c", windows, np.array([0]), ["on"], 30.0, ApproachSettings(min_duration_s=0.1), 5.0, "px"
        )
        == []
    )


def test_flyer_mask_matches_the_per_track_rule() -> None:
    from buzzphono.metrics.approach import flyer_mask

    rng = np.random.default_rng(5)
    owners, pts = [], []
    for k in range(40):
        n = int(rng.integers(1, 25))
        speed = rng.choice([0.5, 10.0])
        owners.append(np.full(n, k))
        pts.append(np.cumsum(rng.normal(0, speed, (n, 2)), axis=0))
    own, xy = np.concatenate(owners), np.concatenate(pts)
    mask = flyer_mask(own, xy, scale=1.0, min_points=10)
    for k in range(40):
        p = xy[own == k]
        expect = len(p) >= 10 and np.hypot(*np.diff(p, axis=0).T).mean() >= 4.0
        assert mask[own == k].all() == expect and mask[own == k].any() == expect
    assert flyer_mask(np.empty(0, dtype=np.int32), np.empty((0, 2)), 1.0, 10).size == 0
