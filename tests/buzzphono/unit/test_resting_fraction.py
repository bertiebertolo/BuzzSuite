from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from buzzphono import thesis
from buzzphono.metrics.resting_fraction import FrameCounts, RestingGrids, frame_counts, polygon_test
from buzzphono.timebase import NS_PER_MIN, NS_PER_S, to_ns

ZONE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
T0 = to_ns(datetime(2026, 6, 4, 12, 0, 0))


def _runs(rows: list[tuple[int, int, float, float]]) -> dict[str, np.ndarray]:
    """rows: (first frame, length, x, y)."""
    a = np.array(rows, dtype=float).reshape(-1, 4)
    return {
        "run_f0": a[:, 0].astype(np.int64),
        "run_len": a[:, 1].astype(np.int32),
        "run_xy": a[:, 2:4],
        "exc_frame": np.empty(0),
        "exc_xy": np.empty((0, 2)),
    }


def test_frame_counts_from_runs() -> None:
    arrays = _runs([(0, 3, 5, 5), (1, 2, 50, 50), (2, 1, 5, 5), (6, 1, 50, 50)])
    fc = frame_counts(arrays, polygon_test(ZONE))
    assert fc.frames.tolist() == [0.0, 1.0, 2.0, 6.0]
    assert fc.n_rest.tolist() == [1, 2, 3, 1]
    assert fc.n_in.tolist() == [1, 1, 2, 0]


def test_frame_counts_with_noninteger_frames_and_no_runs() -> None:
    arrays = _runs([])
    arrays["exc_frame"] = np.array([3.5, 3.5, 4.25])
    arrays["exc_xy"] = np.array([[1, 1], [20, 20], [1, 1]], dtype=float)
    fc = frame_counts(arrays, polygon_test(ZONE))
    assert fc.frames.tolist() == [3.5, 4.25] and fc.n_rest.tolist() == [2, 1] and fc.n_in.tolist() == [1, 1]
    empty = frame_counts(_runs([]), polygon_test(ZONE))
    assert empty.frames.size == 0


def test_grid_cells_cap_and_window() -> None:
    start, end = T0 + 30 * NS_PER_MIN, T0 + 3 * 3600 * NS_PER_S
    grids = RestingGrids.empty(start, end, colony=30)
    assert grids.t0_ns == T0 and grids.n_cycles == 4
    times = np.array(
        [
            T0 + 10 * NS_PER_MIN,  # before the window: dropped
            T0 + 40 * NS_PER_MIN,  # cycle 0, minute 40
            T0 + 40 * NS_PER_MIN + 59 * NS_PER_S,  # same minute
            T0 + 61 * NS_PER_MIN,  # cycle 1, minute 1
            end,  # end is exclusive
        ]
    )
    counts = FrameCounts(np.arange(5, dtype=float), np.array([5, 40, 10, 3, 1]), np.array([5, 35, 2, 3, 1]))
    assert grids.add(counts, times, start, end) == 3
    assert grids.total[0, 40] == 30 + 10  # 40 capped at 30
    assert grids.in_zone[0, 40] == 30 + 2  # 35 capped at the capped total (30)
    assert grids.raw_total[0, 40] == 50 and grids.raw_in[0, 40] == 37
    assert grids.n_frames[0, 40] == 2 and grids.frac_sum[0, 40] == pytest.approx(1.0 + 0.2)
    assert grids.total[1, 1] == 3 and grids.in_zone[1, 1] == 3
    assert grids.abs_total[40] == 40 and grids.abs_total[61] == 3


def _toy_grids() -> RestingGrids:
    g = RestingGrids.empty(T0, T0 + 3 * 3600 * NS_PER_S, colony=30)
    rng = np.random.default_rng(1)
    g.total[:3] = rng.integers(0, 3000, size=(3, 60)).astype(float)
    g.in_zone[:3] = np.floor(g.total[:3] * rng.uniform(0, 0.3, size=(3, 60)))
    g.total[2, 28:38] = 0  # cycle 2 has no OFF data
    g.in_zone[2, 28:38] = 0
    g.abs_total[:180] = g.total[:3].ravel()
    g.abs_in[:180] = g.in_zone[:3].ravel()
    return g


def test_paired_table_matches_hand_computation() -> None:
    g = _toy_grids()
    t = thesis.paired_by_cycle(g)
    assert t["cycle"].tolist() == [0, 1]
    for c in (0, 1):
        off = g.in_zone[c, 28:38].sum() / g.total[c, 28:38].sum()
        on = g.in_zone[c, 40:50].sum() / g.total[c, 40:50].sum()
        assert t.loc[t.cycle == c, "prop_off"].item() == off
        assert t.loc[t.cycle == c, "prop_on"].item() == on


def test_profile_recruitment_and_minute_series() -> None:
    g = _toy_grids()
    prof = thesis.minute_of_hour_profile(g)
    assert prof["n_cycles"].iloc[0] == 3 and prof["n_cycles"].iloc[30] == 2
    p = g.in_zone[:3, 5] / g.total[:3, 5]
    assert prof["mean"].iloc[5] == pytest.approx(p.mean(), rel=1e-15)
    rec = thesis.recruitment(g)
    base = g.in_zone[:2, 28:38].sum(1) / g.total[:2, 28:38].sum(1)
    assert rec["recruit_mean"].iloc[45] == pytest.approx(np.mean(g.in_zone[:2, 45] / g.total[:2, 45] - base))
    ms = thesis.minute_series(g)
    assert ms["time"].iloc[0] == "2026-06-04 12:00:30" and len(ms) == g.n_cycles * 60
    assert np.isnan(ms["speaker_proportion"].iloc[200])  # beyond the data


def test_legacy_tables_omit_what_the_published_code_skipped() -> None:
    g = RestingGrids.empty(T0, T0 + 3600 * NS_PER_S, colony=30)
    assert thesis.legacy_tables(g) == {}


def test_phase_assignment_wraps_midnight() -> None:
    assert thesis.phase_of_hour(5) == "dawn (05-06h)"
    assert thesis.phase_of_hour(15) == "day (06-16h)"
    assert thesis.phase_of_hour(16) == "dusk (16-17h)"
    assert thesis.phase_of_hour(23) == "night (17-05h)" and thesis.phase_of_hour(4) == "night (17-05h)"


def test_phase_table_on_toy_minutes() -> None:
    rows = []
    for day in (4, 5):
        for hour in (5, 12, 16, 22):
            for m in range(60):
                val = 0.1 + (0.05 if 40 <= m < 50 else 0.0) + 0.001 * hour
                rows.append(
                    {
                        "time": f"2026-06-0{day} {hour:02d}:{m:02d}:30",
                        "speaker_proportion": val,
                        "n_resting": 1500.0,
                    }
                )
    res = {r.phase: r for r in thesis.phase_table([pd.DataFrame(rows)])}
    assert set(res) == {"dawn (05-06h)", "day (06-16h)", "dusk (16-17h)", "night (17-05h)"}
    assert all(r.n == 2 and r.test is None for r in res.values())  # n < 4: no test, as published
    assert res["day (06-16h)"].delta_mean == pytest.approx(0.05)


def test_load_zone_errors(tmp_path) -> None:
    from buzzphono.errors import DataError

    with pytest.raises(DataError, match="not found"):
        thesis.load_zone(tmp_path / "custom_zones.json")
    bad = tmp_path / "z.json"
    bad.write_text("{not json")
    with pytest.raises(DataError, match="not valid JSON"):
        thesis.load_zone(bad)
    bad.write_text('{"right": {"polygon": [[0, 0], [1, 0], [1, 1]]}}')
    with pytest.raises(DataError, match="is not in"):
        thesis.load_zone(bad)
    bad.write_text('{"speaker": {"polygon": [[0, 0], [1, 0]]}}')
    with pytest.raises(DataError, match="at least 3"):
        thesis.load_zone(bad)
    bad.write_text('{"speaker": [[0, 0], [4, 0], [4, 4]]}')
    assert thesis.load_zone(bad).shape == (3, 2)
