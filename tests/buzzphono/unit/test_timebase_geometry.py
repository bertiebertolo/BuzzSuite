from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from buzzphono import geometry
from buzzphono.errors import DataError
from buzzphono.io.naming import ChunkName
from buzzphono.timebase import (
    NS_PER_HOUR,
    NS_PER_S,
    ThesisClock,
    floor_hour,
    from_ns,
    legacy_seconds_to_ns,
    to_ns,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_ns_round_trip() -> None:
    t = datetime(2026, 6, 4, 12, 0, 9, 123456)
    assert from_ns(to_ns(t)) == t
    assert to_ns(datetime(1970, 1, 1, 0, 0, 1)) == NS_PER_S


def test_legacy_rounding_matches_pandas_2_0() -> None:
    """The published clock converted float seconds with pandas 2.0's rounding; reproduce it exactly."""
    data = np.load(FIXTURES / "pandas203_seconds_to_ns.npz")
    assert np.array_equal(legacy_seconds_to_ns(data["frames"] / 25.0), data["ns"])


def test_whole_frames_are_exact_multiples_of_40_ms() -> None:
    frames = np.arange(0, 40000, dtype=np.float64)
    assert np.array_equal(legacy_seconds_to_ns(frames / 25.0), np.arange(40000, dtype=np.int64) * 40_000_000)


def test_thesis_clock_places_chunks_every_1200_s() -> None:
    clock = ThesisClock()
    c = ChunkName("Cage", "20260604", "120009", None, 3)
    start = to_ns(datetime(2026, 6, 4, 12, 0, 9)) + 3 * 1200 * NS_PER_S
    assert clock.chunk_start_ns(c) == start
    assert clock.frame_ns(c, np.array([0.0, 25.0]))[1] == start + NS_PER_S


def test_thesis_clock_refuses_impossible_dates() -> None:
    with pytest.raises(DataError, match="not a real date"):
        ThesisClock().chunk_start_ns(ChunkName("Cage", "20261399", "120009", None, 0))


def test_floor_hour() -> None:
    t = to_ns(datetime(2026, 6, 4, 12, 59, 59))
    assert floor_hour(t) == to_ns(datetime(2026, 6, 4, 12))
    assert floor_hour(np.array([t, t + NS_PER_S]))[1] == to_ns(datetime(2026, 6, 4, 13))
    assert int(floor_hour(NS_PER_HOUR)) == NS_PER_HOUR


SQUARE = geometry.as_polygon([[0, 0], [10, 0], [10, 10], [0, 10]])


def test_polygon_basics() -> None:
    assert geometry.area(SQUARE) == 100.0
    assert geometry.centroid(SQUARE).tolist() == [5.0, 5.0]
    closed = geometry.as_polygon([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
    assert closed.shape == (4, 2)
    with pytest.raises(ValueError):
        geometry.as_polygon([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        geometry.as_polygon([[0, 0], [1, np.nan], [2, 2]])


def test_contains_uses_the_published_rule() -> None:
    from matplotlib.path import Path as MplPath

    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.integers(-2, 13, size=(400, 2)).astype(float), rng.uniform(-2, 12, (400, 2))])
    expected = MplPath(SQUARE).contains_points(pts)
    assert np.array_equal(geometry.contains(SQUARE, pts), expected)
    assert geometry.contains(SQUARE, np.empty((0, 2))).shape == (0,)


def test_distances() -> None:
    pts = np.array([[5, 5], [15, 5], [5, -3], [13, 14]], dtype=float)
    assert geometry.distance_to_polygon(SQUARE, pts).tolist() == [0.0, 5.0, 3.0, 5.0]
    sd = geometry.signed_distance(SQUARE, pts)
    assert sd[0] == -5.0 and sd[1] == 5.0
