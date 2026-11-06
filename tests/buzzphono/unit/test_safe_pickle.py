from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from buzzphono.errors import DataError
from buzzphono.io.safe_pickle import PandasPlaceholder, TrackerObject, load_tracking_pickle
from buzzphono.synthetic.writers import tracker_pickle_bytes

OBJECTS = {0: {"coordinates": [(1.5, 2.0), (1.5, 2.0)], "time_stamp": [10, 11], "start_type": np.nan}}


def test_loads_a_tracker_without_the_tracker_code(tmp_path) -> None:
    path = tmp_path / "resting_x.pkl"
    path.write_bytes(tracker_pickle_bytes(OBJECTS, "resting"))
    obj = load_tracking_pickle(path)
    assert isinstance(obj, TrackerObject)
    assert obj.type == "resting" and obj.number_objects == 1
    assert obj.objects[0]["time_stamp"] == [10, 11]


def test_final_file_with_pandas_table_and_datetimes(tmp_path) -> None:
    extra = {
        "time_stamp": [datetime(2026, 6, 4, 12, 0, 9)] * 3,
        "population_variables": pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2026-06-04", periods=2)),
        "settings": {"fps": 25.0},
        "video_name": "Cage_20260604_120009_00000",
        "positions": np.arange(6, dtype=np.int32).reshape(3, 2),
    }
    path = tmp_path / "forward_mosq_tracks_x"
    path.write_bytes(tracker_pickle_bytes(OBJECTS, "mixed", **extra))
    obj = load_tracking_pickle(path)
    assert isinstance(obj.population_variables, PandasPlaceholder)
    assert len(obj.time_stamp) == 3 and obj.settings == {"fps": 25.0}
    assert np.array_equal(obj.positions, np.arange(6).reshape(3, 2))


def test_numpy1_module_paths_are_accepted(tmp_path) -> None:
    """Files written under numpy 1 refer to numpy.core, numpy 2 files to numpy._core."""
    proto2 = _protocol2_with_numpy_core()
    path = tmp_path / "moving_x.pkl"
    path.write_bytes(proto2)
    obj = load_tracking_pickle(path)
    assert float(obj.objects[1]["v"]) == 2.5
    assert np.array_equal(obj.objects[1]["a"], [1.0, 2.0])


def _protocol2_with_numpy_core() -> bytes:
    """A tracker pickle whose numpy globals say numpy.core (protocol 2 names globals as text)."""
    from buzzphono.synthetic import writers

    with writers._tracker_class() as cls:
        obj = cls.__new__(cls)
        obj.__dict__.update(
            objects=OrderedDict(
                {1: {"coordinates": [], "time_stamp": [], "v": np.float64(2.5), "a": np.array([1.0, 2.0])}}
            ),
            number_objects=1,
            type="moving",
        )
        raw = pickle.dumps(obj, protocol=2)
    return raw.replace(b"numpy._core.multiarray", b"numpy.core.multiarray")


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


@pytest.mark.parametrize(
    "payload",
    [pickle.dumps(_Evil()), pickle.dumps(OrderedDict(a=1)), b"\x80\x04garbage", b"not a pickle at all"],
)
def test_refuses_anything_that_is_not_a_tracker(tmp_path, payload) -> None:
    path = tmp_path / "bad.pkl"
    path.write_bytes(payload)
    with pytest.raises(DataError):
        load_tracking_pickle(path)


def test_empty_and_truncated_files(tmp_path) -> None:
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(b"")
    with pytest.raises(DataError, match="empty"):
        load_tracking_pickle(empty)
    good = tracker_pickle_bytes(OBJECTS, "resting")
    cut = tmp_path / "cut.pkl"
    cut.write_bytes(good[: len(good) // 2])
    with pytest.raises(DataError, match="corrupt or truncated"):
        load_tracking_pickle(cut)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(DataError, match="Could not read"):
        load_tracking_pickle(tmp_path / "nope.pkl")
