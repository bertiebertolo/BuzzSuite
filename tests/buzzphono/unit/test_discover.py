from __future__ import annotations

import pytest

from buzzphono.errors import DataError
from buzzphono.io.discover import discover
from buzzphono.io.naming import ChunkName
from buzzphono.synthetic.writers import write_tracking_file


def _chunk(index: int, time: str = "120009", attempt: int | None = None) -> ChunkName:
    return ChunkName("Cage", "20260604", time, attempt, index)


def test_not_an_experiment(tmp_path) -> None:
    with pytest.raises(DataError, match="not found"):
        discover(tmp_path / "missing")
    with pytest.raises(DataError, match="has none of"):
        discover(tmp_path)


def test_lists_chunks_and_missing_tiers(tmp_path) -> None:
    for i in range(3):
        write_tracking_file(tmp_path, "resting", _chunk(i), {})
    write_tracking_file(tmp_path, "moving", _chunk(0), {})
    write_tracking_file(tmp_path, "moving", _chunk(1, attempt=2), {})
    (tmp_path / "tracking_resting" / "._resting_Cage_20260604_120009_00000.pkl").write_bytes(b"x")
    (tmp_path / "tracking_resting" / "resting_activity_timeseries.csv").write_text("x")
    (tmp_path / "tracking_resting" / "resting_badname.pkl").write_bytes(b"x")
    (tmp_path / "tracking_resting" / "resting_Cage_20260604_120009_00009.pkl").write_bytes(b"")
    layout = discover(tmp_path)
    ids = [c.chunk_id for c in layout.chunks]
    assert ids == [
        "Cage_20260604_120009_00000",
        "Cage_20260604_120009_00001",
        "Cage_20260604_120009_00002",
        "Cage_20260604_120009_00009",
        "Cage_20260604_120009-2_00001",
    ]
    assert layout.metadata_files_skipped == 1
    assert [p.name for p in layout.unparsable] == ["resting_badname.pkl"]
    assert [p.name for p in layout.empty_files] == ["resting_Cage_20260604_120009_00009.pkl"]
    assert layout.missing("moving") == [
        "Cage_20260604_120009_00001",
        "Cage_20260604_120009_00002",
        "Cage_20260604_120009_00009",
    ]
    assert layout.missing("tracks") == []  # the folder does not exist, so nothing is "missing"
    assert set(layout.sessions()) == {"Cage_20260604_120009", "Cage_20260604_120009-2"}
