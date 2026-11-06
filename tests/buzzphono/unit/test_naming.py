from __future__ import annotations

from datetime import datetime

from buzzphono.io.naming import ChunkName, legacy_recognises, parse_filename, parse_stem, stem_from_filename


def test_parse_plain_stem() -> None:
    c = parse_stem("CageAedesAegypti475M_20260604_120009_00042")
    assert c == ChunkName("CageAedesAegypti475M", "20260604", "120009", None, 42)
    assert c.session == "CageAedesAegypti475M_20260604_120009"
    assert c.session_start() == datetime(2026, 6, 4, 12, 0, 9)
    assert c.stem == "CageAedesAegypti475M_20260604_120009_00042"


def test_label_with_underscores_and_dashes() -> None:
    c = parse_stem("CageAedesAegyptiF_250_test_20260710_120009_00071")
    assert c is not None and c.label == "CageAedesAegyptiF_250_test" and c.index == 71
    c = parse_stem("TbambusaM-510Hz-1212LD-5am5pm_20261007_151051_00415")
    assert c is not None and c.label == "TbambusaM-510Hz-1212LD-5am5pm" and c.index == 415


def test_attempt_suffix() -> None:
    c = parse_stem("Cage_20260604_120009-2_00003")
    assert c is not None and c.attempt == 2 and c.session == "Cage_20260604_120009-2"
    assert c.stem == "Cage_20260604_120009-2_00003"


def test_rejects_non_matching() -> None:
    assert parse_stem("Cage_2026064_120009_00003") is None
    assert parse_stem("Cage_20260604_120009_3") is None
    assert parse_stem("random") is None


def test_impossible_session_stamp_is_reported_not_raised() -> None:
    c = parse_stem("Cage_20261399_256199_00001")
    assert c is not None and c.session_start() is None


def test_filenames_per_kind() -> None:
    c = ChunkName("Cage", "20260604", "120009", None, 5)
    assert c.filename("resting") == "resting_Cage_20260604_120009_00005.pkl"
    assert c.filename("moving") == "moving_Cage_20260604_120009_00005.pkl"
    assert c.filename("tracks") == "forward_mosq_tracks_Cage_20260604_120009_00005"
    for kind in ("resting", "moving", "tracks"):
        assert parse_filename(kind, c.filename(kind)) == c


def test_stem_from_filename_rejects_other_files() -> None:
    assert stem_from_filename("resting", "._resting_Cage_20260604_120009_00005.pkl") is None
    assert stem_from_filename("resting", "moving_Cage_20260604_120009_00005.pkl") is None
    assert stem_from_filename("resting", "resting_activity_timeseries.csv") is None
    assert stem_from_filename("tracks", "forward_mosq_tracks_Cage_20260604_120009_00005.tmp") is None


def test_legacy_recognition_matches_published_regex() -> None:
    assert legacy_recognises("resting", "resting_Cage_20260604_120009_00005.pkl")
    assert not legacy_recognises("resting", "resting_Cage_20260604_120009-2_00005.pkl")
