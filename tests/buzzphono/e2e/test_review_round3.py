"""Regression tests for the review round 3 follow-ups and the fresh-clone check (docs/REVIEW_LOG.md)."""

from __future__ import annotations

import shutil

import pytest

from buzzphono.cli import _Printer, run
from buzzphono.synthetic.l2 import L2Design, simulate_experiment
from buzzphono.validate import validate_experiment


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    root = tmp_path_factory.mktemp("r3")
    simulate_experiment(L2Design(hours=4), seed=5, out_dir=root / "exp")
    return root / "exp"


def test_validate_reports_overlapping_sessions_before_any_analysis(exp, tmp_path) -> None:
    dup = tmp_path / "dup"
    shutil.copytree(exp, dup)
    stamp = sorted(p.name for p in (dup / "tracking_resting").iterdir())[0][len("resting_") :].rsplit("_", 1)[
        0
    ]
    for folder, prefix, suffix in (
        ("tracking_resting", "resting_", ".pkl"),
        ("tracking_moving", "moving_", ".pkl"),
        ("final_tracking_data", "forward_mosq_tracks_", ""),
    ):
        for k in range(3):
            shutil.copy(
                dup / folder / f"{prefix}{stamp}_{5 + k:05d}{suffix}",
                dup / folder / f"{prefix}{stamp}-1_{k:05d}{suffix}",
            )
    # no cache exists yet: frame counts fall back to the chunk indices
    checks = validate_experiment(dup)
    assert any(c.severity == "error" and "overlap" in c.message for c in checks)
    assert not any("overlap" in c.message for c in validate_experiment(exp))


def test_flight_only_that_cannot_run_is_a_data_error(exp, tmp_path, capsys) -> None:
    # the simulator records at 2 fps; the flight analysis is defined for 25 fps
    code = run(
        [
            "analyse",
            str(exp),
            "-a",
            "flight",
            "--cache",
            str(tmp_path / "c"),
            "--run-dir",
            str(tmp_path / "r"),
            "-q",
            "--no-figures",
        ]
    )
    err = capsys.readouterr().err
    assert code == 4 and "flight analysis was requested but could not run" in err and "Traceback" not in err


def test_progress_is_throttled_within_a_stage(capsys) -> None:
    p = _Printer(quiet=False)
    for k in range(1000):
        p(k / 1000 * 0.5, f"reading chunk {k + 1}/1000")
    p(0.6, "binning")
    p(1.0, "done")
    lines = capsys.readouterr().err.splitlines()
    assert 20 <= len(lines) <= 30
    assert lines[0].endswith("reading chunk 1/1000") and lines[-1].endswith("done")
