"""Regression tests for the problems found in review round 1 (docs/REVIEW_LOG.md)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime

import pytest

from buzzphono import jobs
from buzzphono.cli import run
from buzzphono.config import build_config
from buzzphono.errors import BuzzPhonoError, CalibrationError
from buzzphono.synthetic.l2 import L2Design, simulate_experiment

DESIGN = L2Design(hours=12)


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    root = tmp_path_factory.mktemp("r1")
    simulate_experiment(DESIGN, seed=8, out_dir=root / "exp")
    return root


def test_labels_and_duplicates_are_refused(exp) -> None:
    for bad in ("..", "../../x", "a/b", "a\\b"):
        with pytest.raises(BuzzPhonoError):
            build_config({"experiments": [{"path": str(exp / "exp"), "label": bad}]})
    with pytest.raises(BuzzPhonoError, match="counted twice"):
        build_config({"experiments": [{"path": str(exp / "exp")}, {"path": str(exp / "exp" / ".")}]})


def test_yaml_on_off_keys_and_all(exp) -> None:
    import yaml

    text = (
        f'experiments: [{{path: "{(exp / "exp").as_posix()}"}}]\n'
        "analyses: all\nstimulus:\n  on: 40-50\n  off: 28-38\n"  # bare on/off: YAML booleans
    )
    cfg = build_config(yaml.safe_load(text))
    assert cfg.stimulus.on.as_tuple() == (40, 50) and "flight" in cfg.analyses


def test_frame_rate_comes_from_the_experiment(exp) -> None:
    cfg = build_config({"experiments": [{"path": str(exp / "exp")}]})
    assert cfg.clock.fps is None and cfg.fps_for(cfg.experiments[0]) == DESIGN.fps


def test_failed_start_counts_even_outside_the_log_span(tmp_path) -> None:
    from buzzphono.io.playback_log import read_playback_log, window_status

    log_file = tmp_path / "sound.log"
    log_file.write_text(
        "2026-10-10 00:39:58 [INFO] Sound schedule running: 510 Hz tone on dev, minutes 40-50 of every hour\n"
        "2026-10-10 00:40:00 [INFO] Window open - playing 510 Hz tone at 80% volume\n"
        "2026-10-10 00:50:00 [INFO] Window closed - playback stopped\n"
        "2026-10-10 01:40:00 [ERROR] Could not start playback: [Errno 2] No such file (retry in 10s)\n",
        encoding="utf-8",
    )
    log = read_playback_log(log_file)
    status, _ = window_status(log, datetime(2026, 10, 10, 1, 40), datetime(2026, 10, 10, 1, 50))
    assert status == "failed"


def test_broken_zone_files_give_calibration_errors(exp, tmp_path) -> None:
    from buzzphono.calibration import import_existing

    for content in (
        "{not json",
        "[1, 2]",
        '{"speaker": {"polygon": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}}',
        '{"speaker": {"polygon": [[1, 2], [3, 4]]}}',
    ):
        work = tmp_path / f"z{abs(hash(content))}"
        shutil.copytree(exp / "exp", work)
        (work / "custom_zones.json").write_text(content, encoding="utf-8")
        with pytest.raises(CalibrationError):
            import_existing(work)
    assert (
        run(
            [
                "calibrate",
                str(exp / "exp"),
                "--width-mm",
                "-5",
                "--height-mm",
                "100",
                "--out",
                str(tmp_path / "c.json"),
            ]
        )
        == 3
    )


def test_corrupt_cache_file_is_rebuilt(exp, tmp_path) -> None:
    from buzzphono.io.discover import discover
    from buzzphono.io.ingest import ingest

    layout = discover(exp / "exp")
    rep = ingest(layout, tiers=("resting",), cache_root=tmp_path / "cache", workers=1)
    victim = next((rep.cache_dir / "resting").glob("*.npz"))
    victim.write_bytes(victim.read_bytes()[:100])
    rep2 = ingest(layout, tiers=("resting",), cache_root=tmp_path / "cache", workers=1)
    assert rep2.built == 1 and not rep2.errors


def test_status_of_a_killed_run_and_compare_twice(exp, tmp_path, capsys) -> None:
    run_dir = tmp_path / "dead"
    run_dir.mkdir()
    (run_dir / jobs.STATUS).write_text(
        json.dumps({"state": "running", "pid": 2**22 + 99, "progress": 0.3}), encoding="utf-8"
    )
    assert run(["status", str(run_dir)]) == 0
    assert "stopped" in capsys.readouterr().out
    out = tmp_path / "r"
    assert (
        run(
            [
                "analyse",
                str(exp / "exp"),
                "-a",
                "resting_fraction",
                "--cache",
                str(tmp_path / "c"),
                "--run-dir",
                str(out),
                "-q",
                "--no-figures",
            ]
        )
        == 0
    )
    assert run(["compare", str(out), str(out), "-o", str(tmp_path / "cmp")]) == 4
    assert run(["compare", str(out), "-o", str(exp / "exp")]) == 4
    (out / "experiments" / "exp" / "flight").mkdir(parents=True, exist_ok=True)
    (out / "experiments" / "exp" / "flight" / "empty.csv").write_text("", encoding="utf-8")
    assert run(["plot", str(out)]) == 0


def test_no_usable_cycles_is_an_error_and_overlaps_are_excluded(exp, tmp_path) -> None:
    import yaml

    cfg = tmp_path / "late.yml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {"path": str(exp / "exp"), "start": "2030-01-01 00:00", "end": "2030-01-02 00:00"}
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        run(
            [
                "analyse",
                "-c",
                str(cfg),
                "--cache",
                str(tmp_path / "c2"),
                "--run-dir",
                str(tmp_path / "late"),
                "-q",
                "--no-figures",
            ]
        )
        == 4
    )
    # duplicate a few chunks as an '-attempt' session over the same hours
    dup = tmp_path / "dup"
    shutil.copytree(exp / "exp", dup)
    for folder, prefix, suffix in (
        ("tracking_resting", "resting_", ".pkl"),
        ("tracking_moving", "moving_", ".pkl"),
        ("final_tracking_data", "forward_mosq_tracks_", ""),
    ):
        for k in range(3):
            src = dup / folder / f"{prefix}SynthCage_20260110_120009_{5 + k:05d}{suffix}"
            shutil.copy(src, dup / folder / f"{prefix}SynthCage_20260110_120009-1_{k:05d}{suffix}")
    from buzzphono import pipeline

    res = pipeline.run(
        build_config(
            {"experiments": [{"path": str(dup)}], "workers": 1, "output": {"cache_dir": str(tmp_path / "c3")}}
        ),
        run_dir=tmp_path / "dupres",
        make_figures=False,
        make_report=False,
    )
    assert any("overlap in time" in w for w in res.warnings)
    c = res.experiments[0].cycles
    assert (c.loc[c["usable"], "on_frame_fraction"] <= 1.15).all()


def test_flight_is_skipped_at_other_frame_rates(exp, tmp_path) -> None:
    from buzzphono import pipeline

    res = pipeline.run(
        build_config(
            {
                "experiments": [{"path": str(exp / "exp")}],
                "analyses": ["flight"],
                "workers": 1,
                "output": {"cache_dir": str(tmp_path / "c")},
            }
        ),
        run_dir=tmp_path / "f",
        make_figures=False,
        make_report=False,
    )
    assert any("25-fps" in w for w in res.warnings)


def test_sensitivity_flags_scale_with_zone_width(exp, tmp_path) -> None:
    from buzzphono.calibration import import_existing
    from buzzphono.frames import experiment_clock
    from buzzphono.io.discover import discover
    from buzzphono.io.ingest import ingest
    from buzzphono.metrics.sensitivity import SensitivityPlan, run_sensitivity

    layout = discover(exp / "exp")
    rep = ingest(layout, tiers=("resting", "moving", "tracks"), cache_root=tmp_path / "c", workers=1)
    cal = import_existing(exp / "exp")
    ec = experiment_clock(layout, rep.cache_dir, cage=cal.cage_polygon(), fps=DESIGN.fps)
    plan = SensitivityPlan(
        margins_px=[], shifts_px=[200.0], colonies=[], walking_mm_s=[], on_windows={}, off_windows={}
    )
    table = run_sensitivity(
        layout,
        rep.cache_dir,
        {"corrected": ec},
        speaker=cal.polygon("speaker"),
        cage=cal.cage_polygon(),
        frame_size=cal.frame_size,
        fps=DESIGN.fps,
        plan=plan,
        n_resamples=99,
    )
    far = table[table["variant"] == "+200 px"].iloc[0]
    assert far["zone_change_rel_width"] > 1 and "sign flips" not in str(far["flag"])
