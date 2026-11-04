"""The job protocol (status, cancel, experiment lock) and the pre-flight validation checks."""

from __future__ import annotations

import json
import os
import shutil
import time

import pytest

from buzzphono import jobs
from buzzphono.errors import DataError
from buzzphono.synthetic.l2 import L2Design, simulate_experiment
from buzzphono.validate import validate_experiment


def test_job_status_events_cancel_and_cleanup(tmp_path) -> None:
    run = tmp_path / "run"
    job = jobs.Job(run, min_interval_s=0.0)
    assert jobs.read_status(run)["state"] == "running"
    job.progress(0.5, "half way")
    job.progress(2.0, "clamped")
    st = jobs.read_status(run)
    assert st["progress"] == 1.0 and st["message"] == "clamped"
    assert (run / jobs.EVENTS).read_text(encoding="utf-8").count("\n") == 2
    assert not job.cancelled()
    jobs.request_cancel(run)
    assert job.cancelled()
    (run / "experiments").mkdir()
    (run / "report.html").write_text("x", encoding="utf-8")
    jobs.remove_partial_run(run)
    assert sorted(p.name for p in run.iterdir()) == [jobs.CANCEL, jobs.EVENTS, jobs.STATUS]
    job.finish("failed", DataError("bad", hint="do this"), 4)
    st = jobs.read_status(run)
    assert st["state"] == "failed" and st["hint"] == "do this" and st["exit_code"] == 4
    job.finish("done")
    assert jobs.read_status(run)["progress"] == 1.0
    assert jobs.read_status(tmp_path / "nowhere") is None
    jobs.remove_partial_run(tmp_path / "nowhere")  # nothing to do, no error


def test_experiment_lock(tmp_path) -> None:
    cache = tmp_path / "cache"
    with jobs.experiment_lock(cache):
        assert (cache / jobs.LOCK).is_file()
        assert jobs.lock_holder(cache) is None  # our own lock does not block us
    assert not (cache / jobs.LOCK).exists()
    # a live process holding a fresh lock blocks; a dead or stale one is taken over
    parent = os.getppid()
    (cache / jobs.LOCK).write_text(
        json.dumps({"pid": parent, "started": "x", "heartbeat": time.time()}), encoding="utf-8"
    )
    assert jobs.lock_holder(cache) is not None
    with pytest.raises(DataError, match="Another BuzzPhono run"), jobs.experiment_lock(cache):
        pass
    (cache / jobs.LOCK).write_text(
        json.dumps({"pid": parent, "heartbeat": time.time() - 3600}), encoding="utf-8"
    )
    assert jobs.lock_holder(cache) is None
    (cache / jobs.LOCK).write_text(
        json.dumps({"pid": 2**22 + 12345, "heartbeat": time.time()}), encoding="utf-8"
    )
    assert jobs.lock_holder(cache) is None
    (cache / jobs.LOCK).write_text("not json", encoding="utf-8")
    with jobs.experiment_lock(cache):
        pass


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    root = tmp_path_factory.mktemp("validate")
    simulate_experiment(L2Design(hours=4), seed=3, out_dir=root / "exp")
    return root / "exp"


def _areas(checks):
    return {(c.severity, c.area) for c in checks}


def test_validate_good_and_degraded_experiments(exp, tmp_path) -> None:
    checks = validate_experiment(
        exp, pi_config=exp / "pi_config.ini", playback_logs=[exp / "sound.log"], frequency_hz=475
    )
    assert not [c for c in checks if c.severity == "error"]
    assert ("ok", "playback log") in _areas(checks)
    bad = tmp_path / "bad"
    shutil.copytree(exp, bad)
    shutil.rmtree(bad / "tracking_moving")
    (bad / "final_tracking_data" / next(iter(os.listdir(bad / "final_tracking_data")))).unlink()
    (bad / "tracking_resting" / "resting_notastamp.pkl").write_bytes(b"")
    cal = json.loads((bad / "custom_zones.json").read_text(encoding="utf-8"))
    cal = {"speaker": cal["speaker"]}
    (bad / "custom_zones.json").write_text(json.dumps(cal), encoding="utf-8")
    checks = validate_experiment(
        bad, pi_config=bad / "missing.ini", playback_logs=[bad / "missing.log"], frequency_hz=200
    )
    msgs = " ".join(c.message for c in checks)
    assert "No moving files" in msgs and "no tracks file" in msgs
    assert "No control zones" in msgs
    assert ("error", "Pi config") in _areas(checks) and ("error", "playback log") in _areas(checks)


def test_validate_empty_and_missing_folders(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert [c.severity for c in validate_experiment(empty)] == ["error"]
    assert validate_experiment(tmp_path / "missing")[0].message.endswith("is not a folder.")


def test_validate_reports_a_resized_camera(exp, tmp_path) -> None:
    from PIL import Image

    from buzzphono.calibration import CALIBRATION_FILE, import_existing, save_calibration

    work = tmp_path / "resized"
    shutil.copytree(exp, work)
    save_calibration(import_existing(work), work / CALIBRATION_FILE)  # calibrated at the old size
    for p in (work / "individual_images").glob("*.png"):
        Image.new("L", (800, 600)).save(p)
    checks = validate_experiment(work)
    assert any(c.severity == "error" and "camera or its resolution changed" in c.message for c in checks)


def test_runs_never_touch_an_experiment_or_foreign_folder(exp, tmp_path) -> None:
    from buzzphono.errors import ConfigError

    with pytest.raises(ConfigError, match="experiment folder"):
        jobs.Job(exp)
    other = tmp_path / "other"
    other.mkdir()
    (other / "thesis.docx").write_text("precious", encoding="utf-8")
    with pytest.raises(ConfigError, match="already holds other files"):
        jobs.Job(other)
    # clean-up removes only a run's own outputs, even if called on a folder with other files
    (other / "experiments").mkdir()
    jobs.remove_partial_run(other)
    assert (other / "thesis.docx").read_text(encoding="utf-8") == "precious" and not (
        other / "experiments"
    ).exists()
    before = sorted(p.name for p in exp.iterdir())
    jobs.remove_partial_run(exp)
    assert sorted(p.name for p in exp.iterdir()) == before
