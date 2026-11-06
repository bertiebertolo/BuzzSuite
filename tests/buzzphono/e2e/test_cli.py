"""The command line, end to end on a simulated experiment (in-process and as a real subprocess)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from buzzphono.cli import run
from buzzphono.synthetic.l2 import L2Design, simulate_experiment

DESIGN = L2Design(hours=12)


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    simulate_experiment(DESIGN, seed=4, out_dir=root / "exp")
    return root


def _cfg(root, **extra):
    import yaml

    data = {
        "experiments": [{"path": "exp", "label": "sim", "frequency_hz": 475}],
        "clock": {"fps": DESIGN.fps},
        "stats": {"n_resamples": 199},
        **extra,
    }
    p = root / "settings.yml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_version_and_help(capsys) -> None:
    assert run(["--version"]) == 0
    assert "buzzphono" in capsys.readouterr().out
    assert run(["--help"]) == 0
    assert "analyse" in capsys.readouterr().out
    assert run([]) in (0, 2)


def test_analyse_writes_results_and_status(exp, capsys) -> None:
    cfg = _cfg(exp)
    out = exp / "out"
    code = run(
        [
            "analyse",
            "-c",
            str(cfg),
            "-a",
            "resting_fraction,psth",
            "-o",
            str(out),
            "--cache",
            str(exp / "cache"),
            "--run-dir",
            str(out / "r1"),
            "-q",
        ]
    )
    assert code == 0, capsys.readouterr().err
    assert (out / "r1" / "report.html").is_file() and (out / "r1" / "run_manifest.json").is_file()
    st = json.loads((out / "r1" / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "done" and st["progress"] == 1.0
    assert (out / "r1" / "events.jsonl").read_text(encoding="utf-8").count("\n") > 3
    # redraw and compare from the written results only
    assert run(["plot", str(out / "r1")]) == 0
    assert run(["compare", str(out / "r1"), "-o", str(out / "cmp")]) == 0
    assert (out / "cmp" / "experiments.csv").is_file()
    assert run(["report", str(out / "r1"), "--no-open"]) == 0
    assert run(["status", str(out / "r1")]) == 0


def test_settings_errors_name_the_problem(exp, capsys) -> None:
    code = run(["analyse", str(exp / "exp"), "-a", "resting_fractoin", "-o", str(exp / "bad")])
    err = capsys.readouterr().err
    assert code == 3 and "resting_fraction" in err
    assert run(["analyse", str(exp / "nope")]) == 3
    assert run(["analyse"]) == 3
    assert "No experiment" in capsys.readouterr().err
    assert run(["analyse", str(exp / "exp"), "--clock", "sundial"]) == 3


def test_validate_and_calibrate(exp, capsys) -> None:
    work = exp / "copy"
    if not work.exists():
        shutil.copytree(exp / "exp", work)
    assert (
        run(
            [
                "validate",
                str(work),
                "--pi-config",
                str(work / "pi_config.ini"),
                "--playback-log",
                str(work / "sound.log"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "calibration" in out and "playback log" in out and "Ready to analyse" in out
    assert run(["calibrate", str(work), "--width-mm", "300", "--height-mm", "170"]) == 0
    cal = json.loads((work / "buzzphono_calibration.json").read_text(encoding="utf-8"))
    assert cal["physical"]["cage_width_mm"] == 300
    assert run(["calibrate", str(work), "--speaker", "1,1 2,1"]) == 3  # a zone needs 3 points
    assert (
        run(
            [
                "calibrate",
                str(work),
                "--speaker",
                "10,10 20,10 20,20",
                "--control",
                "far=500,100 520,100 520,120",
            ]
        )
        == 0
    )
    assert run(["validate", str(exp / "missing")]) == 3


def test_cache_commands(exp, capsys) -> None:
    assert run(["cache", "info", str(exp / "exp"), "--cache", str(exp / "cache")]) == 0
    assert "resting" in capsys.readouterr().out
    assert run(["cache", "verify", str(exp / "exp"), "--cache", str(exp / "cache")]) == 0
    assert run(["cache", "clean", str(exp / "exp"), "--cache", str(exp / "cache2"), "-y"]) == 0


def test_cancel_and_internal_errors_have_exit_codes(exp, monkeypatch) -> None:
    from buzzphono import pipeline
    from buzzphono.errors import Cancelled

    def cancelled(*a, **k):
        raise Cancelled("stop")

    monkeypatch.setattr(pipeline, "run", cancelled)
    rd = exp / "out" / "cancelled"
    assert run(["analyse", str(exp / "exp"), "--run-dir", str(rd), "-q"]) == 130
    assert json.loads((rd / "status.json").read_text(encoding="utf-8"))["state"] == "cancelled"


def test_real_subprocess_with_workers(exp) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    rd = exp / "out" / "sub"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "buzzphono",
            "analyse",
            str(exp / "exp"),
            "-a",
            "resting_fraction",
            "--workers",
            "2",
            "--cache",
            str(exp / "cache_sub"),
            "--run-dir",
            str(rd),
            "--no-figures",
            "-q",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert (rd / "experiments").is_dir()


def test_demo(tmp_path) -> None:
    assert run(["demo", str(tmp_path / "demo"), "--hours", "8", "-q"]) == 0
    reports = list((tmp_path / "demo" / "results").rglob("report.html"))
    assert len(reports) == 1


def test_cli_error_paths_have_clear_messages(exp, tmp_path, capsys) -> None:
    assert run(["analyse", "-c", str(tmp_path / "none.yml")]) == 3
    bad_yaml = tmp_path / "bad.yml"
    bad_yaml.write_text("experiments: [\n", encoding="utf-8")
    assert run(["analyse", "-c", str(bad_yaml)]) == 3
    listy = tmp_path / "list.yml"
    listy.write_text("- just a list\n", encoding="utf-8")
    assert run(["analyse", "-c", str(listy)]) == 3
    assert run(["analyse", str(exp / "exp"), str(exp / "exp"), "--label", "x"]) == 3
    err = capsys.readouterr().err
    assert "not valid YAML" in err and "--label needs exactly one experiment" in err
    work = exp / "copy2"
    if not work.exists():
        shutil.copytree(exp / "exp", work)
    assert run(["calibrate", str(work), "--width-mm", "300"]) == 3
    assert run(["calibrate", str(work), "--control", "no-points"]) == 3
    assert run(["calibrate", str(work), "--speaker", "1,a 2,2 3,3"]) == 3
    assert run(["calibrate", str(work), "--speaker", "1,2,3 2,2 3,3"]) == 3
    assert run(["cache", "clean", str(tmp_path / "nothing"), "-y"]) == 0
    assert run(["status", str(tmp_path)]) == 4
    assert run(["report", str(tmp_path), "--no-open"]) == 4
    assert run(["compare", str(tmp_path), "-o", str(tmp_path / "c")]) == 4
    assert run(["status", str(tmp_path / "r"), "--cancel"]) == 4 and not (tmp_path / "r").exists()
