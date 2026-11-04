"""Regression tests for the problems found in review round 2 (docs/REVIEW_LOG.md)."""

from __future__ import annotations

import shutil

import pytest
import yaml

from buzzphono import pipeline
from buzzphono.cli import run
from buzzphono.config import build_config
from buzzphono.errors import BuzzPhonoError
from buzzphono.synthetic.l2 import L2Design, simulate_experiment


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    root = tmp_path_factory.mktemp("r2")
    simulate_experiment(L2Design(hours=10), seed=21, out_dir=root / "exp")
    return root


def test_nothing_usable_exits_4_without_a_traceback(exp, tmp_path, capsys) -> None:
    cfg = tmp_path / "s.yml"
    cfg.write_text(
        yaml.safe_dump({"experiments": [{"path": str(exp / "exp"), "start": "2030-01-01 00:00"}]}),
        encoding="utf-8",
    )
    code = run(
        ["analyse", "-c", str(cfg), "--cache", str(tmp_path / "c"), "--run-dir", str(tmp_path / "r"), "-q"]
    )
    assert code == 4 and "Traceback" not in capsys.readouterr().err
    short = tmp_path / "short.yml"
    short.write_text(
        yaml.safe_dump(
            {"experiments": [{"path": str(exp / "exp")}], "stimulus": {"on": "40-45", "off": "30-35"}}
        ),
        encoding="utf-8",
    )
    assert (
        run(
            [
                "analyse",
                "-c",
                str(short),
                "--cache",
                str(tmp_path / "c"),
                "--run-dir",
                str(tmp_path / "r2"),
                "-q",
            ]
        )
        == 4
    )


def test_a_copy_of_an_experiment_is_the_same_data(exp, tmp_path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(exp / "exp", copy)
    cfg = build_config({"experiments": [{"path": str(exp / "exp")}, {"path": str(copy)}]})
    with pytest.raises(BuzzPhonoError, match="same recordings"):
        pipeline.run(cfg, run_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_labels_that_collide_on_windows(exp, tmp_path) -> None:
    for bad in ("nul", "CON", "com1.txt"):
        with pytest.raises(BuzzPhonoError, match="reserved"):
            build_config({"experiments": [{"path": str(exp / "exp"), "label": bad}]})
    from buzzphono.pipeline import _folder_key

    assert _folder_key("Cage.") == _folder_key("cage")


def test_sham_must_not_contain_the_tone(exp) -> None:
    with pytest.raises(BuzzPhonoError, match="sham"):
        build_config(
            {"experiments": [{"path": str(exp / "exp")}], "stimulus": {"on": "20-30", "off": "40-50"}}
        )


def test_wrong_colony_size_is_warned(exp, tmp_path) -> None:
    res = pipeline.run(
        build_config(
            {
                "experiments": [{"path": str(exp / "exp")}],
                "metric": {"colony_size": 1},
                "workers": 1,
                "output": {"cache_dir": str(tmp_path / "c")},
            }
        ),
        run_dir=tmp_path / "out",
        make_figures=False,
        make_report=False,
    )
    assert any("colony size" in w for w in res.warnings)


def test_results_never_go_inside_an_experiment(exp, tmp_path) -> None:
    cfg = build_config({"experiments": [{"path": str(exp / "exp")}]})
    for target in (exp / "exp" / "run1", exp / "exp" / "tracking_resting" / "run1"):
        with pytest.raises(BuzzPhonoError, match="inside an experiment"):
            pipeline.run(cfg, run_dir=target)
        assert not target.exists()
