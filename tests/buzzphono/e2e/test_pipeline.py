"""The whole pipeline on a simulated experiment: every analysis runs, writes its tables, figures,
report and manifest, and a rerun gives identical outputs."""

from __future__ import annotations

import json

import pytest

from buzzphono import pipeline
from buzzphono.config import build_config
from buzzphono.synthetic.l2 import L2Design, simulate_experiment

ALL = [
    "resting_fraction",
    "attraction_index",
    "psth",
    "zt_profile",
    "spatial",
    "approach",
    "reliability",
    "sensitivity",
]
DESIGN = L2Design(hours=24, drift_s_per_day=40.0)


def _config(root, analyses=ALL, **extra):
    return build_config(
        {
            "experiments": [{"path": str(root / "exp"), "label": "sim", "frequency_hz": 475}],
            "analyses": analyses,
            "clock": {"fps": DESIGN.fps},
            "stats": {"n_resamples": 499},
            "output": {"dir": str(root / "out"), "cache_dir": str(root / "cache"), "formats": ["png", "svg"]},
            "workers": 2,
            **extra,
        }
    )


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    simulate_experiment(DESIGN, seed=7, out_dir=root / "exp")
    return root


@pytest.fixture(scope="module")
def result(sim):
    return pipeline.run(_config(sim), run_dir=sim / "out" / "run1")


def test_every_analysis_writes_its_tables(result) -> None:
    (e,) = result.experiments
    expected = {
        "cycles/cycle_table",
        "resting_fraction/statistics",
        "attraction_index/statistics",
        "psth/psth",
        "zt_profile/hour_of_day",
        "spatial/occupancy_maps",
        "spatial/radial_profile",
        "approach/per_cycle_window",
        "reliability/summary",
        "sensitivity/variants",
    }
    assert expected <= set(e.tables), set(e.tables)
    for name in expected:
        assert (e.out_dir / f"{name}.csv").is_file()
    st = e.tables["resting_fraction/statistics"]
    main = st[(st["comparison"] == "ON vs OFF (speaker zone)") & (st["phase"] == "all")].iloc[0]
    sham = st[(st["comparison"].str.startswith("sham")) & (st["phase"] == "all")].iloc[0]
    assert main["mean_delta"] > 0 and main["wilcoxon_p_value"] < 0.01
    assert abs(sham["mean_delta"]) < main["mean_delta"] / 2


def test_figures_report_and_manifest(result) -> None:
    (e,) = result.experiments
    names = {p.stem for p in e.figures}
    assert {
        "resting_on_vs_off",
        "psth_colony_frac_in_zone",
        "zt_profile",
        "occupancy_maps",
        "sensitivity",
    } <= names
    assert {p.suffix for p in e.figures} == {".png", ".svg"}
    html = result.report.read_text(encoding="utf-8")
    assert "BuzzPhono run" in html and "data:image/png;base64," in html and "<script" not in html
    m = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert m["settings"]["analyses"] == ALL and m["inputs"][0]["files"]
    assert any(k.endswith("resting_fraction/statistics.csv") for k in m["outputs"])


def test_rerun_gives_identical_outputs(sim, result) -> None:
    again = pipeline.run(_config(sim), run_dir=sim / "out" / "run2", make_report=False)
    a = json.loads(result.manifest.read_text(encoding="utf-8"))["outputs"]
    b = json.loads(again.manifest.read_text(encoding="utf-8"))["outputs"]
    assert a == b


def test_thesis_clock_mode_writes_the_published_tables(sim) -> None:
    res = pipeline.run(
        _config(sim, analyses=["resting_fraction"], clock={"mode": "thesis", "fps": 25.0}),
        run_dir=sim / "out" / "thesis",
        make_figures=False,
    )
    assert any(k.startswith("resting_fraction/thesis_") for k in res.experiments[0].tables)


def test_cancel_stops_the_run(sim) -> None:
    from buzzphono.errors import Cancelled

    with pytest.raises(Cancelled):
        pipeline.run(_config(sim), run_dir=sim / "out" / "cancelled", cancel=lambda: True)


def test_pi_config_schedule_and_zone_settings(sim) -> None:
    from buzzphono.errors import ConfigError

    exp = sim / "exp"
    res = pipeline.run(
        _config(
            sim,
            analyses=["resting_fraction"],
            stimulus={
                "pi_config": str(exp / "pi_config.ini"),
                "playback_log": [str(exp / "sound.log")],
                "active_from": "13:00",
                "active_until": "20:00",
            },
            metric={"speaker_zone": "opposite"},
        ),
        run_dir=sim / "out" / "settings",
        make_figures=False,
        make_report=False,
    )
    (e,) = res.experiments
    c = e.cycles
    assert set(c.loc[c["usable"], "hour"]) <= set(range(13, 20))
    assert (~c["tone_scheduled"]).any()
    assert "stimulus_status" in c and (c["stimulus_status"] != "not checked (no playback log)").any()
    st = e.tables["resting_fraction/statistics"]
    main = st[(st["comparison"] == "ON vs OFF (speaker zone)") & (st["phase"] == "all")].iloc[0]
    assert abs(main["mean_delta"]) < 0.05  # the opposite wall used as "speaker": no effect there
    with pytest.raises(ConfigError, match="playback_log"):
        pipeline.run(
            _config(sim, analyses=["resting_fraction"], stimulus={"source": "playback_log"}),
            run_dir=sim / "out" / "bad",
        )
