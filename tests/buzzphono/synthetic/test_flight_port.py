"""The flight analyses on a simulated 25-fps experiment: they run end to end and see the planted arousal."""

from __future__ import annotations

import numpy as np
import pytest

from buzzphono.calibration import Calibration, Zone
from buzzphono.flight_phono.build import FlightParams
from buzzphono.flight_phono.geometry import build_geometry, role_order
from buzzphono.flight_phono.pipeline import run_flight
from buzzphono.io.discover import discover
from buzzphono.io.ingest import ingest
from buzzphono.synthetic.l2 import L2Design, simulate_experiment

DESIGN = L2Design(hours=4, fps=25.0, colony=6)


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    out = tmp_path_factory.mktemp("flight")
    simulate_experiment(DESIGN, seed=3, out_dir=out / "exp")
    layout = discover(out / "exp")
    report = ingest(layout, tiers=("resting", "moving", "tracks"), cache_root=out / "cache", workers=2)
    zones = DESIGN.zones()
    cal = Calibration(
        frame_width_px=DESIGN.width,
        frame_height_px=DESIGN.height,
        cage=[tuple(p) for p in DESIGN.cage],
        zones=[
            Zone(name=k, role="speaker" if k == "speaker" else "control", polygon=[tuple(p) for p in v])
            for k, v in zones.items()
        ],
    )
    return layout, report, cal


def test_geometry_roles() -> None:
    geom = build_geometry(DESIGN.cage, DESIGN.zones(), FlightParams().r_fracs())
    order, roles = role_order(geom)
    assert roles == ["speaker", "opposite", "top", "bottom"] and order == [0, 1, 2, 3]
    assert sorted(geom.sec2zone.tolist()) == [0, 1, 2, 3]
    with pytest.raises(Exception, match="exactly three other zones"):
        build_geometry(DESIGN.cage, {"speaker": DESIGN.zones()["speaker"]}, (0.05, 0.1, 0.2))


@pytest.mark.parametrize("legacy", [False, True])
def test_flight_pipeline_runs_and_sees_arousal(sim, legacy) -> None:
    layout, report, cal = sim
    res = run_flight(layout, report.cache_dir, cal, legacy=legacy)
    assert res.chunks_used > 0 and not res.problems
    cm = res.tables["flight_cycle_metrics"]
    assert set(cm["window"]) >= {"ON", "OFF"}
    st = res.tables["stats_flight_on_vs_off"]
    assert {"p_on_gt_off", "delta", "zone_label"} <= set(st.columns)
    on = cm[cm.window == "ON"].set_index("hour")["flyers_per_frame"]
    off = cm[cm.window == "OFF"].set_index("hour")["flyers_per_frame"]
    both = on.index.intersection(off.index)
    assert len(both) >= 1 and np.all(on[both] > off[both])  # planted arousal during the tone
