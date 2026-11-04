"""Spatial maps, sample tracks and reliability on a simulated experiment with a planted effect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buzzphono.frames import bin_experiment, experiment_clock
from buzzphono.io.discover import discover
from buzzphono.io.ingest import ingest
from buzzphono.metrics.cycles import cycle_table, usable
from buzzphono.metrics.reliability import reliability, variance_components
from buzzphono.metrics.spatial import occupancy_maps, radial_profile, sample_tracks, window_labels
from buzzphono.synthetic.l2 import L2Design, simulate_experiment
from buzzphono.timebase import NS_PER_HOUR, NS_PER_MIN

WINDOWS = {"on": (40, 50), "off": (28, 38)}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("spatial")
    design = L2Design(hours=12, recruitment=4.0)
    simulate_experiment(design, seed=21, out_dir=out / "exp")
    layout = discover(out / "exp")
    report = ingest(layout, tiers=("resting", "moving", "tracks"), cache_root=out / "cache", workers=1)
    cage = np.asarray(design.cage, dtype=float)
    eclock = experiment_clock(layout, report.cache_dir, cage=cage, fps=design.fps)
    b = bin_experiment(layout, report.cache_dir, eclock, design.zones(), colony=design.colony, cage=cage)
    cycles = cycle_table(b, WINDOWS, fps=design.fps, cage="sim")
    starts = cycles.loc[usable(cycles, tuple(WINDOWS)), "cycle_start_ns"].to_numpy()
    return design, layout, report, eclock, b, starts


def test_window_labels() -> None:
    h = 5 * NS_PER_HOUR
    t = np.array(
        [h + 41 * NS_PER_MIN, h + 30 * NS_PER_MIN, h + 5 * NS_PER_MIN, 9 * NS_PER_HOUR + 41 * NS_PER_MIN]
    )
    assert list(window_labels(t, WINDOWS, np.array([h]))) == [0, 1, -1, -1]


def test_maps_count_every_resting_frame(run) -> None:
    design, layout, report, eclock, b, starts = run
    maps = occupancy_maps(layout, report.cache_dir, eclock, starts, WINDOWS, (design.width, design.height))
    # mean resting detections per frame across the map equals the binned total over the same frames
    sel = np.isin(b.bin_start_ns() - b.bin_start_ns() % NS_PER_HOUR, starts)
    minute = (b.bin_start_ns() // NS_PER_MIN) % 60
    on = sel & (minute >= 40) & (minute < 50)
    assert maps.frames["on"] == pytest.approx(b.recorded[on].sum())
    assert np.nansum(maps.maps["on"]) == pytest.approx(b.raw_total[on].sum() / b.recorded[on].sum())
    tidy = maps.tidy()
    assert set(tidy["window"]) == {"on", "off"} and len(tidy) == 2 * maps.maps["on"].size


def test_planted_recruitment_shows_at_the_speaker(run) -> None:
    design, layout, report, eclock, _, starts = run
    maps = occupancy_maps(layout, report.cache_dir, eclock, starts, WINDOWS, (design.width, design.height))
    prof = radial_profile(maps, design.zones()["speaker"], np.asarray(design.cage, dtype=float))
    near = prof.iloc[0]
    assert near["distance_from_px"] == 0
    assert near["on_per_frame_per_1000px2"] > near["off_per_frame_per_1000px2"]
    assert (prof["cage_area_px2"] > 0).all()


def test_sample_tracks_is_seeded(run) -> None:
    _, layout, report, eclock, _, starts = run
    a = sample_tracks(layout, report.cache_dir, eclock, starts, WINDOWS, per_window=5)
    b = sample_tracks(layout, report.cache_dir, eclock, starts, WINDOWS, per_window=5)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["window"]) <= {"on", "off"} and a.groupby("window")["track"].nunique().max() <= 5


def test_variance_components_recover_known_values() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for c in range(60):
        uc = rng.normal(0, 0.03)
        for d in range(4):
            ud = rng.normal(0, 0.02)
            for _ in range(int(rng.integers(8, 20))):
                rows.append((f"c{c}", f"d{d}", uc + ud + rng.normal(0, 0.04)))
    df = pd.DataFrame(rows, columns=["cage", "cage_day", "delta"])
    comp = variance_components(df["delta"].to_numpy(), df["cage"].to_numpy(), df["cage_day"].to_numpy())
    assert comp.var_cage == pytest.approx(0.03**2, rel=0.35)
    assert comp.var_cage_day == pytest.approx(0.02**2, rel=0.35)
    assert comp.var_cycle == pytest.approx(0.04**2, rel=0.1)
    assert comp.n_cages == 60 and comp.n_cage_days == 240


def test_reliability_tables_and_small_designs() -> None:
    df = pd.DataFrame(
        {
            "cage": ["a"] * 12,
            "cage_day": ["d1"] * 6 + ["d2"] * 6,
            "delta": [0.1, 0.12, 0.09, 0.11, 0.1, 0.13, 0.02, 0.01, 0.03, 0.02, 0.0, 0.01],
        }
    )
    summary, plan = reliability(df, n_resamples=99)
    row = summary.iloc[0]
    assert row["n_cages"] == 1 and "imprecise" in row["warnings"] and row["icc_cage_day"] > 0.8
    assert np.isfinite(row["mde_analytic"]) and (plan["mde"].diff().dropna() < 0).all()
    one = reliability(df.iloc[:2], n_resamples=20)[0].iloc[0]
    assert np.isnan(one["icc_cage_day"])


def test_sensitivity_grid(run) -> None:
    from buzzphono.metrics.sensitivity import (
        SensitivityPlan,
        grown_zone,
        run_sensitivity,
        shifted_zone,
        zone_area,
    )

    design, layout, report, eclock, _, _ = run
    speaker = design.zones()["speaker"]
    cage = np.asarray(design.cage, dtype=float)
    nominal = experiment_clock(layout, report.cache_dir, mode="nominal", fps=design.fps)
    plan = SensitivityPlan(margins_px=[-5.0, 10.0], shifts_px=[200.0], colonies=[25], walking_mm_s=[2.0])
    plan.extra_zones = {"wide": speaker + np.array([[-20, 0], [20, 0], [20, 0], [-20, 0]])}
    table = run_sensitivity(
        layout,
        report.cache_dir,
        {"corrected": eclock, "nominal": nominal},
        speaker=speaker,
        cage=cage,
        frame_size=(design.width, design.height),
        fps=design.fps,
        colony=design.colony,
        plan=plan,
        n_resamples=199,
    )
    assert table.iloc[0]["family"] == "baseline" and table.iloc[0]["relative_change"] == 0
    assert len(table) == 1 + 2 + 1 + 1 + 1 + 2 + 2 + 1 + 1 + 1
    base = table.iloc[0]["mean_delta"]
    assert base > 0
    small = table[table["variant"] == "-5 px"].iloc[0]
    assert np.sign(small["mean_delta"]) == np.sign(base)
    far = table[table["variant"] == "+200 px"].iloc[0]  # far from the speaker: the effect is gone
    assert abs(far["mean_delta"]) < 0.5 * base
    assert table["q_bh"].notna().all() and set(table["family"]) >= {"clock", "walking resters removed"}
    # zone helpers
    assert grown_zone(speaker, 0) is speaker
    assert zone_area(grown_zone(speaker, 5.0), (design.width, design.height)) > zone_area(
        speaker, (design.width, design.height)
    )
    moved = shifted_zone(speaker, cage, 10.0)
    assert np.allclose(np.hypot(*(moved - speaker).T), 10.0)


def test_planted_directed_flight_is_recovered(run) -> None:
    from buzzphono.metrics.approach import approach_analysis, approach_cycle_table, wide_by_window

    design, layout, report, eclock, b, starts = run
    zones = design.zones()
    controls = {k: v for k, v in zones.items() if k != "speaker"}
    res = approach_analysis(
        layout,
        report.cache_dir,
        eclock,
        starts,
        WINDOWS,
        speaker=zones["speaker"],
        controls=controls,
        cage=np.asarray(design.cage, dtype=float),
        fps=design.fps,
    )
    assert res.unit == "px" and res.jitter >= 2.0 and not res.steps.empty
    cycles = cycle_table(b, WINDOWS, fps=design.fps, cage="sim")
    table = approach_cycle_table(res, cycles, list(controls))
    assert {"directed_velocity_px_per_s", "bouts_per_flight_s", "validated_radial_velocity"} <= set(
        table.columns
    )
    wide = wide_by_window(table, "directed_velocity_px_per_s")
    delta = wide["on_directed_velocity_px_per_s"] - wide["off_directed_velocity_px_per_s"]
    assert delta.mean() > 0  # flyers head for the speaker during the tone (planted)
