"""Binned counts and hour-cycle tables against the simulator's ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from buzzphono.frames import ExperimentClock, bin_experiment, experiment_clock
from buzzphono.io.cache import CacheIndex
from buzzphono.io.discover import discover
from buzzphono.io.ingest import ingest
from buzzphono.metrics.cycles import Coverage, cycle_table, paired_comparison, usable
from buzzphono.synthetic.l2 import L2Design, simulate_experiment
from buzzphono.timebase import ClockModel, CorrectedClock, build_sessions, frames_per_chunk

CLEAN = L2Design(hours=6, miss_prob=0, specks=0, fragment_mean_s=0, warmup_s=0, jump_prob=0)
WINDOWS = {"on": (40, 50), "off": (28, 38), "sham_on": (20, 30), "sham_off": (8, 18)}


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    truth = simulate_experiment(CLEAN, seed=11, out_dir=out / "exp")
    layout = discover(out / "exp")
    report = ingest(layout, tiers=("resting", "moving", "tracks"), cache_root=out / "cache", workers=1)
    index = CacheIndex.open(report.cache_dir, layout.root)
    nf = frames_per_chunk(index.data["chunks"])
    sessions = build_sessions([c.name for c in layout.chunks], nf, fps=CLEAN.fps)
    (name,) = sessions
    model = ClockModel({name: truth.clock_offset_s}, truth.drift_s_per_day, {name: "true"}, True)
    eclock = ExperimentClock(CorrectedClock(sessions, model), sessions, None, nf)
    return truth, layout, report, eclock


def test_every_recorded_frame_is_counted_once(clean) -> None:
    truth, layout, report, eclock = clean
    b = bin_experiment(layout, report.cache_dir, eclock, CLEAN.zones(), colony=CLEAN.colony)
    assert b.recorded.sum() == truth.n_frames
    assert b.chunks_used == len(layout.chunks) and not b.chunks_skipped
    assert b.zones[0] == "speaker" and np.all(b.zone_area_px > 0)


def test_cycle_table_equals_the_truth(clean) -> None:
    truth, layout, report, eclock = clean
    b = bin_experiment(layout, report.cache_dir, eclock, CLEAN.zones(), colony=CLEAN.colony)
    cycles = cycle_table(b, WINDOWS, fps=CLEAN.fps, cage="sim", coverage=Coverage(enabled=False))
    merged = cycles.merge(truth.cycles, on="cycle_start_ns")
    assert len(merged) == len(truth.cycles)
    assert np.array_equal(merged["on_colony_frac_in_zone"], merged["on_frac"])
    assert np.array_equal(merged["off_colony_frac_in_zone"], merged["off_frac"])
    # a drifting camera records slightly more or fewer frames per wall-clock minute than nominal
    assert np.allclose(merged["on_frame_fraction"], 1.0, atol=0.002) and (merged["on_max_gap_s"] == 0).all()


def test_coverage_rule_drops_partial_cycles(clean) -> None:
    _, layout, report, eclock = clean
    b = bin_experiment(layout, report.cache_dir, eclock, CLEAN.zones(), colony=CLEAN.colony)
    cycles = cycle_table(b, WINDOWS, fps=CLEAN.fps, cage="sim")
    ok = usable(cycles, ("on", "off"))
    assert ok.sum() >= len(cycles) - 2  # only the partial first/last hours can fail
    first_two = cycles["cycle_start_ns"].iloc[:2].to_numpy()
    for start in first_two:  # pretend minutes 30-45 of the first two hours were not recorded
        k = (int(start) - b.t0_ns) // b.bin_ns
        b.recorded[k + 60 : k + 90] = 0
    cycles = cycle_table(b, WINDOWS, fps=CLEAN.fps, cage="sim")
    hit = cycles["cycle_start_ns"].isin(first_two)
    assert not usable(cycles, ("on", "off"))[hit].any()
    assert (cycles.loc[hit, "on_max_gap_s"] == 300).all() and (cycles.loc[hit, "off_max_gap_s"] == 480).all()


def test_corrected_clock_from_bursts_and_paired_tests(tmp_path) -> None:
    design = L2Design(hours=24, drift_s_per_day=50.0, first_frame_delay_s=12.0)
    truth = simulate_experiment(design, seed=5, out_dir=tmp_path / "exp")
    layout = discover(tmp_path / "exp")
    report = ingest(layout, tiers=("resting", "moving", "tracks"), cache_root=tmp_path / "cache", workers=1)
    eclock = experiment_clock(layout, report.cache_dir, cage=np.asarray(design.cage), fps=design.fps)
    assert eclock.mode == "corrected" and eclock.fit is not None and eclock.fit.model.fitted
    assert eclock.clock.model.drift_s_per_day == pytest.approx(truth.drift_s_per_day, abs=5.0)
    b = bin_experiment(
        layout, report.cache_dir, eclock, design.zones(), colony=design.colony, cage=np.asarray(design.cage)
    )
    assert b.fly_px.sum() > 0 and b.moving_total.sum() > 0
    cycles = cycle_table(b, WINDOWS, fps=design.fps, cage="sim")
    cycles = cycles[usable(cycles, tuple(WINDOWS))]
    main = paired_comparison(
        cycles, "off_colony_frac_in_zone", "on_colony_frac_in_zone", label="on_vs_off", n_resamples=999
    )
    sham = paired_comparison(
        cycles, "sham_off_colony_frac_in_zone", "sham_on_colony_frac_in_zone", label="sham", n_resamples=999
    )
    pooled = main[main["phase"] == "all"].iloc[0]
    assert pooled["mean_delta"] > 0 and pooled["wilcoxon_p_value"] < 0.01
    assert abs(sham[sham["phase"] == "all"].iloc[0]["mean_delta"]) < 0.5 * pooled["mean_delta"]


def test_nominal_and_thesis_modes(clean) -> None:
    _, layout, report, _ = clean
    nominal = experiment_clock(layout, report.cache_dir, mode="nominal", fps=CLEAN.fps)
    assert nominal.mode == "corrected" and not nominal.clock.model.fitted
    thesis = experiment_clock(layout, report.cache_dir, mode="thesis", fps=CLEAN.fps)
    assert thesis.mode == "thesis" and "Thesis clock" in thesis.message
    manual = experiment_clock(
        layout,
        report.cache_dir,
        mode="nominal",
        fps=CLEAN.fps,
        manual_offset_s={next(iter(nominal.sessions)): 7.0},
    )
    assert set(manual.clock.model.source.values()) == {"manual offset"}
    with pytest.raises(Exception, match="not in this experiment"):
        experiment_clock(layout, report.cache_dir, mode="nominal", manual_offset_s={"nope": 1.0})
