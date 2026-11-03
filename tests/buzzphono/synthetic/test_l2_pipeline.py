"""End-to-end checks on simulated experiments with known truth (buzzphono.synthetic.l2)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from buzzphono.flight import camera_scale, fly_px_per_frame
from buzzphono.io.cache import CacheIndex, load_npz
from buzzphono.io.discover import discover
from buzzphono.io.ingest import ingest
from buzzphono.metrics.resting_fraction import RestingGrids, frame_counts, polygon_test
from buzzphono.stimulus import find_bursts, fit_clock
from buzzphono.synthetic.l2 import L2Design, simulate_experiment
from buzzphono.timebase import NS_PER_HOUR, ClockModel, CorrectedClock, build_sessions, frames_per_chunk

CLEAN = L2Design(hours=6, miss_prob=0, specks=0, fragment_mean_s=0, warmup_s=0, jump_prob=0)


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("l2clean")
    truth = simulate_experiment(CLEAN, seed=11, out_dir=out / "exp")
    layout = discover(out / "exp")
    report = ingest(layout, tiers=("resting", "moving", "tracks"), cache_root=out / "cache", workers=1)
    return truth, layout, report


def test_ingest_reads_every_simulated_file(clean_run) -> None:
    truth, layout, report = clean_run
    assert not report.errors and report.built == 3 * len(layout.chunks)
    assert len(layout.chunks) == int(np.ceil(truth.n_frames / CLEAN.nf_full))


def test_frame_counts_come_from_the_tracker_frame_list(clean_run) -> None:
    truth, layout, report = clean_run
    index = CacheIndex.open(report.cache_dir, layout.root)
    nf = frames_per_chunk(index.data["chunks"])
    full = [n for n, src in nf.values()]
    assert {src for _, src in nf.values()} == {"tracker frame list"}
    assert max(full) == CLEAN.nf_full and sum(full) == truth.n_frames


def test_headline_metric_equals_the_truth_exactly(clean_run) -> None:
    """With no artefacts and the true clock, the measured ON/OFF fractions are the true ones."""
    truth, layout, report = clean_run
    index = CacheIndex.open(report.cache_dir, layout.root)
    sessions = build_sessions(
        [c.name for c in layout.chunks], frames_per_chunk(index.data["chunks"]), fps=CLEAN.fps
    )
    (name,) = sessions
    model = ClockModel({name: truth.clock_offset_s}, truth.drift_s_per_day, {name: "true"}, True)
    clock = CorrectedClock(sessions, model)
    first = int(truth.cycles["cycle_start_ns"].min())
    last = int(truth.cycles["cycle_start_ns"].max()) + NS_PER_HOUR
    grids = RestingGrids.empty(first, last, CLEAN.colony)
    inside = polygon_test(CLEAN.zones()["speaker"])
    for c in layout.chunks:
        counts = frame_counts(load_npz(report.cache_dir / "resting" / f"{c.chunk_id}.npz"), inside)
        grids.add(counts, clock.frame_ns(c.name, counts.frames), first, last)
    off = grids.in_zone[:, 28:38].sum(1) / grids.total[:, 28:38].sum(1)
    on = grids.in_zone[:, 40:50].sum(1) / grids.total[:, 40:50].sum(1)
    cyc = ((truth.cycles["cycle_start_ns"] - first) // NS_PER_HOUR).to_numpy()
    assert np.array_equal(off[cyc], truth.cycles["off_frac"].to_numpy())
    assert np.array_equal(on[cyc], truth.cycles["on_frac"].to_numpy())
    assert (truth.cycles["on_frac"] > truth.cycles["off_frac"]).mean() > 0.6  # the planted effect is there


def test_clock_fit_recovers_the_drift(tmp_path) -> None:
    design = L2Design(
        hours=24, drift_s_per_day=50.0, first_frame_delay_s=12.0, start=datetime(2026, 1, 10, 12, 0, 9)
    )
    truth = simulate_experiment(design, seed=5, out_dir=tmp_path / "exp", tiers=("moving",))
    layout = discover(tmp_path / "exp")
    report = ingest(layout, tiers=("moving",), cache_root=tmp_path / "cache", workers=1)
    index = CacheIndex.open(report.cache_dir, layout.root)
    nf = frames_per_chunk(index.data["chunks"])
    sessions = build_sessions([c.name for c in layout.chunks], nf, fps=design.fps)
    (name,) = sessions
    s = camera_scale(np.asarray(design.cage))
    activity = {
        name: {
            c.name.index: fly_px_per_frame(
                load_npz(report.cache_dir / "moving" / f"{c.chunk_id}.npz"),
                sessions[name].nf[c.name.index],
                s,
                fps=design.fps,
            )
            for c in layout.chunks
        }
    }
    fit = fit_clock(find_bursts(sessions, activity), list(sessions))
    assert fit.model.fitted, fit.model.message
    assert fit.model.n_kept >= 20 and fit.model.residual_sd_s < 3.0
    assert fit.model.drift_s_per_day == pytest.approx(truth.drift_s_per_day, abs=5.0)
    # The burst fit anchors the clock on the response, so the offset is the camera's first-frame
    # delay plus the response latency: latency analyses are relative to the fitted onset.
    latency = design.effect_latency_s
    assert fit.model.offset_s[name] == pytest.approx(truth.clock_offset_s + latency, abs=3.0)


def test_no_tone_means_no_clock_fit(tmp_path) -> None:
    design = L2Design(hours=24, recruitment=1, arousal=1, directed=1, startle=0)
    simulate_experiment(design, seed=6, out_dir=tmp_path / "exp", tiers=("moving",))
    layout = discover(tmp_path / "exp")
    report = ingest(layout, tiers=("moving",), cache_root=tmp_path / "cache", workers=1)
    index = CacheIndex.open(report.cache_dir, layout.root)
    sessions = build_sessions(
        [c.name for c in layout.chunks], frames_per_chunk(index.data["chunks"]), fps=design.fps
    )
    (name,) = sessions
    s = camera_scale(np.asarray(design.cage))
    activity = {
        name: {
            c.name.index: fly_px_per_frame(
                load_npz(report.cache_dir / "moving" / f"{c.chunk_id}.npz"),
                sessions[name].nf[c.name.index],
                s,
                fps=design.fps,
            )
            for c in layout.chunks
        }
    }
    fit = fit_clock(find_bursts(sessions, activity), list(sessions))
    assert not fit.model.fitted and fit.model.drift_s_per_day == 0.0
    assert "nominal clock" in fit.model.message
