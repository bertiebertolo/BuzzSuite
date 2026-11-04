"""Hand-computed toy cases for the window analyses (cycle table, attraction index, PSTH, ZT profile)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from buzzphono.frames import BIN_NS, Binned, resting_frame_counts
from buzzphono.metrics.attraction_index import INDEX_COLUMN, add_attraction_index, attraction_summary
from buzzphono.metrics.cycles import Coverage, cycle_table, flatten_summary, paired_comparison, usable
from buzzphono.metrics.psth import bin_quantities, cluster_band, event_matrix, psth
from buzzphono.metrics.zt_profile import hour_of_day_profile, minute_series
from buzzphono.timebase import NS_PER_HOUR, NS_PER_MIN, to_ns

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
FAR = SQUARE + 100
T0 = to_ns(datetime(2026, 1, 1, 0, 0))
FPS = 2.0
FRAMES_PER_BIN = int(BIN_NS / 1e9 * FPS)


def toy(hours: int = 4, on_in: int = 6, off_in: int = 3, colony: int = 10) -> Binned:
    """Every frame has 12 resters (capped at 10); ``off_in``/``on_in`` of them on the speaker."""
    b = Binned.empty(T0, T0 + hours * NS_PER_HOUR, {"speaker": SQUARE, "far": FAR}, colony)
    b.recorded[:] = FRAMES_PER_BIN
    b.rest_frames[:] = FRAMES_PER_BIN
    b.raw_total[:] = 12 * FRAMES_PER_BIN
    b.cap_total[:] = colony * FRAMES_PER_BIN
    minute = (np.arange(b.n_bins) * BIN_NS // NS_PER_MIN) % 60
    n_in = np.where((minute >= 40) & (minute < 50), on_in, off_in)
    b.raw_in[:, 0] = n_in * FRAMES_PER_BIN
    b.cap_in[:, 0] = n_in * FRAMES_PER_BIN
    b.frac_sum[:, 0] = n_in / colony * FRAMES_PER_BIN
    b.raw_in[:, 1] = 1 * FRAMES_PER_BIN
    b.cap_in[:, 1] = 1 * FRAMES_PER_BIN
    b.frac_sum[:, 1] = 1 / colony * FRAMES_PER_BIN
    b.fly_px[:] = np.where((minute >= 40) & (minute < 41), 500.0, 50.0)
    b.moving_total[:] = 3 * FRAMES_PER_BIN
    return b


WINDOWS = {"on": (40, 50), "off": (28, 38)}


def test_cycle_table_hand_values() -> None:
    c = cycle_table(toy(), WINDOWS, fps=FPS, cage="toy", phases=dict.fromkeys(range(24), "night"))
    assert len(c) == 4 and (c["hour_index"] == np.arange(4)).all()
    assert np.allclose(c["on_colony_frac_in_zone"], 0.6) and np.allclose(c["off_colony_frac_in_zone"], 0.3)
    assert np.allclose(c["on_resting_frac_in_zone"], 0.5) and np.allclose(c["off_resting_frac_in_zone"], 0.25)
    assert np.allclose(c["on_mean_frame_frac"], 0.6) and np.allclose(c["on_in_zone_per_frame"], 6)
    assert np.allclose(c["on_colony_frac_in_zone__far"], 0.1)
    assert c["on_ok"].all() and (c["on_minutes_with_resting"] == 10).all()
    assert (c["cage_day"] == "toy 2026-01-01").all() and (c["phase"] == "night").all()
    assert (c["stimulus_status"] == "not checked (no playback log)").all()


def test_coverage_rule_and_empty_hours() -> None:
    b = toy()
    b.recorded[2 * 120 : 3 * 120] = 0  # hour 2 not recorded at all: no row
    b.cap_total[3 * 120 + 80 : 3 * 120 + 100] = 0  # hour 3: no resting during ON
    c = cycle_table(b, WINDOWS, fps=FPS, cage="toy")
    assert list(c["hour_index"]) == [0, 1, 3]
    assert list(usable(c, ("on", "off"))) == [True, True, False]
    c = cycle_table(b, WINDOWS, fps=FPS, cage="toy", coverage=Coverage(enabled=False))
    assert not c["on_ok"].iloc[-1]  # no resting at all is never usable


def test_resting_frame_counts_merge_runs_and_exceptions() -> None:
    resting = {
        "run_f0": np.array([0, 2]),
        "run_len": np.array([3, 2]),
        "run_xy": np.array([[5.0, 5.0], [50.0, 50.0]]),
        "exc_frame": np.array([1.0, 7.0]),
        "exc_xy": np.array([[5.0, 5.0], [5.0, 5.0]]),
    }
    f, t, i = resting_frame_counts(resting, [SQUARE, FAR])
    assert list(f) == [0, 1, 2, 3, 7]
    assert list(t) == [1, 2, 2, 1, 1]
    assert list(i[:, 0]) == [1, 2, 1, 0, 1] and list(i[:, 1]) == [0, 0, 0, 0, 0]


def test_paired_comparison_and_flatten() -> None:
    rng = np.random.default_rng(1)
    n = 48
    df = pd.DataFrame(
        {
            "cage": "a",
            "cage_day": np.repeat(["a d1", "a d2"], n // 2),
            "hour_index": np.arange(n),
            "phase": np.tile(["day", "night"], n // 2),
            "off": rng.random(n),
        }
    )
    df["on"] = df["off"] + 0.2
    t = paired_comparison(df, "off", "on", label="x", n_resamples=199, with_mixed_model=False)
    assert list(t["phase"]) == ["all", "day", "night"]
    pooled = t.iloc[0]
    assert pooled["mean_delta"] == pytest.approx(0.2) and pooled["wilcoxon_p_value"] < 1e-6
    assert "wilcoxon_p_holm_phases" in t and np.isnan(pooled["wilcoxon_p_holm_phases"])
    assert flatten_summary({"effect": {"n": 3}, "w": (1, 2)}) == {"n": 3, "w": "1; 2"}


def test_attraction_index() -> None:
    df = pd.DataFrame({"on_colony_frac_in_zone": [0.6, 0.0, 0.2], "off_colony_frac_in_zone": [0.2, 0.0, 0.6]})
    out = add_attraction_index(df)
    assert out[INDEX_COLUMN].iloc[0] == pytest.approx(0.5)
    assert np.isnan(out[INDEX_COLUMN].iloc[1]) and out[INDEX_COLUMN].iloc[2] == pytest.approx(-0.5)
    c = add_attraction_index(cycle_table(toy(hours=8), WINDOWS, fps=FPS, cage="toy"))
    assert np.allclose(c[INDEX_COLUMN], (0.6 - 0.3) / 0.9)
    s = attraction_summary(c, n_resamples=199, with_mixed_model=False)
    assert s.iloc[0]["n_cycles_index_undefined"] == 0 and "fold_change" not in s


def test_psth_shapes_and_values() -> None:
    b = toy(hours=12)  # enough cage-days for the sign-flip test to reach p < 0.05
    q = bin_quantities(b, FPS)
    assert np.allclose(q["colony_frac_in_zone"][80:100], 0.6) and np.allclose(q["moving_per_frame"], 3)
    onsets = np.array([T0 + NS_PER_HOUR + 40 * NS_PER_MIN])
    mat, rel = event_matrix(q["fly_px_per_s"], b, onsets, 12, 20)
    assert mat.shape == (1, 64) and rel[0] == -720 and rel[24] == 0
    assert mat[0, 24] == pytest.approx(500 / FRAMES_PER_BIN * FPS)
    c = cycle_table(b, WINDOWS, fps=FPS, cage="toy")
    c["cage_day"] = [f"d{i}" for i in range(len(c))]
    table = psth(b, c, fps=FPS, n_resamples=199, by_phase=False)
    on = table[(table["event"] == "onset") & (table["quantity"] == "colony_frac_in_zone")]
    assert np.allclose(on.loc[on["rel_time_s"].between(0, 590), "mean"], 0.6)
    assert set(table["event"]) == {"onset", "sham_onset"}
    assert (on.loc[on["rel_time_s"].between(0, 590), "q_bh"] < 0.05).all()


def test_cluster_band_needs_two_clusters() -> None:
    mean, low, high = cluster_band(np.array([[1.0, np.nan], [3.0, 2.0]]), np.array(["a", "a"]))
    assert list(mean) == [2.0, 2.0] and np.isnan(low).all()
    mean, low, high = cluster_band(np.array([[1.0], [3.0]]), np.array(["a", "b"]), n_resamples=200)
    assert mean[0] == 2.0 and 1.0 <= low[0] <= high[0] <= 3.0


def test_zt_profile_and_minute_series() -> None:
    b = toy(hours=26)
    c = cycle_table(b, WINDOWS, fps=FPS, cage="toy")
    prof = hour_of_day_profile(c, lights_on_hour=5, n_resamples=199)
    assert len(prof) == 24 and set(prof["zt_hour"]) == set(range(24))
    assert np.allclose(prof["delta_mean"], 0.3)
    assert prof.loc[prof["hour"] == 0, "n_cycles"].item() == 2
    ms = minute_series(b, fps=FPS, windows=WINDOWS)
    assert len(ms) == 26 * 60 and set(ms["window"]) == {"", "on", "off"}
    assert np.allclose(ms.loc[ms["window"] == "on", "colony_frac_in_zone"], 0.6)
