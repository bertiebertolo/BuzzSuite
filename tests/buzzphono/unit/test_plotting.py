"""Figures: each builds from a toy table, files are deterministic, and flags survive a CSV round trip."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from buzzphono.plotting import static
from buzzphono.plotting.style import SERIES, save

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)


def test_sensitivity_flags_after_csv_round_trip() -> None:
    t = pd.DataFrame(
        {
            "family": ["baseline", "zone shift", "colony cap"],
            "variant": ["b", "+40 px", "25"],
            "mean_delta": [0.1, -0.01, 0.11],
            "ci_low": [0.05, -0.03, 0.06],
            "ci_high": [0.15, 0.01, 0.16],
            "flag": ["", "sign flips", ""],
        }
    )
    buf = io.StringIO()
    t.to_csv(buf, index=False)
    back = pd.read_csv(io.StringIO(buf.getvalue()))  # empty flags come back as NaN
    fig = static.sensitivity_plot(back)
    reds = [tk for tk in fig.axes[0].get_yticklabels() if tk.get_color() == SERIES[7]]
    assert len(reds) == 1 and "flagged" in reds[0].get_text()


def test_every_figure_builds(tmp_path) -> None:
    cycles = pd.DataFrame({"off": [0.1, 0.2, 0.15], "on": [0.3, 0.25, 0.4]})
    stats = pd.DataFrame(
        {
            "phase": ["all", "day"],
            "mean_delta": [0.1, 0.2],
            "wild_cluster_ci_low": [0.0, np.nan],
            "wild_cluster_ci_high": [0.2, np.nan],
            "n": [10, 5],
        }
    )
    rel = np.arange(-24, 40) * 30.0
    psth = pd.concat(
        [
            pd.DataFrame(
                {
                    "event": e,
                    "phase": "all",
                    "quantity": "colony_frac_in_zone",
                    "rel_time_s": rel,
                    "mean": 0.1,
                    "ci_low": 0.05,
                    "ci_high": 0.15,
                }
            )
            for e in ("onset", "sham_onset")
        ]
    )
    zt = pd.DataFrame({"zt_hour": range(24), "delta_mean": 0.1, "delta_ci_low": 0.0, "delta_ci_high": 0.2})
    minutes = pd.DataFrame(
        {
            "minute_start": pd.date_range("2026-01-01", periods=120, freq="min").strftime("%Y-%m-%d %H:%M"),
            "window": ["on" if i % 60 >= 40 and i % 60 < 50 else "" for i in range(120)],
            "colony_frac_in_zone": np.linspace(0, 1, 120),
        }
    )
    maps = {"on": np.random.default_rng(0).random((4, 5)), "off": np.random.default_rng(1).random((4, 5))}
    radial = pd.DataFrame(
        {
            "distance_from_px": [0, 20],
            "distance_to_px": [20, 40],
            "on_per_frame_per_1000px2": [1.0, 0.5],
            "off_per_frame_per_1000px2": [0.5, 0.5],
        }
    )
    tuning = pd.DataFrame(
        {
            "frequency_hz": [300, 475],
            "mean_delta": [0.01, 0.07],
            "ci_low": [0.0, 0.03],
            "ci_high": [0.02, 0.1],
            "label": ["a", "b"],
        }
    )
    plan = pd.DataFrame({"cage_days": [4, 8], "mde": [0.05, 0.03]})
    flight = pd.DataFrame(
        {
            "pair": "ON vs OFF",
            "zone": "all",
            "metric": ["flyers_per_frame", "takeoffs_per_min"],
            "grouping": "all",
            "group": "all",
            "fold": [2.0, 1.5],
        }
    )
    figs = [
        static.paired_on_off(cycles, "off", "on", ylabel="y"),
        static.delta_by_group(stats, ylabel="y"),
        static.psth_plot(psth, "colony_frac_in_zone"),
        static.zt_profile_plot(zt, ylabel="y"),
        static.minute_series_plot(minutes),
        static.occupancy_maps_plot(maps, 10.0, speaker=SQUARE, cage=SQUARE * 4),
        static.radial_profile_plot(radial),
        static.tuning_plot(tuning),
        static.planning_plot(plan, observed=0.04),
        static.flight_summary_plot(flight),
    ]
    for i, fig in enumerate(figs):
        paths = save(fig, tmp_path / f"f{i}", ["png", "svg"])
        assert all(p.stat().st_size > 1000 for p in paths)


def test_svg_and_pdf_are_deterministic(tmp_path) -> None:
    cycles = pd.DataFrame({"off": [0.1, 0.2], "on": [0.3, 0.25]})
    a = save(static.paired_on_off(cycles, "off", "on", ylabel="y"), tmp_path / "a", ["svg", "pdf", "png"])
    b = save(static.paired_on_off(cycles, "off", "on", ylabel="y"), tmp_path / "b", ["svg", "pdf", "png"])
    for x, y in zip(a, b, strict=True):
        assert x.read_bytes() == y.read_bytes(), x.suffix
