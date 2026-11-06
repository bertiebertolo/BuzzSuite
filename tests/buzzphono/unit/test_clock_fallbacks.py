"""Clock fit fallbacks (rig drift, rejected fits, cross-fitting) and the flight-activity median filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buzzphono.flight import fly_px_per_frame
from buzzphono.stimulus import cross_fitted_models, describe_bursts, fit_clock
from buzzphono.timebase import NS_PER_HOUR


def _bursts(
    n: int, drift: float, offset: float = 5.0, sessions=("s1",), noise: float = 0.5, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        days = i / 24
        rows.append(
            {
                "session": sessions[i % len(sessions)],
                "hour_ns": 1_700_000_000 * 10**9 + i * NS_PER_HOUR,
                "onset_recording_s": 0.0,
                "offset_s": offset + drift * days + rng.normal(0, noise),
                "days": days,
                "peak_ratio": 5.0,
            }
        )
    return pd.DataFrame(rows)


def test_good_fit_and_described_bursts() -> None:
    fit = fit_clock(_bursts(30, 50.0), ["s1"])
    assert fit.model.fitted and fit.model.drift_s_per_day == pytest.approx(50.0, abs=1.0)
    table = describe_bursts(fit)
    assert "hour" in table and "hour_ns" not in table and table["kept"].all()
    assert describe_bursts(fit_clock(_bursts(0, 0.0), ["s1"])).empty


def test_too_few_bursts_use_the_rig_drift_or_the_nominal_clock() -> None:
    few = _bursts(4, 50.0)
    rig = fit_clock(few, ["s1"], rig_drift_s_per_day=51.0)
    assert not rig.model.fitted and rig.model.drift_s_per_day == 51.0
    assert (
        set(rig.model.source.values()) == {"rig drift profile"} and "rig's known drift" in rig.model.message
    )
    nominal = fit_clock(few, ["s1"])
    assert nominal.model.drift_s_per_day == 0.0 and "nominal clock" in nominal.model.message


def test_implausible_or_inconsistent_fits_are_rejected() -> None:
    wild = fit_clock(_bursts(30, 900.0), ["s1"])
    assert not wild.model.fitted and "implausible" in wild.model.message
    noisy = _bursts(30, 50.0, noise=0.5)
    noisy.loc[::2, "offset_s"] += np.random.default_rng(1).normal(0, 400, len(noisy.loc[::2]))
    rejected = fit_clock(noisy, ["s1"])
    assert not rejected.model.fitted and "residual SD" in rejected.model.message


def test_sessions_with_few_bursts_get_the_median_offset() -> None:
    b = pd.concat(
        [_bursts(20, 50.0, offset=5.0, sessions=("s1",)), _bursts(2, 50.0, offset=9.0, sessions=("s2",))]
    )
    fit = fit_clock(b, ["s1", "s2"])
    assert fit.model.fitted and fit.model.offset_s["s2"] == pytest.approx(fit.model.offset_s["s1"])
    assert "median offset" in fit.model.source["s2"]


def test_cross_fitting_uses_the_other_hours() -> None:
    models = cross_fitted_models(_bursts(40, 50.0), ["s1"])
    assert set(models) == {0, 1} and all(m.fitted for m in models.values())
    empty = cross_fitted_models(_bursts(0, 0.0), ["s1"])
    assert not empty[0].fitted and not empty[1].fitted


def test_fly_px_median_filter_drops_jittery_tracks() -> None:
    # track 0: steady fast flight; track 1: stationary with one big mis-link jump
    f0 = np.arange(20, dtype=float)
    xy0 = np.column_stack([np.arange(20) * 20.0, np.zeros(20)])
    f1 = np.arange(20, dtype=float)
    xy1 = np.zeros((20, 2))
    xy1[10:] = [400.0, 0.0]
    moving = {
        "det_obj": np.r_[np.zeros(20, np.int32), np.ones(20, np.int32)],
        "det_frame": np.r_[f0, f1],
        "det_xy": np.vstack([xy0, xy1]),
    }
    plain = fly_px_per_frame(moving, 30, scale=1.0)
    filtered = fly_px_per_frame(moving, 30, scale=1.0, min_median=3.0)
    assert plain.sum() == pytest.approx(19 * 20 + 400)
    assert filtered.sum() == pytest.approx(19 * 20)
    empty = {"det_obj": np.zeros(1, np.int32), "det_frame": np.zeros(1), "det_xy": np.zeros((1, 2))}
    assert fly_px_per_frame(empty, 5, 1.0).sum() == 0
