"""Experiment comparison: tuning tables, confound notes and group contrasts."""

from __future__ import annotations

import numpy as np
import pytest

from buzzphono.metrics.comparison import confounds, experiment_table, group_contrast


def _rows():
    return [
        {
            "label": "a",
            "frequency_hz": 475,
            "species": "Aedes",
            "group": "x",
            "mean_delta": 0.07,
            "p_value": 0.01,
        },
        {
            "label": "b",
            "frequency_hz": 475,
            "species": "Aedes",
            "group": "x",
            "mean_delta": 0.05,
            "p_value": 0.02,
        },
        {
            "label": "c",
            "frequency_hz": 475,
            "species": "Aedes",
            "group": "x",
            "mean_delta": 0.06,
            "p_value": 0.03,
        },
        {
            "label": "d",
            "frequency_hz": 510,
            "species": "Tbambusa",
            "group": "y",
            "mean_delta": 0.01,
            "p_value": 0.2,
        },
        {
            "label": "e",
            "frequency_hz": 510,
            "species": "Tbambusa",
            "group": "y",
            "mean_delta": 0.02,
            "p_value": 0.4,
        },
        {
            "label": "f",
            "frequency_hz": 510,
            "species": "Tbambusa",
            "group": "y",
            "mean_delta": 0.00,
            "p_value": 0.9,
        },
    ]


def test_table_sorted_with_bh() -> None:
    t = experiment_table(_rows())
    assert list(t["frequency_hz"]) == [475] * 3 + [510] * 3
    assert (t["q_bh"] >= t["p_value"]).all() and "sex" in t
    assert experiment_table([]).empty


def test_confounds_named() -> None:
    notes = confounds(experiment_table(_rows()))
    assert any("frequency_hz and species" in n for n in notes)
    single = experiment_table(_rows()[:1] + _rows()[3:4])
    assert any("single experiment" in n for n in confounds(single))


def test_group_contrast_exact_permutation() -> None:
    g = group_contrast(experiment_table(_rows()), "group")
    row = g.iloc[0]
    assert row["difference"] == pytest.approx(0.01 - 0.06)
    assert row["p_permutation"] == pytest.approx(2 / 20)  # the two most extreme of C(6,3) splits
    small = group_contrast(experiment_table(_rows()[:2] + _rows()[3:4]), "group")
    assert np.isnan(small.iloc[0]["p_permutation"]) and "not tested" in small.iloc[0]["note"]
