"""The thesis statistics must come out exactly as SciPy 1.10.1 computed them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from buzzphono.stats import average_ranks, wilcoxon_null_pmf, wilcoxon_scipy110

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "wilcoxon_scipy110.json").read_text()
)


def test_fixture_comes_from_scipy_1_10() -> None:
    assert FIXTURE["scipy"] == "1.10.1"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: f"n{len(c['x'])}-{c['alternative']}")
def test_matches_scipy_1_10(case) -> None:
    r = wilcoxon_scipy110(np.array(case["x"]), np.array(case["y"]), alternative=case["alternative"])
    assert r.statistic == case["statistic"]
    assert r.p_value == pytest.approx(case["p"], rel=1e-12, abs=1e-300)


def test_average_ranks() -> None:
    assert average_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_null_pmf_sums_to_one_and_is_symmetric() -> None:
    for n in (1, 5, 20):
        pmf = wilcoxon_null_pmf(n)
        assert pmf.sum() == pytest.approx(1.0)
        assert np.allclose(pmf, pmf[::-1])
        assert pmf.size == n * (n + 1) // 2 + 1


def test_all_zero_differences_are_refused() -> None:
    x = np.ones(60)
    with pytest.raises(ValueError, match="every difference is zero"):
        wilcoxon_scipy110(x, x)


def test_empty_input() -> None:
    r = wilcoxon_scipy110(np.array([]), np.array([]))
    assert np.isnan(r.p_value) and r.n == 0
