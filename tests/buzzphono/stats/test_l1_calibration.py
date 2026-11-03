"""Do the tests keep their promises on data with known truth? (docs/STATS_PLAN.md §9)

Fixed seeds keep these deterministic in CI. The weekly slow job reruns them with 5,000 seeds.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from buzzphono import stats
from buzzphono.synthetic.l1 import L1Design, simulate_cycles

N_SEEDS = int(os.environ.get("BUZZPHONO_L1_SEEDS", "300"))
ALPHA = 0.05


def _rejections(design: L1Design, n_seeds: int, offset: int = 0) -> dict[str, float]:
    hits = {"wilcoxon": 0, "block": 0, "wild": 0}
    for seed in range(offset, offset + n_seeds):
        df = simulate_cycles(design, seed)
        d = (df.on - df.off).to_numpy()
        blocks = stats.blocks_within_clusters(df.cage_day.to_numpy(), df.hour.to_numpy())
        hits["wilcoxon"] += stats.wilcoxon_signed_rank(d).p_value < ALPHA
        hits["block"] += stats.block_signflip_test(d, blocks, n_resamples=1999, seed=seed).p_value < ALPHA
        hits["wild"] += (
            stats.wild_cluster_bootstrap(d, df.cage_day.to_numpy(), n_resamples=1999, seed=seed).p_value
            < ALPHA
        )
    return {k: v / n_seeds for k, v in hits.items()}


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6])
def test_false_positive_rates_with_serial_correlation(rho) -> None:
    rates = _rejections(L1Design(rho=rho), N_SEEDS)
    assert rates["wild"] <= 0.08, rates
    assert rates["block"] <= 0.08, rates
    if rho == 0.0:
        assert rates["wilcoxon"] <= 0.08, rates
    if rho == 0.6:
        # Documents the pseudo-replication problem: treating correlated hours as independent
        # rejects a true null far too often. This is why the plan relies on the cluster methods.
        assert rates["wilcoxon"] > 0.08, rates


def test_day_level_effects_need_the_cluster_bootstrap() -> None:
    """A random cage-day shift in Δ with zero mean: only the cage-day cluster bootstrap stays valid."""
    rates = _rejections(L1Design(sd_day_delta=0.02), N_SEEDS, offset=10_000)
    assert rates["wild"] <= 0.08, rates
    assert rates["wilcoxon"] > rates["wild"], rates


def test_interval_coverage() -> None:
    design = L1Design(
        base={"dawn": 0.3, "day": 0.3, "dusk": 0.3, "night": 0.3},
        effect={"dawn": 0.03, "day": 0.03, "dusk": 0.03, "night": 0.03},
        zero_inflation=0.0,
        sd_day_delta=0.01,
        rho=0.3,
    )
    n = min(N_SEEDS, 200)
    covered = 0
    for seed in range(20_000, 20_000 + n):
        df = simulate_cycles(design, seed)
        r = stats.wild_cluster_bootstrap(
            (df.on - df.off).to_numpy(), df.cage_day.to_numpy(), n_resamples=1999, seed=seed
        )
        covered += r.ci_low <= 0.03 <= r.ci_high
    assert 0.90 <= covered / n <= 0.99, covered / n


def test_power_grows_with_the_effect() -> None:
    small = _rejections(L1Design.uniform_effect(0.01), 60, offset=30_000)
    large = _rejections(L1Design.uniform_effect(0.05), 60, offset=30_000)
    assert large["wild"] > small["wild"] and large["wild"] >= 0.8


@pytest.mark.slow
def test_false_positive_rate_many_seeds() -> None:
    rates = _rejections(L1Design(rho=0.3), 5000, offset=50_000)
    assert rates["wild"] <= 0.06 and rates["block"] <= 0.065, rates
    assert np.isfinite(list(rates.values())).all()
