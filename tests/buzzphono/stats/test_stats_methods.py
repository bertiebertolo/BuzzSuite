from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from buzzphono import stats


def _brute_force_p(d: np.ndarray, alternative: str) -> float:
    """Exact sign-flip p-value of R+ by enumerating every sign pattern (small n only)."""
    nz = d[d != 0]
    r = stats.average_ranks(np.abs(nz))
    obs = r[nz > 0].sum()
    sums = np.array(
        [sum(r[i] for i in range(nz.size) if s[i]) for s in itertools.product([0, 1], repeat=nz.size)]
    )
    g, le = np.mean(sums >= obs - 1e-9), np.mean(sums <= obs + 1e-9)
    return {"greater": g, "less": le, "two-sided": min(1.0, 2 * min(g, le))}[alternative]


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("seed", range(6))
def test_exact_wilcoxon_with_ties_matches_enumeration(alternative, seed) -> None:
    rng = np.random.default_rng(seed)
    d = np.round(rng.normal(0.3, 1, size=11), 0)  # many ties and some zeros
    if not (d != 0).any():
        d[0] = 1.0
    got = stats.wilcoxon_signed_rank(d, alternative)
    assert got.method == "exact"
    assert got.p_value == pytest.approx(_brute_force_p(d, alternative), rel=1e-12)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_exact_wilcoxon_without_ties_matches_scipy(alternative) -> None:
    rng = np.random.default_rng(3)
    d = rng.normal(0.2, 1, size=30)
    ours = stats.wilcoxon_signed_rank(d, alternative)
    ref = sps.wilcoxon(d, alternative=alternative, method="exact")
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10)


def test_large_sample_uses_normal_approximation_like_scipy() -> None:
    rng = np.random.default_rng(4)
    d = np.round(rng.normal(0.1, 1, size=200), 1)
    d = d[d != 0]
    ours = stats.wilcoxon_signed_rank(d, "two-sided")
    ref = sps.wilcoxon(d, method="approx", correction=False)
    assert ours.method == "approx"
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-9)


def test_wilcoxon_all_zero() -> None:
    r = stats.wilcoxon_signed_rank(np.zeros(5))
    assert r.n == 0 and r.n_zero == 5 and np.isnan(r.p_value)


def test_effect_sizes() -> None:
    off = np.array([0.1, 0.2, 0.0, 0.3])
    on = np.array([0.3, 0.2, 0.1, 0.6])
    e = stats.paired_effect_sizes(off, on)
    d = on - off
    assert e["mean_delta"] == pytest.approx(d.mean())
    assert e["fold_change"] == pytest.approx(on.mean() / off.mean())
    assert e["dz"] == pytest.approx(d.mean() / d.std(ddof=1))
    walsh = [(d[i] + d[j]) / 2 for i in range(4) for j in range(i, 4)]
    assert e["hodges_lehmann"] == pytest.approx(np.median(walsh))
    assert e["rank_biserial"] == 1.0  # every non-zero difference is positive


def test_multiplicity_matches_statsmodels() -> None:
    p = np.array([0.001, 0.04, 0.03, 0.2, np.nan, 0.01])
    ok = np.isfinite(p)
    h = stats.holm(p)
    b = stats.benjamini_hochberg(p)
    assert np.isnan(h[4]) and np.isnan(b[4])
    assert np.allclose(h[ok], multipletests(p[ok], method="holm")[1])
    assert np.allclose(b[ok], multipletests(p[ok], method="fdr_bh")[1])


def test_block_signflip_exact_small_case() -> None:
    d = np.array([1.0, 1.0, 2.0, -0.5])
    blocks = np.array([0, 0, 1, 2])
    r = stats.block_signflip_test(d, blocks)
    assert r.method == "exact" and r.n_blocks == 3 and r.n_resamples == 8
    # block sums 2, 2, -0.5; |mean| >= |observed| = 3.5/4 for patterns (+,+,-),(+,+,+),(-,-,+),(-,-,-)
    assert r.p_value == pytest.approx(4 / 8)


def test_block_signflip_monte_carlo_agrees_with_exact() -> None:
    rng = np.random.default_rng(5)
    d = rng.normal(0.2, 1, 64)
    blocks = np.repeat(np.arange(16), 4)
    exact = stats.block_signflip_test(d, blocks, n_resamples=70_000)
    mc = stats.block_signflip_test(d, blocks, n_resamples=20_000, seed=1)
    assert exact.method == "exact" and mc.method == "monte-carlo"
    assert mc.p_value == pytest.approx(exact.p_value, abs=0.01)


def test_wild_cluster_bootstrap_basics() -> None:
    rng = np.random.default_rng(6)
    cl = np.repeat(np.arange(8), 24)
    d = 0.05 + rng.normal(0, 0.05, cl.size) + np.repeat(rng.normal(0, 0.02, 8), 24)
    r = stats.wild_cluster_bootstrap(d, cl)
    assert r.n_clusters == 8 and r.ci_low < d.mean() < r.ci_high
    assert r.p_value < 0.05
    assert any("approximate" in w for w in r.warnings)
    one = stats.wild_cluster_bootstrap(d, np.zeros_like(cl))
    assert np.isnan(one.p_value) and one.method.startswith("undefined")


def test_cluster_bootstrap_ci_contains_estimate() -> None:
    rng = np.random.default_rng(7)
    clusters = [rng.normal(1.0, 0.3, 20) for _ in range(10)]
    lo, hi = stats.cluster_bootstrap_ci(clusters, lambda v: float(np.median(v)), n_resamples=999)
    assert lo < float(np.median(np.concatenate(clusters))) < hi


def test_lag_autocorrelation_recovers_ar1() -> None:
    rng = np.random.default_rng(8)
    rho, n_days = 0.6, 400
    d, cl, hr = [], [], []
    for k in range(n_days):
        e = rng.normal()
        for h in range(24):
            e = rho * e + np.sqrt(1 - rho**2) * rng.normal() if h else e
            d.append(e)
            cl.append(k)
            hr.append(h)
    ac = stats.lag_autocorrelation(np.array(d), np.array(cl), np.array(hr))
    assert ac[0] == pytest.approx(rho, abs=0.05) and ac[1] == pytest.approx(rho**2, abs=0.05)


def test_mixed_model_recovers_mean_and_flags_singular_fit() -> None:
    rng = np.random.default_rng(9)
    cd = np.repeat(np.arange(12), 24)
    d = 0.04 + np.repeat(rng.normal(0, 0.02, 12), 24) + rng.normal(0, 0.03, cd.size)
    m = stats.mixed_model(d, cd)
    assert m.converged and m.ci_low < 0.04 < m.ci_high and not m.singular
    flat = stats.mixed_model(0.04 + rng.normal(0, 0.03, cd.size), cd)
    assert flat.var_cage_day < m.var_cage_day


def test_summarise_paired_runs_everything() -> None:
    from buzzphono.synthetic.l1 import L1Design, simulate_cycles

    df = simulate_cycles(L1Design.uniform_effect(0.03), seed=1)
    s = stats.summarise_paired(
        df.off, df.on, df.cage_day, df.hour, cage=df.cage, phase=df.phase, n_resamples=999
    )
    assert {"effect", "wilcoxon", "block_signflip", "wild_cluster", "median_delta_ci", "mixed_model"} <= set(
        s
    )
    assert s["wild_cluster"].n_clusters == 6
