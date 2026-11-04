"""Statistics for paired ON/OFF comparisons of hour-cycles.

Everything is implemented directly on numpy (``scipy.special`` for the normal distribution,
statsmodels only for the mixed model), so results do not change when SciPy changes its defaults
between releases. The methods, and why each is there, are in docs/STATS_PLAN.md:

* :func:`wilcoxon_signed_rank`: Wilcoxon signed-rank test, exact (ties handled exactly) up to 50
  pairs, normal approximation with tie correction above. Treats cycles as independent.
* :func:`block_signflip_test`: permutation test that flips the sign of whole blocks of neighbouring
  cycles, so serial correlation within a cage-day does not inflate the evidence.
* :func:`wild_cluster_bootstrap`: confidence interval and test for the mean difference that treat
  each cage-day as one independent unit (wild bootstrap with Webb weights, good for few units).
* :func:`mixed_model`: linear mixed model with a random intercept per cage-day (sensitivity).
* effect sizes (:func:`paired_effect_sizes`) with cluster-bootstrap CIs, and Holm / Benjamini-
  Hochberg adjustment.

:func:`wilcoxon_scipy110` reproduces ``scipy.stats.wilcoxon`` from SciPy 1.10.1 exactly, as used for
the published (2026 thesis) results, including its automatic choice between the exact and the
normal-approximation p-value. Use it only to reproduce those numbers.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.special import ndtr

Alternative = Literal["two-sided", "greater", "less"]

ASSUMPTIONS: dict[str, str] = {
    "wilcoxon": (
        "Wilcoxon signed-rank: pairs (hour-cycles) independent and differences symmetric about their "
        "median under the null. Neighbouring hours of one cage-day are correlated, so this p-value can "
        "overstate the evidence; read it with the block and cluster results."
    ),
    "block_signflip": (
        "Block sign-flip permutation: under the null the ON/OFF labels are exchangeable, so the sign of "
        "a block's differences can flip; blocks of neighbouring cycles within a cage-day flip together "
        "and blocks are independent of each other."
    ),
    "wild_cluster": (
        "Wild cluster bootstrap-t (Webb weights): cage-days are independent; cycles within a cage-day may "
        "be correlated in any way. With fewer than about 10 cage-days the interval is approximate."
    ),
    "mixed_model": (
        "Linear mixed model: normally distributed cage-day intercepts and residuals with constant "
        "variance; with fewer than 5 cages the cage term is fixed, and variance components from few "
        "units are imprecise. A singular fit means the cage-day variance was estimated as zero."
    ),
    "cluster_bootstrap": (
        "Cluster bootstrap: cage-days resampled with replacement; percentile intervals are too narrow "
        "when there are few cage-days."
    ),
}


@dataclass(frozen=True)
class WilcoxonResult:
    """Outcome of a Wilcoxon signed-rank test of x - y."""

    statistic: float  # r_plus (one-sided) or min(r_plus, r_minus) (two-sided), SciPy's convention
    p_value: float
    n: int  # pairs used
    n_zero: int  # pairs with x == y
    method: str  # "exact" or "approx"
    z: float | None = None


def average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties given their average rank (SciPy's ``rankdata(method='average')``)."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def wilcoxon_null_pmf(n: int) -> np.ndarray:
    """Null distribution of the positive rank sum for n untied, non-zero differences (SciPy 1.10)."""
    c = np.ones(1, dtype=np.float64)
    for k in range(1, n + 1):
        prev = c
        c = np.zeros(k * (k + 1) // 2 + 1, dtype=np.float64)
        m = len(prev)
        c[:m] = prev * 0.5
        c[-m:] += prev * 0.5
    return c


def wilcoxon_scipy110(x: np.ndarray, y: np.ndarray, alternative: Alternative = "two-sided") -> WilcoxonResult:
    """``scipy.stats.wilcoxon(x, y, alternative=...)`` as in SciPy 1.10.1 (defaults: zero_method
    'wilcox', no continuity correction, method 'auto').

    SciPy 1.10's rule: exact null distribution if there are at most 50 pairs (counted *before*
    dropping zero differences), but the normal approximation (with zeros dropped and the tie
    correction) as soon as any difference is zero. In exact mode a tied rank sum is truncated to
    an integer: a quirk reproduced on purpose.
    """
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if d.size == 0:
        return WilcoxonResult(float("nan"), float("nan"), 0, 0, "none")
    method = "exact" if d.size <= 50 else "approx"
    n_zero = int(np.sum(d == 0))
    if n_zero > 0 and method == "exact":
        method = "approx"
    if method == "approx":
        if n_zero == d.size:
            raise ValueError("every difference is zero: the Wilcoxon test is undefined")
        d = d[d != 0]
    count = d.size
    r = average_ranks(np.abs(d))
    r_plus = float(np.sum((d > 0) * r))
    r_minus = float(np.sum((d < 0) * r))
    stat = min(r_plus, r_minus) if alternative == "two-sided" else r_plus
    if method == "approx":
        mn = count * (count + 1.0) * 0.25
        se = count * (count + 1.0) * (2.0 * count + 1.0)
        _, reps = np.unique(r, return_counts=True)
        reps = reps[reps > 1]
        if reps.size:
            se -= 0.5 * float((reps * (reps * reps - 1)).sum())
        se = float(np.sqrt(se / 24))
        z = (stat - mn) / se
        if alternative == "two-sided":
            p = 2.0 * float(ndtr(-abs(z)))
        elif alternative == "greater":
            p = float(ndtr(-z))
        else:
            p = float(ndtr(z))
        return WilcoxonResult(stat, p, count, n_zero, "approx", z)
    pmf = wilcoxon_null_pmf(count)
    rp = int(r_plus)
    if alternative == "two-sided":
        if rp == (len(pmf) - 1) // 2:
            p = 1.0
        else:
            p = 2 * min(float(np.sum(pmf[rp:])), float(np.sum(pmf[: rp + 1])))
    elif alternative == "greater":
        p = float(np.sum(pmf[rp:]))
    else:
        p = float(np.sum(pmf[: rp + 1]))
    return WilcoxonResult(stat, float(np.clip(p, 0.0, 1.0)), count, n_zero, "exact")


# ============================================================================ current methods
def _tail_p(greater: float, less: float, alternative: Alternative) -> float:
    if alternative == "greater":
        return float(min(1.0, greater))
    if alternative == "less":
        return float(min(1.0, less))
    return float(min(1.0, 2.0 * min(greater, less)))


def wilcoxon_signed_rank(
    d: np.ndarray, alternative: Alternative = "two-sided", exact_max_n: int = 50
) -> WilcoxonResult:
    """Wilcoxon signed-rank test of the differences ``d`` (e.g. ON − OFF per cycle) against 0.

    Zero differences are dropped (Wilcoxon's convention) and counted. Up to ``exact_max_n``
    non-zero pairs the p-value comes from the exact distribution of the positive rank sum given
    the observed (tied) ranks; above that, from the normal approximation with the tie correction.
    ``statistic`` is the positive rank sum R+.
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n_zero = int(np.sum(d == 0))
    nz = d[d != 0]
    n = nz.size
    if n == 0:
        return WilcoxonResult(float("nan"), float("nan"), 0, n_zero, "undefined: all differences are zero")
    r = average_ranks(np.abs(nz))
    r_plus = float(r[nz > 0].sum())
    if n <= exact_max_n:
        w2 = np.rint(2 * r).astype(np.int64)  # midranks are multiples of 1/2
        counts = np.zeros(int(w2.sum()) + 1)
        counts[0] = 1.0
        top = 0
        for w in w2:
            counts[w : top + w + 1] += counts[: top + 1].copy()
            top += int(w)
        pmf = counts / counts.sum()
        obs = round(2 * r_plus)
        p = _tail_p(float(pmf[obs:].sum()), float(pmf[: obs + 1].sum()), alternative)
        return WilcoxonResult(r_plus, p, n, n_zero, "exact")
    mean = n * (n + 1) / 4.0
    _, reps = np.unique(r, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(((reps**3) - reps).sum()) / 48.0
    z = (r_plus - mean) / np.sqrt(var) if var > 0 else 0.0
    p = _tail_p(float(ndtr(-z)), float(ndtr(z)), alternative)
    return WilcoxonResult(r_plus, p, n, n_zero, "approx", float(z))


def hodges_lehmann(d: np.ndarray) -> float:
    """Hodges-Lehmann estimate of the median difference: median of all Walsh averages."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    i, j = np.triu_indices(d.size)
    return float(np.median((d[i] + d[j]) / 2.0))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (R+ − R−)/(R+ + R−), zeros dropped; in [−1, 1]."""
    d = np.asarray(d, dtype=np.float64)
    nz = d[np.isfinite(d) & (d != 0)]
    if nz.size == 0:
        return float("nan")
    r = average_ranks(np.abs(nz))
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())


def paired_effect_sizes(off: np.ndarray, on: np.ndarray) -> dict[str, float]:
    """Effect sizes of ON − OFF: means, mean and median difference, Hodges-Lehmann, rank-biserial,
    Cohen's dz (mean / SD of the differences) and the fold change mean(ON) / mean(OFF)."""
    off = np.asarray(off, dtype=np.float64)
    on = np.asarray(on, dtype=np.float64)
    ok = np.isfinite(off) & np.isfinite(on)
    off, on = off[ok], on[ok]
    d = on - off
    sd = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
    mean_off = float(off.mean()) if d.size else float("nan")
    return {
        "n": float(d.size),
        "mean_off": mean_off,
        "mean_on": float(on.mean()) if d.size else float("nan"),
        "mean_delta": float(d.mean()) if d.size else float("nan"),
        "median_delta": float(np.median(d)) if d.size else float("nan"),
        "hodges_lehmann": hodges_lehmann(d),
        "rank_biserial": rank_biserial(d),
        "dz": float(d.mean() / sd) if d.size > 1 and sd > 0 else float("nan"),
        "fold_change": float(on.mean() / mean_off) if d.size and mean_off > 0 else float("nan"),
    }


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error); NaNs are left out and stay NaN."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = idx.size
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="mergesort")]
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate); NaNs stay NaN."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = idx.size
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="mergesort")]
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


# ============================================================================ cluster-aware methods
# 12-hour blocks keep the block test at its nominal level with hour-to-hour correlation up to
# rho = 0.6 in simulation; 4-hour blocks reached a false-positive rate of 0.11 (docs/STATS_PLAN.md).
DEFAULT_BLOCK_HOURS = 12
BLOCK_TEST_RESAMPLES = 100_000  # the plan: every sign pattern up to 2^16, else 100,000 random ones


def blocks_within_clusters(
    cluster: np.ndarray, hour_index: np.ndarray, block_hours: int = DEFAULT_BLOCK_HOURS
) -> np.ndarray:
    """Block label per cycle: the cluster (cage-day) and which ``block_hours``-hour slice of it."""
    cluster = np.asarray(cluster)
    slot = np.asarray(hour_index, dtype=np.int64) // max(1, block_hours)
    keys = np.array([f"{c}|{s}" for c, s in zip(cluster, slot, strict=True)])
    return np.unique(keys, return_inverse=True)[1]


@dataclass(frozen=True)
class PermutationResult:
    statistic: float  # mean difference
    p_value: float
    n_blocks: int
    n_resamples: int
    method: str  # "exact" (all sign patterns) or "monte-carlo"


def block_signflip_test(
    d: np.ndarray,
    blocks: np.ndarray,
    alternative: Alternative = "two-sided",
    n_resamples: int = 100_000,
    seed: int = 20261104,
) -> PermutationResult:
    """Permutation test of mean(d) = 0 that flips the sign of whole blocks at once.

    Uses every sign pattern when there are at most ``log2(n_resamples)`` blocks, otherwise
    ``n_resamples`` random patterns (the observed one included, so p >= 1/(n_resamples+1)).
    """
    d = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(d)
    d, blocks = d[ok], np.asarray(blocks)[ok]
    n = d.size
    if n == 0:
        return PermutationResult(float("nan"), float("nan"), 0, 0, "undefined: no data")
    _, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    nb = sums.size
    observed = float(d.mean())
    if nb <= 62 and 2**nb <= n_resamples:
        codes = np.arange(2**nb, dtype=np.uint64)[:, None]
        bits = (codes >> np.arange(nb, dtype=np.uint64)[None, :]) & np.uint64(1)
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        stats = signs @ sums / n
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, nb))
        stats = np.concatenate([[observed], signs @ sums / n])
        method = "monte-carlo"
    tol = 1e-12 * max(1.0, abs(observed))
    if alternative == "two-sided":
        p = float(np.mean(np.abs(stats) >= abs(observed) - tol))
    elif alternative == "greater":
        p = float(np.mean(stats >= observed - tol))
    else:
        p = float(np.mean(stats <= observed + tol))
    return PermutationResult(observed, p, nb, int(stats.size), method)


_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


@dataclass(frozen=True)
class ClusterBootstrapResult:
    mean: float
    se: float  # cluster-robust (CR1) standard error
    ci_low: float
    ci_high: float
    p_value: float
    n_clusters: int
    n_resamples: int
    method: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _cr1_se(cluster_resid_sums: np.ndarray, n: int, g: int) -> np.ndarray:
    return np.sqrt(g / (g - 1) * np.sum(cluster_resid_sums**2, axis=-1)) / n


MIN_WILD_CLUSTERS = 6  # below this the wild bootstrap interval under-covers (docs/STATS_PLAN.md, Changes)


def _cluster_robust_t(
    d: np.ndarray, inv: np.ndarray, g: int, level: float, alternative: Alternative
) -> ClusterBootstrapResult:
    """Cluster-robust (CR1) t-test of the mean with g − 1 degrees of freedom. Used instead of the
    wild bootstrap below :data:`MIN_WILD_CLUSTERS` clusters: same estimate (the mean over cycles),
    honest coverage with very few clusters (docs/STATS_PLAN.md, Changes)."""
    from scipy import stats as sps

    n = d.size
    m = float(d.mean())
    se = float(_cr1_se(np.bincount(inv, weights=d - m, minlength=g), n, g))
    q = float(sps.t.ppf(1 - (1 - level) / 2, g - 1))
    if se > 0:
        tstat = m / se
        if alternative == "two-sided":
            p = float(2 * sps.t.sf(abs(tstat), g - 1))
        elif alternative == "greater":
            p = float(sps.t.sf(tstat, g - 1))
        else:
            p = float(sps.t.cdf(tstat, g - 1))
    else:
        p = float("nan")
    note = (
        f"only {g} cage-days: cluster-robust t-test ({g - 1} df) instead of the wild bootstrap, whose "
        "interval is too narrow with so few units"
    )
    return ClusterBootstrapResult(
        m, se, m - q * se, m + q * se, p, g, 0, f"cluster-robust t-test ({g - 1} df)", (note,)
    )


def wild_cluster_bootstrap(
    d: np.ndarray,
    cluster: np.ndarray,
    level: float = 0.95,
    alternative: Alternative = "two-sided",
    n_resamples: int = 9_999,
    seed: int = 20261104,
) -> ClusterBootstrapResult:
    """Wild cluster bootstrap-t for the mean of ``d`` with clusters (e.g. cage-days) as units.

    The test (H0: mean = 0) imposes the null (restricted bootstrap); the interval uses the
    unrestricted residuals (percentile-t). Webb's six-point weights are used because they behave
    well with few clusters; every weight pattern is used when that is at most 50,000 patterns.
    """
    d = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(d)
    d, cluster = d[ok], np.asarray(cluster)[ok]
    n = d.size
    _, inv = np.unique(cluster, return_inverse=True)
    g = int(np.max(inv)) + 1 if n else 0
    notes: list[str] = []
    if g < 2:
        nan = float("nan")
        return ClusterBootstrapResult(
            float(d.mean()) if n else nan, nan, nan, nan, nan, g, 0, "undefined: needs at least 2 clusters"
        )
    if g < MIN_WILD_CLUSTERS:
        return _cluster_robust_t(d, inv, g, level, alternative)
    if g < 10:
        notes.append(f"only {g} independent units (cage-days): the interval is approximate")
    sizes = np.bincount(inv, minlength=g).astype(np.float64)
    sums = np.bincount(inv, weights=d, minlength=g)
    mean = float(d.mean())
    resid_sums = sums - sizes * mean
    se = float(_cr1_se(resid_sums, n, g))
    if 6**g <= 50_000:
        grid = np.array(np.meshgrid(*[_WEBB] * g, indexing="ij")).reshape(g, -1).T
        method = "wild cluster bootstrap-t (Webb weights, all patterns)"
        enumerated = True
    else:
        rng = np.random.default_rng(seed)
        grid = rng.choice(_WEBB, size=(n_resamples, g))
        method = "wild cluster bootstrap-t (Webb weights)"
        enumerated = False
    # restricted (null imposed): residuals are d itself
    s_star = grid * sums[None, :]
    m_star = s_star.sum(axis=1) / n
    se_star = _cr1_se(s_star - sizes[None, :] * m_star[:, None], n, g)
    if se > 0:
        t_obs = mean / se
    else:  # every cluster identical: no spread to judge the mean against
        t_obs = 0.0 if mean == 0 else float(np.copysign(np.inf, mean))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_null = np.where(se_star > 0, m_star / se_star, 0.0)
    if alternative == "two-sided":
        hits = np.abs(t_null) >= abs(t_obs)
    elif alternative == "greater":
        hits = t_null >= t_obs
    else:
        hits = t_null <= t_obs
    # every pattern enumerated: the observed one is among them; random draws: count it, (k+1)/(B+1)
    p = float(hits.mean()) if enumerated else float((hits.sum() + 1) / (hits.size + 1))
    # unrestricted, for the interval
    e_star = grid * resid_sums[None, :]
    m_dev = e_star.sum(axis=1) / n
    se_u = _cr1_se(e_star - sizes[None, :] * m_dev[:, None], n, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_u = np.where(se_u > 0, m_dev / se_u, 0.0)
    a = (1 - level) / 2
    q_lo, q_hi = np.quantile(t_u, [a, 1 - a])
    return ClusterBootstrapResult(
        mean, se, mean - q_hi * se, mean - q_lo * se, p, g, int(grid.shape[0]), method, tuple(notes)
    )


def cluster_bootstrap_ci(
    clusters: list[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    level: float = 0.95,
    n_resamples: int = 9_999,
    seed: int = 20261104,
) -> tuple[float, float]:
    """Percentile interval of ``statistic`` when whole clusters (lists of values) are resampled."""
    g = len(clusters)
    if g < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, g, size=(n_resamples, g))
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        values[b] = statistic(np.concatenate([clusters[i] for i in picks[b]]))
    a = (1 - level) / 2
    lo, hi = np.nanquantile(values, [a, 1 - a])
    return float(lo), float(hi)


def lag_autocorrelation(
    d: np.ndarray, cluster: np.ndarray, hour_index: np.ndarray, max_lag: int = 6
) -> np.ndarray:
    """Autocorrelation of the differences at lags 1..max_lag hours, pooled within clusters."""
    d = np.asarray(d, dtype=np.float64)
    cluster = np.asarray(cluster)
    hour_index = np.asarray(hour_index, dtype=np.int64)
    ok = np.isfinite(d)
    centred = np.where(ok, d - np.nanmean(d), np.nan)
    var = np.nanmean(centred[ok] ** 2) if ok.any() else np.nan
    out = np.full(max_lag, np.nan)
    if not ok.any() or not var > 0:
        return out
    lookup = {(c, h): v for c, h, v in zip(cluster, hour_index, centred, strict=True) if np.isfinite(v)}
    for lag in range(1, max_lag + 1):
        prods = [v * lookup[(c, h + lag)] for (c, h), v in lookup.items() if (c, h + lag) in lookup]
        if len(prods) >= 3:
            out[lag - 1] = float(np.mean(prods) / var)
    return out


@dataclass(frozen=True)
class MixedModelResult:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    var_cage_day: float
    var_residual: float
    converged: bool
    singular: bool
    formula: str
    messages: tuple[str, ...] = ()
    var_cage: float = float("nan")  # only when cages enter as a variance component (>= 5 cages)


def mixed_model(
    d: np.ndarray, cage_day: np.ndarray, cage: np.ndarray | None = None, phase: np.ndarray | None = None
) -> MixedModelResult:
    """Linear mixed model ``delta ~ 1 [+ phase] [+ cage]`` with a random intercept per cage-day.

    Sum-to-zero coding keeps the intercept the overall mean difference. With fewer than 5 cages
    the cage enters as a fixed effect; otherwise as a variance component.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(
        {"delta": np.asarray(d, dtype=np.float64), "cage_day": np.asarray(cage_day).astype(str)}
    )
    terms = ["1"]
    vc: dict[str, str] | None = None
    groups = "cage_day"
    if phase is not None:
        df["phase"] = np.asarray(phase).astype(str)
        if len(set(df["phase"])) > 1:
            terms.append("C(phase, Sum)")
    if cage is not None:
        df["cage"] = np.asarray(cage).astype(str)
        n_cages = df["cage"].nunique()
        if 2 <= n_cages < 5:
            terms.append("C(cage, Sum)")
        elif n_cages >= 5:
            groups = "cage"
            vc = {"cage_day": "0 + C(cage_day)"}
    df = df[np.isfinite(df["delta"])]
    formula = "delta ~ " + " + ".join(terms)
    nan = float("nan")
    if df["cage_day"].nunique() < 2:
        return MixedModelResult(
            nan, nan, nan, nan, nan, nan, nan, False, False, formula, ("needs at least 2 cage-days",)
        )
    # Fit on a unit scale (fractions differ by ~0.01, which upsets the optimisers), then rescale.
    scale = float(df["delta"].std()) or 1.0
    df["delta"] = df["delta"] / scale
    model = smf.mixedlm(formula, df, groups=df[groups], vc_formula=vc)
    best: Any = None
    used = ""
    messages: list[str] = []
    for method in (["nm", "bfgs"], ["powell"], ["lbfgs"]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = model.fit(reml=True, method=method)
            except (ValueError, np.linalg.LinAlgError, OverflowError):
                continue
            se_try = float(fit.bse_fe["Intercept"])  # NaN (with a warning) when the Hessian is singular
        # A failed optimisation can "converge" with an absurd standard error; reject those.
        if not (fit.converged and np.isfinite(se_try) and se_try < 10.0):
            continue
        if best is None or fit.llf > best.llf:
            best, used = fit, "+".join(method)
            messages = sorted({str(w.message).split("\n")[0] for w in caught})
    if best is None:
        return MixedModelResult(
            nan, nan, nan, nan, nan, nan, nan, False, False, formula, ("no optimiser produced a usable fit",)
        )
    fit = best
    est = float(fit.fe_params["Intercept"]) * scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # SEs of other terms can be NaN; not used
        se = float(fit.bse_fe["Intercept"]) * scale
        ci = fit.conf_int().loc["Intercept"] * scale
        p_value = float(fit.pvalues["Intercept"])
    var_group = float(np.asarray(fit.cov_re)[0, 0]) if np.asarray(fit.cov_re).size else 0.0
    var_cage = float("nan")
    if vc is not None and len(fit.vcomp):
        var_cage = var_group * scale**2  # groups are cages here; cage-days are the variance component
        var_group = float(fit.vcomp[0])
    var_group *= scale**2
    var_resid = float(fit.scale) * scale**2
    singular = var_group < 1e-8 * max(var_resid, 1e-300)
    messages.append(f"optimiser: {used}")
    return MixedModelResult(
        est,
        se,
        float(ci.iloc[0]),
        float(ci.iloc[1]),
        p_value,
        var_group,
        var_resid,
        bool(fit.converged),
        bool(singular),
        formula,
        tuple(messages),
        var_cage,
    )


def summarise_paired(
    off: np.ndarray,
    on: np.ndarray,
    cage_day: np.ndarray,
    hour_index: np.ndarray,
    *,
    cage: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    alternative: Alternative = "two-sided",
    block_hours: int = DEFAULT_BLOCK_HOURS,
    n_resamples: int = 9_999,
    seed: int = 20261104,
    with_mixed_model: bool = True,
    hour_of_day: np.ndarray | None = None,
) -> dict[str, Any]:
    """Every pre-registered test and effect size for one ON/OFF comparison (docs/STATS_PLAN.md).

    ``hour_of_day`` (clock hour 0-23) anchors the permutation blocks to the clock (00-12, 12-24 for
    12-h blocks); without it blocks start at each cage-day's first recorded hour.
    """
    off = np.asarray(off, dtype=np.float64)
    on = np.asarray(on, dtype=np.float64)
    d = on - off
    ok = np.isfinite(d)
    cd = np.asarray(cage_day)[ok]
    hi = np.asarray(hour_index)[ok]
    dd = d[ok]
    anchor = np.asarray(hour_of_day)[ok] if hour_of_day is not None else hi
    blocks = blocks_within_clusters(cd, anchor, block_hours) if dd.size else np.empty(0, dtype=int)
    out: dict[str, Any] = {"effect": paired_effect_sizes(off[ok], on[ok])}
    out["wilcoxon"] = wilcoxon_signed_rank(dd, alternative)
    out["block_signflip"] = block_signflip_test(
        dd,
        blocks,
        alternative,
        n_resamples=BLOCK_TEST_RESAMPLES,
        seed=seed,  # exact up to 16 blocks
    )
    out["wild_cluster"] = wild_cluster_bootstrap(
        dd, cd, alternative=alternative, n_resamples=n_resamples, seed=seed
    )
    clusters = [dd[cd == c] for c in np.unique(cd)]
    out["median_delta_ci"] = cluster_bootstrap_ci(
        clusters,
        lambda v: float(np.median(v)) if v.size else np.nan,
        n_resamples=min(n_resamples, 4_999),
        seed=seed,
    )
    out["autocorrelation"] = lag_autocorrelation(dd, cd, hi)
    if with_mixed_model:
        out["mixed_model"] = mixed_model(
            dd,
            cd,
            None if cage is None else np.asarray(cage)[ok],
            None if phase is None else np.asarray(phase)[ok],
        )
    return out
