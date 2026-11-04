"""Reliability: how much of the ON − OFF response belongs to cages, cage-days and single hours.

The per-cycle response Δ is split into three nested variance components – between cages, between
days within a cage, and between hour-cycles within a cage-day – with the unbalanced nested ANOVA
(moment) estimators, negative estimates set to 0. From them:

* the intraclass correlations ``icc_cage`` and ``icc_cage_day`` (share of the variance at each
  level) with 95% intervals from a cluster bootstrap (cages, or cage-days with a single cage);
* the minimum detectable effect (MDE) of the mean Δ at α = 0.05 (two-sided) and 80% power for the
  present design, analytically from the components and, as a cross-check, from the bootstrap
  standard error of the mean;
* the MDE for other numbers of cage-days, so a new experiment can be sized.

Fewer than 5 cages make the cage component, and so every between-cage statement, very imprecise:
the table says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class Components:
    var_cage: float
    var_cage_day: float
    var_cycle: float
    n_cycles: int
    n_cage_days: int
    n_cages: int

    @property
    def total(self) -> float:
        return self.var_cage + self.var_cage_day + self.var_cycle

    @property
    def icc_cage(self) -> float:
        return self.var_cage / self.total if self.total > 0 else float("nan")

    @property
    def icc_cage_day(self) -> float:
        return self.var_cage_day / self.total if self.total > 0 else float("nan")


def variance_components(y: np.ndarray, cage: np.ndarray, cage_day: np.ndarray) -> Components:
    """Nested ANOVA estimates (cycles within cage-days within cages)."""
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    y = y[ok]
    cage = np.asarray(cage).astype(str)[ok]
    day = np.char.add(np.char.add(cage, "|"), np.asarray(cage_day).astype(str)[ok])
    n = y.size
    _, d_inv, n_ij = np.unique(day, return_inverse=True, return_counts=True)
    c_lab, c_inv = np.unique(cage, return_inverse=True)
    a, b = c_lab.size, n_ij.size
    nan = float("nan")
    if n < 3 or b < 2:
        return Components(nan, nan, nan, n, b, a)
    mean_ij = np.bincount(d_inv, weights=y) / n_ij
    n_i = np.bincount(c_inv, minlength=a).astype(np.float64)
    mean_i = np.bincount(c_inv, weights=y, minlength=a) / n_i
    grand = y.mean()
    day_cage = np.zeros(b, dtype=np.int64)
    day_cage[d_inv] = c_inv
    ss_e = float(np.sum((y - mean_ij[d_inv]) ** 2))
    ss_d = float(np.sum(n_ij * (mean_ij - mean_i[day_cage]) ** 2))
    df_e, df_d = n - b, b - a
    ms_e = ss_e / df_e if df_e > 0 else nan
    sum_nij2_over_ni = float(
        np.sum(np.bincount(day_cage, weights=n_ij.astype(float) ** 2, minlength=a) / n_i)
    )
    if df_d > 0:
        k1 = (n - sum_nij2_over_ni) / df_d
        var_d = max(0.0, (ss_d / df_d - ms_e) / k1)
    else:
        var_d = 0.0
    var_c = 0.0
    if a >= 2:
        ss_c = float(np.sum(n_i * (mean_i - grand) ** 2))
        k2 = (sum_nij2_over_ni - float(np.sum(n_ij.astype(float) ** 2)) / n) / (a - 1)
        k3 = (n - float(np.sum(n_i**2)) / n) / (a - 1)
        var_c = max(0.0, (ss_c / (a - 1) - ms_e - k2 * var_d) / k3)
    return Components(var_c, var_d, max(ms_e, 0.0), n, b, a)


def power_multiplier(df: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """k such that a true effect of k standard errors is detected with the given power by a
    two-sided t-test on ``df`` degrees of freedom (noncentral t; about 2.8 when df is large)."""
    from scipy.optimize import brentq

    df = max(float(df), 1.0)
    crit = sps.t.ppf(1 - alpha / 2, df)

    def gap(k: float) -> float:
        upper = float(sps.nct.sf(crit, df, k))
        lower = float(np.nan_to_num(sps.nct.cdf(-crit, df, k)))  # negligible, and NaN, far from 0
        return float(np.nan_to_num(upper, nan=1.0)) + lower - power

    hi = 4.0
    while gap(hi) < 0:  # widen the bracket step by step (the noncentral t is NaN far out)
        hi *= 1.5
        if hi > 1e4:
            return float("inf")
    return float(brentq(gap, 0.0, hi))


def mde_from_components(
    comp: Components,
    n_cages: float,
    n_cage_days: float,
    n_cycles: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest true mean Δ detected with the given power by the primary test, whose degrees of
    freedom are the number of cage-days − 1 (noncentral t)."""
    var_mean = comp.var_cage / n_cages + comp.var_cage_day / n_cage_days + comp.var_cycle / n_cycles
    return float(power_multiplier(n_cage_days - 1, alpha, power) * np.sqrt(var_mean))


def reliability(
    cycles: pd.DataFrame,
    value_col: str = "delta",
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    n_resamples: int = 999,
    seed: int = 20261104,
    planning_cage_days: tuple[int, ...] = (4, 6, 8, 12, 16, 24, 32),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary table, planning table) for one response column (see the module docstring)."""
    y = cycles[value_col].to_numpy(dtype=np.float64)
    cage = cycles["cage"].astype(str).to_numpy()
    day = cycles["cage_day"].astype(str).to_numpy()
    comp = variance_components(y, cage, day)
    ok = np.isfinite(y)
    rng = np.random.default_rng(seed)
    unit = cage if comp.n_cages >= 5 else day
    labels = np.unique(unit[ok])
    boot_icc_c, boot_icc_d, boot_mean = [], [], []
    if labels.size >= 2:
        members = {u: np.flatnonzero(ok & (unit == u)) for u in labels}
        for _ in range(n_resamples):
            pick = rng.choice(labels, size=labels.size, replace=True)
            idx = np.concatenate([members[u] for u in pick])
            # relabel the resampled units so duplicates count as separate units
            rep = np.concatenate([np.full(members[u].size, k) for k, u in enumerate(pick)])
            if comp.n_cages >= 5:
                bc = rep.astype(str)
                bd = np.char.add(bc, np.char.add("|", day[idx]))
            else:
                bc = cage[idx]
                bd = rep.astype(str)
            bcomp = variance_components(y[idx], bc, bd)
            boot_icc_c.append(bcomp.icc_cage)
            boot_icc_d.append(bcomp.icc_cage_day)
            boot_mean.append(float(np.mean(y[idx])))

    def ci(v: list[float]) -> tuple[float, float]:
        a = np.asarray(v, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size < 10:
            return float("nan"), float("nan")
        lo, hi = np.percentile(a, [2.5, 97.5])
        return float(lo), float(hi)

    z = power_multiplier(comp.n_cage_days - 1, alpha, power) if comp.n_cage_days >= 2 else float("nan")
    se_boot = float(np.std(boot_mean, ddof=1)) if len(boot_mean) > 1 else float("nan")
    warnings_: list[str] = []
    if comp.n_cages < 5:
        warnings_.append(
            f"Only {comp.n_cages} cage(s): the between-cage variance and ICC are very imprecise; intervals "
            "resample cage-days instead of cages."
        )
    if comp.n_cage_days < 10:
        warnings_.append(f"Only {comp.n_cage_days} cage-days: bootstrap intervals are approximate.")
    icc_c_ci, icc_d_ci = ci(boot_icc_c), ci(boot_icc_d)
    summary = pd.DataFrame(
        [
            {
                "response": value_col,
                "n_cycles": comp.n_cycles,
                "n_cage_days": comp.n_cage_days,
                "n_cages": comp.n_cages,
                "mean": float(np.nanmean(y)) if ok.any() else float("nan"),
                "var_cage": comp.var_cage,
                "var_cage_day": comp.var_cage_day,
                "var_cycle": comp.var_cycle,
                "icc_cage": comp.icc_cage,
                "icc_cage_ci_low": icc_c_ci[0],
                "icc_cage_ci_high": icc_c_ci[1],
                "icc_cage_day": comp.icc_cage_day,
                "icc_cage_day_ci_low": icc_d_ci[0],
                "icc_cage_day_ci_high": icc_d_ci[1],
                "mde_analytic": mde_from_components(
                    comp, max(comp.n_cages, 1), comp.n_cage_days, comp.n_cycles, alpha, power
                )
                if comp.n_cage_days >= 2
                else float("nan"),
                "mde_bootstrap": float(z * se_boot),
                "alpha": alpha,
                "power": power,
                "warnings": " ".join(warnings_),
            }
        ]
    )
    rows = []
    if comp.n_cage_days >= 2 and np.isfinite(comp.total):
        days_per_cage = comp.n_cage_days / max(comp.n_cages, 1)
        cycles_per_day = comp.n_cycles / comp.n_cage_days
        for g in planning_cage_days:
            n_cages = max(1.0, g / days_per_cage)
            rows.append(
                {
                    "response": value_col,
                    "cage_days": g,
                    "cages_assumed": n_cages,
                    "cycles_per_cage_day": cycles_per_day,
                    "mde": mde_from_components(comp, n_cages, g, g * cycles_per_day, alpha, power),
                }
            )
    return summary, pd.DataFrame(rows)
