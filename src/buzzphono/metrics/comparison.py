"""Comparing experiments: tuning curves and group contrasts, with the confounds stated.

Each experiment contributes one row: its ON − OFF effect with the 95% interval and p-value of
its own analysis, plus the metadata given for it in the settings (frequency, stimulus type,
species, sex, group). Metadata is never guessed from folder names.

Experiments differ in more than the stimulus: different cages, dates, rigs and colonies. When two
descriptors always change together (every 475 Hz run was also the only Aedes run, say), their
effects cannot be told apart, and :func:`confounds` says so in plain words. Groups are compared
at the experiment level (each experiment is one unit), by an exact permutation test of the
difference in mean effect, and only when each group has at least 3 experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.stats import benjamini_hochberg

DESCRIPTORS = ("frequency_hz", "stimulus_type", "species", "sex", "group")
MIN_PER_GROUP = 3


def experiment_table(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per experiment; adds Benjamini-Hochberg q-values across experiments."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    for col in DESCRIPTORS:
        if col not in df:
            df[col] = None
    if "p_value" in df:
        p = df["p_value"].to_numpy(dtype=np.float64)
        q = np.full(p.size, np.nan)
        fin = np.isfinite(p)
        if fin.any():
            q[fin] = benjamini_hochberg(p[fin])
        df["q_bh"] = q
    sort = [c for c in ("frequency_hz", "label") if c in df]
    return df.sort_values(sort, na_position="last").reset_index(drop=True)


def confounds(df: pd.DataFrame) -> list[str]:
    """Plain-English notes on descriptors that cannot be separated in these experiments."""
    notes = []
    present = [c for c in DESCRIPTORS if c in df and len(set(df[c].dropna())) > 1]
    for a, b in itertools.combinations(present, 2):
        sub = df[[a, b]].dropna()
        if sub.empty:
            continue
        a_to_b = sub.groupby(a)[b].nunique().max() == 1
        b_to_a = sub.groupby(b)[a].nunique().max() == 1
        if a_to_b and b_to_a:
            notes.append(
                f"{a} and {b} change together in every experiment: their effects cannot be separated."
            )
    if len(df) and "label" in df:
        for col in present:
            single = df.groupby(col)["label"].nunique()
            lonely = [str(v) for v, n in single.items() if n == 1]
            if lonely and len(lonely) == len(single):
                notes.append(
                    f"Every {col} value comes from a single experiment, so a {col} effect is confounded with "
                    "cage, date and rig."
                )
    return notes


def group_contrast(df: pd.DataFrame, by: str, value: str = "mean_delta") -> pd.DataFrame:
    """Pairwise exact permutation tests of the mean effect between levels of ``by``."""
    rows = []
    sub = df[[by, value]].dropna()
    levels = sorted(sub[by].unique(), key=str)
    for a, b in itertools.combinations(levels, 2):
        xa = sub.loc[sub[by] == a, value].to_numpy(dtype=np.float64)
        xb = sub.loc[sub[by] == b, value].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "descriptor": by,
            "level_a": a,
            "level_b": b,
            "n_a": xa.size,
            "n_b": xb.size,
            "mean_a": float(xa.mean()) if xa.size else np.nan,
            "mean_b": float(xb.mean()) if xb.size else np.nan,
        }
        row["difference"] = row["mean_b"] - row["mean_a"]
        if xa.size >= MIN_PER_GROUP and xb.size >= MIN_PER_GROUP:
            row["p_permutation"] = _perm_p(xa, xb)
            row["note"] = "exact permutation test, experiments as units"
        else:
            row["p_permutation"] = np.nan
            row["note"] = f"not tested: needs at least {MIN_PER_GROUP} experiments per group"
        rows.append(row)
    return pd.DataFrame(rows)


def _perm_p(xa: np.ndarray, xb: np.ndarray, max_exact: int = 200_000, seed: int = 20261104) -> float:
    """Two-sided permutation p of the difference in means: every split when there are at most
    ``max_exact``, otherwise that many random splits (the observed one counted, (k+1)/(B+1))."""
    import math

    pooled = np.concatenate([xa, xb])
    n, na = pooled.size, xa.size
    observed = abs(xb.mean() - xa.mean()) - 1e-12
    total = pooled.sum()

    def diff(sa: np.ndarray | float) -> np.ndarray:
        return np.abs((total - np.asarray(sa)) / (n - na) - np.asarray(sa) / na)

    if math.comb(n, na) <= max_exact:
        sums = np.array([pooled[list(idx)].sum() for idx in itertools.combinations(range(n), na)])
        return float(np.mean(diff(sums) >= observed))
    rng = np.random.default_rng(seed)
    sums = np.array([pooled[rng.permutation(n)[:na]].sum() for _ in range(max_exact)])
    return float((np.sum(diff(sums) >= observed) + 1) / (max_exact + 1))
