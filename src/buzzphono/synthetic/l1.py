"""Level-1 simulator: per-hour-cycle OFF and ON fractions with known truth.

It generates the numbers the statistics see (one OFF and one ON speaker-zone fraction per
hour-cycle) without simulating mosquitoes, so thousands of datasets take seconds. It is used to
check that each test keeps its false-positive rate, that intervals cover the truth, and how much
power a design has.

Model, for cage k, day j, hour h (cycle c = (k, j, h)):

* shared baseline ``b_c = base(phase) + u_k + v_kj + w_c`` (cage, cage-day and hour effects),
  set to exactly 0 with probability ``zero_inflation`` (no resters at the speaker that hour);
* ``OFF_c = clip(b_c + e_c / 2, 0, 1)``, ``ON_c = clip(b_c + effect(phase) + a_kj - e_c / 2, 0, 1)``,
  where ``e_c`` follows an AR(1) process over the hours of each cage-day (correlation ``rho``)
  and ``a_kj`` is an optional cage-day effect on the ON−OFF difference itself (``sd_day_delta``).

With ``effect = 0`` and ``sd_day_delta = 0``, ON and OFF are exchangeable: the null is true.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

THESIS_PHASE_OF_HOUR = {
    h: ("dawn" if h == 5 else "day" if 6 <= h < 16 else "dusk" if h == 16 else "night") for h in range(24)
}


@dataclass(frozen=True)
class L1Design:
    n_cages: int = 3
    n_days: int = 2
    base: Mapping[str, float] = field(
        default_factory=lambda: {"dawn": 0.04, "day": 0.04, "dusk": 0.05, "night": 0.02}
    )
    effect: Mapping[str, float] = field(
        default_factory=lambda: {"dawn": 0.0, "day": 0.0, "dusk": 0.0, "night": 0.0}
    )
    sd_cage: float = 0.01
    sd_cage_day: float = 0.01
    sd_hour: float = 0.01
    sd_noise: float = 0.03
    rho: float = 0.0
    sd_day_delta: float = 0.0
    zero_inflation: float = 0.15

    @classmethod
    def uniform_effect(cls, effect: float, **kw: float) -> L1Design:
        return cls(effect=dict.fromkeys(("dawn", "day", "dusk", "night"), effect), **kw)  # type: ignore[arg-type]


def simulate_cycles(design: L1Design, seed: int) -> pd.DataFrame:
    """One dataset: a row per hour-cycle with cage, day, cage_day, hour, phase, off, on, true_effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(design.n_cages):
        u = rng.normal(0, design.sd_cage)
        for j in range(design.n_days):
            v = rng.normal(0, design.sd_cage_day)
            a = rng.normal(0, design.sd_day_delta) if design.sd_day_delta > 0 else 0.0
            e = np.empty(24)
            e[0] = rng.normal(0, design.sd_noise)
            innov = np.sqrt(max(0.0, 1 - design.rho**2)) * design.sd_noise
            for h in range(1, 24):
                e[h] = design.rho * e[h - 1] + rng.normal(0, innov)
            for h in range(24):
                phase = THESIS_PHASE_OF_HOUR[h]
                b = design.base[phase] + u + v + rng.normal(0, design.sd_hour)
                if rng.random() < design.zero_inflation:
                    b = 0.0
                eff = design.effect[phase]
                off = float(np.clip(b + e[h] / 2, 0, 1))
                on = float(np.clip(b + eff + a - e[h] / 2, 0, 1))
                rows.append((f"cage{k}", j, f"cage{k}-day{j}", h, phase, off, on, eff))
    return pd.DataFrame(
        rows, columns=["cage", "day", "cage_day", "hour", "phase", "off", "on", "true_effect"]
    )
