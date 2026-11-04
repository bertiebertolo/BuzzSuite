"""Sensitivity: does the ON − OFF result survive reasonable changes to the analysis choices?

The headline comparison is recomputed with one choice changed at a time:

* **zone margin** – the speaker zone grown or shrunk by m px (−10 … +20);
* **zone shift** – the zone moved m px towards the cage centre (−40 … +60), plus any extra zone
  given (for example a polygon drawn before a data-driven adjustment, disclosed as such);
* **colony cap** – 25, 30, 35 mosquitoes, and no cap at all (the uncapped fraction);
* **walking resters removed** – resting detections that moved faster than 1, 2, 4 or 8 mm/s;
* **window variants** – other OFF and ON windows;
* **clock** – every other time base supplied (thesis, nominal, corrected).

Every variant uses the same hour-cycles as the baseline (the baseline's usable cycles), so only the
choice changes. Flags apply only when the baseline interval excludes 0 (otherwise there is no
effect whose robustness could be judged). A variant is flagged when the mean Δ changes sign (for
zone changes only when the change is at most half the zone's width: moving the zone further takes
it off the speaker, and the effect is then expected to vanish), or when a change of at most 5 px in
the zone changes the mean Δ by more than half.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.frames import ExperimentClock, ZoneTest, bin_experiment
from buzzphono.geometry import centroid, contains, signed_distance
from buzzphono.io.discover import ExperimentLayout
from buzzphono.metrics.cycles import Coverage, cycle_table, paired_comparison, usable
from buzzphono.stats import benjamini_hochberg

MM_PER_PX_UNCALIBRATED = 0.4  # 5 px/s ≈ 2 mm/s when the cage size is unknown


class _ZoneDistances:
    """Signed distance to one polygon, shared by all its grown/shrunk variants.

    Every variant asks about the same array of positions in turn, so the distances are computed once
    per array. Points farther than ``reach`` outside the polygon's bounding box cannot be within any
    margin and get +inf without the (costly) exact distance.
    """

    def __init__(self, polygon: np.ndarray, reach: float) -> None:
        self.polygon = polygon
        self.lo = polygon.min(axis=0) - reach
        self.hi = polygon.max(axis=0) + reach
        self._xy: np.ndarray | None = None  # held, so its identity cannot be reused by a new array
        self._value = np.empty(0)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        if xy is not self._xy:
            pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
            out = np.full(len(pts), np.inf)
            near = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
            if near.any():
                out[near] = signed_distance(self.polygon, pts[near])
            self._xy, self._value = xy, out
        return self._value


def grown_zone(polygon: np.ndarray, margin_px: float, distances: _ZoneDistances | None = None) -> ZoneTest:
    """The zone grown (``margin_px`` > 0) or shrunk (< 0); 0 returns the polygon itself."""
    if margin_px == 0:
        return polygon
    dist = distances or _ZoneDistances(polygon, abs(margin_px) + 1.0)

    def inside(xy: np.ndarray) -> np.ndarray:
        return dist(xy) <= margin_px

    return inside


def shifted_zone(polygon: np.ndarray, cage: np.ndarray, shift_px: float) -> np.ndarray:
    """The zone moved ``shift_px`` towards the cage centre (away from it when negative)."""
    v = centroid(np.asarray(cage, dtype=np.float64)) - centroid(polygon)
    norm = float(np.hypot(*v))
    return polygon if norm == 0 else polygon + shift_px * v / norm


def zone_area(zone: ZoneTest, frame_size: tuple[int, int], step: float = 1.0) -> float:
    """Area in px² of any zone test, by counting a ``step``-px grid of points inside it."""
    w, h = frame_size
    xs = np.arange(step / 2, w, step)
    ys = np.arange(step / 2, h, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = zone(pts) if callable(zone) else contains(np.asarray(zone, dtype=np.float64), pts)
    return float(np.count_nonzero(inside) * step * step)


@dataclass
class SensitivityPlan:
    margins_px: list[float] = field(default_factory=lambda: [-10.0, -5.0, 5.0, 10.0, 20.0])
    shifts_px: list[float] = field(default_factory=lambda: [-40.0, -20.0, 20.0, 40.0, 60.0])
    colonies: list[int] = field(default_factory=lambda: [25, 35])
    walking_mm_s: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    on_windows: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"on_40_45": (40, 45), "on_42_50": (42, 50)}
    )
    off_windows: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"off_18_28": (18, 28), "off_30_40": (30, 40)}
    )
    extra_zones: dict[str, np.ndarray] = field(default_factory=dict)  # name -> polygon (disclosed)


def run_sensitivity(
    layout: ExperimentLayout,
    cache_dir: Path,
    clocks: Mapping[str, ExperimentClock],
    *,
    speaker: np.ndarray,
    cage: np.ndarray,
    frame_size: tuple[int, int],
    fps: float,
    colony: int = 30,
    on: tuple[int, int] = (40, 50),
    off: tuple[int, int] = (28, 38),
    phases: Mapping[int, str] | None = None,
    coverage: Coverage | None = None,
    px_per_mm: float | None = None,
    plan: SensitivityPlan | None = None,
    cage_label: str = "",
    n_resamples: int = 999,
    seed: int = 20261104,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """One row per variant (see the module docstring); the first clock in ``clocks`` is the baseline."""
    plan = plan or SensitivityPlan()
    base_clock_name = next(iter(clocks))
    base_clock = clocks[base_clock_name]
    windows = {"on": on, "off": off, **plan.on_windows, **plan.off_windows}
    kw: dict[str, Any] = {"fps": fps, "cage": cage_label, "phases": phases, "coverage": coverage}

    zones: dict[str, ZoneTest] = {"speaker": speaker}
    shared = _ZoneDistances(speaker, max([abs(m) for m in plan.margins_px], default=0.0) + 1.0)
    meta: dict[str, tuple[str, str, float]] = {}  # zone -> (family, label, px)
    for m in plan.margins_px:
        name = f"margin_{m:+g}px"
        zones[name] = grown_zone(speaker, m, shared)
        meta[name] = ("zone margin", f"{m:+g} px", m)
    for s in plan.shifts_px:
        name = f"shift_{s:+g}px"
        zones[name] = shifted_zone(speaker, cage, s)
        meta[name] = ("zone shift towards cage centre", f"{s:+g} px", s)
    for name, poly in plan.extra_zones.items():
        zones[f"extra_{name}"] = np.asarray(poly, dtype=np.float64)
        meta[f"extra_{name}"] = ("other zone (disclosed)", name, np.nan)
    areas = {k: zone_area(z, frame_size, 2.0) for k, z in zones.items() if callable(z)}

    steps = 1 + len(plan.colonies) + len(plan.walking_mm_s) + len(clocks) - 1
    done = 0

    def tick(msg: str) -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done / steps, msg)

    def binned(eclock: ExperimentClock, zs: Mapping[str, ZoneTest], **extra: Any) -> pd.DataFrame:
        b = bin_experiment(
            layout,
            cache_dir,
            eclock,
            zs,
            cage=None,
            with_moving=False,
            cancel=cancel,
            arrays_cache=loaded,
            **extra,
        )
        return cycle_table(b, windows, **kw)

    loaded: dict[str, dict[str, np.ndarray]] = {}  # resting arrays, read once for every pass
    base = binned(base_clock, zones, colony=colony, zone_areas=areas)
    tick("zone variants")
    keep = base.loc[usable(base, ("on", "off")), "cycle_start_ns"]
    rows: list[dict[str, Any]] = []

    def compare(family: str, label: str, cycles: pd.DataFrame, a: str, b: str, px: float = np.nan) -> None:
        sub = cycles[cycles["cycle_start_ns"].isin(keep)]
        t = paired_comparison(
            sub, a, b, label=label, n_resamples=n_resamples, seed=seed, with_mixed_model=False, by_phase=False
        )
        r = t.iloc[0].to_dict()
        rows.append(
            {
                "family": family,
                "variant": label,
                "zone_change_px": px,
                "n_cycles": r.get("n", 0),
                "mean_delta": r.get("mean_delta", np.nan),
                "median_delta": r.get("median_delta", np.nan),
                "ci_low": r.get("wild_cluster_ci_low", np.nan),
                "ci_high": r.get("wild_cluster_ci_high", np.nan),
                "p_wild_cluster": r.get("wild_cluster_p_value", np.nan),
                "p_block_signflip": r.get("block_signflip_p_value", np.nan),
                "p_wilcoxon": r.get("wilcoxon_p_value", np.nan),
            }
        )

    compare(
        "baseline",
        f"baseline ({base_clock_name} clock, colony {colony})",
        base,
        "off_colony_frac_in_zone",
        "on_colony_frac_in_zone",
        0.0,
    )
    for name, (family, label, px) in meta.items():
        compare(
            family, label, base, f"off_colony_frac_in_zone__{name}", f"on_colony_frac_in_zone__{name}", px
        )
    compare(
        "colony cap",
        "no cap (uncapped fraction)",
        base,
        "off_resting_frac_in_zone",
        "on_resting_frac_in_zone",
    )
    for w in plan.on_windows:
        compare(
            "ON window",
            f"ON {windows[w][0]}-{windows[w][1]}",
            base,
            "off_colony_frac_in_zone",
            f"{w}_colony_frac_in_zone",
        )
    for w in plan.off_windows:
        compare(
            "OFF window",
            f"OFF {windows[w][0]}-{windows[w][1]}",
            base,
            f"{w}_colony_frac_in_zone",
            "on_colony_frac_in_zone",
        )
    speaker_only: dict[str, ZoneTest] = {"speaker": speaker}
    for c in plan.colonies:
        cyc = binned(base_clock, speaker_only, colony=c)
        compare("colony cap", f"colony {c}", cyc, "off_colony_frac_in_zone", "on_colony_frac_in_zone")
        tick(f"colony {c}")
    mm = px_per_mm if px_per_mm else 1.0 / MM_PER_PX_UNCALIBRATED
    for v in plan.walking_mm_s:
        cyc = binned(base_clock, speaker_only, colony=colony, walking_px_per_s=v * mm)
        note = "" if px_per_mm else " (uncalibrated: 0.4 mm/px assumed)"
        compare(
            "walking resters removed",
            f"> {v:g} mm/s{note}",
            cyc,
            "off_colony_frac_in_zone",
            "on_colony_frac_in_zone",
        )
        tick(f"walking {v:g} mm/s")
    for name, ec in clocks.items():
        if name == base_clock_name:
            continue
        cyc = binned(ec, speaker_only, colony=colony)
        compare("clock", f"{name} clock", cyc, "off_colony_frac_in_zone", "on_colony_frac_in_zone")
        tick(f"{name} clock")

    table = pd.DataFrame(rows)
    base_delta = float(table.iloc[0]["mean_delta"])
    with np.errstate(invalid="ignore", divide="ignore"):
        table["relative_change"] = (table["mean_delta"] - base_delta) / abs(base_delta)
    half_width = 0.5 * float(np.min(speaker.max(axis=0) - speaker.min(axis=0)))
    table["zone_change_rel_width"] = table["zone_change_px"] / (2 * half_width) if half_width > 0 else np.nan
    # Flags judge the robustness of an effect. With no clear baseline effect (its interval includes
    # 0) every variant "flips sign" by chance, so flags would only be noise.
    b_lo, b_hi = float(table.iloc[0]["ci_low"]), float(table.iloc[0]["ci_high"])
    baseline_effect = bool(np.isfinite(b_lo) and np.isfinite(b_hi) and (b_lo > 0 or b_hi < 0))
    table["flag_note"] = (
        "" if baseline_effect else "no flags: the baseline interval includes 0 (no clear effect to test)"
    )
    flags = []
    for _, r in table.iterrows():
        f = []
        if not baseline_effect:
            flags.append("")
            continue
        # moving or growing the zone by more than half its width takes it off the speaker: a sign
        # flip there is expected, not a sign of a fragile result
        small_zone_change = not str(r["family"]).startswith("zone") or abs(r["zone_change_px"]) <= half_width
        if (
            np.isfinite(r["mean_delta"])
            and np.sign(r["mean_delta"]) != np.sign(base_delta)
            and r["mean_delta"] != 0
            and small_zone_change
        ):
            f.append("sign flips")
        if (
            r["family"].startswith("zone")
            and abs(r["zone_change_px"]) <= 5
            and abs(r["relative_change"]) > 0.5
        ):
            f.append("changes by more than half within 5 px")
        flags.append("; ".join(f))
    table["flag"] = flags
    # BH over the primary cluster-robust p (wild cluster; the block test when too few cage-days)
    wild = table["p_wild_cluster"].to_numpy(dtype=np.float64)
    block = table["p_block_signflip"].to_numpy(dtype=np.float64)
    p = np.where(np.isfinite(wild), wild, block)
    table["p_method"] = np.where(
        np.isfinite(wild), "wild cluster", np.where(np.isfinite(block), "block sign-flip", "")
    )
    q = np.full(p.size, np.nan)
    fin = np.isfinite(p)
    if fin.any():
        q[fin] = benjamini_hochberg(p[fin])
    table["q_bh"] = q
    return table
