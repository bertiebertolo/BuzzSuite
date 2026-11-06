"""Which figures a run draws, from which tables. A figure that cannot be drawn (too little data) is
skipped and the reason recorded, never fatal."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from buzzphono.plotting import static
from buzzphono.plotting.style import save

if TYPE_CHECKING:
    from buzzphono.config import AnalysisConfig
    from buzzphono.pipeline import ExperimentResult, RunResult

YLABEL = "Speaker-zone resters (fraction of colony)"


def _try(
    out: list[Path], res: Any, name: str, build: Any, base: Path, formats: Sequence[str], dpi: int
) -> None:
    try:
        fig = build()
    except (KeyError, ValueError, IndexError) as exc:
        res.warnings.append(f"figure {name} not drawn: {exc}")
        return
    out += save(fig, base / name, formats, dpi)


def experiment_figures(res: ExperimentResult, ctx: Any, config: AnalysisConfig) -> list[Path]:
    formats, dpi = list(config.output.formats), config.output.dpi
    out: list[Path] = []
    fig_dir = res.out_dir / "figures"
    t = res.tables
    good = ctx.good
    if len(good) >= 1:
        _try(
            out,
            res,
            "resting_on_vs_off",
            lambda: static.paired_on_off(
                good, "off_colony_frac_in_zone", "on_colony_frac_in_zone", ylabel=YLABEL, title=res.label
            ),
            fig_dir,
            formats,
            dpi,
        )
    st = t.get("resting_fraction/statistics")
    if st is not None and len(st):
        main = st[(st["comparison"] == "ON vs OFF (speaker zone)") & st["mean_delta"].notna()]
        if len(main):
            _try(
                out,
                res,
                "resting_delta_by_phase",
                lambda: static.delta_by_group(main, ylabel="ON − OFF (fraction of colony)"),
                fig_dir,
                formats,
                dpi,
            )
    if "psth/psth" in t and len(t["psth/psth"]):
        on = ctx.windows["on"]
        for q in ("colony_frac_in_zone", "fly_px_per_s"):
            _try(
                out,
                res,
                f"psth_{q}",
                lambda q=q: static.psth_plot(t["psth/psth"], q, on_minutes=(0.0, float(on[1] - on[0]))),
                fig_dir,
                formats,
                dpi,
            )
    if "zt_profile/hour_of_day" in t and len(t["zt_profile/hour_of_day"]):
        light = getattr(ctx, "light", None) or config.light  # the schedule the analyses used
        lh = (int(light.lights_on[:2]), int(light.lights_off[:2]))
        _try(
            out,
            res,
            "zt_profile",
            lambda: static.zt_profile_plot(
                t["zt_profile/hour_of_day"], ylabel="ON − OFF (fraction of colony)", light_hours=lh
            ),
            fig_dir,
            formats,
            dpi,
        )
        _try(
            out,
            res,
            "minute_series",
            lambda: static.minute_series_plot(t["zt_profile/minute_series"]),
            fig_dir,
            formats,
            dpi,
        )
    maps = res.extras.get("maps")
    if maps is not None:
        sp = ctx.cal.polygon(ctx.cal.speaker.name)
        _try(
            out,
            res,
            "occupancy_maps",
            lambda: static.occupancy_maps_plot(
                maps.maps, maps.cell_px, speaker=sp, cage=ctx.cal.cage_polygon()
            ),
            fig_dir,
            formats,
            dpi,
        )
        _try(
            out,
            res,
            "radial_profile",
            lambda: static.radial_profile_plot(t["spatial/radial_profile"]),
            fig_dir,
            formats,
            dpi,
        )
    if "sensitivity/variants" in t and len(t["sensitivity/variants"]):
        _try(
            out,
            res,
            "sensitivity",
            lambda: static.sensitivity_plot(t["sensitivity/variants"]),
            fig_dir,
            formats,
            dpi,
        )
    if "reliability/planning" in t and len(t["reliability/planning"]):
        obs = (
            float(np.nanmean(good["on_colony_frac_in_zone"] - good["off_colony_frac_in_zone"]))
            if len(good)
            else None
        )
        _try(
            out,
            res,
            "reliability_planning",
            lambda: static.planning_plot(t["reliability/planning"], observed=obs),
            fig_dir,
            formats,
            dpi,
        )
    if "flight/stats_flight_on_vs_off" in t and len(t["flight/stats_flight_on_vs_off"]):
        _try(
            out,
            res,
            "flight_summary",
            lambda: static.flight_summary_plot(t["flight/stats_flight_on_vs_off"]),
            fig_dir,
            formats,
            dpi,
        )
    return out


def pooled_figures(result: RunResult) -> list[Path]:
    cfg = result.config
    formats, dpi = list(cfg.output.formats), cfg.output.dpi
    out: list[Path] = []
    fig_dir = result.run_dir / "pooled" / "figures"
    exp = result.pooled.get("experiments")
    if exp is not None and exp["frequency_hz"].notna().sum() >= 2:
        _try(out, result, "tuning_curve", lambda: static.tuning_plot(exp), fig_dir, formats, dpi)
    plan = result.pooled.get("reliability_planning")
    if plan is not None and len(plan):
        _try(out, result, "reliability_planning", lambda: static.planning_plot(plan), fig_dir, formats, dpi)
    return out
