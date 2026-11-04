"""Publication figures, each drawn from one tidy result table (so a figure can always be redrawn
from its CSV). Every function returns a matplotlib Figure; :func:`buzzphono.plotting.style.save`
writes it as PNG (300 dpi), SVG and PDF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.plotting.style import (
    DIVERGING,
    INK,
    INK_2,
    MUTED,
    OFF,
    ON,
    SEQUENTIAL,
    SERIES,
    SHADE,
    figure,
    styled,
)

LABELS = {
    "colony_frac_in_zone": "Speaker-zone resters (fraction of colony)",
    "rest_in_zone_per_frame": "Resting detections in speaker zone per frame",
    "fly_px_per_s": "Flight activity (px/s)",
    "moving_per_frame": "Moving detections per frame",
}


def _mean_ci(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size else np.nan), np.nan
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(v.size))


@styled
def paired_on_off(cycles: pd.DataFrame, off_col: str, on_col: str, *, ylabel: str, title: str = "") -> Any:
    """Each hour-cycle as a grey line from OFF to ON; the means with 95% intervals on top."""
    fig, ax = figure(70, 70)
    off = cycles[off_col].to_numpy(dtype=float)
    on = cycles[on_col].to_numpy(dtype=float)
    ok = np.isfinite(off) & np.isfinite(on)
    for a, b in zip(off[ok], on[ok], strict=True):
        ax.plot([0, 1], [a, b], color=MUTED, linewidth=0.4, alpha=0.35, zorder=1)
    for x, v, c, lab in ((0, off[ok], OFF, "OFF"), (1, on[ok], ON, "ON")):
        m, ci = _mean_ci(v)
        ax.errorbar(
            [x],
            [m],
            yerr=[[ci], [ci]] if np.isfinite(ci) else None,
            fmt="o",
            color=c,
            markersize=7,
            capsize=4,
            linewidth=2,
            zorder=3,
            label=lab,
        )
    ax.set_xticks([0, 1], ["OFF", "ON"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"n = {int(ok.sum())} hour-cycles", loc="left")
    return fig


@styled
def delta_by_group(stats: pd.DataFrame, *, ylabel: str, group_col: str = "phase") -> Any:
    """Mean ON − OFF difference per group (e.g. light phase) with the wild-cluster 95% interval."""
    df = stats.reset_index(drop=True)
    fig, ax = figure(85, 65)
    x = np.arange(len(df))
    lo = df.get("wild_cluster_ci_low", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    hi = df.get("wild_cluster_ci_high", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    m = df["mean_delta"].to_numpy(dtype=float)
    err = np.vstack([m - lo, hi - m])
    err = np.where(np.isfinite(err), err, 0.0)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.errorbar(x, m, yerr=err, fmt="o", color=ON, markersize=7, capsize=4, linewidth=2)
    for xi, row in zip(x, df.itertuples(), strict=True):
        n = getattr(row, "n", np.nan)
        ax.annotate(
            f"n={int(n)}" if np.isfinite(n) else "",
            (xi, 0),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=MUTED,
        )
    ax.set_xticks(x, df[group_col].astype(str))
    ax.set_ylabel(ylabel)
    return fig


@styled
def psth_plot(
    table: pd.DataFrame, quantity: str, *, on_minutes: tuple[float, float] = (0.0, 10.0), phase: str = "all"
) -> Any:
    """Mean ± 95% band around the tone onset, with the sham onset as a control line."""
    fig, ax = figure(110, 65)
    sub = table[(table["quantity"] == quantity) & (table["phase"] == phase)]
    ax.axvspan(on_minutes[0], on_minutes[1], color=SHADE, zorder=0, label="tone")
    for event, color, label in (("sham_onset", MUTED, "sham onset"), ("onset", ON, "tone onset")):
        s = sub[sub["event"] == event]
        if s.empty:
            continue
        t = (s["rel_time_s"].to_numpy() + 15.0) / 60.0  # bin centres, minutes
        ax.fill_between(t, s["ci_low"], s["ci_high"], color=color, alpha=0.2, linewidth=0)
        ax.plot(t, s["mean"], color=color, label=label)
    ax.axvline(0, color=INK_2, linewidth=0.6)
    ax.set_xlabel("Time from onset (min)")
    ax.set_ylabel(LABELS.get(quantity, quantity))
    ax.legend(loc="upper left")
    return fig


@styled
def zt_profile_plot(profile: pd.DataFrame, *, ylabel: str, light_hours: tuple[int, int] = (5, 17)) -> Any:
    """ON − OFF difference per hour after lights-on (ZT), with 95% intervals and the dark phase shaded."""
    fig, ax = figure(110, 65)
    df = profile.sort_values("zt_hour")
    dark_from = (light_hours[1] - light_hours[0]) % 24
    ax.axvspan(dark_from - 0.5, 23.5, color=SHADE, zorder=0, label="dark")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.fill_between(df["zt_hour"], df["delta_ci_low"], df["delta_ci_high"], color=ON, alpha=0.2, linewidth=0)
    ax.plot(df["zt_hour"], df["delta_mean"], color=ON, marker="o", markersize=4, label="ON − OFF")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("Zeitgeber time (h after lights on)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return fig


@styled
def minute_series_plot(
    series: pd.DataFrame, *, value: str = "colony_frac_in_zone", window: str = "on"
) -> Any:
    """The whole recording minute by minute, with every tone window marked."""
    fig, ax = figure(175, 60)
    t = pd.to_datetime(series["minute_start"])
    on = (series["window"] == window).to_numpy()
    ax.plot(t, series[value], color=INK_2, linewidth=0.5)
    ax.scatter(t[on], series.loc[on, value], s=3, color=ON, label="tone on", zorder=3)
    ax.set_ylabel(LABELS.get(value, value))
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    return fig


@styled
def occupancy_maps_plot(
    maps: Mapping[str, np.ndarray],
    cell_px: float,
    *,
    speaker: np.ndarray,
    cage: np.ndarray,
    a: str = "on",
    b: str = "off",
) -> Any:
    """OFF, ON (shared sequential scale) and ON − OFF (diverging, symmetric around 0)."""
    fig, axes = figure(175, 60, ncols=3)
    seq_max = float(np.nanmax([np.nanmax(maps[b]), np.nanmax(maps[a])]))
    diff = maps[a] - maps[b]
    lim = float(np.nanmax(np.abs(diff))) or 1.0
    ny, nx = diff.shape
    extent = (0, nx * cell_px, ny * cell_px, 0)
    for ax, data, cmap, vmin, vmax, title in (
        (axes[0], maps[b], SEQUENTIAL, 0, seq_max, b.upper()),
        (axes[1], maps[a], SEQUENTIAL, 0, seq_max, a.upper()),
        (axes[2], diff, DIVERGING, -lim, lim, f"{a.upper()} − {b.upper()}"),
    ):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
        for poly, color in ((cage, MUTED), (speaker, INK)):
            p = np.vstack([poly, poly[:1]])
            ax.plot(p[:, 0], p[:, 1], color=color, linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(
            im, ax=ax, shrink=0.7, label="Δ resters/frame/cell" if data is diff else "resters/frame/cell"
        )
    return fig


@styled
def radial_profile_plot(profile: pd.DataFrame, *, a: str = "on", b: str = "off") -> Any:
    fig, ax = figure(85, 65)
    x = (profile["distance_from_px"] + profile["distance_to_px"]) / 2
    for w, color in ((b, OFF), (a, ON)):
        ax.plot(
            x, profile[f"{w}_per_frame_per_1000px2"], color=color, marker="o", markersize=3, label=w.upper()
        )
    ax.set_xlabel("Distance from the speaker zone (px)")
    ax.set_ylabel("Resters per frame per 1,000 px²")
    ax.legend()
    return fig


@styled
def sensitivity_plot(table: pd.DataFrame) -> Any:
    """Forest plot: the mean ON − OFF difference and 95% interval under each analysis choice."""
    df = table.reset_index(drop=True)
    fig, ax = figure(120, max(50.0, 6.0 * len(df) + 20))
    y = np.arange(len(df))[::-1]
    base = float(df.iloc[0]["mean_delta"])
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.axvline(base, color=MUTED, linewidth=0.6, linestyle="--")
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    m = df["mean_delta"].to_numpy(dtype=float)
    flagged = df.get("flag", pd.Series("", index=df.index)).fillna("").astype(str).str.len().gt(0).to_numpy()
    for yi, mi, low, high, f in zip(y, m, lo, hi, flagged, strict=True):
        color = SERIES[7] if f else INK_2
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [yi, yi], color=color, linewidth=1.5)
        ax.plot([mi], [yi], "o", color=color, markersize=5)
    labels = [
        f"{fam}: {var}" + (" (flagged)" if f else "")
        for fam, var, f in zip(df["family"], df["variant"], flagged, strict=True)
    ]
    ax.set_yticks(y, labels, fontsize=7)
    for tick, f in zip(ax.get_yticklabels(), flagged, strict=True):
        if f:
            tick.set_color(SERIES[7])
    ax.grid(axis="x")
    ax.set_xlabel("ON − OFF (speaker-zone fraction); dashed line = baseline")
    return fig


@styled
def tuning_plot(table: pd.DataFrame, *, x: str = "frequency_hz", ylabel: str = "ON − OFF") -> Any:
    fig, ax = figure(85, 65)
    df = table.dropna(subset=[x]).sort_values(x)
    m = df["mean_delta"].to_numpy(dtype=float)
    err = np.vstack([m - df["ci_low"].to_numpy(dtype=float), df["ci_high"].to_numpy(dtype=float) - m])
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.errorbar(df[x], m, yerr=np.where(np.isfinite(err), err, 0), fmt="o", color=ON, capsize=4, markersize=6)
    for _, r in df.iterrows():
        ax.annotate(
            str(r.get("label", "")),
            (r[x], r["mean_delta"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color=INK_2,
        )
    ax.set_xlabel("Tone frequency (Hz)" if x == "frequency_hz" else x)
    ax.set_ylabel(ylabel)
    return fig


@styled
def planning_plot(plan: pd.DataFrame, *, observed: float | None = None) -> Any:
    fig, ax = figure(85, 65)
    ax.plot(
        plan["cage_days"],
        plan["mde"],
        color=OFF,
        marker="o",
        markersize=4,
        label="minimum detectable ON − OFF",
    )
    if observed is not None and np.isfinite(observed):
        ax.axhline(abs(observed), color=ON, linestyle="--", linewidth=1, label="observed |ON − OFF|")
    ax.set_xlabel("Cage-days")
    ax.set_ylabel("Effect detectable with 80% power")
    ax.legend()
    return fig


@styled
def flight_summary_plot(
    stats: pd.DataFrame,
    metrics: Sequence[str] = ("flyers_per_frame", "takeoffs_per_min", "flight_px_per_frame"),
) -> Any:
    """Fold change ON / OFF of the whole-cage flight measures, per light phase."""
    sub = stats[
        (stats["pair"] == "ON vs OFF")
        & (stats["zone"] == "all")
        & (stats["metric"].isin(metrics))
        & (stats["grouping"].isin(["all", "light"]))
    ]
    fig, ax = figure(110, 65)
    groups = list(dict.fromkeys(sub["group"]))
    width = 0.8 / max(len(groups), 1)
    for gi, g in enumerate(groups):
        s = sub[sub["group"] == g].set_index("metric").reindex(list(metrics))
        x = np.arange(len(metrics)) + (gi - (len(groups) - 1) / 2) * width
        ax.bar(x, s["fold"], width=width * 0.9, color=SERIES[gi % len(SERIES)], label=g)
    ax.axhline(1, color=INK, linewidth=0.8)
    ax.set_xticks(np.arange(len(metrics)), [m.replace("_", " ") for m in metrics], fontsize=7)
    ax.set_ylabel("ON / OFF")
    ax.legend()
    return fig
