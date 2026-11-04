"""Explore a finished run: the key statistics and interactive figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from buzzphono.app.common import known_runs, page
from buzzphono.plotting.style import MUTED, OFF, ON

page("5 · Results")
runs = known_runs()
typed = st.text_input("Result folder", value=runs[0] if runs else "")
run = Path(typed) if typed else None
if run is None or not (run / "run_manifest.json").is_file():
    st.info("Choose a finished run (its folder holds run_manifest.json).")
    st.stop()
exp_root = run / "experiments"
exps = sorted(p.name for p in exp_root.iterdir() if p.is_dir()) if exp_root.is_dir() else []
if not exps:
    st.warning("This run has no experiment results.")
    st.stop()
label = st.selectbox("Experiment", exps)
folder = exp_root / str(label)


def read(name: str) -> pd.DataFrame | None:
    p = folder / f"{name}.csv"
    return pd.read_csv(p) if p.is_file() else None


st.subheader("Speaker-zone resting fraction")
stats = read("resting_fraction/statistics")
if stats is not None:
    cols = [
        "comparison",
        "phase",
        "n",
        "mean_off",
        "mean_on",
        "mean_delta",
        "wild_cluster_ci_low",
        "wild_cluster_ci_high",
        "wild_cluster_p_value",
        "wilcoxon_p_value",
    ]
    st.dataframe(stats[[c for c in cols if c in stats]], use_container_width=True, hide_index=True)
cycles = read("cycles/cycle_table")
if cycles is not None and len(cycles):
    good = cycles[cycles["usable"].astype(bool)]
    fig = go.Figure()
    for _, r in good.iterrows():
        fig.add_trace(
            go.Scatter(
                x=["OFF", "ON"],
                y=[r["off_colony_frac_in_zone"], r["on_colony_frac_in_zone"]],
                mode="lines",
                line={"color": MUTED, "width": 1},
                opacity=0.35,
                showlegend=False,
                hovertemplate=f"{r['cycle_start']}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Box(
            x=["OFF"] * len(good),
            y=good["off_colony_frac_in_zone"],
            name="OFF",
            marker_color=OFF,
            boxmean=True,
        )
    )
    fig.add_trace(
        go.Box(
            x=["ON"] * len(good), y=good["on_colony_frac_in_zone"], name="ON", marker_color=ON, boxmean=True
        )
    )
    fig.update_layout(yaxis_title="Speaker-zone resters (fraction of colony)", height=420)
    st.plotly_chart(fig, use_container_width=True)
psth = read("psth/psth")
if psth is not None and len(psth):
    st.subheader("Around the tone onset")
    q = st.selectbox("Measure", sorted(psth["quantity"].unique()))
    sub = psth[(psth["quantity"] == q) & (psth["phase"] == "all")]
    fig = go.Figure()
    for event, color in (("sham_onset", MUTED), ("onset", ON)):
        e = sub[sub["event"] == event]
        t = list((e["rel_time_s"] + 15) / 60)
        fig.add_trace(
            go.Scatter(
                x=t + t[::-1],
                y=list(e["ci_high"]) + list(e["ci_low"])[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(x=t, y=e["mean"], name=event.replace("_", " "), line={"color": color, "width": 2})
        )
    fig.update_layout(xaxis_title="Minutes from onset", yaxis_title=q, height=380)
    st.plotly_chart(fig, use_container_width=True)
zt = read("zt_profile/hour_of_day")
if zt is not None and len(zt):
    st.subheader("Time of day")
    z = zt.sort_values("zt_hour")
    err = {
        "type": "data",
        "symmetric": False,
        "array": z["delta_ci_high"] - z["delta_mean"],
        "arrayminus": z["delta_mean"] - z["delta_ci_low"],
    }
    fig = go.Figure(
        go.Scatter(x=z["zt_hour"], y=z["delta_mean"], error_y=err, mode="markers+lines", line={"color": ON})
    )
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_layout(xaxis_title="ZT (h after lights on)", yaxis_title="ON − OFF", height=360)
    st.plotly_chart(fig, use_container_width=True)
for name, title in (
    ("sensitivity/variants", "Sensitivity"),
    ("reliability/summary", "Reliability"),
    ("approach/statistics", "Directed flight"),
    ("attraction_index/statistics", "Attraction index"),
):
    tbl = read(name)
    if tbl is not None and len(tbl):
        st.subheader(title)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
fig_dir = folder / "figures"
figs = sorted(fig_dir.glob("*.png")) if fig_dir.is_dir() else []
if figs:
    st.subheader("Publication figures")
    cols2 = st.columns(2)
    for i, p in enumerate(figs):
        cols2[i % 2].image(str(p), caption=p.stem.replace("_", " "))
