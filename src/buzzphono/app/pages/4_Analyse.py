"""Choose the analyses and run them; follow the progress or cancel."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from buzzphono.app.common import (
    ANALYSES,
    experiments,
    job_status,
    new_run_dir,
    page,
    settings_from_state,
    start_job,
    stop_job,
)
from buzzphono.config import build_config
from buzzphono.errors import BuzzPhonoError

page("4 · Analyse")
chosen = experiments()
if not chosen:
    st.info("Load an experiment first (page 1).")
    st.stop()
s = st.session_state.setdefault("settings", {})
picked = st.multiselect(
    "Analyses",
    list(ANALYSES),
    default=s.get("analyses", ["resting_fraction", "psth", "zt_profile", "reliability"]),
    format_func=lambda k: f"{k.replace('_', ' ')} — {ANALYSES[k]}",
)
with st.expander("Settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    on = c1.slider(
        "Tone ON (minutes of each hour)",
        0,
        60,
        (40, 50),
        help="When the tone played; 40-50 on the Penzance and Bangkok rigs.",
    )
    off = c2.slider(
        "Baseline OFF",
        0,
        60,
        (28, 38),
        help="Compared with ON; 28-38 keeps the same position within the 20-min chunk.",
    )
    colony = c3.number_input(
        "Mosquitoes in the cage",
        1,
        1000,
        30,
        help="Caps the per-frame resting count (the tracker over-detects).",
    )
    c4, c5, c6 = st.columns(3)
    clock = c4.selectbox(
        "Clock",
        ["corrected", "nominal", "thesis"],
        help="corrected: fitted on the tone-evoked flight burst (recommended). thesis: reproduces the published numbers.",
    )
    alt = c5.selectbox(
        "Test",
        ["two-sided", "greater", "less"],
        help="two-sided unless the direction was fixed before seeing the data.",
    )
    workers = c6.number_input(
        "Processes reading files", 1, 61, 4, help="4 suits a hard drive; more only helps on an SSD."
    )
    out = st.text_input("Results folder (empty = inside the first experiment's plots/buzzphono)", value="")
s["analyses"] = picked
s["stimulus"] = {**s.get("stimulus", {}), "on": list(on), "off": list(off)}
s["metric"] = {**s.get("metric", {}), "colony_size": int(colony)}
s["clock"] = {**s.get("clock", {}), "mode": clock}
s["stats"] = {**s.get("stats", {}), "alternative": alt}
s["workers"] = int(workers)
s["output"] = (
    {**s.get("output", {}), "dir": out}
    if out
    else {k: v for k, v in s.get("output", {}).items() if k != "dir"}
)
settings = settings_from_state()
valid = True
try:
    build_config(settings, "these settings")
except BuzzPhonoError as exc:
    st.error(str(exc))
    valid = False
job = st.session_state.get("job")
running = job is not None and job_status(job).get("state") in ("running", "starting")
if st.button("Run", type="primary", disabled=not valid or not picked or running):
    try:
        start_job(settings, new_run_dir(settings))
    except BuzzPhonoError as exc:
        st.error(exc.message)
        if exc.hint:
            st.info(exc.hint)
    else:
        st.rerun()


@st.fragment(run_every="1s")
def follow() -> None:
    job = st.session_state.get("job")
    if not job:
        return
    run_dir = Path(job["run_dir"])
    status = job_status(job)
    st.progress(float(status.get("progress", 0.0)), text=str(status.get("message", "")))
    state = status.get("state")
    if state in ("running", "starting"):
        if st.button("Cancel", key="cancel"):
            stop_job(run_dir, job.get("pid"))
            st.rerun()
    elif state == "done":
        st.success(f"Done. Results in {run_dir} — see page 5.")
    elif state == "cancelled":
        st.warning("Cancelled; partial results were deleted.")
    elif state == "failed":
        st.error(f"The run failed: {status.get('error', 'see process.log in the run folder')}")
        if status.get("hint"):
            st.info(status["hint"])


follow()
