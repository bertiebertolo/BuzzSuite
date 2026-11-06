"""A plain-English checklist of anything that would spoil the analysis."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from buzzphono.app.common import experiments, page
from buzzphono.validate import validate_experiment

page("3 · Validate")
chosen = experiments()
if not chosen:
    st.info("Load an experiment first (page 1).")
    st.stop()
st.write(
    "Optional: the Pi's settings and playback log make the checks stronger "
    "(wrong home folder, failed playback, frame size)."
)
c1, c2 = st.columns(2)
pi_config = c1.text_input("Pi config.ini (path)", key="pi_config")
playback = c2.text_input("Playback log: sound.log or the Pi log zip (path)", key="playback_log")
icons = {"ok": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌"}
any_error = False
for e in chosen:
    st.subheader(e.get("label") or e["path"])
    checks = validate_experiment(
        Path(e["path"]),
        pi_config=Path(pi_config) if pi_config else None,
        playback_logs=[Path(playback)] if playback else [],
        frequency_hz=e.get("frequency_hz"),
    )
    for c in checks:
        st.write(f"{icons.get(c.severity, '•')} **{c.area}** — {c.message}")
    any_error |= any(c.severity == "error" for c in checks)
if any_error:
    st.error("Fix the ❌ problems before analysing.")
else:
    st.success("Ready to analyse (page 4).")
