"""BuzzPhono app: home page."""

from __future__ import annotations

import streamlit as st

from buzzphono import __version__
from buzzphono.app.common import experiments, page, recent

page("BuzzPhono")
st.caption(f"Phonotaxis analysis of BuzzWatch mosquito tracking data · version {__version__}")
st.markdown(
    """
Does a sound draw mosquitoes to the speaker? BuzzPhono answers this from BuzzWatch's tracking data:
it counts how many resting mosquitoes sit in the speaker zone while the tone plays (ON) compared with
just before (OFF), every hour, and tests whether the difference is real.

**Work through the pages on the left, in order:**

1. **Load** – choose the experiment folder(s) and describe them (frequency, species, group).
2. **Calibrate** – check or draw the cage, the speaker zone and the control zones.
3. **Validate** – a plain-English checklist of anything that would spoil the analysis.
4. **Analyse** – choose the analyses and start; you can cancel at any time.
5. **Results** – the numbers and figures, explorable.
6. **Export** – the report and every table and figure in one zip file.

New to BuzzPhono? Run `buzzphono demo` in a terminal for a simulated experiment with a known effect,
then load `buzzphono_demo/tone` here.
"""
)
chosen = experiments()
if chosen:
    st.success(
        f"{len(chosen)} experiment(s) loaded: " + ", ".join(e.get("label") or e["path"] for e in chosen)
    )
items = recent()
if items:
    st.subheader("Recent experiments")
    for e in items[:10]:
        cols = st.columns([5, 1])
        cols[0].write(f"**{e.get('label') or e['path']}** — {e['path']}")
        if cols[1].button("Load", key=f"recent_{e['path']}"):
            if all(x["path"] != e["path"] for x in chosen):
                chosen.append(dict(e))
            st.rerun()
