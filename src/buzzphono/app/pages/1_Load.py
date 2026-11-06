"""Choose experiments and describe them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from buzzphono.app.common import experiments, page, pick_folder, remember
from buzzphono.io.discover import discover

page("1 · Load experiments")
st.write("An experiment folder is BuzzWatch's analysis folder for one cage (it holds `tracking_resting/`).")
chosen = experiments()
c1, c2 = st.columns([4, 1])
path_text = c1.text_input(
    "Experiment folder", key="load_path", placeholder=r"e.g. E:\Buzzwatch\Analysis\Phonotaxis\Penzance\MyCage"
)
if c2.button("Browse…", help="Opens a folder window on this computer."):
    picked = pick_folder(path_text or None)
    if picked:
        st.session_state["load_path_pending"] = picked
        st.rerun()
if "load_path_pending" in st.session_state:
    path_text = st.session_state.pop("load_path_pending")
if st.button("Add experiment", type="primary", disabled=not path_text):
    p = Path(path_text).expanduser()
    if not p.is_dir():
        st.error(f"{p} is not a folder.")
    else:
        layout = discover(p)
        if not layout.chunks:
            st.error("No tracking files in this folder: track it with BuzzWatch first.")
        elif any(e["path"] == str(p) for e in chosen):
            st.info("Already loaded.")
        else:
            entry = {
                "path": str(p),
                "label": p.name,
                "group": None,
                "frequency_hz": None,
                "species": None,
                "sex": None,
            }
            chosen.append(entry)
            remember(entry)
            st.success(f"Added {p.name}: {len(layout.chunks)} chunks.")
if chosen:
    st.subheader("Loaded experiments")
    st.caption("Edit the descriptions here; they label the results and drive the experiment comparison.")
    df = pd.DataFrame(chosen, columns=["path", "label", "group", "frequency_hz", "species", "sex"])
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "path": st.column_config.TextColumn("Folder", disabled=True),
            "label": st.column_config.TextColumn("Label", help="Short name used in tables and figures."),
            "group": st.column_config.TextColumn(
                "Group", help="e.g. treatment or strain; experiments are compared by group."
            ),
            "frequency_hz": st.column_config.NumberColumn(
                "Tone (Hz)", min_value=1, help="Tone frequency of this experiment."
            ),
            "species": st.column_config.TextColumn("Species"),
            "sex": st.column_config.SelectboxColumn("Sex", options=["male", "female", "mixed"]),
        },
        key="exp_editor",
    )
    st.session_state["experiments"] = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in edited.to_dict("records")
        if row.get("path")
    ]
else:
    st.info("No experiment loaded yet.")
