"""Download the report, or everything in one zip file."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import streamlit as st

from buzzphono.app.common import known_runs, page

page("6 · Export")
runs = known_runs()
typed = st.text_input("Result folder", value=runs[0] if runs else "")
run = Path(typed) if typed else None
if run is None or not run.is_dir():
    st.info("Choose a run's result folder.")
    st.stop()
report = run / "report.html"
if report.is_file():
    st.download_button(
        "Download the report (HTML)",
        report.read_bytes(),
        file_name=f"{run.name}_report.html",
        mime="text/html",
    )


def zip_run(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.name not in ("CANCEL", "process.log"):
                zf.write(p, p.relative_to(folder.parent).as_posix())
    return buf.getvalue()


if st.button("Prepare a zip of the whole run"):
    st.session_state["zip"] = zip_run(run)
if "zip" in st.session_state:
    st.download_button(
        "Download the zip", st.session_state["zip"], file_name=f"{run.name}.zip", mime="application/zip"
    )
st.caption(
    "The zip holds every table (CSV), figure (PNG 300 dpi, SVG, PDF), the report and run_manifest.json."
)
