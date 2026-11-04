"""Check or draw the cage, the speaker zone and the control zones on a reference frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from pydantic import ValidationError

from buzzphono.app.common import experiments, page
from buzzphono.calibration import (
    CALIBRATION_FILE,
    Calibration,
    PhysicalSize,
    Zone,
    reference_images,
    save_calibration,
)
from buzzphono.config import plain_english
from buzzphono.errors import BuzzPhonoError
from buzzphono.pipeline import resolve_calibration
from buzzphono.plotting.style import SERIES

DISPLAY_W = 960

page("2 · Calibrate")
chosen = experiments()
if not chosen:
    st.info("Load an experiment first (page 1).")
    st.stop()
label = st.selectbox("Experiment", [e.get("label") or e["path"] for e in chosen])
exp = next(e for e in chosen if (e.get("label") or e["path"]) == label)
folder = Path(exp["path"])
refs = reference_images(folder)
bg = folder / "background_with_borders.png"
image_path = refs[len(refs) // 2] if refs else (bg if bg.is_file() else None)
if image_path is None:
    st.error("No reference frame (individual_images/*.png) in this experiment.")
    st.stop()
with Image.open(image_path) as im:
    img = im.convert("RGB")
w, h = img.size

key = f"cal_{folder}"
if key not in st.session_state:
    try:
        st.session_state[key] = resolve_calibration(folder).model_dump()
        st.caption(f"Loaded the existing zones ({CALIBRATION_FILE} or custom_zones.json).")
    except BuzzPhonoError as exc:
        st.warning(f"No calibration yet: {exc}")
        st.session_state[key] = {"frame_width_px": w, "frame_height_px": h, "cage": [], "zones": []}
cal = st.session_state[key]
if (cal.get("frame_width_px"), cal.get("frame_height_px")) != (w, h):
    st.error(
        f"The video frames are {w}×{h} px but this calibration is for {cal.get('frame_width_px')}×{cal.get('frame_height_px')} px. "
        "The camera or its resolution changed: draw the zones again for this recording."
    )

# ---- picture with every zone
fig = go.Figure(go.Image(z=np.asarray(img)))
shapes = [("cage", cal.get("cage", []), "#898781")] + [
    (z["name"], z["polygon"], SERIES[i % len(SERIES)]) for i, z in enumerate(cal.get("zones", []))
]
for name, pts, color in shapes:
    if len(pts) >= 2:
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                hovertemplate=f"{name}<br>x=%{{x:.0f}} y=%{{y:.0f}}<extra></extra>",
            )
        )
fig.update_layout(
    height=int(DISPLAY_W * h / w) + 40, margin={"l": 0, "r": 0, "t": 20, "b": 0}, legend={"orientation": "h"}
)
st.plotly_chart(fig, use_container_width=True)

# ---- edit one zone
st.subheader("Edit a zone")
names = ["cage"] + [z["name"] for z in cal.get("zones", [])]
c1, c2, c3 = st.columns([2, 2, 1])
target = c1.selectbox("Zone", [*names, "(new control zone)"])
if target == "(new control zone)":
    new_name = c2.text_input("Name of the new zone", value=f"control_{len(names)}")
    if c3.button("Create") and new_name and new_name not in names:
        cal.setdefault("zones", []).append({"name": new_name, "role": "control", "polygon": []})
        st.rerun()
    st.stop()
current = cal["cage"] if target == "cage" else next(z for z in cal["zones"] if z["name"] == target)["polygon"]
if target != "cage":
    zone = next(z for z in cal["zones"] if z["name"] == target)
    zone["role"] = c2.selectbox(
        "Role",
        ["speaker", "control", "other"],
        index=["speaker", "control", "other"].index(zone.get("role", "control")),
        help="Exactly one zone is the speaker. Controls are compared with it; 'other' zones (e.g. feeders) are only shown.",
    )
st.caption(
    "Click the picture to add a corner (in order around the zone), or type the corners in the table. The cage needs 4 corners: top-left, top-right, bottom-right, bottom-left."
)
try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    scale = w / DISPLAY_W
    click = streamlit_image_coordinates(
        img.resize((DISPLAY_W, int(h / scale))), key=f"click_{folder}_{target}"
    )
    last = st.session_state.get(f"last_click_{target}")
    if click and click != last:
        st.session_state[f"last_click_{target}"] = click
        current.append((round(click["x"] * scale, 1), round(click["y"] * scale, 1)))
        st.rerun()
except ImportError:
    st.info("Clicking needs the streamlit-image-coordinates package; type the corners in the table instead.")
st.caption(f"{len(current)} corner(s) in {target}.")
table = st.data_editor(
    pd.DataFrame(current, columns=["x_px", "y_px"]), num_rows="dynamic", key=f"verts_{folder}_{target}"
)
current[:] = [
    (float(r.x_px), float(r.y_px)) for r in table.itertuples() if pd.notna(r.x_px) and pd.notna(r.y_px)
]
b1, b2 = st.columns(2)
if b1.button("Clear this zone"):
    current.clear()
    st.rerun()
if target != "cage" and b2.button("Delete this zone"):
    cal["zones"] = [z for z in cal["zones"] if z["name"] != target]
    st.rerun()

# ---- size in mm and save
st.subheader("Cage size (optional)")
phys = cal.get("physical") or {}
m1, m2 = st.columns(2)
wmm = m1.number_input(
    "Inner width of the filmed cage face (mm)",
    min_value=0.0,
    value=float(phys.get("cage_width_mm", 0.0)),
    help="Lets BuzzPhono report distances and speeds in mm. Leave 0 for pixels.",
)
hmm = m2.number_input("Inner height (mm)", min_value=0.0, value=float(phys.get("cage_height_mm", 0.0)))
if st.button("Save calibration", type="primary"):
    data = dict(cal)
    data["frame_width_px"], data["frame_height_px"] = w, h
    data["physical"] = (
        PhysicalSize(cage_width_mm=wmm, cage_height_mm=hmm).model_dump() if wmm > 0 and hmm > 0 else None
    )
    data["reference_image"] = str(image_path.relative_to(folder)).replace("\\", "/")
    data.pop("created", None)
    data.pop("created_by", None)
    try:
        model = Calibration(
            **{**data, "zones": [Zone(**z) for z in data.get("zones", []) if len(z["polygon"]) >= 3]}
        )
    except (ValidationError, TypeError) as exc:
        st.error(plain_english(exc, "the calibration") if isinstance(exc, ValidationError) else str(exc))
    else:
        path = save_calibration(model, folder / CALIBRATION_FILE)
        st.success(f"Saved {path}")
