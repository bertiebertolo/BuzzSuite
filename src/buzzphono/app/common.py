"""Shared helpers for the BuzzPhono app pages: session state, recent experiments, and running jobs.

The app never analyses anything itself. "Run" writes the settings to the run folder and starts
``python -m buzzphono analyse`` as a separate process; the page then follows the run's
``status.json`` (see :mod:`buzzphono.jobs`). Closing the browser does not stop a run; the Cancel
button does.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from buzzphono import jobs

STATE_DIR = Path(os.environ.get("BUZZPHONO_HOME", Path.home() / ".buzzphono"))
RECENT = STATE_DIR / "recent.json"
ANALYSES = {
    "resting_fraction": "Speaker-zone resting fraction, ON vs OFF (the headline)",
    "attraction_index": "Attraction index (ON − OFF) / (ON + OFF)",
    "psth": "Time course around the tone onset",
    "zt_profile": "Time of day (ZT) profile and whole-recording timeline",
    "spatial": "Where they rest: occupancy maps and distance from the speaker",
    "approach": "Directed flight towards the speaker",
    "reliability": "How reliable is the effect; how many cage-days are needed",
    "sensitivity": "Does the result survive other reasonable choices?",
    "flight": "Flight phonotaxis (arousal, take-offs, landings, wall preference)",
    "comparison": "Compare experiments (tuning curve, groups)",
}


def page(title: str) -> None:
    st.set_page_config(page_title=f"BuzzPhono · {title}", page_icon="🦟", layout="wide")
    st.title(title)


def experiments() -> list[dict[str, Any]]:
    return st.session_state.setdefault("experiments", [])


def recent() -> list[dict[str, Any]]:
    try:
        data = json.loads(RECENT.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def remember(entry: dict[str, Any]) -> None:
    items = [e for e in recent() if e.get("path") != entry.get("path")]
    items.insert(0, entry)
    with contextlib.suppress(OSError):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        RECENT.write_text(json.dumps(items[:20], indent=1), encoding="utf-8")


def remember_run(run_dir: Path) -> None:
    runs = st.session_state.setdefault("runs", [])
    if str(run_dir) not in runs:
        runs.insert(0, str(run_dir))
    path = STATE_DIR / "runs.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = []
    items = [str(run_dir)] + [r for r in old if r != str(run_dir)]
    with contextlib.suppress(OSError):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items[:50], indent=1), encoding="utf-8")


def known_runs() -> list[str]:
    try:
        items = json.loads((STATE_DIR / "runs.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        items = []
    out = list(dict.fromkeys(st.session_state.get("runs", []) + items))
    return [r for r in out if Path(r).is_dir()]


def pick_folder(initial: str | None = None) -> str | None:  # pragma: no cover - opens a window
    """A native folder dialog, run in its own process so Streamlit's thread is never blocked by Tk."""
    code = (
        "import tkinter as tk, tkinter.filedialog as fd, sys;"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True);"
        "print(fd.askdirectory(initialdir=sys.argv[1] if len(sys.argv) > 1 else None) or '')"
    )
    args = [sys.executable, "-c", code] + ([initial] if initial else [])
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=600).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out or None


def new_run_dir(settings: dict[str, Any]) -> Path:
    base = Path(
        settings.get("output", {}).get("dir")
        or Path(settings["experiments"][0]["path"]) / "plots" / "buzzphono"
    )
    return base / datetime.now().strftime("%Y%m%d-%H%M%S-app")


def start_job(settings: dict[str, Any], run_dir: Path) -> subprocess.Popen[bytes]:
    """Write the settings into the run folder and start the analysis as its own process."""
    jobs.claim_run_dir(run_dir)
    cfg = run_dir / "settings.yml"
    cfg.write_text(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "buzzphono",
        "analyse",
        "--config",
        str(cfg),
        "--run-dir",
        str(run_dir),
        "--quiet",
    ]
    log = (run_dir / "process.log").open("wb")
    flags = getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )  # Windows: Ctrl-C in the app stays out of the run
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=os.name != "nt"
    )
    st.session_state["job"] = {"run_dir": str(run_dir), "pid": proc.pid, "started": time.time()}
    remember_run(run_dir)
    return proc


def stop_job(run_dir: Path, pid: int | None, grace_s: float = 10.0) -> None:
    """Ask the run to stop; if it is still alive after ``grace_s``, end its whole process tree."""
    jobs.request_cancel(run_dir)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        s = jobs.read_status(run_dir) or {}
        if s.get("state") in ("cancelled", "failed", "done"):
            return
        time.sleep(0.5)
    if pid:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                getattr(os, "killpg")(pid, 15)  # noqa: B009 (POSIX only)
        jobs.remove_partial_run(run_dir)


def settings_from_state() -> dict[str, Any]:
    s = dict(st.session_state.get("settings", {}))
    s["experiments"] = [{k: v for k, v in e.items() if v not in (None, "")} for e in experiments()]
    return s
