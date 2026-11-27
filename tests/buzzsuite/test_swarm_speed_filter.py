"""BuzzSwarm's speed filter switch (Plotting > Swarming > "Speed filter"): on (the default) only real
flight counts (smoothed speed >= the camera's fly_px_frame); off, every tracked flight point counts.
Only the frame-level metrics, the centre profile and the heatmap cells change; the tracks-based
metrics and the day caches do not."""
import json
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import conftest  # noqa: F401  -- repo root on sys.path

from buzzswarm import clock_analysis as ca

CENTROID = (400.0, 400.0)
SQUARE = [[100.0, 100.0], [700.0, 100.0], [700.0, 700.0], [100.0, 700.0]]
DATE = "2026-07-04"
WINDOW = "16:00-16:30"
NS = 10**9
FPS = 25.0
FEEDER = (650.0, 400.0)          # 250 px from the centre, beyond every fast flight
RESTING = (150.0, 150.0)


def _day(seed=5):
    """Between 16:00 and 16:10: fast flights circling within 200 px of the centre (9 px/frame),
    slow hovering at a 'feeder' (1.5 px/frame) and near-still pieces in a corner."""
    rng = np.random.default_rng(seed)
    d0 = np.datetime64(DATE, "ns").astype(np.int64)
    coords, times, offsets = [], [], [0]

    def piece(xy):
        n = len(xy)
        start = d0 + 16 * 3600 * NS + int(rng.uniform(0, 600) * NS)
        coords.append(xy)
        times.append(start + np.arange(n, dtype=np.int64) * NS // int(FPS))
        offsets.append(offsets[-1] + n)

    for i in range(240):
        n, r0 = 100, rng.uniform(15, 200)
        ang = rng.uniform(0, 2 * np.pi) + np.arange(n) * 9.0 / r0
        piece(np.c_[CENTROID[0] + r0 * np.cos(ang), CENTROID[1] + r0 * np.sin(ang)])
    for i in range(80):
        n = 80
        ang = rng.uniform(0, 2 * np.pi) + np.arange(n) * 1.5 / 10.0
        piece(np.c_[FEEDER[0] + 10 * np.cos(ang), FEEDER[1] + 10 * np.sin(ang)])
    for i in range(40):
        piece(np.asarray(RESTING) + rng.normal(0, 0.3, (60, 2)))
    pop = {"coords": np.vstack(coords), "t_ns": np.concatenate(times), "offsets": np.asarray(offsets)}
    return {"date": DATE, "fps": FPS, "spans": np.array([[d0, d0 + 86400 * NS]], dtype=np.int64),
            "tracks": pop, "flight": pop}


def _metrics(speed_filter=None):
    geo = ca.geometry_from(CENTROID, SQUARE, 100)
    if speed_filter is not None:
        geo["speed_filter"] = speed_filter
    return ca.day_metrics(_day(), geo, bin_minutes=30, windows=[WINDOW])


def test_speed_filter_off_counts_every_flight_point():
    default, on, off = _metrics(), _metrics(True), _metrics(False)
    for key in ("windows", "bins", "heatmap", "profile", "minutes"):
        assert pd.DataFrame(default[key]).equals(pd.DataFrame(on[key])), "the default is not 'on': %s" % key
    w_on, w_off = on["windows"][0], off["windows"][0]
    # off: every point in the window, per recorded frame
    day = _day()
    d0 = np.datetime64(DATE, "ns").astype(np.int64)
    frame_ns = NS / FPS
    f = np.rint((day["flight"]["t_ns"] - d0) / frame_ns).astype(np.int64)
    f_lo, f_hi = int(round(16 * 3600 * NS / frame_ns)), int(round(16.5 * 3600 * NS / frame_ns))
    expected = np.sum((f >= f_lo) & (f < f_hi)) / float(f_hi - f_lo)
    assert math.isclose(w_off["n_flying"], expected, rel_tol=1e-9), (w_off["n_flying"], expected)
    assert w_off["n_flying"] > w_on["n_flying"] * 1.2
    # the slow points lower the median speed; the turning rate is still measured
    assert np.isfinite(w_off["speed_px_s"]) and w_off["speed_px_s"] < w_on["speed_px_s"]
    assert np.isfinite(w_on["turning_deg_s"]) and np.isfinite(w_off["turning_deg_s"])
    # the heatmap: the feeder and the resting corner appear only without the filter
    def far_share(rows):
        return sum(h["share"] for h in rows if math.hypot(h["x_px"], h["y_px"]) > 230)
    assert far_share(on["heatmap"]) == 0
    assert far_share(off["heatmap"]) > 0.1
    for rows in (on["heatmap"], off["heatmap"]):
        assert math.isclose(sum(h["share"] for h in rows), 1.0, rel_tol=1e-9)
    # flights >= 2 s are the same either way
    for k in ("activity", "r50", "prop_within_100", "prop_within_150", "straightness", "n_traj"):
        assert w_on[k] == w_off[k], k


class _Exp(object):
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="bs_speed_")
        self.folder = os.path.join(self.root, "Penzance", "ExpA")
        os.makedirs(self.folder)
        with open(os.path.join(self.folder, "buzzwatch_track_settings.yml"), "w") as f:
            f.write("cage_border_points: %s\n" % json.dumps(SQUARE))
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)


def _run(e, **kwargs):
    saved = {k: getattr(ca, k) for k in ("list_segments", "day_files", "load_or_build_day")}
    ca.list_segments = lambda folder: []
    ca.day_files = lambda segments, date: ["x"]
    ca.load_or_build_day = lambda *a, **k: _day()
    try:
        return ca.run([{"folder": e.folder, "group": "G", "days": [DATE]}], 30, [WINDOW],
                      out_root=os.path.join(e.root, "out"), workers=1, **kwargs)
    finally:
        for k, v in saved.items():
            setattr(ca, k, v)


def test_run_records_the_choice_and_the_plots_say_so():
    from buzzswarm import clock_plots as cp
    with _Exp() as e:
        on = _run(e)
        assert on["geometry"]["Penzance_ExpA"]["speed_filter"] is True
        assert cp.speed_filter_note(on["geometry"]) is None
        off = _run(e, speed_filter=False)
        with open(os.path.join(e.root, "out", "Penzance_ExpA", "geometry.json")) as f:
            assert json.load(f)["speed_filter"] is False
        assert off["windows"].n_flying.iloc[0] > on["windows"].n_flying.iloc[0]
        back = ca.load_outputs([os.path.join(e.root, "out", "Penzance_ExpA")], 30)
        assert cp.speed_filter_note(back["geometry"]).startswith("Speed filter off:")
        paths = cp.render_all(back, os.path.join(e.root, "fig"), metrics=["n_flying"])
        assert paths and all(os.path.isfile(p) for p in paths)
    # several experiments, some with the filter off: the note names them
    mixed = {"A": {"speed_filter": False}, "B": {"speed_filter": True}, "C": {}}
    assert cp.speed_filter_note(mixed).startswith("Speed filter off for A:")


def test_tab_saves_the_switch_and_passes_it_on():
    import tkinter as tk
    from tkinter import ttk
    import buzzswarm_tab_manager as btm

    class EM(object):
        config = {"buzzswarm_view": {"speed_filter": False}}

        def save_config(self, config):
            self.saved = dict(config)

        def get_recent_experiments(self):
            return []

    class UI(object):
        experiment_manager = EM()
        errors = []

        def log(self, msg):
            pass

        def show_error(self, title, msg):
            self.errors.append(msg)

        def set_status(self, **kwargs):
            pass

    root = tk.Tk()
    root.withdraw()
    try:
        tab = btm.BuzzswarmTabManager(root, UI(), None)
        tab.init_buzzswarm_tab(ttk.Frame(root))
        tab._ensure_engine_widgets()
        assert tab.speed_filter_var.get() is False, "the saved setting was not restored"
        with _Exp() as e:
            tab.entries = [{"folder": e.folder, "alias": "A", "group": "A", "coverage": {DATE: 1.0},
                            "included": {DATE: True}, "night_start": 17, "night_end": 5}]
            tab._sync_date_widgets_to_entry = lambda: None
            assert tab._snapshot()["view"]["speed_filter"] is False
            tab.speed_filter_var.set(True)
            assert tab._snapshot()["view"]["speed_filter"] is True
            assert UI.experiment_manager.saved["buzzswarm_view"]["speed_filter"] is True
            # Redraw with results made the other way says that Run analysis is needed
            tab._on_done("redraw", {"paths": [], "dir": e.root, "speed_filter_other": ["A"]})
            assert "Run analysis to apply the new setting" in tab.status_var.get()
        # a fresh install defaults to on
        UI.experiment_manager.config = {}
        tab2 = btm.BuzzswarmTabManager(root, UI(), None)
        tab2.init_buzzswarm_tab(ttk.Frame(root))
        assert tab2.speed_filter_var.get() is True
    finally:
        root.destroy()
