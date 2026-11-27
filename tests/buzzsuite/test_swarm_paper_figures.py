"""BuzzSwarm's fruitless-paper style figures (buzzswarm/flight_maps.py): smooth heatmaps with a chosen
number of bins and colour, flight paths, the longest flights and GIFs of each clock window, all from
the engine's strict flights cut to the window (as fru2's flights_in_zt_window)."""
import json
import os
import shutil
import tempfile
import threading

import numpy as np

import conftest  # noqa: F401  -- repo root on sys.path

from buzzswarm import clock_analysis as ca
from buzzswarm import flight_maps as fm

SQUARE = [[100, 100], [700, 100], [700, 700], [100, 700]]
CENTROID = np.array([400.0, 400.0])
DATE = "2026-07-04"
WINDOWS = ["05:00-05:30", "16:30-17:00"]
NS = 10**9


def _day(seed=1):
    """Strict flights circling the centre around 05:10 and 16:40, plus some crossing 16:30."""
    rng = np.random.default_rng(seed)
    fps = 25.0
    d0 = np.datetime64(DATE, "ns").astype(np.int64)
    coords, times, offsets = [], [], [0]
    for i in range(60):
        n = int(rng.integers(60, 400))
        start_min = [5 * 60 + 10, 16 * 60 + 40, 16 * 60 + 29][i % 3]
        start = d0 + int((start_min * 60 + rng.uniform(0, 60)) * NS)
        r0 = rng.uniform(40, 250)
        ang = rng.uniform(0, 2 * np.pi) + np.arange(n) * 9.0 / r0
        coords.append(np.c_[CENTROID[0] + r0 * np.cos(ang), CENTROID[1] + r0 * np.sin(ang)])
        times.append(start + (np.arange(n, dtype=np.int64) * NS // int(fps)))
        offsets.append(offsets[-1] + n)
    pop = {"coords": np.vstack(coords), "t_ns": np.concatenate(times), "offsets": np.asarray(offsets)}
    return {"date": DATE, "fps": fps, "spans": np.array([[d0, d0 + 86400 * NS]]), "tracks": pop, "flight": pop}


class _Exp(object):
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="bs_paper_")
        self.folder = os.path.join(self.root, "Penzance", "ExpA")
        os.makedirs(self.folder)
        with open(os.path.join(self.folder, "buzzwatch_track_settings.yml"), "w") as f:
            f.write("cage_border_points: %s\n" % json.dumps(SQUARE))
        self.out = os.path.join(self.root, "figures")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)


def test_window_flights_are_cut_to_the_window():
    day = _day()
    fl = fm.window_flights(day["tracks"], DATE, "16:30-17:00")
    d0 = np.datetime64(DATE, "ns").astype(np.int64)
    lo, hi = d0 + (16 * 60 + 30) * 60 * NS, d0 + 17 * 60 * 60 * NS
    t = day["tracks"]["t_ns"]
    assert sum(len(f[1]) for f in fl) == int(np.sum((t >= lo) & (t < hi)))
    for xy, tt in fl:
        assert len(xy) == len(tt) >= 2 and tt.min() >= lo and tt.max() < hi
        assert np.all(np.diff(tt) > 0), "a flight piece mixes two flights"
    # the flights that started at 16:29 are cut at 16:30, not dropped
    assert any(tt[0] - lo < 60 * NS for _xy, tt in fl)
    assert fm.window_flights(day["tracks"], DATE, "12:00-12:30") == []


def test_paper_figures_heatmap_bins_colour_and_flights():
    with _Exp() as e:
        exps = [{"folder": e.folder, "alias": "Penzance_ExpA", "group": "Males", "days": [DATE]}]
        figs = fm.PaperFigures(exps, WINDOWS, e.out, bins=50, colour="viridis")
        figs.add_day(exps[0], DATE, _day())
        paths = figs.finish()
        names = [os.path.basename(p) for p in paths]
        assert names[0] == "heatmaps_paper.png", names
        assert "flights_Penzance_ExpA_2026-07-04.png" in names
        assert "top30_Penzance_ExpA_2026-07-04_1630-1700.png" in names
        assert all(os.path.isfile(p) for p in paths)
        npz = np.load(os.path.join(e.out, "heatmaps_paper_counts.npz"))
        assert len(npz["edges_px"]) == 51
        counts = [npz[k] for k in npz.files if k.startswith("counts_")]
        assert all(c.shape == (50, 50) for c in counts) and len(counts) == 2
        n_points = sum(len(xy) for w in WINDOWS for xy, _t in fm.window_flights(_day()["tracks"], DATE, w))
        assert sum(int(c.sum()) for c in counts) == n_points, "points fell outside the heatmap"
        csv = os.path.join(e.out, "flights", "flights_Penzance_ExpA_2026-07-04.csv")
        import pandas as pd
        table = pd.read_csv(csv)
        assert set(table.window) == set(WINDOWS) and (table.n_frames >= 2).all()


def test_settings_are_checked():
    assert fm.check_bins("1000") == 1000
    assert fm.check_colour("Group colour") == fm.GROUP_COLOUR
    assert fm.check_colour("Greys_r") == "Greys_r"
    for bad in ("x", 1, 5000):
        try:
            fm.check_bins(bad)
            raise AssertionError("accepted bins %r" % (bad,))
        except ValueError:
            pass
    try:
        fm.check_colour("not-a-colormap")
        raise AssertionError("accepted a bad colormap")
    except ValueError:
        pass
    try:
        fm.gif_frames(["05:00-05:30"], 0.5)      # 36000 frames
        raise AssertionError("accepted a GIF that long")
    except ValueError as exc:
        assert "speed-up of at least" in str(exc)


def test_gif_has_one_frame_per_step_and_can_be_cancelled():
    from PIL import Image
    with _Exp() as e:
        exps = [{"folder": e.folder, "alias": "A", "group": "A", "days": [DATE]}]
        figs = fm.PaperFigures(exps, ["16:30-16:40"], e.out, heatmaps=False, flights=False, gifs=True,
                               speedup=120)
        figs.add_day(exps[0], DATE, _day())
        (path,) = figs.finish()
        assert path.endswith(os.path.join("gifs", "A_2026-07-04_1630-1640.gif")), path
        img = Image.open(path)
        assert img.n_frames == fm.gif_frames(["16:30-16:40"], 120) == 50
        cancel = threading.Event()
        cancel.set()
        try:
            fm.make_gif([], DATE, "16:30-16:40", os.path.join(e.out, "x.gif"), "#1f77b4",
                        np.asarray(SQUARE, float) - CENTROID, 320, 120, cancel=cancel)
            raise AssertionError("not cancelled")
        except ca.Cancelled:
            pass


def test_run_hands_each_day_to_on_day():
    """run() calls on_day(experiment, date, day) once per analysed day, with alias and group."""
    with _Exp() as e:
        seen = []
        saved = {k: getattr(ca, k) for k in ("list_segments", "day_files", "load_or_build_day")}
        ca.list_segments = lambda folder: []
        ca.day_files = lambda segments, date: ["x"]
        ca.load_or_build_day = lambda *a, **k: _day()
        try:
            ca.run([{"folder": e.folder, "group": "G", "days": [DATE]}], 30, ["16:30-17:00"],
                   out_root=os.path.join(e.root, "out"), workers=1,
                   on_day=lambda exp, date, day: seen.append((exp["alias"], exp["group"], date,
                                                              len(day["tracks"]["t_ns"]))))
        finally:
            for k, v in saved.items():
                setattr(ca, k, v)
        assert seen == [("Penzance_ExpA", "G", DATE, len(_day()["tracks"]["t_ns"]))], seen


def test_tab_controls_are_saved_and_plots_labelled():
    """The tab builds headless; its heatmap bins, colour, flight paths and GIF speed-up go into the
    saved view; the new files get readable names in the plot list."""
    import tkinter as tk
    from tkinter import ttk
    import buzzswarm_tab_manager as btm

    class EM(object):
        config = {"buzzswarm_view": {"heat_bins": 200, "heat_colour": "viridis"}}

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

    root = tk.Tk()
    root.withdraw()
    try:
        tab = btm.BuzzswarmTabManager(root, UI(), None)
        frame = ttk.Frame(root)
        tab.init_buzzswarm_tab(frame)
        tab._ensure_engine_widgets()
        assert tab.heat_bins_var.get() == "200" and tab.heat_colour_var.get() == "viridis"
        assert fm.GROUP_COLOUR in tab.heat_colour_combo["values"]
        with _Exp() as e:
            tab.entries = [{"folder": e.folder, "alias": "A", "group": "A", "coverage": {DATE: 1.0},
                            "included": {DATE: True}, "night_start": 17, "night_end": 5}]
            tab._sync_date_widgets_to_entry = lambda: None
            tab.heat_bins_var.set("5000")
            tab.heat_colour_var.set("Greys_r")
            tab.gif_speedup_var.set("abc")
            snap = tab._snapshot()
            view = snap["view"]
            assert (view["heat_bins"], view["heat_colour"], view["gif_speedup"]) == (2000, "Greys_r", 60.0)
            assert view["flight_paths"] is True
            assert UI.experiment_manager.saved["buzzswarm_view"]["heat_colour"] == "Greys_r"
            tab.heat_colour_var.set("nope")
            assert tab._snapshot() is None and "neither a colour" in UI.errors[-1]
    finally:
        root.destroy()
    label = btm._plot_label
    assert label("x/heatmaps_paper.png").startswith("Heatmaps, paper style")
    assert label("x/flights/flights_Penzance_Exp_1_2026-07-04.png") == "Flight paths: Penzance_Exp_1 2026-07-04"
    assert label("x/flights/top30_Pz_E_2026-07-04_1630-1700.png") == "Top 30 flights: Pz_E 2026-07-04 16:30-17:00"
    assert label("x/gifs/Pz_E_2026-07-04_0500-0530.gif") == "GIF: Pz_E 2026-07-04 05:00-05:30"


def test_preview_plays_a_gif():
    import tkinter as tk
    from PIL import Image
    from image_preview import ImagePreview
    with _Exp() as e:
        path = os.path.join(e.root, "a.gif")
        frames = [Image.new("RGB", (40, 30), (i * 60, 0, 0)) for i in range(4)]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
        png = os.path.join(e.root, "b.png")
        frames[1].save(png)
        root = tk.Tk()
        root.withdraw()
        try:
            prev = ImagePreview(root)
            prev.show_path(path)
            assert prev._anim is not None and prev._anim_job is not None
            for want in (1, 2, 3, 0):
                prev._next_frame()
                assert prev._anim.tell() == want
            prev.show_path(png)                      # stops the animation and closes the GIF
            assert prev._anim is None and prev._anim_job is None
        finally:
            root.destroy()
