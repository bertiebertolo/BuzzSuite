"""BuzzSwarm heatmap colours (Plotting > Swarming > Colour): white -> each group's colour (the
default, for the grid and the paper-style heatmaps alike), white -> one chosen colour (the
Colour... picker, a name or a colour number), or a matplotlib colormap."""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import conftest  # noqa: F401  -- repo root on sys.path

from buzzswarm import clock_plots as cp
from buzzswarm import flight_maps as fm

SQUARE = [[100.0, 100.0], [700.0, 100.0], [700.0, 700.0], [100.0, 700.0]]
GEOMETRY = {e: {"centroid": [400.0, 400.0], "polygon": SQUARE, "swarm_radius_px": 100.0}
            for e in ("ExpA", "ExpB")}
WINDOW = "16:00-16:30"
WHITE = (1.0, 1.0, 1.0)


def _heatmap():
    """Share of flight per cell in one window for two groups, flight near the centre only."""
    rng = np.random.default_rng(3)
    rows = []
    for exp, group in (("ExpA", "A"), ("ExpB", "B")):
        cells = np.round(rng.normal(0, 60, size=(300, 2)) / 20.0) * 20.0
        df = pd.DataFrame({"x_px": cells[:, 0], "y_px": cells[:, 1], "n": 1.0})
        df = df.groupby(["x_px", "y_px"], as_index=False)["n"].sum()
        df["share"] = df["n"] / df["n"].sum()
        rows.append(df.drop(columns="n").assign(experiment=exp, group=group, date="2026-07-04",
                                                window=WINDOW))
    return pd.concat(rows, ignore_index=True)


def _panels(fig):
    """The heatmap images (colour bars have no image) and the number of colour bars."""
    ims = [ax.images[0] for ax in fig.axes if ax.images]
    return ims, len(fig.axes) - len(ims)


def test_colour_settings_are_checked():
    for name in ("", "group colour", "Group Colour"):
        assert cp.check_heat_colour(name) == cp.GROUP_COLOUR
    for name in ("magma", "Greys_r", "red", "tab:red", "#d62728", "C0"):
        assert cp.check_heat_colour(name) == name
    # colour numbers: hex without the #, or red, green, blue (0-255)
    for name in ("d62728", "D62728", "214, 39, 40", "214,39,40", "(214, 39, 40)", "214 39 40"):
        assert cp.check_heat_colour(name) == "#d62728", name
    for bad in ("nope", "none", "#12345", "256, 0, 0", "1, 2"):
        try:
            cp.check_heat_colour(bad)
            raise AssertionError("accepted %r" % bad)
        except ValueError as exc:
            assert "neither a colour" in str(exc)
    assert cp.one_heat_colour("red") == "#ff0000" and cp.one_heat_colour("214, 39, 40") == "#d62728"
    for name in ("group colour", "magma", "nope"):
        assert cp.one_heat_colour(name) is None
    assert fm.check_colour is cp.check_heat_colour and fm.GROUP_COLOUR == cp.GROUP_COLOUR


def test_grid_heatmaps_are_white_to_each_groups_colour_by_default():
    fig, _table = cp.heatmaps(_heatmap(), GEOMETRY)
    ims, n_bars = _panels(fig)
    colors = cp.group_colors(["A", "B"])
    assert len(ims) == 2 and n_bars == 2, "a colour bar for each group's panel"
    for im, group in zip(ims, ("A", "B")):
        assert np.allclose(im.cmap(0.0)[:3], WHITE), "empty cells are not white"
        assert np.allclose(im.cmap(1.0)[:3], to_rgb(colors[group]))
        assert (im.get_array() == 0).any() and im.get_clim()[0] == 0
    # one scale along the row, so the panels can be compared
    assert ims[0].get_clim() == ims[1].get_clim()
    # on white: a grey cage border and a black swarm zone
    ax = fig.axes[0]
    assert [ln.get_color() for ln in ax.lines[:2]] == ["0.45", "k"]
    assert "black = swarm zone" in fig._suptitle.get_text()


def test_one_colour_or_a_colormap():
    for colour in ("#d62728", "red"):
        fig, _table = cp.heatmaps(_heatmap(), GEOMETRY, colour)
        ims, n_bars = _panels(fig)
        assert n_bars == 1, "one colour bar for the row when every group has the same colours"
        for im in ims:
            assert np.allclose(im.cmap(0.0)[:3], WHITE)
            assert np.allclose(im.cmap(1.0)[:3], to_rgb(colour))
    fig, _table = cp.heatmaps(_heatmap(), GEOMETRY, "magma")
    ims, n_bars = _panels(fig)
    assert all(im.cmap.name == "magma" for im in ims) and n_bars == 1
    assert [ln.get_color() for ln in fig.axes[0].lines[:2]] == ["w", "c"]
    # one group alone: a single colour bar
    one = _heatmap()
    fig, _table = cp.heatmaps(one[one.group == "A"], GEOMETRY)
    assert _panels(fig)[1] == 1


def test_paper_heatmaps_take_one_colour():
    edges = np.linspace(-300.0, 300.0, 51)
    H = np.zeros((50, 50))
    H[20:30, 20:30] = 5.0
    heat = {(WINDOW, "A"): H, (WINDOW, "B"): H}
    colors = cp.group_colors(["A", "B"])
    for colour, tops in ((fm.GROUP_COLOUR, [colors["A"], colors["B"]]), ("#2ca02c", ["#2ca02c"] * 2)):
        fig = fm.paper_heatmaps(heat, {}, [WINDOW], ["A", "B"], colors, edges, colour)
        ims = [ax.images[0] for ax in fig.axes if ax.images]
        for im, top in zip(ims, tops):
            assert np.allclose(im.cmap(0.0)[:3], WHITE) and np.allclose(im.cmap(1.0)[:3], to_rgb(top))


def test_render_all_passes_the_colour_on():
    seen = []
    original = cp.heatmaps

    def spy(heatmap, geometry=None, colour=cp.GROUP_COLOUR):
        seen.append(colour)
        return original(heatmap, geometry, colour)

    out = tempfile.mkdtemp(prefix="heat_colour_")
    cp.heatmaps = spy
    try:
        tables = {"heatmap": _heatmap(), "geometry": GEOMETRY}
        paths = cp.render_all(tables, out, metrics=[], heat_colour="#9467bd")
        paths += cp.render_all(tables, out, metrics=[])
    finally:
        cp.heatmaps = original
        shutil.rmtree(out, ignore_errors=True)
    assert seen == ["#9467bd", cp.GROUP_COLOUR]
    assert sum(os.path.basename(p) == "heatmaps.png" for p in paths) == 2


def test_tab_picks_or_takes_a_colour_number():
    import tkinter as tk
    from tkinter import ttk
    import buzzswarm_tab_manager as btm

    class EM(object):
        config = {"buzzswarm_view": {"heat_colour": "group colour"}}

        def save_config(self, config):
            self.saved = dict(config)

        def get_recent_experiments(self):
            return []

    class UI(object):
        experiment_manager = EM()

        def log(self, msg):
            pass

        def show_error(self, title, msg):
            raise AssertionError(msg)

    asked = []

    def askcolor(color=None, **kw):
        asked.append(color)
        return (214, 39, 40), "#d62728"

    root = tk.Tk()
    root.withdraw()
    real = btm.colorchooser.askcolor
    btm.colorchooser.askcolor = askcolor
    try:
        tab = btm.BuzzswarmTabManager(root, UI(), None)
        frame = ttk.Frame(root)
        tab.init_buzzswarm_tab(frame)
        tab._ensure_engine_widgets()
        blank = tab.heat_swatch.cget("background")
        assert "red" in tab.heat_colour_combo["values"]
        tab._on_pick_colour()
        assert asked == ["#1f77b4"], "the picker opens on a default colour"
        assert tab.heat_colour_var.get() == "#d62728"
        assert tab.heat_swatch.cget("background") == "#d62728"
        tab._on_pick_colour()
        assert asked[-1] == "#d62728", "the picker opens on the chosen colour"
        tab.heat_colour_var.set("44, 160, 44")
        assert tab.heat_swatch.cget("background") == "#2ca02c"
        tab.heat_colour_var.set("magma")
        assert tab.heat_swatch.cget("background") == blank
        # cancelling the picker keeps the colour
        btm.colorchooser.askcolor = lambda **kw: (None, None)
        tab._on_pick_colour()
        assert tab.heat_colour_var.get() == "magma"
    finally:
        btm.colorchooser.askcolor = real
        root.destroy()
