"""BuzzSwarm with a missing cage border (e.g. Penzance Ae. aegypti M, Jul 4-7): the run, the
swarm-zone picker and the CLI stop with a clear message and an offer to draw it, never a crash and
never a guessed border."""
import json
import os
import shutil
import tempfile

import conftest  # noqa: F401  -- repo root on sys.path

from buzzswarm import clock_analysis as ca

SQUARE = [[100, 100], [700, 100], [700, 700], [100, 700]]


class _Tmp(object):
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="bs_cage_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def _exp(root, name, settings_text):
    d = os.path.join(root, name)
    os.makedirs(d)
    if settings_text is not None:
        with open(os.path.join(d, "buzzwatch_track_settings.yml"), "w") as f:
            f.write(settings_text)
    return d


def test_cage_border_problem_reasons():
    with _Tmp() as t:
        good = _exp(t, "good", "cage_border_points: %s\n" % json.dumps(SQUARE))
        cases = {
            "no_yml": (_exp(t, "no_yml", None), "no buzzwatch_track_settings.yml"),
            "no_key": (_exp(t, "no_key", "fps: 25\n"), "never been drawn"),
            "empty": (_exp(t, "empty", "cage_border_points: []\n"), "never been drawn"),
            "two": (_exp(t, "two", "cage_border_points: [[1, 2], [3, 4]]\n"), "at least 3 points"),
            "ragged": (_exp(t, "ragged", "cage_border_points: [[1, 2], [3], [5, 6]]\n"), "at least 3 points"),
            "bad_yaml": (_exp(t, "bad_yaml", "cage_border_points: [[1, 2\n"), "could not be read"),
            "not_dict": (_exp(t, "not_dict", "- 1\n- 2\n"), "never been drawn"),
        }
        assert ca.cage_border_problem(good) is None
        centroid, polygon = ca.cage_geometry(good)
        assert polygon.shape == (4, 2)
        assert abs(centroid[0] - 400) < 1e-6 and abs(centroid[1] - 400) < 1e-6
        for name, (d, expect) in cases.items():
            why = ca.cage_border_problem(d)
            assert why and expect in why, (name, why)
            try:
                ca.cage_geometry(d)
                raise AssertionError("cage_geometry accepted %s" % name)
            except ca.CageBorderMissing as exc:
                assert isinstance(exc, ValueError)          # old callers catch ValueError
                assert exc.problems == [(d, why)]
                assert "Draw Cage Borders" in str(exc)


def test_run_checks_every_border_before_any_work():
    """One experiment without a border stops the run up front (naming it), before any day of the
    others is loaded, instead of failing half way through a long run."""
    with _Tmp() as t:
        good = _exp(t, "good", "cage_border_points: %s\n" % json.dumps(SQUARE))
        bad = _exp(t, "penzance", "cage_border_points: []\n")
        calls = []
        orig = ca.load_or_build_day
        ca.load_or_build_day = lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("loaded"))
        try:
            try:
                ca.run([{"folder": good, "alias": "good", "days": ["2026-07-04"]},
                        {"folder": bad, "alias": "penzance", "days": ["2026-07-04"]}])
                raise AssertionError("run did not stop")
            except ca.CageBorderMissing as exc:
                assert [d for d, _ in exc.problems] == [bad]
        finally:
            ca.load_or_build_day = orig
        assert calls == [], "a day was loaded before the border check"


def test_cli_swarm_clock_stops_with_message():
    import buzzsuite_cli
    with _Tmp() as t:
        bad = _exp(t, "penzance", "cage_border_points: []\n")
        os.makedirs(os.path.join(bad, "final_tracking_data"))
        orig = (ca.date_coverage, ca.default_days)
        ca.date_coverage = lambda d: {"2026-07-04": 1.0}
        ca.default_days = lambda cov, sch: ["2026-07-04"]
        try:
            buzzsuite_cli.main(["swarm", "-e", bad, "--which", "clock"])
            raise AssertionError("the CLI did not stop")
        except SystemExit as exc:
            msg = str(exc.code)
            assert "No cage border saved" in msg and bad in msg and "never been drawn" in msg, msg
        finally:
            ca.date_coverage, ca.default_days = orig


# ---------------------------------------------------------------- the Swarming tab (no window shown)
class _Notebook(object):
    def __init__(self, titles):
        self._titles = dict(("t%d" % i, s) for i, s in enumerate(titles))
        self.selected = None

    def tabs(self):
        return list(self._titles)

    def tab(self, tab_id, key):
        return self._titles[tab_id]

    def select(self, tab_id=None):
        self.selected = self._titles.get(tab_id)


class _EM(object):
    def __init__(self):
        self.settings = {"cage_border_points": SQUARE}      # the previously opened experiment's
        self.loaded = []
        self.experiment_alias = None

    def load_experiment_from_json(self, path):
        self.loaded.append((path, self.settings))
        self.experiment_alias = "penzance"


class _UI(object):
    def __init__(self):
        self.experiment_manager = _EM()
        self.errors, self.logs, self.status = [], [], []
        self.notebook = _Notebook(["1 · Setup", "2 · Analysis", "3 · Plotting"])
        self.setup_notebook = _Notebook(["Video", "Background & Cage Border"])
        self.updated = 0

    def show_error(self, title, message):
        self.errors.append((title, message))

    def log(self, msg):
        self.logs.append(msg)

    def set_status(self, **kw):
        self.status.append(kw)

    def update_ui_after_loading_experiment(self):
        self.updated += 1


def _tab(entries):
    import buzzswarm_tab_manager as btm
    tab = btm.BuzzswarmTabManager.__new__(btm.BuzzswarmTabManager)
    tab.ui_manager = _UI()
    tab.experiment_manager = tab.ui_manager.experiment_manager
    tab.log = tab.ui_manager.log
    tab.root = None
    tab.entries = entries
    tab._status = []
    tab._set_status = tab._status.append
    tab._selected_entry_idx = lambda: 0
    return tab


def _patch_askyesno(answer):
    from tkinter import messagebox
    asked = []
    orig = messagebox.askyesno
    messagebox.askyesno = lambda title, msg, **kw: asked.append((title, msg)) or answer
    return asked, lambda: setattr(messagebox, "askyesno", orig)


def test_swarm_zone_picker_offers_to_draw_the_border():
    import swarm_zone_dialog
    with _Tmp() as t:
        bad = _exp(t, "penzance", "cage_border_points: []\n")
        tab = _tab([{"folder": bad, "alias": "Penzance Ae. aegypti M"}])
        opened = []
        orig_dialog = swarm_zone_dialog.SwarmZoneDialog
        swarm_zone_dialog.SwarmZoneDialog = lambda *a, **k: opened.append(a)
        asked, undo = _patch_askyesno(False)
        try:
            tab._on_swarm_zone_clicked()
        finally:
            undo()
            swarm_zone_dialog.SwarmZoneDialog = orig_dialog
        assert opened == [], "the picker opened without a border"
        assert len(asked) == 1 and "cage border" in asked[0][0]
        msg = asked[0][1]
        assert "Penzance Ae. aegypti M" in msg and "never been drawn" in msg and "won't guess" in msg
        assert "Draw Cage Borders" in msg
        assert tab.experiment_manager.loaded == [], "answering No must not open anything"
        assert tab.ui_manager.errors == []


def test_run_offers_to_draw_the_border_and_runs_nothing():
    with _Tmp() as t:
        bad = _exp(t, "penzance", "cage_border_points: []\n")
        tab = _tab([])
        dispatched = []
        tab._dispatch = lambda work, describe: dispatched.append(describe)
        tab._save_days = lambda entries: dispatched.append("saved days")
        entry = {"folder": bad, "alias": "penzance"}
        tab._snapshot = lambda: {"entries": [entry], "experiments": [], "view": {}, "night": (20, 8),
                                 "fig_dir": t}
        asked, undo = _patch_askyesno(False)
        try:
            tab._on_run_clicked()
        finally:
            undo()
        assert dispatched == [] and len(asked) == 1


def test_yes_opens_the_experiment_in_setup_with_its_own_settings():
    with _Tmp() as t:
        bad = _exp(t, "penzance", "cage_border_points: []\n")
        with open(os.path.join(bad, "experiment_penz.json"), "w") as f:
            f.write("{}")
        tab = _tab([{"folder": bad, "alias": "penzance"}])
        asked, undo = _patch_askyesno(True)
        try:
            tab._on_swarm_zone_clicked()
        finally:
            undo()
        em = tab.experiment_manager
        assert em.loaded == [(os.path.join(bad, "experiment_penz.json"), None)], \
            "the experiment must be loaded with its own settings, not the previous experiment's"
        assert tab.ui_manager.updated == 1
        assert tab.ui_manager.notebook.selected == "1 · Setup"
        assert tab.ui_manager.setup_notebook.selected == "Background & Cage Border"
        assert any("Draw Cage Borders" in s for s in tab._status)


def test_yes_without_experiment_json_explains():
    with _Tmp() as t:
        bad = _exp(t, "penzance", "cage_border_points: []\n")
        tab = _tab([{"folder": bad, "alias": "penzance"}])
        asked, undo = _patch_askyesno(True)
        try:
            tab._on_swarm_zone_clicked()
        finally:
            undo()
        assert tab.experiment_manager.loaded == []
        assert len(tab.ui_manager.errors) == 1 and "experiment_*.json" in tab.ui_manager.errors[0][1]
