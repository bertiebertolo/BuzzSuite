"""Opening a second experiment in one app session reads that experiment's own settings (its cage
border above all), not the first one's; a failed open leaves the first experiment as it was."""
import conftest  # noqa: F401  -- repo root on sys.path


class _EM(object):
    def __init__(self, fail=False):
        self.settings = {"cage_border_points": [[0, 0], [1, 0], [1, 1]]}   # experiment A's
        self.seen = []
        self.fail = fail
        self.experiment_alias = "A"

    def load_experiment_from_json(self, path):
        self.seen.append(self.settings)
        if self.fail:
            raise FileNotFoundError("Settings file not found: x.yml")
        self.settings = {"cage_border_points": []}                          # B's own, read from disk
        self.experiment_alias = "B"


class _UI(object):
    def update_ui_after_loading_experiment(self):
        pass

    def set_status(self, **kw):
        pass


class _App(object):
    def __init__(self, em):
        self.experiment_manager, self.ui_manager, self.logs = em, _UI(), []

    def log(self, msg):
        self.logs.append(msg)


def _open(app, path):
    import buzzsuite_app
    from tkinter import messagebox
    shown = []
    orig = messagebox.showerror
    messagebox.showerror = lambda *a, **k: shown.append(a)
    try:
        buzzsuite_app.VideoAnalyzerApp._open_experiment_path(app, path)
    finally:
        messagebox.showerror = orig
    return shown


def test_second_experiment_reads_its_own_settings():
    em = _EM()
    shown = _open(_App(em), "B/experiment_b.json")
    assert shown == []
    assert em.seen == [None], "load_experiment_from_json saw the previous experiment's settings"
    assert em.settings == {"cage_border_points": []} and em.experiment_alias == "B"


def test_failed_open_keeps_the_previous_experiment():
    em = _EM(fail=True)
    before = em.settings
    shown = _open(_App(em), "B/experiment_b.json")
    assert len(shown) == 1 and "Settings file not found" in shown[0][1]
    assert em.settings is before and em.experiment_alias == "A"
