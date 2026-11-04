"""The app's pages driven headlessly with Streamlit's AppTest, on a simulated experiment."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

os.environ.setdefault(
    "BUZZPHONO_HOME", tempfile.mkdtemp(prefix="buzzphono_home_")
)  # never touch ~/.buzzphono

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from buzzphono.synthetic.l2 import L2Design, simulate_experiment

pytestmark = pytest.mark.e2e
APP = Path(__file__).resolve().parents[3] / "src" / "buzzphono" / "app"
DESIGN = L2Design(hours=8)


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    root = tmp_path_factory.mktemp("app")
    simulate_experiment(DESIGN, seed=9, out_dir=root / "exp")
    return root


def _page(name: str, exp_root: Path | None = None, **state) -> AppTest:
    at = AppTest.from_file(str(APP / name), default_timeout=120)
    if exp_root is not None:
        at.session_state["experiments"] = [
            {"path": str(exp_root / "exp"), "label": "sim", "frequency_hz": 475.0}
        ]
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def test_home_renders() -> None:
    at = _page("Home.py")
    assert not at.exception and "BuzzPhono" in at.title[0].value


def test_load_adds_an_experiment(exp) -> None:
    at = AppTest.from_file(str(APP / "pages" / "1_Load.py"), default_timeout=60).run()
    at.text_input(key="load_path").input(str(exp / "exp")).run()
    at.button[1].click().run()  # "Add experiment"
    assert not at.exception
    assert at.session_state["experiments"][0]["path"] == str(exp / "exp")
    at.text_input(key="load_path").input(str(exp / "missing")).run()
    at.button[1].click().run()
    assert any("not a folder" in e.value for e in at.error)


def test_calibrate_shows_the_zones(exp) -> None:
    at = _page("pages/2_Calibrate.py", exp)
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
    saves = [b for b in at.button if b.label == "Save calibration"]
    saves[0].click().run()
    assert not at.exception and any("Saved" in s.value for s in at.success)
    assert (exp / "exp" / "buzzphono_calibration.json").is_file()


def test_validate_checklist(exp) -> None:
    at = _page("pages/3_Validate.py", exp)
    assert not at.exception and any("Ready to analyse" in s.value for s in at.success)


def test_analyse_runs_a_job_and_results_show_it(exp) -> None:
    out = exp / "runs"
    at = _page(
        "pages/4_Analyse.py",
        exp,
        settings={"analyses": ["resting_fraction", "psth"], "clock": {"fps": DESIGN.fps}},
    )
    assert not at.exception
    at.text_input[0].input(str(out)).run()
    run_button = next(b for b in at.button if b.label == "Run")
    run_button.click().run()
    assert not at.exception
    job = at.session_state["job"]
    run_dir = Path(job["run_dir"])
    deadline = time.time() + 300
    status = {}
    from buzzphono import jobs

    while time.time() < deadline:
        status = jobs.read_status(run_dir) or {}
        if status.get("state") in ("done", "failed", "cancelled"):
            break
        time.sleep(1)
    assert status.get("state") == "done", (
        (run_dir / "process.log").read_text(errors="replace")
        if (run_dir / "process.log").exists()
        else status
    )
    res = _page("pages/5_Results.py", runs=[str(run_dir)])
    assert not res.exception and len(res.dataframe) >= 1
    exp_page = _page("pages/6_Export.py", runs=[str(run_dir)])
    assert not exp_page.exception
    next(b for b in exp_page.button if "zip" in b.label).click().run()
    assert not exp_page.exception and exp_page.session_state["zip"][:2] == b"PK"


def test_pages_without_experiments_explain_what_to_do() -> None:
    for name in ("pages/2_Calibrate.py", "pages/3_Validate.py", "pages/4_Analyse.py"):
        at = AppTest.from_file(str(APP / name), default_timeout=60).run()
        assert not at.exception and any("Load an experiment first" in i.value for i in at.info)
