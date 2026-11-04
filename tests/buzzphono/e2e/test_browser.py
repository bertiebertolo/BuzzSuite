"""Smoke test of the real app in a real browser (Playwright, Edge or Chrome channel).

Starts ``buzzphono app --headless`` on a free local port, then: home page, load an experiment,
click a calibration point on the reference frame, see the validation checklist. With
BUZZPHONO_SCREENSHOTS=<folder> it also saves the screenshots used in the guide.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.e2e]
playwright = pytest.importorskip("playwright.sync_api")

from buzzphono.synthetic.l2 import L2Design, simulate_experiment  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    # screenshots for the guide must not show a personal path: put the data in a neutral folder
    neutral = os.environ.get("BUZZPHONO_SCREENSHOT_DATA")
    root = Path(neutral) if neutral else tmp_path_factory.mktemp("browser")
    simulate_experiment(L2Design(hours=6), seed=12, out_dir=root / "exp")
    port = _free_port()
    env = {**os.environ, "BUZZPHONO_HOME": str(root / "home"), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "buzzphono", "app", "--port", str(port), "--headless"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(90):
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=2)
            break
        except OSError:
            time.sleep(1)
    else:
        proc.kill()
        pytest.fail("the app did not start")
    yield url, root
    proc.kill()


@pytest.fixture(scope="module")
def page(app):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = None
        for channel in ("msedge", "chrome", None):
            try:
                browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no Chromium-based browser available")
        pg = browser.new_page(viewport={"width": 1400, "height": 1900})
        yield pg
        browser.close()


def _shot(page, name: str, height: int = 900) -> None:
    folder = os.environ.get("BUZZPHONO_SCREENSHOTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        page.set_viewport_size({"width": 1400, "height": height})
        page.wait_for_timeout(1500)  # let tables and charts finish drawing
        page.screenshot(path=str(Path(folder) / f"{name}.png"))
        page.set_viewport_size({"width": 1400, "height": 1900})


def test_home_load_calibrate_validate(app, page) -> None:
    url, root = app
    page.goto(url)
    page.get_by_text("Does a sound draw mosquitoes to the speaker?").wait_for(timeout=60_000)
    _shot(page, "app_home", 1000)
    page.goto(url + "/Load")
    page.get_by_label("Experiment folder").fill(str(root / "exp"))
    page.get_by_label("Experiment folder").press("Enter")
    page.get_by_role("button", name="Add experiment").click()
    page.get_by_text("Added exp").wait_for(timeout=60_000)
    _shot(page, "app_load", 640)
    page.get_by_role("link", name="Calibrate").click()
    page.locator(".js-plotly-plot").first.wait_for(timeout=60_000)
    frame = page.frame_locator("iframe[title*='streamlit_image_coordinates']").first
    img = frame.locator("img").first
    img.wait_for(timeout=60_000)
    page.get_by_text("4 corner(s) in cage.").wait_for(timeout=60_000)
    img.click(position={"x": 50, "y": 40})
    page.get_by_text("5 corner(s) in cage.").wait_for(timeout=60_000)  # the click became a corner
    _shot(page, "app_calibrate", 1700)
    page.get_by_role("link", name="Validate").click()
    page.get_by_text("Ready to analyse").wait_for(timeout=60_000)
    _shot(page, "app_validate", 900)
