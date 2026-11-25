"""A test that leaves another module as __main__ (Streamlit's AppTest does) must not break the spawn
worker pools of the tests after it: conftest puts __main__ back after every test. These two tests run
in this order; without that fixture the second fails with BrokenProcessPool, as the multi-worker
ingest tests did after the app tests."""

from __future__ import annotations

import sys
import types
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

_ORIGINAL_MAIN = sys.modules["__main__"]


def test_a_test_that_leaves_a_page_as_main(tmp_path: Path) -> None:
    page = tmp_path / "page.py"
    page.write_text("raise StopIteration('a Streamlit page run outside Streamlit')\n", encoding="utf-8")
    fake = types.ModuleType("__main__")
    fake.__file__ = str(page)
    sys.modules["__main__"] = fake  # what Streamlit's script runner leaves behind


def test_spawn_workers_still_start_after_it() -> None:
    assert sys.modules["__main__"] is _ORIGINAL_MAIN
    with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
        assert list(pool.map(abs, [-1, -2, -3])) == [1, 2, 3]
