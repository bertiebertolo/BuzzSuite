"""Shared pytest set-up for the BuzzPhono test suite.

* Plots never open a window: matplotlib runs headless (Agg).
* Hypothesis runs a derandomised, deadline-free profile so CI is deterministic; set
  HYPOTHESIS_PROFILE=dev locally for a randomised run.
* Tests that need real data (marked ``golden`` / ``realdata``) find it through environment
  variables or, on the analysis PC only, tests/buzzphono/golden/local_paths.json (never published).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci", derandomize=True, deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("dev", deadline=None, max_examples=200)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

TESTS_DIR = Path(__file__).resolve().parent


def _local_paths() -> dict[str, str]:
    path = TESTS_DIR / "golden" / "local_paths.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(v, str)}
    return {}


def local_setting(key: str) -> str | None:
    """A machine-specific test setting: env var BUZZPHONO_<KEY>, else the local paths file."""
    return os.environ.get("BUZZPHONO_" + key.upper()) or _local_paths().get(key)


def real_data_path(key: str) -> Path | None:
    """Folder for a real-data test, or None when it is not available on this machine."""
    value = local_setting(key)
    if value and Path(value).exists():
        return Path(value)
    return None


@pytest.fixture(scope="session")
def real_data():
    """``real_data(key)`` -> Path or None (session-scoped, for module-scoped real-data fixtures)."""
    return real_data_path


@pytest.fixture(scope="session")
def golden_cache(tmp_path_factory) -> Path:
    """Where real-data tests keep their BuzzPhono cache: a fixed local folder if configured
    (so repeated runs reuse it), else a temporary one."""
    value = local_setting("golden_cache")
    return Path(value) if value else tmp_path_factory.mktemp("golden_cache")


@pytest.fixture
def require_real_data():
    """Return a getter that skips the test when a real-data folder is not available here."""

    def _get(key: str) -> Path:
        path = real_data_path(key)
        if path is None:
            pytest.skip(f"real data '{key}' not available on this machine")
        return path

    return _get
