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


def real_data_path(key: str) -> Path | None:
    """Folder for a real-data test, from env var BUZZPHONO_<KEY> or the local paths file; None if absent."""
    value = os.environ.get("BUZZPHONO_" + key.upper()) or _local_paths().get(key)
    if value and Path(value).exists():
        return Path(value)
    return None


@pytest.fixture
def require_real_data():
    """Return a getter that skips the test when a real-data folder is not available here."""

    def _get(key: str) -> Path:
        path = real_data_path(key)
        if path is None:
            pytest.skip(f"real data '{key}' not available on this machine")
        return path

    return _get
