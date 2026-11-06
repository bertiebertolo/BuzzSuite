"""Golden test: the flight analyses reproduce the original flight-phonotaxis package (legacy mode).

Reference tables come from running ``buzzphono_legacy.flight_phonotaxis`` on one thesis cage; they
are local-only fixtures (``fixtures/flight_4-5/``). Skipped when the data or fixtures are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.golden, pytest.mark.realdata, pytest.mark.slow]

REF = Path(__file__).resolve().parent / "fixtures" / "flight_4-5"
_EXPECT = REF / "expect.json"
EXPECT: dict[str, Any] = json.loads(_EXPECT.read_text(encoding="utf-8")) if _EXPECT.is_file() else {}
TABLES = {
    "flight_per_minute": "flight_per_minute.csv",
    "flight_cycle_metrics": "flight_cycle_metrics.csv",
    "links_cycle_metrics": "links_cycle_metrics.csv",
    "distance_cycle_metrics": "distance_cycle_metrics.csv",
    "trajectory_pass_summary": "trajectory_pass_summary.csv",
    "stats_flight_on_vs_off": "stats_flight_on_vs_off.csv",
    "stats_links_on_vs_off": "stats_links_on_vs_off.csv",
    "stats_distance_on_vs_off": "stats_distance_on_vs_off.csv",
}


@pytest.fixture(scope="module")
def flight(real_data, golden_cache):
    from buzzphono.calibration import import_existing
    from buzzphono.flight_phono.pipeline import run_flight
    from buzzphono.io.discover import discover
    from buzzphono.io.ingest import ingest

    root = real_data("thesis_analysis_root")
    if root is None or not EXPECT or not (root / EXPECT["cage"]).is_dir():
        pytest.skip("thesis cage or flight reference fixtures not available on this machine")
    exp = root / EXPECT["cage"]
    layout = discover(exp)
    report = ingest(layout, tiers=("resting", "moving", "tracks"), cache_root=golden_cache, workers=4)
    return run_flight(layout, report.cache_dir, import_existing(exp), legacy=True)


def test_clock_and_window(flight) -> None:
    assert flight.nf == EXPECT["nf"] and flight.chunks_used == EXPECT["chunks_used"]
    assert flight.clock.fitted and flight.clock.n_kept == EXPECT["n_kept"]


@pytest.mark.parametrize("name", list(TABLES))
def test_tables_match_the_original_package(flight, name, tmp_path) -> None:
    ref = pd.read_csv(REF / TABLES[name])
    mine_path = tmp_path / TABLES[name]
    flight.tables[name].to_csv(mine_path, index=False)
    mine = pd.read_csv(mine_path)
    assert list(mine.columns) == list(ref.columns)
    # sums are formed in a different order by today's pandas: allow the last bits of a double
    pd.testing.assert_frame_equal(mine, ref, check_exact=False, rtol=1e-12, atol=1e-13)
    assert np.isfinite(mine.select_dtypes("number").to_numpy(dtype=float)).any()
