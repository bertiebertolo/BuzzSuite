"""Golden test: reproduce the published (2026 thesis) phonotaxis result from the raw tracking data.

Runs only on the analysis PC, which has the thesis cages, the published per-cage CSVs and the
local-only fixtures in tests/buzzphono/golden/fixtures/:

* ``thesis_cages.json``: which experiment folders and analysis windows the thesis used;
* ``published_stats.json``: the published statistics, recomputed with SciPy 1.10.1 from the
  published CSVs;
* ``legacy_grids_<cage>.npz``: the cycle × minute grids of the published code on today's files.

Data locations come from BUZZPHONO_THESIS_ANALYSIS_ROOT / BUZZPHONO_THESIS_RESULTS_ROOT or
golden/local_paths.json. Without them every test here is skipped.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from buzzphono import thesis

pytestmark = [pytest.mark.golden, pytest.mark.realdata, pytest.mark.slow]

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
_CAGES_FILE = FIXTURES / "thesis_cages.json"
CAGES: dict[str, dict[str, str]] = (
    json.loads(_CAGES_FILE.read_text(encoding="utf-8"))["cages"] if _CAGES_FILE.is_file() else {}
)


def _snapshot(folders: list[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns) for f in folders for p in f.rglob("*") if p.is_file()
    }


@pytest.fixture(scope="module")
def reproduced(tmp_path_factory, real_data, golden_cache):
    analysis = real_data("thesis_analysis_root")
    results = real_data("thesis_results_root")
    stats_file = FIXTURES / "published_stats.json"
    if analysis is None or results is None or not stats_file.is_file() or not CAGES:
        pytest.skip("thesis data, published CSVs or golden fixtures not available on this machine")
    out = tmp_path_factory.mktemp("golden_out")
    folders = [analysis / spec["folder"] for spec in CAGES.values()]
    before = _snapshot(folders)
    runs = {}
    for cage, spec in CAGES.items():
        run = thesis.run_thesis(
            analysis / spec["folder"],
            datetime.fromisoformat(spec["start"]),
            datetime.fromisoformat(spec["end"]),
            cache_root=golden_cache,
            workers=4,
        )
        thesis.write_tables(run.tables, out / cage)
        runs[cage] = run
    after = _snapshot(folders)
    return {
        "runs": runs,
        "out": out,
        "results": results,
        "stats": json.loads(stats_file.read_text()),
        "unchanged": before == after,
    }


def test_inputs_are_never_modified(reproduced) -> None:
    assert reproduced["unchanged"]


@pytest.mark.parametrize("cage", list(CAGES))
@pytest.mark.parametrize("name", thesis.LEGACY_CSVS)
def test_published_csv_reproduced_byte_for_byte(reproduced, cage, name) -> None:
    ours = (reproduced["out"] / cage / name).read_bytes()
    published = (reproduced["results"] / f"{cage}_split" / "speaker_distance" / name).read_bytes()
    assert ours == published.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("cage", list(CAGES))
def test_grids_equal_the_published_code_exactly(reproduced, cage) -> None:
    legacy = np.load(FIXTURES / f"legacy_grids_{cage}.npz")
    g = reproduced["runs"][cage].grids
    assert np.array_equal(g.in_zone, legacy["in_cyc"]) and np.array_equal(g.total, legacy["tot_cyc"])
    assert np.array_equal(g.abs_in, legacy["abs_in"]) and np.array_equal(g.abs_total, legacy["abs_tot"])


def test_pooled_paired_test(reproduced) -> None:
    """The headline result: one-sided Wilcoxon (ON > OFF) over every cycle of every cage."""
    paired = [reproduced["runs"][c].tables["speaker_proportion_by_cycle.csv"] for c in CAGES]
    r = thesis.pooled_paired_test(paired)
    expected = reproduced["stats"]["pooled"]["paired"]
    assert r.n + r.n_zero == expected["n"]
    assert r.statistic == expected["W"]
    assert r.p_value == pytest.approx(expected["p_one_sided"], rel=1e-9)


def test_per_phase_table(reproduced) -> None:
    """Δ = mean(ON − OFF) per photoperiod phase, with the published estimator and tests."""
    minutes = [reproduced["runs"][c].tables["resting_speaker_proportion_timeseries.csv"] for c in CAGES]
    expected = reproduced["stats"]["pooled"]["by_phase"]
    got = {r.phase: r for r in thesis.phase_table(minutes)}
    assert set(got) == set(expected)
    for phase, exp in expected.items():
        assert got[phase].n == exp["n"]
        assert got[phase].delta_mean == pytest.approx(exp["delta_mean"], abs=1e-12)
        if "p_one_sided" in exp:
            assert got[phase].test is not None
            assert got[phase].test.p_value == pytest.approx(exp["p_one_sided"], rel=1e-9)
