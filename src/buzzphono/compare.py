"""Working from finished runs: compare several runs' experiments, or redraw a run's figures.

Both read only what a run wrote (its tables and ``run_manifest.json``), so they work on another
computer without the tracking data.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.errors import DataError
from buzzphono.manifest import MANIFEST
from buzzphono.metrics.comparison import confounds, experiment_table, group_contrast
from buzzphono.metrics.cycles import paired_comparison


def _manifest(run: Path) -> dict[str, Any]:
    try:
        return json.loads((run / MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataError(f"{run} is not a BuzzPhono result folder (no {MANIFEST}).") from exc
    except ValueError as exc:
        raise DataError(f"{run / MANIFEST} is not valid JSON ({exc}).") from exc


def _experiments(run: Path) -> list[tuple[dict[str, Any], Path]]:
    m = _manifest(run)
    return [(e, run / "experiments" / e["label"]) for e in m.get("inputs", [])]


def compare_runs(runs: list[Path], out: Path) -> list[Path]:
    """Experiment table (effect, CI, metadata), confounds, group contrasts, pooled tests, tuning figure."""
    resolved = [r.resolve() for r in runs]
    if len(set(resolved)) != len(resolved):
        raise DataError("The same run is given more than once: its experiments would be counted twice.")
    from buzzphono.jobs import EXPERIMENT_MARKERS

    if any((out / m).exists() for m in EXPERIMENT_MARKERS):
        raise DataError(f"{out} is an experiment folder; choose another folder for the comparison.")
    rows = []
    cycles = []
    settings: dict[str, Any] = {}
    for run in runs:
        settings = settings or _manifest(run).get("settings", {})
        for info, folder in _experiments(run):
            meta = info.get("metadata", {})
            label = f"{run.name}/{info['label']}" if len(runs) > 1 else info["label"]
            st_path = folder / "resting_fraction" / "statistics.csv"
            r: dict[str, Any] = {}
            if st_path.is_file():
                st = pd.read_csv(st_path)
                sel = st[(st["comparison"] == "ON vs OFF (speaker zone)") & (st["phase"] == "all")]
                r = sel.iloc[0].to_dict() if len(sel) else {}
            rows.append(
                {
                    "label": label,
                    "frequency_hz": meta.get("frequency_hz"),
                    "stimulus_type": meta.get("stimulus_type"),
                    "species": meta.get("species"),
                    "sex": meta.get("sex"),
                    "group": meta.get("group"),
                    "n_cycles": r.get("n", np.nan),
                    "mean_delta": r.get("mean_delta", np.nan),
                    "ci_low": r.get("wild_cluster_ci_low", np.nan),
                    "ci_high": r.get("wild_cluster_ci_high", np.nan),
                    "p_value": r.get("wild_cluster_p_value", np.nan),
                }
            )
            cyc_path = folder / "cycles" / "cycle_table.csv"
            if cyc_path.is_file():
                c = pd.read_csv(cyc_path)
                c = c[c["usable"].astype(bool)].copy()
                c["cage"] = label
                c["cage_day"] = label + " " + c["date"].astype(str)
                cycles.append(c)
    if not rows:
        raise DataError("The given runs contain no experiments.")
    out.mkdir(parents=True, exist_ok=True)
    table = experiment_table(rows)
    written = []
    tables: dict[str, pd.DataFrame] = {
        "experiments": table,
        "confounds": pd.DataFrame({"note": confounds(table)}),
    }
    contrasts = [
        group_contrast(table, d)
        for d in ("group", "frequency_hz", "species", "sex")
        if len(set(table[d].dropna())) > 1
    ]
    if contrasts:
        tables["group_contrasts"] = pd.concat(contrasts, ignore_index=True)
    if cycles:
        pool = pd.concat(cycles, ignore_index=True)
        stats = settings.get("stats", {})
        tables["pooled_statistics"] = paired_comparison(
            pool,
            "off_colony_frac_in_zone",
            "on_colony_frac_in_zone",
            label="ON vs OFF (all experiments)",
            alternative=stats.get("alternative", "two-sided"),
            block_hours=stats.get("block_hours", 12),
            n_resamples=stats.get("n_resamples", 9999),
            seed=stats.get("seed", 20261104),
        )
    for name, t in tables.items():
        p = out / f"{name}.csv"
        t.to_csv(p, index=False, lineterminator="\n")
        written.append(p)
    if table["frequency_hz"].notna().sum() >= 2:
        from buzzphono.plotting.static import tuning_plot
        from buzzphono.plotting.style import save

        written += save(
            tuning_plot(table),
            out / "figures" / "tuning_curve",
            settings.get("output", {}).get("formats", ["png", "svg", "pdf"]),
        )
    return written


def redraw(run: Path) -> list[Path]:
    """Redraw every figure that can be made from the run's tables (maps need the tracking data)."""
    from buzzphono.plotting.figures import experiment_figures

    settings = _manifest(run).get("settings", {})
    output = settings.get("output", {})
    light = settings.get("light", {})
    stim = settings.get("stimulus", {})
    config = SimpleNamespace(
        output=SimpleNamespace(
            formats=output.get("formats", ["png", "svg", "pdf"]), dpi=output.get("dpi", 300)
        ),
        light=SimpleNamespace(
            lights_on=light.get("lights_on", "05:00"), lights_off=light.get("lights_off", "17:00")
        ),
    )
    on = stim.get("on", {"start_min": 40, "end_min": 50})
    windows = {"on": (on["start_min"], on["end_min"])}
    paths: list[Path] = []
    for info, folder in _experiments(run):
        tables = {}
        for p in folder.rglob("*.csv"):
            try:
                tables[p.relative_to(folder).with_suffix("").as_posix()] = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue  # an analysis that produced no rows
        cyc = tables.get("cycles/cycle_table", pd.DataFrame())
        good = cyc[cyc["usable"].astype(bool)] if len(cyc) else cyc
        res = SimpleNamespace(label=info["label"], tables=tables, out_dir=folder, extras={}, warnings=[])
        ctx = SimpleNamespace(good=good, windows=windows, cal=None)
        paths += experiment_figures(res, ctx, config)  # type: ignore[arg-type]
    return paths
