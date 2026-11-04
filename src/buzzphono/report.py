"""The run report: one self-contained HTML file (figures embedded, every value escaped).

Sections: the run at a glance, data quality (clock, usable hour-cycles, every warning), each
experiment's results (the key statistics and its figures, each linking to the full-resolution
file), pooled results, the statistical assumptions, and the software and settings used.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from buzzphono import __version__
from buzzphono.io.cache import atomic_write_text
from buzzphono.stats import ASSUMPTIONS

if TYPE_CHECKING:
    from buzzphono.pipeline import ExperimentResult, RunResult

KEY_COLUMNS = [
    "comparison",
    "phase",
    "n",
    "mean_off",
    "mean_on",
    "mean_delta",
    "wild_cluster_ci_low",
    "wild_cluster_ci_high",
    "wild_cluster_p_value",
    "block_signflip_p_value",
    "wilcoxon_p_value",
    "hodges_lehmann",
    "dz",
]
HEADERS = {
    "n": "cycles",
    "mean_off": "mean OFF",
    "mean_on": "mean ON",
    "mean_delta": "mean Δ",
    "wild_cluster_ci_low": "95% CI low",
    "wild_cluster_ci_high": "95% CI high",
    "wild_cluster_p_value": "p (wild cluster)",
    "block_signflip_p_value": "p (block sign-flip)",
    "wilcoxon_p_value": "p (Wilcoxon)",
    "hodges_lehmann": "Hodges-Lehmann",
    "dz": "dz",
}

CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e1e0d9; --surface:#fcfcfb; --accent:#eb6834; }
body { font: 14px/1.5 Arial, Calibri, "DejaVu Sans", sans-serif; color: var(--ink);
       background: var(--surface);
       max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 18px; border-bottom: 1px solid var(--rule); padding-bottom: 4px; margin-top: 32px; }
h3 { font-size: 15px; margin-top: 20px; } .muted { color: var(--muted); }
table { border-collapse: collapse; font-size: 12px; margin: 8px 0; display: block; overflow-x: auto; }
th, td { padding: 3px 8px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: var(--ink2); font-weight: 600; }
.warn { border-left: 3px solid var(--accent); padding: 4px 10px; margin: 4px 0; background: #fff6f1; }
.figs { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
.figs figure { margin: 0; } .figs img { width: 100%; border: 1px solid var(--rule); }
.figs figcaption { font-size: 12px; color: var(--ink2); }
code { font-size: 12px; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _table(df: pd.DataFrame, columns: list[str] | None = None, headers: dict[str, str] | None = None) -> str:
    if df is None or df.empty:
        return '<p class="muted">No rows.</p>'
    d = df[[c for c in (columns or list(df.columns)) if c in df.columns]].copy()
    d = d.rename(columns=headers or {})
    return d.to_html(index=False, escape=True, float_format=lambda v: f"{v:.4g}", na_rep="–", border=0)


def _figure(path: Path, run_dir: Path) -> str:
    rel = path.relative_to(run_dir).as_posix()
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    full = path.with_suffix(".pdf") if path.with_suffix(".pdf").exists() else path
    full_rel = full.relative_to(run_dir).as_posix()
    return (
        f'<figure><a href="{_e(full_rel)}"><img alt="{_e(path.stem)}" src="data:image/png;base64,{data}"></a>'
        f"<figcaption>{_e(path.stem.replace('_', ' '))} — <code>{_e(rel)}</code></figcaption></figure>"
    )


def _figures(paths: list[Path], run_dir: Path) -> str:
    pngs = [p for p in paths if p.suffix == ".png" and p.exists()]
    if not pngs:
        return ""
    return '<div class="figs">' + "".join(_figure(p, run_dir) for p in pngs) + "</div>"


def _experiment(e: ExperimentResult, run_dir: Path) -> str:
    parts = [
        f"<h2>{_e(e.label)}</h2>",
        f'<p class="muted">{_e(e.spec.path)}</p>',
        f"<p>Clock: {_e(e.clock_message)}</p>",
    ]
    n_usable = int(e.cycles["usable"].sum()) if len(e.cycles) else 0
    parts.append(
        f"<p>Hour-cycles: {len(e.cycles)} with data, {n_usable} usable "
        "(coverage rule passed, tone not known to have failed).</p>"
    )
    for w in e.warnings:
        parts.append(f'<div class="warn">{_e(w)}</div>')
    for name, title in (
        ("resting_fraction/statistics", "Speaker-zone resting fraction (headline)"),
        ("attraction_index/statistics", "Attraction index"),
        ("approach/statistics", "Directed flight (approach)"),
        ("reliability/summary", "Reliability"),
    ):
        t = e.tables.get(name)
        if t is not None and len(t):
            parts.append(f"<h3>{_e(title)}</h3>")
            cols = (
                KEY_COLUMNS + (["validation"] if "validation" in t else [])
                if name != "reliability/summary"
                else None
            )
            parts.append(_table(t, cols, HEADERS))
    sens = e.tables.get("sensitivity/variants")
    if sens is not None and len(sens):
        flagged = sens[sens["flag"].fillna("").astype(str).str.len() > 0]
        parts.append("<h3>Sensitivity</h3>")
        parts.append(
            f"<p>{len(sens)} variants; {len(flagged)} flagged.</p>"
            + _table(flagged if len(flagged) else sens.head(0))
        )
    parts.append(_figures(e.figures, run_dir))
    return "\n".join(parts)


def render(result: RunResult) -> str:
    run_dir = result.run_dir
    cfg = result.config
    exps = result.experiments
    head = (
        f'<h1>BuzzPhono run</h1><p class="muted">{_e(run_dir.name)} · '
        f"{_e(result.started)} – {_e(result.finished)} · "
        f"BuzzPhono {_e(__version__)}</p>"
        f"<p>{len(exps)} experiment(s); analyses: {_e(', '.join(cfg.analyses))}; "
        f"clock: {_e(cfg.clock.mode)}; "
        f"ON {cfg.stimulus.on.start_min}-{cfg.stimulus.on.end_min} "
        f"vs OFF {cfg.stimulus.off.start_min}-{cfg.stimulus.off.end_min} "
        f"min of each hour; tests {_e(cfg.stats.alternative)}.</p>"
    )
    quality = ["<h2>Data quality</h2>"]
    rows = [
        {
            "experiment": e.label,
            "clock": e.clock_message,
            "cycles": len(e.cycles),
            "usable": int(e.cycles["usable"].sum()) if len(e.cycles) else 0,
            **{f"ingest {k}": v for k, v in e.ingest.items()},
        }
        for e in exps
    ]
    quality.append(_table(pd.DataFrame(rows)))
    if result.warnings:
        quality.append(
            f"<p>{len(result.warnings)} warning(s):</p>"
            + "".join(f'<div class="warn">{_e(w)}</div>' for w in result.warnings[:200])
        )
    else:
        quality.append("<p>No warnings.</p>")
    body = [head, *quality, *(_experiment(e, run_dir) for e in exps)]
    if result.pooled:
        body.append("<h2>All experiments</h2>")
        for name, t in result.pooled.items():
            body.append(f"<h3>{_e(name.replace('_', ' '))}</h3>")
            body.append(_table(t, KEY_COLUMNS if name == "statistics" else None, HEADERS))
        body.append(_figures(result.figures, run_dir))
    body.append("<h2>Statistical assumptions</h2>")
    body.append("".join(f"<p><b>{_e(k)}</b>: {_e(v)}</p>" for k, v in ASSUMPTIONS.items()))
    body.append("<h2>Settings</h2>")
    body.append(f"<pre>{_e(json.dumps(json.loads(cfg.model_dump_json()), indent=1))}</pre>")
    body.append(
        '<p class="muted">Full provenance (input files, software versions, output hashes): '
        "run_manifest.json.</p>"
    )
    return (
        '<!doctype html><html lang="en-GB"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>BuzzPhono report</title><style>{CSS}</style></head><body>"
        + "\n".join(body)
        + "</body></html>"
    )


def write_report(result: RunResult) -> Path:
    path = result.run_dir / "report.html"
    atomic_write_text(path, render(result))
    if result.config.output.report_pdf:
        _pdf(result)
    return path


def _pdf(result: RunResult) -> Path:
    """All figures of the run, one per page (report.pdf)."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = result.run_dir / "report.pdf"
    pngs = [p for e in result.experiments for p in e.figures if p.suffix == ".png"] + [
        p for p in result.figures if p.suffix == ".png"
    ]
    with PdfPages(path, metadata={"CreationDate": None, "ModDate": None}) as pdf:
        for p in pngs:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes((0.05, 0.05, 0.9, 0.9))
            ax.imshow(mpimg.imread(p))
            ax.set_axis_off()
            ax.set_title(p.stem.replace("_", " "))
            pdf.savefig(fig)
            plt.close(fig)
    return path
