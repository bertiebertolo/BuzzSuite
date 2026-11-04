"""Command-line interface: ``buzzphono <command>``.

The commands are thin wrappers: they parse options, call into the buzzphono package, print
progress and map errors to exit codes (see :mod:`buzzphono.errors`). No analysis lives here.

Exit codes: 0 done, 1 internal error, 2 usage, 3 settings or validation, 4 data, 5 calibration,
130 cancelled.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

import typer

from buzzphono import __version__
from buzzphono.errors import BuzzPhonoError, Cancelled, ConfigError, DataError, ValidationFailed

app = typer.Typer(
    name="buzzphono",
    help="BuzzPhono: phonotaxis analysis of BuzzWatch mosquito tracking data.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
cache_app = typer.Typer(
    help="Inspect, check or delete an experiment's BuzzPhono cache.", no_args_is_help=True
)
app.add_typer(cache_app, name="cache")

ANALYSIS_NAMES = (
    "resting_fraction, attraction_index, psth, approach, zt_profile, spatial, comparison, reliability, "
    "sensitivity, flight (or 'all')"
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"buzzphono {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """BuzzPhono: phonotaxis analysis of BuzzWatch mosquito tracking data."""


@app.command()
def version() -> None:
    """Print the BuzzPhono version."""
    typer.echo(f"buzzphono {__version__}")


# ----------------------------------------------------------------------------- analyse
def _analyses(text: str | None) -> list[str] | None:
    if not text:
        return None
    names = [a.strip().replace("-", "_") for a in text.split(",") if a.strip()]
    if names == ["all"]:
        return [
            "resting_fraction",
            "attraction_index",
            "psth",
            "zt_profile",
            "spatial",
            "approach",
            "reliability",
            "sensitivity",
            "flight",
            "comparison",
        ]
    return names


def build_run_config(
    paths: list[Path],
    config_file: Path | None,
    analyses: str | None,
    clock: str | None,
    out: Path | None,
    cache: Path | None,
    workers: int | None,
    label: str | None,
    colony: int | None,
) -> Any:
    """Settings from a file (if given) with the command-line options on top, checked."""
    import yaml

    from buzzphono.config import build_config

    data: dict[str, Any] = {}
    where = "the command-line options"
    if config_file is not None:
        try:
            loaded = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise ConfigError(f"Settings file not found: {config_file}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"{config_file} is not valid YAML: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"{config_file} must contain settings (key: value pairs).")
        data = loaded
        where = str(config_file)
        for e in data.get("experiments", []) or []:  # relative experiment paths are relative to the file
            if isinstance(e, dict) and "path" in e and not Path(str(e["path"])).is_absolute():
                e["path"] = str((config_file.parent / str(e["path"])).resolve())
    if paths:
        data["experiments"] = [{"path": str(p)} for p in paths]
    if label is not None:
        if len(data.get("experiments", [])) != 1:
            raise ConfigError("--label needs exactly one experiment.")
        data["experiments"][0]["label"] = label
    if not data.get("experiments"):
        raise ConfigError("No experiment given.", hint="Give an experiment folder, or --config settings.yml.")
    names = _analyses(analyses)
    if names is not None:
        data["analyses"] = names
    if clock is not None:
        data.setdefault("clock", {})["mode"] = clock
    if out is not None:
        data.setdefault("output", {})["dir"] = str(out)
    if cache is not None:
        data.setdefault("output", {})["cache_dir"] = str(cache)
    if workers is not None:
        data["workers"] = workers
    if colony is not None:
        data.setdefault("metric", {})["colony_size"] = colony
    return build_config(data, where)


class _Printer:
    """Progress on stderr: one line per new message (no flicker in logs or the app)."""

    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet
        self.last = ""

    def __call__(self, fraction: float, message: str) -> None:
        if self.quiet or message == self.last:
            return
        self.last = message
        typer.echo(f"[{fraction:6.1%}] {message}", err=True)


@app.command()
def analyse(
    paths: list[Path] = typer.Argument(None, help="Experiment folder(s). Omit when --config lists them."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Settings file (YAML); see the guide."),
    batch: Path | None = typer.Option(None, "--batch", help="A settings file listing several experiments."),
    analyses: str | None = typer.Option(None, "--analyses", "-a", help=f"Comma-separated: {ANALYSIS_NAMES}."),
    clock: str | None = typer.Option(None, "--clock", help="corrected (default), nominal or thesis."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Folder for results (default: <experiment>/plots/buzzphono)."
    ),
    cache: Path | None = typer.Option(None, "--cache", help="Folder for the tracking cache."),
    workers: int | None = typer.Option(
        None, "--workers", "-w", min=1, max=61, help="Processes reading files."
    ),
    label: str | None = typer.Option(None, "--label", help="Name for the experiment in outputs."),
    colony: int | None = typer.Option(None, "--colony", min=1, help="Mosquitoes in the cage (default 30)."),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Exact folder for this run (used by the app)."
    ),
    max_bad_chunks: int = typer.Option(
        0, "--max-bad-chunks", min=0, help="Unreadable chunks allowed before exit code 4."
    ),
    no_figures: bool = typer.Option(False, "--no-figures", help="Tables only."),
    open_report: bool = typer.Option(False, "--open", help="Open the report when done."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="No progress messages."),
) -> None:
    """Run the analyses and write tables, figures, the report and run_manifest.json."""
    from buzzphono import jobs, pipeline

    cfg = build_run_config(paths or [], config or batch, analyses, clock, out, cache, workers, label, colony)
    target = run_dir or pipeline.default_run_dir(cfg)
    job = jobs.Job(target)
    printer = _Printer(quiet)

    def progress(fraction: float, message: str) -> None:
        job.progress(fraction, message)
        printer(fraction, message)

    try:
        result = pipeline.run(
            cfg, run_dir=target, progress=progress, cancel=job.cancelled, make_figures=not no_figures
        )
    except KeyboardInterrupt as exc:
        jobs.remove_partial_run(target)
        job.finish("cancelled", Cancelled("Cancelled with Ctrl-C."), 130)
        raise Cancelled("Cancelled with Ctrl-C; partial results were deleted.") from exc
    except Cancelled as exc:
        jobs.remove_partial_run(target)
        job.finish("cancelled", exc, 130)
        raise
    except BuzzPhonoError as exc:
        job.finish("failed", exc, exc.exit_code)
        raise
    except Exception as exc:
        job.finish("failed", exc, 1)
        raise
    bad = sum(e.ingest.get("errors", 0) for e in result.experiments)
    for w in result.warnings[:20]:
        typer.echo(f"Warning: {w}", err=True)
    if len(result.warnings) > 20:
        typer.echo(f"... and {len(result.warnings) - 20} more warnings (see the report).", err=True)
    typer.echo(f"Results: {result.run_dir}")
    if result.report is not None:
        typer.echo(f"Report:  {result.report}")
        if open_report:
            webbrowser.open(result.report.resolve().as_uri())
    if not any(int(e.cycles["usable"].sum()) for e in result.experiments if len(e.cycles)):
        job.finish("failed", None, 4)
        raise DataError(
            "No usable hour-cycle in any experiment, so there are no results to report.",
            hint="Check the analysis window (start/end), the frame rate and the warnings above.",
        )
    if bad > max_bad_chunks:
        job.finish("failed", None, 4)
        raise DataError(
            f"{bad} tracking file(s) could not be read (allowed: {max_bad_chunks}). The results above exclude them.",
            hint="See the warnings in the report; use --max-bad-chunks to accept them.",
        )
    job.finish("done")


# ----------------------------------------------------------------------------- validate
@app.command()
def validate(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    pi_config: Path | None = typer.Option(None, "--pi-config", help="The Pi's config.ini to check."),
    playback_log: list[Path] = typer.Option(None, "--playback-log", help="sound.log or the Pi log zip."),
    on: str = typer.Option("40-50", "--on", help="Tone window, minutes of each hour."),
    frequency: float | None = typer.Option(
        None, "--frequency", help="Tone frequency of this experiment, Hz."
    ),
) -> None:
    """Check an experiment before analysing it: files, calibration, Pi settings, playback log."""
    from buzzphono.validate import validate_experiment

    checks = validate_experiment(
        experiment, pi_config=pi_config, playback_logs=playback_log or [], on=on, frequency_hz=frequency
    )
    marks = {"ok": "OK  ", "info": "info", "warning": "WARN", "error": "FAIL"}
    for c in checks:
        typer.echo(f"{marks.get(c.severity, c.severity):4}  {c.area}: {c.message}")
    errors = [c for c in checks if c.severity == "error"]
    if errors:
        raise ValidationFailed(f"{len(errors)} problem(s) must be fixed before analysing {experiment}.")
    typer.echo("Ready to analyse.")


# ----------------------------------------------------------------------------- calibrate
def _points(text: str) -> list[tuple[float, float]]:
    try:
        pts = [tuple(float(v) for v in p.split(",")) for p in text.replace(";", " ").split()]
    except ValueError as exc:
        raise ConfigError(f"'{text}' is not a list of x,y points (e.g. '10,20 30,20 30,40').") from exc
    if any(len(p) != 2 for p in pts):
        raise ConfigError(f"'{text}': every point needs exactly two numbers, x,y.")
    return [(p[0], p[1]) for p in pts]


@app.command()
def calibrate(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    speaker: str | None = typer.Option(
        None, "--speaker", help="Speaker zone vertices: 'x,y x,y x,y ...' (pixels)."
    ),
    control: list[str] = typer.Option(
        None, "--control", help="A control zone: 'name=x,y x,y ...'. Repeatable."
    ),
    cage: str | None = typer.Option(None, "--cage", help="Cage corners TL TR BR BL: 'x,y x,y x,y x,y'."),
    width_mm: float | None = typer.Option(
        None, "--width-mm", help="Inner width of the filmed cage face, mm."
    ),
    height_mm: float | None = typer.Option(
        None, "--height-mm", help="Inner height of the filmed cage face, mm."
    ),
    click: bool = typer.Option(False, "--click", help="Click the speaker zone on the reference frame."),
    out: Path | None = typer.Option(
        None, "--out", help="Where to write (default: <experiment>/buzzphono_calibration.json)."
    ),
) -> None:
    """Write the calibration (cage, speaker and control zones, optional cage size in mm)."""
    from buzzphono.calibration import (
        CALIBRATION_FILE,
        Calibration,
        PhysicalSize,
        Zone,
        import_existing,
        save_calibration,
    )

    base: dict[str, Any]
    try:
        base = import_existing(experiment).model_dump()
    except BuzzPhonoError:
        if speaker is None and not click:
            raise
        base = {}
    if click:
        speaker = " ".join(f"{x:.1f},{y:.1f}" for x, y in _click_polygon(experiment))
    if speaker is not None:
        others = [z for z in base.get("zones", []) if z["role"] != "speaker"]
        base["zones"] = [{"name": "speaker", "role": "speaker", "polygon": _points(speaker)}, *others]
    if control:
        zones = [z for z in base.get("zones", []) if z["role"] == "speaker"]
        for c in control:
            name, _, pts = c.partition("=")
            if not pts:
                raise ConfigError(f"--control '{c}' must look like 'name=x,y x,y x,y'.")
            zones.append({"name": name.strip(), "role": "control", "polygon": _points(pts)})
        base["zones"] = zones
    if cage is not None:
        base["cage"] = _points(cage)
    if width_mm is not None or height_mm is not None:
        if width_mm is None or height_mm is None:
            raise ConfigError("Give both --width-mm and --height-mm.")
        if width_mm <= 0 or height_mm <= 0:
            raise ConfigError("--width-mm and --height-mm must be positive (the cage's inner size in mm).")
        base["physical"] = PhysicalSize(cage_width_mm=width_mm, cage_height_mm=height_mm).model_dump()
    if "frame_width_px" not in base:
        from buzzphono.calibration import image_size, reference_images

        refs = reference_images(experiment)
        if not refs:
            raise ConfigError(f"{experiment} has no reference frame to take the frame size from.")
        base["frame_width_px"], base["frame_height_px"] = image_size(refs[0])
    base.pop("created", None)
    base.pop("created_by", None)
    from pydantic import ValidationError

    from buzzphono.config import plain_english

    try:
        cal = Calibration(**{**base, "zones": [Zone(**z) for z in base.get("zones", [])]})
    except ValidationError as exc:
        raise ConfigError(plain_english(exc, "the calibration")) from exc
    path = save_calibration(cal, out or experiment / CALIBRATION_FILE)
    typer.echo(f"Calibration written: {path}")
    typer.echo(
        f"  frame {cal.frame_width_px}×{cal.frame_height_px} px; zones: "
        + ", ".join(f"{z.name} ({z.role})" for z in cal.zones)
    )


def _click_polygon(experiment: Path) -> list[tuple[float, float]]:  # pragma: no cover - interactive
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from buzzphono.calibration import reference_images

    refs = reference_images(experiment)
    if not refs:
        raise ConfigError(f"{experiment} has no reference frame to click on.")
    with Image.open(refs[len(refs) // 2]) as im:
        img = im.convert("RGB")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.set_title("Click the speaker zone's corners, then press Enter (right-click removes the last point)")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    if len(pts) < 3:
        raise ConfigError("A zone needs at least 3 points.")
    return [(float(x), float(y)) for x, y in pts]


# ----------------------------------------------------------------------------- demo
@app.command()
def demo(
    folder: Path = typer.Argument(
        Path("buzzphono_demo"), help="Where to create the demo experiments and results."
    ),
    hours: int = typer.Option(
        72, "--hours", min=6, max=144, help="Hours to simulate per experiment (72 = 4 calendar days)."
    ),
    seed: int = typer.Option(1, "--seed", help="Random seed."),
    open_report: bool = typer.Option(False, "--open", help="Open the report when done."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Simulate a tone experiment and a no-tone control with known truth, analyse both, and report."""
    from buzzphono import pipeline
    from buzzphono.config import build_config
    from buzzphono.synthetic.l2 import L2Design, simulate_experiment

    folder.mkdir(parents=True, exist_ok=True)
    tone = L2Design(hours=hours)
    null = L2Design(hours=hours, recruitment=1.0, arousal=1.0, directed=1.0, startle=0.0, label="NoToneCage")
    for name, design, s in (("tone", tone, seed), ("no_tone", null, seed + 1)):
        if not (folder / name / "synthetic_truth.json").is_file():
            typer.echo(f"Simulating {name} ({hours} h)...", err=True)
            simulate_experiment(design, seed=s, out_dir=folder / name)
    cfg = build_config(
        {
            "experiments": [
                {
                    "path": str(folder / "tone"),
                    "label": "tone",
                    "group": "tone",
                    "frequency_hz": tone.frequency_hz,
                },
                {"path": str(folder / "no_tone"), "label": "no_tone", "group": "no tone"},
            ],
            "analyses": [
                "resting_fraction",
                "attraction_index",
                "psth",
                "zt_profile",
                "spatial",
                "approach",
                "reliability",
                "sensitivity",
                "comparison",
            ],
            "clock": {"fps": tone.fps},
            "output": {"dir": str(folder / "results"), "cache_dir": str(folder / "cache")},
        },
        "the demo settings",
    )
    result = pipeline.run(cfg, progress=_Printer(quiet))
    typer.echo(f"Results: {result.run_dir}")
    typer.echo(f"Report:  {result.report}")
    if open_report and result.report is not None:
        webbrowser.open(result.report.resolve().as_uri())


# ----------------------------------------------------------------------------- compare / plot / report
@app.command()
def compare(
    runs: list[Path] = typer.Argument(..., help="Result folders of earlier runs."),
    out: Path = typer.Option(..., "--out", "-o", help="Folder for the comparison."),
) -> None:
    """Pool earlier runs: per-experiment effects side by side, tuning curve, group contrasts."""
    from buzzphono.compare import compare_runs

    paths = compare_runs(runs, out)
    for p in paths:
        typer.echo(str(p))


@app.command()
def plot(run: Path = typer.Argument(..., help="A run's result folder.")) -> None:
    """Redraw a run's figures from its tables (e.g. after changing the style)."""
    from buzzphono.compare import redraw

    paths = redraw(run)
    typer.echo(f"{len(paths)} figure file(s) written under {run}.")


@app.command()
def report(
    run: Path = typer.Argument(..., help="A run's result folder."),
    open_report: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Show where a run's report is (and open it)."""
    path = run / "report.html"
    if not path.is_file():
        raise DataError(f"No report in {run}.", hint="Run `buzzphono analyse` first.")
    typer.echo(str(path))
    if open_report:
        webbrowser.open(path.resolve().as_uri())


@app.command()
def status(
    run: Path = typer.Argument(..., help="A run's result folder."),
    cancel: bool = typer.Option(False, "--cancel", help="Ask the run to stop."),
) -> None:
    """Show a run's progress, or ask it to stop."""
    from buzzphono import jobs

    if cancel:
        jobs.request_cancel(run)
        typer.echo("Cancel requested.")
        return
    st = jobs.read_status(run)
    if st is None:
        raise DataError(f"No run status in {run}.")
    if st.get("state") == "running" and not jobs.pid_alive(int(st.get("pid", -1))):
        st["state"] = "stopped (its process has ended without finishing)"
    typer.echo(json.dumps(st, indent=1))


@app.command(name="app")
def app_cmd(
    port: int = typer.Option(8501, "--port", help="Local port for the app."),
    headless: bool = typer.Option(False, "--headless", help="Do not open a browser (for servers and tests)."),
) -> None:  # pragma: no cover
    """Open the BuzzPhono app in the browser (stays on this computer)."""
    import importlib.util

    home = Path(__file__).resolve().parent / "app" / "Home.py"
    if not home.is_file():
        raise DataError("The app is not installed with this copy of BuzzPhono.")
    if importlib.util.find_spec("streamlit") is None:
        raise DataError("The app needs Streamlit, which is not installed.", hint="pip install -e .[app]")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(home),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true" if headless else "false",
    ]
    raise SystemExit(subprocess.call(cmd))


# ----------------------------------------------------------------------------- cache
@cache_app.command("info")
def cache_info(experiment: Path, cache: Path | None = typer.Option(None, "--cache")) -> None:
    """What the cache holds for this experiment."""
    from buzzphono.io.cache import CacheIndex, cache_dir_for
    from buzzphono.io.discover import discover

    discover(experiment)  # a clear error if this is not an experiment folder
    d = cache_dir_for(experiment, cache)
    index = CacheIndex.open(d, experiment)
    counts: dict[str, int] = {}
    for tiers in index.data["chunks"].values():
        for tier, entry in tiers.items():
            if entry.get("status") == "ok":
                counts[tier] = counts.get(tier, 0) + 1
    size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0
    typer.echo(f"Cache: {d}")
    typer.echo(
        f"  {size / 1e6:.1f} MB; chunks per tier: "
        + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none")
    )


@cache_app.command("verify")
def cache_verify(experiment: Path, cache: Path | None = typer.Option(None, "--cache")) -> None:
    """Check every cached chunk still matches its tracking file and loads."""
    from buzzphono.io.cache import CacheIndex, cache_dir_for, load_npz, source_identity
    from buzzphono.io.discover import discover

    d = cache_dir_for(experiment, cache)
    index = CacheIndex.open(d, experiment)
    stale = broken = ok = 0
    for chunk in discover(experiment).chunks:
        for tier, src in chunk.files.items():
            if index.entry(chunk.chunk_id, tier) is None:
                continue
            if not index.is_current(chunk.chunk_id, tier, source_identity(src)):
                stale += 1
                continue
            try:
                load_npz(index.tier_path(tier, chunk.chunk_id))
                ok += 1
            except Exception:
                broken += 1
    typer.echo(f"{ok} current, {stale} out of date (rebuilt on the next run), {broken} unreadable.")
    if broken:
        raise DataError(
            f"{broken} cache file(s) are unreadable.", hint="Run `buzzphono cache clean` and analyse again."
        )


@cache_app.command("clean")
def cache_clean(
    experiment: Path,
    cache: Path | None = typer.Option(None, "--cache"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete this experiment's cache (the tracking files are never touched)."""
    from buzzphono.io.cache import cache_dir_for
    from buzzphono.jobs import lock_holder

    d = cache_dir_for(experiment, cache)
    if not d.is_dir():
        typer.echo("Nothing to delete.")
        return
    if lock_holder(d) is not None:
        raise DataError("A run is using this cache right now.", hint="Wait for it to finish.")
    if not yes and not typer.confirm(f"Delete {d}?"):
        raise typer.Exit(1)
    shutil.rmtree(d)
    typer.echo(f"Deleted {d}")


# ----------------------------------------------------------------------------- entry
def _safe_streams() -> None:
    """Never crash on a character the console cannot show (e.g. a Japanese-locale Windows console)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def run(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code (used by :func:`main` and the tests).

    Typer handles help and usage errors itself (exit code 2). BuzzPhono's own errors are printed
    in plain English and mapped to their exit codes; Ctrl-C becomes exit code 130.
    """
    _safe_streams()
    try:
        app(args=argv, prog_name="buzzphono")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    except BuzzPhonoError as exc:
        typer.echo(f"Error: {exc}", err=True)
        return exc.exit_code
    except KeyboardInterrupt:
        typer.echo("Cancelled.", err=True)
        return 130
    return 0


def main() -> None:
    """Console-script entry point (``buzzphono``)."""
    raise SystemExit(run())
