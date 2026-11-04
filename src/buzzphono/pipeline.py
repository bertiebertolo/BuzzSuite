"""One run of BuzzPhono: every selected analysis on every experiment, plus the pooled results.

``run(config)`` is the single entry point of the command line and the app. For each experiment
it reads the tracking files into the cache, checks the calibration, builds the clock, counts
detections into 30-s bins and makes the hour-cycle table; each selected analysis then writes its
tidy CSV tables (and figures) under ``<run>/experiments/<label>/<analysis>/``. Experiments are
pooled into ``<run>/pooled/``; the report and ``run_manifest.json`` sit at the top of the run.
Inputs are only read: nothing is ever written next to the tracking data except the cache.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.calibration import CALIBRATION_FILE, Calibration, import_existing, load_calibration
from buzzphono.config import AnalysisConfig, ExperimentSpec, LightSettings
from buzzphono.errors import Cancelled, ConfigError, DataError
from buzzphono.frames import Binned, ExperimentClock, bin_experiment, experiment_clock
from buzzphono.io.cache import cache_dir_for
from buzzphono.io.discover import ExperimentLayout, discover
from buzzphono.io.ingest import IngestReport, ingest
from buzzphono.io.playback_log import PlaybackLog, read_playback_log
from buzzphono.jobs import claim_run_dir, experiment_lock
from buzzphono.metrics import attraction_index as ai
from buzzphono.metrics.cycles import (
    Coverage,
    add_control_contrast,
    assumptions_table,
    cycle_table,
    paired_comparison,
    usable,
)
from buzzphono.timebase import to_ns

Progress = Callable[[float, str], None]
CancelFn = Callable[[], bool]


@dataclass
class ExperimentResult:
    label: str
    spec: ExperimentSpec
    out_dir: Path
    clock_message: str
    cycles: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # "analysis/name" -> table
    figures: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ingest: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)  # in-memory objects for figures (not written)


@dataclass
class RunResult:
    run_dir: Path
    config: AnalysisConfig
    experiments: list[ExperimentResult]
    pooled: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started: str = ""
    finished: str = ""
    report: Path | None = None
    manifest: Path | None = None


def config_hash(config: AnalysisConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()[:8]


def default_run_dir(config: AnalysisConfig, now: datetime | None = None) -> Path:
    moment = now or datetime.now()
    stamp = moment.strftime("%Y%m%d-%H%M%S-") + f"{moment.microsecond // 1000:03d}"
    run_id = f"{stamp}-{config_hash(config)}"
    if config.output.dir is not None:
        return Path(config.output.dir) / run_id
    return config.experiments[0].path / "plots" / "buzzphono" / run_id


def resolve_calibration(experiment: Path, calibration: Path | None = None) -> Calibration:
    """The experiment's calibration: the given file, else its sidecar, else imported from BuzzWatch."""
    if calibration is not None:
        return load_calibration(calibration)
    sidecar = experiment / CALIBRATION_FILE
    if sidecar.is_file():
        return load_calibration(sidecar)
    return import_existing(experiment)


def windows_of(config: AnalysisConfig) -> dict[str, tuple[int, int]]:
    s = config.stimulus
    w = {
        "on": (s.on.start_min, s.on.end_min),
        "off": (s.off.start_min, s.off.end_min),
        "sham_on": (s.sham_on.start_min, s.sham_on.end_min),
        "sham_off": (s.sham_off.start_min, s.sham_off.end_min),
    }
    if s.pre is not None:
        w["pre"] = (s.pre.start_min, s.pre.end_min)
    return w


def _tiers(config: AnalysisConfig) -> tuple[str, ...]:
    tiers = ["resting"]
    if config.clock.mode != "thesis" or set(config.analyses) & {"psth", "approach", "spatial", "flight"}:
        tiers += ["moving", "tracks"]
    return tuple(tiers)


def _write(tables: dict[str, pd.DataFrame], root: Path) -> None:
    for name, table in tables.items():
        path = root / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, lineterminator="\n")


class _Steps:
    """Maps each experiment's progress into its share of the whole run and checks for cancel."""

    def __init__(self, progress: Progress | None, cancel: CancelFn | None, n: int) -> None:
        self.progress, self.cancel, self.n = progress, cancel, max(n, 1)
        self.i = 0

    def check(self) -> None:
        if self.cancel is not None and self.cancel():
            raise Cancelled("The run was cancelled.")

    def sub(self, lo: float, hi: float) -> Progress:
        def report(frac: float, msg: str) -> None:
            self.check()
            if self.progress is not None:
                self.progress((self.i + lo + (hi - lo) * frac) / self.n, msg)

        return report


def run(
    config: AnalysisConfig,
    *,
    run_dir: Path | None = None,
    progress: Progress | None = None,
    cancel: CancelFn | None = None,
    make_figures: bool = True,
    make_report: bool = True,
) -> RunResult:
    run_dir = run_dir or default_run_dir(config)
    _refuse_inside_experiment(run_dir)
    _refuse_same_data_twice(config)  # before anything is created
    claim_run_dir(run_dir)  # never write into an experiment folder, a foreign or a finished one
    result = RunResult(run_dir, config, [], started=datetime.now().isoformat(timespec="seconds"))
    steps = _Steps(progress, cancel, len(config.experiments) + 1)
    labels: set[str] = set()
    for i, spec in enumerate(config.experiments):
        steps.i = i
        label = spec.name
        # Windows folder names ignore case and trailing dots: compare labels the same way
        if _folder_key(label) in labels:
            label = f"{label}_{i + 1}"
        labels.add(_folder_key(label))
        with experiment_lock(cache_dir_for(spec.path, config.output.cache_dir)):
            exp = _run_experiment(config, spec, label, run_dir / "experiments" / label, steps, make_figures)
        result.experiments.append(exp)
        result.warnings += [f"{label}: {w}" for w in exp.warnings]
    steps.i = len(config.experiments)
    _pooled(config, result, steps.sub(0, 1), make_figures)
    result.finished = datetime.now().isoformat(timespec="seconds")
    from buzzphono.manifest import write_manifest

    result.manifest = write_manifest(result)
    if make_report:
        from buzzphono.report import write_report

        result.report = write_report(result)
    if progress is not None:
        progress(1.0, "Done")
    return result


# ----------------------------------------------------------------------------- one experiment
def _run_experiment(
    config: AnalysisConfig, spec: ExperimentSpec, label: str, out: Path, steps: _Steps, figs: bool
) -> ExperimentResult:
    steps.check()
    layout = discover(spec.path)
    if not layout.chunks:
        raise DataError(
            f"{spec.path} has no tracking files (tracking_resting/, tracking_moving/, final_tracking_data/).",
            hint="Track the experiment with BuzzWatch first.",
        )
    cal = resolve_calibration(spec.path, spec.calibration)
    warnings: list[str] = []
    report = ingest(
        layout,
        tiers=_tiers(config),  # type: ignore[arg-type]
        cache_root=config.output.cache_dir,
        workers=config.workers,
        progress=steps.sub(0.0, 0.5),
        cancel=steps.cancel,
    )
    warnings += _ingest_warnings(report)
    windows = windows_of(config)
    fps = config.fps_for(spec)
    eclock = experiment_clock(
        layout,
        report.cache_dir,
        mode=config.clock.mode,
        cage=cal.cage_polygon(),
        fps=fps,
        on_start_min=config.stimulus.on.start_min,
        rig_drift_s_per_day=config.clock.rig_drift_s_per_day,
        manual_offset_s=config.clock.manual_offset_s,
    )
    if eclock.fit is not None and not eclock.fit.model.fitted:
        warnings.append(eclock.message)
    warnings += _fps_check(eclock, fps, config.clock.fps is None)
    playback = _playback(config)
    light = config.light
    if config.stimulus.pi_config is not None:
        light, pi_warnings = _pi_config_checks(config, cal, layout)
        warnings += pi_warnings
    speaker_name = config.metric.speaker_zone
    if speaker_name not in {z.name for z in cal.zones}:
        speaker_name = cal.speaker.name
    controls = [z.name for z in cal.controls if z.name != speaker_name]
    zones: dict[str, Any] = {"speaker": cal.polygon(speaker_name), **{c: cal.polygon(c) for c in controls}}
    if "spatial" in config.analyses:
        from buzzphono.metrics.sensitivity import grown_zone

        zones[BAND_ZONE] = grown_zone(cal.polygon(speaker_name), BAND_PX)  # spatial F2 contrast (§5)
    binned = bin_experiment(
        layout,
        report.cache_dir,
        eclock,
        zones,
        colony=config.metric.colony_size,
        cage=cal.cage_polygon(),
        t_start_ns=to_ns(spec.start) if spec.start else None,
        t_end_ns=to_ns(spec.end) if spec.end else None,
        progress=steps.sub(0.5, 0.65),
        cancel=steps.cancel,
    )
    for chunk, why in binned.chunks_skipped:
        warnings.append(f"chunk {chunk} skipped: {why}")
    cov = config.metric.coverage
    coverage = Coverage(
        cov.min_frame_fraction,
        cov.max_gap_s,
        cov.min_minutes_with_resting,
        enabled=config.clock.mode != "thesis",
    )
    phases = light.phases()
    cycles = cycle_table(
        binned, windows, fps=fps, cage=label, phases=phases, coverage=coverage, playback=playback
    )
    cycles = add_control_contrast(cycles, binned, tuple(windows), controls)
    if len(cycles):
        cycles["tone_scheduled"] = _scheduled(config, cycles)
        from buzzphono.metrics.cycles import MAX_FRAME_FRACTION

        doubled = int(
            (cycles[[f"{w}_frame_fraction" for w in windows]] > MAX_FRAME_FRACTION).any(axis=1).sum()
        )
        if doubled:
            warnings.append(
                f"{doubled} hour-cycle(s) hold more frames than their windows last: recordings overlap in "
                "time (e.g. an '-attempt' session repeating another). Those cycles are excluded; remove the "
                "duplicate recording to use them."
            )
    cycles["usable"] = (
        (usable(cycles, ("on", "off")) & cycles["tone_scheduled"]) if len(cycles) else pd.Series(dtype=bool)
    )
    res = ExperimentResult(
        label,
        spec,
        out,
        eclock.message,
        cycles,
        warnings=warnings,
        ingest={
            "built": report.built,
            "reused": report.reused,
            "errors": len(report.errors),
            "seconds": report.seconds,
        },
    )
    res.tables["cycles/cycle_table"] = cycles
    if eclock.fit is not None and len(eclock.fit.bursts):
        from buzzphono.stimulus import describe_bursts

        res.tables["clock/tone_onset_bursts"] = describe_bursts(eclock.fit)
    good = cycles[cycles["usable"]] if len(cycles) else cycles
    if len(good) < 3:
        warnings.append(f"only {len(good)} usable hour-cycles: statistics need at least 3")
    warnings += _colony_check(binned, config.metric.colony_size)
    if spec.fps_from_settings() is None and _yml_states_fps(spec.path) and config.clock.fps is None:
        warnings.append(
            "buzzwatch_track_settings.yml states an fps that is not a valid number; 25 fps was used. "
            "Set clock.fps to the camera's frame rate."
        )
    ctx = _Ctx(
        config,
        layout,
        report,
        cal,
        eclock,
        binned,
        cycles,
        good,
        windows,
        controls,
        res,
        steps.sub(0.65, 1.0),
        steps.cancel,
        fps,
    )
    analyses = list(dict.fromkeys(config.analyses)) if len(good) else []
    if not len(good):
        warnings.append("no usable hour-cycle: the analyses were not run (see the warnings above)")
    for k, name in enumerate(analyses):
        steps.check()
        ctx.say(k / max(len(analyses), 1), f"{label}: {name.replace('_', ' ')}")
        if name in ANALYSES:
            ANALYSES[name](ctx)
    family = secondary_family(res.tables)
    if len(family):
        res.tables["multiplicity/secondary_family_F2"] = family
    _write(res.tables, out)
    if figs and len(good):
        from buzzphono.plotting.figures import experiment_figures

        res.figures = experiment_figures(res, ctx, config)
    return res


def _scheduled(config: AnalysisConfig, cycles: pd.DataFrame) -> pd.Series:
    """Whether the tone was scheduled to play in each cycle's ON window (stimulus.active_from/until)."""
    from buzzphono.config import _hhmm_to_min

    lo = _hhmm_to_min(config.stimulus.active_from)
    hi = _hhmm_to_min(config.stimulus.active_until)
    on_min = cycles["hour"].astype(int) * 60 + config.stimulus.on.start_min
    inside = (on_min >= lo) & (on_min < hi) if lo <= hi else (on_min >= lo) | (on_min < hi)
    return inside


def _refuse_inside_experiment(run_dir: Path) -> None:
    """Results may sit in <experiment>/plots/buzzphono/, never directly in an experiment folder or in
    one of its tracking folders."""
    from buzzphono.jobs import EXPERIMENT_MARKERS

    parent = run_dir.parent
    in_tracking = any(p.name in EXPERIMENT_MARKERS for p in [run_dir, *run_dir.parents])
    if in_tracking or any((parent / m).is_dir() for m in EXPERIMENT_MARKERS):
        raise ConfigError(
            f"The results folder {run_dir} would be inside an experiment's own folders.",
            hint="Leave out --out (results go to <experiment>/plots/buzzphono/) or choose another folder.",
        )


def _folder_key(label: str) -> str:
    return label.rstrip(". ").casefold()


def _refuse_same_data_twice(config: AnalysisConfig) -> None:
    """Two experiments with the same recording chunks are the same data (a copy, or another path
    spelling such as a network share): counting both would double n."""
    seen: dict[frozenset[str], str] = {}
    for spec in config.experiments:
        chunks = frozenset(c.chunk_id for c in discover(spec.path).chunks)
        if not chunks:
            continue
        if chunks in seen:
            raise ConfigError(
                f"{spec.path} holds the same recordings as {seen[chunks]}: "
                "the same data would be counted twice.",
                hint="List each experiment once (a copy or another path to one folder is the same data).",
            )
        seen[chunks] = str(spec.path)


def _colony_check(binned: Binned, colony: int) -> list[str]:
    """Warn when resting detections per frame are far above the colony size (wrong colony_size)."""
    frames = binned.rest_frames > 0
    if not frames.any():
        return []
    per_frame = float(np.median(binned.raw_total[frames] / binned.rest_frames[frames]))
    if per_frame > 2 * colony:
        return [
            f"Resting detections per frame (median {per_frame:.0f}) are far above the colony size "
            f"({colony}): check metric.colony_size (the number of mosquitoes in the cage)."
        ]
    return []


def _yml_states_fps(path: Path) -> bool:
    try:
        return "fps" in (path / "buzzwatch_track_settings.yml").read_text(encoding="utf-8")
    except OSError:
        return False


def _spec_of(config: AnalysisConfig, layout: ExperimentLayout) -> ExperimentSpec:
    return next(e for e in config.experiments if e.path.resolve() == layout.root.resolve())


def _fps_check(eclock: ExperimentClock, fps: float, automatic: bool) -> list[str]:
    """Warn when the frames per 20-min chunk say the camera ran at a clearly different rate."""
    full = max((n for n, src in eclock.nf.values() if src == "tracker frame list"), default=0)
    if not full:
        return []
    implied = full / 1200.0
    if abs(implied / fps - 1) <= 0.05:
        return []
    where = "read from the experiment's settings (or the default)" if automatic else "set in clock.fps"
    return [
        f"The chunks hold {full} frames, about {implied:.3g} fps for 20-minute chunks, but the analysis "
        f"uses {fps:g} fps ({where}). If the chunks are 20 minutes long, set clock.fps: {implied:.3g}."
    ]


def _ingest_warnings(report: IngestReport) -> list[str]:
    out = [f"{chunk} ({tier}) could not be read: {msg}" for chunk, tier, msg in report.errors]
    for tier, missing in report.missing.items():
        if missing and tier == "resting":
            out.append(f"{len(missing)} chunk(s) have no resting tracking file")
    return out


def _playback(config: AnalysisConfig) -> PlaybackLog | None:
    """The playback log, when the stimulus source allows it (``schedule`` ignores logs)."""
    s = config.stimulus
    if s.source == "playback_log" and not s.playback_log:
        raise ConfigError("stimulus.source is 'playback_log' but no stimulus.playback_log file is given.")
    if s.source == "pi_config" and s.pi_config is None:
        raise ConfigError("stimulus.source is 'pi_config' but no stimulus.pi_config file is given.")
    if not s.playback_log or s.source in ("schedule", "pi_config"):
        return None
    return read_playback_log(s.playback_log)


def _pi_config_checks(
    config: AnalysisConfig, cal: Calibration, layout: ExperimentLayout
) -> tuple[LightSettings, list[str]]:
    """Cross-check the Pi's config.ini with the recording; with ``light.source: auto`` its light
    ramps also give the light phases."""
    from buzzphono.io.pi_config import cross_check, read_pi_config

    assert config.stimulus.pi_config is not None
    pi = read_pi_config(config.stimulus.pi_config)
    s = config.stimulus
    problems = pi.checks + cross_check(
        pi,
        frame_size=cal.frame_size,
        labels={c.name.label for c in layout.chunks},
        on_window=(s.on.start_min, s.on.end_min),
        fps=config.fps_for(_spec_of(config, layout)),
    )
    warnings = [f"Pi config {c.field}: {c.message}" for c in problems if c.severity != "info"]
    light = config.light
    if light.source == "auto" and pi.ramp_up_start_min is not None and pi.ramp_down_end_min is not None:
        on_min = pi.ramp_up_start_min
        off_min = pi.ramp_down_end_min
        light = LightSettings(
            lights_on=f"{on_min // 60:02d}:{on_min % 60:02d}",
            lights_off=f"{off_min // 60:02d}:{off_min % 60:02d}",
        )
    return light, warnings


@dataclass
class _Ctx:
    config: AnalysisConfig
    layout: ExperimentLayout
    report: IngestReport
    cal: Calibration
    eclock: ExperimentClock
    binned: Binned
    cycles: pd.DataFrame
    good: pd.DataFrame
    windows: dict[str, tuple[int, int]]
    controls: list[str]
    res: ExperimentResult
    say: Progress
    cancel: CancelFn | None
    fps: float = 25.0

    def compare(
        self, a: str, b: str, label: str, data: pd.DataFrame | None = None, **kw: Any
    ) -> pd.DataFrame:
        st = self.config.stats
        return paired_comparison(
            self.good if data is None else data,
            a,
            b,
            label=label,
            alternative=kw.pop("alternative", st.alternative),
            block_hours=st.block_hours,
            n_resamples=st.n_resamples,
            seed=st.seed,
            **kw,
        )


def _resting_fraction(ctx: _Ctx) -> None:
    good = ctx.good.assign(
        _zero=0.0,
        tone_minus_sham=(ctx.good["on_colony_frac_in_zone"] - ctx.good["off_colony_frac_in_zone"])
        - (ctx.good["sham_on_colony_frac_in_zone"] - ctx.good["sham_off_colony_frac_in_zone"]),
    )
    rows = [
        ctx.compare("off_colony_frac_in_zone", "on_colony_frac_in_zone", "ON vs OFF (speaker zone)"),
        ctx.compare(
            "_zero",
            "tone_minus_sham",
            "tone minus sham: (ON − OFF) − (ON' − OFF')",
            data=good,
            by_phase=False,
        ),
        ctx.compare(
            "sham_off_colony_frac_in_zone", "sham_on_colony_frac_in_zone", "sham: ON' vs OFF' (no tone)"
        ),
        ctx.compare(
            "off_resting_frac_in_zone",
            "on_resting_frac_in_zone",
            "ON vs OFF, uncapped fraction",
            by_phase=False,
            with_mixed_model=False,
        ),
    ]
    if ctx.controls and "delta_speaker" in ctx.good:
        rows.append(
            ctx.compare(
                "off_control_area_matched",
                "on_control_area_matched",
                "control zones (area-matched): ON vs OFF",
            )
        )
        rows.append(ctx.compare("delta_control", "delta_speaker", "speaker minus control zones (ON − OFF)"))
    ctx.res.tables["resting_fraction/statistics"] = add_conclusions(pd.concat(rows, ignore_index=True))
    ctx.res.tables["resting_fraction/assumptions"] = assumptions_table()
    if ctx.config.clock.mode == "thesis":
        from buzzphono import thesis

        legacy = thesis.legacy_tables(_thesis_grids(ctx))
        for name, table in legacy.items():
            ctx.res.tables[f"resting_fraction/thesis_{name.removesuffix('.csv')}"] = table


def _thesis_grids(ctx: _Ctx) -> Any:
    """The published cycle × minute grids, recomputed with the thesis clock and file selection."""
    from buzzphono import thesis

    spec = ctx.res.spec
    start = spec.start or pd.Timestamp(ctx.binned.t0_ns).to_pydatetime()
    end = spec.end or pd.Timestamp(ctx.binned.t0_ns + ctx.binned.n_bins * ctx.binned.bin_ns).to_pydatetime()
    run = thesis.run_thesis(spec.path, start, end, cache_root=ctx.config.output.cache_dir, layout=ctx.layout)
    return run.grids


def _attraction(ctx: _Ctx) -> None:
    pre = "pre" if "pre" in ctx.windows else "off"
    cyc = ai.add_attraction_index(ctx.good, pre_col=f"{pre}_colony_frac_in_zone")
    st = ctx.config.stats
    ctx.res.tables["attraction_index/per_cycle"] = cyc[
        [
            "cage",
            "cycle_start",
            "phase",
            "on_colony_frac_in_zone",
            f"{pre}_colony_frac_in_zone",
            ai.INDEX_COLUMN,
        ]
    ]
    ctx.res.tables["attraction_index/statistics"] = ai.attraction_summary(
        cyc, alternative=st.alternative, block_hours=st.block_hours, n_resamples=st.n_resamples, seed=st.seed
    )


def _psth(ctx: _Ctx) -> None:
    from buzzphono.metrics.psth import psth

    on0 = ctx.windows["on"][0]
    ctx.res.tables["psth/psth"] = psth(
        ctx.binned,
        ctx.good,
        fps=ctx.fps,
        onset_min=on0,
        sham_onset_min=ctx.windows["sham_on"][0],
        n_resamples=min(ctx.config.stats.n_resamples, 999),
        seed=ctx.config.stats.seed,
    )
    # F2 contrast: the change from the 5 min before to the 5 min after the tone onset, minus the same
    # change at the sham onset (which also starts a fresh video chunk, so tracker warm-up cancels)
    from buzzphono.metrics.psth import bin_quantities, event_matrix

    values = bin_quantities(ctx.binned, ctx.fps)["colony_frac_in_zone"]
    starts = ctx.good["cycle_start_ns"].to_numpy(dtype=np.int64)

    def onset_change(minute: int) -> np.ndarray:
        mat, rel = event_matrix(values, ctx.binned, starts + minute * 60 * 10**9, 5, 5)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            post = np.nanmean(np.where(rel[None, :] >= 0, mat, np.nan), axis=1)
            pre = np.nanmean(np.where(rel[None, :] < 0, mat, np.nan), axis=1)
        return post - pre

    cyc = ctx.good.assign(
        _zero=0.0,
        onset_change=onset_change(on0),
        sham_onset_change=onset_change(ctx.windows["sham_on"][0]),
    )
    ctx.res.tables["psth/onset_response"] = ctx.compare(
        "sham_onset_change",
        "onset_change",
        "PSTH onset response (5 min after minus 5 min before), tone minus sham onset",
        data=cyc,
        by_phase=False,
    )


def _zt(ctx: _Ctx) -> None:
    from buzzphono.metrics.zt_profile import hour_of_day_profile, minute_series

    on_h = int(ctx.config.light.lights_on.split(":")[0])
    ctx.res.tables["zt_profile/hour_of_day"] = hour_of_day_profile(
        ctx.good,
        lights_on_hour=on_h,
        n_resamples=min(ctx.config.stats.n_resamples, 999),
        seed=ctx.config.stats.seed,
    )
    ctx.res.tables["zt_profile/minute_series"] = minute_series(ctx.binned, fps=ctx.fps, windows=ctx.windows)


def _spatial(ctx: _Ctx) -> None:
    from buzzphono.metrics.spatial import occupancy_maps, radial_profile, sample_tracks

    starts = ctx.good["cycle_start_ns"].to_numpy()
    w = {"on": ctx.windows["on"], "off": ctx.windows["off"]}
    maps = occupancy_maps(
        ctx.layout, ctx.report.cache_dir, ctx.eclock, starts, w, ctx.cal.frame_size, cancel=ctx.cancel
    )
    ctx.res.tables["spatial/occupancy_maps"] = maps.tidy()
    ctx.res.tables["spatial/radial_profile"] = radial_profile(
        maps, ctx.cal.polygon(ctx.cal.speaker.name), ctx.cal.cage_polygon()
    )
    ctx.res.tables["spatial/sample_tracks"] = sample_tracks(
        ctx.layout, ctx.report.cache_dir, ctx.eclock, starts, w, seed=ctx.config.stats.seed
    )
    ctx.res.extras["maps"] = maps
    band_on, band_off = f"on_colony_frac_in_zone__{BAND_ZONE}", f"off_colony_frac_in_zone__{BAND_ZONE}"
    if band_on in ctx.good:
        ctx.res.tables["spatial/band_statistics"] = ctx.compare(
            band_off,
            band_on,
            f"fraction within {BAND_PX:g} px of the speaker zone: ON vs OFF",
            by_phase=False,
        )


def _approach(ctx: _Ctx) -> None:
    from buzzphono.metrics.approach import (
        ApproachSettings,
        approach_analysis,
        approach_cycle_table,
        wide_by_window,
    )

    a = ctx.config.approach
    settings = ApproachSettings(
        a.min_points, a.smoothing_frames, a.min_drop_mm, a.min_drop_cage_fraction, a.min_duration_s
    )
    w = {k: ctx.windows[k] for k in ("on", "off", "sham_on", "sham_off")}
    controls = {c: ctx.cal.polygon(c) for c in ctx.controls}
    to_mm = ctx.cal.to_mm if ctx.cal.physical is not None else None
    res = approach_analysis(
        ctx.layout,
        ctx.report.cache_dir,
        ctx.eclock,
        ctx.good["cycle_start_ns"].to_numpy(),
        w,
        speaker=ctx.cal.polygon(ctx.cal.speaker.name),
        controls=controls,
        cage=ctx.cal.cage_polygon(),
        to_mm=to_mm,
        settings=settings,
        fps=ctx.fps,
        cancel=ctx.cancel,
    )
    table = approach_cycle_table(res, ctx.good, ctx.controls)
    ctx.res.tables["approach/bouts"] = res.bouts
    ctx.res.tables["approach/per_cycle_window"] = table
    if table.empty:
        return
    rows = []
    for value in (f"directed_velocity_{res.unit}_per_s", "bouts_per_flight_s"):
        wide = wide_by_window(table, value)
        if {f"on_{value}", f"off_{value}"} <= set(wide.columns):
            rows.append(
                ctx.compare(f"off_{value}", f"on_{value}", f"{value}: ON vs OFF", data=wide, by_phase=False)
            )
        if {f"sham_on_{value}", f"sham_off_{value}"} <= set(wide.columns):
            rows.append(
                ctx.compare(
                    f"sham_off_{value}", f"sham_on_{value}", f"{value}: sham", data=wide, by_phase=False
                )
            )
    if rows:
        st = pd.concat(rows, ignore_index=True)
        from buzzphono.metrics.approach import VALIDATION

        st["validation"] = [
            VALIDATION["radial_velocity"] if "velocity" in c else VALIDATION["approach_bouts"]
            for c in st["comparison"]
        ]
        ctx.res.tables["approach/statistics"] = st


def _reliability(ctx: _Ctx) -> None:
    from buzzphono.metrics.reliability import reliability

    if "delta_speaker" not in ctx.good:
        df = ctx.good.assign(
            delta_speaker=ctx.good["on_colony_frac_in_zone"] - ctx.good["off_colony_frac_in_zone"]
        )
    else:
        df = ctx.good
    summary, plan = reliability(
        df,
        "delta_speaker",
        alpha=ctx.config.stats.alpha,
        n_resamples=min(ctx.config.stats.n_resamples, 999),
        seed=ctx.config.stats.seed,
    )
    ctx.res.tables["reliability/summary"] = summary
    ctx.res.tables["reliability/planning"] = plan


def _sensitivity(ctx: _Ctx) -> None:
    from buzzphono.metrics.sensitivity import SensitivityPlan, run_sensitivity

    s = ctx.config.sensitivity
    plan = SensitivityPlan(
        margins_px=[m for m in s.zone_margins_px if m != 0],
        shifts_px=[m for m in s.zone_shifts_px if m != 0],
        colonies=[c for c in s.colony_sizes if c != ctx.config.metric.colony_size],
        walking_mm_s=list(s.walking_thresholds_mm_s),
    )
    clocks: dict[str, ExperimentClock] = {ctx.config.clock.mode: ctx.eclock}
    for mode in ("corrected", "nominal", "thesis"):
        if mode not in clocks:
            clocks[mode] = experiment_clock(
                ctx.layout,
                ctx.report.cache_dir,
                mode=mode,
                cage=ctx.cal.cage_polygon(),
                fps=ctx.fps,
            )
    px_per_mm = None
    if ctx.cal.physical is not None:
        cage = ctx.cal.cage_polygon()
        px_per_mm = float(np.hypot(*(cage[1] - cage[0])) / ctx.cal.physical.cage_width_mm)
    w, h = ctx.cal.frame_size
    on, off = ctx.windows["on"], ctx.windows["off"]
    cov = ctx.config.metric.coverage
    ctx.res.tables["sensitivity/variants"] = run_sensitivity(
        ctx.layout,
        ctx.report.cache_dir,
        clocks,
        speaker=ctx.cal.polygon(ctx.cal.speaker.name),
        cage=ctx.cal.cage_polygon(),
        frame_size=(w, h),
        fps=ctx.fps,
        colony=ctx.config.metric.colony_size,
        on=on,
        off=off,
        phases=ctx.config.light.phases(),
        coverage=Coverage(cov.min_frame_fraction, cov.max_gap_s, cov.min_minutes_with_resting),
        px_per_mm=px_per_mm,
        plan=plan,
        cage_label=ctx.res.label,
        n_resamples=min(ctx.config.stats.n_resamples, 999),
        seed=ctx.config.stats.seed,
        cancel=ctx.cancel,
    )


def _flight(ctx: _Ctx) -> None:
    from buzzphono.flight_phono.build import FlightParams
    from buzzphono.flight_phono.pipeline import run_flight

    f = ctx.config.flight
    on, off = ctx.windows["on"], ctx.windows["off"]
    lh = (int(ctx.config.light.lights_on.split(":")[0]), int(ctx.config.light.lights_off.split(":")[0]))
    params = FlightParams(
        on=on,
        off=off,
        light_hours=lh,
        flyer_mult=f.flyer_multiplier,
        radius=f.radius_cage_widths,
        min_median=f.min_median,
    )
    if abs(ctx.fps - 25.0) > 0.25:
        ctx.res.warnings.append(
            f"flight analysis skipped: it is defined for 25-fps video (its speed and time thresholds are in "
            f"frames), and this experiment is analysed at {ctx.fps:g} fps"
        )
        return
    walls = [z.name for z in ctx.cal.zones if z.role != "speaker"]
    if len(walls) != 3:
        ctx.res.warnings.append(
            "flight analysis skipped: it needs the speaker zone plus exactly three other zones, one per "
            f"wall (found {len(walls)}: {', '.join(walls) or 'none'})"
        )
        return
    res = run_flight(
        ctx.layout, ctx.report.cache_dir, ctx.cal, params=params, label=ctx.res.label, cancel=ctx.cancel
    )
    for name, table in res.tables.items():
        ctx.res.tables[f"flight/{name}"] = table


ANALYSES: dict[str, Callable[[_Ctx], None]] = {
    "resting_fraction": _resting_fraction,
    "attraction_index": _attraction,
    "psth": _psth,
    "zt_profile": _zt,
    "spatial": _spatial,
    "approach": _approach,
    "reliability": _reliability,
    "sensitivity": _sensitivity,
    "flight": _flight,
}


# ----------------------------------------------------------------------------- pooled
def _pooled(config: AnalysisConfig, result: RunResult, say: Progress, figs: bool) -> None:
    exps = [e for e in result.experiments if len(e.cycles)]
    if not exps:
        return
    say(0.0, "Pooling experiments")
    cycles = pd.concat([e.cycles[e.cycles["usable"]] for e in exps], ignore_index=True)
    st = config.stats
    pooled: dict[str, pd.DataFrame] = {}
    if len(exps) > 1 and len(cycles) >= 3:
        rows = [
            paired_comparison(
                cycles,
                "off_colony_frac_in_zone",
                "on_colony_frac_in_zone",
                label="ON vs OFF (all experiments)",
                alternative=st.alternative,
                block_hours=st.block_hours,
                n_resamples=st.n_resamples,
                seed=st.seed,
            ),
            paired_comparison(
                cycles,
                "sham_off_colony_frac_in_zone",
                "sham_on_colony_frac_in_zone",
                label="sham (all experiments)",
                alternative=st.alternative,
                block_hours=st.block_hours,
                n_resamples=st.n_resamples,
                seed=st.seed,
                by_phase=False,
            ),
            paired_comparison(
                cycles.assign(
                    _zero=0.0,
                    tone_minus_sham=(cycles["on_colony_frac_in_zone"] - cycles["off_colony_frac_in_zone"])
                    - (cycles["sham_on_colony_frac_in_zone"] - cycles["sham_off_colony_frac_in_zone"]),
                ),
                "_zero",
                "tone_minus_sham",
                label="tone minus sham: (ON − OFF) − (ON' − OFF') (all experiments)",
                alternative=st.alternative,
                block_hours=st.block_hours,
                n_resamples=st.n_resamples,
                seed=st.seed,
                by_phase=False,
            ),
        ]
        pooled["statistics"] = add_conclusions(pd.concat(rows, ignore_index=True))
        if "reliability" in config.analyses:
            from buzzphono.metrics.reliability import reliability

            df = cycles.assign(delta=cycles["on_colony_frac_in_zone"] - cycles["off_colony_frac_in_zone"])
            groups = {e.spec.group for e in exps}
            if len(groups) > 1:
                result.warnings.append(
                    "Pooled reliability mixes experiments from different groups "
                    f"({', '.join(sorted(str(g) for g in groups))}): its cage ICC and MDE mostly reflect the "
                    "group difference. Use the per-experiment reliability tables instead."
                )
            pooled["reliability_summary"], pooled["reliability_planning"] = reliability(
                df, "delta", n_resamples=min(st.n_resamples, 999), seed=st.seed
            )
    if "comparison" in config.analyses or len(exps) > 1:
        from buzzphono.metrics.comparison import confounds, experiment_table, group_contrast

        rows_e = []
        for e in exps:
            stats = e.tables.get("resting_fraction/statistics")
            r = (
                stats[(stats["comparison"] == "ON vs OFF (speaker zone)") & (stats["phase"] == "all")].iloc[0]
                if stats is not None and len(stats)
                else {}
            )
            rows_e.append(
                {
                    "label": e.label,
                    "frequency_hz": e.spec.frequency_hz,
                    "stimulus_type": e.spec.stimulus_type,
                    "species": e.spec.species,
                    "sex": e.spec.sex,
                    "group": e.spec.group,
                    "n_cycles": r.get("n", np.nan) if len(r) else np.nan,
                    "mean_delta": r.get("mean_delta", np.nan) if len(r) else np.nan,
                    "ci_low": r.get("wild_cluster_ci_low", np.nan) if len(r) else np.nan,
                    "ci_high": r.get("wild_cluster_ci_high", np.nan) if len(r) else np.nan,
                    "p_value": r.get("wild_cluster_p_value", np.nan) if len(r) else np.nan,
                }
            )
        table = experiment_table(rows_e)
        pooled["experiments"] = table
        pooled["confounds"] = pd.DataFrame({"note": confounds(table)})
        contrasts = [
            group_contrast(table, d)
            for d in ("group", "frequency_hz", "species", "sex")
            if len(set(table[d].dropna())) > 1
        ]
        if contrasts:
            pooled["group_contrasts"] = pd.concat(contrasts, ignore_index=True)
    result.pooled = pooled
    _write(pooled, result.run_dir / "pooled")
    if figs:
        from buzzphono.plotting.figures import pooled_figures

        result.figures = pooled_figures(result)
    say(1.0, "Pooled results written")


def manifest_inputs(result: RunResult) -> list[dict[str, Any]]:
    """Each experiment's identity for the manifest (folder, chunks, calibration)."""
    out = []
    for e in result.experiments:
        out.append(
            {
                "label": e.label,
                "path": str(e.spec.path),
                "clock": e.clock_message,
                "ingest": e.ingest,
                "n_cycles": len(e.cycles),
                "n_usable_cycles": int(e.cycles["usable"].sum()) if len(e.cycles) else 0,
            }
        )
    return out


# ----------------------------------------------------------------------------- plan §5 and §8
BAND_ZONE = "speaker_band"
BAND_PX = 20.0
F2_SOURCES = (
    ("attraction_index/statistics", "attraction index (vs 0)", None),
    ("psth/onset_response", "PSTH onset response", None),
    (
        "approach/statistics",
        "approach: bouts per flight-second, ON vs OFF (validated measure)",
        "bouts_per_flight_s",
    ),
    ("spatial/band_statistics", "spatial: fraction in the speaker's 20-px band, ON vs OFF", None),
)


def secondary_family(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """F2 (docs/STATS_PLAN.md §5): Holm across the primary contrast of each secondary analysis run."""
    rows = []
    for key, label, contains in F2_SOURCES:
        t = tables.get(key)
        if t is None or not len(t) or "wild_cluster_p_value" not in t:
            continue
        sel = t[t["phase"] == "all"] if "phase" in t else t
        if contains is not None:
            sel = sel[sel["comparison"].str.contains(contains) & ~sel["comparison"].str.contains("sham")]
        if not len(sel):
            continue
        r = sel.iloc[0]
        rows.append(
            {
                "analysis": label,
                "n": r.get("n", np.nan),
                "mean_delta": r.get("mean_delta", np.nan),
                "ci_low": r.get("wild_cluster_ci_low", np.nan),
                "ci_high": r.get("wild_cluster_ci_high", np.nan),
                "p_value": r.get("wild_cluster_p_value", np.nan),
                "method": r.get("wild_cluster_method", ""),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        p = out["p_value"].to_numpy(dtype=np.float64)
        adj = np.full(p.size, np.nan)
        fin = np.isfinite(p)
        if fin.any():
            from buzzphono.stats import holm

            adj[fin] = holm(p[fin])
        out["p_holm_F2"] = adj
    return out


def _interval_excludes_zero(row: pd.Series, prefix: str = "wild_cluster") -> bool:
    lo, hi = row.get(f"{prefix}_ci_low", np.nan), row.get(f"{prefix}_ci_high", np.nan)
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def conclusion(main: pd.Series, tone_minus_sham: pd.Series | None, alpha: float = 0.05) -> str:
    """The plan's wording (§8), with the sham tested directly.

    *Evidence of an effect* needs the primary interval to exclude 0, the primary p-value and the
    block sign-flip p-value to be below ``alpha``, and the tone's change to differ from the sham's
    (the per-cycle difference (ON − OFF) − (ON' − OFF') has an interval excluding 0, same sign).
    """
    p = main.get("wild_cluster_p_value", np.nan)
    block_p = main.get("block_signflip_p_value", np.nan)
    wil_p = main.get("wilcoxon_p_value", np.nan)
    primary = _interval_excludes_zero(main) and np.isfinite(p) and p < alpha
    if primary and np.isfinite(block_p) and block_p < alpha:
        if tone_minus_sham is not None:
            same_sign = np.sign(tone_minus_sham.get("mean_delta", np.nan)) == np.sign(
                main.get("mean_delta", np.nan)
            )
            if not (_interval_excludes_zero(tone_minus_sham) and same_sign):
                return "Not attributable to the tone (the sham changes similarly)"
        text = "Evidence of an effect"
        if "cage_cluster_ci_low" in main and not _interval_excludes_zero(main, "cage_cluster"):
            text += " (not confirmed with whole cages as units)"
        return text
    if np.isfinite(wil_p) and wil_p < alpha:
        return "Suggestive (per-cycle Wilcoxon only)"
    return "No clear effect"


def add_conclusions(stats: pd.DataFrame) -> pd.DataFrame:
    """A ``conclusion`` on the headline row (all phases pooled). Phase rows are descriptive (see their
    Holm-adjusted p-values); a phase with too few cycles says it was not tested."""
    out = stats.copy()
    out["conclusion"] = ""
    main = out["comparison"].isin(["ON vs OFF (speaker zone)", "ON vs OFF (all experiments)"])
    tms = out[out["comparison"].str.startswith("tone minus sham") & (out["phase"] == "all")]
    tms_row = tms.iloc[0] if len(tms) else None
    for i in out.index[main]:
        if out.loc[i, "phase"] == "all":
            out.loc[i, "conclusion"] = conclusion(out.loc[i], tms_row)
        elif not np.isfinite(out.loc[i].get("wild_cluster_p_value", np.nan)):
            out.loc[i, "conclusion"] = "not tested (fewer than 4 cycles)"
        else:
            out.loc[i, "conclusion"] = "phase: see Holm-adjusted p"
    return out


__all__ = ["ANALYSES", "ExperimentResult", "RunResult", "default_run_dir", "run"]
