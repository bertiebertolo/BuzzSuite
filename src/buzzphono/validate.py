"""Pre-flight checks of one experiment, in plain English: what will work, what will not, and why.

Checks the tracking files (present, readable names, empty files), the recording sessions (stamps,
missing and short chunks, overlaps, clock changes), the calibration (present, frame size equal
to the reference frames, zones inside the frame), and, when given, the Pi's ``config.ini`` and
playback log (cross-checked against the recording). Nothing is written.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from buzzphono.errors import BuzzPhonoError
from buzzphono.io.discover import discover
from buzzphono.qc import layout_findings
from buzzphono.timebase import build_sessions

NOMINAL_NF = 30060


@dataclass(frozen=True)
class Check:
    severity: str  # "ok", "info", "warning", "error"
    area: str
    message: str


def _window(text: str) -> tuple[int, int]:
    a, _, b = text.partition("-")
    return int(a), int(b)


def validate_experiment(
    experiment: Path,
    *,
    pi_config: Path | None = None,
    playback_logs: Iterable[Path] = (),
    on: str = "40-50",
    frequency_hz: float | None = None,
) -> list[Check]:
    out: list[Check] = []
    if not experiment.is_dir():
        return [Check("error", "folder", f"{experiment} is not a folder.")]
    layout = discover(experiment)
    n = len(layout.chunks)
    if n == 0:
        out.append(
            Check("error", "tracking", "No tracking files: track the experiment with BuzzWatch first.")
        )
        return out
    for kind in ("resting", "moving", "tracks"):
        missing = layout.missing(kind)
        if not missing:
            out.append(Check("ok", "tracking", f"{n} chunks have {kind} files."))
        elif len(missing) == n:
            sev = "error" if kind == "resting" else "warning"
            out.append(
                Check(
                    sev,
                    "tracking",
                    f"No {kind} files (needed for {'every analysis' if kind == 'resting' else 'the clock fit and flight analyses'}).",
                )
            )
        else:
            out.append(Check("warning", "tracking", f"{len(missing)} of {n} chunks have no {kind} file."))
    if layout.empty_files:
        out.append(
            Check("warning", "tracking", f"{len(layout.empty_files)} empty tracking file(s) will be skipped.")
        )
    if layout.unparsable:
        out.append(
            Check(
                "warning",
                "tracking",
                f"{len(layout.unparsable)} file name(s) do not follow <cage>_<YYYYMMDD>_<HHMMSS>_<NNNNN> and are ignored.",
            )
        )
    names = [c.name for c in layout.chunks]
    # frame counts are known only once the tracking files have been read into the cache
    from buzzphono.io.cache import CacheIndex, cache_dir_for
    from buzzphono.timebase import frames_per_chunk

    nf = frames_per_chunk(CacheIndex.open(cache_dir_for(experiment), experiment).data["chunks"])
    sessions = build_sessions(names, nf, default_nf=NOMINAL_NF)
    for f in layout_findings(names, sessions):
        if not nf and "frame counts come from" in f.message:
            continue  # not read yet: says nothing about the recording
        out.append(Check(f.severity, f.area, f.message))
    out.append(
        Check(
            "info",
            "sessions",
            f"{len(sessions)} recording session(s): "
            + ", ".join(sorted(sessions)[:6])
            + (" ..." if len(sessions) > 6 else ""),
        )
    )

    from buzzphono.calibration import image_size, reference_images
    from buzzphono.pipeline import resolve_calibration

    frame: tuple[int, int] | None = None
    labels = {c.name.label for c in layout.chunks}
    try:
        cal = resolve_calibration(experiment)
        out.append(
            Check(
                "ok",
                "calibration",
                f"Zones: {', '.join(f'{z.name} ({z.role})' for z in cal.zones)}; frame {cal.frame_width_px}×{cal.frame_height_px} px.",
            )
        )
        refs = reference_images(experiment)
        if refs:
            frame = image_size(refs[0])
            if frame != cal.frame_size:
                out.append(
                    Check(
                        "error",
                        "calibration",
                        f"The video frames are {frame[0]}×{frame[1]} px but the calibration is for {cal.frame_width_px}×{cal.frame_height_px} px: the camera or its resolution changed. Calibrate again for this recording.",
                    )
                )
        if not cal.controls:
            out.append(
                Check(
                    "warning",
                    "calibration",
                    "No control zones: the speaker-vs-control comparison will be skipped.",
                )
            )
        if cal.physical is None:
            out.append(Check("info", "calibration", "No cage size in mm: distances will be in pixels."))
    except BuzzPhonoError as exc:
        out.append(Check("error", "calibration", str(exc)))
    if pi_config is not None:
        from buzzphono.io.pi_config import cross_check, read_pi_config

        try:
            cfg = read_pi_config(pi_config)
            for c in cfg.checks + cross_check(
                cfg, frame_size=frame, labels=labels, on_window=_window(on), frequency_hz=frequency_hz
            ):
                out.append(Check(c.severity, f"Pi config {c.field}", c.message))
            if not cfg.checks:
                out.append(Check("ok", "Pi config", f"{pi_config.name} parsed with no problems."))
        except BuzzPhonoError as exc:
            out.append(Check("error", "Pi config", str(exc)))
    logs = list(playback_logs)
    if logs:
        from buzzphono.io.playback_log import read_playback_log

        try:
            log = read_playback_log(logs)
            failed = int((log.events["kind"] == "failed_start").sum()) if len(log.events) else 0
            out.append(
                Check(
                    "ok",
                    "playback log",
                    f"{len(log.intervals)} playback window(s); stimuli: {', '.join(log.stimuli) or 'unknown'}.",
                )
            )
            if failed:
                out.append(
                    Check(
                        "warning",
                        "playback log",
                        f"{failed} window(s) failed to start: those hour-cycles will be excluded.",
                    )
                )
        except BuzzPhonoError as exc:
            out.append(Check("error", "playback log", str(exc)))
    return out
