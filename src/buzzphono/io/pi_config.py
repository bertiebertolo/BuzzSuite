"""Read and check the Raspberry Pi recorder's ``config.ini`` (the Picamera2 BuzzWatch recorder).

The recorder, light and sound services all read one ``config.ini``. A mistake there (for example
a ``/home/penzance/`` path on a Pi whose user is ``penzancepi``) silently breaks a recording, so
every field is checked and every problem is named in plain English. The file also tells
BuzzPhono the recording's frame size and rate, the tone schedule and frequency, and the light
ramps, which are cross-checked against the data.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from buzzphono.errors import ConfigError

Severity = Literal["error", "warning", "info"]
_HHMM = re.compile(r"^(\d{1,2}):(\d{2})$")
_WINDOW = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")


@dataclass(frozen=True)
class Check:
    severity: Severity
    field: str  # e.g. "[sound] sound_file"
    message: str


@dataclass
class PiConfig:
    """The parts of the Pi config that matter for analysis (None when absent)."""

    path: Path
    label: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    segment_duration_s: int | None = None
    sound_enabled: bool | None = None
    sound_mode: str | None = None
    frequency_hz: float | None = None
    windows: list[tuple[int, int]] = field(default_factory=list)  # half-open minutes of every hour
    active_from_min: int | None = None  # minutes after midnight
    active_until_min: int | None = None
    ramp_up_start_min: int | None = None
    ramp_up_end_min: int | None = None
    ramp_down_start_min: int | None = None
    ramp_down_end_min: int | None = None
    paths: dict[str, str] = field(default_factory=dict)  # "[section] key" -> path
    checks: list[Check] = field(default_factory=list)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "error"]


def _hhmm(value: str) -> int | None:
    m = _HHMM.match(value.strip())
    if not m:
        return None
    h, mnt = int(m.group(1)), int(m.group(2))
    if (h, mnt) == (24, 0):
        return 24 * 60
    if 0 <= h < 24 and 0 <= mnt < 60:
        return h * 60 + mnt
    return None


def parse_windows(text: str) -> list[tuple[int, int]]:
    """``"40-50"`` or ``"10-20, 40-50"`` -> [(40, 50)] (minutes of the hour, half-open)."""
    out = []
    for part in text.split(","):
        m = _WINDOW.match(part)
        if not m:
            raise ValueError(f"'{part.strip()}' is not a minute range like 40-50")
        a, b = int(m.group(1)), int(m.group(2))
        if not (0 <= a <= 60 and 0 <= b <= 60) or a == b:
            raise ValueError(f"'{part.strip()}' is not a valid minute range (0-60, start different from end)")
        out.append((a, b))
    return out


def _home(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "home":
        return parts[2]
    return None


def read_pi_config(path: str | Path) -> PiConfig:
    """Parse and check a Pi ``config.ini``. Problems go into ``.checks``; unreadable files raise."""
    path = Path(path)
    parser = configparser.ConfigParser(interpolation=None)
    try:
        read = parser.read(path, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path} is not a valid config.ini: {exc}") from exc
    if not read:
        raise ConfigError(f"Could not read the Pi config file {path}.")
    cfg = PiConfig(path=path)
    add = cfg.checks.append

    def get(section: str, key: str) -> str | None:
        return parser.get(section, key, fallback=None) if parser.has_section(section) else None

    def number(section: str, key: str, kind: type, lo: float, hi: float) -> float | int | None:
        raw = get(section, key)
        if raw is None:
            return None
        try:
            value = kind(raw.strip())
        except ValueError:
            add(Check("error", f"[{section}] {key}", f"'{raw}' is not a number."))
            return None
        if not lo <= value <= hi:
            add(Check("error", f"[{section}] {key}", f"{value} is outside the sensible range {lo}-{hi}."))
            return None
        return value

    for section in ("recording", "sound", "light"):
        if not parser.has_section(section):
            add(Check("warning", f"[{section}]", f"The [{section}] section is missing."))
    cfg.label = get("recording", "label")
    cfg.width = number("recording", "width", int, 16, 10000)  # type: ignore[assignment]
    cfg.height = number("recording", "height", int, 16, 10000)  # type: ignore[assignment]
    cfg.fps = number("recording", "fps", float, 1, 240)
    cfg.segment_duration_s = number("recording", "segment_duration", int, 1, 86400)  # type: ignore[assignment]
    enabled = get("sound", "enabled")
    if enabled is not None:
        low = enabled.strip().lower()
        if low in ("true", "yes", "1", "on"):
            cfg.sound_enabled = True
        elif low in ("false", "no", "0", "off"):
            cfg.sound_enabled = False
        else:
            add(Check("error", "[sound] enabled", f"'{enabled}' is not true or false."))
    cfg.sound_mode = (get("sound", "mode") or "").strip().lower() or None
    if cfg.sound_mode not in (None, "tone", "file"):
        add(Check("error", "[sound] mode", f"'{cfg.sound_mode}' is not 'tone' or 'file'."))
    cfg.frequency_hz = number("sound", "frequency_hz", float, 1, 100000)
    if cfg.sound_mode == "tone" and cfg.frequency_hz is None and cfg.sound_enabled:
        add(Check("error", "[sound] frequency_hz", "A tone needs a frequency (frequency_hz)."))
    windows = get("sound", "windows")
    if windows is not None:
        try:
            cfg.windows = parse_windows(windows)
        except ValueError as exc:
            add(Check("error", "[sound] windows", str(exc) + "."))
    for key, attr in (("active_from", "active_from_min"), ("active_until", "active_until_min")):
        raw = get("sound", key)
        if raw is not None:
            value = _hhmm(raw)
            if value is None:
                add(Check("error", f"[sound] {key}", f"'{raw}' is not a time like 07:30."))
            setattr(cfg, attr, value)
    for key in ("ramp_up_start", "ramp_up_end", "ramp_down_start", "ramp_down_end"):
        raw = get("light", key)
        if raw is not None:
            value = _hhmm(raw)
            if value is None:
                add(Check("error", f"[light] {key}", f"'{raw}' is not a time like 05:00."))
            setattr(cfg, f"{key}_min", value)
    for ramp, start, end in (
        ("up", cfg.ramp_up_start_min, cfg.ramp_up_end_min),
        ("down", cfg.ramp_down_start_min, cfg.ramp_down_end_min),
    ):
        if start is not None and end is not None and end < start:
            add(Check("error", f"[light] ramp_{ramp}_end", f"The ramp {ramp} ends before it starts."))
    _check_paths(parser, cfg)
    return cfg


def _check_paths(parser: configparser.ConfigParser, cfg: PiConfig) -> None:
    for section in parser.sections():
        for key, value in parser.items(section):
            v = value.strip()
            if v.startswith("/"):
                cfg.paths[f"[{section}] {key}"] = v
    homes: dict[str, list[str]] = {}
    for fieldname, p in cfg.paths.items():
        user = _home(p)
        if user is not None:
            homes.setdefault(user, []).append(fieldname)
    if len(homes) > 1:
        majority = max(homes, key=lambda u: len(homes[u]))
        for user, fields in homes.items():
            if user == majority:
                continue
            for f in fields:
                used = not (f == "[sound] sound_file" and cfg.sound_mode == "tone")
                where = ", ".join(homes[majority])
                add_msg = (
                    f"{cfg.paths[f]} is under /home/{user}/, but {where} "
                    f"{'is' if len(homes[majority]) == 1 else 'are'} under /home/{majority}/. "
                    "A Pi has one user's home folder, so this is almost certainly a typo."
                )
                if not used:
                    add_msg += " (Not used while [sound] mode = tone, but it will break mode = file.)"
                cfg.checks.append(Check("error" if used else "warning", f, add_msg))


def cross_check(
    cfg: PiConfig,
    *,
    frame_size: tuple[int, int] | None = None,
    labels: set[str] | None = None,
    on_window: tuple[int, int] | None = None,
    frequency_hz: float | None = None,
    fps: float | None = None,
) -> list[Check]:
    """Compare the config with the recording it is supposed to describe."""
    out: list[Check] = []
    if labels and cfg.label and cfg.label not in labels:
        out.append(
            Check(
                "warning",
                "[recording] label",
                f"This config.ini is for label '{cfg.label}', but the recordings are labelled "
                f"{', '.join(sorted(labels))}: it may not be the config used for this recording.",
            )
        )
    if frame_size and cfg.width and cfg.height and (cfg.width, cfg.height) != tuple(frame_size):
        out.append(
            Check(
                "error",
                "[recording] width/height",
                f"The config records {cfg.width}×{cfg.height} but the videos are "
                f"{frame_size[0]}×{frame_size[1]}.",
            )
        )
    if fps and cfg.fps and abs(cfg.fps - fps) > 1e-6:
        out.append(
            Check("warning", "[recording] fps", f"The config says {cfg.fps} fps; the tracking used {fps}.")
        )
    if on_window and cfg.windows and tuple(on_window) not in [tuple(w) for w in cfg.windows]:
        out.append(
            Check(
                "error",
                "[sound] windows",
                f"The tone played in minutes {cfg.windows} but the analysis uses minutes "
                f"{on_window[0]}-{on_window[1]} as ON.",
            )
        )
    if frequency_hz and cfg.frequency_hz and abs(cfg.frequency_hz - frequency_hz) > 0.5:
        out.append(
            Check(
                "warning",
                "[sound] frequency_hz",
                f"The config plays {cfg.frequency_hz:g} Hz; the experiment is described as "
                f"{frequency_hz:g} Hz.",
            )
        )
    if cfg.sound_enabled is False:
        out.append(
            Check("error", "[sound] enabled", "Sound was disabled in this config: no stimulus played.")
        )
    return out
