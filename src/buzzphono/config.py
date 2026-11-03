"""BuzzPhono's settings: every option, its default and unit, checked the moment it is read.

Settings come from a YAML file (``buzzphono analyse --config my.yml``), the command line or the
app; all three end up in :class:`AnalysisConfig`. Mistakes are reported in plain English, naming
the setting, the value given and what was expected; an unknown setting (usually a typo) is an
error, with the closest valid name suggested. docs/BUZZPHONO_GUIDE.md lists every setting.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from buzzphono.errors import ConfigError

ANALYSES = (
    "resting_fraction",
    "attraction_index",
    "psth",
    "approach",
    "zt_profile",
    "spatial",
    "comparison",
    "reliability",
    "sensitivity",
    "flight",
)
Analysis = Literal[
    "resting_fraction",
    "attraction_index",
    "psth",
    "approach",
    "zt_profile",
    "spatial",
    "comparison",
    "reliability",
    "sensitivity",
    "flight",
]
_HHMM = re.compile(r"^(\d{1,2}):(\d{2})$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Window(_Strict):
    """Minutes of every clock hour, start included, end excluded (e.g. 40-50)."""

    start_min: int = Field(ge=0, le=59)
    end_min: int = Field(ge=1, le=60)

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            m = re.fullmatch(r"\s*(\d{1,2})\s*-\s*(\d{1,2})\s*", value)
            if not m:
                raise ValueError(f"'{value}' is not a minute range like 40-50")
            return {"start_min": int(m.group(1)), "end_min": int(m.group(2))}
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"start_min": value[0], "end_min": value[1]}
        return value

    @model_validator(mode="after")
    def _ordered(self) -> Window:
        if self.end_min <= self.start_min:
            raise ValueError(f"the window must end after it starts (got {self.start_min}-{self.end_min})")
        return self

    def as_tuple(self) -> tuple[int, int]:
        return self.start_min, self.end_min

    def __str__(self) -> str:
        return f"{self.start_min}-{self.end_min}"


def _hhmm_to_min(value: str) -> int:
    m = _HHMM.match(value.strip())
    if not m:
        raise ValueError(f"'{value}' is not a time like 05:00")
    h, mm = int(m.group(1)), int(m.group(2))
    if (h, mm) != (24, 0) and not (0 <= h < 24 and 0 <= mm < 60):
        raise ValueError(f"'{value}' is not a valid time of day")
    return h * 60 + mm


class StimulusSettings(_Strict):
    on: Window = Window(start_min=40, end_min=50)
    off: Window = Window(start_min=28, end_min=38)
    sham_on: Window = Window(start_min=20, end_min=30)
    sham_off: Window = Window(start_min=8, end_min=18)
    pre: Window | None = Field(default=None, description="pre-window of the attraction index; default = off")
    source: Literal["auto", "playback_log", "pi_config", "schedule"] = "auto"
    playback_log: list[Path] = Field(default_factory=list)
    pi_config: Path | None = None
    active_from: str = "00:00"
    active_until: str = "24:00"

    @field_validator("active_from", "active_until")
    @classmethod
    def _time(cls, v: str) -> str:
        _hhmm_to_min(v)
        return v

    @model_validator(mode="after")
    def _no_overlap(self) -> StimulusSettings:
        a, b = self.on.as_tuple(), self.off.as_tuple()
        if a[0] < b[1] and b[0] < a[1]:
            raise ValueError(f"the ON window ({self.on}) and the OFF window ({self.off}) overlap")
        return self


class ClockSettings(_Strict):
    mode: Literal["corrected", "thesis", "nominal"] = "corrected"
    fps: float = Field(default=25.0, gt=0, le=240, description="nominal camera frame rate, frames/s")
    rig_drift_s_per_day: float | None = Field(default=None, ge=-300, le=300)
    manual_offset_s: dict[str, float] = Field(default_factory=dict, description="per session, seconds")


class LightSettings(_Strict):
    lights_on: str = "05:00"
    lights_off: str = "17:00"
    source: Literal["auto", "settings"] = "auto"

    @field_validator("lights_on", "lights_off")
    @classmethod
    def _time(cls, v: str) -> str:
        _hhmm_to_min(v)
        return v

    def phases(self) -> dict[int, str]:
        """Hour of day -> phase: dawn = the lights-on hour, dusk = the hour before lights-off."""
        on_h = _hhmm_to_min(self.lights_on) // 60
        off_h = (_hhmm_to_min(self.lights_off) // 60) % 24
        dusk_h = (off_h - 1) % 24
        out = {}
        for h in range(24):
            if h == on_h:
                out[h] = "dawn"
            elif h == dusk_h:
                out[h] = "dusk"
            elif (on_h < dusk_h and on_h < h < dusk_h) or (on_h > dusk_h and (h > on_h or h < dusk_h)):
                out[h] = "day"
            else:
                out[h] = "night"
        return out


class CoverageRule(_Strict):
    min_frame_fraction: float = Field(default=0.8, ge=0, le=1)
    max_gap_s: float = Field(default=120.0, gt=0)
    min_minutes_with_resting: int = Field(default=8, ge=0, le=60)


class MetricSettings(_Strict):
    colony_size: int = Field(default=30, ge=1, le=1000, description="mosquitoes in the cage")
    speaker_zone: str = "speaker"
    coverage: CoverageRule = CoverageRule()


class FlightSettings(_Strict):
    flyer_multiplier: float = Field(default=1.0, gt=0, le=10)
    min_median: float = Field(default=0.0, ge=0, le=20, description="px/frame × camera scale; 0 = off")
    radius_cage_widths: float = Field(default=0.1, gt=0, le=1)


class ApproachSettings(_Strict):
    min_points: int = Field(default=10, ge=2)
    smoothing_frames: int = Field(default=5, ge=1, le=51)
    min_drop_mm: float = Field(default=20.0, ge=0)
    min_drop_cage_fraction: float = Field(default=0.15, ge=0, le=1)
    min_duration_s: float = Field(default=0.4, ge=0)


class SensitivitySettings(_Strict):
    zone_margins_px: list[float] = Field(default_factory=lambda: [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0])
    zone_shifts_px: list[float] = Field(default_factory=lambda: [-40.0, -20.0, 0.0, 20.0, 40.0, 60.0])
    walking_thresholds_mm_s: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    colony_sizes: list[int] = Field(default_factory=lambda: [25, 30, 35])


class StatsSettings(_Strict):
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    alpha: float = Field(default=0.05, gt=0, lt=1)
    n_resamples: int = Field(default=9999, ge=199, le=1_000_000)
    block_hours: int = Field(default=12, ge=1, le=24)
    seed: int = 20261104


def _all_formats() -> list[Literal["png", "svg", "pdf"]]:
    return ["png", "svg", "pdf"]


class OutputSettings(_Strict):
    dir: Path | None = None
    cache_dir: Path | None = None
    formats: list[Literal["png", "svg", "pdf"]] = Field(default_factory=_all_formats)
    dpi: int = Field(default=300, ge=72, le=1200)
    report_pdf: bool = False


class ExperimentSpec(_Strict):
    path: Path
    label: str | None = None
    group: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    frequency_hz: float | None = Field(default=None, gt=0)
    stimulus_type: Literal["tone", "file", "piezo", "none"] = "tone"
    species: str | None = None
    sex: Literal["male", "female", "mixed"] | None = None
    calibration: Path | None = None

    @field_validator("path")
    @classmethod
    def _exists(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"folder not found: {v}")
        return v

    @model_validator(mode="after")
    def _window(self) -> ExperimentSpec:
        if self.start and self.end and self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")
        return self

    @property
    def name(self) -> str:
        return self.label or self.path.name


def _default_analyses() -> list[Analysis]:
    return ["resting_fraction"]


class AnalysisConfig(_Strict):
    experiments: list[ExperimentSpec] = Field(min_length=1)
    analyses: list[Analysis] = Field(default_factory=_default_analyses)
    clock: ClockSettings = ClockSettings()
    stimulus: StimulusSettings = StimulusSettings()
    light: LightSettings = LightSettings()
    metric: MetricSettings = MetricSettings()
    flight: FlightSettings = FlightSettings()
    approach: ApproachSettings = ApproachSettings()
    sensitivity: SensitivitySettings = SensitivitySettings()
    stats: StatsSettings = StatsSettings()
    output: OutputSettings = OutputSettings()
    workers: int | None = Field(default=None, ge=1, le=61)

    @field_validator("analyses")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


# ------------------------------------------------------------------------- errors in plain English
def _fields_at(model: type[BaseModel], loc: tuple[Any, ...]) -> list[str]:
    current: Any = model
    for part in loc:
        if isinstance(part, int):
            continue
        info = getattr(current, "model_fields", {}).get(part)
        if info is None:
            return []
        ann = info.annotation
        args = [a for a in getattr(ann, "__args__", ()) if isinstance(a, type) and issubclass(a, BaseModel)]
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            current = ann
        elif args:
            current = args[0]
        else:
            return []
    return list(getattr(current, "model_fields", {}))


def plain_english(exc: ValidationError, where: str, model: type[BaseModel] | None = None) -> str:
    """One readable line per problem: which setting, what was given, what was expected."""
    lines = [f"Problems in {where}:"]
    for err in exc.errors():
        loc = tuple(err.get("loc", ()))
        name = ".".join(str(p) for p in loc) or "(the file as a whole)"
        given = err.get("input")
        kind = err.get("type", "")
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        if kind == "extra_forbidden":
            known = _fields_at(model, loc[:-1]) if model else []
            close = difflib.get_close_matches(str(loc[-1]), known, n=1) if known else []
            hint = f" Did you mean '{close[0]}'?" if close else ""
            lines.append(f"- '{name}' is not a BuzzPhono setting.{hint}")
        elif kind == "missing":
            lines.append(f"- '{name}' is required but missing.")
        elif given is None or isinstance(given, dict):
            lines.append(f"- {name}: {msg}.")
        else:
            lines.append(f"- {name}: {msg} (you gave {given!r}).")
    return "\n".join(lines)


def build_config(data: dict[str, Any], where: str = "the settings") -> AnalysisConfig:
    try:
        return AnalysisConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(plain_english(exc, where, AnalysisConfig)) from exc


def load_config(path: Path) -> AnalysisConfig:
    """Read a YAML settings file. Relative experiment paths are taken from the file's folder."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Settings file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain settings (key: value pairs), not a {type(data).__name__}.")
    for exp in data.get("experiments") or []:
        if isinstance(exp, dict) and isinstance(exp.get("path"), str) and not Path(exp["path"]).is_absolute():
            exp["path"] = str((path.parent / exp["path"]).resolve())
    return build_config(data, f"settings file {path}")
