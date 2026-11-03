"""Calibration of one recording: frame size, cage corners and zones, in image pixels.

Pixel coordinates mean something only for one camera position and one sensor mode: a Camera Module 3
crops a different field of view from a v2 camera, and cameras get moved or swapped between
experiments. So nothing in BuzzPhono hard-codes a pixel position. Each experiment carries a
calibration sidecar, ``buzzphono_calibration.json``, drawn on (and tied to) one of its own reference
frames. Analyses refuse to run when the frame size of the data does not match it, and warn when
the camera seems to have moved since it was drawn.

The sidecar holds the four cage corners (top-left, top-right, bottom-right, bottom-left), one zone
with role ``speaker``, any number of control zones, and optionally the physical size of the cage
face in millimetres. With that size, positions convert to millimetres through the perspective
transform (homography) of the cage corners, because the camera sees the cage face as a trapezoid.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from buzzphono import __version__
from buzzphono.config import plain_english
from buzzphono.errors import CalibrationError, ConfigError
from buzzphono.geometry import as_polygon, contains
from buzzphono.io.cache import atomic_write_text

CALIBRATION_FILE = "buzzphono_calibration.json"
Point = tuple[float, float]


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    role: Literal["speaker", "control", "other"] = "other"
    polygon: list[Point] = Field(min_length=3)

    @field_validator("polygon")
    @classmethod
    def _finite(cls, v: list[Point]) -> list[Point]:
        if not np.isfinite(np.asarray(v, dtype=float)).all():
            raise ValueError("every vertex must be a finite number")
        return v


class PhysicalSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cage_width_mm: float = Field(gt=0, description="inner width of the filmed cage face, mm")
    cage_height_mm: float = Field(gt=0, description="inner height of the filmed cage face, mm")


class Calibration(BaseModel):
    """Contents of ``buzzphono_calibration.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    frame_width_px: int = Field(gt=0)
    frame_height_px: int = Field(gt=0)
    cage: list[Point] = Field(min_length=4, max_length=4, description="TL, TR, BR, BL corners")
    zones: list[Zone]
    physical: PhysicalSize | None = None
    reference_image: str | None = None
    reference_sha256: str | None = None
    camera: str | None = None
    sessions: list[str] | None = None  # None = every session of the experiment
    created: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    created_by: str = f"buzzphono {__version__}"
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Calibration:
        speakers = [z.name for z in self.zones if z.role == "speaker"]
        if len(speakers) != 1:
            raise ValueError(f"there must be exactly one zone with role 'speaker' (found {len(speakers)})")
        names = [z.name for z in self.zones]
        if len(set(names)) != len(names):
            raise ValueError("zone names must be unique")
        w, h = self.frame_width_px, self.frame_height_px
        tol = 2.0
        for label, pts in [("cage", self.cage)] + [(f"zone '{z.name}'", z.polygon) for z in self.zones]:
            a = np.asarray(pts, dtype=float)
            if (
                (a[:, 0] < -tol).any()
                or (a[:, 0] > w + tol).any()
                or (a[:, 1] < -tol).any()
                or (a[:, 1] > h + tol).any()
            ):
                raise ValueError(f"the {label} has points outside the {w}×{h} frame")
        return self

    # --------------------------------------------------------------------- convenience
    @property
    def frame_size(self) -> tuple[int, int]:
        return self.frame_width_px, self.frame_height_px

    @property
    def speaker(self) -> Zone:
        return next(z for z in self.zones if z.role == "speaker")

    @property
    def controls(self) -> list[Zone]:
        return [z for z in self.zones if z.role == "control"]

    def polygon(self, name: str) -> np.ndarray:
        for z in self.zones:
            if z.name == name:
                return as_polygon(z.polygon)
        raise CalibrationError(
            f"No zone named '{name}' in the calibration (zones: {', '.join(z.name for z in self.zones)})."
        )

    def cage_polygon(self) -> np.ndarray:
        return np.asarray(self.cage, dtype=np.float64)

    def homography(self) -> np.ndarray | None:
        """3×3 matrix taking image pixels to millimetres on the cage face (None without a size)."""
        if self.physical is None:
            return None
        w, h = self.physical.cage_width_mm, self.physical.cage_height_mm
        return homography_from_points(self.cage_polygon(), np.array([[0, 0], [w, 0], [w, h], [0, h]], float))

    def to_mm(self, xy: np.ndarray) -> np.ndarray:
        """Pixel positions -> millimetres on the cage face. Raises if no physical size was given."""
        hmat = self.homography()
        if hmat is None:
            raise CalibrationError(
                "This calibration has no physical cage size, so millimetres are unknown.",
                hint="Add the cage face width and height in mm with `buzzphono calibrate`.",
            )
        return apply_homography(hmat, xy)


# ------------------------------------------------------------------------- homography
def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Perspective transform mapping 4 source points onto 4 destination points (h33 = 1)."""
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst, strict=True)):
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i], b[2 * i + 1] = u, v
    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError("The cage corners are degenerate (three in a line, or repeated).") from exc
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(hmat: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    hom = np.c_[pts, np.ones(len(pts))] @ hmat.T
    return hom[:, :2] / hom[:, 2:3]


# ------------------------------------------------------------------------- files
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_calibration(path: Path) -> Calibration:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CalibrationError(
            f"No calibration at {path}.",
            hint="Run `buzzphono calibrate <experiment>` (or the app's Calibrate page).",
        ) from exc
    except ValueError as exc:
        raise ConfigError(f"{path} is not valid JSON ({exc}).") from exc
    try:
        return Calibration.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(plain_english(exc, f"calibration file {path}")) from exc


def save_calibration(cal: Calibration, path: Path) -> Path:
    atomic_write_text(path, cal.model_dump_json(indent=2) + "\n")
    return path


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def reference_images(experiment_dir: Path) -> list[Path]:
    """Per-chunk reference frames written by BuzzWatch (individual_images/<chunk>.png), sorted."""
    folder = experiment_dir / "individual_images"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.png") if not p.name.startswith("._"))


def import_existing(experiment_dir: Path, zones_file: Path | None = None) -> Calibration:
    """Build a calibration from what BuzzWatch already has: ``custom_zones.json`` (speaker + other
    zones), the cage corners in ``buzzwatch_track_settings.yml`` and the size of a reference frame."""
    import yaml

    images = reference_images(experiment_dir)
    bg = experiment_dir / "background_with_borders.png"
    ref = images[len(images) // 2] if images else (bg if bg.is_file() else None)
    if ref is None:
        raise CalibrationError(
            f"{experiment_dir} has no reference frame (individual_images/*.png).",
            hint="Run BuzzWatch's set-up step (Extract images) for this experiment first.",
        )
    width, height = image_size(ref)
    zones_path = zones_file or experiment_dir / "custom_zones.json"
    try:
        raw = json.loads(zones_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CalibrationError(
            f"No zone file at {zones_path}.", hint="Draw the speaker zone with `buzzphono calibrate --click`."
        ) from exc
    zones = []
    for name, spec in raw.items():
        vertices = spec.get("polygon") if isinstance(spec, dict) else spec
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        role = "speaker" if name == "speaker" else "control"
        zones.append(Zone(name=name, role=role, polygon=[(float(x), float(y)) for x, y in vertices]))
    settings = experiment_dir / "buzzwatch_track_settings.yml"
    try:
        cage = yaml.safe_load(settings.read_text(encoding="utf-8"))["cage_border_points"]
    except (FileNotFoundError, KeyError, TypeError) as exc:
        raise CalibrationError(
            f"No cage corners (cage_border_points) in {settings}.",
            hint="Draw the cage with `buzzphono calibrate --click`.",
        ) from exc
    try:
        return Calibration(
            frame_width_px=width,
            frame_height_px=height,
            cage=[(float(x), float(y)) for x, y in cage],
            zones=zones,
            reference_image=str(ref.relative_to(experiment_dir)).replace("\\", "/"),
            reference_sha256=sha256_of(ref),
            notes=f"Imported from {zones_path.name} and {settings.name}.",
        )
    except ValidationError as exc:
        raise ConfigError(plain_english(exc, f"the zones imported from {zones_path}")) from exc


# ------------------------------------------------------------------------- checks
def polygon_iou(a: np.ndarray, b: np.ndarray, frame: tuple[int, int], step: float = 4.0) -> float:
    """Intersection over union of two polygons, on a grid of ``step``-pixel cells."""
    xs = np.arange(step / 2, frame[0], step)
    ys = np.arange(step / 2, frame[1], step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel()]
    ma, mb = contains(as_polygon(a), pts), contains(as_polygon(b), pts)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def _gray(path: Path, factor: int = 2) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        g = im.convert("L")
        if factor > 1:
            g = g.reduce(factor)
        return np.asarray(g, dtype=np.float64)


def phase_shift(reference: np.ndarray, image: np.ndarray) -> tuple[float, float, float]:
    """Translation (dx, dy) of ``image`` relative to ``reference`` and the correlation peak strength.

    Both are high-pass filtered first (so lighting changes and slow gradients do not dominate) and
    windowed. A peak strength below about 0.05 means the estimate is unreliable.
    """
    from scipy.ndimage import gaussian_filter

    def prep(a: np.ndarray) -> np.ndarray:
        hp = a - gaussian_filter(a, 8)
        win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
        return (hp - hp.mean()) * win

    fa, fb = np.fft.fft2(prep(reference)), np.fft.fft2(prep(image))
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.real(np.fft.ifft2(cross))
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = iy - corr.shape[0] if iy > corr.shape[0] // 2 else iy
    dx = ix - corr.shape[1] if ix > corr.shape[1] // 2 else ix
    return float(dx), float(dy), float(corr.max())
