"""Names of recording chunks and the tracking files made from them.

The Raspberry Pi recorder names a session ``{label}_{YYYYMMDD}_{HHMMSS}`` from its own clock when
recording starts, adding ``-{attempt}`` when that name already exists (a sign the clock went
backwards). ``rpicam-vid --segment`` then cuts the session into ~20-minute chunks numbered
``_{NNNNN}``. BuzzWatch writes three tracking files per chunk:

==============  ==========================================  ========================
kind            file name                                   folder
==============  ==========================================  ========================
``resting``     ``resting_{stem}.pkl``                       ``tracking_resting/``
``moving``      ``moving_{stem}.pkl``                        ``tracking_moving/``
``tracks``      ``forward_mosq_tracks_{stem}`` (no suffix)   ``final_tracking_data/``
==============  ==========================================  ========================

where ``stem = {label}_{YYYYMMDD}_{HHMMSS}[-{attempt}]_{NNNNN}``. Times in a stem are the Pi's
wall clock, which is not trustworthy (no real-time clock): treat them as labels, and let
:mod:`buzzphono.timebase` decide what they mean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

TrackingKind = Literal["resting", "moving", "tracks"]

TRACKING_FOLDERS: dict[TrackingKind, str] = {
    "resting": "tracking_resting",
    "moving": "tracking_moving",
    "tracks": "final_tracking_data",
}
_PREFIX: dict[TrackingKind, str] = {
    "resting": "resting_",
    "moving": "moving_",
    "tracks": "forward_mosq_tracks_",
}
_SUFFIX: dict[TrackingKind, str] = {"resting": ".pkl", "moving": ".pkl", "tracks": ""}

STEM_RE = re.compile(
    r"^(?P<label>.+)_(?P<date>\d{8})_(?P<time>\d{6})(?:-(?P<attempt>\d+))?_(?P<index>\d{5})$"
)
# The legacy (published) code recognised only names without an attempt suffix.
LEGACY_STEM_RE = re.compile(r"_(\d{8})_(\d{6})_(\d{5})$")


@dataclass(frozen=True, order=True)
class ChunkName:
    """One recording chunk, parsed from its file-name stem."""

    label: str
    date: str  # YYYYMMDD as written by the Pi
    time: str  # HHMMSS as written by the Pi
    attempt: int | None
    index: int

    @property
    def session(self) -> str:
        """The recording session this chunk belongs to (label, start stamp and attempt)."""
        base = f"{self.label}_{self.date}_{self.time}"
        return base if self.attempt is None else f"{base}-{self.attempt}"

    @property
    def stem(self) -> str:
        return f"{self.session}_{self.index:05d}"

    def session_start(self) -> datetime | None:
        """The session start written by the Pi clock, or None if it is not a real date/time."""
        try:
            return datetime.strptime(self.date + self.time, "%Y%m%d%H%M%S")
        except ValueError:
            return None

    def filename(self, kind: TrackingKind) -> str:
        return f"{_PREFIX[kind]}{self.stem}{_SUFFIX[kind]}"


def parse_stem(stem: str) -> ChunkName | None:
    """Parse ``{label}_{YYYYMMDD}_{HHMMSS}[-{attempt}]_{NNNNN}``; None if it does not match."""
    m = STEM_RE.match(stem)
    if m is None:
        return None
    attempt = m.group("attempt")
    return ChunkName(
        label=m.group("label"),
        date=m.group("date"),
        time=m.group("time"),
        attempt=int(attempt) if attempt is not None else None,
        index=int(m.group("index")),
    )


def stem_from_filename(kind: TrackingKind, filename: str) -> str | None:
    """The chunk stem of a tracking file name, or None if the name is not a ``kind`` file.

    macOS metadata files (``._*``) are never tracking files.
    """
    if filename.startswith("._"):
        return None
    prefix, suffix = _PREFIX[kind], _SUFFIX[kind]
    if not filename.startswith(prefix):
        return None
    if suffix:
        if not filename.endswith(suffix):
            return None
        return filename[len(prefix) : -len(suffix)]
    if "." in filename[len(prefix) :]:
        return None  # e.g. forward_mosq_tracks_X.tmp: not a finished tracking file
    return filename[len(prefix) :]


def parse_filename(kind: TrackingKind, filename: str) -> ChunkName | None:
    stem = stem_from_filename(kind, filename)
    return parse_stem(stem) if stem is not None else None


def legacy_recognises(kind: TrackingKind, filename: str) -> bool:
    """Whether the published (thesis) code would have read this file (no ``-attempt`` names)."""
    stem = filename[: -len(_SUFFIX[kind])] if _SUFFIX[kind] and filename.endswith(_SUFFIX[kind]) else filename
    return LEGACY_STEM_RE.search(stem) is not None
