"""Turning (chunk, frame) into clock time.

The Raspberry Pis have no real-time clock, the camera's true frame rate is not exactly 25 fps, and
a "20-minute" chunk is cut at the first keyframe after 1200 s. So the time of a video frame is
never read from a timestamp: it is computed here from the chunk's session start, its position in
the session and the frame index, under one of these models:

``thesis``
    The published (2026 thesis) model, reproduced exactly: chunk *k* starts at
    ``session_start + k × 1200 s`` and frame *f* is ``f / 25`` s later. Chunks are really ~1201.7 s
    long, so this drifts by about 2 minutes a day and overlaps consecutive chunks by ~2 s. Use it
    only to reproduce published numbers.
``corrected`` (the default)
    Frames are counted continuously through the session from each chunk's real frame count, at a
    rate and offset fitted per session from the tone-evoked flight bursts or the Pi's playback log.

All times are integer nanoseconds of the Pi's local wall clock, as if it were UTC (no time zone,
no daylight saving): ``ns = (time - 1970-01-01 00:00) in ns``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import overload

import numpy as np

from buzzphono.errors import DataError
from buzzphono.io.naming import ChunkName

NS_PER_S = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_S
NS_PER_HOUR = 3600 * NS_PER_S
_EPOCH = datetime(1970, 1, 1)


def to_ns(value: datetime) -> int:
    """Wall-clock time -> integer ns since 1970-01-01 (exact)."""
    d = value - _EPOCH
    return (d.days * 86400 + d.seconds) * NS_PER_S + d.microseconds * 1000


def from_ns(ns: int) -> datetime:
    """Integer ns -> wall-clock time (to the microsecond)."""
    return _EPOCH + timedelta(microseconds=ns // 1000)


@overload
def floor_hour(ns: int) -> int: ...
@overload
def floor_hour(ns: np.ndarray) -> np.ndarray: ...
def floor_hour(ns: np.ndarray | int) -> np.ndarray | int:
    """Start of the clock hour containing each time."""
    return ns - (ns % NS_PER_HOUR)


def legacy_seconds_to_ns(seconds: np.ndarray) -> np.ndarray:
    """Float seconds -> int ns exactly as pandas 2.0 did it for the published analysis.

    ``pd.to_timedelta(x, unit="s")`` split x into whole seconds and a fraction rounded to 9
    decimals, then truncated ``fraction × 1e9``. For whole frame numbers at 25 fps this is exactly
    ``frame × 40 ms``; the general formula is kept so non-integer frames match too.
    """
    x = np.asarray(seconds, dtype=np.float64)
    base = x.astype(np.int64)
    frac = np.round(x - base, 9)
    return base * NS_PER_S + (frac * 1e9).astype(np.int64)


class ThesisClock:
    """The published time base (see module docstring). ``chunk_s`` and ``fps`` are the thesis values."""

    name = "thesis"

    def __init__(self, chunk_s: float = 1200.0, fps: float = 25.0) -> None:
        self.chunk_s = chunk_s
        self.fps = fps

    def chunk_start_ns(self, chunk: ChunkName) -> int:
        start = chunk.session_start()
        if start is None:
            raise DataError(
                f"Chunk {chunk.stem}: '{chunk.date}_{chunk.time}' is not a real date and time.",
                hint="The Pi clock wrote an impossible session stamp; this chunk cannot be placed in time.",
            )
        return to_ns(start) + round(chunk.index * self.chunk_s * NS_PER_S)

    def frame_ns(self, chunk: ChunkName, frames: np.ndarray) -> np.ndarray:
        """Wall-clock ns of each frame (float frame numbers allowed, as in the tracking files)."""
        frames = np.asarray(frames, dtype=np.float64)
        return self.chunk_start_ns(chunk) + legacy_seconds_to_ns(frames / self.fps)
