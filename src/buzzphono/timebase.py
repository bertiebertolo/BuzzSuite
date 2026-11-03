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

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SessionFrames:
    """How one recording session's chunks map onto its recording clock.

    rpicam cuts one continuous stream into chunks at the first keyframe after each 1200 s of its
    own timer, so a full chunk has a fixed number of frames (``nf_full``: 30,060 on the Penzance
    Camera Module 3 rigs, 30,000 on the Bangkok v2 camera) and chunk *k* starts ``k × nf_full``
    frames into the session. Only a session's first and last chunks are shorter: the last because
    recording stopped, the first because the camera starts a little after its timer, so a short
    first chunk is aligned to end where the second begins.
    """

    session: str
    start_ns: int  # the Pi-clock session start written in the file names
    nf_full: int
    nf: dict[int, int]  # frames per chunk index (known chunks only)
    nf_source: str  # "tracker frame list", "last detection" or "nominal"
    fps: float = 25.0

    def first_frame(self, k: int) -> int:
        """Recording-clock frame number of frame 0 of chunk ``k``.

        A short first chunk is end-aligned only when its frame count is exact (the tracker's frame
        list): a count estimated from the last detection is short whenever the chunk's last frames
        happen to have no detection, which says nothing about where the chunk started.
        """
        start = k * self.nf_full
        if (
            k == 0
            and self.nf_source == "tracker frame list"
            and len(self.nf) > 1
            and self.nf.get(0, self.nf_full) < self.nf_full
        ):
            start += self.nf_full - self.nf[0]
        return start

    def recording_seconds(self, k: int, frames: np.ndarray) -> np.ndarray:
        return (self.first_frame(k) + np.asarray(frames, dtype=np.float64)) / self.fps


@dataclass(frozen=True)
class ClockModel:
    """The fitted relation between recording clock and wall clock (see :mod:`buzzphono.stimulus`).

    Wall time of recording second ``t`` of a session = start + t − offset − drift × t / 86400, with
    one ``offset`` per session (seconds) and one ``drift`` per experiment (seconds per day).
    ``source`` says, per session, where its offset came from.
    """

    offset_s: dict[str, float]
    drift_s_per_day: float
    source: dict[str, str]
    fitted: bool = False
    message: str = ""
    n_bursts: int = 0
    n_kept: int = 0
    residual_sd_s: float = float("nan")

    @classmethod
    def nominal(cls, sessions: list[str], message: str = "nominal clock (no calibration)") -> ClockModel:
        return cls(dict.fromkeys(sessions, 0.0), 0.0, dict.fromkeys(sessions, "nominal"), False, message)


class CorrectedClock:
    """The default time base: real frame counts plus a fitted per-session clock (see module docstring)."""

    name = "corrected"

    def __init__(self, sessions: dict[str, SessionFrames], model: ClockModel) -> None:
        self.sessions = sessions
        self.model = model

    def frame_ns(self, chunk: ChunkName, frames: np.ndarray) -> np.ndarray:
        sf = self.sessions.get(chunk.session)
        if sf is None:
            raise DataError(f"Chunk {chunk.stem}: its session {chunk.session} is not in this clock.")
        t = sf.recording_seconds(chunk.index, frames)
        a = self.model.offset_s.get(chunk.session, 0.0)
        real = t - a - self.model.drift_s_per_day * t / 86400.0
        return sf.start_ns + np.round(real * NS_PER_S).astype(np.int64)


def frames_per_chunk(cache_entries: dict[str, dict[str, dict[str, object]]]) -> dict[str, tuple[int, str]]:
    """Each chunk's frame count and where it came from, from the cache index's extraction summaries.

    Best: the tracker's per-frame list in the final file (exact). Otherwise the last frame with any
    detection + 1 (resting detections appear in almost every frame, so this is usually exact too).
    """
    out: dict[str, tuple[int, str]] = {}
    for chunk_id, tiers in cache_entries.items():
        tracks = tiers.get("tracks") or {}
        nf = tracks.get("nf") if tracks.get("status") == "ok" else None
        if isinstance(nf, int) and nf > 0:
            out[chunk_id] = (nf, "tracker frame list")
            continue
        best = -1.0
        for tier in ("resting", "moving"):
            entry = tiers.get(tier) or {}
            mf = entry.get("max_frame") if entry.get("status") == "ok" else None
            if isinstance(mf, int | float) and mf > best:
                best = float(mf)
        if best >= 0:
            out[chunk_id] = (int(best) + 1, "last detection")
    return out


def build_sessions(
    chunks: list[ChunkName], nf: dict[str, tuple[int, str]], fps: float = 25.0, default_nf: int | None = None
) -> dict[str, SessionFrames]:
    """One :class:`SessionFrames` per recording session (``nf`` from :func:`frames_per_chunk`)."""
    by_session: dict[str, list[ChunkName]] = {}
    for c in chunks:
        by_session.setdefault(c.session, []).append(c)
    exact = [n for n, src in nf.values() if src == "tracker frame list"]
    experiment_full = max(exact) if exact else (max((n for n, _ in nf.values()), default=0) or default_nf)
    out: dict[str, SessionFrames] = {}
    for session, members in by_session.items():
        start = members[0].session_start()
        if start is None:
            continue  # an impossible session stamp: reported by QC, cannot be placed in time
        counts = {c.index: nf[c.stem][0] for c in members if c.stem in nf}
        sources = {nf[c.stem][1] for c in members if c.stem in nf}
        exact_here = [
            nf[c.stem][0] for c in members if c.stem in nf and nf[c.stem][1] == "tracker frame list"
        ]
        full = max(exact_here) if len(exact_here) >= 2 else experiment_full
        if not full:
            continue
        source = (
            "tracker frame list"
            if sources == {"tracker frame list"}
            else ("last detection" if sources else "nominal")
        )
        out[session] = SessionFrames(session, to_ns(start), int(full), counts, source, fps)
    return out


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
