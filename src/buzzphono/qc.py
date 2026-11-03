"""Data-quality checks. Each returns findings in plain English, with counts, never silently.

The checks guard against the known ways phonotaxis numbers go wrong:

* **time**: impossible or future session stamps, sessions that overlap (a clock reset), missing
  chunks, unexpectedly short chunks in mid-session, ``-attempt`` sessions, sessions that cross a
  daylight-saving change, duplicate files;
* **tracking**: implausible jumps in moving tracks (ID swaps), track fragmentation, the tracker's
  warm-up at the start of each chunk, static objects (dead mosquitoes or specks) that inflate
  zone counts, and crowding in a zone (close pairs, a sign of merged blobs);
* **light**: lights-on/off seen in the reference frames against the schedule (an hour-sized
  clock error shows up here, where the tone-burst fit cannot see it).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

from buzzphono.io.naming import ChunkName
from buzzphono.timebase import NS_PER_S, SessionFrames, from_ns

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Finding:
    severity: Severity
    area: str
    message: str
    count: int | None = None
    details: tuple[str, ...] = field(default_factory=tuple)


# ------------------------------------------------------------------------- time and layout
def _ranges(values: Iterable[int]) -> str:
    v = sorted(values)
    parts, start = [], None
    for i, x in enumerate(v):
        if start is None:
            start = x
        if i == len(v) - 1 or v[i + 1] != x + 1:
            parts.append(f"{start}" if start == x else f"{start}-{x}")
            start = None
    return ", ".join(parts)


def _last_sunday(year: int, month: int) -> datetime:
    d = datetime(year, month + 1, 1) - timedelta(days=1) if month < 12 else datetime(year, 12, 31)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def layout_findings(
    chunks: list[ChunkName], sessions: Mapping[str, SessionFrames], *, now: datetime | None = None
) -> list[Finding]:
    """Checks on session stamps, chunk numbering and frame counts."""
    now = now or datetime.now()
    out: list[Finding] = []
    by_session: dict[str, list[ChunkName]] = {}
    for c in chunks:
        by_session.setdefault(c.session, []).append(c)
    bad = [s for s, cs in by_session.items() if cs[0].session_start() is None]
    if bad:
        out.append(
            Finding(
                "error",
                "time",
                "Session stamps that are not real dates; their chunks cannot be placed "
                "in time and are left out.",
                len(bad),
                tuple(bad),
            )
        )
    for s, cs in by_session.items():
        start = cs[0].session_start()
        if start is None:
            continue
        if start.year < 2015 or start > now + timedelta(days=1):
            out.append(
                Finding(
                    "error",
                    "time",
                    f"Session {s} says it started on {start:%Y-%m-%d %H:%M}, which "
                    "is implausible: the Pi clock was probably not set (no network at boot).",
                )
            )
        if cs[0].attempt is not None:
            out.append(
                Finding(
                    "warning",
                    "time",
                    f"Session {s} has an '-attempt' suffix: the Pi reused a name, "
                    "which happens when its clock goes backwards. Check its times.",
                )
            )
        idx = sorted({c.index for c in cs})
        missing = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx))
        if missing:
            out.append(
                Finding(
                    "warning",
                    "chunks",
                    f"Session {s} is missing chunks {_ranges(missing)} (not tracked or not recorded).",
                    len(missing),
                )
            )
        sf = sessions.get(s)
        if sf is not None:
            short = [k for k, n in sf.nf.items() if n < 0.98 * sf.nf_full and k not in (idx[0], idx[-1])]
            if short:
                out.append(
                    Finding(
                        "warning",
                        "chunks",
                        f"Session {s}: chunks {_ranges(short)} are shorter than a "
                        f"full chunk ({sf.nf_full} frames) although they are not at the session's start or "
                        "end; frames were probably dropped.",
                        len(short),
                    )
                )
            if sf.nf_source != "tracker frame list":
                out.append(
                    Finding(
                        "info",
                        "time",
                        f"Session {s}: frame counts come from the {sf.nf_source}, not "
                        "the tracker's frame list (final_tracking_data missing for some chunks).",
                    )
                )
    # overlapping sessions (the clock went back, or two recordings of one cage were mixed)
    spans = []
    for s, sf in sessions.items():
        if sf.nf:
            end = sf.start_ns + int((sf.first_frame(max(sf.nf)) + sf.nf[max(sf.nf)]) / sf.fps * NS_PER_S)
            spans.append((sf.start_ns + int(sf.first_frame(min(sf.nf)) / sf.fps * NS_PER_S), end, s))
    spans.sort()
    for (_a0, a1, sa), (b0, _b1, sb) in itertools.pairwise(spans):
        if b0 < a1 - 60 * NS_PER_S:
            out.append(
                Finding(
                    "error",
                    "time",
                    f"Sessions {sa} and {sb} overlap by "
                    f"{(a1 - b0) / NS_PER_S / 60:.0f} min: the Pi clock jumped back, or recordings of "
                    "different runs are mixed in one folder.",
                )
            )
    # daylight saving (Europe: last Sunday of March and October)
    for s, (start_ns, end_ns) in {sp[2]: (sp[0], sp[1]) for sp in spans}.items():
        t0, t1 = from_ns(start_ns), from_ns(end_ns)
        for year in range(t0.year, t1.year + 1):
            for change in (_last_sunday(year, 3), _last_sunday(year, 10)):
                if t0 < change + timedelta(hours=1) < t1:
                    out.append(
                        Finding(
                            "warning",
                            "time",
                            f"Session {s} spans the daylight-saving change of "
                            f"{change:%Y-%m-%d}: if the Pi follows summer time, hours after it are one "
                            "hour off on the clock.",
                        )
                    )
    return out


def duplicate_sources(index_chunks: Mapping[str, Mapping[str, Mapping[str, object]]]) -> list[Finding]:
    """Chunks whose source files are byte-identical copies (same size and content hash)."""
    seen: dict[tuple[object, object, str], str] = {}
    dups = []
    for chunk_id, tiers in index_chunks.items():
        for tier, entry in tiers.items():
            src = entry.get("source") if isinstance(entry, Mapping) else None
            if not isinstance(src, Mapping):
                continue
            key = (src.get("size"), src.get("phash"), tier)
            if key in seen:
                dups.append(f"{chunk_id} = {seen[key]} ({tier})")
            else:
                seen[key] = chunk_id
    if not dups:
        return []
    return [
        Finding(
            "error",
            "chunks",
            "Some tracking files are copies of others and would be counted twice; remove the copies.",
            len(dups),
            tuple(dups[:20]),
        )
    ]


# ------------------------------------------------------------------------- tracking
def jump_counts(moving: Mapping[str, np.ndarray], max_step_px: float) -> tuple[int, int, int]:
    """(steps, jumps, spike points) in one chunk's moving tracks.

    A jump is a step longer than ``max_step_px`` per frame (an identity swap or a mislink); a spike
    is a single point whose steps in and out are both jumps (a one-frame excursion)."""
    obj = moving["det_obj"]
    fr = moving["det_frame"]
    xy = moving["det_xy"]
    if obj.size < 2:
        return 0, 0, 0
    same = obj[1:] == obj[:-1]
    dfr = np.maximum(np.diff(fr), 1.0)
    step = np.hypot(*(xy[1:] - xy[:-1]).T) / dfr
    jump = same & (step > max_step_px)
    spike = jump[:-1] & jump[1:] & same[:-1] & same[1:]
    return int(same.sum()), int(jump.sum()), int(spike.sum())


@dataclass(frozen=True)
class Fragmentation:
    tracks: int
    flight_bouts: int
    median_bout_s: float
    share_flight_in_long_bouts: float  # fraction of flight frames in bouts >= 2 s
    starts_per_mosquito_hour: float


def fragmentation(tracks: Mapping[str, np.ndarray], colony: int, fps: float = 25.0) -> Fragmentation:
    """How broken-up the assembled tracks of one chunk are (from the tracks tier)."""
    n_tracks = int(tracks["trk_id"].size)
    nf = int(tracks["nf"]) if np.ndim(tracks["nf"]) == 0 else int(tracks["nf"][()])
    run_len = tracks["run_len"].astype(np.int64)
    state = tracks["run_state"]
    off = tracks["trk_run_off"]
    bouts: list[int] = []
    for t in range(n_tracks):
        lens, st = run_len[off[t] : off[t + 1]], state[off[t] : off[t + 1]]
        cur = 0
        for ln, s in zip(lens, st, strict=True):
            if s == 1:
                cur += int(ln)
            elif cur:
                bouts.append(cur)
                cur = 0
        if cur:
            bouts.append(cur)
    b = np.asarray(bouts, dtype=np.float64)
    starts = int((tracks["trk_start"] > 2).sum())
    hours = nf / fps / 3600.0 if nf else float("nan")
    return Fragmentation(
        tracks=n_tracks,
        flight_bouts=int(b.size),
        median_bout_s=float(np.median(b) / fps) if b.size else float("nan"),
        share_flight_in_long_bouts=float(b[b >= 2 * fps].sum() / b.sum()) if b.size else float("nan"),
        starts_per_mosquito_hour=float(starts / (colony * hours)) if hours and hours > 0 else float("nan"),
    )


def warmup_ratio(n_per_frame: np.ndarray, fps: float = 25.0) -> float:
    """Detections per frame in a chunk's first 2 minutes ÷ those in minutes 5-15 (1 = no warm-up)."""
    a = n_per_frame[: int(120 * fps)]
    b = n_per_frame[int(300 * fps) : int(900 * fps)]
    if a.size == 0 or b.size == 0 or b.mean() == 0:
        return float("nan")
    return float(a.mean() / b.mean())


def static_runs(resting: Mapping[str, np.ndarray], nf: int, fraction: float = 0.95) -> np.ndarray:
    """Positions (x, y) of resting runs that never move for at least ``fraction`` of the chunk."""
    long = resting["run_len"] >= fraction * nf
    return resting["run_xy"][long]


def persistent_static_objects(
    per_chunk: list[np.ndarray], tolerance_px: float = 2.0, min_chunks: int = 6
) -> np.ndarray:
    """Positions static in at least ``min_chunks`` consecutive chunks (dead mosquitoes or specks)."""
    if not per_chunk:
        return np.empty((0, 2))
    tracks: list[tuple[np.ndarray, int, int]] = []  # position, run length, last chunk index
    found = []
    for i, pts in enumerate(per_chunk):
        updated = []
        for pos, n, last in tracks:
            if last != i - 1:
                if n >= min_chunks:
                    found.append(pos)
                continue
            d = np.hypot(*(pts - pos).T) if len(pts) else np.empty(0)
            if d.size and np.min(d) <= tolerance_px:
                updated.append((pos, n + 1, i))
            elif n >= min_chunks:
                found.append(pos)
        known = [u[0] for u in updated]
        for p in pts:
            if not known or np.hypot(*(np.asarray(known) - p).T).min() > tolerance_px:
                updated.append((p, 1, i))
                known.append(p)
        tracks = updated
    found += [pos for pos, n, _ in tracks if n >= min_chunks]
    return np.asarray(found).reshape(-1, 2)


def close_pair_rate(xy_by_frame: Iterable[np.ndarray], max_distance_px: float) -> float:
    """Fraction of detections whose nearest neighbour in the same frame is closer than the limit."""
    n_close = n_all = 0
    for pts in xy_by_frame:
        if len(pts) < 2:
            n_all += len(pts)
            continue
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d, np.inf)
        n_close += int((d.min(axis=1) < max_distance_px).sum())
        n_all += len(pts)
    return n_close / n_all if n_all else float("nan")


# ------------------------------------------------------------------------- light
def image_brightness(paths: Iterable[Path]) -> dict[str, float]:
    """Mean grey level of each reference frame (chunk stem -> 0..255)."""
    from PIL import Image

    out = {}
    for p in paths:
        try:
            with Image.open(p) as im:
                out[p.stem] = float(np.asarray(im.convert("L").reduce(8), dtype=np.float64).mean())
        except OSError:
            continue
    return out


def light_findings(
    brightness_by_time: Mapping[datetime, float], lights_on: str, lights_off: str, tolerance_min: float = 45.0
) -> list[Finding]:
    """Compare the strongest daily brightness steps with the light schedule.

    Reference frames are one per 20-minute chunk, so only errors of about an hour or more show up:
    exactly the size of error a Pi that booted with a wrong clock, or a missed summer-time change,
    would make.
    """
    if len(brightness_by_time) < 6:
        return []
    times = sorted(brightness_by_time)
    vals = np.array([brightness_by_time[t] for t in times])
    if np.ptp(vals) < 10:
        return [
            Finding(
                "info",
                "light",
                "The reference frames show no clear lights-on/off change (IR-only or "
                "constant light), so the clock's hour cannot be checked against the light schedule.",
            )
        ]
    on_h, on_m = map(int, lights_on.split(":"))
    off_h, off_m = map(int, lights_off.split(":"))
    out = []
    steps = np.diff(vals)
    thr = 0.5 * np.ptp(vals)
    for i in map(int, np.flatnonzero(np.abs(steps) > thr)):
        t = times[i] + (times[i + 1] - times[i]) / 2
        target = (
            t.replace(hour=on_h, minute=on_m, second=0, microsecond=0)
            if steps[i] > 0
            else t.replace(hour=off_h % 24, minute=off_m, second=0, microsecond=0)
        )
        diff = (
            min(
                abs((t - target).total_seconds()),
                abs((t - target - timedelta(days=1)).total_seconds()),
                abs((t - target + timedelta(days=1)).total_seconds()),
            )
            / 60.0
        )
        if diff > tolerance_min:
            kind = "on" if steps[i] > 0 else "off"
            out.append(
                Finding(
                    "warning",
                    "light",
                    f"The lights seem to switch {kind} around {t:%Y-%m-%d %H:%M} on "
                    f"the recording clock, {diff:.0f} min from the schedule "
                    f"({lights_on if kind == 'on' else lights_off}). "
                    "If the schedule is right, the Pi clock was off by about "
                    f"{round(diff / 60)} h for this recording.",
                )
            )
    return out


def chunk_times(sessions: Mapping[str, SessionFrames], chunks: Iterable[ChunkName]) -> dict[str, datetime]:
    """Nominal mid-time of each chunk (for comparing reference frames with the light schedule)."""
    out = {}
    for c in chunks:
        sf = sessions.get(c.session)
        n = sf.nf.get(c.index) if sf else None
        if sf is None or n is None:
            continue
        mid = sf.start_ns + int((sf.first_frame(c.index) + n / 2) / sf.fps * NS_PER_S)
        out[c.stem] = from_ns(mid)
    return out


__all__ = [
    "Finding",
    "chunk_times",
    "close_pair_rate",
    "duplicate_sources",
    "fragmentation",
    "image_brightness",
    "jump_counts",
    "layout_findings",
    "light_findings",
    "persistent_static_objects",
    "static_runs",
    "warmup_ratio",
]
