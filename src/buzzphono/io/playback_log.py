"""Read the Raspberry Pi's sound-playback log (``sound_schedule.py`` / systemd ``buzzwatch-sound``).

The Pi's sound service writes lines like::

    2026-10-10 00:40:00 [INFO] Window open - playing 510 Hz tone at 80% volume
    2026-10-10 00:50:00 [INFO] Window closed - playback stopped
    2026-10-10 01:40:00 [ERROR] Could not start playback: ... (retry in 10s)

either to ``~/pirecorder/logs/sound.log`` or to the system journal
(``journalctl -u buzzwatch-sound``, which adds its own prefix to every line). Both are read here; the
times are the Pi's local wall clock, the same clock that named the video files.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from buzzphono.errors import DataError

_LINE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[.,]\d+)?\s*\[(?P<level>[A-Z]+)\]\s*(?P<msg>.*)$"
)
_EVENTS = (
    ("open", re.compile(r"^Window open\b")),
    ("close", re.compile(r"^Window closed\b")),
    ("failed_start", re.compile(r"^Could not start playback")),
    ("interrupted", re.compile(r"^(Playback stopped unexpectedly|Could not continue playback)")),
    ("service_start", re.compile(r"^Sound schedule running")),
    ("service_stop", re.compile(r"^(Sound schedule stopped|Received SIG\w+)")),
    ("disabled", re.compile(r"^Sound schedule disabled")),
    ("device_lost", re.compile(r"^Audio device disappeared")),
)
_STIMULUS = re.compile(r"^Window open - playing (?P<what>.+)$")


@dataclass
class PlaybackLog:
    """Events and playback intervals read from one or more log files."""

    events: pd.DataFrame  # time, kind, level, message, source
    intervals: (
        pd.DataFrame
    )  # start, end, end_reason ("closed", "interrupted", "service stopped", "log ended")
    stimuli: list[str] = field(default_factory=list)  # distinct "playing ..." descriptions
    sources: list[str] = field(default_factory=list)

    @property
    def span(self) -> tuple[datetime, datetime] | None:
        if self.events.empty:
            return None
        return self.events["time"].min().to_pydatetime(), self.events["time"].max().to_pydatetime()


def _read_texts(path: Path) -> Iterable[tuple[str, str]]:
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith((".log", ".txt")):
                        yield f"{path.name}:{name}", zf.read(name).decode("utf-8", errors="replace")
        except zipfile.BadZipFile as exc:
            raise DataError(f"{path} is not a readable zip file ({exc}).") from exc
        return
    try:
        yield path.name, path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DataError(f"Could not read the playback log {path}: {exc.strerror or exc}") from exc


def parse_lines(lines: Iterable[str], source: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in lines:
        m = _LINE.search(line.rstrip("\r\n"))
        if m is None:
            continue
        msg = m.group("msg").strip()
        kind = next((k for k, pat in _EVENTS if pat.search(msg)), None)
        if kind is None:
            continue
        try:
            t = datetime.fromisoformat(m.group("ts").replace("T", " "))
        except ValueError:
            continue
        rows.append({"time": t, "kind": kind, "level": m.group("level"), "message": msg, "source": source})
    return rows


def read_playback_log(paths: Path | Iterable[Path]) -> PlaybackLog:
    """Read ``sound.log`` files, journal exports or a zip made by tools/pi_collect_logs.py."""
    paths = [paths] if isinstance(paths, Path) else list(paths)
    rows: list[dict[str, object]] = []
    sources = []
    for path in paths:
        for name, text in _read_texts(Path(path)):
            rows += parse_lines(io.StringIO(text), name)
            sources.append(name)
    events = pd.DataFrame(rows, columns=["time", "kind", "level", "message", "source"])
    if events.empty:
        raise DataError(
            f"No playback events found in {', '.join(map(str, paths))}.",
            hint="Expected lines like '2026-10-10 00:40:00 [INFO] Window open - playing ...' from the Pi's "
            "buzzwatch-sound service (sound.log or `journalctl -u buzzwatch-sound`).",
        )
    events = events.drop_duplicates(subset=["time", "kind", "message"]).sort_values("time", kind="stable")
    events = events.reset_index(drop=True)
    intervals = []
    open_at: datetime | None = None
    for row in events.itertuples(index=False):
        t = row.time
        if row.kind == "open":
            if open_at is not None:
                intervals.append((open_at, t, "reopened without a close"))
            open_at = t
        elif row.kind in ("close", "interrupted", "service_stop", "device_lost") and open_at is not None:
            reason = {
                "close": "closed",
                "interrupted": "interrupted",
                "service_stop": "service stopped",
                "device_lost": "audio device lost",
            }[row.kind]
            intervals.append((open_at, t, reason))
            open_at = None
    if open_at is not None:
        intervals.append((open_at, events["time"].iloc[-1], "log ended"))
    stimuli = sorted({m.group("what") for m in map(_STIMULUS.match, events["message"]) if m})
    return PlaybackLog(
        events, pd.DataFrame(intervals, columns=["start", "end", "end_reason"]), stimuli, sources
    )


def window_status(log: PlaybackLog, window_start: datetime, window_end: datetime) -> tuple[str, float]:
    """Whether the tone played during [start, end): ("played" | "partly played" | "not played" |
    "failed" | "not covered by the log", fraction of the window covered by playback)."""
    span = log.span
    length = (window_end - window_start).total_seconds()
    if span is None or window_start < span[0] or window_end > span[1] or length <= 0:
        return "not covered by the log", float("nan")
    covered = 0.0
    for start, end in zip(log.intervals["start"], log.intervals["end"], strict=True):
        lo, hi = max(start, window_start), min(end, window_end)
        if hi > lo:
            covered += (hi - lo).total_seconds()
    frac = min(1.0, covered / length)
    if frac >= 0.9:
        return "played", frac
    near = log.events[
        (log.events["time"] >= window_start - pd.Timedelta(minutes=1)) & (log.events["time"] < window_end)
    ]
    if frac == 0 and (near["kind"] == "failed_start").any():
        return "failed", frac
    return ("partly played" if frac > 0 else "not played"), frac
