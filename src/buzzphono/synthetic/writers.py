"""Write files in BuzzWatch's own formats (tracking pickles, zone files) from synthetic data.

Tracking pickles are written exactly as BuzzWatch writes them: an instance of
``buzzwatch_data_analysis.mosquito_obj_tracker.mosquito_obj_tracker`` with ``objects``,
``number_objects`` and ``type`` (and, for final files, the per-frame ``time_stamp`` list,
``settings`` and ``video_name``). BuzzPhono does not ship the tracker, so a stand-in class is
registered under the tracker's module name only while pickling.
"""

from __future__ import annotations

import contextlib
import json
import pickle
import sys
import threading
import types
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from buzzphono.io.naming import TRACKING_FOLDERS, ChunkName, TrackingKind

_MODULE = "buzzwatch_data_analysis.mosquito_obj_tracker"
_LOCK = threading.Lock()


@contextlib.contextmanager
def _tracker_class() -> Iterator[type]:
    with _LOCK:
        names = ("buzzwatch_data_analysis", _MODULE)
        saved = {n: sys.modules.get(n) for n in names}
        pkg = types.ModuleType("buzzwatch_data_analysis")
        pkg.__path__ = []
        mod = types.ModuleType(_MODULE)
        cls = type("mosquito_obj_tracker", (), {"__module__": _MODULE})
        mod.mosquito_obj_tracker = cls  # type: ignore[attr-defined]
        sys.modules["buzzwatch_data_analysis"] = pkg
        sys.modules[_MODULE] = mod
        try:
            yield cls
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m


def tracker_pickle_bytes(objects: Mapping[Any, dict[str, Any]], kind: str, **extra: Any) -> bytes:
    """Pickle bytes of a tracker object (``kind``: "resting", "moving" or "mixed")."""
    with _tracker_class() as cls:
        obj: Any = object.__new__(cls)
        obj.__dict__.update(objects=OrderedDict(objects), number_objects=len(objects), type=kind)
        obj.__dict__.update(extra)
        return pickle.dumps(obj, protocol=4)


def write_tracking_file(
    experiment_dir: Path,
    kind: TrackingKind,
    chunk: ChunkName,
    objects: Mapping[Any, dict[str, Any]],
    **extra: Any,
) -> Path:
    """Write one tracking file where BuzzWatch would (``tracks`` = the final assembled file)."""
    folder = experiment_dir / TRACKING_FOLDERS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / chunk.filename(kind)
    tracker_kind = {"resting": "resting", "moving": "moving", "tracks": "mixed"}[kind]
    path.write_bytes(tracker_pickle_bytes(objects, tracker_kind, **extra))
    return path


def frame_time_stamps(chunk: ChunkName, n_frames: int, fps: float = 25.0) -> list[datetime]:
    """The per-frame datetimes BuzzWatch stores in final files (session start + index × 1200 s)."""
    start = chunk.session_start() or datetime(2000, 1, 1)
    t0 = start + timedelta(seconds=chunk.index * 1200)
    return [t0 + timedelta(milliseconds=int((1000.0 / fps) * i)) for i in range(n_frames)]


def write_zones(path: Path, zones: Mapping[str, Any]) -> Path:
    """Write a ``custom_zones.json`` (``{"speaker": {"polygon": [[x, y], ...]}, ...}``)."""
    data = {name: {"polygon": [[float(x), float(y)] for x, y in poly]} for name, poly in zones.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
