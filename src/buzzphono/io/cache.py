"""The canonical cache: each tracking pickle converted once into compact numpy arrays.

Layout (schema v1)::

    <cache_root>/v1/<experiment key>/
        index.json          what was ingested from which source file, and how it went
        resting/<chunk>.npz resting detections, run-length encoded (see buzzphono.io.ingest)
        moving/<chunk>.npz  moving detections
        tracks/<chunk>.npz  assembled tracks with per-frame state, run-length encoded

Times are never stored: the cache holds frame indices only, and :mod:`buzzphono.timebase` turns
them into clock times. Changing the clock or the calibration therefore never invalidates it. A
chunk's arrays are rebuilt only when its source file changes (size, modification time, or a hash
of its first and last MiB) or the extractor version changes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from buzzphono.errors import DataError

SCHEMA = "buzzphono-cache"
SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "1"  # index layout; each tier also has its own version below
# Bumped when a tier's arrays change; only that tier is rebuilt. moving 2: per-track start/end types.
TIER_VERSIONS = {"resting": "1", "moving": "2", "tracks": "1"}
_HASH_SPAN = 1 << 20  # 1 MiB from each end of a source file


def experiment_key(experiment_dir: Path) -> str:
    """A short, stable folder name for an experiment inside a shared cache folder."""
    resolved = str(experiment_dir.resolve()).replace("\\", "/").lower()
    digest = hashlib.blake2b(resolved.encode("utf-8"), digest_size=5).hexdigest()
    return f"{experiment_dir.name}-{digest}"


def cache_dir_for(experiment_dir: Path, cache_root: Path | None = None) -> Path:
    """Where an experiment's cache lives: by default inside it, under plots/buzzphono/cache."""
    root = cache_root if cache_root is not None else experiment_dir / "plots" / "buzzphono" / "cache"
    return root / f"v{SCHEMA_VERSION}" / experiment_key(experiment_dir)


def source_identity(path: Path) -> dict[str, Any]:
    """Cheap identity of a source file: name, size, modification time and a partial content hash."""
    st = path.stat()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(st.st_size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(_HASH_SPAN))
        if st.st_size > 2 * _HASH_SPAN:
            fh.seek(-_HASH_SPAN, os.SEEK_END)
            h.update(fh.read(_HASH_SPAN))
    return {"name": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "phash": h.hexdigest()}


def _replace_with_retry(src: Path, dst: Path, attempts: int = 12) -> None:
    """``Path.replace`` that survives the short locks virus scanners and indexers take on Windows."""
    delay = 0.05
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


@contextlib.contextmanager
def _atomic_target(path: Path, suffix: str) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".partial-", suffix=suffix)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        _replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a compressed .npz so that readers only ever see a complete file."""
    named: dict[str, Any] = dict(arrays)
    with _atomic_target(path, ".npz") as tmp, tmp.open("wb") as fh:
        np.savez_compressed(fh, **named)


def atomic_write_text(path: Path, text: str) -> None:
    with _atomic_target(path, ".tmp") as tmp, tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read a cache file completely and close it (an open .npz blocks replacing it on Windows)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError) as exc:
        raise DataError(
            f"Cache file {path} is unreadable ({exc}).",
            hint="Delete it (or run `buzzphono cache clean`); it will be rebuilt.",
        ) from exc


class CacheIndex:
    """index.json: per chunk and tier, the source identity and the extraction summary."""

    def __init__(self, cache_dir: Path, experiment_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.path = cache_dir / "index.json"
        self.data: dict[str, Any] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "experiment": str(experiment_dir),
            "chunks": {},
        }

    @classmethod
    def open(cls, cache_dir: Path, experiment_dir: Path) -> CacheIndex:
        index = cls(cache_dir, experiment_dir)
        if index.path.is_file():
            try:
                data = json.loads(index.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None  # unreadable index: start again; the tier files are re-validated
            if (
                isinstance(data, dict)
                and data.get("schema") == SCHEMA
                and data.get("schema_version") == SCHEMA_VERSION
                and data.get("extractor_version") == EXTRACTOR_VERSION
                and isinstance(data.get("chunks"), dict)
            ):
                index.data["chunks"] = data["chunks"]
        index.remove_stale_partials()
        return index

    def remove_stale_partials(self, older_than_s: float = 3600.0) -> int:
        """Delete temporary files left by a process that was killed mid-write (an hour old or more,
        so a write in progress elsewhere is never touched)."""
        if not self.cache_dir.is_dir():
            return 0
        cutoff = time.time() - older_than_s
        removed = 0
        for p in self.cache_dir.rglob(".partial-*"):
            with contextlib.suppress(OSError):
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
        return removed

    def tier_path(self, tier: str, chunk_id: str) -> Path:
        return self.cache_dir / tier / f"{chunk_id}.npz"

    def entry(self, chunk_id: str, tier: str) -> dict[str, Any] | None:
        entry = self.data["chunks"].get(chunk_id, {}).get(tier)
        return entry if isinstance(entry, dict) else None

    def is_current(self, chunk_id: str, tier: str, identity: dict[str, Any]) -> bool:
        entry = self.entry(chunk_id, tier)
        if entry is None or entry.get("source") != identity or entry.get("status") != "ok":
            return False
        if entry.get("extractor") != TIER_VERSIONS.get(tier, EXTRACTOR_VERSION):
            return False
        return self.tier_path(tier, chunk_id).is_file()

    def set_entry(self, chunk_id: str, tier: str, entry: dict[str, Any]) -> None:
        self.data["chunks"].setdefault(chunk_id, {})[tier] = entry

    def save(self) -> None:
        atomic_write_text(self.path, json.dumps(self.data, indent=1, sort_keys=True))
