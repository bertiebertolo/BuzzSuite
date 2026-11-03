"""Find the tracking files of an experiment and check that the folder looks right.

An experiment folder is what BuzzWatch writes under ``Analysis/{Assay}/{Location}/{Experiment}``:
``tracking_resting/``, ``tracking_moving/`` and ``final_tracking_data/`` (one file per chunk each),
plus ``custom_zones.json``, ``buzzwatch_track_settings.yml`` and reference images. Discovery only
lists files; it never opens a pickle.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from buzzphono.errors import DataError
from buzzphono.io.naming import TRACKING_FOLDERS, ChunkName, TrackingKind, parse_filename, stem_from_filename

KINDS: tuple[TrackingKind, ...] = ("resting", "moving", "tracks")


@dataclass
class ChunkFiles:
    """One recording chunk and its tracking files (a kind is missing if BuzzWatch did not write it)."""

    name: ChunkName
    files: dict[TrackingKind, Path] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.name.stem


@dataclass
class ExperimentLayout:
    """What is on disk for one experiment."""

    root: Path
    chunks: list[ChunkFiles]
    folders_present: dict[TrackingKind, bool]
    unparsable: list[Path] = field(default_factory=list)
    empty_files: list[Path] = field(default_factory=list)
    metadata_files_skipped: int = 0

    @property
    def name(self) -> str:
        return self.root.name

    def missing(self, kind: TrackingKind) -> list[str]:
        """Chunk ids that have some tracking file but not this kind (only if the folder exists)."""
        if not self.folders_present.get(kind):
            return []
        return [c.chunk_id for c in self.chunks if kind not in c.files]

    def sessions(self) -> dict[str, list[ChunkFiles]]:
        out: dict[str, list[ChunkFiles]] = defaultdict(list)
        for c in self.chunks:
            out[c.name.session].append(c)
        return dict(out)


def _sort_key(c: ChunkFiles) -> tuple[str, str, int, int]:
    n = c.name
    return (n.date + n.time, n.label, n.attempt if n.attempt is not None else -1, n.index)


def discover(experiment_dir: str | Path) -> ExperimentLayout:
    """List the experiment's chunks. Raises :class:`DataError` if it is not an experiment folder."""
    root = Path(experiment_dir)
    if not root.is_dir():
        raise DataError(
            f"Experiment folder not found: {root}",
            hint="Check the path; on Windows a data drive letter can change when a drive is re-plugged.",
        )
    present = {kind: (root / TRACKING_FOLDERS[kind]).is_dir() for kind in KINDS}
    if not any(present.values()):
        raise DataError(
            f"{root} has none of tracking_resting/, tracking_moving/ or final_tracking_data/.",
            hint="Choose a folder under Buzzwatch/Analysis/... that BuzzWatch has tracked.",
        )
    by_stem: dict[str, ChunkFiles] = {}
    unparsable: list[Path] = []
    empty: list[Path] = []
    skipped = 0
    for kind in KINDS:
        folder = root / TRACKING_FOLDERS[kind]
        if not present[kind]:
            continue
        for path in sorted(folder.iterdir()):
            if path.name.startswith("._"):
                skipped += 1
                continue
            if not path.is_file():
                continue
            stem = stem_from_filename(kind, path.name)
            if stem is None:
                continue  # some other file (a plot, a log): not ours to judge
            name = parse_filename(kind, path.name)
            if name is None:
                unparsable.append(path)
                continue
            if path.stat().st_size == 0:
                empty.append(path)
            entry = by_stem.setdefault(name.stem, ChunkFiles(name))
            entry.files[kind] = path
    chunks = sorted(by_stem.values(), key=_sort_key)
    return ExperimentLayout(
        root=root,
        chunks=chunks,
        folders_present=present,
        unparsable=unparsable,
        empty_files=empty,
        metadata_files_skipped=skipped,
    )
