"""``run_manifest.json``: everything needed to know exactly how a run's results were made.

Settings (every value, defaults included), every input file's identity (size, modification time
and a hash of its first and last MiB, as the cache records them), the calibration, the BuzzPhono
version and a hash of its source code, the versions of the scientific packages, the platform, the
command line, timestamps, every warning, and a SHA-256 of every output file. Two runs of the same
settings on the same data give the same output hashes (figures and tables are deterministic).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from buzzphono import __version__
from buzzphono.io.cache import CacheIndex, atomic_write_text, cache_dir_for

if TYPE_CHECKING:
    from buzzphono.pipeline import RunResult

MANIFEST = "run_manifest.json"
PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "matplotlib",
    "pydantic",
    "pyyaml",
    "pillow",
    "opencv-python-headless",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_hash() -> str:
    """SHA-256 over BuzzPhono's own source files (identifies the code even without git)."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _inputs(result: RunResult) -> list[dict[str, Any]]:
    out = []
    cache_root = result.config.output.cache_dir
    for e in result.experiments:
        index = CacheIndex.open(cache_dir_for(e.spec.path, cache_root), e.spec.path)
        files = []
        for chunk, tiers in sorted(index.data["chunks"].items()):
            for tier, entry in sorted(tiers.items()):
                src = entry.get("source", {})
                files.append({"chunk": chunk, "tier": tier, **src, "status": entry.get("status")})
        out.append(
            {
                "label": e.label,
                "path": str(e.spec.path),
                "clock": e.clock_message,
                "ingest": e.ingest,
                "n_cycles": len(e.cycles),
                "n_usable_cycles": int(e.cycles["usable"].sum()) if len(e.cycles) else 0,
                "metadata": {k: v for k, v in e.spec.model_dump(mode="json").items() if k != "path"},
                "files": files,
            }
        )
    return out


def outputs(run_dir: Path) -> dict[str, str]:
    """Relative path -> SHA-256 of every file the run wrote (except the manifest and report)."""
    skip = {
        MANIFEST,
        "report.html",
        "report.pdf",
        ".buzzphono_run",
        "status.json",
        "events.jsonl",
        "process.log",
        "settings.yml",
    }
    return {
        p.relative_to(run_dir).as_posix(): sha256_file(p)
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name not in skip
    }


def build_manifest(result: RunResult) -> dict[str, Any]:
    return {
        "schema": "buzzphono-run-manifest",
        "schema_version": 1,
        "buzzphono_version": __version__,
        "source_sha256": source_hash(),
        "packages": package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "command_line": sys.argv,
        "started": result.started,
        "finished": result.finished,
        "settings": json.loads(result.config.model_dump_json()),
        "inputs": _inputs(result),
        "warnings": result.warnings,
        "outputs": outputs(result.run_dir),
    }


def write_manifest(result: RunResult) -> Path:
    path = result.run_dir / MANIFEST
    atomic_write_text(path, json.dumps(build_manifest(result), indent=1, default=str))
    return path
