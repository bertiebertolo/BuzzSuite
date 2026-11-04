"""The job protocol: how a run reports progress and is cancelled, whoever started it.

A run's folder is the single source of truth, so the command line, the app and the BuzzSuite
launcher can all watch and stop the same run:

* ``status.json`` – state (``running`` / ``done`` / ``failed`` / ``cancelled``), progress 0-1, the
  current message, the process id, start and last-update times, and on failure the error;
* ``events.jsonl`` – one line per progress message (time, progress, message);
* ``CANCEL`` – created by whoever wants the run to stop; the run checks for it between steps,
  stops, deletes its partial outputs and ends with state ``cancelled``.

Two runs must not build the same experiment's cache at once: :func:`experiment_lock` holds a lock
file (process id + heartbeat) in the cache folder; a lock whose process is gone or whose
heartbeat is more than two minutes old is taken over.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from buzzphono.errors import BuzzPhonoError, ConfigError, DataError
from buzzphono.io.cache import atomic_write_text

STATUS = "status.json"
EVENTS = "events.jsonl"
CANCEL = "CANCEL"
LOCK = ".buzzphono.lock"
STALE_S = 120.0


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Job:
    """Records one run's progress in its folder and answers "has someone asked me to stop?"."""

    def __init__(self, run_dir: Path, min_interval_s: float = 0.5) -> None:
        self.run_dir = run_dir
        self.min_interval_s = min_interval_s
        self._last = 0.0
        claim_run_dir(run_dir)
        self.status: dict[str, Any] = {
            "state": "running",
            "progress": 0.0,
            "message": "Starting",
            "pid": os.getpid(),
            "started": _now(),
            "updated": _now(),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / CANCEL).unlink(missing_ok=True)
        self._write()

    def _write(self) -> None:
        self.status["updated"] = _now()
        atomic_write_text(self.run_dir / STATUS, json.dumps(self.status, indent=1))

    def progress(self, fraction: float, message: str) -> None:
        self.status["progress"] = round(float(min(max(fraction, 0.0), 1.0)), 4)
        self.status["message"] = message
        now = time.monotonic()
        with (self.run_dir / EVENTS).open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"time": _now(), "progress": self.status["progress"], "message": message}) + "\n"
            )
        if now - self._last >= self.min_interval_s or fraction >= 1.0:
            self._last = now
            self._write()

    def cancelled(self) -> bool:
        return (self.run_dir / CANCEL).exists()

    def finish(self, state: str, error: BaseException | None = None, exit_code: int = 0) -> None:
        self.status["state"] = state
        self.status["exit_code"] = exit_code
        if state == "done":
            self.status["progress"] = 1.0
            self.status["message"] = "Done"
        if error is not None:
            self.status["error"] = str(error)
            if isinstance(error, BuzzPhonoError):
                self.status["hint"] = error.hint
        self._write()


def request_cancel(run_dir: Path) -> None:
    """Ask a running job to stop (it notices within a few seconds). Only in a run folder."""
    if not is_run_dir(run_dir):
        raise DataError(f"{run_dir} is not a BuzzPhono run folder; nothing to cancel.")
    (run_dir / CANCEL).write_text(_now(), encoding="utf-8")


def read_status(run_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((run_dir / STATUS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# Everything a run writes into its folder. Clean-up deletes only these, never anything else.
RUN_OUTPUTS = ("experiments", "pooled", "report.html", "report.pdf", "run_manifest.json")
MARKER = ".buzzphono_run"  # written when BuzzPhono creates a run folder; required before any clean-up
EXPERIMENT_MARKERS = ("tracking_resting", "tracking_moving", "final_tracking_data", "individual_images")


def is_run_dir(run_dir: Path) -> bool:
    """True only for a folder that BuzzPhono itself created for a run (it holds the marker file)."""
    return (run_dir / MARKER).is_file()


def check_run_dir(run_dir: Path) -> None:
    """Refuse a results folder unless it is new, empty, or a run folder BuzzPhono created and has not
    finished. Names alone prove nothing: only BuzzPhono's own marker file makes a folder a run's."""
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise ConfigError(f"The results folder {run_dir} is a file.")
    if any((run_dir / m).exists() for m in EXPERIMENT_MARKERS):
        raise ConfigError(
            f"{run_dir} is an experiment folder; results cannot go there.",
            hint="Leave out --run-dir (results then go to <experiment>/plots/buzzphono/) "
            "or choose an empty folder.",
        )
    if not any(run_dir.iterdir()):
        return
    if not is_run_dir(run_dir):
        raise ConfigError(
            f"The results folder {run_dir} already holds files BuzzPhono did not create.",
            hint="Choose a new or empty folder for the results.",
        )
    status = read_status(run_dir) or {}
    if (run_dir / "run_manifest.json").exists() or status.get("state") in ("done", "failed", "cancelled"):
        raise ConfigError(
            f"{run_dir} already holds a finished run.",
            hint="Choose a new folder; earlier results are never overwritten or deleted.",
        )


def claim_run_dir(run_dir: Path) -> None:
    """Check the folder (see check_run_dir), create it and mark it as this run's own."""
    check_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / MARKER
    if not marker.exists():
        marker.write_text(
            f"BuzzPhono run folder, created {_now()} by process {os.getpid()}\n", encoding="utf-8"
        )


def remove_partial_run(run_dir: Path) -> None:
    """Delete a cancelled or failed run's own outputs (see RUN_OUTPUTS), keeping status.json and
    events.jsonl as the record. Only in a folder BuzzPhono created (marker file); nothing else in the
    folder is ever touched."""
    if not run_dir.is_dir() or not is_run_dir(run_dir):
        return
    for name in RUN_OUTPUTS:
        p = run_dir / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 (windll exists only on Windows)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_holder(cache_dir: Path) -> dict[str, Any] | None:
    """The live holder of the experiment lock, or None if it is free (or stale)."""
    try:
        info = json.loads((cache_dir / LOCK).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if info.get("pid") == os.getpid():
        return None
    if not pid_alive(int(info.get("pid", -1))) or time.time() - float(info.get("heartbeat", 0)) > STALE_S:
        return None
    return info


@contextlib.contextmanager
def experiment_lock(cache_dir: Path) -> Iterator[None]:
    """Hold the experiment's cache lock for the duration (refreshing its heartbeat)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    holder = lock_holder(cache_dir)
    if holder is not None:
        raise DataError(
            f"Another BuzzPhono run (process {holder.get('pid')}, started {holder.get('started')}) is "
            f"working on this experiment's cache ({cache_dir}).",
            hint="Wait for it to finish, or cancel it, then try again.",
        )
    path = cache_dir / LOCK
    info = {"pid": os.getpid(), "started": _now(), "heartbeat": time.time()}
    atomic_write_text(path, json.dumps(info))
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(30.0):
            info["heartbeat"] = time.time()
            with contextlib.suppress(OSError):
                atomic_write_text(path, json.dumps(info))

    t = threading.Thread(target=beat, name="buzzphono-lock-heartbeat", daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        with contextlib.suppress(OSError):
            if json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid():
                path.unlink()
