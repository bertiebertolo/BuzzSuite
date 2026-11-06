"""Load BuzzWatch tracking pickles safely, without importing the tracker.

``pickle.load`` on an arbitrary file can run arbitrary code. BuzzWatch tracking files only ever
contain one tracker class, standard containers, ``datetime``, numpy arrays and (in the final
files) a pandas DataFrame, so this loader allows exactly those and refuses anything else with a
plain-English error. That also makes BuzzPhono independent of the tracker code and its frozen
environment:

* ``buzzwatch_data_analysis.mosquito_obj_tracker.mosquito_obj_tracker`` becomes a plain
  :class:`TrackerObject` holding the same attributes;
* numpy's module paths from numpy 1 (``numpy.core``) and numpy 2 (``numpy._core``) both resolve;
* every ``pandas.*`` object (only the final files' ``population_variables`` table, which BuzzPhono
  never uses) becomes an inert placeholder. That sidesteps pandas pickle incompatibilities between
  versions entirely.
"""

from __future__ import annotations

import codecs
import collections
import datetime as _dt
import importlib
import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from buzzphono.errors import DataError

TRACKER_CLASS = ("buzzwatch_data_analysis.mosquito_obj_tracker", "mosquito_obj_tracker")


class TrackerObject:
    """Stand-in for BuzzWatch's ``mosquito_obj_tracker``: just the unpickled attributes.

    Attributes that matter to BuzzPhono: ``objects`` (``OrderedDict`` of track id -> dict with
    ``coordinates`` and ``time_stamp`` lists, plus ``start``/``state`` in final files), ``type``
    (``"resting"``, ``"moving"`` or ``"mixed"``) and, in final files, ``time_stamp`` (one entry per
    video frame), ``settings`` and ``video_name``.
    """

    objects: collections.OrderedDict[Any, dict[str, Any]]

    def __repr__(self) -> str:
        n = len(getattr(self, "objects", {}) or {})
        return f"<TrackerObject type={getattr(self, 'type', '?')!r} objects={n}>"


class PandasPlaceholder:
    """Inert stand-in for any pandas object inside a tracking pickle (never used by BuzzPhono)."""

    def __new__(cls, *args: Any, **kwargs: Any) -> PandasPlaceholder:
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> PandasPlaceholder:
        return PandasPlaceholder()

    def __repr__(self) -> str:
        return "<pandas object not loaded>"


def _pandas_factory(*args: Any, **kwargs: Any) -> PandasPlaceholder:
    return PandasPlaceholder()


def _numpy_multiarray() -> Any:
    try:
        return importlib.import_module("numpy._core.multiarray")
    except ImportError:  # numpy < 2
        return importlib.import_module("numpy.core.multiarray")


_MULTIARRAY = _numpy_multiarray()
_ALLOWED: dict[tuple[str, str], Any] = {
    TRACKER_CLASS: TrackerObject,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("collections", "defaultdict"): collections.defaultdict,
    ("builtins", "slice"): slice,
    ("builtins", "bytearray"): bytearray,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "complex"): complex,
    ("datetime", "datetime"): _dt.datetime,
    ("datetime", "date"): _dt.date,
    ("datetime", "time"): _dt.time,
    ("datetime", "timedelta"): _dt.timedelta,
    ("datetime", "timezone"): _dt.timezone,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): _MULTIARRAY._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _MULTIARRAY._reconstruct,
    ("numpy.core.multiarray", "scalar"): _MULTIARRAY.scalar,
    ("numpy._core.multiarray", "scalar"): _MULTIARRAY.scalar,
    ("_codecs", "encode"): codecs.encode,  # how protocol-2/3 pickles store raw bytes
}
# Only ever found inside the pandas table of final files (pandas 3 stores text columns via pyarrow).
_PLACEHOLDER_PACKAGES = ("pandas", "pyarrow")


class _Refused(pickle.UnpicklingError):
    """The file asks for an object type BuzzWatch never writes."""


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        allowed = _ALLOWED.get((module, name))
        if allowed is not None:
            return allowed
        if module == "numpy.dtypes" and isinstance(getattr(np.dtypes, name, None), type):
            return getattr(np.dtypes, name)
        if module.split(".")[0] in _PLACEHOLDER_PACKAGES:
            return PandasPlaceholder if name[:1].isupper() else _pandas_factory
        raise _Refused(f"unexpected object {module}.{name}")


def load_tracking_pickle(path: str | Path) -> TrackerObject:
    """Load one BuzzWatch tracking file. Raises :class:`DataError` with a plain-English reason."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DataError(f"Could not read tracking file {path}: {exc.strerror or exc}") from exc
    if not data:
        raise DataError(
            f"Tracking file {path} is empty (0 bytes).",
            hint="It was probably left behind by an interrupted tracking run; re-track that segment.",
        )
    try:
        obj = _RestrictedUnpickler(io.BytesIO(data)).load()
    except _Refused as exc:
        raise DataError(f"{path} is not a BuzzWatch tracking file ({exc}).") from exc
    except (
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        KeyError,
        MemoryError,
    ) as exc:
        raise DataError(
            f"{path} is corrupt or truncated and cannot be read ({type(exc).__name__}: {exc}).",
            hint="Re-track that segment, or leave it out; BuzzPhono reports skipped chunks.",
        ) from exc
    if not isinstance(obj, TrackerObject):
        raise DataError(f"{path} does not contain a BuzzWatch tracker object (found {type(obj).__name__}).")
    if not isinstance(getattr(obj, "objects", None), dict):
        raise DataError(f"{path} has no track list (`objects`); the file is incomplete.")
    return obj
