"""Publication style: one look for every BuzzPhono figure, and deterministic files.

Colours are fixed by role, never by rank: OFF is always blue and ON always orange, the speaker
zone always takes the first colour and controls the next ones in order. Diverging maps (ON − OFF)
run blue -> grey -> red with grey at zero and a symmetric scale. Text is black, gridlines are
hairlines, and files are written so the same data give byte-identical SVG/PDF (no creation dates,
fixed SVG id salt).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OFF = "#2a78d6"  # blue: baseline window
ON = "#eb6834"  # orange: tone window
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SHADE = "#f0efec"  # stimulus windows / light-phase bands
DIVERGING = LinearSegmentedColormap.from_list(
    "buzzphono_div", ["#104281", "#3987e5", SHADE, "#e66767", "#9b2423"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "buzzphono_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
)

RC: Any = {  # matplotlib rcParams (typed loosely: the stubs list every key as a Literal)
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.hashsalt": "buzzphono",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


F = TypeVar("F", bound=Callable[..., Any])


def styled(fn: F) -> F:
    """Run a plotting function inside the BuzzPhono style (every artist it creates is styled)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with plt.rc_context(RC):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def figure(
    width_mm: float = 85.0, height_mm: float = 65.0, ncols: int = 1, nrows: int = 1, **kw: Any
) -> tuple[Any, Any]:
    """A figure at journal column sizes (85 mm single, 175 mm double), in the BuzzPhono style."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(nrows, ncols, figsize=(width_mm / 25.4, height_mm / 25.4), **kw)
    return fig, ax


def save(fig: Any, base: Path, formats: Iterable[str] = ("png", "svg", "pdf"), dpi: int = 300) -> list[Path]:
    """Write ``base.<fmt>`` for each format, deterministically, and close the figure."""
    base.parent.mkdir(parents=True, exist_ok=True)
    out = []
    meta = {
        "png": {"Software": None},
        "svg": {"Date": None, "Creator": None},
        "pdf": {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None},
    }
    with plt.rc_context(RC):
        for fmt in formats:
            path = base.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=meta.get(fmt, {}))
            out.append(path)
    plt.close(fig)
    return out
