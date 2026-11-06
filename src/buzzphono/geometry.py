"""Polygons in image pixels: membership tests, areas, centroids and distances.

Zone membership uses ``matplotlib.path.Path.contains_points`` with no radius, exactly as the
published analysis did. This matters: zone vertices are whole pixels and about a quarter of
resting detections sit on whole-pixel coordinates, so points exactly on an edge do occur, and
other point-in-polygon rules classify some of them differently.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath


def as_polygon(vertices: object) -> np.ndarray:
    """(n, 2) float array of polygon vertices (n >= 3), not closed."""
    poly = np.asarray(vertices, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("a polygon needs at least 3 (x, y) vertices")
    if not np.isfinite(poly).all():
        raise ValueError("polygon vertices must be finite numbers")
    if poly.shape[0] > 3 and np.array_equal(poly[0], poly[-1]):
        poly = poly[:-1]
    return poly


def contains(polygon: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Boolean mask: which points lie inside the polygon (published rule)."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(MplPath(polygon).contains_points(pts), dtype=bool)


def area(polygon: np.ndarray) -> float:
    """Absolute area (shoelace formula), in square pixels."""
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def centroid(polygon: np.ndarray) -> np.ndarray:
    """Area-weighted centroid; the vertex mean for a degenerate (zero-area) polygon."""
    x, y = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return polygon.mean(axis=0)
    return np.array([((x + xn) * cross).sum() / (6 * a), ((y + yn) * cross).sum() / (6 * a)])


def _boundary_distance(polygon: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest point on any polygon edge."""
    a = polygon
    ab = np.roll(polygon, -1, axis=0) - a  # (m, 2) edge vectors
    ap = pts[:, None, :] - a[None, :, :]  # (n, m, 2)
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip((ap * ab[None]).sum(axis=2) / denom[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - closest) ** 2).sum(axis=2)).min(axis=1)


def distance_to_polygon(polygon: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the polygon's boundary, 0 for points inside it."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    d = _boundary_distance(polygon, pts)
    d[contains(polygon, pts)] = 0.0
    return d


def signed_distance(polygon: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Distance to the boundary, negative inside: ``<= m`` selects the zone grown by m pixels."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    d = _boundary_distance(polygon, pts)
    return np.where(contains(polygon, pts), -d, d)
