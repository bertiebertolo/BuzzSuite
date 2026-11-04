"""Cage and zone geometry for the flight analyses: distance maps, wall sectors and zone roles.

Four zones are needed, one per cage wall: the speaker and three others. Each wall is assigned to
the zone nearest its midpoint, and the zones get roles by position: the speaker, the wall opposite
it ("right" in the original Aedes rigs), and the top and bottom walls (image orientation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath

from buzzphono.errors import CalibrationError

REF_WIDTH = 780.0
DS = 2  # distance-map downsample
ROLES = ("speaker", "right", "top", "bottom")
ROLE_LABELS = {"speaker": "Speaker", "right": "Opposite wall", "top": "Top wall", "bottom": "Bottom wall"}


@dataclass
class FlightGeometry:
    cage: np.ndarray  # (4, 2) TL TR BR BL
    names: list[str]  # speaker first, then the three other zones in file order
    zones: dict[str, np.ndarray]
    width: float
    s: float
    centre: np.ndarray
    bbox: tuple[float, float, float, float]
    dms: list[np.ndarray]  # per zone: distance (px) to the polygon on a DS-downsampled grid
    cents: np.ndarray  # (4, 2) zone pixel centroids
    paths: list[MplPath]
    gw: int
    gh: int
    rel: np.ndarray
    sign: float
    cang0: float
    sec2zone: np.ndarray
    rpx: list[float]  # neighbourhood radii, px

    def distance(self, zone_index: int, xy: np.ndarray) -> np.ndarray:
        xi = np.clip((xy[:, 0] / DS).astype(np.int64), 0, self.gw - 1)
        yi = np.clip((xy[:, 1] / DS).astype(np.int64), 0, self.gh - 1)
        return self.dms[zone_index][yi, xi]


def role_label(zone: str) -> str:
    if zone in ROLE_LABELS:
        return ROLE_LABELS[zone]
    if zone == "speaker - mean(controls)":
        return "Speaker - mean of the 3 other walls"
    if zone.startswith("speaker - "):
        rest = zone[len("speaker - ") :]
        return "Speaker - " + ROLE_LABELS.get(rest, rest)
    if zone == "all":
        return "whole cage"
    return zone


def build_geometry(cage: Any, zones: dict[str, Any], r_fracs: tuple[float, ...]) -> FlightGeometry:
    """``zones``: speaker first (key ``speaker`` or the first key), then exactly three others."""
    import cv2

    cage = np.asarray(cage, dtype=np.float64).reshape(4, 2)
    polys = {k: np.asarray(v, dtype=np.float64) for k, v in zones.items()}
    if "speaker" not in polys:
        raise CalibrationError("Flight phonotaxis needs a zone named 'speaker'.")
    others = [k for k in polys if k != "speaker"]
    if len(others) != 3:
        raise CalibrationError(
            f"Flight phonotaxis needs the speaker zone plus exactly three other zones, one on each "
            f"other wall (found {len(others)}: {', '.join(others) or 'none'}).",
            hint="Add or remove zones with `buzzphono calibrate`.",
        )
    names = ["speaker", *others]
    top = np.hypot(*(cage[1] - cage[0]))
    bot = np.hypot(*(cage[2] - cage[3]))
    width = float(0.5 * (top + bot))
    allpts = np.vstack([cage] + [polys[n] for n in names])
    gw, gh = int(allpts[:, 0].max() + 200) // DS + 2, int(allpts[:, 1].max() + 200) // DS + 2
    dms, cents, paths = [], [], []
    for n in names:
        poly = polys[n]
        m = np.zeros((gh, gw), np.uint8)
        cv2.fillPoly(m, [np.round(poly / DS).astype(np.int32)], 255)
        src = np.where(m > 0, 0, 255).astype(np.uint8)
        dms.append(cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32) * DS)
        yy, xx = np.nonzero(m)
        if len(xx) == 0:
            raise CalibrationError(f"Zone '{n}' is empty (no pixels inside the polygon).")
        cents.append(np.array([xx.mean() * DS, yy.mean() * DS]))
        paths.append(MplPath(poly))
    cents_a = np.array(cents)
    ctr = cage.mean(axis=0)
    cang = np.arctan2(cage[:, 1] - ctr[1], cage[:, 0] - ctr[0])
    sign = 1.0
    rel = (cang - cang[0]) % (2 * np.pi)
    if not (rel[1] < rel[2] < rel[3]):
        sign = -1.0
        rel = (-(cang - cang[0])) % (2 * np.pi)
    if not (rel[1] < rel[2] < rel[3]):
        raise CalibrationError("The four cage corners are not in order around the cage; redraw the cage.")
    mids = 0.5 * (cage + np.roll(cage, -1, axis=0))
    sec2zone = np.array([int(np.argmin(np.hypot(*(cents_a - m).T))) for m in mids])
    if len(set(sec2zone.tolist())) != 4:
        raise CalibrationError(
            f"Each of the four zones must sit on a different cage wall (walls map to zones "
            f"{sec2zone.tolist()} for {names}). Move the zones so there is one per wall."
        )
    return FlightGeometry(
        cage=cage,
        names=names,
        zones={n: polys[n] for n in names},
        width=width,
        s=width / REF_WIDTH,
        centre=ctr,
        bbox=(cage[:, 0].min(), cage[:, 0].max(), cage[:, 1].min(), cage[:, 1].max()),
        dms=dms,
        cents=cents_a,
        paths=paths,
        gw=gw,
        gh=gh,
        rel=rel,
        sign=sign,
        cang0=float(cang[0]),
        sec2zone=sec2zone,
        rpx=[f * width for f in r_fracs],
    )


def role_order(geom: FlightGeometry) -> tuple[list[int], list[str]]:
    """Indices into ``geom.names`` giving [speaker, opposite, top, bottom]."""
    names = geom.names
    cen = {n: geom.zones[n].mean(axis=0) for n in names}
    spk = names[0]
    ctrls = names[1:]
    right = max(ctrls, key=lambda n: float(np.hypot(*(cen[n] - cen[spk]))))
    rest = [n for n in ctrls if n != right]
    top = min(rest, key=lambda n: float(cen[n][1]))
    bottom = next(n for n in rest if n != top)
    ordered = [spk, right, top, bottom]
    return [names.index(n) for n in ordered], ordered
