"""BuzzSwarm's pixel settings follow the chosen swarm-zone radius: with no radius (the default
100 px) they are exactly the old constants; with a camera that sees the cage at twice the scale and
a swarm zone set twice as large, every scale-free metric comes out the same."""
import math

import numpy as np
import pandas as pd

import conftest  # noqa: F401  -- repo root on sys.path

from buzzswarm import clock_analysis as ca

CENTROID = (400.0, 400.0)
SQUARE = [[100.0, 100.0], [700.0, 100.0], [700.0, 700.0], [100.0, 700.0]]
DATE = "2026-07-04"
WINDOW = "16:00-16:30"


def test_default_radius_gives_the_old_constants():
    for radius in (None, 0, ca.SWARM_RADIUS_PX, float(ca.SWARM_RADIUS_PX)):
        px = ca.px_settings(radius)
        assert px == {"scale": 1.0, "central_radius_px": 150, "core_px": 60, "shape_ring_px": 20,
                      "shape_max_px": 240, "heat_cell_px": 20, "fly_px_frame": 3.0}, (radius, px)
    assert ca.px_settings() == ca.px_settings(100)


def test_settings_scale_with_the_radius():
    px = ca.px_settings(200)
    assert px["scale"] == 2.0
    assert (px["central_radius_px"], px["core_px"], px["heat_cell_px"], px["fly_px_frame"]) == (300, 120, 40, 6.0)
    assert (px["shape_ring_px"], px["shape_max_px"]) == (40, 480)
    px = ca.px_settings(137)
    assert math.isclose(px["central_radius_px"], 150 * 1.37) and math.isclose(px["fly_px_frame"], 3 * 1.37)
    for bad in (-50, float("nan"), float("inf")):
        try:
            ca.px_settings(bad)
            raise AssertionError("accepted radius %r" % bad)
        except ValueError:
            pass


def test_default_geometry_is_unchanged():
    """The geometry for the default radius matches what the constants gave before."""
    geo = ca.geometry_from(CENTROID, SQUARE, ca.SWARM_RADIUS_PX)
    assert np.array_equal(geo["areas"]["ring_edges"], np.arange(0, 240 + 20, 20))
    half = np.ceil(np.abs(np.asarray(SQUARE) - CENTROID).max() / 20) * 20
    assert np.array_equal(geo["heat_edges"], np.arange(-half, half + 20, 20))
    # the old _area_shares, inline, with the same seed
    from matplotlib.path import Path
    rng = np.random.default_rng(0)
    poly = np.asarray(SQUARE)
    pts = np.empty((0, 2))
    while len(pts) < 400_000:
        cand = rng.uniform(poly.min(axis=0), poly.max(axis=0), size=(400_000, 2))
        pts = np.vstack([pts, cand[Path(poly).contains_points(cand)]])
    r = np.hypot(pts[:400_000, 0] - CENTROID[0], pts[:400_000, 1] - CENTROID[1])
    assert geo["areas"]["central"] == float(np.mean(r < 150))
    assert geo["areas"]["core"] == float(np.mean(r < 60))
    assert np.array_equal(geo["areas"]["rings"], np.histogram(r, np.arange(0, 260, 20))[0] / 400_000.0)
    # odd scales still give a whole number of rings and heatmap cells
    g = ca.geometry_from(CENTROID, SQUARE, 137)
    assert len(g["areas"]["ring_edges"]) == 13 and math.isclose(g["areas"]["ring_edges"][-1], 240 * 1.37)
    assert np.allclose(np.diff(g["heat_edges"]), 20 * 1.37)


# ---------------------------------------------------------------- a synthetic day at two scales
def _synthetic_day(scale=1.0, seed=3):
    """Flights circling the centre between 16:00 and 16:10 (several at once), some slow pieces just
    under the real-flight threshold and some just over it; coordinates times `scale` about the centre."""
    rng = np.random.default_rng(seed)
    fps = 25.0
    d0 = np.datetime64(DATE, "ns").astype(np.int64)
    t_start = d0 + 16 * 3600 * 10**9
    coords, times, offsets = [], [], [0]
    for i in range(320):
        n = 100
        r0 = rng.uniform(15, 280)
        step = 9.0 if i % 4 else rng.choice([2.6, 3.4])        # px/frame along the circle
        a0 = rng.uniform(0, 2 * np.pi)
        ang = a0 + np.arange(n) * step / r0
        x = CENTROID[0] + r0 * np.cos(ang) + rng.normal(0, 0.3, n)
        y = CENTROID[1] + r0 * np.sin(ang) + rng.normal(0, 0.3, n)
        start = t_start + int(rng.uniform(0, 600) * 10**9)
        coords.append(np.c_[x, y])
        times.append(start + (np.arange(n) * 10**9 / fps).astype(np.int64))
        offsets.append(offsets[-1] + n)
    c = np.vstack(coords)
    c = np.asarray(CENTROID) * scale + (c - np.asarray(CENTROID)) * scale
    pop = {"coords": c, "t_ns": np.concatenate(times), "offsets": np.asarray(offsets)}
    spans = np.array([[d0, d0 + 86400 * 10**9]], dtype=np.int64)
    return {"date": DATE, "fps": fps, "spans": spans, "tracks": pop, "flight": pop}


def _window_row(scale, radius):
    day = _synthetic_day(scale)
    poly = np.asarray(CENTROID) * scale + (np.asarray(SQUARE) - np.asarray(CENTROID)) * scale
    geo = ca.geometry_from(np.asarray(CENTROID) * scale, poly, radius)
    out = ca.day_metrics(day, geo, bin_minutes=30, windows=[WINDOW])
    return out, geo


def test_scale_free_metrics_do_not_depend_on_the_camera():
    a, _ = _window_row(1.0, 100)
    b, geo_b = _window_row(2.0, 200)
    wa, wb = a["windows"][0], b["windows"][0]
    assert wa["n_flying"] > 0.5 and wa["swarm_ge3_frac"] > 0, "the synthetic day has too little flight"
    same = ("n_flying", "swarm_mean", "swarm_ge3_frac", "swarm_ge5_frac", "swarm_p95",
            "central_enrichment", "core_density", "shape_index", "turning_deg_s", "activity", "straightness")
    for k in same:
        assert np.isfinite(wa[k]), k
        assert math.isclose(wa[k], wb[k], rel_tol=1e-6, abs_tol=1e-9), (k, wa[k], wb[k])
    for k in ("speed_px_s", "nn_px"):                     # measured in pixels: twice as large
        assert math.isclose(wb[k], 2 * wa[k], rel_tol=1e-6), (k, wa[k], wb[k])
    # the profile and the heatmap: same shares in rings / cells twice as large
    pa, pb = a["profile"], b["profile"]
    assert len(pa) == len(pb) == 12
    for ra, rb in zip(pa, pb):
        assert math.isclose(rb["r_hi_px"], 2 * ra["r_hi_px"])
        assert (np.isnan(ra["density"]) and np.isnan(rb["density"])) or \
            math.isclose(ra["density"], rb["density"], rel_tol=1e-6), (ra, rb)
    ha = sorted((h["x_px"], h["y_px"], h["share"]) for h in a["heatmap"])
    hb = sorted((h["x_px"] / 2, h["y_px"] / 2, h["share"]) for h in b["heatmap"])
    assert len(ha) == len(hb) and all(np.allclose(p, q) for p, q in zip(ha, hb))
    assert geo_b["px"]["heat_cell_px"] == 40


def test_unscaled_settings_would_differ():
    """Guard for the test above: with the old fixed settings the 2x camera gave other numbers."""
    b_fixed = ca.day_metrics(_synthetic_day(2.0), dict(ca.geometry_from((800.0, 800.0),
                             np.asarray(SQUARE) * 2.0, 200), px=ca.px_settings(100),
                             areas=ca._area_shares(np.asarray(SQUARE) * 2.0, (800.0, 800.0), 200.0)),
                             bin_minutes=30, windows=[WINDOW])["windows"][0]
    a, _ = _window_row(1.0, 100)
    assert not math.isclose(a["windows"][0]["central_enrichment"], b_fixed["central_enrichment"], rel_tol=1e-3)


def test_geometry_json_records_the_settings_used():
    import tempfile
    import json
    import os
    import shutil
    d = tempfile.mkdtemp(prefix="bs_px_")
    try:
        geo = ca.geometry_from(CENTROID, SQUARE, 150)
        payload = ca._write_geometry(d, "exp", "grp", geo)
        with open(os.path.join(d, "geometry.json")) as f:
            on_disk = json.load(f)
        assert on_disk == payload
        assert payload["swarm_radius_px"] == 150 and payload["px_scale"] == 1.5
        assert payload["central_radius_px"] == 225 and payload["core_px"] == 90
        assert payload["heat_cell_px"] == 30 and payload["fly_px_frame"] == 4.5
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_heatmap_plot_uses_the_experiments_cell_size():
    from buzzswarm import clock_plots as cp
    out, geo = _window_row(2.0, 200)
    hm = pd.DataFrame(out["heatmap"]).assign(experiment="cam3", group="cam3", date=DATE)
    geometry = {"cam3": {"centroid": [800.0, 800.0], "polygon": (np.asarray(SQUARE) * 2).tolist(),
                         "swarm_radius_px": 200.0, "heat_cell_px": geo["px"]["heat_cell_px"]}}
    fig, table = cp.heatmaps(hm, geometry)
    assert fig is not None
    assert sorted(set(table.x_px.round(6))) == sorted(set(hm.x_px.round(6))), \
        "the plot's cells are not the experiment's 40 px cells"
    # an older geometry.json without heat_cell_px still plots with the default cell
    a, _ = _window_row(1.0, 100)
    hm1 = pd.DataFrame(a["heatmap"]).assign(experiment="old", group="old", date=DATE)
    fig1, table1 = cp.heatmaps(hm1, {"old": {"centroid": list(CENTROID), "polygon": SQUARE}})
    assert sorted(set(table1.x_px.round(6))) == sorted(set(hm1.x_px.round(6)))
