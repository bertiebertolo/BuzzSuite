"""BuzzSwarm GIFs (Plotting > Swarming > Make GIFs): the flight paths build up over each other, the
last frame (every flight of the window) is held, and the GIF is drawn on the day's camera frame
of the cage when there is one."""
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

import conftest  # noqa: F401  -- repo root on sys.path

from buzzswarm import clock_analysis as ca
from buzzswarm import flight_maps as fm

DATE, WINDOW = "2026-07-04", "16:30-16:40"
CENTROID = np.array([400.0, 400.0])
CAGE = np.array([[100, 100], [700, 100], [700, 700], [100, 700]], float) - CENTROID
NS = 10**9


def _flight(start_s, x0):
    """A 20 s flight (25 fps) along a short circle around (x0, 0), start_s after 16:30."""
    d0 = np.datetime64(DATE, "ns").astype(np.int64) + (16 * 3600 + 30 * 60) * NS
    th = np.linspace(0, 2 * np.pi, 500)
    xy = np.c_[x0 + 60 * np.cos(th), 60 * np.sin(th)]
    return xy, d0 + start_s * NS + np.arange(500, dtype=np.int64) * NS // 25


def _frames(path):
    img = Image.open(path)
    out = []
    for i in range(img.n_frames):
        img.seek(i)
        out.append((np.asarray(img.convert("RGB")).astype(int), img.info.get("duration")))
    return out


def test_paths_build_up_and_the_last_frame_is_held():
    d = tempfile.mkdtemp(prefix="gif_")
    try:
        # one flight at the start (left), one near the end (right); 120x: 12 s per frame
        flights = [_flight(0, -150), _flight(480, 150)]
        path = fm.make_gif(flights, DATE, WINDOW, os.path.join(d, "a.gif"), "#d62728", CAGE, 320, 120)
        frames = _frames(path)
        assert len(frames) == fm.gif_frames([WINDOW], 120) == 50
        red = [(f[40:, :, 0] - f[40:, :, 2]) > 30 for f, _ms in frames]   # path pixels, below the clock
        early, last = red[5], red[-1]          # 16:31:12: the first flight is over
        assert early.sum() > 100, "the first flight is not drawn"
        assert (early & ~last).sum() <= 0.02 * early.sum(), "the first flight did not stay drawn"
        assert last.sum() > 1.6 * early.sum(), "the second flight was not added"
        assert all(red[i].sum() >= red[5].sum() * 0.98 for i in range(5, 50)), "paths vanished"
        assert frames[0][1] == 1000 // fm.GIF_FPS and frames[-1][1] == fm.GIF_HOLD_MS
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gif_is_drawn_on_the_days_camera_frame():
    d = tempfile.mkdtemp(prefix="gif_")
    try:
        exp = os.path.join(d, "Exp")
        frames_dir = os.path.join(exp, "individual_images")
        os.makedirs(frames_dir)
        pic = np.full((800, 800), 255, np.uint8)
        pic[:400, :400] = 0                              # top-left quadrant of the cage dark
        for name in ("Cage_20260704_160000_00000.png", "Cage_20260704_160000_00001.png",
                     "Cage_20260704_160000_00002.png", "Cage_20260705_160000_00001.png", "notes.png"):
            Image.fromarray(pic).save(os.path.join(frames_dir, name))
        # the segment whose middle is nearest the window's (16:30 vs 16:35); other days lose
        assert os.path.basename(fm.gif_picture(exp, DATE, WINDOW)) == "Cage_20260704_160000_00001.png"
        assert os.path.basename(fm.gif_picture(exp, "2026-07-05", WINDOW)) == "Cage_20260705_160000_00001.png"
        # nothing within 6 h: the experiment's reference frame
        assert fm.gif_picture(exp, DATE, "05:00-05:30") == ca.cage_reference_image(exp)
        assert fm.gif_picture(os.path.join(d, "none"), DATE, WINDOW) is None

        path = fm.make_gif([_flight(0, 150)], DATE, WINDOW, os.path.join(d, "b.gif"), "#1f77b4",
                           CAGE, 320, 120, picture=fm.gif_picture(exp, DATE, WINDOW), centroid=CENTROID)
        f = _frames(path)[0][0]
        cx, cy, off = 225, 256, 70                       # the axes' centre in GIF pixels; ~100 px
        dark = 255 * (1 - fm.GIF_PHOTO_FADE)
        assert abs(f[cy - off, cx - off].mean() - dark) < 25, "top-left of the cage is not the dark part"
        # bottom-left and top-right are the light parts (the picture is not flipped)
        assert f[cy + off, cx - off].mean() > 240 and f[cy - off, cx + off].mean() > 240
        # without a picture: white, with the cage outline
        path = fm.make_gif([], DATE, WINDOW, os.path.join(d, "c.gif"), "#1f77b4", CAGE, 320, 120,
                           picture=None, centroid=CENTROID)
        assert _frames(path)[0][0][cy - off, cx - off].mean() > 240
        # an unreadable picture is skipped, not an error
        bad = os.path.join(d, "bad.png")
        with open(bad, "w") as fh:
            fh.write("not a png")
        fm.make_gif([], DATE, WINDOW, os.path.join(d, "d.gif"), "#1f77b4", CAGE, 320, 120,
                    picture=bad, centroid=CENTROID)
    finally:
        shutil.rmtree(d, ignore_errors=True)
