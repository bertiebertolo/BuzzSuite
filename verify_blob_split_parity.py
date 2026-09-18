#!/usr/bin/env python
"""Self-check: the crop-based blob splitting in single_video_analysis.split_merged_contour gives
exactly the same centroids, in the same order, as the original full-frame version.

Run it once on any machine / OpenCV build the tracker is used on (Windows and macOS use different
OpenCV threading back-ends, which pick different connected-component algorithms):

    python verify_blob_split_parity.py

Needs no data (synthetic merged blobs, incl. ones touching every frame edge, at 3 frame sizes).
Prints PASS or FAIL and exits non-zero on any difference. See DEVLOG 2026-09-17.
"""
import struct
import sys

import cv2
import numpy as np

from buzzwatch_data_analysis.single_video_analysis import split_merged_contour


# Verbatim copy of split_merged_contour as it was before the crop change -- the reference.
def _reference_split(shape, contour, single_area, min_length):
    """Split one over-sized contour into multiple centroids (WS2).

    Mosquitoes crowd at the speaker and merge into a single blob that is then *rejected* by the
    ``max_length`` size filter, so the whole cluster vanishes. Here we recover the individuals via
    local-maxima detection on the distance transform: each peak that is locally highest within a
    ~mosquito-radius neighbourhood is one centroid. cv2-only, no skimage.

    Returns a list of (x, y) centroids; falls back to the bounding-box centre if it cannot split.
    """
    import math
    (x, y, w, h) = cv2.boundingRect(contour)
    fallback = [(x + w / 2.0, y + h / 2.0)]
    area = cv2.contourArea(contour)
    if not single_area or single_area <= 0 or area < 1.6 * single_area:
        return fallback
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist_map.max() <= 0:
        return fallback
    # Local-maxima detection: a pixel is a peak if it equals the dilation within a
    # neighbourhood whose radius matches the expected single-mosquito radius.  This finds each
    # mosquito's core independently of how deep the overall blob is (fixes the dense-crowd case
    # where the old global threshold 0.5*max was too high and merged many peaks into one).
    radius = max(2, int(math.sqrt(single_area / math.pi)))
    k = max(3, 2 * radius - 1)   # odd kernel, ~mosquito diameter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(dist_map, kernel)
    floor = max(1.0, radius * 0.4)   # ignore peaks shallower than ~half a mosquito radius
    local_max = ((dist_map == dilated) & (dist_map > floor)).astype(np.uint8) * 255
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max)
    pts = []
    for label in range(1, n_labels):  # skip background label 0
        if stats[label, cv2.CC_STAT_AREA] >= max(1, (min_length * min_length) / 4.0):
            cx, cy = centroids[label]
            pts.append((float(cx), float(cy)))
    return pts if pts else fallback


def _bits(points):
    return [tuple(struct.pack('<d', v) for v in p) for p in points]


def main():
    rng = np.random.default_rng(42)
    checked = split = mismatches = 0
    for thread_setting in (cv2.getNumThreads(), 1):
        cv2.setNumThreads(thread_setting)
        for shape in ((1080, 1920), (720, 1280), (101, 77)):
            height, width = shape
            for _ in range(700):
                img = np.zeros(shape, np.uint8)
                cx = int(rng.choice([0, 1, width - 1, width - 2, int(rng.integers(0, width))]))
                cy = int(rng.choice([0, 1, height - 1, height - 2, int(rng.integers(0, height))]))
                for _ in range(int(rng.integers(1, 12))):
                    center = (int(np.clip(cx + rng.integers(-30, 31), 0, width - 1)),
                              int(np.clip(cy + rng.integers(-30, 31), 0, height - 1)))
                    cv2.circle(img, center, int(rng.integers(2, 12)), 255, -1)
                if rng.random() < 0.3:
                    x = int(rng.integers(0, width))
                    cv2.line(img, (x, 0), (x, height - 1), 255, int(rng.integers(1, 6)))
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                single_area = float(rng.choice([8.0, 50.0, 150.0, 400.0]))
                min_length = int(rng.choice([1, 3, 5]))
                for contour in contours:
                    expected = _reference_split(shape, contour, single_area, min_length)
                    actual = split_merged_contour(shape, contour, single_area, min_length)
                    checked += 1
                    split += len(expected) > 1
                    if _bits(expected) != _bits(actual):
                        mismatches += 1
    print("OpenCV %s: %d contours checked (%d split into several mosquitoes), %d differences"
          % (cv2.__version__, checked, split, mismatches))
    print("PASS" if mismatches == 0 else "FAIL -- do not use this build for tracking; see DEVLOG")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
