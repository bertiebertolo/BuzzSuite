"""Shared definition of the adopted tracking-overhaul config + a helper to apply it.

The BuzzSuite tracker overhaul (WS1 within-video bright-percentile reference, WS3/WS4 motion-aware
flying tracker, WS5 adaptive bright-frame-std resting threshold) is entirely feature-flagged: the
pipeline reads a ``tracking:`` block from the settings dict via ``tracking_flag()`` in
``single_video_analysis.py``.  No block => original behaviour.

This module is the single source of truth for the adopted config (mirrors the ``TRACKING`` dict in
``run_experiment_overhaul.py``) so the GUI can toggle the exact same overhaul the batch scripts use,
WITHOUT writing it into the user's ``buzzwatch_track_settings.yml`` (the toggle injects it only into
the in-memory settings used for a given run).
"""
import copy

# Adopted config = WS1 + WS3 + WS4 + WS5 (p75 / k6). See TRACKING_OVERHAUL_HANDOFF.md "FINAL STATUS".
OVERHAUL_TRACKING = {
    "resting_reference": "bright_percentile",   # WS1 within-video p90 reference
    "resting_reference_percentile": 90,
    "resting_reference_samples": 80,
    "resting_adaptive": True,                   # WS5 adaptive bright-frame-std threshold
    "resting_adaptive_k": 6.0,
    "resting_threshold_floor": 12,              # = the working flat threshold; stable pixels unchanged
    "resting_std_percentile": 75,               # empty-cage cutoff; protects long/low-activity resters
    "resting_reference_window": 0,              # WS7 rolling reference OFF (tested, not adopted)
    "split_blobs": True,                        # WS2 blob-split (the adopted "v7" method)
    "assignment": "hungarian",                  # WS3 flying tracker (moving only)
    "motion_model": "constant_velocity",        # WS3
    "use_backward_pass": True,                  # WS4 forward+backward merge
}

# Non-``tracking`` overrides that the adopted config also sets (see run_experiment_overhaul.py).
OVERHAUL_SEG_RESTING_GRAY_TRESH = 12   # overhaul flat fallback (used only if resting_adaptive is off)
OVERHAUL_TRACK_MOVING_MAXDISAPPEARED = 15  # tolerate flight occlusion
OVERHAUL_SINGLE_BLOB_AREA = 150   # WS2 single-mosquito footprint (px^2); split triggers at 1.6x = 240


def apply_tracking_overhaul(settings, enabled):
    """Return a deep copy of ``settings`` with the overhaul applied (``enabled``) or removed.

    Pure function — never mutates the input dict, so the caller's loaded settings stay clean and the
    overhaul is never written back to the YAML. When enabled, injects the adopted ``tracking`` block
    plus the two related overrides; when disabled, strips the ``tracking`` block so the original
    tracking algorithm is used.
    """
    if settings is None:
        return settings
    out = copy.deepcopy(settings)
    if enabled:
        out["tracking"] = copy.deepcopy(OVERHAUL_TRACKING)
        out.setdefault("seg_resting", {})["gray_tresh"] = OVERHAUL_SEG_RESTING_GRAY_TRESH
        out.setdefault("seg_resting", {})["single_blob_area"] = OVERHAUL_SINGLE_BLOB_AREA
        out.setdefault("seg_moving", {})["single_blob_area"] = OVERHAUL_SINGLE_BLOB_AREA
        out.setdefault("track_moving", {})["maxDisappeared"] = OVERHAUL_TRACK_MOVING_MAXDISAPPEARED
        out["activity_only"] = False
    else:
        out.pop("tracking", None)
    return out
