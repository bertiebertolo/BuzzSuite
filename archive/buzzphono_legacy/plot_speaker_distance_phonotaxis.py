#!/usr/bin/env python3
"""
Per-trajectory speaker-distance analysis + 2D occupancy heatmaps for a
phonotaxis experiment.

Loads forward_mosq_tracks pickles from final_tracking_data/, parses absolute
time from the filename, classifies each trajectory by minute-of-hour
(stim ON = min 40-50, OFF = min 30-40 pre-stim only — balanced 10-vs-10), and produces:
  1. per-trajectory mean speaker-distance distribution: ON vs OFF
  2. 2D occupancy heatmaps: ON vs OFF, with cage borders overlaid

Speaker side: left wall (square_3 polygon in settings). The speaker-distance
metric is the trajectory's mean x-coordinate (pixels). Smaller = closer to
the speaker wall.
"""

import argparse
import glob
import os
import pickle
import re
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from matplotlib.path import Path as MplPath
from scipy.stats import mannwhitneyu, wilcoxon

# ---------------------------------------------------------------------------
STIM_START_MIN   = 40
STIM_END_MIN     = 50
# ON metric is evaluated from ON_EVAL_START_MIN. A tracker WARM-UP exists (recordings chunk at
# :00/:20/:40 so ON min40-50 starts on a fresh video, tracked<<raw for ~2 min then recovers), BUT
# the recruitment metric is a PROPORTION (in_zone/total): a cold tracker shrinks numerator AND
# denominator together, so trimming the warm-up was tested and did NOT help (it slightly lowered
# recruitment). Kept at 0 (= no trim, original ON 40-50). Set >0 only for absolute-count metrics.
ON_WARMUP_TRIM   = 0
ON_EVAL_START_MIN = STIM_START_MIN + ON_WARMUP_TRIM
OFF_START_MIN    = 28   # OFF = 10-min PRE-stim window; 28-38 avoids the ~2-min onset bleed at min 38-40
OFF_END_MIN      = 38
FPS              = 25.0
CHUNK_DURATION_S = 1200            # 20 min per chunk (verified)
STIM_COLOR       = '#ff8c00'
OFF_COLOR        = '#444444'
OFF_LABEL        = f"OFF {OFF_START_MIN}-{OFF_END_MIN}"   # labels derive from the windows above
ON_LABEL         = f"ON {STIM_START_MIN}-{STIM_END_MIN}"
ZT0_HOUR         = 5     # Penzance photoperiod: lights on 05:00, off 17:00

# Same palette as activity_plot_manager.py's/buzzswarm's _GROUP_PALETTE, for visual consistency
# across the app's multi-experiment comparison plots. Only used by run_multi_speaker_comparison.
_GROUP_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def _hours_in(s, e):     # half-open clock-hour range, may wrap past midnight
    return set(range(s, e)) if s <= e else set(range(s, 24)) | set(range(0, e))


# Candidate roots for the `--cage` CLI shortcut below (_resolve_cage / _CAGE_CFG); the GUI-driven
# functions in this module take an experiment path directly and don't use these. Set
# BUZZSUITE_DATA_ROOT / BUZZSUITE_RESULTS_ROOT to point these at your own data drive.
_env_data_root = os.environ.get('BUZZSUITE_DATA_ROOT')
_ANALYSIS_ROOTS = ([os.path.join(_env_data_root, 'Analysis', 'Phonotaxis', 'Penzance')]
                    if _env_data_root else []) + [
    "/Volumes/Mosquito2/Buzzwatch/Analysis/Phonotaxis/Penzance",
    r"E:\Buzzwatch\Analysis\Phonotaxis\Penzance",
]
_env_results_root = os.environ.get('BUZZSUITE_RESULTS_ROOT')
_RESULTS_ROOTS = ([_env_results_root] if _env_results_root else []) + [
    "/Volumes/Mosquito2/Phonotaxis results new",
    r"E:\Phonotaxis results new",
]
_CAGE_CFG = {
    "6_4-5":   {"suffix": "AedesAegyptiLiv475M_6_4-5_overhaul_split",
                "start": "2026-06-04 00:00", "end": "2026-06-06 00:00"},
    "6_9-10":  {"suffix": "AedesAegyptiLiv475M_6_9-10_overhaul_split",
                "start": "2026-06-09 00:00", "end": "2026-06-11 00:00"},
    "6_14-15": {"suffix": "AedesAegyptiLiv475M_6_14-15_overhaul_split",
                "start": "2026-06-14 00:00", "end": "2026-06-16 00:00"},
}


def _resolve_cage(cage):
    """Return (experiment_dir, output_dir, start_str, end_str) from _CAGE_CFG."""
    cfg = _CAGE_CFG[cage]
    exp_dir = None
    for root in _ANALYSIS_ROOTS:
        candidate = os.path.join(root, cfg["suffix"])
        if os.path.isdir(candidate):
            exp_dir = candidate
            break
    if exp_dir is None:
        raise FileNotFoundError(f"Experiment dir for cage {cage!r} not found under {_ANALYSIS_ROOTS}")
    out_root = next((r for r in _RESULTS_ROOTS if os.path.isdir(r)), _RESULTS_ROOTS[0])
    out_dir = os.path.join(out_root, cage + "_split", "speaker_distance")
    return exp_dir, out_dir, cfg["start"], cfg["end"]


PHASE_DEFS = [('dawn (05-06h)', 5, 6, '#7b3294'), ('day (06-16h)', 6, 16, '#1a9850'),
              ('dusk (16-17h)', 16, 17, '#d73027'), ('night (17-05h)', 17, 5, '#4575b4')]
PHASES = [(name, _hours_in(s, e), color) for name, s, e, color in PHASE_DEFS]
# ---------------------------------------------------------------------------


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


FNAME_RE = re.compile(r'_(\d{8})_(\d{6})_(\d{5})$')

def _parse_chunk_start(fname):
    m = FNAME_RE.search(fname)
    if not m:
        return None
    d, t, idx = m.groups()
    sess_start = pd.Timestamp(f'{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}')
    return sess_start + pd.Timedelta(seconds=int(idx) * CHUNK_DURATION_S)


def collect_trajectories(folder, t_start, t_end):
    """Yield (chunk_start, mosquito_dict) for every track in window."""
    files = sorted(glob.glob(os.path.join(folder, 'forward_mosq_tracks_*')))
    print(f"  scanning {len(files)} chunks...")
    for i, f in enumerate(files):
        cs = _parse_chunk_start(os.path.basename(f))
        if cs is None:
            continue
        chunk_end = cs + pd.Timedelta(seconds=CHUNK_DURATION_S)
        if chunk_end < t_start or cs >= t_end:
            continue
        with open(f, 'rb') as fh:
            obj = _CompatUnpickler(fh).load()
        for m in obj.objects.values():
            yield cs, m
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(files)}")


def trajectory_metrics(folder, t_start, t_end):
    """
    Return:
        traj_df: DataFrame with columns mean_x, mean_y, n_frames, mid_time,
                 frac_min_on (fraction of frames inside stim window)
        all_xy_on, all_xy_off: arrays of (x,y) for every frame, by ON/OFF
    """
    rows = []
    xy_on, xy_off = [], []
    for chunk_start, m in collect_trajectories(folder, t_start, t_end):
        coords = m.get('coordinates')
        if not coords:
            continue
        start_frame = int(m.get('start', 0))
        n = len(coords)
        frame_idx  = np.arange(n) + start_frame
        # absolute time per frame
        times = chunk_start + pd.to_timedelta(frame_idx / FPS, unit='s')
        in_win = (times >= t_start) & (times < t_end)
        if not in_win.any():
            continue
        coords_arr = np.asarray(coords, dtype=float)[in_win]
        times_in   = times[in_win]
        mins = times_in.minute + times_in.second / 60.0
        on_mask  = (mins >= STIM_START_MIN) & (mins < STIM_END_MIN)   # ON  = 40-50
        off_mask = (mins >= OFF_START_MIN)  & (mins < OFF_END_MIN)    # OFF = 30-40 (pre-stim only)
        n_cmp = int(on_mask.sum() + off_mask.sum())
        if n_cmp == 0:
            continue   # trajectory not in either compared window -> excluded

        # accumulate per-frame xy (ON vs the 10-min PRE window only)
        if on_mask.any():
            xy_on.append(coords_arr[on_mask])
        if off_mask.any():
            xy_off.append(coords_arr[off_mask])

        # per-trajectory summary; frac_on is among compared (ON+OFF) frames only
        rows.append({
            'mean_x':       coords_arr[:, 0].mean(),
            'mean_y':       coords_arr[:, 1].mean(),
            'n_frames':     len(coords_arr),
            'mid_time':     times_in[len(times_in) // 2],
            'frac_on':      float(on_mask.sum() / n_cmp),
        })

    traj_df = pd.DataFrame(rows)
    xy_on  = np.concatenate(xy_on, axis=0)  if xy_on  else np.empty((0, 2))
    xy_off = np.concatenate(xy_off, axis=0) if xy_off else np.empty((0, 2))
    return traj_df, xy_on, xy_off


def collect_xy_by_state(folder, t_start, t_end):
    """Per-frame (x,y) points from a tracking_resting/ or tracking_moving/ folder, split ON/OFF.

    These folders store one track per object with explicit per-frame 'time_stamp' (frame index) and
    'coordinates', so resting-only / flying-only occupancy is read straight from the state folder.
    """
    files = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    print(f"  {os.path.basename(folder)}: scanning {len(files)} chunks...")
    xy_on, xy_off = [], []
    for f in files:
        cs = _parse_chunk_start(os.path.splitext(os.path.basename(f))[0])  # strip .pkl (regex is $-anchored)
        if cs is None:
            continue
        try:
            with open(f, 'rb') as fh:
                obj = _CompatUnpickler(fh).load()
        except Exception:
            continue
        if not hasattr(obj, 'objects') or not obj.objects:
            continue
        for m in obj.objects.values():
            ts = m.get('time_stamp', [])
            co = m.get('coordinates', [])
            n = min(len(ts), len(co))
            if n == 0:
                continue
            ts = np.asarray(ts[:n], dtype=float)
            co = np.asarray(co[:n], dtype=float)
            times = cs + pd.to_timedelta(ts / FPS, unit='s')
            in_win = (times >= t_start) & (times < t_end)
            if not in_win.any():
                continue
            co = co[in_win]
            tw = times[in_win]
            mins = tw.minute + tw.second / 60.0
            on_mask  = (mins >= STIM_START_MIN) & (mins < STIM_END_MIN)   # ON  = 40-50
            off_mask = (mins >= OFF_START_MIN)  & (mins < OFF_END_MIN)    # OFF = 30-40 (pre-stim only)
            if on_mask.any():
                xy_on.append(co[on_mask])
            if off_mask.any():
                xy_off.append(co[off_mask])

    def _stack(lst):
        a = np.concatenate(lst, axis=0) if lst else np.empty((0, 2))
        return a[np.isfinite(a).all(axis=1)] if len(a) else a
    return _stack(xy_on), _stack(xy_off)


def collect_resting_zone_by_cycle(folder, t_start, t_end, speaker_poly, n_min_bins=60):
    """Stream resting detections; accumulate, per (hour-cycle, minute-of-hour bin), the count of
    points inside the custom speaker zone and the total count.

    One streaming pass feeds BOTH the minute-of-hour phase plot and the per-cycle paired OFF-vs-ON
    plot. Each hour-cycle is treated as an independent replicate, which avoids the frame-level
    pseudoreplication of the pooled bar chart (one mosquito resting for minutes contributes
    thousands of near-identical frames). Counts only, so memory is tiny regardless of point count.

    Also accumulates a per-absolute-minute resting series (whole experiment) for the resting-activity
    time series. Returns (in_zone, total, t0_floor, abs_in, abs_total): in_zone/total are shape
    (n_cycles, n_min_bins); abs_in/abs_total are 1-D, length n_cycles*60 (one bin per wall-clock min).
    """
    speaker_path = MplPath(speaker_poly)
    t0_floor = t_start.floor('H')
    n_cycles = int(np.ceil((t_end - t0_floor) / pd.Timedelta(hours=1))) + 1
    bin_w = 60.0 / n_min_bins
    size = n_cycles * n_min_bins
    in_flat  = np.zeros(size, dtype=np.float64)
    tot_flat = np.zeros(size, dtype=np.float64)
    n_abs    = n_cycles * 60
    abs_in   = np.zeros(n_abs, dtype=np.float64)
    abs_tot  = np.zeros(n_abs, dtype=np.float64)

    files = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    print(f"  {os.path.basename(folder)} (by-cycle): scanning {len(files)} chunks...")
    unreadable = []
    for f in files:
        cs = _parse_chunk_start(os.path.splitext(os.path.basename(f))[0])
        if cs is None:
            continue
        try:
            with open(f, 'rb') as fh:
                obj = _CompatUnpickler(fh).load()
        except Exception:
            unreadable.append(os.path.basename(f))
            continue
        if not hasattr(obj, 'objects') or not obj.objects:
            continue
        # Collect ALL objects from this chunk together so we can group by frame before
        # accumulating. This lets us apply a per-frame colony cap on the total resting
        # count: the split-blob tracker over-detects (one clustered mosquito -> several
        # objects) and the resting channel includes static floor false-positives, so the
        # raw per-frame count routinely exceeds the 30-mosquito colony. Floor FPs inflate
        # only the denominator (not the speaker-zone numerator), deflating the speaker
        # fraction. A per-frame cap of COLONY corrects this. See also
        # plot_resting_activity_timeseries and export_zt_clips._cap_detections (same cap).
        COLONY = 30.0
        chunk_frame_ts  = []   # raw frame indices (float, FPS units) per object-frame
        chunk_frame_inz = []   # True/False inside speaker zone
        for m in obj.objects.values():
            ts = m.get('time_stamp', [])
            co = m.get('coordinates', [])
            n = min(len(ts), len(co))
            if n == 0:
                continue
            ts = np.asarray(ts[:n], dtype=float)
            co = np.asarray(co[:n], dtype=float)
            ok = np.isfinite(ts) & np.isfinite(co).all(axis=1)
            if not ok.any():
                continue
            ts, co = ts[ok], co[ok]
            times = cs + pd.to_timedelta(ts / FPS, unit='s')
            in_win = (times >= t_start) & (times < t_end)
            if not in_win.any():
                continue
            chunk_frame_ts.append(ts[in_win])
            chunk_frame_inz.append(speaker_path.contains_points(co[in_win]))
        if not chunk_frame_ts:
            continue
        all_frame_ts  = np.concatenate(chunk_frame_ts)
        all_frame_inz = np.concatenate(chunk_frame_inz).astype(np.float64)
        # Group by exact frame index, then apply per-frame COLONY cap
        unique_frames, inv = np.unique(all_frame_ts, return_inverse=True)
        n_uf = len(unique_frames)
        per_frame_total  = np.bincount(inv, minlength=n_uf).astype(np.float64)
        per_frame_inzone = np.bincount(inv, weights=all_frame_inz, minlength=n_uf)
        cap_total  = np.minimum(per_frame_total,  COLONY)
        cap_inzone = np.minimum(per_frame_inzone, cap_total)
        # Compute (cycle, minute-bin) for each unique frame and accumulate
        frame_times = cs + pd.to_timedelta(unique_frames / FPS, unit='s')
        cyc  = np.round(((frame_times.floor('H') - t0_floor) / pd.Timedelta(hours=1)).values).astype(int)
        mins = frame_times.minute.values + frame_times.second.values / 60.0
        mb   = np.clip((mins / bin_w).astype(int), 0, n_min_bins - 1)
        good = (cyc >= 0) & (cyc < n_cycles)
        flat = cyc[good] * n_min_bins + mb[good]
        tot_flat += np.bincount(flat, weights=cap_total[good],  minlength=size)
        in_flat  += np.bincount(flat, weights=cap_inzone[good], minlength=size)
        # absolute wall-clock minute (whole-experiment resting series)
        amin = np.floor(((frame_times[good] - t0_floor) / pd.Timedelta(minutes=1)).values).astype(int)
        a_ok = (amin >= 0) & (amin < n_abs)
        abs_tot += np.bincount(amin[a_ok], weights=cap_total[good][a_ok],  minlength=n_abs)
        abs_in  += np.bincount(amin[a_ok], weights=cap_inzone[good][a_ok], minlength=n_abs)

    if unreadable:
        examples = ", ".join(unreadable[:5]) + (", ..." if len(unreadable) > 5 else "")
        print(f"  {os.path.basename(folder)}: skipped {len(unreadable)} unreadable chunk(s): {examples}")

    return (in_flat.reshape(n_cycles, n_min_bins), tot_flat.reshape(n_cycles, n_min_bins),
            t0_floor, abs_in, abs_tot)


def collect_abs_minute_total(folder, t_start, t_end, t0_floor, n_abs):
    """Per-absolute-minute total detection count from a state folder (e.g. tracking_moving), aligned
    to t0_floor with n_abs one-minute bins. Adds the flying series to the activity plot."""
    out = np.zeros(n_abs, dtype=np.float64)
    files = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    print(f"  {os.path.basename(folder)} (abs-min total): scanning {len(files)} chunks...")
    for f in files:
        cs = _parse_chunk_start(os.path.splitext(os.path.basename(f))[0])
        if cs is None:
            continue
        try:
            with open(f, 'rb') as fh:
                obj = _CompatUnpickler(fh).load()
        except Exception:
            continue
        if not hasattr(obj, 'objects') or not obj.objects:
            continue
        for m in obj.objects.values():
            ts = m.get('time_stamp', [])
            n = len(ts)
            if n == 0:
                continue
            ts = np.asarray(ts, dtype=float)
            ts = ts[np.isfinite(ts)]
            if ts.size == 0:
                continue
            times = cs + pd.to_timedelta(ts / FPS, unit='s')
            in_win = (times >= t_start) & (times < t_end)
            if not in_win.any():
                continue
            amin = np.floor(((times[in_win] - t0_floor) / pd.Timedelta(minutes=1)).values).astype(int)
            a_ok = (amin >= 0) & (amin < n_abs)
            out += np.bincount(amin[a_ok], minlength=n_abs)
    return out


def load_cage_borders(yml_path):
    with open(yml_path) as f:
        y = yaml.safe_load(f)
    keys = {
        'cage':   'cage_border_points',
        'left':   'square_3_border_points',     # speaker side
        'right':  'square_4_border_points',
        'top':    'control_border_points',
        'bottom': 'sugar_border_points',
    }
    missing = [name for name, key in keys.items() if not y.get(key)]
    if missing:
        raise ValueError(
            "Cage/speaker-side borders not drawn yet for this experiment (missing: "
            f"{', '.join(missing)}). Draw them in the BuzzPhono tab's \"Cage & speaker-side "
            "borders\" section before running the full (non-resting-only) analysis.")
    return {name: np.array(y[key]) for name, key in keys.items()}


# ---------------------- plot 1: per-trajectory dist ----------------------

def plot_traj_distance(traj_df, borders, speaker_xy, out_dir, exp_name):
    # classify trajectory: ON if majority of its frames are in stim window
    on_traj  = traj_df[traj_df['frac_on'] >= 0.5]
    off_traj = traj_df[traj_df['frac_on'] <  0.5]

    if len(on_traj) < 5 or len(off_traj) < 5:
        print("  Not enough trajectories for ON vs OFF distance plot.")
        return

    # Euclidean distance of each trajectory's mean position from the custom speaker-zone centroid.
    # Drop non-finite distances: a trajectory with a NaN coordinate gap yields a NaN mean, and a
    # single NaN propagates through .max()/np.linspace and blanks the entire plot.
    def _dist(df):
        d = np.hypot(df['mean_x'].values - speaker_xy[0], df['mean_y'].values - speaker_xy[1])
        return d[np.isfinite(d)]
    on_dist  = _dist(on_traj)
    off_dist = _dist(off_traj)
    if len(on_dist) < 5 or len(off_dist) < 5:
        print("  Not enough finite-distance trajectories for ON vs OFF distance plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150,
                                    gridspec_kw={'width_ratios': [2, 1]})

    # left: histograms of distance-from-speaker-zone
    dmax = float(max(on_dist.max(), off_dist.max()))
    bins = np.linspace(0, dmax, 60)
    ax1.hist(off_dist, bins=bins, color=OFF_COLOR, alpha=0.55,
             density=True, label=f'{OFF_LABEL} (n={len(off_dist)})')
    ax1.hist(on_dist,  bins=bins, color=STIM_COLOR, alpha=0.55,
             density=True, label=f'{ON_LABEL} (n={len(on_dist)})')
    ax1.axvline(0, color='red', linestyle='--', linewidth=1.5, label='speaker zone (dist=0)')
    ax1.set_xlabel('Distance of trajectory from speaker zone (px)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Per-trajectory distance from speaker zone', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)

    # right: distance boxplot
    try:
        u_stat, p_val = mannwhitneyu(on_dist, off_dist, alternative='less')
        title = f'Distance from speaker zone\nMann-Whitney (ON<OFF): p={p_val:.3g}'
    except Exception:
        title = 'Distance from speaker zone'
    bp = ax2.boxplot([off_dist, on_dist], labels=['OFF', 'ON'],
                     patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], [OFF_COLOR, STIM_COLOR]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax2.set_ylabel('Mean distance from speaker zone (px)', fontsize=11)
    ax2.set_title(title, fontsize=11, fontweight='bold')

    fig.suptitle(f'{exp_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    for ext in ('png',):
        fig.savefig(os.path.join(out_dir, f'speaker_distance_per_trajectory.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: speaker_distance_per_trajectory.png/.pdf  "
          f"(n_on={len(on_traj)}, n_off={len(off_traj)})")


# ---------------------- plot 1b: speaker proportion (resting in custom zone) ----------------------

def plot_speaker_proportion_resting(rest_on, rest_off, speaker_poly, out_dir, exp_name):
    """Fraction of resting detections inside the custom speaker zone: OFF(30-40) vs ON(40-50).

    Computes per-window proportion and shows a bar+scatter plot with the individual
    point fractions as a sanity reference.
    """
    if len(rest_on) == 0 or len(rest_off) == 0:
        print("  No resting data for speaker proportion plot, skipping.")
        return

    speaker_path = MplPath(speaker_poly)
    in_on  = speaker_path.contains_points(rest_on)
    in_off = speaker_path.contains_points(rest_off)

    prop_on  = in_on.sum()  / max(len(rest_on),  1)
    prop_off = in_off.sum() / max(len(rest_off), 1)

    print(f"  Resting in speaker zone: OFF={prop_off:.3f}  ON={prop_on:.3f}  "
          f"(n_off={len(rest_off):,}, n_on={len(rest_on):,})")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    x       = [0, 1]
    props   = [prop_off, prop_on]
    colors  = [OFF_COLOR, STIM_COLOR]
    bars    = ax.bar(x, props, color=colors, alpha=0.75, width=0.5, zorder=3)
    for bar, v in zip(bars, props):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{OFF_LABEL} min', f'{ON_LABEL} min'], fontsize=12)
    ax.set_ylabel('Fraction resting in speaker zone', fontsize=11)
    ax.set_title(f'Speaker zone resting proportion\n{exp_name}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(props) * 1.25 + 0.01)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    for ext in ('png',):
        fig.savefig(os.path.join(out_dir, f'speaker_proportion_resting.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: speaker_proportion_resting.png/.pdf")


# ---------------------- plot 1c: minute-of-hour phase profile ----------------------

def _minute_of_hour_profile(in_zone, total, n_min_bins=60):
    """Mean +/- SEM minute-of-hour speaker-zone resting-fraction profile from
    collect_resting_zone_by_cycle's (in_zone, total) accumulators, shape (n_cycles, n_min_bins).
    Factored out of plot_speaker_phase so run_multi_speaker_comparison can draw one profile per
    experiment on a shared axis. Returns (centers, mean, sem, n_per_bin) — n_per_bin is the
    per-minute-bin valid-cycle count array (its .max() is the "n≈N cycles" figure used in
    titles/legends) — or None if there aren't enough cycles (mirrors plot_speaker_phase's own
    skip condition)."""
    bin_w = 60.0 / n_min_bins
    centers = (np.arange(n_min_bins) + 0.5) * bin_w
    with np.errstate(invalid='ignore', divide='ignore'):
        prop = np.where(total > 0, in_zone / total, np.nan)          # (n_cycles, n_min_bins)
    n_per_bin = np.sum(np.isfinite(prop), axis=0)
    if n_per_bin.max() < 2:
        return None
    mean = np.nanmean(np.where(n_per_bin > 0, prop, np.nan), axis=0)
    sem  = np.nanstd(prop, axis=0) / np.sqrt(np.maximum(n_per_bin, 1))
    return centers, mean, sem, n_per_bin


def plot_speaker_phase(in_zone, total, out_dir, exp_name, n_min_bins=60):
    """Minute-of-hour profile of the speaker-zone resting fraction, averaged across hour-cycles
    (each cycle = one replicate) with SEM. Shows the proportion rising when the stimulus turns on
    (40-50 min, shaded) and decaying afterwards — i.e. stimulus-locked recruitment, not a static
    before/after."""
    profile = _minute_of_hour_profile(in_zone, total, n_min_bins=n_min_bins)
    if profile is None:
        print("  Not enough cycles for phase plot, skipping.")
        return
    centers, mean, sem, n_per_bin = profile

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.axvspan(OFF_START_MIN, OFF_END_MIN, color=OFF_COLOR, alpha=0.10, zorder=0,
               label=f'OFF compare ({OFF_START_MIN}-{OFF_END_MIN})')
    ax.axvspan(STIM_START_MIN, STIM_END_MIN, color=STIM_COLOR, alpha=0.18, zorder=0,
               label=f'stim ON ({STIM_START_MIN}-{STIM_END_MIN})')
    ax.plot(centers, mean, color='#1f77b4', lw=2.2, zorder=3)
    ax.fill_between(centers, mean - sem, mean + sem, color='#1f77b4', alpha=0.25, zorder=2)
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Minute of hour', fontsize=12)
    ax.set_ylabel('Fraction resting in speaker zone\n(mean ± SEM across cycles)', fontsize=11)
    n_cyc = int(n_per_bin.max())
    ax.set_title(f'Speaker-zone resting — minute-of-hour profile\n{exp_name}  '
                 f'(n≈{n_cyc} cycles)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    pd.DataFrame({'minute': centers, 'mean': mean, 'sem': sem, 'n_cycles': n_per_bin}).to_csv(
        os.path.join(out_dir, 'speaker_phase_minute_of_hour.csv'), index=False)
    plt.tight_layout()
    for ext in ('png',):
        fig.savefig(os.path.join(out_dir, f'speaker_phase_minute_of_hour.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: speaker_phase_minute_of_hour.png/.pdf (+ .csv)")


def plot_speaker_phase_multi(profiles, out_path):
    """Overlay one minute-of-hour speaker-zone resting-fraction profile (mean +/- SEM line) per
    experiment/group on a shared axis — the multi-experiment analog of plot_speaker_phase.

    `profiles` is a list of `(group_label, centers, mean, sem, n_per_bin)` tuples — the same shape
    _minute_of_hour_profile returns, with a group_label prepended (skip any group whose profile
    was None before calling this). Writes out_path (PNG) + a sibling CSV with the same basename.
    """
    if not profiles:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.axvspan(OFF_START_MIN, OFF_END_MIN, color=OFF_COLOR, alpha=0.10, zorder=0,
               label=f'OFF compare ({OFF_START_MIN}-{OFF_END_MIN})')
    ax.axvspan(STIM_START_MIN, STIM_END_MIN, color=STIM_COLOR, alpha=0.18, zorder=0,
               label=f'stim ON ({STIM_START_MIN}-{STIM_END_MIN})')
    csv_rows = []
    for i, (label, centers, mean, sem, n_per_bin) in enumerate(profiles):
        color = _GROUP_PALETTE[i % len(_GROUP_PALETTE)]
        n_cyc = int(n_per_bin.max())
        ax.plot(centers, mean, color=color, lw=2.0, zorder=3, label=f'{label} (n≈{n_cyc})')
        ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.20, zorder=2)
        for c, mn, sm, n in zip(centers, mean, sem, n_per_bin):
            csv_rows.append({'group': label, 'minute': c, 'mean': mn, 'sem': sm, 'n_cycles': n})
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Minute of hour', fontsize=12)
    ax.set_ylabel('Fraction resting in speaker zone\n(mean ± SEM across cycles)', fontsize=11)
    ax.set_title('Speaker-zone resting — minute-of-hour profile (compared across experiments)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(out_path.rsplit('.', 1)[0] + '.csv', index=False)
    print(f"  Saved: {os.path.basename(out_path)} (+ .csv)")


# ---------------------- plot 1d: per-cycle paired OFF vs ON ----------------------

def plot_speaker_proportion_paired(in_zone, total, out_dir, exp_name,
                                   on_eval_start=ON_EVAL_START_MIN):
    """Per-cycle paired OFF(30-40) vs ON(40-50) speaker-zone resting proportion + Wilcoxon test.

    Each hour-cycle is one replicate, so this gives a legitimate n and paired test (vs the pooled
    bar chart, which treats every frame as independent). Also dumps speaker_proportion_by_cycle.csv.
    """
    off_in,  off_tot = in_zone[:, OFF_START_MIN:OFF_END_MIN].sum(1),  total[:, OFF_START_MIN:OFF_END_MIN].sum(1)
    on_in,   on_tot  = in_zone[:, on_eval_start:STIM_END_MIN].sum(1), total[:, on_eval_start:STIM_END_MIN].sum(1)
    valid = (off_tot > 0) & (on_tot > 0)
    if int(valid.sum()) < 3:
        print("  Not enough complete cycles for paired plot, skipping.")
        return
    p_off = off_in[valid] / off_tot[valid]
    p_on  = on_in[valid]  / on_tot[valid]
    n = int(valid.sum())

    try:
        _, pval = wilcoxon(p_on, p_off, alternative='greater')
        ptxt = f'Wilcoxon (ON>OFF): p={pval:.3g}'
    except Exception:
        ptxt = 'Wilcoxon: n/a'
    fold = float(p_on.mean() / p_off.mean()) if p_off.mean() > 0 else float('nan')

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    rng = np.random.default_rng(0)
    xoff = (rng.random(n) - 0.5) * 0.10
    xon  = 1 + (rng.random(n) - 0.5) * 0.10
    for i in range(n):
        ax.plot([xoff[i], xon[i]], [p_off[i], p_on[i]], color='0.75', lw=0.6, alpha=0.6, zorder=1)
    ax.scatter(xoff, p_off, s=24, color=OFF_COLOR, zorder=3)
    ax.scatter(xon,  p_on,  s=24, color=STIM_COLOR, zorder=3)
    ax.plot([-0.20, 0.20], [p_off.mean()] * 2, color='black', lw=2.5, zorder=4)
    ax.plot([0.80, 1.20],  [p_on.mean()]  * 2, color='black', lw=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'{OFF_LABEL} min', f'{ON_LABEL} min'], fontsize=12)
    ax.set_ylabel('Fraction resting in speaker zone (per cycle)', fontsize=11)
    ax.set_title(f'Per-cycle paired speaker proportion\n{exp_name}\n'
                 f'n={n} cycles, {fold:.2f}× — {ptxt}', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    for ext in ('png',):
        fig.savefig(os.path.join(out_dir, f'speaker_proportion_paired.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'cycle': np.where(valid)[0], 'prop_off': p_off, 'prop_on': p_on,
                  'off_total': off_tot[valid], 'on_total': on_tot[valid]}
                 ).to_csv(os.path.join(out_dir, 'speaker_proportion_by_cycle.csv'), index=False)
    print(f"  Saved: speaker_proportion_paired.png/.pdf  (n={n} cycles, {fold:.2f}x, {ptxt})")


# ---------------------- plot 1e: resting-activity time series ----------------------

def plot_resting_activity_timeseries(abs_total, abs_in, abs_fly, t0_floor, rest_dir, exp_name):
    """Whole-experiment activity time series (1-min bins), saved into the tracking_resting folder.

    Top panel: raw mean detections per frame for resting, flying and their total. The total sits
    ABOVE the 30-mosquito colony line because the split-blob tracker over-detects (one clustered
    mosquito can yield several detections) and the resting channel includes static false positives.

    Bottom panel: the same resting/flying split NORMALISED so resting + flying = 30 (colony size) at
    every minute — i.e. the share of the 30 mosquitoes that is resting vs flying over time.
    """
    n = len(abs_total)
    if n == 0 or abs_total.sum() == 0:
        print("  No resting data for activity time series, skipping.")
        return
    COLONY = 30.0
    fpm = 60.0 * FPS
    centers = t0_floor + pd.to_timedelta(np.arange(n) + 0.5, unit='m')
    r   = abs_total / fpm                          # resting detections / frame
    f   = np.asarray(abs_fly, dtype=float) / fpm   # flying detections / frame
    spk = abs_in / fpm                             # resting-in-speaker / frame
    tot = r + f
    valid = tot > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        r_n = np.where(valid, COLONY * r / tot, np.nan)   # resting share scaled to colony = 30
        f_n = np.where(valid, COLONY * f / tot, np.nan)   # flying  share scaled to colony = 30

    fig, (axr, axn) = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True)
    n_hours = int(np.ceil(n / 60.0))
    for ax in (axr, axn):
        for h in range(n_hours):
            ax.axvspan(t0_floor + pd.Timedelta(minutes=h * 60 + STIM_START_MIN),
                       t0_floor + pd.Timedelta(minutes=h * 60 + STIM_END_MIN),
                       color=STIM_COLOR, alpha=0.07, zorder=0)

    axr.plot(centers[valid], tot[valid], color='black',   lw=0.9, label='total (resting+flying)')
    axr.plot(centers[valid], r[valid],   color='#1f77b4', lw=0.8, label='resting')
    axr.plot(centers[valid], f[valid],   color='#2ca02c', lw=0.8, label='flying')
    axr.plot(centers[valid], spk[valid], color='red',     lw=0.8, alpha=0.85, label='resting in speaker zone')
    axr.axhline(COLONY, color='gray', ls='--', lw=1.2, label=f'colony size ({int(COLONY)})')
    axr.set_ylabel('Raw detections per frame', fontsize=11)
    axr.set_ylim(bottom=0)
    axr.set_title(f'Activity time series — {exp_name}\n'
                  '(orange bands = stim ON, min 40-50 each hour)', fontsize=12, fontweight='bold')
    axr.legend(fontsize=8, loc='upper right', ncol=2)
    axr.grid(True, linestyle=':', alpha=0.4)

    axn.fill_between(centers[valid], 0, r_n[valid], color='#1f77b4', alpha=0.7, label='resting')
    axn.fill_between(centers[valid], r_n[valid], COLONY, color='#2ca02c', alpha=0.55, label='flying')
    axn.axhline(COLONY, color='gray', ls='--', lw=1.0)
    axn.set_ylabel(f'Mosquitoes (normalised, resting+flying={int(COLONY)})', fontsize=11)
    axn.set_ylim(0, COLONY)
    axn.set_xlabel('Time', fontsize=12)
    axn.legend(fontsize=8, loc='upper right', ncol=2)
    axn.grid(True, linestyle=':', alpha=0.4)
    for ax in (axr, axn):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(rest_dir, 'resting_activity_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'time': centers, 'rest_per_frame': r, 'fly_per_frame': f,
                  'total_per_frame': tot, 'speaker_per_frame': spk,
                  'resting_norm30': r_n, 'flying_norm30': f_n}).to_csv(
        os.path.join(rest_dir, 'resting_activity_timeseries.csv'), index=False)
    print(f"  Saved: resting_activity_timeseries.png (+ .csv) -> {rest_dir}")


# ---------------------- plot 1f: resting speaker-side proportion over time ----------------------

def make_resting_speaker_proportion_figs(centers, prop, n_resting, out_dir, exp_name):
    """Draw the resting speaker-side proportion time series: one full-span figure PLUS one figure per
    calendar day (00:00-24:00). Stim ON windows (min 40-50 each hour) are shaded. Also writes the
    per-minute CSV. Separated out so it can be regenerated from that CSV without re-scanning PKLs."""
    centers = pd.DatetimeIndex(centers)
    prop = np.asarray(prop, dtype=float)
    valid = np.asarray(n_resting, dtype=float) > 0
    if valid.sum() == 0:
        print("  No resting data for speaker-proportion time series, skipping.")
        return

    def _draw(x0, x1, fname, sub):
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        h = x0.floor('h')
        while h < x1:
            s = max(h + pd.Timedelta(minutes=STIM_START_MIN), x0)
            e = min(h + pd.Timedelta(minutes=STIM_END_MIN), x1)
            if s < e:
                ax.axvspan(s, e, color=STIM_COLOR, alpha=0.12, zorder=0)
            h += pd.Timedelta(hours=1)
        m = valid & (centers >= x0) & (centers < x1)
        ax.plot(centers[m], prop[m], color='red', lw=0.9, zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Fraction of resting in speaker zone', fontsize=11)
        ax.set_title(f'Resting speaker-side proportion over time — {exp_name}{sub}\n'
                     '(orange bands = stim ON, min 40-50 each hour)', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if (x1 - x0) / pd.Timedelta(hours=1) <= 26:      # single-day view: hourly ticks across 0-24h
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
        else:
            fig.autofmt_xdate()
        plt.tight_layout()
        for ext in ('png',):
            fig.savefig(os.path.join(out_dir, f'{fname}.{ext}'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    cv = centers[valid]
    _draw(cv.min().floor('h'), cv.max().ceil('h'), 'resting_speaker_proportion_timeseries', '')
    n_days = 0
    for day in pd.date_range(cv.min().normalize(), cv.max().normalize(), freq='D'):
        x1 = day + pd.Timedelta(days=1)
        if not (valid & (centers >= day) & (centers < x1)).any():
            continue
        _draw(day, x1, f"resting_speaker_proportion_timeseries_{day.strftime('%Y-%m-%d')}",
              f"  ({day.strftime('%b %d')})")
        n_days += 1
    pd.DataFrame({'time': centers, 'speaker_proportion': prop, 'n_resting': n_resting}).to_csv(
        os.path.join(out_dir, 'resting_speaker_proportion_timeseries.csv'), index=False)
    print(f"  Saved: resting_speaker_proportion_timeseries.png/.pdf + {n_days} per-day fig(s) "
          f"(+ .csv) -> {out_dir}")


def plot_resting_speaker_proportion_timeseries(abs_in, abs_total, t0_floor, out_dir, exp_name):
    """Whole-experiment + per-day time series of the speaker-zone resting PROPORTION (resting in the
    speaker zone / all resting, per 1-min bin). Spikes inside each stim ON window. Delegates drawing
    to make_resting_speaker_proportion_figs (also reusable from the saved CSV)."""
    n = len(abs_total)
    if n == 0 or abs_total.sum() == 0:
        print("  No resting data for speaker-proportion time series, skipping.")
        return
    with np.errstate(invalid='ignore', divide='ignore'):
        prop = np.where(abs_total > 0, abs_in / abs_total, np.nan)
    centers = t0_floor + pd.to_timedelta(np.arange(n) + 0.5, unit='m')
    make_resting_speaker_proportion_figs(centers, prop, abs_total, out_dir, exp_name)


# ---------------------- plot 1g: resting recruitment (newly-attracted) ----------------------

def plot_recruitment_resting(in_zone, total, t0_floor, out_dir, exp_name,
                             on_eval_start=ON_EVAL_START_MIN):
    """RESTING recruitment (newly-attracted to the speaker): per hour-cycle speaker-zone resting
    proportion with that cycle's pre-stim baseline (OFF window) subtracted, by minute-of-hour.
    Overall fold + a photoperiod-phase split (dawn/day/dusk/night), as PROPORTION and as
    detections-per-frame COUNT.

    This is the resting-only counterpart to plot_custom_zones.fold_recruitment, which is computed on
    flying (tracking_moving). Built from the same (cycle x minute) resting accumulator as the phase
    and paired plots — no extra pickle pass.
    """
    n_cycles, n_min = in_zone.shape
    moh = np.arange(n_min)
    fpm = 60.0 * FPS
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(total > 0, in_zone / total, np.nan)            # (cyc, min) speaker resting frac
    off_in  = in_zone[:, OFF_START_MIN:OFF_END_MIN].sum(1)
    off_tot = total[:,  OFF_START_MIN:OFF_END_MIN].sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        base_frac = np.where(off_tot > 0, off_in / off_tot, np.nan)        # (cyc,) per-cycle baseline frac
    # np.errstate covers the float divide; catch_warnings covers np.nanmean's 'Mean of empty
    # slice' RuntimeWarning (a warnings.warn, not a float-error) for cycles with an all-NaN OFF window.
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        base_cnt = np.nanmean(np.where(total[:, OFF_START_MIN:OFF_END_MIN] > 0,
                                       in_zone[:, OFF_START_MIN:OFF_END_MIN], np.nan), axis=1)
    recruit     = frac - base_frac[:, None]
    recruit_cnt = (in_zone - base_cnt[:, None]) / fpm                  # extra detections per frame
    cyc_hour = np.array([(t0_floor + pd.Timedelta(hours=int(c))).hour for c in range(n_cycles)])
    valid = np.isfinite(base_frac)
    if int(valid.sum()) < 3:
        print("  Not enough cycles for recruitment plot, skipping.")
        return

    def _mean_sem(M):
        with np.errstate(invalid='ignore'):
            m = np.nanmean(M, axis=0)
            k = np.sum(np.isfinite(M), axis=0)
            s = np.nanstd(M, axis=0) / np.sqrt(np.maximum(k, 1))
        return m, s

    # ---- overall recruitment fold (proportion) ----
    mean, sem = _mean_sem(recruit[valid])
    on_mean = float(np.nanmean(recruit[valid][:, on_eval_start:STIM_END_MIN]))
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.axvspan(STIM_START_MIN, STIM_END_MIN, color=STIM_COLOR, alpha=0.20,
               label=f'stim ON ({STIM_START_MIN}-{STIM_END_MIN})')
    ax.axhline(0.0, color='gray', ls='--', lw=1.3, label='pre-stim baseline (0 = already there)')
    ax.plot(moh, mean, color='red', lw=2.0, label='newly attracted (mean)')
    ax.fill_between(moh, mean - sem, mean + sem, color='red', alpha=0.20)
    ax.set_xlim(0, n_min - 1); ax.set_xticks(range(0, n_min, 5))
    ax.set_xlabel('Minute of hour', fontsize=11)
    ax.set_ylabel('Newly-attracted resting proportion\n(above pre-stim baseline)', fontsize=11)
    ax.set_title(f'Resting recruitment fold — {exp_name}\n'
                 f'mean recruitment in stim window = {on_mean:+.4f}  (n={int(valid.sum())} cycles)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    for ext in ('png',):
        fig.savefig(os.path.join(out_dir, f'recruitment_fold_resting.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- phase split (proportion + count) ----
    def _phase_plot(M, ylabel, fname, fmt):
        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        ax.axvspan(STIM_START_MIN, STIM_END_MIN, color=STIM_COLOR, alpha=0.20,
                   label=f'stim ON ({STIM_START_MIN}-{STIM_END_MIN})')
        ax.axhline(0.0, color='gray', ls='--', lw=1.2)
        for name, hrs, color in PHASES:
            sel = valid & np.isin(cyc_hour, list(hrs))
            if int(sel.sum()) == 0:
                continue
            with np.errstate(invalid='ignore'):
                gm = np.nanmean(M[sel], axis=0)
                on = float(np.nanmean(M[sel][:, on_eval_start:STIM_END_MIN]))
            ax.plot(moh, gm, lw=2.0, color=color, label=f'{name}: {fmt(on)} (n={int(sel.sum())}h)')
        ax.set_xlim(0, n_min - 1); ax.set_xticks(range(0, n_min, 5))
        ax.set_xlabel('Minute of hour', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Resting recruitment by photoperiod phase — {exp_name}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left', title='mean in stim window')
        ax.grid(True, linestyle=':', alpha=0.4)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
        for ext in ('png',):
            fig.savefig(os.path.join(out_dir, f'{fname}.{ext}'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    _phase_plot(recruit, 'Newly-attracted resting proportion\n(above pre-stim baseline)',
                'recruitment_by_phase_resting_proportion', lambda v: f'{v:+.3f}')
    _phase_plot(recruit_cnt, 'Newly-attracted resting detections per frame\n(above pre-stim baseline)',
                'recruitment_by_phase_resting_count', lambda v: f'{v:+.3f}/fr')

    pd.DataFrame({'minute': moh, 'recruit_mean': mean, 'recruit_sem': sem}).to_csv(
        os.path.join(out_dir, 'recruitment_resting_fold.csv'), index=False)
    print(f"  Saved: recruitment_fold_resting.png/.pdf + by-phase (proportion & count)  "
          f"(ON recruit={on_mean:+.4f}, n={int(valid.sum())} cycles)")


# ---------------------- plot 2: 2D occupancy heatmap ----------------------

def _draw_polygons(ax, borders, speaker_poly=None):
    for name, pts in borders.items():
        closed = np.vstack([pts, pts[0:1]])
        col = 'cyan' if name == 'cage' else 'lime'
        ax.plot(closed[:, 0], closed[:, 1], color=col, linewidth=1.2, alpha=0.9)
    if speaker_poly is not None:
        closed = np.vstack([speaker_poly, speaker_poly[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], color='red', linewidth=1.8, alpha=0.95)


def plot_occupancy_heatmap(xy_on, xy_off, borders, speaker_poly, out_dir, exp_name,
                           tag='', label='all states'):
    sfx = f'_{tag}' if tag else ''
    if len(xy_on) == 0 or len(xy_off) == 0:
        print(f"  No xy data ({label}) in one of the windows, skipping heatmap.")
        return
    cage = borders['cage']
    xmin, xmax = cage[:, 0].min() - 5, cage[:, 0].max() + 5
    ymin, ymax = cage[:, 1].min() - 5, cage[:, 1].max() + 5
    bins = [np.linspace(xmin, xmax, 80), np.linspace(ymin, ymax, 80)]

    # Total points (raw counts), not density: OFF(30-40) and ON(40-50) are equal-length windows so
    # the totals are directly comparable.
    H_off, _, _ = np.histogram2d(xy_off[:, 0], xy_off[:, 1], bins=bins, density=False)
    H_on,  _, _ = np.histogram2d(xy_on[:, 0],  xy_on[:, 1],  bins=bins, density=False)

    vmax = max(H_off.max(), H_on.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    titles = [f'{OFF_LABEL} min (total pts={len(xy_off):,})',
              f'{ON_LABEL} min (total pts={len(xy_on):,})',
              'ON − OFF (count diff)']
    diffs = H_on - H_off
    dmax = np.abs(diffs).max() or 1.0

    for ax, H, title, cmap, args in zip(
            axes,
            [H_off.T, H_on.T, diffs.T],
            titles,
            ['inferno', 'inferno', 'RdBu_r'],
            [{'vmin': 0, 'vmax': vmax},
             {'vmin': 0, 'vmax': vmax},
             {'vmin': -dmax, 'vmax': dmax}]):
        im = ax.imshow(H, origin='upper', extent=[xmin, xmax, ymax, ymin],
                       cmap=cmap, aspect='equal', **args)
        _draw_polygons(ax, borders, speaker_poly)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
        ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    fig.suptitle(f'2D occupancy ({label}) — {exp_name}  (red = speaker zone)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('png',):
        fig.savefig(os.path.join(out_dir, f'occupancy_heatmap_on_vs_off{sfx}.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: occupancy_heatmap_on_vs_off{sfx}.png/.pdf")


# ---------------------- main ----------------------

def _load_speaker_zone(zones_path, zone_name):
    """Return (centroid_xy, polygon) for the custom speaker zone in custom_zones.json.

    Raises ValueError if the requested zone is missing (rather than sys.exit, which would
    terminate the whole process when called in-process from the GUI)."""
    import json
    with open(zones_path) as f:
        zones = json.load(f)
    if zone_name not in zones:
        raise ValueError(f"zone '{zone_name}' not in {zones_path} (have: {list(zones)})")
    zd = zones[zone_name]
    poly = np.array(zd['polygon'] if isinstance(zd, dict) else zd, dtype=float)
    return poly.mean(axis=0), poly


def _list_zone_names(zones_path):
    """All zone names defined in custom_zones.json (e.g. ['speaker', 'sugar']), or [] if the file
    is missing/unreadable."""
    import json
    if not zones_path or not os.path.isfile(zones_path):
        return []
    try:
        with open(zones_path) as f:
            zones = json.load(f) or {}
    except Exception:
        return []
    return list(zones.keys())


def plot_zone_fraction_comparison(zone_cycle_data, out_dir, exp_name, on_eval_start=ON_EVAL_START_MIN):
    """Compare the stim-ON resting fraction across multiple named zones (e.g. the speaker zone vs
    an equally-sized comparison region like a sugar feeder) as a bar chart with per-cycle points.

    zone_cycle_data: {zone_name: (in_cyc, tot_cyc)} — each zone's own per-cycle in-zone/total
    resting counts from collect_resting_zone_by_cycle, sharing the same (cycle, minute) binning.
    Needs >=2 zones with at least one valid cycle; otherwise there's nothing to compare and the
    plot is skipped.
    """
    names, means, sems, ns = [], [], [], []
    per_zone_points = {}
    for name, (in_cyc, tot_cyc) in zone_cycle_data.items():
        on_in  = in_cyc[:, on_eval_start:STIM_END_MIN].sum(1)
        on_tot = tot_cyc[:, on_eval_start:STIM_END_MIN].sum(1)
        valid = on_tot > 0
        if not valid.any():
            print(f"  Zone '{name}': no complete ON-window cycles, excluding from comparison.")
            continue
        p = on_in[valid] / on_tot[valid]
        names.append(name)
        means.append(float(p.mean()))
        sems.append(float(p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 else 0.0)
        ns.append(len(p))
        per_zone_points[name] = p

    if len(names) < 2:
        print("  Fewer than 2 zones have usable data, skipping zone comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 5), dpi=150)
    x = np.arange(len(names))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 2)))
    ax.bar(x, means, yerr=sems, capsize=4, color=colors[:len(names)], alpha=0.85, zorder=2)
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        p = per_zone_points[name]
        jitter = (rng.random(len(p)) - 0.5) * 0.25
        ax.scatter(np.full(len(p), i) + jitter, p, s=14, color='black', alpha=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(n={c} cycles)' for n, c in zip(names, ns)], fontsize=10)
    ax.set_ylabel(f'Fraction resting in zone\n(stim ON, min {on_eval_start}-{STIM_END_MIN}, per cycle)',
                  fontsize=10)
    ax.set_title(f'Zone comparison — {exp_name}', fontsize=12, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'zone_fraction_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: zone_fraction_comparison.png ({', '.join(names)})")


def _phono_outputs_current(out_dir, exp_dir):
    """True if the speaker-zone plots already exist and are all newer than the newest tracking
    file (so a re-run would reproduce the same figures). Used to skip regeneration."""
    if not os.path.isdir(out_dir):
        return False
    pngs = [f for f in os.listdir(out_dir) if f.lower().endswith('.png') and not f.startswith('.')]
    if not pngs:
        return False
    newest_out = max(os.path.getmtime(os.path.join(out_dir, f)) for f in pngs)
    newest_track = 0.0
    for sub in ('final_tracking_data', 'tracking_resting', 'tracking_moving'):
        d = os.path.join(exp_dir, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if not f.startswith('.'):
                    try:
                        newest_track = max(newest_track, os.path.getmtime(os.path.join(d, f)))
                    except OSError:
                        pass
    return newest_out >= newest_track


def run_speaker_distance_analysis(experiment_dir, start, end, output_dir=None,
                                  custom_zones_path=None, speaker_zone='speaker',
                                  resting_only=False, on_warmup_trim=ON_WARMUP_TRIM,
                                  force_replot=False):
    """Programmatic entry point for the speaker-distance (custom-zones) phonotaxis analysis.

    Extracted verbatim from main() so the GUI can call it directly (no argparse / no sys.exit).
    Runs the resting (cycle x minute) pass and its five plots and — unless resting_only — the
    trajectory + occupancy-heatmap pass. Output defaults to <experiment_dir>/plots/custom_zones/
    (the documented BuzzPhono output location). With force_replot=False (default) it skips the
    whole run when the output plots already exist and are newer than the tracking data. Returns
    the output directory."""
    exp_dir  = os.path.abspath(experiment_dir)
    out_dir  = output_dir or os.path.join(exp_dir, 'plots', 'custom_zones')
    exp_name = os.path.basename(exp_dir)
    t_start  = pd.Timestamp(start)
    t_end    = pd.Timestamp(end)
    os.makedirs(out_dir, exist_ok=True)

    if not force_replot and _phono_outputs_current(out_dir, exp_dir):
        print(f"BuzzPhono outputs already up to date in {out_dir} "
              f"(pass force_replot=True to regenerate).")
        return out_dir

    print(f"Experiment : {exp_name}")
    print(f"Window     : {t_start} -> {t_end}")
    print(f"Output dir : {out_dir}")

    zones_path = custom_zones_path or os.path.join(exp_dir, 'custom_zones.json')
    speaker_xy, speaker_poly = _load_speaker_zone(zones_path, speaker_zone)
    print(f"Speaker zone '{speaker_zone}' centroid ~ ({speaker_xy[0]:.0f}, {speaker_xy[1]:.0f}) px  [custom_zones]")

    # The (cycle x minute) resting accumulator feeds the phase, paired, recruitment and time-series
    # plots from one resting pass — all that resting_only needs.
    print("\nPer-cycle speaker-zone proportion (resting; hour = independent replicate)...")
    rest_dir = os.path.join(exp_dir, 'tracking_resting')
    in_cyc, tot_cyc, t0_floor, abs_in, abs_tot = collect_resting_zone_by_cycle(
        rest_dir, t_start, t_end, speaker_poly)

    print("\nFlying per-minute totals (for the activity plot resting+flying=colony)...")
    abs_fly = collect_abs_minute_total(
        os.path.join(exp_dir, 'tracking_moving'), t_start, t_end, t0_floor, len(abs_tot))

    on_eval_start = STIM_START_MIN + on_warmup_trim

    print("\nGenerating resting-derived plots...")
    plot_speaker_phase(in_cyc, tot_cyc, out_dir, exp_name)
    plot_speaker_proportion_paired(in_cyc, tot_cyc, out_dir, exp_name,
                                   on_eval_start=on_eval_start)
    plot_recruitment_resting(in_cyc, tot_cyc, t0_floor, out_dir, exp_name,
                             on_eval_start=on_eval_start)
    # Route this into the output plots dir too (was tracking_resting/) so ALL BuzzPhono outputs
    # land under plots/custom_zones/ per the unified plots/ layout.
    plot_resting_activity_timeseries(abs_tot, abs_in, abs_fly, t0_floor, out_dir, exp_name)
    plot_resting_speaker_proportion_timeseries(abs_in, abs_tot, t0_floor, out_dir, exp_name)

    # Comparison zones (e.g. an equally-sized sugar-feeder region, added via BuzzPhono's "Add
    # comparison zone (same shape)…") — any other zone defined in custom_zones.json besides
    # speaker_zone gets its own resting-fraction pass so the plot can compare them directly.
    comparison_names = [z for z in _list_zone_names(zones_path) if z != speaker_zone]
    zone_cycle_data = {speaker_zone: (in_cyc, tot_cyc)}
    for other_name in comparison_names:
        try:
            _, other_poly = _load_speaker_zone(zones_path, other_name)
            other_in_cyc, other_tot_cyc, _, _, _ = collect_resting_zone_by_cycle(
                rest_dir, t_start, t_end, other_poly)
            zone_cycle_data[other_name] = (other_in_cyc, other_tot_cyc)
        except Exception as exc:
            print(f"  Could not compute comparison zone '{other_name}': {exc}")
    if len(zone_cycle_data) > 1:
        plot_zone_fraction_comparison(zone_cycle_data, out_dir, exp_name, on_eval_start=on_eval_start)

    if resting_only:
        print(f"\nDone (resting-only). Plots in: {out_dir}  (+ resting series in {rest_dir})")
        return out_dir

    # ---- full analysis: trajectories, per-trajectory distance, bar chart, occupancy heatmaps ----
    borders = load_cage_borders(os.path.join(exp_dir, 'buzzwatch_track_settings.yml'))

    print("\nLoading trajectories...")
    traj_df, xy_on, xy_off = trajectory_metrics(
        os.path.join(exp_dir, 'final_tracking_data'), t_start, t_end)
    print(f"  total trajectories in window: {len(traj_df)}")
    print(f"  ON-frame points : {len(xy_on):,}")
    print(f"  OFF-frame points: {len(xy_off):,}")

    print("\nResting-only / flying-only occupancy heatmaps...")
    rest_on, rest_off = collect_xy_by_state(os.path.join(exp_dir, 'tracking_resting'), t_start, t_end)
    fly_on,  fly_off  = collect_xy_by_state(os.path.join(exp_dir, 'tracking_moving'),  t_start, t_end)
    print(f"  resting ON/OFF: {len(rest_on):,}/{len(rest_off):,}   flying ON/OFF: {len(fly_on):,}/{len(fly_off):,}")

    # Total = resting + flying from state folders; should peak ~30 (full population)
    total_on  = np.concatenate([rest_on,  fly_on],  axis=0) if (len(rest_on)  or len(fly_on))  else np.empty((0, 2))
    total_off = np.concatenate([rest_off, fly_off], axis=0) if (len(rest_off) or len(fly_off)) else np.empty((0, 2))
    print(f"  total  ON/OFF: {len(total_on):,}/{len(total_off):,}")

    plot_traj_distance(traj_df, borders, speaker_xy, out_dir, exp_name)
    plot_speaker_proportion_resting(rest_on, rest_off, speaker_poly, out_dir, exp_name)
    plot_occupancy_heatmap(total_on, total_off, borders, speaker_poly, out_dir, exp_name,
                           tag='total', label='total points (resting + flying)')
    plot_occupancy_heatmap(rest_on, rest_off, borders, speaker_poly, out_dir, exp_name,
                           tag='resting', label='resting only')
    plot_occupancy_heatmap(fly_on,  fly_off,  borders, speaker_poly, out_dir, exp_name,
                           tag='flying',  label='flying only')

    # save raw per-trajectory CSV for any downstream stats
    traj_df.to_csv(os.path.join(out_dir, 'trajectories_summary.csv'), index=False)
    print(f"\nDone. Plots + CSV in: {out_dir}")
    return out_dir


def run_multi_speaker_comparison(experiments, output_dir=None, speaker_zone='speaker'):
    """Compares the speaker-zone minute-of-hour resting response across MULTIPLE experiments as
    one overlaid plot (plot_speaker_phase_multi).

    `experiments` is a list of `(experiment_dir, group_label, start, end)` tuples. This does NOT
    replace each experiment's own full output — callers should still call
    run_speaker_distance_analysis once per experiment for that (this only adds the comparison
    plot on top). Reads the speaker zone from each experiment's own custom_zones.json. ON/OFF
    window shading uses the module's current STIM_START_MIN/STIM_END_MIN/OFF_START_MIN/
    OFF_END_MIN globals, exactly like run_speaker_distance_analysis/plot_speaker_phase — set them
    via setattr before calling this, the same pattern the GUI already uses for the
    single-experiment path. Returns the output directory."""
    if not experiments:
        raise ValueError('run_multi_speaker_comparison requires at least one experiment')
    if output_dir is None:
        output_dir = os.path.join(experiments[0][0], 'plots', 'buzzphono', 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    profiles = []
    for experiment_dir, group_label, start, end in experiments:
        exp_dir = os.path.abspath(experiment_dir)
        zones_path = os.path.join(exp_dir, 'custom_zones.json')
        try:
            _, speaker_poly = _load_speaker_zone(zones_path, speaker_zone)
        except Exception as exc:
            print(f'  skipped (no speaker zone): {group_label}: {exc}')
            continue
        t_start = pd.Timestamp(start)
        t_end = pd.Timestamp(end)
        rest_dir = os.path.join(exp_dir, 'tracking_resting')
        in_cyc, tot_cyc, _t0_floor, _abs_in, _abs_tot = collect_resting_zone_by_cycle(
            rest_dir, t_start, t_end, speaker_poly)
        profile = _minute_of_hour_profile(in_cyc, tot_cyc)
        if profile is None:
            print(f'  skipped (not enough cycles): {group_label}')
            continue
        centers, mean, sem, n_per_bin = profile
        profiles.append((group_label, centers, mean, sem, n_per_bin))

    if not profiles:
        print('No experiment had enough data for the comparison plot.')
        return output_dir

    out_path = os.path.join(output_dir, 'speaker_phase_minute_of_hour_comparison.png')
    plot_speaker_phase_multi(profiles, out_path)
    print(f'  wrote -> {output_dir}')
    return output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cage', choices=list(_CAGE_CFG),
                   help='cage shorthand (e.g. 6_4-5); auto-resolves --experiment-dir/--start/--end/--output-dir')
    p.add_argument('--experiment-dir', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--start', default=None, help='ISO timestamp')
    p.add_argument('--end',   default=None, help='ISO timestamp')
    p.add_argument('--custom-zones', default=None,
                   help='path to custom_zones.json (default: <experiment-dir>/custom_zones.json)')
    p.add_argument('--speaker-zone', default='speaker',
                   help="zone name in custom_zones.json used as the speaker reference (default 'speaker')")
    p.add_argument('--resting-only', action='store_true',
                   help='only the resting (cycle x minute) pass + resting plots (phase, paired, '
                        'recruitment, both time series); skips the trajectory load and heatmaps')
    args = p.parse_args()

    on_warmup_trim = ON_WARMUP_TRIM
    if args.cage:
        cage_exp, cage_out, cage_start, cage_end = _resolve_cage(args.cage)
        args.experiment_dir = args.experiment_dir or cage_exp
        args.output_dir     = args.output_dir     or cage_out
        args.start          = args.start          or cage_start
        args.end            = args.end            or cage_end
        on_warmup_trim = _CAGE_CFG[args.cage].get('on_warmup_trim', ON_WARMUP_TRIM)
    else:
        missing = [f'--{k}' for k, v in [('experiment-dir', args.experiment_dir),
                                           ('start', args.start), ('end', args.end)] if v is None]
        if missing:
            p.error(f"required without --cage: {', '.join(missing)}")

    run_speaker_distance_analysis(
        args.experiment_dir, args.start, args.end,
        output_dir=args.output_dir, custom_zones_path=args.custom_zones,
        speaker_zone=args.speaker_zone, resting_only=args.resting_only,
        on_warmup_trim=on_warmup_trim)


if __name__ == '__main__':
    main()
