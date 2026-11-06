#!/usr/bin/env python3
"""
Per-day phonotaxis plots for the OVERHAUL tracking data.

First run: analyze tracking_moving/ + tracking_resting/ PKLs (OVERHAUL folder ONLY),
build a GUI-matched 1-minute master CSV, then plot. Subsequent runs: load the master
CSV and replot without touching the PKLs.

Activity y-axis = mean number of mosquitoes per frame (the BuzzSuite GUI scale: per-frame
object count -> resample().mean()) -- NOT a per-hour detection sum.
Speaker-side proportion computed for flying / resting / combined (chance = 0.25).

The master CSV stores per-frame-mean counts for ALL FOUR sides (left/right/top/bottom),
so the speaker side is a plot-time choice (--speaker-side, default 'left') and switching
sides needs NO re-analysis.

Windows are 04:00 -> 04:00: every full day in the data + a combined span.
Combined span is an overlay plot: each calendar day as its own line on a shared 04:00->04:00 x-axis.

IMPORTANT: sources STRICTLY from <exp>_overhaul. It never reads the original
AedesAegyptiLiv475M_<exp>/analyzed_data.pkl, and does not use the _find_pkl/auto loaders.
"""

import os
import sys
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.path import Path as MplPath

# ---- constants -------------------------------------------------------------
FPS               = 25.0
VIDEO_DURATION_S  = 1202     # seconds per video chunk
STIM_START_MIN    = 40       # 475 Hz stim window (min past each hour)
STIM_END_MIN      = 50
STIM_COLOR        = '#ff8c00'
STIM_ALPHA        = 0.30
NIGHT_COLOR       = '#C8C8C8'
NIGHT_ALPHA       = 0.30
CHANCE            = 0.25     # 1 of 4 edge zones
ZT0_HOUR          = 5        # lights ON 05:00
DAY_START_HOUR    = 4        # plotting windows start at 04:00
RESAMPLE_DEFAULT  = '10min'
# Candidate data-drive mount points to search for the Buzzwatch/ tree under. Set
# BUZZSUITE_DATA_ROOT to point at your own data drive instead of editing this list.
DRIVE_ROOTS       = [p for p in [os.environ.get('BUZZSUITE_DATA_ROOT')] if p] + \
                     ["/Volumes/Mosquito2", "E:/", "e:/", "/e"]
# Where per-experiment result PNGs get copied to. Override with BUZZSUITE_RESULTS_ROOT.
RESULTS_ROOT      = os.environ.get('BUZZSUITE_RESULTS_ROOT', r"E:\Phonotaxis results")

SIDE_KEYS = {
    'left':   'sugar_border_points',     # SPEAKER side (default)
    'right':  'square_3_border_points',
    'top':    'square_4_border_points',
    'bottom': 'control_border_points',
}
SPEAKER_SIDE_DEFAULT = 'left'
SIDE_OUTLINE = {'left': 'magenta', 'right': 'orange', 'top': 'lime', 'bottom': 'yellow'}

ACT_COLORS = {'flying': '#1a6faf', 'resting': '#8a6d3b', 'combined': '#2ca02c'}
SPK_COLORS = {'flying': '#b22222', 'resting': '#1a6faf', 'combined': '#2ca02c'}


def _round_up(v, step=10):
    return step if (not np.isfinite(v) or v <= 0) else int(step * np.ceil(v / step))


class _Compat(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


# ---- paths -----------------------------------------------------------------
def find_overhaul_folder(exp):
    rel = f"Buzzwatch/Analysis/Phonotaxis/Penzance/AedesAegyptiLiv475M_{exp}_overhaul"
    for root in DRIVE_ROOTS:
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            return p
    sys.exit(f"Overhaul folder not found for exp {exp}")


def ts_from_fname(fname):
    parts = os.path.basename(fname).replace('.pkl', '').split('_')
    d, t, vid = parts[-3], parts[-2], int(parts[-1])
    session_start = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}")
    return session_start + pd.Timedelta(seconds=vid * VIDEO_DURATION_S)


# ---- analysis (first run only) --------------------------------------------
def _state_series(track_dir, side_paths):
    files = sorted(glob.glob(os.path.join(track_dir, '*.pkl')))
    print(f"  {len(files)} files in {os.path.basename(track_dir)}/", flush=True)
    tot_list = []
    side_lists = {s: [] for s in side_paths}
    for i, f in enumerate(files):
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(files)}", flush=True)
        try:
            video_start = ts_from_fname(f)
            with open(f, 'rb') as fh:
                tracks = _Compat(fh).load()
            frames, coords = [], []
            if hasattr(tracks, 'objects') and tracks.objects:
                for td in tracks.objects.values():
                    ts_ = td.get('time_stamp', [])
                    co_ = td.get('coordinates', [])
                    n = min(len(ts_), len(co_))
                    if n:
                        frames.append(np.asarray(ts_[:n], dtype=np.int64))
                        coords.append(np.asarray(co_[:n], dtype=float))
            del tracks
            if not frames:
                continue
            frames = np.concatenate(frames)
            coords = np.concatenate(coords)
            maxf = int(frames.max())
            idx = video_start + pd.to_timedelta(np.arange(maxf + 1) / FPS, unit='s')
            tot = np.bincount(frames, minlength=maxf + 1).astype(float)
            tot_list.append(pd.Series(tot, index=idx).resample('1s').mean())
            for side, path in side_paths.items():
                ins = path.contains_points(coords)
                cnt = np.bincount(frames[ins], minlength=maxf + 1).astype(float)
                side_lists[side].append(pd.Series(cnt, index=idx).resample('1s').mean())
        except Exception as e:
            print(f"    WARN {os.path.basename(f)}: {e}", flush=True)

    def _join(lst):
        if not lst:
            return pd.Series(dtype=float)
        s = pd.concat(lst).sort_index()
        return s[~s.index.duplicated(keep='first')]

    return _join(tot_list), {s: _join(side_lists[s]) for s in side_paths}


def analyze(overhaul_folder, exp):
    print(f"[analyze] OVERHAUL source: {overhaul_folder}", flush=True)
    assert '_overhaul' in overhaul_folder
    with open(os.path.join(overhaul_folder, 'buzzwatch_track_settings.yml')) as f:
        settings = yaml.safe_load(f)
    side_paths = {side: MplPath(np.asarray(settings[key], dtype=float))
                  for side, key in SIDE_KEYS.items() if settings.get(key)}
    print(f"  zones: " + ", ".join(f"{s}={SIDE_KEYS[s]}" for s in side_paths), flush=True)

    fly_tot, fly_sides   = _state_series(os.path.join(overhaul_folder, 'tracking_moving'),  side_paths)
    rest_tot, rest_sides = _state_series(os.path.join(overhaul_folder, 'tracking_resting'), side_paths)

    def to_min(s):
        return s.resample('1min').mean() if len(s) else s

    fly_tot_m, rest_tot_m = to_min(fly_tot), to_min(rest_tot)
    idx = fly_tot_m.index.union(rest_tot_m.index)

    df = pd.DataFrame(index=idx)
    df.index.name = 'timestamp'
    df['numb_mosquitos_flying']  = fly_tot_m.reindex(idx)
    df['numb_mosquitos_resting'] = rest_tot_m.reindex(idx)
    for side in side_paths:
        df[f'side_flying_{side}']  = to_min(fly_sides[side]).reindex(idx)
        df[f'side_resting_{side}'] = to_min(rest_sides[side]).reindex(idx)
    return df


# ---- speaker fractions / bonus pkl ----------------------------------------
def add_speaker_fractions(df, speaker_side):
    fly  = df['numb_mosquitos_flying']
    rest = df['numb_mosquitos_resting']
    fspk = df[f'side_flying_{speaker_side}']
    rspk = df[f'side_resting_{speaker_side}']
    out = df.copy()
    out['frac_flying_speaker']  = (fspk / fly).where(fly > 0)
    out['frac_resting_speaker'] = (rspk / rest).where(rest > 0)
    comb_tot = fly.fillna(0) + rest.fillna(0)
    comb_spk = fspk.fillna(0) + rspk.fillna(0)
    out['frac_combined_speaker'] = (comb_spk / comb_tot).where(comb_tot > 0)
    out['n_flying']  = fly
    out['n_resting'] = rest
    return out


def write_bonus_pkl(folder, df, speaker_side):
    try:
        others = [s for s in SIDE_KEYS if s != speaker_side and f'side_flying_{s}' in df.columns]
        pop = df[['numb_mosquitos_flying', 'numb_mosquitos_resting']].copy()
        side = pd.DataFrame(index=df.index)
        side['side_flying_speaker']  = df.get(f'side_flying_{speaker_side}', 0.0)
        side['side_resting_speaker'] = df.get(f'side_resting_{speaker_side}', 0.0)
        for i in range(1, 4):
            o = others[i - 1] if i - 1 < len(others) else None
            side[f'side_flying_non_speaker_{i}']  = df[f'side_flying_{o}']  if o else 0.0
            side[f'side_resting_non_speaker_{i}'] = df[f'side_resting_{o}'] if o else 0.0
        with open(os.path.join(folder, 'analyzed_data.pkl'), 'wb') as fh:
            pickle.dump({'population_data': pop, 'speaker_side_data': side}, fh)
        print(f"  saved bonus: {os.path.join(folder, 'analyzed_data.pkl')}", flush=True)
    except Exception as e:
        print(f"  (bonus pkl skipped: {e})", flush=True)


# ---- shading helpers (datetime x-axis) ------------------------------------
def _add_night_shading(ax, t_min, t_max):
    day, zt12 = pd.Timedelta(hours=24), pd.Timedelta(hours=12)
    first = t_min.normalize() + pd.Timedelta(hours=ZT0_HOUR)
    if first > t_min:
        first -= day
    cur = first
    while cur < t_max:
        ns = max(cur + zt12, t_min)
        ne = min(cur + day, t_max)
        if ns < ne:
            ax.axvspan(ns, ne, color=NIGHT_COLOR, alpha=NIGHT_ALPHA, linewidth=0)
        cur += day


def _add_stim_shading(ax, t_min, t_max):
    cur = t_min.floor('h')
    one = pd.Timedelta(hours=1)
    while cur < t_max:
        s = max(cur + pd.Timedelta(minutes=STIM_START_MIN), t_min)
        e = min(cur + pd.Timedelta(minutes=STIM_END_MIN), t_max)
        if s < e:
            ax.axvspan(s, e, color=STIM_COLOR, alpha=STIM_ALPHA, linewidth=0)
        cur += one


def _hourly_axis(ax, t_min, t_max, hours):
    interval = 1 if hours <= 24 else 2
    ax.set_xlim(t_min, t_max)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    day = pd.Timedelta(hours=24)
    b = t_min.normalize() + pd.Timedelta(hours=DAY_START_HOUR)
    if b < t_min:
        b += day
    while b <= t_max:
        ax.axvline(b, color='k', lw=0.8, ls=':', alpha=0.5)
        ax.annotate(b.strftime('%b %d'), xy=(b, 1), xycoords=('data', 'axes fraction'),
                    xytext=(3, -10), textcoords='offset points', fontsize=8,
                    color='k', ha='left', va='top')
        b += day


def _legend_patches():
    return [mpatches.Patch(color=NIGHT_COLOR, alpha=0.5, label='Night (17:00-05:00)'),
            mpatches.Patch(color=STIM_COLOR, alpha=STIM_ALPHA, label='475 Hz stim (min 40-50)')]


# ---- shading helpers (float-hour x-axis, used by overlay plots) -----------
def _add_night_shading_overlay(ax):
    # lights off 17:00 = 13h from 04:00; lights on 05:00 = 1h from 04:00
    lights_off_h = 17 - DAY_START_HOUR  # 13
    lights_on_h  =  5 - DAY_START_HOUR  # 1
    ax.axvspan(lights_off_h, 24, color=NIGHT_COLOR, alpha=NIGHT_ALPHA, linewidth=0)
    ax.axvspan(0, lights_on_h,  color=NIGHT_COLOR, alpha=NIGHT_ALPHA, linewidth=0)


def _add_stim_shading_overlay(ax):
    for h in range(24):
        ax.axvspan(h + STIM_START_MIN / 60, h + STIM_END_MIN / 60,
                   color=STIM_COLOR, alpha=STIM_ALPHA, linewidth=0)


def _overlay_time_axis(ax):
    """Float-hour x-axis 0..24 with clock-time tick labels starting at 04:00."""
    ax.set_xlim(0, 24)
    ticks = np.arange(0, 25)
    labels = [f"{(DAY_START_HOUR + int(h)) % 24:02d}:00" for h in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)


def _float_hours(series, day_start):
    """Float hours since day_start for each timestamp in series."""
    return np.array((series.index - day_start).total_seconds() / 3600)


# ---- per-day plots (datetime x-axis) --------------------------------------
def plot_activity(series, t_min, t_max, hours, kind, ymax, out_dir, exp, win_label, win_title):
    color = ACT_COLORS[kind]
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    _add_night_shading(ax, t_min, t_max)
    _add_stim_shading(ax, t_min, t_max)
    ax.plot(series.index, series.values, '-', lw=1.0, color=color, label=f'{kind} activity')
    ax.fill_between(series.index, 0, series.values, color=color, alpha=0.15)
    ax.set_ylim(0, ymax)
    _hourly_axis(ax, t_min, t_max, hours)
    ax.set_ylabel('Mean number of mosquitoes', fontsize=11)
    ax.set_xlabel('Clock time', fontsize=11)
    ax.set_title(f'Activity ({kind}) - {win_title}\n{exp} (overhaul)', fontsize=12, fontweight='bold')
    ax.legend(handles=[ax.lines[0]] + _legend_patches(), fontsize=9, loc='upper right')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'activity_{kind}_{win_label}.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_speaker(series, t_min, t_max, hours, kind, speaker_side, out_dir, exp, win_title, win_label):
    color = SPK_COLORS[kind]
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    _add_night_shading(ax, t_min, t_max)
    _add_stim_shading(ax, t_min, t_max)
    ax.axhline(CHANCE, color='gray', ls='--', lw=1.2, label=f'Chance ({CHANCE:.2f})')
    ax.plot(series.index, series.values, '-', lw=1.1, color=color, label=f'{kind} on speaker side')
    ax.set_ylim(0, 1)
    _hourly_axis(ax, t_min, t_max, hours)
    ax.set_ylabel(f'Fraction on speaker side ({speaker_side})', fontsize=11)
    ax.set_xlabel('Clock time', fontsize=11)
    ax.set_title(f'Speaker-side proportion ({kind}, speaker={speaker_side}) - {win_title}\n{exp} (overhaul)',
                 fontsize=12, fontweight='bold')
    ax.legend(handles=[ax.lines[0], ax.lines[1]] + _legend_patches(), fontsize=9, loc='upper right')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'speaker_{kind}_{win_label}.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---- overlay plots (float-hour x-axis, one line per calendar day) ---------
def plot_activity_overlay(day_series, kind, out_dir, exp, win_label, win_title):
    """Each calendar day as a separate line on a shared 04:00->04:00 axis."""
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    _add_night_shading_overlay(ax)
    _add_stim_shading_overlay(ax)

    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(day_series), 1)))
    lines = []
    all_vals = []
    for i, (day_label, (day_start, series)) in enumerate(day_series.items()):
        x = _float_hours(series, day_start)
        mask = (x >= 0) & (x <= 24)
        if not mask.any():
            continue
        vals = series.values[mask]
        l, = ax.plot(x[mask], vals, '-', lw=1.3, color=palette[i], label=day_label)
        lines.append(l)
        all_vals.extend(vals[np.isfinite(vals)])

    ymax = _round_up(max(all_vals) if all_vals else 1)
    ax.set_ylim(0, ymax)
    _overlay_time_axis(ax)
    ax.set_ylabel('Mean number of mosquitoes', fontsize=11)
    ax.set_xlabel('Clock time (04:00 → 04:00)', fontsize=11)
    ax.set_title(f'Activity ({kind}) - {win_title}\n{exp} (overhaul)', fontsize=12, fontweight='bold')
    ax.legend(handles=lines + _legend_patches(), fontsize=9, loc='upper right')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'activity_{kind}_{win_label}.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_speaker_overlay(day_series, kind, speaker_side, out_dir, exp, win_label, win_title):
    """Each calendar day as a separate line on a shared 04:00->04:00 axis."""
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    _add_night_shading_overlay(ax)
    _add_stim_shading_overlay(ax)
    chance_line = ax.axhline(CHANCE, color='gray', ls='--', lw=1.2, label=f'Chance ({CHANCE:.2f})')

    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(day_series), 1)))
    lines = [chance_line]
    for i, (day_label, (day_start, series)) in enumerate(day_series.items()):
        x = _float_hours(series, day_start)
        mask = (x >= 0) & (x <= 24)
        if not mask.any():
            continue
        l, = ax.plot(x[mask], series.values[mask], '-', lw=1.3, color=palette[i], label=day_label)
        lines.append(l)

    ax.set_ylim(0, 1)
    _overlay_time_axis(ax)
    ax.set_ylabel(f'Fraction on speaker side ({speaker_side})', fontsize=11)
    ax.set_xlabel('Clock time (04:00 → 04:00)', fontsize=11)
    ax.set_title(f'Speaker-side proportion ({kind}, speaker={speaker_side}) - {win_title}\n{exp} (overhaul)',
                 fontsize=12, fontweight='bold')
    ax.legend(handles=lines + _legend_patches(), fontsize=9, loc='upper right')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'speaker_{kind}_{win_label}.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---- speaker-zone QC -------------------------------------------------------
def plot_speaker_zone(overhaul_folder, settings, out_dir, exp, speaker_side):
    speaker_key = SIDE_KEYS.get(speaker_side)
    spk = settings.get(speaker_key)
    if not spk:
        print(f"  (speaker_zone.png skipped: no polygon for {speaker_side})", flush=True)
        return
    imgs = sorted(glob.glob(os.path.join(overhaul_folder, 'individual_images', '*.png')))
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    w, h = 1280, 720
    if imgs:
        try:
            img = plt.imread(imgs[0])
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            h, w = img.shape[0], img.shape[1]
        except Exception as e:
            print(f"  (frame load failed: {e})", flush=True)
    for lbl, col, key in [('cage', 'cyan', 'cage_border_points'),
                          ('center', 'white', 'center_border_points')]:
        pts = settings.get(key)
        if pts:
            p = np.array(pts + [pts[0]], dtype=float)
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.5, label=lbl)
    for side, key in SIDE_KEYS.items():
        if side == speaker_side:
            continue
        pts = settings.get(key)
        if pts:
            p = np.array(pts + [pts[0]], dtype=float)
            ax.plot(p[:, 0], p[:, 1], color=SIDE_OUTLINE.get(side, 'white'), lw=1.5,
                    label=f'{side} ({key})')
    sp = np.array(spk, dtype=float)
    ax.add_patch(mpatches.Polygon(sp, closed=True, facecolor='red', alpha=0.30, edgecolor='red', lw=2.5))
    spc = np.array(spk + [spk[0]], dtype=float)
    ax.plot(spc[:, 0], spc[:, 1], color='red', lw=2.5,
            label=f'SPEAKER side ({speaker_side}, {speaker_key})')
    ax.text(sp[:, 0].mean(), sp[:, 1].mean(), 'SPEAKER\nSIDE', color='red',
            fontsize=13, fontweight='bold', ha='center', va='center')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f'Speaker-side zone ({speaker_side}) - {exp} (overhaul)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'speaker_zone.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: speaker_zone.png (speaker={speaker_side})", flush=True)


# ---- windows ---------------------------------------------------------------
def build_windows(min_ts, max_ts):
    day = pd.Timedelta(hours=24)
    d4  = pd.Timedelta(hours=DAY_START_HOUR)
    first = min_ts.normalize() + d4
    while first < min_ts:
        first += day
    last_b = max_ts.normalize() + d4
    while last_b > max_ts:
        last_b -= day

    wins = []
    s = first
    while s + day <= last_b + pd.Timedelta(seconds=1):
        e = s + day
        wins.append((f"day_{s.strftime('%b%d')}_{e.strftime('%b%d')}",
                     f"{s.strftime('%b %d %H:%M')} -> {e.strftime('%b %d %H:%M')}", s, e, 24))
        s = e
    if len(wins) > 1:
        cs, ce = first, wins[-1][3]
        hours = int((ce - cs) / pd.Timedelta(hours=1))
        wins.append((f"span_{cs.strftime('%b%d')}_{ce.strftime('%b%d')}",
                     f"Combined {cs.strftime('%b %d %H:%M')} -> {ce.strftime('%b %d %H:%M')}",
                     cs, ce, hours))
    return wins


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='6_4-5')
    ap.add_argument('--speaker-side', default=SPEAKER_SIDE_DEFAULT, choices=list(SIDE_KEYS))
    ap.add_argument('--resample', default=RESAMPLE_DEFAULT,
                    help='pandas resample rule for plot bins (default 10min)')
    ap.add_argument('--force', action='store_true', help='re-analyze PKLs even if master CSV exists')
    args = ap.parse_args()

    folder = find_overhaul_folder(args.exp)
    assert '_overhaul' in folder
    out_dir = os.path.join(folder, 'direct_plots', 'per_day')
    os.makedirs(out_dir, exist_ok=True)
    master = os.path.join(folder, 'direct_plots', f'phonotaxis_minute_{args.exp}.csv')

    print(f"Experiment   : {args.exp}")
    print(f"OVERHAUL     : {folder}")
    print(f"Speaker side : {args.speaker_side} ({SIDE_KEYS[args.speaker_side]})")
    print(f"Resample     : {args.resample}")
    print(f"Master CSV   : {master}")
    print(f"Output       : {out_dir}")

    with open(os.path.join(folder, 'buzzwatch_track_settings.yml')) as f:
        settings = yaml.safe_load(f)

    df = None
    if os.path.exists(master) and not args.force:
        cand = pd.read_csv(master, index_col='timestamp', parse_dates=True)
        if f'side_flying_{args.speaker_side}' in cand.columns:
            print("[load] master CSV exists -> skipping PKL analysis")
            df = cand
        else:
            print("[load] master CSV missing side columns -> re-analyzing")
    if df is None:
        df = analyze(folder, args.exp)
        df.to_csv(master, index_label='timestamp')
        print(f"  saved master CSV: {master}", flush=True)

    # Resample from 1-min master to desired bin width, then compute fractions
    df = df[~df.index.isna()].sort_index()
    df = df.resample(args.resample).mean()
    df = add_speaker_fractions(df, args.speaker_side)
    write_bonus_pkl(folder, df, args.speaker_side)
    print(f"Data span    : {df.index.min()} -> {df.index.max()}  ({len(df)} bins at {args.resample})")

    windows = build_windows(df.index.min(), df.index.max())
    day_wins  = [(l, t, s, e, h) for l, t, s, e, h in windows if not l.startswith('span_')]
    span_wins = [(l, t, s, e, h) for l, t, s, e, h in windows if l.startswith('span_')]
    print(f"Windows      : {[w[0] for w in windows]}")

    # --- per-day windows: datetime x-axis, y-axis auto-scaled per plot -----
    for label, title, s, e, hours in day_wins:
        wdf = df[(df.index >= s) & (df.index < e)]
        if wdf.empty:
            print(f"  [{label}] no data, skipping")
            continue
        print(f"  [{label}] {title}  ({len(wdf)} rows)")
        flying  = wdf['numb_mosquitos_flying']
        resting = wdf['numb_mosquitos_resting']
        combined = (flying.fillna(0) + resting.fillna(0)).where(~(flying.isna() & resting.isna()))

        ymax_fly = _round_up(flying.max())
        print(f"    y-max  fly={ymax_fly}")

        plot_activity(flying, s, e, hours, 'flying', ymax_fly, out_dir, args.exp, label, title)

        plot_speaker(wdf['frac_flying_speaker'],   s, e, hours, 'flying',   args.speaker_side, out_dir, args.exp, title, label)
        plot_speaker(wdf['frac_resting_speaker'],  s, e, hours, 'resting',  args.speaker_side, out_dir, args.exp, title, label)
        plot_speaker(wdf['frac_combined_speaker'], s, e, hours, 'combined', args.speaker_side, out_dir, args.exp, title, label)

        act_csv = wdf[['numb_mosquitos_flying', 'numb_mosquitos_resting']].copy()
        act_csv['numb_mosquitos_combined'] = combined
        act_csv.to_csv(os.path.join(out_dir, f'activity_{label}.csv'), index_label='timestamp')
        wdf[['frac_flying_speaker', 'frac_resting_speaker', 'frac_combined_speaker',
             'n_flying', 'n_resting']].to_csv(
            os.path.join(out_dir, f'speaker_{label}.csv'), index_label='timestamp')

    # --- combined/span windows: linear 48h datetime x-axis, per-plot y-scale -
    for label, title, s, e, hours in span_wins:
        wdf = df[(df.index >= s) & (df.index < e)]
        if wdf.empty:
            print(f"  [{label}] no data, skipping")
            continue
        print(f"  [{label}] {title}  ({len(wdf)} rows, {hours}h)")
        flying  = wdf['numb_mosquitos_flying']
        resting = wdf['numb_mosquitos_resting']
        combined = (flying.fillna(0) + resting.fillna(0)).where(~(flying.isna() & resting.isna()))

        ymax_fly = _round_up(flying.max())
        print(f"    y-max  fly={ymax_fly}")

        plot_activity(flying, s, e, hours, 'flying', ymax_fly, out_dir, args.exp, label, title)

        plot_speaker(wdf['frac_flying_speaker'],   s, e, hours, 'flying',   args.speaker_side, out_dir, args.exp, title, label)
        plot_speaker(wdf['frac_resting_speaker'],  s, e, hours, 'resting',  args.speaker_side, out_dir, args.exp, title, label)
        plot_speaker(wdf['frac_combined_speaker'], s, e, hours, 'combined', args.speaker_side, out_dir, args.exp, title, label)

        act_csv = wdf[['numb_mosquitos_flying', 'numb_mosquitos_resting']].copy()
        act_csv['numb_mosquitos_combined'] = combined
        act_csv.to_csv(os.path.join(out_dir, f'activity_{label}.csv'), index_label='timestamp')
        wdf[['frac_flying_speaker', 'frac_resting_speaker', 'frac_combined_speaker',
             'n_flying', 'n_resting']].to_csv(
            os.path.join(out_dir, f'speaker_{label}.csv'), index_label='timestamp')

    plot_speaker_zone(folder, settings, out_dir, args.exp, args.speaker_side)
    print(f"[DONE] {out_dir}")

    # Copy PNGs to central results folder, split by plot type
    import shutil, glob as _glob
    res = os.path.join(RESULTS_ROOT, args.exp)
    act_dir  = os.path.join(res, 'activity');    os.makedirs(act_dir,  exist_ok=True)
    spk_dir  = os.path.join(res, 'speaker');     os.makedirs(spk_dir,  exist_ok=True)
    qc_dir   = os.path.join(res, 'QC');          os.makedirs(qc_dir,   exist_ok=True)
    QC_NAMES = {'speaker_zone.png', 'custom_zones_qc.png'}
    copied = 0
    for png in _glob.glob(os.path.join(out_dir, '*.png')):
        name = os.path.basename(png)
        if name in QC_NAMES:
            dst = qc_dir
        elif name.startswith('activity_'):
            dst = act_dir
        else:
            dst = spk_dir
        shutil.copy2(png, dst)
        copied += 1
    if copied:
        print(f"[results] {copied} PNGs → {res}")


if __name__ == '__main__':
    main()
