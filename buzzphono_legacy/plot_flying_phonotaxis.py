#!/usr/bin/env python3
"""
Flying-only phonotaxis plots from tracking_moving/.

For a given --start/--end window, produces:
  - polygons_tracking_coords.png      (sanity: polygons in tracking-coord frame)
  - occupancy_flying_on_vs_off.png    (2D heatmap ON, OFF, ON-OFF)
  - distance_flying_per_trajectory.png (histogram + boxplot, Mann-Whitney)
  - flying_near_speaker_timeseries.png (fraction within THRESH of speaker through time)
  - flying_near_speaker_on_vs_off.png  (per-hour ON vs OFF bars)
"""
import argparse
import glob
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))

class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if m.startswith('numpy._core'):
            m = m.replace('numpy._core', 'numpy.core')
        return super().find_class(m, n)

FPS, CHUNK = 25.0, 1200
STIM_LO, STIM_HI = 40, 50
THRESH = 200   # px, distance threshold from speaker

FNAME_RE = re.compile(r'_(\d{8})_(\d{6})_(\d{5})\.pkl$')

def parse_start(name):
    m = FNAME_RE.search(name)
    d, t, idx = m.groups()
    sess = pd.Timestamp(f'{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}')
    return sess + pd.Timedelta(seconds=int(idx) * CHUNK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end',   required=True)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--track-type', choices=['moving', 'resting'], default='moving',
                    help='which trajectory set to use (tracking_moving or tracking_resting)')
    args = ap.parse_args()

    TT = args.track_type
    track_dir   = f'tracking_{TT}'
    track_glob  = f'{TT}_*.pkl'
    label_word  = 'FLYING' if TT == 'moving' else 'RESTING'
    file_tag    = '' if TT == 'moving' else '_resting'
    # resting tracks are long and stationary -> subsample to ~1 Hz to bound memory
    stride      = 1 if TT == 'moving' else 25

    folder  = os.path.abspath(args.experiment_dir)
    out_dir = args.output_dir or os.path.join(
        folder, f'speaker_side_plots_{pd.Timestamp(args.start).strftime("%Y%m%d_%H")}_to_{pd.Timestamp(args.end).strftime("%Y%m%d_%H")}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'track type: {TT}')

    T0 = pd.Timestamp(args.start)
    T1 = pd.Timestamp(args.end)

    with open(os.path.join(folder, 'buzzwatch_track_settings.yml')) as f:
        yset = yaml.safe_load(f)
    cage  = np.array(yset['cage_border_points'], dtype=float)
    spkrp = np.array(yset['square_3_border_points'], dtype=float)
    sq4   = np.array(yset['square_4_border_points'], dtype=float)
    ctrl  = np.array(yset['control_border_points'], dtype=float)
    sugar = np.array(yset['sugar_border_points'], dtype=float)

    # The trajectory pkls are in the ANALYSIS coordinate frame (~0-640). If the
    # settings file was re-drawn in CAMERA resolution (max coord >> 640), the
    # polygons no longer match the trajectory data. Detect this and map the
    # camera-space polygons back into the trajectory frame so the speaker lands
    # on the actual left wall, not mid-cage.
    TRAJ_MAX = 645.0   # analysis ROI is a ~640x640 square
    if cage.max() > TRAJ_MAX:
        cyl = cage
        cxmin, cxmax = cyl[:, 0].min(), cyl[:, 0].max()
        cymin, cymax = cyl[:, 1].min(), cyl[:, 1].max()
        # canonical analysis-space cage (original settings, recorded before edit)
        ax0, ax1 = 33.0, 639.0
        ay0, ay1 = 29.0, 639.0
        def to_traj(p):
            q = p.copy()
            q[:, 0] = (p[:, 0] - cxmin) / (cxmax - cxmin) * (ax1 - ax0) + ax0
            q[:, 1] = (p[:, 1] - cymin) / (cymax - cymin) * (ay1 - ay0) + ay0
            return q
        print(f'WARNING: yml is camera-resolution (cage max={cage.max():.0f}); '
              f'rescaling polygons into trajectory frame')
        cage, spkrp, sq4, ctrl, sugar = map(to_traj, (cage, spkrp, sq4, ctrl, sugar))

    SPEAKER = ((cage[0, 0] + cage[3, 0]) / 2, (cage[0, 1] + cage[3, 1]) / 2)

    print(f'window  : {T0} -> {T1}')
    print(f'output  : {out_dir}')
    print(f'speaker : x={SPEAKER[0]:.1f}, y={SPEAKER[1]:.1f}')

    # (a) polygons in tracking coords
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    for pts, col, lbl in [(cage, 'cyan', 'cage'),
                          (spkrp, 'red', 'square_3 (left band)'),
                          (sq4,  'lime', 'square_4 (right)'),
                          (ctrl, 'yellow', 'control (top)'),
                          (sugar,'magenta', 'sugar (bottom)')]:
        c = np.vstack([pts, pts[:1]])
        ax.plot(c[:, 0], c[:, 1], color=col, lw=2, label=lbl)
    ax.scatter([SPEAKER[0]], [SPEAKER[1]], marker='*', s=400, c='red',
               edgecolor='black', lw=1.5, label='speaker (mid-left wall)', zorder=5)
    ax.set_xlim(0, 660); ax.set_ylim(660, 0); ax.set_aspect('equal')
    ax.set_title('Polygons in tracking coordinates', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)'); ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    fig.savefig(os.path.join(out_dir, 'polygons_tracking_coords.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  + polygons_tracking_coords.png')

    # scan flying tracks
    xy_on, xy_off, trows, mrows = [], [], [], []
    files = sorted(glob.glob(os.path.join(folder, track_dir, track_glob)))
    print(f'scanning {len(files)} chunks')
    for i, f in enumerate(files):
        cs = parse_start(os.path.basename(f))
        if cs + pd.Timedelta(seconds=CHUNK) < T0 or cs >= T1:
            continue
        with open(f, 'rb') as fh:
            obj = _U(fh).load()
        for m in obj.objects.values():
            coords = m.get('coordinates'); ts = m.get('time_stamp')
            if not coords or not ts:
                continue
            c = np.asarray(coords, dtype=float)[::stride]
            frames = np.asarray(ts, dtype=float)[::stride]
            times = cs + pd.to_timedelta(frames / FPS, unit='s')
            msk = (times >= T0) & (times < T1)
            if not msk.any():
                continue
            c = c[msk]; t = times[msk]
            moh = t.minute + t.second / 60.0
            on  = (moh >= STIM_LO) & (moh < STIM_HI)
            d   = np.hypot(c[:, 0] - SPEAKER[0], c[:, 1] - SPEAKER[1])
            near = d < THRESH

            if on.any():    xy_on.append(c[on])
            if (~on).any(): xy_off.append(c[~on])
            trows.append({'mean_dist': d.mean(), 'n_frames': len(c),
                          'frac_on': float(on.mean()),
                          'frac_near': float(near.mean())})
            minute = t.floor('min')
            for mn in np.unique(minute):
                mm = minute == mn
                mrows.append({'minute': mn,
                              'n_flying': int(mm.sum()),
                              'n_near':   int(near[mm].sum())})
        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(files)}')

    xy_on  = np.concatenate(xy_on)  if xy_on  else np.empty((0, 2))
    xy_off = np.concatenate(xy_off) if xy_off else np.empty((0, 2))
    traj    = pd.DataFrame(trows)
    minutes = pd.DataFrame(mrows).groupby('minute').sum().sort_index()
    minutes['frac_near'] = minutes['n_near'] / minutes['n_flying'].where(minutes['n_flying'] > 0)
    print(f'flying tracks: {len(traj):,}, ON pts: {len(xy_on):,}, OFF pts: {len(xy_off):,}')

    # (b) 2D heatmaps
    xmin, xmax = cage[:, 0].min() - 5, cage[:, 0].max() + 5
    ymin, ymax = cage[:, 1].min() - 5, cage[:, 1].max() + 5
    bx = np.linspace(xmin, xmax, 60); by = np.linspace(ymin, ymax, 60)
    H_off, _, _ = np.histogram2d(xy_off[:, 0], xy_off[:, 1], bins=[bx, by])
    H_on,  _, _ = np.histogram2d(xy_on[:, 0],  xy_on[:, 1],  bins=[bx, by])
    # per-second density
    dur_h = (T1 - T0).total_seconds() / 3600
    sec_off = 50 * 60 * dur_h; sec_on = 10 * 60 * dur_h
    R_off = H_off / sec_off; R_on = H_on / sec_on
    cage_closed = np.vstack([cage, cage[:1]])

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=150)
    vmax = max(R_off.max(), R_on.max(), 1e-9)
    for ax, R, ttl in zip(axes[:2], [R_off, R_on],
                          [f'{label_word} OFF (silence)\n50 min/h x {dur_h:.0f} h',
                           f'{label_word} ON (475 Hz)\n10 min/h x {dur_h:.0f} h']):
        im = ax.imshow(R.T, origin='upper', extent=[xmin, xmax, ymax, ymin],
                       cmap='magma', vmin=0, vmax=vmax, aspect='equal',
                       interpolation='gaussian')
        ax.plot(cage_closed[:, 0], cage_closed[:, 1], color='white', lw=1.5)
        ax.scatter([SPEAKER[0]], [SPEAKER[1]], marker='*', s=350, c='cyan',
                   edgecolor='black', lw=1.5, zorder=5, label='speaker')
        ax.set_title(ttl, fontsize=13, fontweight='bold')
        ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)'); ax.invert_yaxis()
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label='pts/s/cell')
    diff = R_on - R_off
    dmax = np.percentile(np.abs(diff), 99) or 1.0
    im3 = axes[2].imshow(diff.T, origin='upper', extent=[xmin, xmax, ymax, ymin],
                         cmap='RdBu_r', vmin=-dmax, vmax=dmax, aspect='equal',
                         interpolation='gaussian')
    axes[2].plot(cage_closed[:, 0], cage_closed[:, 1], color='black', lw=1.5)
    axes[2].scatter([SPEAKER[0]], [SPEAKER[1]], marker='*', s=350, c='yellow',
                    edgecolor='black', lw=1.5, zorder=5)
    axes[2].set_title(f'ON - OFF ({TT})\nred = more in ON', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('x (px)'); axes[2].set_ylabel('y (px)'); axes[2].invert_yaxis()
    plt.colorbar(im3, ax=axes[2], fraction=0.045, pad=0.04, label='delta pts/s/cell')
    fig.suptitle(f'{label_word}-only occupancy  {T0.strftime("%b %d %H:%M")} - {T1.strftime("%b %d %H:%M")}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'occupancy{file_tag}_on_vs_off.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  + occupancy_flying_on_vs_off.png')

    # (c) per-trajectory distance histogram + boxplot
    on_t  = traj[traj['frac_on'] >= 0.5]
    off_t = traj[traj['frac_on'] <  0.5]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150,
                                    gridspec_kw={'width_ratios': [3, 1]})
    bins = np.linspace(0, traj['mean_dist'].quantile(0.99), 50)
    ax1.hist(off_t['mean_dist'], bins=bins, density=True, alpha=0.55,
             color='#444', label=f'OFF (n={len(off_t):,})')
    ax1.hist(on_t['mean_dist'],  bins=bins, density=True, alpha=0.55,
             color='#ff8c00', label=f'ON (n={len(on_t):,})')
    ax1.axvline(THRESH, color='red', linestyle='--', lw=1.5, label=f'threshold {THRESH} px')
    ax1.set_xlabel('trajectory mean distance to speaker (px)', fontsize=12)
    ax1.set_ylabel('density', fontsize=12)
    ax1.set_title(f'{label_word}: per-trajectory distance to speaker', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    try:
        _, p = mannwhitneyu(on_t['mean_dist'], off_t['mean_dist'], alternative='less')
        sub = f'Mann-Whitney (ON<OFF): p={p:.3g}'
    except Exception:
        sub = ''
    bp = ax2.boxplot([off_t['mean_dist'], on_t['mean_dist']],
                      tick_labels=['OFF', 'ON'], patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], ['#444', '#ff8c00']):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax2.set_ylabel('distance (px)', fontsize=11); ax2.set_title(sub, fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'distance{file_tag}_per_trajectory.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  + distance_flying_per_trajectory.png')

    # (d) timeseries of fraction within THRESH of speaker
    m = minutes.copy()
    m['frac_near_smooth'] = m['frac_near'].rolling(20, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    t_min, t_max = m.index.min(), m.index.max()
    cur = t_min.floor('h')
    while cur < t_max:
        s = cur + pd.Timedelta(minutes=STIM_LO); e = cur + pd.Timedelta(minutes=STIM_HI)
        if s < t_max and e > t_min:
            ax.axvspan(max(s, t_min), min(e, t_max), color='#ff8c00', alpha=0.30, lw=0)
        cur += pd.Timedelta(hours=1)
    chance = (np.pi * THRESH ** 2) / ((xmax - xmin) * (ymax - ymin))
    ax.axhline(chance, color='gray', linestyle='--', lw=1,
               label=f'uniform chance ({chance:.2f})')
    ax.plot(m.index, m['frac_near_smooth'], color='#b22222', lw=1.4,
            label=f'20-min rolling fraction within {THRESH} px of speaker')
    ax.set_ylim(0, 1); ax.set_xlim(t_min, t_max)
    ax.set_xlabel('Date / Time', fontsize=12)
    ax.set_ylabel(f'fraction of {TT} points near speaker', fontsize=12)
    ax.set_title(f'{label_word}-only: fraction near speaker through time',
                 fontsize=13, fontweight='bold')
    stim_patch = mpatches.Patch(color='#ff8c00', alpha=0.4, label='475 Hz stim (min 40-50)')
    ax.legend(handles=[ax.lines[0], ax.lines[1], stim_patch], fontsize=10, loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    fig.autofmt_xdate(rotation=0, ha='center')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{TT}_near_speaker_timeseries.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  + flying_near_speaker_timeseries.png')

    # per-hour ON vs OFF bars
    m['hour'] = m.index.floor('h')
    m['moh'] = m.index.minute + m.index.second / 60.0
    m['on'] = (m['moh'] >= STIM_LO) & (m['moh'] < STIM_HI)
    hourly = m.groupby(['hour', 'on'])[['n_flying', 'n_near']].sum().unstack('on')
    on_frac  = (hourly['n_near'][True]  / hourly['n_flying'][True]).dropna()
    off_frac = (hourly['n_near'][False] / hourly['n_flying'][False]).dropna()
    common = on_frac.index.intersection(off_frac.index)
    on_frac, off_frac = on_frac.loc[common], off_frac.loc[common]
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(common)), 5), dpi=150)
    x = np.arange(len(common)); w = 0.38
    ax.bar(x - w/2, off_frac.values, w, color='#888', label='OFF')
    ax.bar(x + w/2, on_frac.values,  w, color='#ff8c00', label='ON (min 40-50)')
    ax.axhline(chance, color='gray', linestyle='--', lw=1,
               label=f'uniform chance ({chance:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels([t.strftime('%b %d\n%H:00') for t in common], fontsize=8)
    ax.set_ylabel(f'frac {TT} within {THRESH} px of speaker', fontsize=12)
    ax.set_title(f'{label_word}: per-hour ON vs OFF fraction near speaker',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1); ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{TT}_near_speaker_on_vs_off.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  + flying_near_speaker_on_vs_off.png')

    print(f'\nDone: {out_dir}')


if __name__ == '__main__':
    main()
