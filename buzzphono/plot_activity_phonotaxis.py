#!/usr/bin/env python3
"""
Activity plots for a phonotaxis experiment (stim ON during min 40-50 of every hour).

Same three plots as plot_activity_single_experiment.py, with an extra orange
shading layer marking the stim windows.
"""

import argparse
import os
import pickle
import sys
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import sem


STIM_START_MIN = 40
STIM_END_MIN   = 50
STIM_COLOR     = '#ff8c00'
STIM_ALPHA     = 0.30


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def _find_pkl(folder, data_mode):
    candidates = {
        'normal':        ['analyzed_data.pkl'],
        'speed-filtered': ['analyzed_data_filtered.pkl', 'analyzed_data_speed_filtered.pkl'],
        'filtered':      ['analyzed_data_filtered.pkl', 'analyzed_data_speed_filtered.pkl'],
        'auto':          ['analyzed_data_filtered.pkl', 'analyzed_data_speed_filtered.pkl', 'analyzed_data.pkl'],
    }
    for fname in candidates.get(data_mode, candidates['auto']):
        p = os.path.join(folder, fname)
        if os.path.exists(p):
            return p
    return None


def load_population(folder, data_mode, resample):
    pkl_path = _find_pkl(folder, data_mode)
    if pkl_path is None:
        sys.exit(f"No analyzed data found in {folder} for mode '{data_mode}'")
    print(f"Loading: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = _CompatUnpickler(f).load()

    pop = data.get('population_data')
    if pop is None or len(pop) == 0:
        sys.exit("population_data is empty in the pickle file")

    pop = pop.copy()
    pop.index = pd.to_datetime(pop.index, errors='coerce')
    pop = pop[~pop.index.isna()].sort_index()

    if 'numb_mosquitos_flying' not in pop.columns:
        sys.exit("numb_mosquitos_flying column not found in population_data")

    pop = pop[['numb_mosquitos_flying']].resample(resample).mean(numeric_only=True)
    return pop


def _add_night_shading(ax, t_min, t_max, zt0_hour,
                       night_color='#C8C8C8', alpha=0.35):
    day = pd.Timedelta(hours=24)
    zt12 = pd.Timedelta(hours=12)
    first_zt0 = t_min.normalize() + pd.Timedelta(hours=zt0_hour)
    if first_zt0 > t_min:
        first_zt0 -= day
    cur = first_zt0
    while cur < t_max:
        ns = max(cur + zt12, t_min)
        ne = min(cur + day, t_max)
        if ns < ne:
            ax.axvspan(ns, ne, color=night_color, alpha=alpha, linewidth=0)
        cur += day


def _add_stim_shading_datetime(ax, t_min, t_max):
    """Shade min 40-50 of every hour across a datetime axis."""
    start = t_min.floor('h')
    end   = t_max.ceil('h')
    cur = start
    one_hour = pd.Timedelta(hours=1)
    while cur < end:
        s = cur + pd.Timedelta(minutes=STIM_START_MIN)
        e = cur + pd.Timedelta(minutes=STIM_END_MIN)
        s = max(s, t_min); e = min(e, t_max)
        if s < e:
            ax.axvspan(s, e, color=STIM_COLOR, alpha=STIM_ALPHA, linewidth=0)
        cur += one_hour


def _add_stim_shading_zt(ax):
    """Shade min 40-50 of every hour across a ZT (0-24) axis."""
    for h in range(24):
        ax.axvspan(h + STIM_START_MIN/60, h + STIM_END_MIN/60,
                   color=STIM_COLOR, alpha=STIM_ALPHA, linewidth=0)


def plot_timeseries(pop, out_dir, exp_name, zt0_hour, rolling_window):
    series = pop['numb_mosquitos_flying']
    rolled = series.rolling(window=rolling_window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    t_min, t_max = rolled.index.min(), rolled.index.max()

    _add_night_shading(ax, t_min, t_max, zt0_hour)
    _add_stim_shading_datetime(ax, t_min, t_max)

    ax.plot(rolled.index, rolled.values, color='#1a6faf', linewidth=1.2,
            label=f'{rolling_window}-min rolling mean')
    ax.fill_between(rolled.index, 0, rolled.values, color='#1a6faf', alpha=0.15)

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Date / Time', fontsize=12)
    ax.set_ylabel('Number of flying mosquitoes', fontsize=12)
    ax.set_title(f'Flight activity – full time-series\n{exp_name}',
                 fontsize=13, fontweight='bold')

    night_patch = mpatches.Patch(color='#C8C8C8', alpha=0.5, label='Subjective night (ZT12–ZT24)')
    stim_patch  = mpatches.Patch(color=STIM_COLOR, alpha=STIM_ALPHA, label='475 Hz stim (min 40–50)')
    ax.legend(handles=[ax.lines[0], night_patch, stim_patch], fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    fig.autofmt_xdate(rotation=0, ha='center')
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'activity_timeseries.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: activity_timeseries.png/.pdf")


def _project_to_zt(series, zt0_hour):
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    zt = ((s.index.hour - zt0_hour) * 60 + s.index.minute) / 60.0
    zt = zt % 24
    return pd.Series(s.values, index=zt).sort_index()


def _split_days(pop, zt0_hour):
    series = pop['numb_mosquitos_flying'].dropna()
    days = {}
    date_range = pd.date_range(
        start=(series.index.min().normalize() + pd.Timedelta(hours=zt0_hour)),
        end=(series.index.max() + pd.Timedelta(hours=24)),
        freq='24h'
    )
    for i in range(len(date_range) - 1):
        d_start, d_end = date_range[i], date_range[i + 1]
        mask = (series.index >= d_start) & (series.index < d_end)
        chunk = series[mask]
        if len(chunk) < 10:
            continue
        label = d_start.strftime('%b %d')
        days[label] = chunk
    return days


def plot_per_day_overlay(pop, out_dir, exp_name, zt0_hour):
    days = _split_days(pop, zt0_hour)
    if not days:
        print("  No days found, skipping per-day overlay plot.")
        return

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(days)))
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    ax.axvspan(12, 24, color='#C8C8C8', alpha=0.35, linewidth=0, label='Subjective night')
    _add_stim_shading_zt(ax)

    for (label, chunk), col in zip(days.items(), colors):
        zt = _project_to_zt(chunk, zt0_hour)
        binned = zt.groupby(pd.cut(zt.index, bins=np.arange(0, 24.05, 1/60))).mean()
        binned.index = [b.mid for b in binned.index]
        ax.plot(binned.index, binned.values, color=col, linewidth=1.0, alpha=0.85, label=label)

    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f'ZT{h}' for h in range(0, 25, 2)])
    ax.set_xlabel('Zeitgeber Time', fontsize=12)
    ax.set_ylabel('Number of flying mosquitoes', fontsize=12)
    ax.set_title(f'Flight activity per day (projected to ZT) — stim min 40–50/hour\n{exp_name}',
                 fontsize=13, fontweight='bold')
    stim_patch = mpatches.Patch(color=STIM_COLOR, alpha=STIM_ALPHA, label='475 Hz stim')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [stim_patch], labels + ['475 Hz stim'], fontsize=9, loc='upper right', ncol=2)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'activity_per_day_overlay.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: activity_per_day_overlay.png/.pdf")


def plot_zt_average(pop, out_dir, exp_name, zt0_hour):
    days = _split_days(pop, zt0_hour)
    if not days:
        print("  No days found, skipping ZT-average plot.")
        return

    bin_edges = np.arange(0, 24 + 1/60, 1/60)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
    day_profiles = []
    for chunk in days.values():
        zt = _project_to_zt(chunk, zt0_hour)
        binned = zt.groupby(pd.cut(zt.index, bins=bin_edges)).mean()
        binned.index = bin_mids
        day_profiles.append(binned)

    all_profiles = pd.concat(day_profiles, axis=1)
    mean_profile = all_profiles.mean(axis=1)
    sem_profile  = all_profiles.apply(lambda row: sem(row.dropna()) if row.dropna().size > 1 else 0.0, axis=1)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.axvspan(12, 24, color='#C8C8C8', alpha=0.35, linewidth=0, label='Subjective night')
    _add_stim_shading_zt(ax)

    ax.fill_between(mean_profile.index, mean_profile - sem_profile,
                    mean_profile + sem_profile, color='#1a6faf', alpha=0.25)
    ax.plot(mean_profile.index, mean_profile.values, color='#1a6faf', linewidth=2.0,
            label=f'Mean ± SEM (n={len(days)} days)')

    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f'ZT{h}' for h in range(0, 25, 2)])
    ax.set_xlabel('Zeitgeber Time', fontsize=12)
    ax.set_ylabel('Number of flying mosquitoes', fontsize=12)
    ax.set_title(f'Flight activity – ZT-averaged daily profile — stim min 40–50/hour\n{exp_name}',
                 fontsize=13, fontweight='bold')
    stim_patch = mpatches.Patch(color=STIM_COLOR, alpha=STIM_ALPHA, label='475 Hz stim')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [stim_patch], labels + ['475 Hz stim'], fontsize=10)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'activity_zt_average.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: activity_zt_average.png/.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment-dir', required=True)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--data-mode', choices=['auto', 'normal', 'speed-filtered', 'filtered'], default='auto')
    p.add_argument('--zt0-hour', type=int, default=5)
    p.add_argument('--resample', default='1min')
    p.add_argument('--rolling-window', type=int, default=20)
    p.add_argument('--start', default=None, help='ISO timestamp; trim data before this')
    p.add_argument('--end',   default=None, help='ISO timestamp; trim data at/after this')
    args = p.parse_args()

    exp_dir  = os.path.abspath(args.experiment_dir)
    out_dir  = args.output_dir or os.path.join(exp_dir, 'activity_plots')
    exp_name = os.path.basename(exp_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Experiment : {exp_name}")
    print(f"Output dir : {out_dir}")

    pop = load_population(exp_dir, args.data_mode, args.resample)
    if args.start:
        pop = pop[pop.index >= pd.Timestamp(args.start)]
    if args.end:
        pop = pop[pop.index <  pd.Timestamp(args.end)]
    print(f"Data loaded: {len(pop)} rows | {pop.index.min()} -> {pop.index.max()}")

    print("\nGenerating plots...")
    plot_timeseries(pop, out_dir, exp_name, args.zt0_hour, args.rolling_window)
    plot_per_day_overlay(pop, out_dir, exp_name, args.zt0_hour)
    plot_zt_average(pop, out_dir, exp_name, args.zt0_hour)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == '__main__':
    main()
