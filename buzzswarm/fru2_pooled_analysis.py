#!/usr/bin/env python3
"""
Pooled-across-sessions analysis: rather than computing per-session bins and
plotting their mean, pool all trajectories from the 3 sessions of each sex
into one big dataset and compute per-bin r50 / prop_within_R / activity from
the pooled trajectory pool. One single line per sex.

Runs three filter configurations and writes each to its own folder:
  - plots_pooled_no_split    : no per-step split (only avg-speed + density)
  - plots_pooled_max80       : per-step split at 80 px/frame
  - plots_pooled_max60       : per-step split at 60 px/frame

Each folder contains:
  pooled_r50.png
  pooled_prop_within_100.png
  pooled_prop_within_150.png
  pooled_activity.png
  + matching CSV per plot
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    # Imported as part of the buzzswarm package (GUI/CLI): share the SAME engine module object
    # that callers configure via `from buzzswarm import fru2_zt_normalized_analysis` so parameter
    # overrides (setattr on the engine) actually reach the analysis.
    from buzzswarm import fru2_zt_normalized_analysis as m
except ImportError:
    # Standalone (run from inside buzzswarm/): fall back to the sibling top-level module.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fru2_zt_normalized_analysis as m


CONFIGS = [
    ('plots_pooled_max40',    {'USE_STEP_SPLIT': True,  'STEP_SPLIT_MAX': 40.0}),
]

# Same palette as activity_plot_manager.py's _GROUP_PALETTE, for visual consistency across the
# app's multi-experiment comparison plots. Only used by run_multi (run() keeps the single-group
# blue default).
_GROUP_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def collect_traj_records(sessions):
    """For every trajectory in every session, compute its normalized ZT (midpoint)
    and its per-frame distances to its session's cage centroid. Returns a list of
    dicts, one per trajectory."""
    out = []
    for sd in sessions:
        first_ts = sd['first_ts']; last_ts = sd['last_ts']
        total_s = (last_ts - first_ts).total_seconds()
        if total_s <= 0:
            continue
        centroid = sd['cage_centroid']
        for fl in sd['flights']:
            ts_list = [ts for ts in fl['timestamps'] if ts is not None]
            if len(ts_list) < 2:
                continue
            mid_elapsed = ((ts_list[0] - first_ts).total_seconds()
                           + (ts_list[-1] - first_ts).total_seconds()) / 2.0
            if mid_elapsed < 0 or mid_elapsed > total_s:
                continue
            zt_mid = 12.0 * mid_elapsed / total_s
            dists = np.asarray([m.distance_to_center(pt, centroid) for pt in fl['coords']])
            if len(dists) == 0:
                continue
            out.append({
                'sex': sd['sex'],
                'zt_mid': zt_mid,
                'dists': dists,
                'prop100': float(np.mean(dists <= 100)),
                'prop150': float(np.mean(dists <= 150)),
            })
    return out


def pooled_bins(records, sex):
    """Bin all `sex` trajectories on a single 0-12 ZT axis with BIN_MINUTES bins.
    Returns (lower, upper, r50, prop100, prop150, activity, n_traj)."""
    n_bins = int(round(12 * 60 / m.BIN_MINUTES))
    bin_w = 12.0 / n_bins
    buckets = [[] for _ in range(n_bins)]
    p100 = [[] for _ in range(n_bins)]
    p150 = [[] for _ in range(n_bins)]
    for r in records:
        if r['sex'] != sex:
            continue
        b = min(n_bins - 1, max(0, int(r['zt_mid'] / bin_w)))
        buckets[b].append(r['dists'])
        p100[b].append(r['prop100'])
        p150[b].append(r['prop150'])

    r50 = np.full(n_bins, np.nan)
    prop100 = np.full(n_bins, np.nan)
    prop150 = np.full(n_bins, np.nan)
    n_traj = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        td = buckets[b]
        n_traj[b] = len(td)
        if len(td) < m.MIN_TRAJECTORIES_PER_BIN:
            continue
        nT = len(td)
        d_parts, w_parts = [], []
        for d_arr in td:
            n_pts = len(d_arr)
            d_parts.append(d_arr)
            w_parts.append(np.full(n_pts, 1.0 / (nT * n_pts)))
        all_d = np.concatenate(d_parts)
        all_w = np.concatenate(w_parts)
        order = np.argsort(all_d)
        sd_arr = all_d[order]; sw_arr = all_w[order]
        cumw = np.cumsum(sw_arr)
        if cumw[-1] < 0.5:
            r50[b] = np.nan
        else:
            idx = int(np.searchsorted(cumw, 0.5))
            if idx == 0:
                r50[b] = float(sd_arr[0])
            else:
                d0, d1 = sd_arr[idx - 1], sd_arr[idx]
                c0, c1 = cumw[idx - 1], cumw[idx]
                r50[b] = float(d0 + (0.5 - c0) / (c1 - c0) * (d1 - d0)) if c1 > c0 else float(d1)
        prop100[b] = float(np.mean(p100[b]))
        prop150[b] = float(np.mean(p150[b]))

    lower = np.arange(n_bins) * bin_w
    upper = lower + bin_w
    activity = n_traj / float(m.BIN_MINUTES)
    return lower, upper, r50, prop100, prop150, activity, n_traj


def step_xy(lower, upper, vals):
    xs, ys = [], []
    for lo, up, v in zip(lower, upper, vals):
        if np.isnan(v):
            continue
        xs.extend([lo, up]); ys.extend([v, v])
    return xs, ys


def _finalize_series_identity(ax, groups):
    """Presentation-only: identity should never be colour-alone, and a one-item legend box is
    noise. For a single group (the single-experiment GUI case, where the group is the experiment
    name) name it in the title and drop the legend; for >=2 groups keep the legend (colour =
    identity). Cosmetic — does not touch any plotted data."""
    groups = [g for g in groups]
    if len(groups) <= 1:
        if groups:
            ax.set_title(str(groups[0]), fontsize=12)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    else:
        ax.legend(loc='upper left', frameon=False)


def plot_pooled(records, value_idx, ylabel, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    csv_rows = []
    for sex in m.SEX_ORDER:
        lower, upper, r50, p100, p150, activity, n_traj = pooled_bins(records, sex)
        vals = (r50, p100, p150, activity)[value_idx]
        xs, ys = step_xy(lower, upper, vals)
        if xs:
            ax.plot(xs, ys, color=m.SEX_COLORS[sex], lw=2.4, label=sex)
        for lo, up, v, n in zip(lower, upper, vals, n_traj):
            csv_rows.append({'sex': sex, 'zt_lo': lo, 'zt_hi': up,
                              'value': v, 'n_trajectories': int(n)})
    ax.set_xlabel('ZT  (session squashed to 0-12)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 12); ax.set_xticks(np.arange(0, 13, 1))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    _finalize_series_identity(ax, m.SEX_ORDER)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(out_path.replace('.png', '.csv'),
                                  index=False, float_format='%.4f')


def run(experiment_dir, output_dir=None, settings=None, group_label=None):
    """GUI entry point: pooled r50 / prop-within / activity plots for one experiment folder.

    Loads the single experiment via process_session_from_dir (no BASE_PATH/SESSIONS globals).
    "Pooled" normally pools the multiple same-sex sessions of a study; the single-experiment
    case is a degenerate one-group pooling — the experiment is treated as one group
    (group_label, default the folder name) so plot_pooled renders one line. Writes 4 PNG+CSV
    pairs under output_dir, defaulting to <experiment_dir>/plots/buzzswarm/pooled/. Returns the
    output directory."""
    if settings is None:
        settings = dict(CONFIGS[0][1])   # {'USE_STEP_SPLIT': True, 'STEP_SPLIT_MAX': 40.0}
    if group_label is None:
        group_label = os.path.basename(os.path.normpath(experiment_dir)) or 'experiment'
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, 'plots', 'buzzswarm', 'pooled')
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n=== pooled: {group_label}  settings={settings} ===')

    for k, v in settings.items():
        setattr(m, k, v)

    sd = m.load_or_build_session(experiment_dir, verbose=False)   # cached session (plots/sholl_cache.pkl)
    if sd is not None:
        sd['sex'] = sd['strain'] = sd['batch'] = group_label
    sessions = [sd] if sd is not None else []
    records = collect_traj_records(sessions)
    print(f'  pooled trajectories: {group_label}={len(records)}')

    # Temporarily register this one group so plot_pooled (which iterates m.SEX_ORDER /
    # m.SEX_COLORS) renders it; restore afterwards so the module globals stay clean.
    saved_order, saved_colors = m.SEX_ORDER, dict(m.SEX_COLORS)
    try:
        m.SEX_ORDER = [group_label]
        m.SEX_COLORS = {group_label: saved_colors.get(group_label, '#0070FF')}
        plot_pooled(records, 0, 'Radius at which mean per-trajectory fraction within = 0.5  (px)',
                    os.path.join(output_dir, 'pooled_r50.png'))
        plot_pooled(records, 1, 'Mean fraction of trajectory within 100 px of centre',
                    os.path.join(output_dir, 'pooled_prop_within_100.png'), ylim=(-0.02, 1.02))
        plot_pooled(records, 2, 'Mean fraction of trajectory within 150 px of centre',
                    os.path.join(output_dir, 'pooled_prop_within_150.png'), ylim=(-0.02, 1.02))
        plot_pooled(records, 3, 'Trajectories per real minute (pooled)',
                    os.path.join(output_dir, 'pooled_activity.png'))
    finally:
        m.SEX_ORDER, m.SEX_COLORS = saved_order, saved_colors
    print(f'  wrote -> {output_dir}')
    return output_dir


def run_multi(experiments, output_dir=None, settings=None):
    """GUI entry point: pooled r50 / prop-within / activity plots overlaid across MULTIPLE
    experiments, one line per experiment/group. `experiments` is a list of
    `(experiment_dir, group_label)` pairs.

    Each experiment is loaded via the same cached load_or_build_session (plots/sholl_cache.pkl)
    run()/the Dashboard already use, so nothing is recomputed for an experiment that was already
    tracked. Writes the same 4 PNG+CSV pairs as run(), just with N lines instead of 1.
    output_dir defaults to <first experiment_dir>/plots/buzzswarm/pooled_multi/ if not given, but
    callers comparing experiments should pass a package-level folder instead (a comparison plot
    doesn't semantically belong to any single one of the experiments it compares)."""
    if not experiments:
        raise ValueError('run_multi requires at least one (experiment_dir, group_label) pair')
    if settings is None:
        settings = dict(CONFIGS[0][1])
    if output_dir is None:
        output_dir = os.path.join(experiments[0][0], 'plots', 'buzzswarm', 'pooled_multi')
    os.makedirs(output_dir, exist_ok=True)
    group_labels = [g for _, g in experiments]
    print(f'\n=== pooled (multi): {group_labels}  settings={settings} ===')

    for k, v in settings.items():
        setattr(m, k, v)

    sessions = []
    loaded_labels = []
    for experiment_dir, group_label in experiments:
        sd = m.load_or_build_session(experiment_dir, verbose=False)
        if sd is None:
            print(f'  skipped (no session data): {group_label}')
            continue
        sd = dict(sd)  # don't mutate the cached session dict, it may be shared across calls
        sd['sex'] = sd['strain'] = sd['batch'] = group_label
        sessions.append(sd)
        loaded_labels.append(group_label)
    records = collect_traj_records(sessions)
    for g in loaded_labels:
        print(f'  pooled trajectories: {g}={sum(1 for r in records if r["sex"] == g)}')

    # Temporarily register all loaded groups so plot_pooled (which iterates m.SEX_ORDER /
    # m.SEX_COLORS) renders one line per group; restore afterwards so the module globals stay
    # clean for any other caller (mirrors run()'s save/restore pattern).
    saved_order, saved_colors = m.SEX_ORDER, dict(m.SEX_COLORS)
    try:
        m.SEX_ORDER = loaded_labels
        m.SEX_COLORS = {g: _GROUP_PALETTE[i % len(_GROUP_PALETTE)] for i, g in enumerate(loaded_labels)}
        plot_pooled(records, 0, 'Radius at which mean per-trajectory fraction within = 0.5  (px)',
                    os.path.join(output_dir, 'pooled_r50.png'))
        plot_pooled(records, 1, 'Mean fraction of trajectory within 100 px of centre',
                    os.path.join(output_dir, 'pooled_prop_within_100.png'), ylim=(-0.02, 1.02))
        plot_pooled(records, 2, 'Mean fraction of trajectory within 150 px of centre',
                    os.path.join(output_dir, 'pooled_prop_within_150.png'), ylim=(-0.02, 1.02))
        plot_pooled(records, 3, 'Trajectories per real minute (pooled)',
                    os.path.join(output_dir, 'pooled_activity.png'))
    finally:
        m.SEX_ORDER, m.SEX_COLORS = saved_order, saved_colors
    print(f'  wrote -> {output_dir}')
    return output_dir
