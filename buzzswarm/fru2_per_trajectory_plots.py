#!/usr/bin/env python3
"""Per-trajectory r50 / prop_within_100 / prop_within_150 outputs.

Writes everything under <experiment_dir>/plots/buzzswarm/per_trajectory/:

  all_day\\
    scatter_<metric>_by_sex_all_day_ZT.png + .csv
    line_<metric>_by_sex_all_day_ZT.png    + .csv
    violin_<metric>_by_sex_all_day_ZT.png  + .csv

  ZT0.0-0.5\\, ZT5.75-6.25\\, ZT11.5-12.0\\
    strip_<metric>_<window>.png + .csv

Every CSV uses the same flat row format:
    Sex, date, ZT, traj_id, r_50, prop_150, prop_100

Each row is one trajectory. The CSV is restricted to trajectories whose data
falls in one of the three named ZT windows (dawn, midday, dusk). Inside each
window subfolder the CSV holds only that window's trajectories. Inside
all_day\\ the CSV pools all three windows. r_50, prop_100, prop_150 are
computed from the points inside the window only (window-trimmed coords).
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    # Imported as part of the buzzswarm package (GUI/CLI): share the SAME engine module object
    # that callers configure via `from buzzswarm import fru2_zt_normalized_analysis` so parameter
    # overrides (setattr on the engine) actually reach the analysis.
    from buzzswarm import fru2_zt_normalized_analysis as m
except ImportError:
    # Standalone (run from inside buzzswarm/): fall back to the sibling top-level module.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fru2_zt_normalized_analysis as m

# ==== Window definitions (same anchors as existing Sholl/strip plots) ====
WINDOWS = [
    ('ZT0.0-0.5',    0.0,  0.5),
    ('ZT5.75-6.25',  5.75, 6.25),
    ('ZT11.5-12.0', 11.5, 12.0),
]

METRICS = [
    ('r50',             'Per-trajectory r50  (median distance to centre, px)', None),
    ('prop_within_100', 'Per-trajectory fraction within 100 px of centre',     (-0.02, 1.02)),
    ('prop_within_150', 'Per-trajectory fraction within 150 px of centre',     (-0.02, 1.02)),
]

CSV_COLS = ['Sex', 'date_genotype', 'ZT', 'traj_id', 'n_frames', 'r_50', 'prop_150', 'prop_100']

# Same palette as activity_plot_manager.py's _GROUP_PALETTE, for visual consistency across the
# app's multi-experiment comparison plots. Only used by run_multi (run() keeps the single-group
# blue default).
_GROUP_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# ==== Per-trajectory extraction restricted to a single window ====
def extract_window_df(all_sessions_data, zt_lo, zt_hi, window_label):
    """One row per trajectory whose window-trimmed coords have >= 2 points.
    Columns include the requested CSV columns + internal `r50` /
    `prop_within_100` / `prop_within_150` / `sex` for plotting."""
    rows = []
    for sd in all_sessions_data:
        sex = sd['sex']
        batch = sd['batch']
        sid = f"{sex}_{batch}"
        centroid = sd['cage_centroid']
        if (sd['last_ts'] - sd['first_ts']).total_seconds() <= 0:
            continue
        fl_in = m.flights_in_zt_window(sd, zt_lo, zt_hi)
        for idx, fl in enumerate(fl_in, start=1):
            dists = np.array([m.distance_to_center(p, centroid) for p in fl['coords']])
            if len(dists) < 2:
                continue
            r50 = float(np.median(dists))
            p100 = float(np.mean(dists <= 100))
            p150 = float(np.mean(dists <= 150))
            rows.append({
                'Sex': sex,
                'date_genotype': f'{batch}_{sex}',
                'ZT': window_label,
                'traj_id': f'{sid}_{idx:03d}',
                'n_frames': int(len(dists)),
                'r_50': r50,
                'prop_150': p150,
                'prop_100': p100,
                # internal aliases for plotting
                'sex': sex,
                'r50': r50,
                'prop_within_100': p100,
                'prop_within_150': p150,
                'zt_lo': zt_lo,
                'zt_hi': zt_hi,
            })
    return pd.DataFrame(rows)


def write_csv(df, path):
    """Write only the requested 7-column CSV (drop internal helper cols)."""
    if df.empty:
        pd.DataFrame(columns=CSV_COLS).to_csv(path, index=False)
        return
    df[CSV_COLS].to_csv(path, index=False, float_format='%.4f')


# ==== Plotting (all-day = full 12 hours, per-trajectory midpoint binning) ====
def _finalize_series_identity(ax, groups):
    """Presentation-only: for a single group (the single-experiment GUI case, where the group is
    the experiment name) name it in the title and drop the redundant one-item legend; for >=2
    groups keep the legend (colour = identity). Cosmetic — does not touch any plotted data."""
    groups = [g for g in groups]
    if len(groups) <= 1:
        if groups:
            ax.set_title(str(groups[0]), fontsize=12)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    else:
        ax.legend(loc='upper left', frameon=False)


def plot_all_day(df_full, value_key, ylabel, outdir, name_base, ylim, mode):
    """All-day plots use ALL trajectories across the 12 h ZT axis (continuous).
    Mode in {'scatter', 'line', 'violin'}. df_full must have a `zt_midpoint`
    column (0-12) and a `sex` column.
    The CSV alongside is written separately and only contains the 3-window
    rows (see write_csv calls in main)."""
    os.makedirs(outdir, exist_ok=True)
    if df_full.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    if mode == 'scatter':
        for sex in m.SEX_ORDER:
            sub = df_full[df_full['sex'] == sex]
            if sub.empty:
                continue
            ax.scatter(sub['zt_midpoint'].values, sub[value_key].values,
                       s=8, alpha=0.25, color=m.SEX_COLORS[sex],
                       edgecolors='none', label='_nolegend_', zorder=2)

    if mode in ('scatter', 'line'):
        summary = m._pertraj_bin_summary(df_full, value_key, m.BIN_MINUTES)
        for sex, s in summary.items():
            mask = ~np.isnan(s['mean'])
            if not mask.any():
                continue
            centers = 0.5 * (s['lower_zt'] + s['upper_zt'])
            ax.plot(centers[mask], s['mean'][mask],
                    color=m.SEX_COLORS[sex], lw=2.8, label=sex, zorder=5)
            sem_mask = mask & ~np.isnan(s['sem'])
            if sem_mask.any():
                ax.fill_between(centers[sem_mask],
                                s['mean'][sem_mask] - s['sem'][sem_mask],
                                s['mean'][sem_mask] + s['sem'][sem_mask],
                                color=m.SEX_COLORS[sex], alpha=0.20, zorder=1)

    if mode == 'violin':
        # One violin per (30-min ZT bin, sex)
        bin_minutes = 30
        n_bins = int(round(12 * 60 / bin_minutes))
        bin_width = 12.0 / n_bins
        edges = np.linspace(0.0, 12.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        offsets = {sex: (i - 1) * (bin_width * 0.25) for i, sex in enumerate(m.SEX_ORDER)}
        violin_w = bin_width * 0.22
        for sex in m.SEX_ORDER:
            sub = df_full[df_full['sex'] == sex]
            if sub.empty:
                continue
            idx = np.clip(np.floor(sub['zt_midpoint'].values / bin_width).astype(int),
                          0, n_bins - 1)
            vals = sub[value_key].values
            data, positions = [], []
            for b in range(n_bins):
                mask = idx == b
                if mask.sum() >= m.MIN_TRAJECTORIES_PER_BIN:
                    data.append(vals[mask])
                    positions.append(centers[b] + offsets[sex])
            if data:
                parts = ax.violinplot(data, positions=positions, widths=violin_w,
                                      showmeans=False, showmedians=True,
                                      showextrema=False)
                for body in parts['bodies']:
                    body.set_facecolor(m.SEX_COLORS[sex])
                    body.set_edgecolor(m.SEX_COLORS[sex])
                    body.set_alpha(0.45)
                if 'cmedians' in parts:
                    parts['cmedians'].set_color(m.SEX_COLORS[sex])
                    parts['cmedians'].set_linewidth(1.5)
            ax.plot([], [], color=m.SEX_COLORS[sex], lw=6, alpha=0.6, label=sex)

    ax.set_xlabel('ZT  (session squashed to 0-12)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 12)
    ax.set_xticks(np.arange(0, 12.1, 1))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    _finalize_series_identity(ax, m.SEX_ORDER)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{name_base}.png'), dpi=200)
    plt.close(fig)


def plot_window_strip(df_win, value_key, ylabel, outdir, name_base, ylim):
    """Strip plot for a single ZT window with pairwise Mann-Whitney brackets."""
    os.makedirs(outdir, exist_ok=True)
    if df_win.empty:
        return

    fig, ax = plt.subplots(figsize=(5.5, 6))
    rng = np.random.default_rng(0)
    positions = {sex: i for i, sex in enumerate(m.SEX_ORDER)}

    finite = df_win[value_key].replace([np.inf, -np.inf], np.nan).dropna().values
    if finite.size == 0:
        plt.close(fig); return
    data_min = float(np.min(finite)); data_max = float(np.max(finite))
    data_span = max(data_max - data_min, 1e-9)

    for sex in m.SEX_ORDER:
        sub = df_win[df_win['sex'] == sex]
        if len(sub) == 0:
            continue
        x = np.full(len(sub), positions[sex], dtype=float) \
            + rng.uniform(-0.22, 0.22, len(sub))
        ax.scatter(x, sub[value_key].values,
                   color=m.SEX_COLORS[sex], s=14, alpha=0.45, edgecolor='none')
        med = float(np.median(sub[value_key]))
        ax.hlines(med, positions[sex] - 0.28, positions[sex] + 0.28,
                  colors='black', lw=2.5, zorder=5)

    ax.set_xticks([positions[s] for s in m.SEX_ORDER])
    tick_labels = {'Female': '♀', 'Male': '♂', 'FruM': 'fruM♂'}
    ax.set_xticklabels([tick_labels[s] for s in m.SEX_ORDER], fontsize=14)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    y_top = data_max + 0.05 * data_span
    bracket_step = 0.06 * data_span
    pairs = [('Female', 'Male'), ('Male', 'FruM'), ('Female', 'FruM')]
    for i, (a, b) in enumerate(pairs):
        sa = df_win.loc[df_win['sex'] == a, value_key].to_numpy()
        sb = df_win.loc[df_win['sex'] == b, value_key].to_numpy()
        if sa.size and sb.size:
            try:
                res = stats.mannwhitneyu(sa, sb, alternative='two-sided')
                p = float(res.pvalue)
            except Exception:
                p = 1.0
        else:
            p = 1.0
        x1, x2 = positions[a], positions[b]
        y = y_top + i * bracket_step
        ax.plot([x1, x1, x2, x2],
                [y, y + 0.25 * bracket_step, y + 0.25 * bracket_step, y],
                color='black', lw=1.1)
        ax.text((x1 + x2) / 2.0, y + 0.35 * bracket_step, m.stars(p),
                ha='center', va='bottom', fontsize=12)

    upper_y = y_top + len(pairs) * bracket_step + 0.04 * data_span
    if ylim is not None:
        lo, hi = ylim
        ax.set_ylim(lo, max(hi, upper_y))
    else:
        ax.set_ylim(data_min - 0.05 * data_span, upper_y)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{name_base}.png'), dpi=200)
    plt.close(fig)


def run(experiment_dir, output_dir=None, group_label=None):
    """GUI entry point: per-trajectory metric outputs (r50 / prop-within) for one experiment.

    Loads the single experiment via process_session_from_dir (no BASE_PATH/SESSIONS globals)
    and writes, under output_dir (default <experiment_dir>/plots/buzzswarm/per_trajectory/):
      - all_three_windows.csv (combined per-trajectory metrics for the 3 ZT windows)
      - all_day/ scatter+line+violin plots and CSVs per metric
      - <window>/ per-window metric CSVs
    Unlike main(), it does NOT delete any pre-existing output folders. The experiment is
    treated as one group, so the by-sex all-day plots render a single series; the per-window
    *strip* plots (which are hardcoded Female/Male/FruM pairwise comparisons) are skipped as
    they are meaningless for a single experiment — their data is still written as CSV.
    Returns the output directory."""
    if group_label is None:
        group_label = os.path.basename(os.path.normpath(experiment_dir)) or 'experiment'
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, 'plots', 'buzzswarm', 'per_trajectory')
    os.makedirs(output_dir, exist_ok=True)

    print('Loading experiment...')
    sd = m.load_or_build_session(experiment_dir, verbose=False)   # cached session (plots/sholl_cache.pkl)
    if sd is not None:
        sd['sex'] = sd['strain'] = sd['batch'] = group_label
    all_sessions_data = [sd] if sd is not None else []
    if not all_sessions_data:
        print('No experiment data loaded. Nothing to plot.')
        return output_dir

    # Temporarily register this one group so the by-sex all-day plot helpers (which iterate
    # m.SEX_ORDER / m.SEX_COLORS) render it; restore afterwards so module globals stay clean.
    saved_order, saved_colors = m.SEX_ORDER, dict(m.SEX_COLORS)
    try:
        m.SEX_ORDER = [group_label]
        m.SEX_COLORS = {group_label: saved_colors.get(group_label, '#0070FF')}

        win_dfs = {}
        for label, lo, hi in WINDOWS:
            win_dfs[label] = extract_window_df(all_sessions_data, lo, hi, label)
            print(f'  {label}: {len(win_dfs[label])} trajectories')

        df_3win = pd.concat([d for d in win_dfs.values() if not d.empty],
                            ignore_index=True) if win_dfs else pd.DataFrame()
        write_csv(df_3win, os.path.join(output_dir, 'all_three_windows.csv'))

        df_full = m.extract_per_trajectory_metrics(all_sessions_data)
        print(f'  full all-day: {len(df_full)} trajectories')

        all_day_dir = os.path.join(output_dir, 'all_day')
        os.makedirs(all_day_dir, exist_ok=True)
        for key, ylabel, ylim in METRICS:
            for mode in ('scatter', 'line', 'violin'):
                name_base = f'{mode}_{key}_by_sex_all_day_ZT'
                plot_all_day(df_full, key, ylabel, all_day_dir, name_base, ylim, mode)
                write_csv(df_3win, os.path.join(all_day_dir, f'{name_base}.csv'))

        for label, _, _ in WINDOWS:
            win_dir = os.path.join(output_dir, label)
            os.makedirs(win_dir, exist_ok=True)
            df = win_dfs[label]
            for key, ylabel, ylim in METRICS:
                write_csv(df, os.path.join(win_dir, f'strip_{key}_{label}.csv'))
    finally:
        m.SEX_ORDER, m.SEX_COLORS = saved_order, saved_colors
    print(f'  wrote -> {output_dir}')
    return output_dir


def run_multi(experiments, output_dir=None):
    """GUI entry point: per-trajectory metric outputs (r50 / prop-within) overlaid across
    MULTIPLE experiments, one series per experiment/group in the all-day scatter/line/violin
    plots. `experiments` is a list of `(experiment_dir, group_label)` pairs.

    Mirrors run(): writes under output_dir (default <first experiment_dir>/plots/buzzswarm/
    per_trajectory_multi/) the combined all_three_windows.csv, the all_day/ scatter+line+violin
    plots+CSVs (now with N series), and per-window per-group CSVs. Like run(), the per-window
    *strip* plots are skipped — plot_window_strip is hardcoded to Female/Male/FruM pairwise
    comparisons and isn't meaningful for arbitrary experiment group labels; their data is still
    written as CSV. Returns the output directory."""
    if not experiments:
        raise ValueError('run_multi requires at least one (experiment_dir, group_label) pair')
    if output_dir is None:
        output_dir = os.path.join(experiments[0][0], 'plots', 'buzzswarm', 'per_trajectory_multi')
    os.makedirs(output_dir, exist_ok=True)

    print('Loading experiments...')
    all_sessions_data = []
    loaded_labels = []
    for experiment_dir, group_label in experiments:
        sd = m.load_or_build_session(experiment_dir, verbose=False)
        if sd is None:
            print(f'  skipped (no session data): {group_label}')
            continue
        sd = dict(sd)  # don't mutate the cached session dict, it may be shared across calls
        sd['sex'] = sd['strain'] = sd['batch'] = group_label
        all_sessions_data.append(sd)
        loaded_labels.append(group_label)
    if not all_sessions_data:
        print('No experiment data loaded. Nothing to plot.')
        return output_dir

    # Temporarily register all loaded groups so the by-sex all-day plot helpers (which iterate
    # m.SEX_ORDER / m.SEX_COLORS) render one series per group; restore afterwards so module
    # globals stay clean for any other caller (mirrors run()'s save/restore pattern).
    saved_order, saved_colors = m.SEX_ORDER, dict(m.SEX_COLORS)
    try:
        m.SEX_ORDER = loaded_labels
        m.SEX_COLORS = {g: _GROUP_PALETTE[i % len(_GROUP_PALETTE)] for i, g in enumerate(loaded_labels)}

        win_dfs = {}
        for label, lo, hi in WINDOWS:
            win_dfs[label] = extract_window_df(all_sessions_data, lo, hi, label)
            print(f'  {label}: {len(win_dfs[label])} trajectories')

        df_3win = pd.concat([d for d in win_dfs.values() if not d.empty],
                            ignore_index=True) if win_dfs else pd.DataFrame()
        write_csv(df_3win, os.path.join(output_dir, 'all_three_windows.csv'))

        df_full = m.extract_per_trajectory_metrics(all_sessions_data)
        print(f'  full all-day: {len(df_full)} trajectories')

        all_day_dir = os.path.join(output_dir, 'all_day')
        os.makedirs(all_day_dir, exist_ok=True)
        for key, ylabel, ylim in METRICS:
            for mode in ('scatter', 'line', 'violin'):
                name_base = f'{mode}_{key}_by_sex_all_day_ZT'
                plot_all_day(df_full, key, ylabel, all_day_dir, name_base, ylim, mode)
                write_csv(df_3win, os.path.join(all_day_dir, f'{name_base}.csv'))

        for label, _, _ in WINDOWS:
            win_dir = os.path.join(output_dir, label)
            os.makedirs(win_dir, exist_ok=True)
            df = win_dfs[label]
            for key, ylabel, ylim in METRICS:
                write_csv(df, os.path.join(win_dir, f'strip_{key}_{label}.csv'))
    finally:
        m.SEX_ORDER, m.SEX_COLORS = saved_order, saved_colors
    print(f'  wrote -> {output_dir}')
    return output_dir
