#!/usr/bin/env python3
"""Plots for clock_analysis outputs (BuzzSwarm, real clock time, LD days).

Every function draws from the DataFrames that clock_analysis.run() / load_outputs() return, so
re-plotting never recomputes. Figures are plain matplotlib Figure objects (no pyplot state), safe
to build in a worker thread. A "unit" is one experiment-date; group lines are the mean across a
group's units, with the units themselves drawn faintly behind.

  timecourse            one metric through the day: overlay (00:00-24:00, group mean +- SEM),
                        individual (a panel per experiment-date) or continuous (real calendar
                        time, a row per experiment)
  window_summary        each metric per clock window, one point per day, line = group mean
  swarm_timecourse      swarm size minute by minute around lights-on and lights-off
  swarm_distribution    how often each swarm size occurs per window, against independent joining
  swarm_vs_flying       swarm size and share of flyers in the swarm at the same number flying
  centre_profile        real-flight density vs distance from the centre (filled swarm vs ring)
  heatmaps              where the real flight is, per window and group

render_all() writes each figure as PNG plus the numbers it shows as CSV.
"""

import math
import os
import re
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.dates as mdates

try:
    from buzzswarm import clock_analysis as ca
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import clock_analysis as ca
try:
    import ld_dd
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ld_dd


GROUP_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MODES = ('overlay', 'individual', 'continuous')
DEFAULT_METRICS = ['swarm_mean', 'n_flying', 'shape_index', 'speed_px_s']
WINDOW_METRICS = ['swarm_mean', 'swarm_ge3_frac', 'n_flying', 'shape_index', 'core_density',
                  'central_enrichment', 'speed_px_s', 'turning_deg_s', 'nn_px', 'near_ratio',
                  'r50', 'straightness']
LABELS = dict(ca.METRICS)
LABELS['prop_within_100'] = 'Fraction of flight within 100 px of centre'
REF_LINES = {'near_ratio': 1.0, 'central_enrichment': 1.0, 'core_density': 1.0}
SWARM_SMOOTH_MIN = 5            # rolling minutes for the swarm time course
MERGE_NOTE = ('Two flyers closer than ~30 px are seen as one, so large swarms are undercounted.')


# ==== helpers ====

def group_colors(groups):
    return {g: GROUP_PALETTE[i % len(GROUP_PALETTE)] for i, g in enumerate(groups)}


def _groups(df):
    return list(dict.fromkeys(df['group']))


def _units(df):
    """(experiment, group, date, ld_dd) per experiment-date, in table order."""
    cols = ['experiment', 'group', 'date', 'ld_dd']
    return [tuple(r) for r in df[cols].drop_duplicates().itertuples(index=False)]


def _window_order(labels):
    def start(label):
        try:
            return ca.parse_window(label)[1]
        except ValueError:
            return 1e9
    return sorted(dict.fromkeys(labels), key=start)


def _label(metric):
    return LABELS.get(metric, metric)


def _n_days(n):
    return f'n = {n} day' if n == 1 else f'n = {n} days'


def _bin_hours(df):
    t = np.sort(df['tod_h'].unique())
    return float(np.median(np.diff(t))) if len(t) > 1 else 1.0


def _mean_sem(wide):
    """Row-wise mean, SEM (NaN where < 2 values) and count of a units-as-columns frame."""
    n = wide.count(axis=1)
    mean = wide.mean(axis=1)
    sem = wide.std(axis=1, ddof=1) / np.sqrt(n.where(n >= 2))
    return mean, sem, n


def _hour_axis(ax, lo=0.0, hi=24.0):
    ax.set_xlim(lo, hi)
    step = 1 if hi - lo <= 6 else 2
    ax.set_xticks(np.arange(math.ceil(lo), math.floor(hi) + 1, step))
    ax.set_xlabel('Time of day (h)')
    ax.grid(alpha=0.3)


def _finish_y(ax, metric):
    if metric in REF_LINES:
        ax.axhline(REF_LINES[metric], color='0.4', ls=':', lw=1)
    lo, hi = ax.get_ylim()
    if lo < 0 <= hi and metric != 'near_ratio':
        ax.set_ylim(bottom=0)


def _lights_off_on(ax, night):
    for h in night:
        ax.axvline(h, color='k', ls='--', lw=0.8)


def _poisson_pmf(k, lam):
    k = np.asarray(k, dtype=float)
    if lam <= 0:
        return (k == 0).astype(float)
    lg = np.array([math.lgamma(v + 1) for v in k])
    return np.exp(k * math.log(lam) - lam - lg)


def _is_night(tod_h, night):
    tod_h = np.asarray(tod_h, dtype=float)
    out = np.zeros(len(tod_h), dtype=bool)
    for lo, hi in ld_dd.night_spans(*night):
        out |= (tod_h >= lo) & (tod_h < hi)
    return out


def safe_name(text):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(text)).strip('_') or 'group'


def comparison_dir(root, groups):
    """{root}/buzzswarm_comparison/{GroupA}__{GroupB}: one folder per set of groups compared."""
    return os.path.join(root, 'buzzswarm_comparison', '__'.join(safe_name(g) for g in groups))


# ==== metric through the day ====

def timecourse(bins, metric, mode='overlay', night=ld_dd.DEFAULT_NIGHT):
    """(fig, summary) or (None, None) when the metric has no data."""
    if bins.empty or metric not in bins or bins[metric].notna().sum() == 0:
        return None, None
    if mode == 'individual':
        return _timecourse_individual(bins, metric, night)
    if mode == 'continuous':
        return _timecourse_continuous(bins, metric, night)
    return _timecourse_overlay(bins, metric, night)


def _timecourse_overlay(bins, metric, night):
    colors = group_colors(_groups(bins))
    half = _bin_hours(bins) / 2.0
    fig = Figure(figsize=(10, 5.2), layout='constrained')
    ax = fig.add_subplot(1, 1, 1)
    ld_dd.shade_night(ax, *night)
    rows = []
    for g, sub in bins.groupby('group', sort=False):
        wide = sub.pivot_table(index='tod_h', columns=['experiment', 'date'], values=metric, dropna=False)
        x = wide.index.values + half
        for col in wide.columns:
            ax.plot(x, wide[col].values, color=colors[g], lw=0.8, alpha=0.35)
        mean, sem, n = _mean_sem(wide)
        ax.plot(x, mean.values, color=colors[g], lw=2.2, label=f'{g} ({_n_days(wide.shape[1])})')
        ax.fill_between(x, (mean - sem).values, (mean + sem).values, color=colors[g], alpha=0.2, lw=0)
        rows.append(pd.DataFrame({'group': g, 'tod_h': wide.index.values, 'mean': mean.values,
                                  'sem': sem.values, 'n_days': n.values}))
    _hour_axis(ax)
    _finish_y(ax, metric)
    ax.set_ylabel(_label(metric))
    ax.set_title(f'{_label(metric)} - overlay of days (thick = group mean +- SEM, '
                 f'thin = single days; night shaded)', fontsize=10)
    ax.legend(fontsize=8)
    return fig, pd.concat(rows, ignore_index=True)


def _timecourse_individual(bins, metric, night):
    colors = group_colors(_groups(bins))
    half = _bin_hours(bins) / 2.0
    units = _units(bins)
    ncols = 1 if len(units) == 1 else 2
    nrows = int(math.ceil(len(units) / float(ncols)))
    fig = Figure(figsize=(6.5 * ncols, 2.6 * nrows + 0.6), layout='constrained')
    axes = fig.subplots(nrows, ncols, squeeze=False, sharey=True)
    flat = [a for row in axes for a in row]
    for i, (ax, (exp, g, date, tag)) in enumerate(zip(flat, units)):
        sub = bins[(bins.experiment == exp) & (bins.date == date)].sort_values('tod_h')
        ld_dd.shade_day(ax, night[0], night[1], tag)
        ax.plot(sub.tod_h.values + half, sub[metric].values, color=colors[g], lw=1.4)
        _hour_axis(ax)
        ax.set_title(f'{exp}  {date}  [{g}]', fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel(_label(metric), fontsize=7)
    for ax in flat[len(units):]:
        ax.axis('off')
    _finish_y(flat[0], metric)
    fig.suptitle(f'{_label(metric)} - each day (night shaded)', fontsize=11)
    summary = bins[['experiment', 'group', 'date', 'tod_h', metric]].copy()
    return fig, summary


def _timecourse_continuous(bins, metric, night):
    colors = group_colors(_groups(bins))
    half = pd.Timedelta(hours=_bin_hours(bins) / 2.0)
    exps = list(dict.fromkeys(bins['experiment']))
    fig = Figure(figsize=(11, 2.8 * len(exps) + 0.6), layout='constrained')
    axes = fig.subplots(len(exps), 1, squeeze=False, sharey=True)[:, 0]
    for ax, exp in zip(axes, exps):
        sub = bins[bins.experiment == exp].sort_values('bin_start')
        g = sub['group'].iloc[0]
        for date, tag in sub[['date', 'ld_dd']].drop_duplicates().itertuples(index=False):
            ld_dd.shade_real_span(ax, pd.Timestamp(date), night[0], night[1], tag)
        x = pd.to_datetime(sub['bin_start']) + half
        y = sub[metric].astype(float).values.copy()
        # break the line across days that are not consecutive
        gap = np.r_[False, np.diff(x.values).astype('timedelta64[m]').astype(float) > 3 * half.total_seconds() / 60.0]
        xs, ys = [], []
        for xi, yi, gi in zip(x, y, gap):
            if gi:
                xs.append(xi)
                ys.append(np.nan)
            xs.append(xi)
            ys.append(yi)
        ax.plot(xs, ys, color=colors[g], lw=1.3)
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.margins(x=0)
        ax.grid(alpha=0.3)
        ax.set_title(f'{exp}  [{g}]', fontsize=9)
        ax.set_ylabel(_label(metric), fontsize=7)
    _finish_y(axes[0], metric)
    fig.suptitle(f'{_label(metric)} - continuous (night shaded)', fontsize=11)
    summary = bins[['experiment', 'group', 'date', 'bin_start', metric]].copy()
    return fig, summary


# ==== windows ====

def window_summary(windows, metrics=None):
    """A panel per metric: x = clock window, one point per day, short line = group mean."""
    if windows.empty:
        return None, None
    metrics = [m for m in (metrics or WINDOW_METRICS) if m in windows and windows[m].notna().any()]
    if not metrics:
        return None, None
    groups = _groups(windows)
    colors = group_colors(groups)
    order = _window_order(windows['window'])
    ncols = min(4, len(metrics))
    nrows = int(math.ceil(len(metrics) / float(ncols)))
    fig = Figure(figsize=(3.9 * ncols, 3.3 * nrows + 0.8), layout='constrained')
    axes = fig.subplots(nrows, ncols, squeeze=False)
    flat = [a for row in axes for a in row]
    width = 0.8 / max(1, len(groups))
    rows = []
    legend = {}
    for ax, metric in zip(flat, metrics):
        for gi, g in enumerate(groups):
            for wi, w in enumerate(order):
                vals = windows.loc[(windows.group == g) & (windows.window == w), metric].dropna().values
                if not len(vals):
                    continue
                x0 = wi - 0.4 + width * (gi + 0.5)
                jitter = np.linspace(-width * 0.25, width * 0.25, len(vals)) if len(vals) > 1 else [0.0]
                pts = ax.scatter(x0 + np.asarray(jitter), vals, s=18, color=colors[g], zorder=3)
                legend.setdefault(g, pts)
                ax.hlines(vals.mean(), x0 - width * 0.4, x0 + width * 0.4, color=colors[g], lw=2.2)
                rows.append({'metric': metric, 'window': w, 'group': g, 'mean': float(vals.mean()),
                             'sd': float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                             'n_days': int(len(vals)),
                             'values': ' '.join('%.4g' % v for v in vals)})
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, fontsize=7)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(alpha=0.3, axis='y')
        _finish_y(ax, metric)
        ax.set_title(_label(metric), fontsize=8)
    for ax in flat[len(metrics):]:
        ax.axis('off')
    fig.legend(list(legend.values()), list(legend.keys()), loc='outside lower center',
               ncol=len(groups), fontsize=8)
    fig.suptitle('Per clock window: one point per day, line = group mean', fontsize=11)
    return fig, pd.DataFrame(rows)


# ==== swarm size ====

def _minute_wide(sub, col, smooth):
    """Units-as-columns frame of `col` per clock minute (0..1439), rolling-smoothed per unit."""
    out = {}
    for (exp, date), sd in sub.groupby(['experiment', 'date'], sort=False):
        s = pd.Series(sd[col].values, index=np.rint(sd['tod_h'].values * 60).astype(int))
        s = s[~s.index.duplicated()].reindex(range(1440))
        out[(exp, date)] = s.rolling(smooth, center=True, min_periods=1).mean() if smooth > 1 else s
    return pd.DataFrame(out)


def swarm_timecourse(minutes, night=ld_dd.DEFAULT_NIGHT, smooth=SWARM_SMOOTH_MIN):
    """Swarm size per minute around lights-on and lights-off: mean in the swarm, and the
    largest swarm in that minute."""
    if minutes.empty:
        return None, None
    colors = group_colors(_groups(minutes))
    off, on = night
    spans = [(on - 0.5, on + 2.0, 'Lights on (%02d:00)' % on), (off - 2.5, off + 0.5, 'Lights off (%02d:00)' % off)]
    radius = int(minutes['swarm_radius_px'].iloc[0])
    fig = Figure(figsize=(14, 7.5), layout='constrained')
    axes = fig.subplots(2, 2, squeeze=False)
    summary = {}
    for row, (col, ylab) in enumerate([('swarm_mean', 'Swarm size (mean in the swarm at once)'),
                                       ('swarm_max', 'Largest swarm in that minute')]):
        for g, sub in minutes.groupby('group', sort=False):
            wide = _minute_wide(sub, col, smooth)
            x = wide.index.values / 60.0
            mean, _sem, n = _mean_sem(wide)
            s = summary.setdefault(g, pd.DataFrame({'group': g, 'tod_h': x, 'n_days': n.values}))
            s[col] = mean.values
            for c, (lo, hi, _title) in enumerate(spans):
                ax = axes[row, c]
                sel = (x >= lo) & (x <= hi)
                for unit in wide.columns:
                    ax.plot(x[sel], wide[unit].values[sel], color=colors[g], lw=0.8, alpha=0.35)
                ax.plot(x[sel], mean.values[sel], color=colors[g], lw=2.2,
                        label=f'{g} ({_n_days(wide.shape[1])})')
        for c, (lo, hi, title) in enumerate(spans):
            ax = axes[row, c]
            _lights_off_on(ax, night)
            _hour_axis(ax, lo, hi)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f'{title}: thick = group mean, thin = single days ({smooth}-min rolling)',
                             fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Swarm size = mosquitoes in real flight inside the {radius} px centre zone '
                 f'at the same moment. {MERGE_NOTE}', fontsize=10)
    return fig, pd.concat(summary.values(), ignore_index=True)


def swarm_distribution(swarm_size, minutes, max_size=15):
    """Per window: share of frames with each swarm size (group mean of days), with the
    distribution expected if mosquitoes joined independently (a Poisson mixture over the
    window's minute means)."""
    if swarm_size.empty:
        return None, None
    colors = group_colors(_groups(swarm_size))
    order = _window_order(swarm_size['window'])
    ncols = min(4, len(order))
    nrows = int(math.ceil(len(order) / float(ncols)))
    fig = Figure(figsize=(5.2 * ncols, 4.0 * nrows + 0.7), layout='constrained')
    axes = fig.subplots(nrows, ncols, squeeze=False)
    flat = [a for row in axes for a in row]
    rows = []
    for ax, w in zip(flat, order):
        _, lo_min, hi_min = ca.parse_window(w)
        top = 1
        for g in _groups(swarm_size):
            sub = swarm_size[(swarm_size.group == g) & (swarm_size.window == w)]
            if sub.empty:
                continue
            obs = sub.groupby('swarm_size').share_of_frames.mean()
            k = obs.index.values
            mm = minutes[(minutes.group == g) & (minutes.tod_h * 60 >= lo_min - 1e-6)
                         & (minutes.tod_h * 60 < hi_min - 1e-6)]
            lams = mm.swarm_mean.dropna().values
            indep = (np.mean([_poisson_pmf(k, lam) for lam in lams], axis=0) if len(lams)
                     else np.full(len(k), np.nan))
            nz = k[obs.values > 0]
            top = max(top, int(nz.max()) if len(nz) else 0)
            sel = k <= max_size
            ax.semilogy(k[sel], np.where(obs.values[sel] > 0, obs.values[sel], np.nan), '-o',
                        color=colors[g], ms=3.5, lw=1.8, label=f'{g}')
            ax.semilogy(k[sel], np.where(indep[sel] > 0, indep[sel], np.nan), ':', color=colors[g], lw=1.5)
            rows.append(pd.DataFrame({'window': w, 'group': g, 'swarm_size': k, 'observed': obs.values,
                                      'independent_joining': indep}))
        ax.set_xlim(-0.3, min(max_size, top + 1) + 0.3)
        ax.set_ylim(1e-5, 1.2)
        ax.grid(alpha=0.3, which='both')
        ax.set_xlabel('Swarm size (in flight in the centre at once)')
        ax.set_ylabel('Share of frames')
        ax.set_title(w, fontsize=10)
    for ax in flat[len(order):]:
        ax.axis('off')
    flat[0].legend(fontsize=8, title='solid = observed, dotted = if joining independently',
                   title_fontsize=7)
    fig.suptitle('How often each swarm size occurs (mean of days). ' + MERGE_NOTE, fontsize=10)
    return fig, (pd.concat(rows, ignore_index=True) if rows else None)


def swarm_vs_flying(minutes, night=ld_dd.DEFAULT_NIGHT, bin_width=0.5, min_minutes=10):
    """Lights-on minutes: swarm size against the number in flight, and the share of flyers in
    the swarm at the same number flying (is a group more drawn to the centre, or just more
    active?)."""
    if minutes.empty:
        return None, None
    day = minutes[~_is_night(minutes['tod_h'].values, night)]
    day = day[day.n_flying.notna() & day.swarm_mean.notna()]
    if day.empty:
        return None, None
    colors = group_colors(_groups(day))
    edges = np.arange(0, float(day.n_flying.max()) + bin_width, bin_width)
    fig = Figure(figsize=(13, 5.2), layout='constrained')
    ax1, ax2 = fig.subplots(1, 2)
    rows = []
    for g, sub in day.groupby('group', sort=False):
        ax1.scatter(sub.n_flying, sub.swarm_mean, s=4, color=colors[g], alpha=0.25, lw=0)
        idx = np.digitize(sub.n_flying.values, edges)
        for i in range(1, len(edges)):
            s = sub[idx == i]
            if len(s) < min_minutes:
                continue
            rows.append({'group': g, 'n_flying_lo': edges[i - 1], 'n_flying_hi': edges[i],
                         'n_minutes': len(s), 'n_flying': s.n_flying.mean(),
                         'swarm_mean': s.swarm_mean.mean(),
                         'share_in_swarm': s.swarm_mean.sum() / s.n_flying.sum()})
        r = pd.DataFrame([x for x in rows if x['group'] == g])
        if not r.empty:
            ax1.plot(r.n_flying, r.swarm_mean, '-o', color=colors[g], lw=2, ms=4, label=g)
            ax2.plot(r.n_flying, r.share_in_swarm, '-o', color=colors[g], lw=2, ms=4, label=g)
    ax1.set_xlabel('Mosquitoes in flight (minute mean)')
    ax1.set_ylabel('Swarm size (minute mean)')
    ax1.set_title('Each dot = 1 minute with lights on; line = mean per 0.5 flyers', fontsize=9)
    ax2.set_xlabel('Mosquitoes in flight (minute mean)')
    ax2.set_ylabel('Share of flyers in the swarm zone')
    ax2.set_title('Same number flying -> what share is in the swarm?', fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.suptitle('Swarm size vs number in flight (lights on)', fontsize=11)
    return fig, pd.DataFrame(rows)


# ==== where they fly ====

def centre_profile(profile):
    """Per window: real-flight density per unit cage area vs distance from the centre."""
    if profile.empty:
        return None, None
    colors = group_colors(_groups(profile))
    order = _window_order(profile['window'])
    fig = Figure(figsize=(4.6 * len(order), 4.4), layout='constrained')
    axes = fig.subplots(1, len(order), squeeze=False, sharey=True)[0]
    rows = []
    for ax, w in zip(axes, order):
        for g, sub in profile[profile.window == w].groupby('group', sort=False):
            sub = sub.assign(r_mid=(sub.r_lo_px + sub.r_hi_px) / 2.0)
            wide = sub.pivot_table(index='r_mid', columns=['experiment', 'date'], values='density')
            for col in wide.columns:
                ax.plot(wide.index, wide[col].values, color=colors[g], lw=0.8, alpha=0.35)
            mean, sem, n = _mean_sem(wide)
            ax.plot(wide.index, mean.values, color=colors[g], lw=2.2, label=g)
            rows.append(pd.DataFrame({'window': w, 'group': g, 'r_mid_px': wide.index.values,
                                      'density': mean.values, 'sem': sem.values, 'n_days': n.values}))
        ax.axhline(1, color='0.4', ls=':', lw=1)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.set_xlabel('Distance from cage centre (px)')
        ax.set_title(w, fontsize=10)
    axes[0].set_ylabel('Real-flight density per unit cage area\n(1 = spread evenly)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Filled swarm (highest at the centre) vs ring (hollow centre); '
                 'thick = group mean, thin = single days', fontsize=10)
    return fig, pd.concat(rows, ignore_index=True)


def heatmaps(heatmap, geometry=None):
    """Rows = windows, columns = groups: share of real flight per cell around the cage centre
    (mean of the group's days; image orientation, camera below). Cage outline and the swarm
    zone are drawn on top. Colours are shared along each row."""
    if heatmap.empty:
        return None, None
    geometry = geometry or {}
    groups = _groups(heatmap)
    order = _window_order(heatmap['window'])
    cell = ca.HEAT_CELL_PX
    half = float(max(heatmap.x_px.abs().max(), heatmap.y_px.abs().max())) + cell / 2.0
    edges = np.arange(-half, half + cell / 2.0, cell)
    n = len(edges) - 1
    fig = Figure(figsize=(3.6 * len(groups) + 1.0, 3.4 * len(order) + 0.6), layout='constrained')
    axes = fig.subplots(len(order), len(groups), squeeze=False)
    rows = []
    for ri, w in enumerate(order):
        grids = {}
        for g in groups:
            sub = heatmap[(heatmap.group == g) & (heatmap.window == w)]
            units = sub[['experiment', 'date']].drop_duplicates()
            if units.empty:
                continue
            ix = np.clip(((sub.x_px.values + half) // cell).astype(int), 0, n - 1)
            iy = np.clip(((sub.y_px.values + half) // cell).astype(int), 0, n - 1)
            grid = np.zeros((n, n))
            np.add.at(grid, (iy, ix), sub.share.values)
            grids[g] = grid / float(len(units))
            yy, xx = np.nonzero(grids[g])
            rows.append(pd.DataFrame({'window': w, 'group': g, 'x_px': edges[xx] + cell / 2.0,
                                      'y_px': edges[yy] + cell / 2.0, 'share': grids[g][yy, xx],
                                      'n_days': len(units)}))
        vmax = max([gr.max() for gr in grids.values()] + [1e-9])
        im = None
        for ci, g in enumerate(groups):
            ax = axes[ri, ci]
            ax.set_xticks([])
            ax.set_yticks([])
            if g not in grids:
                ax.text(0.5, 0.5, 'too little flight', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='0.5')
                ax.set_title(f'{w}  |  {g}', fontsize=9)
                continue
            im = ax.imshow(grids[g], origin='upper', extent=[-half, half, half, -half],
                           cmap='magma', vmin=0, vmax=vmax, interpolation='nearest')
            for exp in heatmap.loc[heatmap.group == g, 'experiment'].unique():
                geo = geometry.get(exp)
                if not geo:
                    continue
                cx, cy = geo['centroid']
                poly = np.asarray(geo['polygon'], dtype=float) - [cx, cy]
                poly = np.vstack([poly, poly[:1]])
                ax.plot(poly[:, 0], poly[:, 1], color='w', lw=0.8, alpha=0.7)
                t = np.linspace(0, 2 * np.pi, 120)
                rz = geo.get('swarm_radius_px', ca.SWARM_RADIUS_PX)
                ax.plot(rz * np.cos(t), rz * np.sin(t), color='c', lw=0.8, ls='--')
            ax.set_xlim(-half, half)
            ax.set_ylim(half, -half)
            ax.set_title(f'{w}  |  {g}', fontsize=9)
        if im is not None:
            fig.colorbar(im, ax=list(axes[ri, :]), shrink=0.85, label='Share of real flight per cell')
    fig.suptitle('Where the real flight is (seen from below; cyan = swarm zone, white = cage border)',
                 fontsize=10)
    return fig, (pd.concat(rows, ignore_index=True) if rows else None)


# ==== everything ====

def render_all(tables, out_dir, mode='overlay', metrics=None, night=ld_dd.DEFAULT_NIGHT, dpi=120,
               progress=None, cancel=None):
    """Draw every plot from run()/load_outputs() tables into out_dir (PNG + CSV of the numbers
    shown). Returns the PNG paths, headline swarm plots first. progress(fraction, message);
    a set `cancel` Event raises clock_analysis.Cancelled between plots."""
    os.makedirs(out_dir, exist_ok=True)
    metrics = metrics or DEFAULT_METRICS
    empty = pd.DataFrame()
    t = {k: tables.get(k, empty) for k in ('bins', 'windows', 'swarm_size', 'minutes', 'profile', 'heatmap')}
    jobs = [
        ('swarm_timecourse', lambda: swarm_timecourse(t['minutes'], night)),
        ('swarm_size_distribution', lambda: swarm_distribution(t['swarm_size'], t['minutes'])),
        ('swarm_vs_flying', lambda: swarm_vs_flying(t['minutes'], night)),
        ('heatmaps', lambda: heatmaps(t['heatmap'], tables.get('geometry'))),
        ('centre_profile', lambda: centre_profile(t['profile'])),
        ('window_summary', lambda: window_summary(t['windows'])),
    ]
    for metric in metrics:
        jobs.append((f'timecourse_{metric}_{mode}',
                     lambda metric=metric: timecourse(t['bins'], metric, mode, night)))
    paths = []
    for i, (name, make) in enumerate(jobs):
        if cancel is not None and cancel.is_set():
            raise ca.Cancelled()
        if progress:
            progress(i / float(len(jobs)), f'plotting {name}')
        fig, summary = make()
        if fig is None:
            continue
        path = os.path.join(out_dir, name + '.png')
        fig.savefig(path, dpi=dpi)
        if summary is not None:
            summary.to_csv(os.path.join(out_dir, name + '.csv'), index=False)
        paths.append(path)
    if progress:
        progress(1.0, f'{len(paths)} plots written')
    return paths
