#!/usr/bin/env python3
"""Per-trajectory percentile-radius CSV for the 3 named ZT 30-min windows.

Output:
  <experiment_dir>/plots/buzzswarm/sholl_per_trajectory.csv

Columns:
  Sex, date_genotype, ZT, traj_id, n_frames, r_1, r_2, ..., r_100

One row per trajectory. r_p = the radius (px) containing p% of that
trajectory's in-window frames (= p-th percentile of distance-to-centre,
linear interpolation). r_50 = median distance. Same trajectory population
as all_three_windows.csv.
"""

import os
import sys

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

WINDOWS = [
    ('ZT0.0-0.5',    0.0,  0.5),
    ('ZT5.75-6.25',  5.75, 6.25),
    ('ZT11.5-12.0', 11.5, 12.0),
]

PERCENTILES = np.arange(1, 101)            # 1, 2, ..., 100
R_COLS = [f'r_{int(p)}' for p in PERCENTILES]
CSV_COLS = ['Sex', 'date_genotype', 'ZT', 'traj_id', 'n_frames'] + R_COLS


def build_sholl_rows(all_sessions_data):
    """Build the per-trajectory percentile-radius rows for the 3 ZT windows from a list of
    session dicts (as returned by process_session / process_session_from_dir). Shared by the
    CLI main() and the GUI run() so there is one implementation of the metric."""
    rows = []
    for label, zt_lo, zt_hi in WINDOWS:
        n_win = 0
        for sd in all_sessions_data:
            sex, batch = sd['sex'], sd['batch']
            sid = f'{sex}_{batch}'
            centroid = sd['cage_centroid']
            if (sd['last_ts'] - sd['first_ts']).total_seconds() <= 0:
                continue
            for idx, fl in enumerate(m.flights_in_zt_window(sd, zt_lo, zt_hi), start=1):
                dists = np.array([m.distance_to_center(p, centroid)
                                  for p in fl['coords']])
                if len(dists) < 2:
                    continue
                # r_p = radius (px) containing p% of frames = p-th percentile
                radii_p = np.percentile(dists, PERCENTILES)
                row = {
                    'Sex':           sex,
                    'date_genotype': f'{batch}_{sex}',
                    'ZT':            label,
                    'traj_id':       f'{sid}_{label}_{idx:03d}',
                    'n_frames':      int(len(dists)),
                }
                for col, val in zip(R_COLS, radii_p):
                    row[col] = float(val)
                rows.append(row)
                n_win += 1
        print(f'  {label}: {n_win} trajectories')
    return rows


def run(experiment_dir, output_dir=None):
    """GUI entry point: per-trajectory Sholl percentile-radius CSV for one experiment folder.

    Loads the single experiment via process_session_from_dir (no BASE_PATH/SESSIONS globals)
    and writes sholl_per_trajectory.csv under output_dir, defaulting to
    <experiment_dir>/plots/buzzswarm/. Returns the output CSV path."""
    print('Loading experiment...')
    sd = m.load_or_build_session(experiment_dir, verbose=False)   # cached session (plots/sholl_cache.pkl)
    all_sessions_data = [sd] if sd is not None else []
    rows = build_sholl_rows(all_sessions_data)

    df = pd.DataFrame(rows, columns=CSV_COLS)
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, 'plots', 'buzzswarm')
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'sholl_per_trajectory.csv')
    df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f'wrote {out_csv}  ({len(df)} rows x {len(df.columns)} cols)')
    return out_csv
