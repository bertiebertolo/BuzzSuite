"""
Behavioral time-series analysis with custom 04:00–04:00 day windows.

Plots:
1. Activity time-series (L/D shaded, minute 40-50 highlighted)
2. Speaker side proportion over time
3. Filtered versions (activity > threshold)
4. State-separated plots (resting, flying, combined)
5. Box plots (light condition, speaker side vs activity)

Data assumptions:
- timestamp: datetime
- activity: numeric (0-1 or 0-100)
- light_condition: 'L' or 'D'
- speaker_side: 'Left', 'Right', or numeric
- state: 'resting' or 'flying'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DAY_START_HOUR = 4  # 04:00 AM
ACTIVITY_THRESHOLD = 10  # Adjust to your data scale (0-100 or 0-1)
ROLLING_WINDOW = 3  # hours, for smoothing speaker side proportion
MINUTE_HIGHLIGHT_START = 40  # highlight 40-50 minutes within each hour
MINUTE_HIGHLIGHT_END = 50
BINS_PER_HOUR = 1  # 1 = hourly bins; 4 = 15-min bins; etc.

# Color scheme
COLOR_LIGHT = 'lightyellow'
COLOR_DARK = 'lightgray'
COLOR_RESTING = '#1f77b4'  # blue
COLOR_FLYING = '#ff7f0e'   # orange
COLOR_SPEAKER_LEFT = '#2ca02c'  # green
COLOR_SPEAKER_RIGHT = '#d62728'  # red

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def align_to_custom_day(ts):
    """
    Return the start of the 04:00–04:00 day containing the timestamp.

    Example:
        2024-06-05 10:00 → 2024-06-05 04:00
        2024-06-05 02:00 → 2024-06-04 04:00
    """
    if ts.hour < DAY_START_HOUR:
        return (ts - timedelta(days=1)).replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)
    else:
        return ts.replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)


def add_custom_day_column(df):
    """Add a column 'custom_day' for the 04:00-aligned day."""
    df = df.copy()
    df['custom_day'] = df['timestamp'].apply(align_to_custom_day)
    return df


def resample_to_bins(df, interval_minutes=60):
    """
    Resample activity data into equal-width bins aligned to custom day start.

    Returns a new dataframe with columns:
    - bin_start: start of the bin (timestamp)
    - activity_mean: mean activity in the bin
    - activity_std: std dev (for error bars)
    - light_condition: L or D (majority vote)
    - speaker_side: majority
    - state: majority (resting/flying)
    """
    df = df.copy()
    df = add_custom_day_column(df)

    # Create bin labels: time elapsed since custom day start
    df['custom_day_seconds'] = (df['timestamp'] - df['custom_day']).dt.total_seconds()
    bin_width_seconds = interval_minutes * 60
    df['bin_idx'] = (df['custom_day_seconds'] / bin_width_seconds).astype(int)
    df['bin_start'] = df['custom_day'] + pd.to_timedelta(df['bin_idx'] * bin_width_seconds, unit='s')

    # Aggregate by bin
    agg_dict = {
        'activity': ['mean', 'std', 'count'],
        'light_condition': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'speaker_side': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'state': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
    }

    binned = df.groupby('bin_start', as_index=False).agg(agg_dict)
    binned.columns = ['bin_start', 'activity_mean', 'activity_std', 'activity_count',
                      'light_condition', 'speaker_side', 'state']

    return binned


def generate_sample_data(n_hours=48):
    """
    Generate realistic sample behavioral data for testing.

    Returns a DataFrame with timestamp, activity, light_condition, speaker_side, state.
    """
    # Start 48 hours ago
    start_time = datetime.now() - timedelta(hours=n_hours)
    times = pd.date_range(start=start_time, periods=n_hours * 60, freq='1min')

    data = []
    for ts in times:
        hour = ts.hour
        # Light condition: L from 08:00-20:00, D otherwise
        light = 'L' if 8 <= hour < 20 else 'D'

        # Activity: higher during light, circadian rhythm
        base_activity = 40 if light == 'L' else 20
        circadian = 10 * np.sin(2 * np.pi * (hour - 8) / 24)
        noise = np.random.normal(0, 5)
        activity = max(0, min(100, base_activity + circadian + noise))

        # Speaker side: slight bias toward left
        speaker_side = 'Left' if np.random.random() < 0.6 else 'Right'

        # State: mostly resting at night, more flying during day
        if light == 'L' and activity > 30:
            state = 'flying' if np.random.random() < 0.4 else 'resting'
        else:
            state = 'resting' if np.random.random() < 0.7 else 'flying'

        data.append({
            'timestamp': ts,
            'activity': activity,
            'light_condition': light,
            'speaker_side': speaker_side,
            'state': state,
        })

    return pd.DataFrame(data)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_activity_timeseries(binned_df, title="Activity Over Time", ax=None, filtered=False):
    """
    Plot activity time-series with L/D shading and minute 40-50 highlight.

    Parameters:
    - binned_df: resampled data from resample_to_bins()
    - title: plot title
    - ax: matplotlib axis (creates new if None)
    - filtered: if True, only plot rows where activity_mean > ACTIVITY_THRESHOLD
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    if filtered:
        plot_df = binned_df[binned_df['activity_mean'] > ACTIVITY_THRESHOLD].copy()
    else:
        plot_df = binned_df.copy()

    # Add hour-of-day for x-axis labels and L/D backgrounds
    plot_df['hour_of_day'] = plot_df['bin_start'].dt.hour + plot_df['bin_start'].dt.minute / 60

    # Plot L/D background
    for i, row in plot_df.iterrows():
        color = COLOR_LIGHT if row['light_condition'] == 'L' else COLOR_DARK
        ax.axvspan(row['bin_start'], row['bin_start'] + timedelta(hours=1),
                   alpha=0.2, color=color, zorder=0)

    # Plot activity line with error bars
    ax.errorbar(plot_df['bin_start'], plot_df['activity_mean'],
                yerr=plot_df['activity_std'], fmt='o-', linewidth=1.5, markersize=4,
                capsize=3, capthick=1, label='Activity (mean ± std)', color='black', zorder=3)

    # Highlight minute 40-50 regions (vertical bands)
    for bin_start in plot_df['bin_start']:
        minute_start_time = bin_start.replace(minute=MINUTE_HIGHLIGHT_START)
        minute_end_time = bin_start.replace(minute=MINUTE_HIGHLIGHT_END)
        ax.axvspan(minute_start_time, minute_end_time, alpha=0.15, color='red', zorder=1)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Activity Level', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return ax


def plot_speaker_proportion(binned_df, speaker_side_col='speaker_side',
                           title="Speaker Side Proportion", ax=None, filtered=False):
    """
    Plot proportion of speaker side (e.g., Left vs Right) over time with optional smoothing.

    Parameters:
    - binned_df: resampled data
    - speaker_side_col: column name for speaker side
    - title: plot title
    - ax: matplotlib axis
    - filtered: if True, filter by activity threshold
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    if filtered:
        plot_df = binned_df[binned_df['activity_mean'] > ACTIVITY_THRESHOLD].copy()
    else:
        plot_df = binned_df.copy()

    # Compute proportion of "Left" (or first unique value)
    unique_sides = plot_df[speaker_side_col].unique()
    left_side = sorted(unique_sides)[0]  # e.g., 'Left'

    plot_df['left_proportion'] = (plot_df[speaker_side_col] == left_side).astype(float)

    # Apply rolling average if requested
    plot_df['left_proportion_smooth'] = (
        plot_df['left_proportion'].rolling(window=ROLLING_WINDOW, center=True).mean()
    )

    # Plot L/D background
    for i, row in plot_df.iterrows():
        color = COLOR_LIGHT if row['light_condition'] == 'L' else COLOR_DARK
        ax.axvspan(row['bin_start'], row['bin_start'] + timedelta(hours=1),
                   alpha=0.2, color=color, zorder=0)

    # Plot raw and smoothed
    ax.plot(plot_df['bin_start'], plot_df['left_proportion'], 'o-', alpha=0.4,
            linewidth=1, markersize=3, label='Raw', color='gray')
    ax.plot(plot_df['bin_start'], plot_df['left_proportion_smooth'], 's-', linewidth=2,
            markersize=5, label=f'Smoothed ({ROLLING_WINDOW}h window)', color='navy')

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(f'Proportion ({left_side})', fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return ax


def plot_state_separated(binned_df, plot_type='activity', filtered=False):
    """
    Create three subplots: resting only, flying only, combined overlay.

    Parameters:
    - binned_df: resampled data
    - plot_type: 'activity' or 'speaker_proportion'
    - filtered: if True, filter by activity threshold
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Separate by state
    resting_df = binned_df[binned_df['state'] == 'resting'].copy()
    flying_df = binned_df[binned_df['state'] == 'flying'].copy()

    if filtered:
        resting_df = resting_df[resting_df['activity_mean'] > ACTIVITY_THRESHOLD]
        flying_df = flying_df[flying_df['activity_mean'] > ACTIVITY_THRESHOLD]

    if plot_type == 'activity':
        # Resting only
        plot_activity_timeseries(resting_df, title='Activity: Resting Only',
                                ax=axes[0], filtered=False)
        # Flying only
        plot_activity_timeseries(flying_df, title='Activity: Flying Only',
                                ax=axes[1], filtered=False)
        # Combined overlay
        all_data = binned_df if not filtered else binned_df[binned_df['activity_mean'] > ACTIVITY_THRESHOLD]
        plot_activity_timeseries(all_data, title='Activity: Combined (Resting + Flying)',
                                ax=axes[2], filtered=False)

    elif plot_type == 'speaker_proportion':
        plot_speaker_proportion(resting_df, title='Speaker Side: Resting Only',
                               ax=axes[0], filtered=False)
        plot_speaker_proportion(flying_df, title='Speaker Side: Flying Only',
                               ax=axes[1], filtered=False)
        plot_speaker_proportion(all_data, title='Speaker Side: Combined',
                               ax=axes[2], filtered=False)

    plt.tight_layout()
    return fig, axes


def plot_boxplots(binned_df, filtered=False):
    """
    Create box plots comparing light condition and speaker side vs activity.

    Creates two main subplots:
    1. Light (L vs D) vs Activity (optionally separated by state)
    2. Speaker side vs Activity (optionally separated by state)
    """
    if filtered:
        plot_df = binned_df[binned_df['activity_mean'] > ACTIVITY_THRESHOLD].copy()
    else:
        plot_df = binned_df.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Light vs Activity (combined)
    sns.boxplot(data=plot_df, x='light_condition', y='activity_mean', ax=axes[0, 0],
                palette={('L' if plot_df['light_condition'].iloc[0] == 'L' else 'D'): COLOR_LIGHT})
    axes[0, 0].set_title('Activity by Light Condition', fontweight='bold')
    axes[0, 0].set_ylabel('Activity')
    axes[0, 0].set_xlabel('Light Condition (L=Light, D=Dark)')

    # 2. Light vs Activity (separated by state)
    sns.boxplot(data=plot_df, x='light_condition', y='activity_mean', hue='state', ax=axes[0, 1],
                palette={'resting': COLOR_RESTING, 'flying': COLOR_FLYING})
    axes[0, 1].set_title('Activity by Light Condition (separated by state)', fontweight='bold')
    axes[0, 1].set_ylabel('Activity')
    axes[0, 1].set_xlabel('Light Condition')
    axes[0, 1].legend(title='State', fontsize=9)

    # 3. Speaker side vs Activity (combined)
    sns.boxplot(data=plot_df, x='speaker_side', y='activity_mean', ax=axes[1, 0])
    axes[1, 0].set_title('Activity by Speaker Side', fontweight='bold')
    axes[1, 0].set_ylabel('Activity')
    axes[1, 0].set_xlabel('Speaker Side')

    # 4. Speaker side vs Activity (separated by state)
    sns.boxplot(data=plot_df, x='speaker_side', y='activity_mean', hue='state', ax=axes[1, 1],
                palette={'resting': COLOR_RESTING, 'flying': COLOR_FLYING})
    axes[1, 1].set_title('Activity by Speaker Side (separated by state)', fontweight='bold')
    axes[1, 1].set_ylabel('Activity')
    axes[1, 1].set_xlabel('Speaker Side')
    axes[1, 1].legend(title='State', fontsize=9)

    plt.tight_layout()
    return fig, axes


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis(df, output_dir='./plots'):
    """
    Run full analysis pipeline: generate all plots.

    Parameters:
    - df: input dataframe with columns: timestamp, activity, light_condition, speaker_side, state
    - output_dir: directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Resampling data to 1-hour bins aligned to 04:00 day start...")
    binned = resample_to_bins(df, interval_minutes=60)
    print(f"  Binned data shape: {binned.shape}")

    # Plot 1: Activity time-series (all data + filtered)
    print("\nGenerating activity time-series plots...")
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_activity_timeseries(binned, title='Activity Over Time (All Data)', ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '01_activity_timeseries_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    plot_activity_timeseries(binned, title=f'Activity Over Time (Filtered: > {ACTIVITY_THRESHOLD})',
                            ax=ax, filtered=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '02_activity_timeseries_filtered.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Speaker side proportion (all data + filtered)
    print("Generating speaker proportion plots...")
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_speaker_proportion(binned, title='Speaker Side Proportion (All Data)', ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '03_speaker_proportion_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    plot_speaker_proportion(binned, title=f'Speaker Side Proportion (Filtered: > {ACTIVITY_THRESHOLD})',
                           ax=ax, filtered=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '04_speaker_proportion_filtered.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: State-separated activity
    print("Generating state-separated activity plots...")
    fig, axes = plot_state_separated(binned, plot_type='activity', filtered=False)
    fig.savefig(os.path.join(output_dir, '05_activity_by_state_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plot_state_separated(binned, plot_type='activity', filtered=True)
    fig.savefig(os.path.join(output_dir, '06_activity_by_state_filtered.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 4: State-separated speaker proportion
    print("Generating state-separated speaker proportion plots...")
    fig, axes = plot_state_separated(binned, plot_type='speaker_proportion', filtered=False)
    fig.savefig(os.path.join(output_dir, '07_speaker_by_state_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plot_state_separated(binned, plot_type='speaker_proportion', filtered=True)
    fig.savefig(os.path.join(output_dir, '08_speaker_by_state_filtered.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 5: Box plots
    print("Generating box plots...")
    fig, axes = plot_boxplots(binned, filtered=False)
    fig.savefig(os.path.join(output_dir, '09_boxplots_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plot_boxplots(binned, filtered=True)
    fig.savefig(os.path.join(output_dir, '10_boxplots_filtered.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✓ All plots saved to {output_dir}/")
    print(f"  Generated 10 figure files (activity, speaker side, state-separated, box plots)")


if __name__ == '__main__':
    # Generate sample data for demonstration
    print("Generating sample data...")
    sample_df = generate_sample_data(n_hours=72)  # 3 days
    print(f"Sample data shape: {sample_df.shape}")
    print(f"Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")

    # Run full analysis
    run_analysis(sample_df, output_dir='./behavioral_plots')
    print("\n✓ Analysis complete!")
