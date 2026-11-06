"""Light schedule shared by the Activity and BuzzSwarm (Aggregation) tabs.

Each experiment keeps its light schedule in ld_dd_schedule.json (sibling to custom_zones.json):
night start/end hours, which calendar dates are included, and each date's LD (light:dark) or DD
(constant dark) tag. Both tabs read and write the same file: the night hours are shared, while
each tab keeps its own included dates (Activity: 'included_dates', BuzzSwarm: BUZZSWARM_DAYS_KEY). The shading helpers draw the night window on a 0-24 h axis or at a date's real
calendar position; DD dates are shaded across the whole day (no "day" portion to leave unshaded).
"""
import json
import os

import pandas as pd

NIGHT_COLOR = '#C8C8C8'
NIGHT_ALPHA = 0.30
SCHEDULE_FILE = 'ld_dd_schedule.json'
DEFAULT_NIGHT = (17, 5)
BUZZSWARM_DAYS_KEY = 'buzzswarm_included_dates'   # BuzzSwarm's own day selection (Activity uses
                                                  # 'included_dates'; only the night hours are shared)


def night_spans(night_start, night_end):
    """(lo, hi) hour spans of the night window on a 0-24h axis. Wraps midnight when
    night_start > night_end (the usual 17..05 case)."""
    if night_start > night_end:
        return [(night_start, 24.0), (0.0, night_end)]
    return [(night_start, night_end)]


def shade_night(ax, night_start, night_end):
    """Shade the night window (fixed gray band) on a time-of-day (0-24h) axis."""
    for lo, hi in night_spans(night_start, night_end):
        if lo < hi:
            ax.axvspan(lo, hi, color=NIGHT_COLOR, alpha=NIGHT_ALPHA, linewidth=0, zorder=0)


def shade_day(ax, night_start, night_end, tag):
    """One date's 0-24h panel: DD shades the whole panel, LD only the night window."""
    if tag == 'DD':
        ax.axvspan(0, 24, color=NIGHT_COLOR, alpha=NIGHT_ALPHA, linewidth=0, zorder=0)
    else:
        shade_night(ax, night_start, night_end)


def shade_real_span(ax, day_start, night_start, night_end, tag):
    """Same rule as shade_day, but positioned at this date's real calendar span instead of a
    folded 0-24h axis."""
    if tag == 'DD':
        ax.axvspan(day_start, day_start + pd.Timedelta(days=1),
                   color=NIGHT_COLOR, alpha=NIGHT_ALPHA, linewidth=0, zorder=0)
        return
    for lo, hi in night_spans(night_start, night_end):
        if lo < hi:
            ax.axvspan(day_start + pd.Timedelta(hours=lo), day_start + pd.Timedelta(hours=hi),
                       color=NIGHT_COLOR, alpha=NIGHT_ALPHA, linewidth=0, zorder=0)


def schedule_path(folder):
    return os.path.join(folder, SCHEDULE_FILE)


def load_schedule(folder):
    """The experiment's saved schedule dict, or {} when there is none. A file that exists but
    can't be read raises (the caller reports it)."""
    path = schedule_path(folder)
    if not os.path.isfile(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_schedule(folder, dates, included, tags, night_start, night_end):
    """Write the Activity tab's schedule; `included` and `tags` map date -> bool / 'LD'|'DD'.
    Keys written by other tabs (e.g. BUZZSWARM_DAYS_KEY) are kept. Returns the path."""
    return update_schedule(folder, **{
        'schema_version': 1,
        'night_start_hour': int(night_start),
        'night_end_hour': int(night_end),
        'included_dates': [d for d in dates if included.get(d, True)],
        'dates': {d: tags.get(d, 'LD') for d in dates},
    })


def update_schedule(folder, **fields):
    """Set the given top-level keys of the schedule, keeping every other key. An unreadable file
    is replaced. Returns the path."""
    try:
        data = load_schedule(folder)
    except Exception:
        data = {}
    data.setdefault('schema_version', 1)
    data.update(fields)
    path = schedule_path(folder)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path
