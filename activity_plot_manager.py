"""Activity plotting tab: LD/DD-shaded flight-activity plots across experiments and dates.

Replaces the old ~2000-line single-video Activity Plotting tab (removed 2026-07-15) and folds in
the former standalone "Compare experiments" tab. The user picks one or more already-analyzed
experiments (each with its own colour Group), chooses which calendar dates within each to include
and tags every date LD (light:dark) or DD (constant dark), then plots a chosen activity variable —
default **total_pixels_moved** (movement magnitude per minute, computed during Concatenate) — in
one of three output modes:

  - Overlay:     every included experiment-date on one 00:00-24:00 (time-of-day) axis. LD dates
                 use the experiment's normal Group colour; DD dates are always drawn bright red,
                 so DD is identifiable by colour alone regardless of group (no dashing).
  - Individual:  one panel per included date. LD panels shade only the configured night window
                 (default 17:00-05:00); DD panels shade the *entire* panel (constant darkness has
                 no "day" portion to leave unshaded).
  - Continuous:  one panel, real calendar time on the x-axis, every included date's data
                 concatenated in chronological order (a gap in the line where dates were
                 excluded). Night shading repeats at each LD date's real night window; DD dates
                 are shaded across their entire 24h span, same as Individual, just laid out on a
                 real timeline instead of folded onto a single 0-24h axis.

Night/DD shading never changes position (DD's "subjective night" reuses the identical LD clock
window, standard circadian practice) — only how much of the axis gets shaded, or which colour a
line is drawn in.

Reads each experiment's already-produced analyzed_data.pkl -> population_data (the same file/key
the data-prep step writes). Picker/recents/preview code is ported from the retired
compare_tab_manager.py; LD/DD tags persist per experiment in ld_dd_schedule.json (sibling to
custom_zones.json, same inline-JSON pattern as BuzzPhono). matplotlib/pandas are already a sunk
cold-start cost via common_imports, so importing them at module scope here adds nothing new.
"""
import os
import sys
import json
import pickle
import math
import tkinter as tk
from tkinter import ttk, filedialog

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from tooltip import add_tooltip
from path_utils import get_package_root, newest_mtime
from scrollable_frame import make_scrollable_column
from image_preview import ImagePreview

_GROUP_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

_NIGHT_COLOR = '#C8C8C8'
_NIGHT_ALPHA = 0.30
_DD_COLOR = '#FF0000'  # Overlay mode: DD lines are always bright red, regardless of group colour.

# Variables that are displacement TOTALS (resample by sum); everything else is an instantaneous
# count/level (resample by mean).
_SUM_VARIABLES = {'total_pixels_moved'}


def _ylabel_for(variable):
    if variable == 'total_pixels_moved':
        return 'total_pixels_moved (px per bin)'
    return variable


def _apply_pixel_axis_format(ax, variable):
    """total_pixels_moved's per-bin magnitude depends on the chosen resample width -- anywhere
    from hundreds to hundreds of thousands of px. Auto-scale tick labels (e.g. '26k px', '1.2M px')
    instead of a fixed 'thousands'/'millions' label in the axis title that would be wrong at a
    different bin size."""
    if variable == 'total_pixels_moved':
        ax.yaxis.set_major_formatter(mticker.EngFormatter(unit='px'))


def _format_recent(entry):
    """Mirrors ExperimentDashboard._format_recent's label so recents look the same everywhere."""
    alias = entry.get('alias') or os.path.basename(os.path.dirname(entry.get('path', '')))
    ts = entry.get('ts')
    if ts:
        try:
            import time as _time
            return f"{alias}   —   last opened {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(ts))}"
        except Exception:
            pass
    return alias


def _night_spans(night_start, night_end):
    """(lo, hi) hour spans of the night window on a 0-24h axis. Wraps midnight when
    night_start > night_end (the usual 17..05 case)."""
    if night_start > night_end:
        return [(night_start, 24.0), (0.0, night_end)]
    return [(night_start, night_end)]


def _shade_night(ax, night_start, night_end):
    """Shade the night window (fixed gray band) on a time-of-day (0-24h) axis."""
    for lo, hi in _night_spans(night_start, night_end):
        if lo < hi:
            ax.axvspan(lo, hi, color=_NIGHT_COLOR, alpha=_NIGHT_ALPHA, linewidth=0, zorder=0)


def _shade_panel(ax, entry, tag):
    """Individual-mode shading for one date's panel: DD has no "day" portion to leave unshaded,
    so the whole 0-24h panel is shaded; LD shades only the configured night window."""
    if tag == 'DD':
        ax.axvspan(0, 24, color=_NIGHT_COLOR, alpha=_NIGHT_ALPHA, linewidth=0, zorder=0)
    else:
        _shade_night(ax, entry['night_start'], entry['night_end'])


def _shade_real_span(ax, day_start, night_start, night_end, tag):
    """Continuous-mode shading: same rule as _shade_panel, but positioned at this date's real
    calendar span instead of a folded 0-24h axis."""
    if tag == 'DD':
        ax.axvspan(day_start, day_start + pd.Timedelta(days=1),
                   color=_NIGHT_COLOR, alpha=_NIGHT_ALPHA, linewidth=0, zorder=0)
        return
    for lo, hi in _night_spans(night_start, night_end):
        if lo < hi:
            ax.axvspan(day_start + pd.Timedelta(hours=lo), day_start + pd.Timedelta(hours=hi),
                       color=_NIGHT_COLOR, alpha=_NIGHT_ALPHA, linewidth=0, zorder=0)


class ActivityPlotManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.tab = None
        # entries: list of dicts with keys: path, folder, alias, group, population_data,
        #   dates (sorted 'YYYY-MM-DD' list), included ({date: bool}), tags ({date: 'LD'|'DD'}),
        #   night_start (int), night_end (int), pkl_path, pkl_mtime (float, for staleness/reload
        #   detection -- see _reload_stale_entries), tracking_mtime (float, informational)
        self.entries = []
        self._picker_paths = {}
        self._group_colors = {}
        self._current_entry_idx = None
        self._date_widgets = []   # per-row {date, include_var, tag_var} for the shown experiment
        self.preview = None
        self._last_fig = None

    # ================================================================ UI build
    def init_activity_tab(self, tab: ttk.Frame):
        self.tab = tab
        controls_outer, controls = make_scrollable_column(tab)
        controls_outer.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(controls, text="Activity (in-cage)", font=("TkDefaultFont", 12, "bold")
                 ).pack(side=tk.TOP, anchor="w")
        tk.Label(controls, text="Flight-activity plots with light/dark shading.",
                 fg="gray", wraplength=300, justify="left").pack(side=tk.TOP, anchor="w", pady=(0, 8))

        # ---- Add an experiment (ported from compare_tab_manager) ----
        add_frame = tk.LabelFrame(controls, text="Add an experiment")
        add_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        pick_row = tk.Frame(add_frame); pick_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.picker_var = tk.StringVar()
        self.picker_combo = ttk.Combobox(pick_row, textvariable=self.picker_var, width=28, state="readonly")
        self.picker_combo.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(self.picker_combo, "Pick any of your last 10 opened experiments.")
        refresh_btn = tk.Button(pick_row, text="⟳", width=2, command=self._refresh_recents)
        refresh_btn.pack(side=tk.LEFT)
        add_tooltip(refresh_btn, "Reload the recents list.")
        browse_btn = tk.Button(add_frame, text="Browse…", command=self._browse_and_stage)
        browse_btn.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        add_tooltip(browse_btn, "Pick an experiment_*.json that isn't in the recents list above.")
        add_btn = tk.Button(add_frame, text="Add experiment", command=self._on_add_clicked)
        add_btn.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))
        self._refresh_recents()

        # ---- Experiments list ----
        list_frame = tk.LabelFrame(controls, text="Experiments")
        list_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        self.exp_listbox = tk.Listbox(list_frame, height=5, exportselection=False)
        self.exp_listbox.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))
        self.exp_listbox.bind("<<ListboxSelect>>", self._on_experiment_select)
        group_row = tk.Frame(list_frame); group_row.pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Label(group_row, text="Group:").pack(side=tk.LEFT)
        self.group_var = tk.StringVar()
        group_entry = tk.Entry(group_row, textvariable=self.group_var, width=12)
        group_entry.pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(group_row, text="Set", command=self._on_set_group_clicked).pack(side=tk.LEFT)
        add_tooltip(group_entry, "Colour code: experiment-dates sharing a Group are drawn in the "
                    "same colour (e.g. 'Control' / 'Treatment').")
        list_btn_row = tk.Frame(list_frame); list_btn_row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(list_btn_row, text="Remove selected", command=self._on_remove_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        reload_btn = tk.Button(list_btn_row, text="Reload data", command=self._on_reload_clicked)
        reload_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        add_tooltip(reload_btn, "Re-read analyzed_data.pkl for the selected experiment from disk "
                    "(use after re-running tracking/concatenation on it).")

        # ---- Dates & light schedule (per selected experiment) ----
        dates_frame = tk.LabelFrame(controls, text="Dates & light schedule (selected experiment)")
        dates_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        night_row = tk.Frame(dates_frame); night_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        tk.Label(night_row, text="Night start").pack(side=tk.LEFT)
        self.night_start_var = tk.StringVar(value="17")
        ns = tk.Entry(night_row, textvariable=self.night_start_var, width=4); ns.pack(side=tk.LEFT, padx=(2, 6))
        tk.Label(night_row, text="end").pack(side=tk.LEFT)
        self.night_end_var = tk.StringVar(value="5")
        ne = tk.Entry(night_row, textvariable=self.night_end_var, width=4); ne.pack(side=tk.LEFT, padx=(2, 0))
        add_tooltip(ns, "Hour lights turn OFF (24h clock). Default 17 = 17:00. Shaded as night.")
        add_tooltip(ne, "Hour lights turn ON (24h clock). Default 5 = 05:00.")

        bulk_row = tk.Frame(dates_frame); bulk_row.pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Button(bulk_row, text="All LD", command=lambda: self._set_all_tags("LD")).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(bulk_row, text="All DD", command=lambda: self._set_all_tags("DD")).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(bulk_row, text="Save LD/DD tags", command=self._on_save_ld_dd_tags_clicked).pack(side=tk.LEFT)

        # scrollable date table
        table_holder = tk.Frame(dates_frame, height=140)
        table_holder.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))
        table_holder.pack_propagate(False)
        self._date_canvas = tk.Canvas(table_holder, highlightthickness=0)
        dscroll = tk.Scrollbar(table_holder, orient="vertical", command=self._date_canvas.yview)
        self._date_inner = tk.Frame(self._date_canvas)
        self._date_inner.bind("<Configure>",
                              lambda e: self._date_canvas.configure(scrollregion=self._date_canvas.bbox("all")))
        self._date_canvas.create_window((0, 0), window=self._date_inner, anchor="nw")
        self._date_canvas.configure(yscrollcommand=dscroll.set)
        self._date_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel / trackpad scrolling while hovered, same pattern as scrollable_frame.py's
        # outer controls column -- bind_all only while the pointer is over this canvas (not the
        # whole app) so it doesn't fight the outer column's own wheel scrolling when the pointer
        # is elsewhere.
        def _date_wheel(event):
            if sys.platform == "darwin":
                self._date_canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                self._date_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_date_wheel(_e=None):
            self._date_canvas.bind_all("<MouseWheel>", _date_wheel)
            self._date_canvas.bind_all("<Button-4>", lambda e: self._date_canvas.yview_scroll(-1, "units"))
            self._date_canvas.bind_all("<Button-5>", lambda e: self._date_canvas.yview_scroll(1, "units"))

        def _unbind_date_wheel(_e=None):
            self._date_canvas.unbind_all("<MouseWheel>")
            self._date_canvas.unbind_all("<Button-4>")
            self._date_canvas.unbind_all("<Button-5>")

        self._date_canvas.bind("<Enter>", _bind_date_wheel)
        self._date_canvas.bind("<Leave>", _unbind_date_wheel)

        # ---- Plot controls ----
        plot_frame = tk.LabelFrame(controls, text="Plot")
        plot_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        var_row = tk.Frame(plot_frame); var_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(var_row, text="Variable:").pack(side=tk.LEFT)
        self.variable_var = tk.StringVar()
        self.variable_combo = ttk.Combobox(var_row, textvariable=self.variable_var, width=20, state="readonly")
        self.variable_combo.pack(side=tk.LEFT, padx=(4, 0))
        add_tooltip(self.variable_combo,
                    "total_pixels_moved sums flight displacement per bin, so it drops to true 0 "
                    "when nothing moves — best for seeing whether activity actually stopped.\n"
                    "Count variables like numb_mosquitos_flying are averaged per bin, so they stay "
                    "a smooth trend line but rarely hit exact 0 even when most seconds are 0.")
        bin_row = tk.Frame(plot_frame); bin_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(bin_row, text="Resample (min):").pack(side=tk.LEFT)
        self.resample_var = tk.StringVar(value="1")
        rs = tk.Entry(bin_row, textvariable=self.resample_var, width=6); rs.pack(side=tk.LEFT, padx=(4, 0))
        add_tooltip(rs, "Bin width in minutes before plotting. total_pixels_moved is summed per bin; "
                    "count variables are averaged.")
        mode_row = tk.Frame(plot_frame); mode_row.pack(fill=tk.X, padx=4, pady=(2, 2))
        tk.Label(mode_row, text="Output:").pack(side=tk.LEFT)
        self.output_mode_var = tk.StringVar(value="individual")
        tk.Radiobutton(mode_row, text="Overlay", variable=self.output_mode_var, value="overlay").pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="Individual", variable=self.output_mode_var, value="individual").pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="Continuous", variable=self.output_mode_var, value="continuous").pack(side=tk.LEFT)
        add_tooltip(mode_row,
                    "Overlay = every included experiment-date on one 00:00–24:00 axis; DD dates "
                    "always drawn bright red, regardless of the experiment's colour.\n"
                    "Individual = one panel per included date; LD shades only the night window, "
                    "DD shades the whole panel.\n"
                    "Continuous = one panel, real calendar time; night/DD shading repeats at "
                    "each included date's real position.")
        tk.Button(plot_frame, text="Plot", command=self._on_plot_clicked).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))
        tk.Button(plot_frame, text="Save PNG", command=self._on_save_png_clicked).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))

        self.status_var = tk.StringVar(value="Add an experiment, tick dates + LD/DD, then Plot.")
        tk.Label(controls, textvariable=self.status_var, fg="blue",
                 wraplength=300, justify="left").pack(side=tk.TOP, anchor="w", pady=(4, 0))

        preview = tk.Frame(tab)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preview = ImagePreview(preview, placeholder="Add experiments, tick dates, then Plot.",
                                    log=lambda msg: self.log(f"Activity: {msg}"))

    # ================================================================ recents picker (ported)
    def _refresh_recents(self):
        em = getattr(self.ui_manager, 'experiment_manager', None)
        recent = em.get_recent_experiments() if em else []
        self._picker_paths = {}
        values = []
        for r in recent:
            path = r.get('path')
            if not path:
                continue
            label = _format_recent(r)
            self._picker_paths[label] = path
            values.append(label)
        self.picker_combo['values'] = values
        if values:
            self.picker_var.set(values[0])

    def _browse_and_stage(self):
        path = filedialog.askopenfilename(
            title="Select experiment JSON",
            # Extension-only patterns: macOS's native open panel takes file extensions, not
            # globs, so a prefixed pattern like "experiment_*.json" makes Tk hand it a nil and
            # the whole process aborts (NSInvalidArgumentException). Don't narrow this again.
            filetypes=[("Experiment JSON", "*.json")])
        if not path:
            return
        alias = os.path.basename(os.path.dirname(path))
        label = f"{alias}   —   browsed"
        values = list(self.picker_combo['values'])
        if label not in values:
            values.insert(0, label)
            self.picker_combo['values'] = values
        self._picker_paths[label] = path
        self.picker_var.set(label)

    # ================================================================ experiment list
    def _on_add_clicked(self):
        path = self._picker_paths.get(self.picker_var.get())
        if not path:
            self.ui_manager.show_error("Activity plotting",
                                       "Pick an experiment (recents dropdown or Browse…) first.")
            return
        folder = os.path.dirname(path)
        alias = os.path.basename(os.path.normpath(folder))
        existing = next((e for e in self.entries if e['folder'] == folder), None)
        if existing is not None:
            # Re-adding an already-loaded experiment reloads its data instead of refusing outright
            # -- the common case is the user just re-ran tracking/concatenation and wants this
            # tab to pick up the refreshed analyzed_data.pkl without removing and re-adding it
            # (which would also lose its saved included/tags/group state).
            if self._reload_entry(existing):
                self._refresh_listbox()
                self._refresh_variable_options()
                idx = self.entries.index(existing)
                self.exp_listbox.selection_clear(0, tk.END)
                self.exp_listbox.selection_set(idx)
                self._on_experiment_select()
                self.status_var.set(f"'{alias}' was already added -- reloaded its data.")
            return
        pop = self._load_population_data(folder)
        if pop is None:
            return
        dates = sorted({ts.strftime("%Y-%m-%d") for ts in pd.DatetimeIndex(pop.index).normalize()})
        sched = self._load_ld_dd_schedule(folder)
        saved_tags = sched.get('dates', {}) if isinstance(sched, dict) else {}
        saved_incl = sched.get('included_dates', None)
        tags = {d: (saved_tags.get(d, 'LD') if saved_tags.get(d) in ('LD', 'DD') else 'LD') for d in dates}
        if isinstance(saved_incl, list):
            included = {d: (d in saved_incl) for d in dates}
        else:
            included = {d: True for d in dates}
        entry = {'path': path, 'folder': folder, 'alias': alias, 'group': alias,
                 'population_data': pop, 'dates': dates, 'included': included, 'tags': tags,
                 'night_start': int(sched.get('night_start_hour', 17)) if isinstance(sched, dict) else 17,
                 'night_end': int(sched.get('night_end_hour', 5)) if isinstance(sched, dict) else 5,
                 'pkl_path': self._pkl_path(folder), 'pkl_mtime': self._pkl_mtime(folder),
                 'tracking_mtime': self._tracking_mtime(folder)}
        self.entries.append(entry)
        self._refresh_listbox()
        self._refresh_variable_options()
        # select the newly-added experiment so its date table shows
        self.exp_listbox.selection_clear(0, tk.END)
        self.exp_listbox.selection_set(len(self.entries) - 1)
        self._on_experiment_select()
        self.status_var.set(f"{len(self.entries)} experiment(s) added.")

    @staticmethod
    def _pkl_path(folder):
        return os.path.join(folder, "analyzed_data.pkl")

    def _pkl_mtime(self, folder):
        try:
            return os.path.getmtime(self._pkl_path(folder))
        except OSError:
            return 0.0

    @staticmethod
    def _tracking_mtime(folder):
        return newest_mtime(os.path.join(folder, "final_tracking_data"), "forward_mosq_tracks_")

    def _reload_entry(self, entry):
        """Re-load population_data from disk for an already-added entry (its analyzed_data.pkl
        changed since it was added or last reloaded — e.g. tracking/concatenation re-ran). Dates
        that still exist keep their included/tag state; new dates default to included+LD; group
        and night_start/night_end are untouched. Returns False (leaving the old data in place) if
        the pkl can no longer be loaded — the caller decides whether/how to report that."""
        pop = self._load_population_data(entry['folder'])
        if pop is None:
            return False
        dates = sorted({ts.strftime("%Y-%m-%d") for ts in pd.DatetimeIndex(pop.index).normalize()})
        old_included, old_tags = entry['included'], entry['tags']
        entry['population_data'] = pop
        entry['dates'] = dates
        entry['included'] = {d: old_included.get(d, True) for d in dates}
        entry['tags'] = {d: old_tags.get(d, 'LD') for d in dates}
        entry['pkl_mtime'] = self._pkl_mtime(entry['folder'])
        entry['tracking_mtime'] = self._tracking_mtime(entry['folder'])
        return True

    def _reload_stale_entries(self):
        """Auto-reload any entry whose analyzed_data.pkl changed on disk since it was loaded (most
        commonly: the user re-ran 2 · Analysis -> Run tracking, which rebuilds the pkl
        automatically). Also warns — without blocking the plot — when final_tracking_data/ has
        segments newer than the loaded analyzed_data.pkl, meaning the pkl is stale relative to
        tracking and hasn't been rebuilt yet."""
        reloaded, stale = [], []
        for e in self.entries:
            current_pkl_mtime = self._pkl_mtime(e['folder'])
            if current_pkl_mtime and current_pkl_mtime != e.get('pkl_mtime'):
                if self._reload_entry(e):
                    reloaded.append(e['alias'])
            if self._tracking_mtime(e['folder']) > e.get('pkl_mtime', 0):
                stale.append(e['alias'])
        if reloaded:
            msg = "Reloaded updated data for: " + ", ".join(reloaded)
            self.log(f"Activity: {msg}")
            self.status_var.set(msg)
            self._refresh_listbox()
        if stale:
            msg = ("Newer tracking data exists but analyzed_data.pkl hasn't caught up for: "
                   + ", ".join(stale) + " (2 · Analysis -> Run tracking rebuilds it automatically).")
            self.log(f"Activity: {msg}")
            if not reloaded:
                self.status_var.set(msg)

    def _load_population_data(self, folder):
        """Load analyzed_data.pkl's population_data. Plain pickle.load — batch_processing's
        _patch_numpy_compat() ran at app import time so legacy numpy module paths already work."""
        pkl_path = self._pkl_path(folder)
        if not os.path.isfile(pkl_path):
            self.ui_manager.show_error(
                "Activity plotting",
                f"No analyzed_data.pkl in:\n{folder}\n\n"
                "Go to \"2 · Analysis\" and click \"Run tracking\" — analyzed_data.pkl is built "
                "automatically once tracking finishes (no separate concatenate step).")
            return None
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            self.ui_manager.show_error(
                "Activity plotting",
                f"The saved analysis file appears corrupt or incomplete:\n{pkl_path}\n\n{exc}")
            return None
        except Exception as exc:
            self.ui_manager.show_error("Activity plotting", f"Could not load {pkl_path}:\n{exc}")
            return None
        pop = data.get('population_data') if isinstance(data, dict) else None
        if not isinstance(pop, pd.DataFrame) or pop.empty:
            self.ui_manager.show_error("Activity plotting",
                                       f"analyzed_data.pkl has no population_data:\n{pkl_path}")
            return None
        pop = pop.copy()
        pop.index = pd.to_datetime(pop.index, errors='coerce')
        pop = pop[~pd.isna(pop.index)].sort_index()
        return pop

    def _on_set_group_clicked(self):
        idx = self._selected_entry_idx()
        if idx is None:
            self.log("Activity: select an experiment in the list first.")
            return
        group = self.group_var.get().strip()
        if not group:
            self.log("Activity: type a group name first.")
            return
        self.entries[idx]['group'] = group
        self._refresh_listbox()
        self.exp_listbox.selection_set(idx)

    def _on_remove_clicked(self):
        idx = self._selected_entry_idx()
        if idx is None:
            return
        del self.entries[idx]
        self._current_entry_idx = None
        self._refresh_listbox()
        self._refresh_variable_options()
        self._clear_date_table()

    def _on_reload_clicked(self):
        idx = self._selected_entry_idx()
        if idx is None:
            self.log("Activity: select an experiment in the list first.")
            return
        e = self.entries[idx]
        if self._reload_entry(e):
            self._refresh_listbox()
            self._refresh_variable_options()
            self.exp_listbox.selection_set(idx)
            self._refresh_date_table(idx)
            self.status_var.set(f"Reloaded '{e['alias']}'.")

    def _refresh_listbox(self):
        self.exp_listbox.delete(0, tk.END)
        for e in self.entries:
            n_incl = sum(1 for d in e['dates'] if e['included'].get(d, True))
            self.exp_listbox.insert(tk.END, f"{e['alias']}  [{e['group']}]  ({n_incl}/{len(e['dates'])} dates)")

    def _refresh_variable_options(self):
        cols = []
        for e in self.entries:
            for c in e['population_data'].select_dtypes(include='number').columns:
                if c not in cols:
                    cols.append(c)
        self.variable_combo['values'] = cols
        if cols and self.variable_var.get() not in cols:
            for preferred in ('total_pixels_moved', 'numb_mosquitos_flying'):
                if preferred in cols:
                    self.variable_var.set(preferred)
                    break
            else:
                self.variable_var.set(cols[0])

    def _selected_entry_idx(self):
        sel = self.exp_listbox.curselection()
        return sel[0] if sel else None

    # ================================================================ date table + LD/DD
    def _on_experiment_select(self, event=None):
        # persist the currently-shown experiment's widget state before switching away
        self._sync_date_widgets_to_entry()
        idx = self._selected_entry_idx()
        if idx is None:
            return
        self._current_entry_idx = idx
        e = self.entries[idx]
        self.group_var.set(e['group'])
        self.night_start_var.set(str(e['night_start']))
        self.night_end_var.set(str(e['night_end']))
        self._refresh_date_table(idx)

    def _clear_date_table(self):
        for w in list(self._date_inner.winfo_children()):
            w.destroy()
        self._date_widgets = []

    def _refresh_date_table(self, entry_idx):
        self._clear_date_table()
        e = self.entries[entry_idx]
        for d in e['dates']:
            row = tk.Frame(self._date_inner); row.pack(side=tk.TOP, fill=tk.X, anchor="w")
            inc_var = tk.BooleanVar(value=bool(e['included'].get(d, True)))
            tk.Checkbutton(row, variable=inc_var).pack(side=tk.LEFT)
            tk.Label(row, text=d, width=11, anchor="w").pack(side=tk.LEFT)
            tag_var = tk.StringVar(value=e['tags'].get(d, 'LD'))
            ttk.Combobox(row, textvariable=tag_var, values=["LD", "DD"], width=4,
                         state="readonly").pack(side=tk.LEFT, padx=(2, 0))
            self._date_widgets.append({'date': d, 'include_var': inc_var, 'tag_var': tag_var})

    def _sync_date_widgets_to_entry(self):
        if self._current_entry_idx is None or not self._date_widgets:
            return
        if self._current_entry_idx >= len(self.entries):
            return
        e = self.entries[self._current_entry_idx]
        for w in self._date_widgets:
            e['included'][w['date']] = bool(w['include_var'].get())
            tag = w['tag_var'].get()
            e['tags'][w['date']] = tag if tag in ('LD', 'DD') else 'LD'
        e['night_start'] = self._int_or(self.night_start_var, 17, minimum=0, maximum=24)
        e['night_end'] = self._int_or(self.night_end_var, 5, minimum=0, maximum=24)

    def _set_all_tags(self, tag):
        for w in self._date_widgets:
            w['tag_var'].set(tag)

    def _on_save_ld_dd_tags_clicked(self):
        self._sync_date_widgets_to_entry()
        idx = self._current_entry_idx
        if idx is None:
            self.log("Activity: select an experiment first.")
            return
        e = self.entries[idx]
        self._save_ld_dd_schedule(e)
        self._refresh_listbox()
        self.exp_listbox.selection_set(idx)
        self.status_var.set(f"Saved LD/DD schedule for {e['alias']}.")

    # ---- ld_dd_schedule.json (per experiment, sibling to custom_zones.json) ----
    @staticmethod
    def _ld_dd_schedule_path(folder):
        return os.path.join(folder, "ld_dd_schedule.json")

    def _load_ld_dd_schedule(self, folder):
        path = self._ld_dd_schedule_path(folder)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            self.log(f"Activity: could not read {path}: {exc}")
            return {}

    def _save_ld_dd_schedule(self, entry):
        payload = {
            'schema_version': 1,
            'night_start_hour': int(entry['night_start']),
            'night_end_hour': int(entry['night_end']),
            'included_dates': [d for d in entry['dates'] if entry['included'].get(d, True)],
            'dates': {d: entry['tags'].get(d, 'LD') for d in entry['dates']},
        }
        path = self._ld_dd_schedule_path(entry['folder'])
        try:
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)
            self.log(f"Activity: saved {path}")
        except Exception as exc:
            self.ui_manager.show_error("Activity plotting", f"Could not save LD/DD schedule:\n{exc}")

    def _int_or(self, var, default, minimum=None, maximum=None):
        try:
            val = int(str(var.get()).strip())
        except (ValueError, TypeError):
            val = default
        if minimum is not None:
            val = max(minimum, val)
        if maximum is not None:
            val = min(maximum, val)
        return val

    # ================================================================ plotting
    def _group_color(self, group):
        if group not in self._group_colors:
            self._group_colors[group] = _GROUP_PALETTE[len(self._group_colors) % len(_GROUP_PALETTE)]
        return self._group_colors[group]

    def _selected_units(self):
        """Flatten (entry, date_str, tag) for every included date across all entries, in list order."""
        units = []
        for e in self.entries:
            for d in e['dates']:
                if e['included'].get(d, True):
                    units.append((e, d, e['tags'].get(d, 'LD')))
        return units

    def _day_series(self, entry, date_str, variable, resample_min, as_time_of_day=True):
        """Resampled y for one experiment-date, or (None, None). x is either time-of-day
        (hours-since-midnight, for Overlay/Individual's folded 0-24h axis) or the real
        timestamps (for Continuous's real calendar-time axis)."""
        pop = entry['population_data']
        if variable not in pop.columns:
            return None, None
        day_start = pd.Timestamp(date_str)
        mask = (pop.index >= day_start) & (pop.index < day_start + pd.Timedelta(days=1))
        series = pop.loc[mask, variable].dropna()
        if series.empty:
            return None, None
        if resample_min:
            how = 'sum' if variable in _SUM_VARIABLES else 'mean'
            r = series.resample(f"{resample_min}min")
            series = r.sum() if how == 'sum' else r.mean()
            series = series.dropna()
            if series.empty:
                return None, None
        if as_time_of_day:
            x = series.index.hour + series.index.minute / 60.0 + series.index.second / 3600.0
            return x, series.values
        return series.index, series.values

    def _render_figure(self):
        """Returns None (a validation/data error was already shown to the user) or
        (fig, missing) where `missing` is a list of (alias, date_str) pairs that had no data for
        the chosen variable -- callers (_on_plot_clicked/_on_save_png_clicked) report those."""
        self._sync_date_widgets_to_entry()
        self._reload_stale_entries()
        variable = self.variable_var.get()
        if not self.entries:
            self.ui_manager.show_error("Activity plotting", "Add at least one experiment first.")
            return None
        if not variable:
            self.ui_manager.show_error("Activity plotting", "Pick a variable to plot first.")
            return None
        units = self._selected_units()
        if not units:
            self.ui_manager.show_error("Activity plotting",
                                       "Tick at least one date to include (Dates & light schedule).")
            return None
        resample_min = self._int_or(self.resample_var, 1, minimum=1)
        mode = self.output_mode_var.get()
        if mode == 'overlay':
            return self._render_overlay(units, variable, resample_min)
        if mode == 'continuous':
            return self._render_continuous(units, variable, resample_min)
        return self._render_individual(units, variable, resample_min)

    def _render_individual(self, units, variable, resample_min):
        n = len(units)
        ncols = 1 if n == 1 else 2
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 2.6 * nrows + 0.5),
                                 squeeze=False)
        flat = [ax for row in axes for ax in row]
        plotted = 0
        missing = []
        for i, (entry, date_str, tag) in enumerate(units):
            ax = flat[i]
            x, y = self._day_series(entry, date_str, variable, resample_min)
            _shade_panel(ax, entry, tag)
            if x is not None:
                ax.plot(x, y, color=self._group_color(entry['group']), linewidth=1.4)
                plotted += 1
            else:
                missing.append((entry['alias'], date_str))
                ax.text(0.5, 0.5, "no data", ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 1))
            ax.set_ylim(bottom=0)
            _apply_pixel_axis_format(ax, variable)
            ax.tick_params(axis='x', labelsize=6, rotation=90)
            ax.set_title(f"{entry['alias']}  {date_str}  [{tag}]", fontsize=9)
            ax.set_xlabel("Time of day (h)", fontsize=8)
            ax.set_ylabel(_ylabel_for(variable), fontsize=8)
        for j in range(n, len(flat)):
            flat[j].axis('off')
        if plotted == 0:
            plt.close(fig)
            self.ui_manager.show_error("Activity plotting",
                                       f"No data for '{variable}' on the selected dates.")
            return None
        fig.suptitle(f"{variable} — individual dates (LD: night shaded, DD: fully shaded)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig, missing

    def _render_overlay(self, units, variable, resample_min):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        # single fixed night band (all dates share the 0-24h axis); use the first unit's night hours
        first_entry = units[0][0]
        _shade_night(ax, first_entry['night_start'], first_entry['night_end'])
        plotted = 0
        missing = []
        seen_labels = set()
        for entry, date_str, tag in units:
            x, y = self._day_series(entry, date_str, variable, resample_min)
            if x is None:
                missing.append((entry['alias'], date_str))
                continue
            color = _DD_COLOR if tag == 'DD' else self._group_color(entry['group'])
            label = f"{entry['alias']} ({entry['group']}) [{tag}]"
            legend_label = label if label not in seen_labels else "_nolegend_"
            seen_labels.add(label)
            ax.plot(x, y, color=color, linewidth=1.3, label=legend_label, alpha=0.9)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            self.ui_manager.show_error("Activity plotting",
                                       f"No data for '{variable}' on the selected dates.")
            return None
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 1))
        ax.set_ylim(bottom=0)
        _apply_pixel_axis_format(ax, variable)
        ax.tick_params(axis='x', labelsize=7, rotation=90)
        ax.set_xlabel("Time of day (h)")
        ax.set_ylabel(_ylabel_for(variable))
        ax.set_title(f"{variable} — overlay (DD shown in bright red)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, missing

    def _render_continuous(self, units, variable, resample_min):
        """One panel, real calendar time on the x-axis. Every included entry gets one line per
        run of *consecutive* included dates (a gap in the calendar breaks the line rather than
        drawing a false connector across an excluded day). Night/DD shading is placed at each
        included date's real position, independent of which entry's line is being drawn."""
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for entry, date_str, tag in units:
            _shade_real_span(ax, pd.Timestamp(date_str), entry['night_start'], entry['night_end'], tag)

        plotted = 0
        missing = []
        for entry in self.entries:
            included_dates = sorted(d for d in entry['dates'] if entry['included'].get(d, True))
            if not included_dates:
                continue
            runs = [[included_dates[0]]]
            for prev, curr in zip(included_dates, included_dates[1:]):
                if pd.Timestamp(curr) - pd.Timestamp(prev) == pd.Timedelta(days=1):
                    runs[-1].append(curr)
                else:
                    runs.append([curr])

            color = self._group_color(entry['group'])
            label = f"{entry['alias']} ({entry['group']})"
            first_segment = True
            for run in runs:
                xs, ys = [], []
                for d in run:
                    x, y = self._day_series(entry, d, variable, resample_min, as_time_of_day=False)
                    if x is None:
                        missing.append((entry['alias'], d))
                        continue
                    xs.extend(x)
                    ys.extend(y)
                if not xs:
                    continue
                ax.plot(xs, ys, color=color, linewidth=1.3,
                        label=(label if first_segment else "_nolegend_"))
                first_segment = False
                plotted += 1

        if plotted == 0:
            plt.close(fig)
            self.ui_manager.show_error("Activity plotting",
                                       f"No data for '{variable}' on the selected dates.")
            return None
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_ylim(bottom=0)
        ax.margins(x=0)  # no default autoscale padding -- line starts flush at the left edge
        _apply_pixel_axis_format(ax, variable)
        ax.set_xlabel("Date / time")
        ax.set_ylabel(_ylabel_for(variable))
        ax.set_title(f"{variable} — continuous (night shaded; DD fully shaded)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, missing

    @staticmethod
    def _sanitize_filename_part(s):
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))

    def _output_dir_for_current_selection(self):
        """One loaded experiment -> that experiment's own plots/activity_plots/ (so saved plots
        live alongside the rest of its analysis output); several -> a package-local shared folder,
        since there's no single experiment folder that would make sense to write into. Falls back
        to the package folder if the preferred directory can't be created (e.g. the data drive
        unmounted)."""
        if len(self.entries) == 1:
            out_dir = os.path.join(self.entries[0]['folder'], "plots", "activity_plots")
        else:
            out_dir = os.path.join(get_package_root(), "plots", "activity")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            fallback = os.path.join(get_package_root(), "plots", "activity")
            self.log(f"Activity: could not create {out_dir} ({exc}); using {fallback} instead.")
            out_dir = fallback
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _report_plotted(self, missing, n_total):
        n_plotted = n_total - len(missing)
        msg = f"Plotted {n_plotted} date(s)."
        if missing:
            variable = self.variable_var.get()
            examples = "; ".join(f"{a} {d}" for a, d in missing[:5])
            if len(missing) > 5:
                examples += ", ..."
            msg += f" {len(missing)} had no '{variable}' data: {examples}"
            self.log(f"Activity: {msg}")
        self.status_var.set(msg)

    def _on_plot_clicked(self):
        fig = None
        try:
            result = self._render_figure()
            if result is None:
                return
            fig, missing = result
            if self._last_fig is not None:
                plt.close(self._last_fig)
            self._last_fig = fig
            # The preview is a transient scratch image (not a saved deliverable), so it always
            # lives in the package's own plots/ folder regardless of _output_dir_for_current_
            # selection() -- Save PNG (below) is what writes into the experiment's own folder.
            preview_dir = os.path.join(get_package_root(), "plots", "activity")
            os.makedirs(preview_dir, exist_ok=True)
            preview_path = os.path.join(preview_dir, "_preview.png")
            fig.savefig(preview_path, dpi=110)
            self.preview.show_path(preview_path)
            self._report_plotted(missing, len(self._selected_units()))
        except Exception as exc:
            # fig may already be stored as self._last_fig by the time savefig/preview fails --
            # closing it must not leave self._last_fig pointing at a closed figure (a later Save
            # PNG click would then try to save a closed matplotlib Figure and fail confusingly).
            if fig is not None:
                plt.close(fig)
                if self._last_fig is fig:
                    self._last_fig = None
            self.ui_manager.show_error("Activity plotting", f"Could not render plot:\n{exc}")

    def _on_save_png_clicked(self):
        fig = None
        try:
            missing = []
            if self._last_fig is None:
                result = self._render_figure()
                if result is None:
                    return
                fig, missing = result
                self._last_fig = fig
            out_dir = self._output_dir_for_current_selection()
            variable = self.variable_var.get() or "activity"
            mode = self.output_mode_var.get()
            if len(self.entries) == 1:
                name_part = self._sanitize_filename_part(self.entries[0]['alias'])
            else:
                name_part = "multi"
            import time as _time
            stamp = _time.strftime("%Y%m%d-%H%M")
            initial_name = f"activity_{name_part}_{self._sanitize_filename_part(variable)}_{mode}_{stamp}.png"
            path = filedialog.asksaveasfilename(
                title="Save activity plot", initialdir=out_dir, initialfile=initial_name,
                defaultextension=".png", filetypes=[("PNG", "*.png")])
            if not path:
                return
            self._last_fig.savefig(path, dpi=150)
            self.log(f"Activity: saved {path}")
            if missing:
                self._report_plotted(missing, len(self._selected_units()))
            self.status_var.set(f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            if fig is not None:
                plt.close(fig)
                if self._last_fig is fig:
                    self._last_fig = None
            self.ui_manager.show_error("Activity plotting", f"Could not save PNG:\n{exc}")

