"""BuzzSwarm (aggregation) tab: swarm size and where mosquitoes fly, by real clock time, per day.

The user adds one or more tracked experiments (each with a colour Group), ticks which calendar
days to analyse (by default the full days not tagged DD), edits the clock windows, and picks how
the data is shown. "Run analysis" calls ``buzzswarm/clock_analysis.run`` (one cached session per
calendar day; real-flight swarm metrics) on a worker thread under a cancellable progress dialog,
then ``buzzswarm/clock_plots.render_all``. "Redraw plots" re-reads the CSVs the last run wrote, so
changing the view, the metrics, the groups or unticking days never recomputes. Every figure goes
to one folder and can be browsed on the right.

Outputs:
  - per experiment: CSVs + day caches in {exp}/plots/buzzswarm/ (day_cache/ holds ~150 MB a day);
  - figures: {exp}/plots/buzzswarm/figures/ for one experiment, or a package-level
    plots/buzzswarm_comparison/{GroupA}__{GroupB}/ when several are compared.
Each experiment's day selection and night hours are saved on Run in its ld_dd_schedule.json
(ld_dd.BUZZSWARM_DAYS_KEY; the night hours are shared with the Activity tab, the day selection
is not). Windows / bin / view / metrics persist in config.json under 'buzzswarm_view'.

The older fru2 analyses (Sholl CSV, pooled r50, per-trajectory, which squash a whole recording
onto ZT 0-12) are no longer in this tab; `buzzsuite_cli.py swarm --which sholl|pooled|per-traj`
still runs them. The engine and matplotlib are imported lazily (first Add / first time the tab
is shown), on the main thread, so the app cold-starts fast.
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import ld_dd
from tooltip import add_tooltip
from path_utils import get_package_root
from scrollable_frame import make_scrollable_column
from image_preview import ImagePreview


_PLOT_LABELS = {
    'swarm_timecourse': 'Swarm size around lights on / off',
    'swarm_size_distribution': 'How often each swarm size occurs',
    'swarm_vs_flying': 'Swarm size at the same number flying',
    'heatmaps': 'Heatmaps of flight position',
    'centre_profile': 'Flight density vs distance from the centre',
    'window_summary': 'All metrics per clock window',
}
_MODES = (('overlay', 'Overlay'), ('individual', 'Individual'), ('continuous', 'Continuous'))


def _engine():
    """clock_analysis + clock_plots, imported on first use (main thread)."""
    from buzzswarm import clock_analysis as ca
    from buzzswarm import clock_plots as cp
    return ca, cp


def _plot_label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in _PLOT_LABELS:
        return _PLOT_LABELS[stem]
    if stem.startswith('timecourse_') and '_' in stem[len('timecourse_'):]:
        metric, mode = stem[len('timecourse_'):].rsplit('_', 1)
        _ca, cp = _engine()
        return f"Through the day: {cp.LABELS.get(metric, metric)} ({mode})"
    return stem


def _open_path(path):
    if sys.platform.startswith('win'):
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])


def _select_tables(tables, units, groups):
    """Keep only the (experiment, date) units asked for and relabel each experiment's group.
    Returns (tables, missing_units)."""
    out = {}
    have = set()
    for key, df in tables.items():
        if key == 'geometry' or df is None or df.empty or 'experiment' not in df:
            out[key] = df
            continue
        pairs = list(zip(df['experiment'], df['date'].astype(str)))
        have.update(pairs)
        keep = [p in units for p in pairs]
        sub = df[keep].copy()
        sub['group'] = sub['experiment'].map(groups)
        out[key] = sub
    return out, sorted(set(units) - have)


class BuzzswarmTabManager:
    CONFIG_KEY = 'buzzswarm_view'     # config.json: windows, bin_minutes, mode, metrics

    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.tab = None
        self.status_var = None
        self.preview = None
        self._running = False
        self._progress = None
        # entries: dicts with path, folder, alias (Location_Experiment, also the output folder
        # name), group, coverage {date: fraction}, included {date: bool}, night_start, night_end,
        # schedule (the ld_dd_schedule.json read on Add)
        self.entries = []
        self._picker_paths = {}
        self._current_entry_idx = None
        self._date_widgets = []       # [{date, var}] for the shown experiment
        self._metric_vars = {}        # metric -> BooleanVar (filled when the tab is first shown)
        self._out_paths = []
        self._out_dir = None
        # Test hooks: None = the experiments' own plots/buzzswarm/ and the package plots/ folder.
        self._out_root = None
        self._cache_root = None
        self._comparison_root = None

    # ================================================================ UI build
    def init_buzzswarm_tab(self, tab: ttk.Frame):
        self.tab = tab
        saved = self._load_view()
        controls_outer, controls = make_scrollable_column(tab)
        controls_outer.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(controls, text="BuzzSwarm (Aggregation)",
                 font=("TkDefaultFont", 12, "bold")).pack(side=tk.TOP, anchor="w")
        tk.Label(controls, text="Swarm size and where mosquitoes fly, by clock time, per day.",
                 fg="gray", wraplength=300, justify="left").pack(side=tk.TOP, anchor="w", pady=(0, 8))

        # ---- Add an experiment ----
        add_frame = tk.LabelFrame(controls, text="Add an experiment")
        add_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        pick_row = tk.Frame(add_frame); pick_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.picker_var = tk.StringVar()
        self.picker_combo = ttk.Combobox(pick_row, textvariable=self.picker_var, width=26, state="readonly")
        self.picker_combo.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(self.picker_combo, "Pick any of your last 10 opened experiments.")
        refresh_btn = tk.Button(pick_row, text="⟳", width=2, command=self._refresh_recents)
        refresh_btn.pack(side=tk.LEFT)
        add_tooltip(refresh_btn, "Reload the recents list.")
        browse_btn = tk.Button(add_frame, text="Browse…", command=self._browse_and_stage)
        browse_btn.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        add_tooltip(browse_btn, "Pick an experiment_*.json that isn't in the recents list above.")
        tk.Button(add_frame, text="Add experiment", command=self._on_add_clicked).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))
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
        group_entry = tk.Entry(group_row, textvariable=self.group_var, width=16)
        group_entry.pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(group_row, text="Set", command=self._on_set_group_clicked).pack(side=tk.LEFT)
        add_tooltip(group_entry, "Colour and legend label (e.g. 'Male' / 'Female'). Experiments "
                    "with the same Group are averaged together.")
        tk.Button(list_frame, text="Remove selected", command=self._on_remove_clicked).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))

        # ---- Days (selected experiment) ----
        days_frame = tk.LabelFrame(controls, text="Days (selected experiment)")
        days_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        night_row = tk.Frame(days_frame); night_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        tk.Label(night_row, text="Lights off").pack(side=tk.LEFT)
        self.night_start_var = tk.StringVar(value="17")
        ns = tk.Entry(night_row, textvariable=self.night_start_var, width=4); ns.pack(side=tk.LEFT, padx=(2, 6))
        tk.Label(night_row, text="on").pack(side=tk.LEFT)
        self.night_end_var = tk.StringVar(value="5")
        ne = tk.Entry(night_row, textvariable=self.night_end_var, width=4); ne.pack(side=tk.LEFT, padx=(2, 0))
        add_tooltip(ns, "Hour the lights turn off (24 h clock), e.g. 17. Shared with the Activity tab.")
        add_tooltip(ne, "Hour the lights turn on (24 h clock), e.g. 5. Shared with the Activity tab.")
        bulk_row = tk.Frame(days_frame); bulk_row.pack(fill=tk.X, padx=4, pady=(0, 2))
        full_btn = tk.Button(bulk_row, text="Full days", command=self._tick_full_days)
        full_btn.pack(side=tk.LEFT, padx=(0, 3))
        add_tooltip(full_btn, "Tick every day recorded for at least 95% of its 24 h (except days "
                    "tagged DD in the Activity tab), untick the rest.")
        tk.Button(bulk_row, text="Untick all", command=lambda: self._set_all_days(False)).pack(side=tk.LEFT)
        self._build_date_table(days_frame)

        # ---- Clock windows ----
        win_frame = tk.LabelFrame(controls, text="Clock windows")
        win_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        self.window_listbox = tk.Listbox(win_frame, height=4, exportselection=False)
        self.window_listbox.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))
        for w in saved.get('windows') or []:
            self.window_listbox.insert(tk.END, w)
        add_row = tk.Frame(win_frame); add_row.pack(fill=tk.X, padx=4, pady=(0, 2))
        self.window_entry_var = tk.StringVar(value="16:00-16:30")
        win_entry = tk.Entry(add_row, textvariable=self.window_entry_var, width=13)
        win_entry.pack(side=tk.LEFT, padx=(0, 4))
        win_entry.bind("<Return>", lambda e: self._on_add_window())
        add_tooltip(win_entry, "A clock window as HH:MM-HH:MM, e.g. 16:30-17:00. Each window is "
                    "scored on every ticked day.")
        tk.Button(add_row, text="Add", command=self._on_add_window).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(add_row, text="Remove", command=self._on_remove_window).pack(side=tk.LEFT, padx=(0, 3))
        defaults_btn = tk.Button(add_row, text="Defaults", command=self._on_default_windows)
        defaults_btn.pack(side=tk.LEFT)
        add_tooltip(defaults_btn, "Just after lights on, mid-light and just before lights off "
                    "(30 min each), from the selected experiment's light hours.")

        # ---- Plots ----
        plot_frame = tk.LabelFrame(controls, text="Plots")
        plot_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        mode_row = tk.Frame(plot_frame); mode_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        tk.Label(mode_row, text="Through the day:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value=saved.get('mode') if saved.get('mode') in dict(_MODES) else 'overlay')
        for value, text in _MODES:
            tk.Radiobutton(mode_row, text=text, variable=self.mode_var, value=value).pack(side=tk.LEFT)
        add_tooltip(mode_row, "Overlay = every day on one 00:00-24:00 axis (group mean +- SEM).\n"
                    "Individual = one panel per day.\nContinuous = real calendar time, one row "
                    "per experiment.")
        bin_row = tk.Frame(plot_frame); bin_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(bin_row, text="Bin (min):").pack(side=tk.LEFT)
        self.bin_var = tk.StringVar(value=str(saved.get('bin_minutes', 10)))
        bin_entry = tk.Entry(bin_row, textvariable=self.bin_var, width=5)
        bin_entry.pack(side=tk.LEFT, padx=(4, 0))
        add_tooltip(bin_entry, "Width of each time bin for the through-the-day plots. Changing "
                    "it needs Run analysis.")
        tk.Label(plot_frame, text="Metrics through the day:").pack(side=tk.TOP, anchor="w", padx=4)
        self._metrics_frame = tk.Frame(plot_frame)
        self._metrics_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))
        self._saved_metrics = saved.get('metrics')
        controls.bind("<Map>", lambda e: self._ensure_engine_widgets(), add="+")

        run_btn = tk.Button(controls, text="Run analysis", command=self._on_run_clicked)
        run_btn.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))
        add_tooltip(run_btn, "Analyse the ticked days of every experiment (days already analysed "
                    "load from the cache in seconds), then draw all plots.")
        redraw_btn = tk.Button(controls, text="Redraw plots", command=self._on_redraw_clicked)
        redraw_btn.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))
        add_tooltip(redraw_btn, "Redraw from the last results with the current view, metrics, "
                    "groups and ticked days. No analysis is rerun.")

        self.status_var = tk.StringVar(value="Add experiments, tick days, then Run analysis.")
        tk.Label(controls, textvariable=self.status_var, fg="blue",
                 wraplength=300, justify="left").pack(side=tk.TOP, anchor="w", pady=(8, 0))

        # ---- Output browser ----
        right = tk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        nav = tk.Frame(right); nav.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        tk.Label(nav, text="Plot:").pack(side=tk.LEFT)
        self.plot_combo = ttk.Combobox(nav, state="readonly", width=60)
        self.plot_combo.pack(side=tk.LEFT, padx=(4, 4))
        self.plot_combo.bind("<<ComboboxSelected>>", lambda e: self._show_plot(self.plot_combo.current()))
        tk.Button(nav, text="◀", width=3, command=lambda: self._step_plot(-1)).pack(side=tk.LEFT)
        tk.Button(nav, text="▶", width=3, command=lambda: self._step_plot(1)).pack(side=tk.LEFT, padx=(2, 8))
        open_btn = tk.Button(nav, text="Open folder", command=self._on_open_folder)
        open_btn.pack(side=tk.LEFT)
        add_tooltip(open_btn, "Open the folder with every plot (PNG) and the numbers behind it (CSV).")
        holder = tk.Frame(right); holder.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.preview = ImagePreview(holder, placeholder="Run the analysis to see the plots here.",
                                    log=lambda msg: self.log(f"BuzzSwarm: {msg}"))

    def _build_date_table(self, parent):
        """Scrollable per-date rows (ported from activity_plot_manager.py's date table)."""
        holder = tk.Frame(parent, height=130)
        holder.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))
        holder.pack_propagate(False)
        self._date_canvas = tk.Canvas(holder, highlightthickness=0)
        scroll = tk.Scrollbar(holder, orient="vertical", command=self._date_canvas.yview)
        self._date_inner = tk.Frame(self._date_canvas)
        self._date_inner.bind("<Configure>",
                              lambda e: self._date_canvas.configure(scrollregion=self._date_canvas.bbox("all")))
        self._date_canvas.create_window((0, 0), window=self._date_inner, anchor="nw")
        self._date_canvas.configure(yscrollcommand=scroll.set)
        self._date_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def wheel(event):
            if sys.platform == "darwin":
                self._date_canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                self._date_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_wheel(_e=None):
            self._date_canvas.bind_all("<MouseWheel>", wheel)
            self._date_canvas.bind_all("<Button-4>", lambda e: self._date_canvas.yview_scroll(-1, "units"))
            self._date_canvas.bind_all("<Button-5>", lambda e: self._date_canvas.yview_scroll(1, "units"))

        def unbind_wheel(_e=None):
            self._date_canvas.unbind_all("<MouseWheel>")
            self._date_canvas.unbind_all("<Button-4>")
            self._date_canvas.unbind_all("<Button-5>")

        self._date_canvas.bind("<Enter>", bind_wheel)
        self._date_canvas.bind("<Leave>", unbind_wheel)

    def _ensure_engine_widgets(self):
        """Widgets that need the engine (metric list, default windows), built the first time the
        tab is shown rather than at app start."""
        if self._metric_vars:
            return
        ca, cp = _engine()
        if not self.window_listbox.size():
            self._on_default_windows()
        chosen = self._saved_metrics if isinstance(self._saved_metrics, list) else cp.DEFAULT_METRICS
        for metric, label in ca.METRICS:
            var = tk.BooleanVar(value=metric in chosen)
            tk.Checkbutton(self._metrics_frame, text=label, variable=var, anchor="w", justify="left",
                           wraplength=280).pack(side=tk.TOP, fill=tk.X)
            self._metric_vars[metric] = var

    def refresh_experiment(self):
        # No-op: this tab isn't tied to "the currently loaded experiment" (ui_manager.py calls it
        # on experiment load).
        pass

    # ================================================================ recents picker
    def _refresh_recents(self):
        em = getattr(self.ui_manager, 'experiment_manager', None)
        recent = em.get_recent_experiments() if em else []
        self._picker_paths = {}
        values = []
        for r in recent:
            path = r.get('path')
            if not path:
                continue
            label = r.get('alias') or os.path.basename(os.path.dirname(path))
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
        label = f"{os.path.basename(os.path.dirname(path))}   —   browsed"
        values = list(self.picker_combo['values'])
        if label not in values:
            values.insert(0, label)
            self.picker_combo['values'] = values
        self._picker_paths[label] = path
        self.picker_var.set(label)

    # ================================================================ experiment list
    @staticmethod
    def _count_tracking_files(exp):
        tdir = os.path.join(exp, 'final_tracking_data')
        if not os.path.isdir(tdir):
            return 0
        return sum(1 for f in os.listdir(tdir)
                   if f.startswith('forward_mosq_tracks_') and not f.startswith('.'))

    def _on_add_clicked(self):
        path = self._picker_paths.get(self.picker_var.get())
        if not path:
            self.log("BuzzSwarm: pick an experiment (recents dropdown or Browse…) first.")
            return
        self.add_experiment(path)

    def add_experiment(self, path):
        """Add the experiment whose experiment_*.json is `path`. Returns the entry or None."""
        folder = os.path.dirname(path)
        if any(e['folder'] == folder for e in self.entries):
            self._set_status(f"'{os.path.basename(folder)}' is already added.")
            return None
        if self._count_tracking_files(folder) == 0:
            self.ui_manager.show_error(
                "BuzzSwarm — no tracking data",
                f"'{os.path.basename(folder)}' has no tracked flights yet (final_tracking_data is "
                "empty).\n\nRun tracking on this experiment first, then add it again.")
            return None
        ca, _cp = _engine()
        coverage = ca.date_coverage(folder)
        try:
            schedule = ld_dd.load_schedule(folder)
        except Exception as exc:
            self.log(f"BuzzSwarm: could not read {ld_dd.schedule_path(folder)}: {exc}")
            schedule = {}
        days = set(ca.default_days(coverage, schedule))
        alias = ca.default_alias(folder)
        entry = {'path': path, 'folder': folder, 'alias': alias, 'group': alias,
                 'coverage': coverage, 'included': {d: d in days for d in coverage},
                 'night_start': int(schedule.get('night_start_hour', ld_dd.DEFAULT_NIGHT[0])),
                 'night_end': int(schedule.get('night_end_hour', ld_dd.DEFAULT_NIGHT[1])),
                 'schedule': schedule}
        self._sync_date_widgets_to_entry()
        self.entries.append(entry)
        self._refresh_listbox()
        self.exp_listbox.selection_clear(0, tk.END)
        self.exp_listbox.selection_set(len(self.entries) - 1)
        self._on_experiment_select()
        self._set_status(f"{len(self.entries)} experiment(s) added.")
        return entry

    def _on_set_group_clicked(self):
        idx = self._selected_entry_idx()
        if idx is None:
            self.log("BuzzSwarm: select an experiment in the list first.")
            return
        group = self.group_var.get().strip()
        if not group:
            self.log("BuzzSwarm: type a group name first.")
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
        self._clear_date_table()
        self._refresh_listbox()

    def _refresh_listbox(self):
        self._sync_date_widgets_to_entry()
        self.exp_listbox.delete(0, tk.END)
        for e in self.entries:
            n = sum(1 for v in e['included'].values() if v)
            self.exp_listbox.insert(tk.END, f"{e['alias']}  [{e['group']}]  ({n} days)")

    def _selected_entry_idx(self):
        sel = self.exp_listbox.curselection()
        return sel[0] if sel else None

    # ================================================================ days
    def _on_experiment_select(self, event=None):
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

    def _refresh_date_table(self, idx):
        ca, _cp = _engine()
        self._clear_date_table()
        e = self.entries[idx]
        for d, frac in e['coverage'].items():
            row = tk.Frame(self._date_inner); row.pack(side=tk.TOP, fill=tk.X, anchor="w")
            var = tk.BooleanVar(value=bool(e['included'].get(d)))
            tk.Checkbutton(row, variable=var).pack(side=tk.LEFT)
            full = frac >= ca.FULL_DAY_COVERAGE
            tk.Label(row, text=d, width=11, anchor="w", fg="black" if full else "gray").pack(side=tk.LEFT)
            tk.Label(row, text=f"{frac * 100:.0f}% recorded", fg="gray").pack(side=tk.LEFT, padx=(4, 0))
            self._date_widgets.append({'date': d, 'var': var})

    def _sync_date_widgets_to_entry(self):
        idx = self._current_entry_idx
        if idx is None or idx >= len(self.entries):
            return
        e = self.entries[idx]
        for w in self._date_widgets:
            e['included'][w['date']] = bool(w['var'].get())
        e['night_start'] = self._int_or(self.night_start_var, e['night_start'], 0, 24)
        e['night_end'] = self._int_or(self.night_end_var, e['night_end'], 0, 24)

    def _set_all_days(self, value):
        for w in self._date_widgets:
            w['var'].set(value)

    def _tick_full_days(self):
        idx = self._current_entry_idx
        if idx is None:
            return
        ca, _cp = _engine()
        e = self.entries[idx]
        schedule = dict(e['schedule'])
        schedule.pop(ld_dd.BUZZSWARM_DAYS_KEY, None)
        days = set(ca.default_days(e['coverage'], schedule))
        for w in self._date_widgets:
            w['var'].set(w['date'] in days)

    # ================================================================ windows
    def _windows(self):
        return list(self.window_listbox.get(0, tk.END))

    def _on_add_window(self):
        ca, _cp = _engine()
        text = self.window_entry_var.get().strip()
        try:
            label, _lo, _hi = ca.parse_window(text)
        except (ValueError, TypeError):
            self.ui_manager.show_error("BuzzSwarm", f"'{text}' is not a clock window.\n\nType it "
                                       "as HH:MM-HH:MM within one day, e.g. 16:30-17:00.")
            return
        if label in self._windows():
            return
        windows = sorted(self._windows() + [label], key=lambda w: ca.parse_window(w)[1])
        self.window_listbox.delete(0, tk.END)
        for w in windows:
            self.window_listbox.insert(tk.END, w)

    def _on_remove_window(self):
        for i in reversed(self.window_listbox.curselection()):
            self.window_listbox.delete(i)

    def _on_default_windows(self):
        ca, _cp = _engine()
        start = self._int_or(self.night_start_var, ld_dd.DEFAULT_NIGHT[0], 0, 24)
        end = self._int_or(self.night_end_var, ld_dd.DEFAULT_NIGHT[1], 0, 24)
        self.window_listbox.delete(0, tk.END)
        for w in ca.default_windows(start, end):
            self.window_listbox.insert(tk.END, w)

    # ================================================================ settings
    def _load_view(self):
        try:
            saved = self.experiment_manager.config.get(self.CONFIG_KEY, {})
            return saved if isinstance(saved, dict) else {}
        except Exception:
            return {}

    def _save_view(self, view):
        try:
            self.experiment_manager.config[self.CONFIG_KEY] = dict(view)
            self.experiment_manager.save_config(self.experiment_manager.config)
        except Exception as exc:
            self.log(f"BuzzSwarm: could not save settings: {exc}")

    @staticmethod
    def _int_or(var, default, minimum=None, maximum=None):
        try:
            val = int(str(var.get()).strip())
        except (ValueError, TypeError):
            return default
        if minimum is not None:
            val = max(minimum, val)
        if maximum is not None:
            val = min(maximum, val)
        return val

    def _set_status(self, text):
        self.status_var.set(text)

    # ================================================================ run / redraw
    def _snapshot(self):
        """Everything a run needs, read from Tk on the main thread. None if not runnable (the
        reason was already shown)."""
        self._sync_date_widgets_to_entry()
        self._ensure_engine_widgets()
        ca, _cp = _engine()
        if not self.entries:
            self.ui_manager.show_error("BuzzSwarm", "Add at least one experiment first.")
            return None
        units = [e for e in self.entries if any(e['included'].values())]
        skipped = [e['alias'] for e in self.entries if e not in units]
        if not units:
            self.ui_manager.show_error("BuzzSwarm", "Tick at least one day to analyse "
                                       "(Days, for the selected experiment).")
            return None
        windows = self._windows()
        if not windows:
            self.ui_manager.show_error("BuzzSwarm", "Add at least one clock window (or click Defaults).")
            return None
        for w in windows:
            try:
                ca.parse_window(w)
            except (ValueError, TypeError):
                self.ui_manager.show_error("BuzzSwarm", f"'{w}' is not a clock window. Remove it "
                                           "and add it again as HH:MM-HH:MM.")
                return None
        bin_minutes = self._int_or(self.bin_var, 10, 1, 240)
        self.bin_var.set(str(bin_minutes))
        view = {'windows': windows, 'bin_minutes': bin_minutes, 'mode': self.mode_var.get(),
                'metrics': [m for m, v in self._metric_vars.items() if v.get()]}
        self._save_view(view)
        if skipped:
            self.log(f"BuzzSwarm: no days ticked for {', '.join(skipped)}; leaving them out.")
        nights = {(e['night_start'], e['night_end']) for e in units}
        if len(nights) > 1:
            self.log("BuzzSwarm: the experiments have different light hours; plots shade the "
                     "first experiment's night.")
        experiments = [{'folder': e['folder'], 'alias': e['alias'], 'group': e['group'],
                        'days': [d for d in e['coverage'] if e['included'].get(d)]} for e in units]
        return {'experiments': experiments, 'entries': units, 'view': view,
                'night': (units[0]['night_start'], units[0]['night_end']),
                'fig_dir': self._figures_dir(units)}

    def _experiment_out_dir(self, entry):
        ca, _cp = _engine()
        if self._out_root:
            return os.path.join(self._out_root, entry['alias'])
        return ca.experiment_output_dir(entry['folder'])

    def _figures_dir(self, entries):
        _ca, cp = _engine()
        if len(entries) == 1:
            return os.path.join(self._experiment_out_dir(entries[0]), 'figures')
        root = self._comparison_root or os.path.join(get_package_root(), 'plots')
        return cp.comparison_dir(root, list(dict.fromkeys(e['group'] for e in entries)))

    def _save_days(self, entries):
        """Persist each experiment's ticked days + light hours (BuzzSwarm's own key)."""
        for e in entries:
            try:
                ld_dd.update_schedule(e['folder'], **{
                    ld_dd.BUZZSWARM_DAYS_KEY: [d for d in e['coverage'] if e['included'].get(d)],
                    'night_start_hour': int(e['night_start']), 'night_end_hour': int(e['night_end'])})
            except Exception as exc:
                self.log(f"BuzzSwarm: could not save the day selection for {e['alias']}: {exc}")

    def _on_run_clicked(self):
        snap = self._snapshot()
        if snap is None:
            return
        self._save_days(snap['entries'])
        ca, cp = _engine()
        view = snap['view']
        n_days = sum(len(x['days']) for x in snap['experiments'])

        def work(report, cancel):
            tables = ca.run(snap['experiments'], view['bin_minutes'], view['windows'],
                            cache_root=self._cache_root, out_root=self._out_root,
                            progress=lambda f, msg: report(0.9 * f, msg), cancel=cancel)
            paths = cp.render_all(tables, snap['fig_dir'], view['mode'], view['metrics'], snap['night'],
                                  progress=lambda f, msg: report(0.9 + 0.1 * f, msg), cancel=cancel)
            return {'paths': paths, 'dir': snap['fig_dir'], 'missing': []}
        self._dispatch(work, f"analysis of {n_days} day(s)")

    def _on_redraw_clicked(self):
        snap = self._snapshot()
        if snap is None:
            return
        ca, cp = _engine()
        view = snap['view']
        out_dirs = [self._experiment_out_dir(e) for e in snap['entries']]
        units = {(x['alias'], d) for x in snap['experiments'] for d in x['days']}
        groups = {x['alias']: x['group'] for x in snap['experiments']}

        def work(report, cancel):
            report(0.05, "reading saved results")
            tables = ca.load_outputs(out_dirs, view['bin_minutes'])
            tables, missing = _select_tables(tables, units, groups)
            if all(df.empty for k, df in tables.items() if k != 'geometry'):
                raise RuntimeError("No saved results for these days yet. Click Run analysis first.")
            paths = cp.render_all(tables, snap['fig_dir'], view['mode'], view['metrics'], snap['night'],
                                  progress=lambda f, msg: report(0.1 + 0.9 * f, msg), cancel=cancel)
            return {'paths': paths, 'dir': snap['fig_dir'], 'missing': missing,
                    'no_bins': tables['bins'].empty}
        self._dispatch(work, "redraw")

    def _dispatch(self, work, describe):
        """Run work(report, cancel_event) on a worker thread under a cancellable progress dialog.
        The worker never touches Tk: report(fraction, message) and the outcome go through a queue
        that the main thread drains every 100 ms (which also passes Cancel on to the worker)."""
        if self._running:
            self.log("BuzzSwarm: a run is already in progress.")
            return
        ca, _cp = _engine()
        self._running = True
        cancel = threading.Event()
        events = queue.Queue()
        self._set_status(f"Running {describe}…")
        self.ui_manager.set_status(action=f"BuzzSwarm: {describe}…", busy=True)
        self.log(f"BuzzSwarm: running {describe}…")
        try:
            from progress_dialog import ProgressDialog
            self._progress = ProgressDialog(self.root, title="BuzzSwarm", message=f"Running {describe}…",
                                            cancelable=True)
        except Exception:
            self._progress = None

        def report(frac, msg):
            events.put(('progress', (frac, msg)))

        def worker():
            try:
                result = work(report, cancel)
            except ca.Cancelled:
                events.put(('cancelled', None))
            except Exception as exc:
                events.put(('error', exc))
            else:
                events.put(('done', result))

        threading.Thread(target=worker, daemon=True).start()
        self._poll(describe, events, cancel)

    def _poll(self, describe, events, cancel):
        if self._progress is not None and self._progress.cancelled:
            cancel.set()
        progress = None
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == 'progress':
                progress = payload          # only the latest one is worth drawing
                continue
            if kind == 'done':
                self._on_done(describe, payload)
            elif kind == 'cancelled':
                self._on_cancelled(describe)
            else:
                self._on_error(describe, payload)
            return
        if progress is not None and self._progress is not None:
            self._progress.update(*progress)
        self.root.after(100, lambda: self._poll(describe, events, cancel))

    def _finish(self, status, action):
        self._running = False
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._set_status(status)
        self.ui_manager.set_status(action=f"BuzzSwarm: {action}", busy=False)

    def _on_done(self, describe, result):
        paths = result['paths']
        note = ""
        if result.get('missing'):
            days = ", ".join(f"{a} {d}" for a, d in result['missing'])
            note = f" Not analysed yet (Run analysis to add): {days}."
        if result.get('no_bins'):
            note += " No through-the-day results for this bin width; Run analysis to make them."
        self._finish(f"Done: {len(paths)} plots in {result['dir']}.{note}", f"{describe} done")
        self.log(f"BuzzSwarm: {describe} finished -> {result['dir']}")
        self._set_outputs(paths, result['dir'])

    def _on_cancelled(self, describe):
        self._finish("Cancelled.", f"{describe} cancelled")
        self.log(f"BuzzSwarm: {describe} cancelled.")

    def _on_error(self, describe, exc):
        self._finish(f"Error in {describe}.", f"{describe} failed")
        self.ui_manager.show_error("BuzzSwarm", f"{describe} failed:\n{exc}")

    # ================================================================ output browser
    def _set_outputs(self, paths, folder):
        self._out_paths = list(paths)
        self._out_dir = folder
        self.plot_combo['values'] = [_plot_label(p) for p in self._out_paths]
        if self._out_paths:
            self._show_plot(0)

    def _show_plot(self, idx):
        if not self._out_paths or idx is None:
            return
        idx = idx % len(self._out_paths)        # wraps both ways (step back from the first plot)
        self.plot_combo.current(idx)
        self.preview.show_path(self._out_paths[idx])

    def _step_plot(self, delta):
        if self._out_paths:
            self._show_plot(max(0, self.plot_combo.current()) + delta)

    def _on_open_folder(self):
        if not self._out_dir or not os.path.isdir(self._out_dir):
            self._set_status("Run the analysis first; then this opens the folder with its plots.")
            return
        try:
            _open_path(self._out_dir)
        except Exception as exc:
            self.ui_manager.show_error("BuzzSwarm", f"Could not open {self._out_dir}:\n{exc}")
