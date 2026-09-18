"""BuzzSwarm (aggregation / Sholl) tab.

Wires the refactored buzzswarm engine into the shared app. Unlike the old single-experiment
design, this tab now has its own multi-experiment picker (add several already-tracked
experiments, compare across them) — ported from ``activity_plot_manager.py``'s picker, the same
pattern used across all three "3 · Plotting" sub-tabs. Analysis runs on a background thread
(never the Tk main thread) via the importable ``run(...)``/``run_multi(...)`` callables in
``buzzswarm/fru2_sholl_csv.py`` / ``fru2_pooled_analysis.py`` / ``fru2_per_trajectory_plots.py``
(all built on the real r50/Sholl engine ``fru2_zt_normalized_analysis.py`` — not the excluded
``batch_centroid_analysis.py``). Per-experiment CSV export stays single-experiment (whichever
entry is selected in the list); the pooled/per-trajectory plots overlay every added experiment
and write to a package-level ``plots/buzzswarm_comparison/`` folder (a comparison plot doesn't
belong inside any single experiment's own folder).

Heavy imports (the engine + matplotlib) are deferred to the run handlers so opening the tab
stays cheap and the app cold-starts fast.
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from tooltip import add_tooltip
from path_utils import get_package_root
from scrollable_frame import make_scrollable_column
from image_preview import ImagePreview


class BuzzswarmTabManager:
    # Engine tunables exposed in the UI: (engine attribute, label, default-as-string, help).
    # A '.' in the default marks a float field; otherwise it is parsed as int.
    PARAMS = [
        ('MIN_FLIGHT_FRAMES',    'Min flight frames',           '50',
         "Minimum consecutive frames a flight track must span to be counted. Higher = stricter "
         "(fewer, more reliable flights). Typical: 30–60."),
        ('SPEED_FILTER_MIN',     'Speed filter min (px/frame)', '1.0',
         "Discard track points slower than this (pixels moved per frame) as resting/jitter. "
         "Typical: 1.0."),
        ('SPEED_FILTER_MAX',     'Speed filter max (px/frame)', '40.0',
         "Discard track points faster than this (px/frame) as tracking errors/teleports. "
         "Typical: 40."),
        ('BIN_MINUTES',          'Window minutes (bin)',        '10',
         "Width of each time bin (minutes) for the windowed r50/activity aggregation. "
         "Smaller = finer time resolution. Typical: 10."),
        ('SHOLL_RADIUS_STEP_PX', 'Sholl radius step (px)',      '5',
         "Radial ring spacing (pixels) for the Sholl-style shell counts around the cage centre. "
         "Smaller = finer spatial resolution. Typical: 5."),
    ]
    # config.json namespace for persisted BuzzSwarm parameters.
    CONFIG_KEY = 'buzzswarm_params'

    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.tab = None
        self.param_vars = {}
        self.status_var = None
        self.preview = None
        self._running = False
        # entries: list of dicts with keys: path, folder, alias, group
        self.entries = []
        self._picker_paths = {}

    # ---------------------------------------------------------------- UI build
    def init_buzzswarm_tab(self, tab: ttk.Frame):
        self.tab = tab
        controls_outer, controls = make_scrollable_column(tab)
        controls_outer.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(controls, text="BuzzSwarm (Aggregation)",
                 font=("TkDefaultFont", 12, "bold")).pack(side=tk.TOP, anchor="w")
        tk.Label(controls, text="Sholl-style r50 aggregation from tracked flights.",
                 fg="gray", wraplength=260, justify="left").pack(side=tk.TOP, anchor="w", pady=(0, 8))

        # ---- Add an experiment (ported from activity_plot_manager.py) ----
        add_frame = tk.LabelFrame(controls, text="Add an experiment")
        add_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        pick_row = tk.Frame(add_frame); pick_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.picker_var = tk.StringVar()
        self.picker_combo = ttk.Combobox(pick_row, textvariable=self.picker_var, width=22, state="readonly")
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
        group_row = tk.Frame(list_frame); group_row.pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Label(group_row, text="Group:").pack(side=tk.LEFT)
        self.group_var = tk.StringVar()
        group_entry = tk.Entry(group_row, textvariable=self.group_var, width=12)
        group_entry.pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(group_row, text="Set", command=self._on_set_group_clicked).pack(side=tk.LEFT)
        add_tooltip(group_entry, "Colour/legend label for this experiment (e.g. 'Control' / "
                    "'Treatment'). Also the label used for the single-experiment CSV export.")
        tk.Button(list_frame, text="Remove selected", command=self._on_remove_clicked).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))

        param_frame = tk.LabelFrame(controls, text="Parameters")
        param_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        saved = self._load_saved_params()
        for row, (attr, label, default, help_text) in enumerate(self.PARAMS):
            lbl = tk.Label(param_frame, text=label)
            lbl.grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=saved.get(attr, default))
            entry = tk.Entry(param_frame, textvariable=var, width=8)
            entry.grid(row=row, column=1, padx=4, pady=2)
            self.param_vars[attr] = var
            add_tooltip(lbl, help_text)
            add_tooltip(entry, help_text)

        csv_btn = tk.Button(controls, text="Run Sholl per-trajectory CSV",
                            command=self.run_sholl_csv)
        csv_btn.pack(side=tk.TOP, fill=tk.X, pady=3)
        add_tooltip(csv_btn, "Single-experiment: exports the selected experiment's per-trajectory "
                    "CSV (no plot).")
        pooled_btn = tk.Button(controls, text="Run pooled r50 plots", command=self.run_pooled)
        pooled_btn.pack(side=tk.TOP, fill=tk.X, pady=3)
        add_tooltip(pooled_btn, "Compares every added experiment: one overlaid line per "
                    "experiment/Group.")
        traj_btn = tk.Button(controls, text="Run per-trajectory plots",
                             command=self.run_per_trajectory)
        traj_btn.pack(side=tk.TOP, fill=tk.X, pady=3)
        add_tooltip(traj_btn, "Compares every added experiment: one overlaid series per "
                    "experiment/Group.")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(controls, textvariable=self.status_var, fg="blue",
                 wraplength=260, justify="left").pack(side=tk.TOP, anchor="w", pady=(8, 0))

        preview = tk.Frame(tab)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preview = ImagePreview(preview, placeholder="Run an analysis to preview its plot here.",
                                    log=lambda msg: self.log(f"BuzzSwarm: {msg}"))

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _count_tracking_files(exp):
        """Number of forward_mosq_tracks_* segment files in the experiment's final_tracking_data.
        This is the canonical 'has this experiment been tracked yet?' signal — BuzzSwarm's engine
        returns an empty session (and produces empty plots/CSV) when there are none."""
        tdir = os.path.join(exp, 'final_tracking_data')
        if not os.path.isdir(tdir):
            return 0
        return sum(1 for f in os.listdir(tdir)
                   if f.startswith('forward_mosq_tracks_') and not f.startswith('.'))

    def refresh_experiment(self):
        # No-op: this tab is no longer tied to "the currently loaded experiment" (ui_manager.py
        # still calls this on experiment load; kept harmless rather than editing that call site).
        pass

    # ================================================================ recents picker (ported
    # from activity_plot_manager.py, near-verbatim — see that file's "Confirmed decisions" note
    # on why this is duplicated per-tab rather than factored into a shared module)
    def _refresh_recents(self):
        em = getattr(self.ui_manager, 'experiment_manager', None)
        recent = em.get_recent_experiments() if em else []
        self._picker_paths = {}
        values = []
        for r in recent:
            path = r.get('path')
            if not path:
                continue
            alias = r.get('alias') or os.path.basename(os.path.dirname(path))
            label = alias
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
            self.log("BuzzSwarm: pick an experiment (recents dropdown or Browse…) first.")
            return
        folder = os.path.dirname(path)
        alias = os.path.basename(os.path.normpath(folder))
        if any(e['folder'] == folder for e in self.entries):
            self.log(f"BuzzSwarm: '{alias}' is already added.")
            return
        self.entries.append({'path': path, 'folder': folder, 'alias': alias, 'group': alias})
        self._refresh_listbox()
        self.exp_listbox.selection_clear(0, tk.END)
        self.exp_listbox.selection_set(len(self.entries) - 1)
        self._set_status(f"{len(self.entries)} experiment(s) added.")

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
        self._refresh_listbox()

    def _refresh_listbox(self):
        self.exp_listbox.delete(0, tk.END)
        for e in self.entries:
            self.exp_listbox.insert(tk.END, f"{e['alias']}  [{e['group']}]")

    def _selected_entry_idx(self):
        sel = self.exp_listbox.curselection()
        return sel[0] if sel else None

    def _entries_with_tracking_data(self):
        """Validate self.entries and return the (folder, group) pairs that have tracking data,
        for a run_multi() call — or None if none do (an error dialog was already shown)."""
        if not self.entries:
            messagebox.showerror("BuzzSwarm", "Add at least one experiment first.")
            return None
        pairs = [(e['folder'], e['group']) for e in self.entries]
        ready = [(f, g) for f, g in pairs if self._count_tracking_files(f) > 0]
        missing = [g for f, g in pairs if self._count_tracking_files(f) == 0]
        if missing:
            self.log(f"BuzzSwarm: skipping {len(missing)} experiment(s) with no tracking data: "
                     f"{', '.join(missing)}")
        if not ready:
            messagebox.showerror(
                "BuzzSwarm — no tracking data",
                "None of the added experiments have tracked flights yet "
                "(final_tracking_data is empty).\n\n"
                "BuzzSwarm aggregates tracked flights, so there is nothing to analyse. "
                "Run Activity tracking on these experiments first, then try again.")
            return None
        return ready

    def _comparison_output_dir(self):
        """Package-level folder for multi-experiment comparison plots — not inside any single
        experiment's own folder, since a comparison plot doesn't belong to any one of them."""
        out_dir = os.path.join(get_package_root(), "plots", "buzzswarm_comparison")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _snapshot_params(self):
        """Read the parameter Entry values on the main thread (Tk vars must not be read from a
        worker thread) and cast each to int/float per its default's type. Also persists the
        (validated) values back to config.json so they survive across sessions."""
        out = {}
        for attr, _label, default, _help in self.PARAMS:
            raw = self.param_vars[attr].get().strip()
            caster = float if '.' in default else int
            try:
                out[attr] = caster(raw)
            except (ValueError, TypeError):
                out[attr] = caster(default)
                self.log(f"BuzzSwarm: invalid value for {attr!r} ({raw!r}); using default {default}.")
        self._save_params(out)
        return out

    def _load_saved_params(self):
        """Return the persisted BuzzSwarm params from config.json as {attr: str}, or {}."""
        try:
            saved = self.experiment_manager.config.get(self.CONFIG_KEY, {})
            if isinstance(saved, dict):
                return {k: str(v) for k, v in saved.items()}
        except Exception:
            pass
        return {}

    def _save_params(self, params):
        """Persist the current (validated) params back to config.json. Best-effort."""
        try:
            self.experiment_manager.config[self.CONFIG_KEY] = dict(params)
            self.experiment_manager.save_config(self.experiment_manager.config)
        except Exception as exc:
            self.log(f"BuzzSwarm: could not save parameters: {exc}")

    @staticmethod
    def _apply_params(engine, params):
        for attr, value in params.items():
            setattr(engine, attr, value)

    def _set_status(self, text):
        self.status_var.set(text)

    def _dispatch(self, work, describe):
        """Run `work()` (no args — the handler already snapshotted whatever it needs) on a
        worker thread, never the Tk main thread. `work` returns a path to a PNG to preview (or
        an output dir)."""
        if self._running:
            self.log("BuzzSwarm: a run is already in progress.")
            return
        self._running = True
        self._set_status(f"Running {describe}...")
        self.ui_manager.set_status(action=f"BuzzSwarm: {describe}…", busy=True)
        self._open_progress("BuzzSwarm", f"Running {describe}…")
        self.log(f"BuzzSwarm: running {describe}...")

        def worker():
            try:
                result = work()
                self.root.after(0, lambda: self._on_done(describe, result))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_error(describe, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, describe, result):
        self._running = False
        self._close_progress()
        self._set_status(f"Done: {describe}.")
        self.ui_manager.set_status(action=f"BuzzSwarm: {describe} done", busy=False)
        self.log(f"BuzzSwarm: {describe} finished -> {result}")
        if isinstance(result, str) and result.lower().endswith(".png") and os.path.isfile(result):
            self.preview.show_path(result)

    def _on_error(self, describe, exc):
        self._running = False
        self._close_progress()
        self._set_status(f"Error in {describe}.")
        self.ui_manager.set_status(action=f"BuzzSwarm: {describe} failed", busy=False)
        self.log(f"BuzzSwarm: {describe} failed: {exc}")
        messagebox.showerror("BuzzSwarm", f"{describe} failed:\n{exc}")

    def _open_progress(self, title, message):
        try:
            from progress_dialog import ProgressDialog
            self._progress = ProgressDialog(self.root, title=title, message=message)
            self._progress.pulse()   # indeterminate: the engine reports no incremental %
        except Exception:
            self._progress = None

    def _close_progress(self):
        prog = getattr(self, '_progress', None)
        if prog is not None:
            prog.close()
            self._progress = None

    # ---------------------------------------------------------------- handlers
    def run_sholl_csv(self):
        """Single-experiment: exports whichever entry is selected in the list."""
        idx = self._selected_entry_idx()
        if idx is None:
            messagebox.showerror("BuzzSwarm", "Select an experiment in the list first.")
            return
        entry = self.entries[idx]
        exp = entry['folder']
        if self._count_tracking_files(exp) == 0:
            messagebox.showerror(
                "BuzzSwarm — no tracking data",
                f"'{entry['alias']}' has no tracked flights yet (final_tracking_data is empty).\n\n"
                "Run Activity tracking on this experiment first, then try again.")
            return
        params = self._snapshot_params()

        def work():
            from buzzswarm import fru2_zt_normalized_analysis as engine
            from buzzswarm import fru2_sholl_csv
            self._apply_params(engine, params)
            return fru2_sholl_csv.run(exp)   # returns the CSV path (no preview)
        self._dispatch(work, "Sholl per-trajectory CSV")

    def run_pooled(self):
        """Compares every added experiment: one overlaid r50/prop-within/activity line per
        experiment/Group, written to the shared plots/buzzswarm_comparison/ folder."""
        experiments = self._entries_with_tracking_data()
        if experiments is None:
            return
        params = self._snapshot_params()
        out_dir = self._comparison_output_dir()

        def work():
            from buzzswarm import fru2_zt_normalized_analysis as engine
            from buzzswarm import fru2_pooled_analysis
            self._apply_params(engine, params)
            result_dir = fru2_pooled_analysis.run_multi(experiments, output_dir=out_dir)
            return os.path.join(result_dir, "pooled_r50.png")
        self._dispatch(work, "pooled r50 plots")

    def run_per_trajectory(self):
        """Compares every added experiment: one overlaid series per experiment/Group in the
        all-day scatter/line/violin plots, written to the shared plots/buzzswarm_comparison/
        folder."""
        experiments = self._entries_with_tracking_data()
        if experiments is None:
            return
        params = self._snapshot_params()
        out_dir = self._comparison_output_dir()

        def work():
            from buzzswarm import fru2_zt_normalized_analysis as engine
            from buzzswarm import fru2_per_trajectory_plots
            self._apply_params(engine, params)
            result_dir = fru2_per_trajectory_plots.run_multi(experiments, output_dir=out_dir)
            return os.path.join(result_dir, "all_day", "scatter_r50_by_sex_all_day_ZT.png")
        self._dispatch(work, "per-trajectory plots")
