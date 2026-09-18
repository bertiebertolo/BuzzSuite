"""BuzzPhono (phonotaxis) tab.

Wires the refactored custom-zones phonotaxis analysis into the shared app. Like BuzzSwarm, this
tab now has its own multi-experiment picker (add several already-tracked experiments, compare
across them) — ported from ``activity_plot_manager.py``'s picker. "Run BuzzPhono analysis" loops
over every added experiment, calling the unchanged, per-experiment
``buzzphono/plot_speaker_distance_phonotaxis.run_speaker_distance_analysis(...)`` for each (full
existing output, unaffected), then produces one new combined plot via
``run_multi_speaker_comparison(...)`` overlaying every experiment's speaker-response profile.
Per-experiment outputs stay in that experiment's own ``plots/custom_zones/``; the combined plot
goes to a package-level ``plots/buzzphono_comparison/`` folder (it doesn't belong to any one of
the experiments it compares).

The "Speaker zone" section (define/clone the speaker polygon) stays a per-experiment action, tied
to whichever entry is currently selected in the list — drawing a zone on N backgrounds at once
doesn't make sense, and this only ever consumed tracking output (never raw video), so it's fine
to sit alongside the picker without affecting the tracking pipeline at all.

BuzzPhono needs a ``custom_zones.json`` describing the speaker polygon
(``{"speaker": {"polygon": [[x, y], ...]}}``) per experiment. The interactive polygon editor
(``SpeakerZoneEditor``/``ZoneClonerEditor``) is below; the tab reports a clear error if a
selected experiment doesn't have one yet.

Heavy imports (the phonotaxis module + matplotlib) are deferred to the run handler.
"""
import os
import re
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

from PIL import Image, ImageTk

from tooltip import add_tooltip
from path_utils import get_package_root
from scrollable_frame import make_scrollable_column
from image_preview import ImagePreview

_TS_RE = re.compile(r'_(\d{8})_(\d{6})_')   # _{YYYYMMDD}_{HHMMSS}_ in segment filenames


class BuzzphonoTabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.tab = None
        self.start_var = None
        self.end_var = None
        self.on_start_var = None
        self.on_end_var = None
        self.off_start_var = None
        self.off_end_var = None
        self.resting_only_var = None
        self.status_var = None
        self.preview = None
        self._running = False
        # entries: list of dicts with keys: path, folder, alias, group, start, end
        self.entries = []
        self._picker_paths = {}
        self._current_entry_idx = None

    # ---------------------------------------------------------------- UI build
    def init_buzzphono_tab(self, tab: ttk.Frame):
        self.tab = tab
        controls_outer, controls = make_scrollable_column(tab)
        controls_outer.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(controls, text="BuzzPhono (Phonotaxis)",
                 font=("TkDefaultFont", 12, "bold")).pack(side=tk.TOP, anchor="w")
        tk.Label(controls, text="Speaker-zone resting fraction (custom zones).",
                 fg="gray", wraplength=280, justify="left").pack(side=tk.TOP, anchor="w", pady=(0, 8))

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
        self.exp_listbox.bind("<<ListboxSelect>>", self._on_experiment_select)
        group_row = tk.Frame(list_frame); group_row.pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Label(group_row, text="Group:").pack(side=tk.LEFT)
        self.group_var = tk.StringVar()
        group_entry = tk.Entry(group_row, textvariable=self.group_var, width=12)
        group_entry.pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(group_row, text="Set", command=self._on_set_group_clicked).pack(side=tk.LEFT)
        add_tooltip(group_entry, "Colour/legend label for this experiment in the combined plot "
                    "(e.g. 'Control' / 'Treatment').")
        tk.Button(list_frame, text="Remove selected", command=self._on_remove_clicked).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))

        zones_frame = tk.LabelFrame(controls, text="Speaker zone (selected experiment)")
        zones_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        tk.Button(zones_frame, text="Show background with borders",
                  command=self.show_background_with_zones).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=2)

        tk.Button(zones_frame, text="Define speaker zone…", command=self.define_speaker_zone).pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        comparison_btn = tk.Button(zones_frame, text="Add comparison zone (same shape)…",
                                   command=self.add_comparison_zone)
        comparison_btn.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        add_tooltip(comparison_btn, "Clone the speaker zone's exact shape/size and drag it to a "
                    "new spot (e.g. a sugar feeder) so the analysis can compare resting fraction "
                    "between regions of equal size.")

        win_frame = tk.LabelFrame(controls, text="Time window (ISO, selected experiment)")
        win_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        tk.Label(win_frame, text="Start").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.start_var = tk.StringVar(value="")
        tk.Entry(win_frame, textvariable=self.start_var, width=22).grid(row=0, column=1, padx=4, pady=2)
        tk.Label(win_frame, text="End").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.end_var = tk.StringVar(value="")
        tk.Entry(win_frame, textvariable=self.end_var, width=22).grid(row=1, column=1, padx=4, pady=2)
        add_tooltip(win_frame, "Auto-filled from the selected experiment's recording dates when "
                    "added; editable, and saved per experiment (switching the selected experiment "
                    "shows its own window).")

        stim_frame = tk.LabelFrame(controls, text="Stimulus windows (minute of hour)")
        stim_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        saved = self._load_saved_params()
        self.on_start_var = tk.StringVar(value=str(saved.get('on_start', 40)))
        self.on_end_var = tk.StringVar(value=str(saved.get('on_end', 50)))
        self.off_start_var = tk.StringVar(value=str(saved.get('off_start', 28)))
        self.off_end_var = tk.StringVar(value=str(saved.get('off_end', 38)))
        stim_help = {
            "ON  min": "Minutes within each hour when the speaker stimulus is ON (playback "
                       "window). Points in this window count as 'stimulus on'. Typical: 40–50.",
            "OFF min": "Minutes within each hour used as the silent baseline (stimulus OFF) for "
                       "comparison. Typical: 28–38.",
        }
        for row, (lbl, a, b) in enumerate([("ON  min", self.on_start_var, self.on_end_var),
                                           ("OFF min", self.off_start_var, self.off_end_var)]):
            tk.Label(stim_frame, text=lbl).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            e_a = tk.Entry(stim_frame, textvariable=a, width=5)
            e_a.grid(row=row, column=1, padx=2, pady=2)
            tk.Label(stim_frame, text="to").grid(row=row, column=2, padx=2)
            e_b = tk.Entry(stim_frame, textvariable=b, width=5)
            e_b.grid(row=row, column=3, padx=2, pady=2)
            add_tooltip(e_a, stim_help[lbl])
            add_tooltip(e_b, stim_help[lbl])

        self.resting_only_var = tk.BooleanVar(value=bool(saved.get('resting_only', True)))
        resting_chk = tk.Checkbutton(controls, text="Resting only (faster; skip heatmaps)",
                                     variable=self.resting_only_var)
        resting_chk.pack(side=tk.TOP, anchor="w")
        add_tooltip(resting_chk,
                    "Compute only the speaker-zone resting fraction (fast). Untick to also "
                    "generate the fuller set of heatmap/flight plots (slower).")

        run_btn = tk.Button(controls, text="Run BuzzPhono analysis", command=self.run_analysis)
        run_btn.pack(side=tk.TOP, fill=tk.X, pady=(6, 3))
        add_tooltip(run_btn, "Runs every added experiment's full analysis (unchanged, own "
                    "plots/custom_zones/ output each), then adds one combined plot overlaying "
                    "every experiment's speaker-response profile.")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(controls, textvariable=self.status_var, fg="blue",
                 wraplength=280, justify="left").pack(side=tk.TOP, anchor="w", pady=(8, 0))

        preview = tk.Frame(tab)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preview = ImagePreview(preview, placeholder="Add experiments, then Run BuzzPhono analysis.",
                                    log=lambda msg: self.log(f"BuzzPhono: {msg}"))

    # ---------------------------------------------------------------- picker/list
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
            self._picker_paths[alias] = path
            values.append(alias)
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

    def _on_add_clicked(self):
        path = self._picker_paths.get(self.picker_var.get())
        if not path:
            self.log("BuzzPhono: pick an experiment (recents dropdown or Browse…) first.")
            return
        folder = os.path.dirname(path)
        alias = os.path.basename(os.path.normpath(folder))
        if any(e['folder'] == folder for e in self.entries):
            self.log(f"BuzzPhono: '{alias}' is already added.")
            return
        span = self._derive_date_range(folder)
        entry = {'path': path, 'folder': folder, 'alias': alias, 'group': alias,
                 'start': span[0] if span else '', 'end': span[1] if span else ''}
        self.entries.append(entry)
        self._refresh_listbox()
        self.exp_listbox.selection_clear(0, tk.END)
        self.exp_listbox.selection_set(len(self.entries) - 1)
        self._on_experiment_select()
        self._set_status(f"{len(self.entries)} experiment(s) added.")

    def _on_set_group_clicked(self):
        idx = self._selected_entry_idx()
        if idx is None:
            self.log("BuzzPhono: select an experiment in the list first.")
            return
        group = self.group_var.get().strip()
        if not group:
            self.log("BuzzPhono: type a group name first.")
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

    def _refresh_listbox(self):
        self.exp_listbox.delete(0, tk.END)
        for e in self.entries:
            self.exp_listbox.insert(tk.END, f"{e['alias']}  [{e['group']}]")

    def _selected_entry_idx(self):
        sel = self.exp_listbox.curselection()
        return sel[0] if sel else None

    def _selected_experiment_dir(self):
        """The folder of whichever entry is selected in the list, or None (zone/preview handlers
        show their own error dialog when this is None)."""
        idx = self._selected_entry_idx()
        if idx is None:
            return None
        return self.entries[idx]['folder']

    def _on_experiment_select(self, event=None):
        # persist the currently-shown start/end into whichever entry was previously selected
        self._sync_start_end_to_entry()
        idx = self._selected_entry_idx()
        if idx is None:
            return
        self._current_entry_idx = idx
        e = self.entries[idx]
        self.group_var.set(e['group'])
        self.start_var.set(e['start'])
        self.end_var.set(e['end'])

    def _sync_start_end_to_entry(self):
        if self._current_entry_idx is None or self._current_entry_idx >= len(self.entries):
            return
        e = self.entries[self._current_entry_idx]
        e['start'] = self.start_var.get().strip()
        e['end'] = self.end_var.get().strip()

    def _comparison_output_dir(self):
        """Package-level folder for the combined comparison plot — not inside any single
        experiment's own folder, since it doesn't belong to any one of them."""
        out_dir = os.path.join(get_package_root(), "plots", "buzzphono_comparison")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _zones_path(exp):
        """custom_zones.json always lives at the experiment's own root — always this experiment's
        file, never browsed from elsewhere or reused across experiments."""
        return os.path.join(exp, "custom_zones.json")

    @staticmethod
    def _count_tracking_files(exp):
        """Number of forward_mosq_tracks_* segment files in final_tracking_data — the canonical
        'has this experiment been tracked yet?' signal. BuzzPhono reads tracked mosquitoes, so an
        empty final_tracking_data means there is nothing to analyse."""
        tdir = os.path.join(exp, 'final_tracking_data')
        if not os.path.isdir(tdir):
            return 0
        return sum(1 for f in os.listdir(tdir)
                   if f.startswith('forward_mosq_tracks_') and not f.startswith('.'))

    def refresh_experiment(self):
        # No-op: this tab is no longer tied to "the currently loaded experiment" (ui_manager.py
        # still calls this on experiment load; kept harmless rather than editing that call site).
        pass

    def _derive_date_range(self, exp):
        """Parse {YYYYMMDD}_{HHMMSS} out of final_tracking_data segment filenames and return
        (start_iso, end_iso) covering them, or None. Cheap: only lists filenames."""
        tdir = os.path.join(exp, 'final_tracking_data')
        if not os.path.isdir(tdir):
            return None
        stamps = []
        for name in os.listdir(tdir):
            match = _TS_RE.search(name)
            if match:
                d, t = match.group(1), match.group(2)
                stamps.append(f"{d[0:4]}-{d[4:6]}-{d[6:8]} {t[0:2]}:{t[2:4]}:{t[4:6]}")
        if not stamps:
            return None
        stamps.sort()
        # pad the end by ~a day so the last segment's full hour is included
        import pandas as pd
        end = (pd.Timestamp(stamps[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        return stamps[0], end

    def _find_background_image(self, exp):
        """Reference frame to draw the zone on: prefer background_with_borders.png, else any
        images_mortality/Cage*.png, else background.png."""
        candidates = [os.path.join(exp, "background_with_borders.png"),
                      os.path.join(exp, "background.png")]
        for c in candidates:
            if os.path.isfile(c):
                return c
        mort = os.path.join(exp, "images_mortality")
        if os.path.isdir(mort):
            pngs = sorted(f for f in os.listdir(mort) if f.lower().endswith(".png") and not f.startswith("."))
            if pngs:
                return os.path.join(mort, pngs[0])
        return None

    def define_speaker_zone(self):
        """Open the polygon editor to draw the speaker zone and save it to custom_zones.json, for
        whichever experiment is currently selected in the list."""
        exp = self._selected_experiment_dir()
        if not exp or not os.path.isdir(exp):
            messagebox.showerror("BuzzPhono", "Select an experiment in the list first.")
            return
        bg = self._find_background_image(exp)
        if not bg:
            messagebox.showerror("BuzzPhono", "No reference image found (looked for "
                                             "background_with_borders.png / images_mortality/*.png).\n\n"
                                             "Run the Setup tab -> \"Extract Images from Video\" / "
                                             "\"Get Background from Images\" first to extract a background "
                                             "frame, then define the zone.")
            return
        save_path = self._zones_path(exp)

        def on_saved(path):
            self.log(f"BuzzPhono: speaker zone saved to {path}")
            self._refresh_zone_preview()

        SpeakerZoneEditor(self.root, bg, save_path, zone_name="speaker",
                          on_saved=on_saved, log=self.log)

    def add_comparison_zone(self):
        """Clone the speaker zone's exact polygon (same size/shape) and let the user drag it to
        a new position — e.g. an equally-sized sugar-feeder region — so the analysis can compare
        resting fraction between the speaker zone and other regions on equal footing. Operates on
        whichever experiment is currently selected in the list."""
        exp = self._selected_experiment_dir()
        if not exp or not os.path.isdir(exp):
            messagebox.showerror("BuzzPhono", "Select an experiment in the list first.")
            return
        zones_path = self._zones_path(exp)
        if not os.path.isfile(zones_path):
            messagebox.showerror("BuzzPhono", "Define the speaker zone first — a comparison zone "
                                             "is cloned from its exact shape/size.")
            return
        try:
            with open(zones_path) as f:
                zones = json.load(f) or {}
        except Exception as exc:
            messagebox.showerror("BuzzPhono", f"Could not read {zones_path}:\n{exc}")
            return
        if "speaker" not in zones:
            messagebox.showerror("BuzzPhono", "No 'speaker' zone found in custom_zones.json yet — "
                                             "define it first (button above).")
            return
        base_polygon = zones["speaker"]["polygon"] if isinstance(zones["speaker"], dict) else zones["speaker"]

        bg = self._find_background_image(exp)
        if not bg:
            messagebox.showerror("BuzzPhono", "No reference image found (looked for "
                                             "background_with_borders.png / images_mortality/*.png).\n\n"
                                             "Run the Setup tab -> \"Extract Images from Video\" / "
                                             "\"Get Background from Images\" first to extract a background "
                                             "frame, then define the zone.")
            return

        existing_names = set(zones.keys())
        name = simpledialog.askstring(
            "Comparison zone", "Name for this zone (e.g. sugar):", initialvalue="sugar", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name or name == "speaker":
            messagebox.showerror("BuzzPhono", "Give the comparison zone a name other than 'speaker'.")
            return
        if name in existing_names and not messagebox.askyesno(
                "Comparison zone", f"'{name}' already exists in custom_zones.json — replace it?"):
            return

        # The cage is square, so the natural comparable position for a same-size region is
        # either the mirror image of the speaker zone (opposite wall, same orientation — for a
        # region on the other side wall) or a 90-degree rotation about the cage center (for a
        # region on the top/bottom wall, which needs to be turned to fit that wall).
        axis = self._ask_reflection_axis()
        if axis is None:
            return
        center = self._reflection_center(exp, bg)
        if axis == 'horizontal':
            start_polygon = self._mirror_horizontal(base_polygon, center)
        elif axis == 'up':
            start_polygon = self._rotate_polygon(base_polygon, 'cw', center)
        else:  # 'down'
            start_polygon = self._rotate_polygon(base_polygon, 'ccw', center)

        def on_saved(path):
            self.log(f"BuzzPhono: comparison zone '{name}' saved to {path}")
            self._refresh_zone_preview()

        ZoneClonerEditor(self.root, bg, zones_path, start_polygon, zone_name=name,
                         on_saved=on_saved, log=self.log)

    def _ask_reflection_axis(self):
        """Small modal: how to position the new comparison zone relative to the speaker zone.
        Left/Right mirrors across the cage center (opposite side wall, same orientation). Up/Down
        instead rotates the shape 90 degrees about the cage center — a zone on a side wall has to
        be turned, not just mirrored, to fit a top or bottom wall on a square cage. Returns
        'horizontal', 'up', 'down', or None if cancelled."""
        result = {'axis': None}
        dlg = tk.Toplevel(self.root)
        dlg.title("Comparison zone orientation")
        dlg.resizable(False, False)
        tk.Label(dlg, text="Position the new zone relative to the speaker zone:").pack(padx=14, pady=(14, 8))
        btns = tk.Frame(dlg)
        btns.pack(padx=14, pady=(0, 14))

        def pick(axis):
            result['axis'] = axis
            dlg.destroy()

        tk.Button(btns, text="Left ↔ Right\n(mirror)", width=14, command=lambda: pick('horizontal')).pack(
            side=tk.LEFT, padx=4)
        tk.Button(btns, text="Up\n(rotate 90° CW)", width=14, command=lambda: pick('up')).pack(
            side=tk.LEFT, padx=4)
        tk.Button(btns, text="Down\n(rotate 90° CCW)", width=14, command=lambda: pick('down')).pack(
            side=tk.LEFT, padx=4)
        dlg.transient(self.root)
        dlg.grab_set()
        self.root.wait_window(dlg)
        return result['axis']

    def _reflection_center(self, exp, bg_path):
        """Reference point to mirror across: the cage border's own centroid if it's been drawn,
        otherwise the background image's center (a reasonable default since the cage is square
        and roughly centered in frame). Reads `exp`'s own settings file — NOT
        self.experiment_manager's, which may be a different experiment than the one selected
        here."""
        settings_file = os.path.join(exp, 'buzzwatch_track_settings.yml')
        if os.path.isfile(settings_file):
            try:
                import yaml
                with open(settings_file) as f:
                    settings = yaml.safe_load(f) or {}
                cage_points = settings.get('cage_border_points') or []
                if len(cage_points) >= 3:
                    xs = [p[0] for p in cage_points]
                    ys = [p[1] for p in cage_points]
                    return (sum(xs) / len(xs), sum(ys) / len(ys))
            except Exception:
                pass
        with Image.open(bg_path) as img:
            w, h = img.size
        return (w / 2.0, h / 2.0)

    @staticmethod
    def _mirror_horizontal(polygon, center):
        """Left/Right mirror across the cage center — same orientation, opposite side wall."""
        cx, _ = center
        return [[2 * cx - x, y] for x, y in polygon]

    @staticmethod
    def _rotate_polygon(polygon, direction, center):
        """Rotate a polygon 90 degrees about `center`. Image coordinates (y increases downward),
        so 'cw'/'ccw' match what the user sees on screen: cw maps (dx, dy) -> (-dy, dx), ccw maps
        (dx, dy) -> (dy, -dx), relative to center."""
        cx, cy = center
        result = []
        for x, y in polygon:
            dx, dy = x - cx, y - cy
            if direction == 'cw':
                nx, ny = -dy, dx
            else:
                nx, ny = dy, -dx
            result.append([cx + nx, cy + ny])
        return result

    # Distinct colours cycled across zones in the preview overlay (speaker first).
    _ZONE_COLORS = [(255, 140, 0), (0, 170, 255), (0, 200, 90), (220, 60, 200),
                    (240, 200, 0), (255, 60, 60)]

    def show_background_with_zones(self):
        """Show the background-with-borders image for whichever experiment is selected in the
        list, with every zone in custom_zones.json overlaid (speaker + any comparison zones).

        The border-redraw call (experiment_manager.update_border_image()) only operates on
        whichever experiment is currently loaded in "1 · Setup" (it has no path parameter — deep,
        frozen Experiment-object machinery, not something this tab should reach into). So: only
        trigger a fresh redraw when the selected entry IS that loaded experiment; otherwise just
        show whatever background image already exists on disk for it (still correct, just not
        freshly redrawn) with a log line explaining why."""
        exp = self._selected_experiment_dir()
        if not exp:
            messagebox.showerror("BuzzPhono", "Select an experiment in the list first.")
            return
        current = getattr(self.experiment_manager, 'folder_analysis', None)
        if current and os.path.normpath(current) == os.path.normpath(exp):
            try:
                self.experiment_manager.update_border_image()
            except Exception as exc:
                self.log(f"BuzzPhono: error drawing borders: {exc}")
        else:
            idx = self._selected_entry_idx()
            alias = self.entries[idx]['alias'] if idx is not None else exp
            self.log(f"BuzzPhono: '{alias}' isn't the experiment currently loaded in 1 · Setup, "
                     "so its border image can't be freshly redrawn from here — showing whatever "
                     "background image already exists for it.")
        border_png = os.path.join(exp, "background_with_borders.png")
        base = border_png if os.path.isfile(border_png) else self._find_background_image(exp)
        if not base or not os.path.isfile(base):
            messagebox.showerror("BuzzPhono", "No reference image found for this experiment.\n\n"
                                             "Run the Setup tab -> \"Extract Images from Video\" / "
                                             "\"Get Background from Images\" first.")
            return
        composited = self._overlay_zones(base, self._zones_path(exp))
        if composited is not None:
            self.preview.show_image(composited)
        else:
            self.preview.show_path(base)

    def _overlay_zones(self, base_image_path, zones_path):
        """Return a PIL image of the reference frame with each custom zone drawn as a translucent
        filled polygon + labelled outline, or None if there are no zones to draw. Never modifies
        the source PNG on disk."""
        if not os.path.isfile(zones_path):
            return None
        try:
            with open(zones_path) as f:
                zones = json.load(f) or {}
        except Exception as exc:
            self.log(f"BuzzPhono: could not read {zones_path}: {exc}")
            return None
        if not zones:
            return None
        try:
            from PIL import ImageDraw
            base = Image.open(base_image_path).convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            # speaker first, then the rest in a stable order
            names = (["speaker"] if "speaker" in zones else []) + sorted(
                n for n in zones if n != "speaker")
            for i, name in enumerate(names):
                zone = zones[name]
                polygon = zone.get("polygon") if isinstance(zone, dict) else zone
                if not polygon or len(polygon) < 3:
                    continue
                color = self._ZONE_COLORS[i % len(self._ZONE_COLORS)]
                pts = [(float(x), float(y)) for x, y in polygon]
                draw.polygon(pts, fill=color + (70,), outline=color + (255,))
                # thicker outline by re-stroking the closed loop
                draw.line(pts + [pts[0]], fill=color + (255,), width=3)
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                draw.text((cx, cy), name, fill=(255, 255, 255, 255))
            return Image.alpha_composite(base, overlay).convert("RGB")
        except Exception as exc:
            self.log(f"BuzzPhono: could not overlay zones: {exc}")
            return None

    def _refresh_zone_preview(self):
        """Re-composite the zone overlay onto whatever reference frame is already on disk, without
        regenerating the border image. Called after a zone is saved so the new zone (speaker,
        mirrored, or rotated up/down comparison zone) shows up in the preview immediately."""
        exp = self._selected_experiment_dir()
        if not exp:
            return
        border_png = os.path.join(exp, "background_with_borders.png")
        base = border_png if os.path.isfile(border_png) else self._find_background_image(exp)
        if not base:
            return
        composited = self._overlay_zones(base, self._zones_path(exp))
        if composited is not None:
            self.preview.show_image(composited)

    def preview_zones(self):
        """Standalone zone preview over the plain reference frame (no border redraw). Not
        currently wired to a button; kept correct (selected-entry-aware) for future use."""
        exp = self._selected_experiment_dir()
        if not exp:
            messagebox.showerror("BuzzPhono", "Select an experiment in the list first.")
            return
        base = self._find_background_image(exp)
        if not base:
            messagebox.showerror("BuzzPhono", "No reference image found. Extract images / get "
                                             "background first.")
            return
        composited = self._overlay_zones(base, self._zones_path(exp))
        if composited is not None:
            self.preview.show_image(composited)
        else:
            self.log("BuzzPhono: no zones defined yet to preview.")
            self.preview.show_path(base)

    def _set_status(self, text):
        self.status_var.set(text)

    def _int_or(self, var, default):
        try:
            return int(str(var.get()).strip())
        except (ValueError, TypeError):
            return default

    def _entries_ready_to_run(self):
        """Validate self.entries, returning ready (folder, group, start, end, zones_path) tuples.
        Entries missing tracking data, a speaker zone, or a Start/End window are skipped (logged,
        not a blocking error) rather than dead-ending the whole run — this loops over several
        experiments at once, unlike the old single-experiment "define zone now?" prompt."""
        if not self.entries:
            messagebox.showerror("BuzzPhono", "Add at least one experiment first.")
            return None
        ready = []
        for e in self.entries:
            exp, alias = e['folder'], e['alias']
            if self._count_tracking_files(exp) == 0:
                self.log(f"BuzzPhono: skipping '{alias}' — no tracking data "
                         "(final_tracking_data is empty).")
                continue
            zones = self._zones_path(exp)
            if not os.path.isfile(zones):
                self.log(f"BuzzPhono: skipping '{alias}' — no speaker zone defined yet (select "
                         "it in the list and use \"Define speaker zone…\").")
                continue
            start, end = e['start'].strip(), e['end'].strip()
            if not start or not end:
                self.log(f"BuzzPhono: skipping '{alias}' — no Start/End time set.")
                continue
            ready.append((exp, e['group'], start, end, zones))
        if not ready:
            messagebox.showerror(
                "BuzzPhono — nothing ready to run",
                "None of the added experiments are ready: each needs tracking data, a defined "
                "speaker zone, and a Start/End time. Check the log for what's missing.")
            return None
        return ready

    def run_analysis(self):
        if self._running:
            self.log("BuzzPhono: a run is already in progress.")
            return
        self._sync_start_end_to_entry()
        ready = self._entries_ready_to_run()
        if ready is None:
            return

        # Snapshot everything on the main thread. ON/OFF windows + resting-only are shared across
        # every experiment in this run (same as before — these were never per-experiment).
        on_start = self._int_or(self.on_start_var, 40)
        on_end = self._int_or(self.on_end_var, 50)
        off_start = self._int_or(self.off_start_var, 28)
        off_end = self._int_or(self.off_end_var, 38)
        resting_only = bool(self.resting_only_var.get())
        self._save_params(on_start, on_end, off_start, off_end, resting_only)
        comparison_dir = self._comparison_output_dir()

        self._running = True
        self._set_status(f"Running BuzzPhono analysis on {len(ready)} experiment(s)...")
        self.ui_manager.set_status(action="BuzzPhono: running analysis…", busy=True)
        self._open_progress("BuzzPhono", f"Running {len(ready)} experiment(s)…")
        self.log(f"BuzzPhono: running on {len(ready)} experiment(s) "
                 f"(resting_only={resting_only})...")

        def worker():
            try:
                from buzzphono import plot_speaker_distance_phonotaxis as ph
                # Apply the ON/OFF window overrides onto the module globals the analysis reads.
                ph.STIM_START_MIN = on_start
                ph.STIM_END_MIN = on_end
                ph.OFF_START_MIN = off_start
                ph.OFF_END_MIN = off_end

                succeeded = []
                last_out_dir = None
                for exp, group, start, end, zones in ready:
                    try:
                        # Explicit user "Run" always regenerates (honors any changed params);
                        # the skip-if-current optimisation applies to automated/dashboard runs.
                        # Unchanged, per-experiment full output -- must not regress.
                        last_out_dir = ph.run_speaker_distance_analysis(
                            exp, start=start, end=end, custom_zones_path=zones,
                            resting_only=resting_only, force_replot=True)
                        succeeded.append((exp, group, start, end))
                    except Exception as exc:
                        self.root.after(0, lambda a=group, e=exc:
                                        self.log(f"BuzzPhono: '{a}' failed: {e}"))

                # New: one combined plot overlaying every successfully-run experiment's
                # speaker-response profile, on top of (not instead of) each one's own output.
                comparison_png = None
                if succeeded:
                    ph.run_multi_speaker_comparison(succeeded, output_dir=comparison_dir)
                    candidate = os.path.join(
                        comparison_dir, "speaker_phase_minute_of_hour_comparison.png")
                    if os.path.isfile(candidate):
                        comparison_png = candidate

                preview = comparison_png or self._pick_preview_png(last_out_dir)
                self.root.after(0, lambda: self._on_done(len(succeeded), len(ready), preview))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, n_succeeded, n_total, preview):
        self._running = False
        self._close_progress()
        self._set_status(f"Done: {n_succeeded}/{n_total} experiment(s).")
        self.ui_manager.set_status(action="BuzzPhono: analysis complete", busy=False)
        self.log(f"BuzzPhono: finished ({n_succeeded}/{n_total} experiment(s) succeeded).")
        if preview:
            self.preview.show_path(preview)

    def _on_error(self, exc):
        self._running = False
        self._close_progress()
        self._set_status("Error in BuzzPhono analysis.")
        self.ui_manager.set_status(action="BuzzPhono: analysis failed", busy=False)
        self.log(f"BuzzPhono: failed: {exc}")
        messagebox.showerror("BuzzPhono", f"Analysis failed:\n{exc}")

    # ---------------------------------------------------------------- progress + persistence
    def _open_progress(self, title, message):
        try:
            from progress_dialog import ProgressDialog
            self._progress = ProgressDialog(self.root, title=title, message=message)
            self._progress.pulse()   # indeterminate: the analysis reports no incremental %
        except Exception:
            self._progress = None

    def _close_progress(self):
        prog = getattr(self, '_progress', None)
        if prog is not None:
            prog.close()
            self._progress = None

    CONFIG_KEY = 'buzzphono_params'

    def _load_saved_params(self):
        """Return persisted BuzzPhono params from config.json, or {}."""
        try:
            saved = self.experiment_manager.config.get(self.CONFIG_KEY, {})
            if isinstance(saved, dict):
                return saved
        except Exception:
            pass
        return {}

    def _save_params(self, on_start, on_end, off_start, off_end, resting_only):
        """Persist the BuzzPhono stimulus windows + resting-only flag to config.json."""
        try:
            self.experiment_manager.config[self.CONFIG_KEY] = {
                'on_start': on_start, 'on_end': on_end,
                'off_start': off_start, 'off_end': off_end,
                'resting_only': bool(resting_only),
            }
            self.experiment_manager.save_config(self.experiment_manager.config)
        except Exception as exc:
            self.log(f"BuzzPhono: could not save parameters: {exc}")

    def _pick_preview_png(self, out_dir):
        if not out_dir or not os.path.isdir(out_dir):
            return None
        # Prefer the paired/recruitment proportion figures if present, else any PNG.
        pngs = [f for f in sorted(os.listdir(out_dir)) if f.lower().endswith(".png") and not f.startswith(".")]
        if not pngs:
            return None
        for pref in ("paired", "recruitment", "phase", "proportion"):
            for f in pngs:
                if pref in f.lower():
                    return os.path.join(out_dir, f)
        return os.path.join(out_dir, pngs[0])


class SpeakerZoneEditor:
    """Interactive polygon editor for the BuzzPhono speaker zone.

    Opens a Toplevel canvas over a reference frame; the user left-clicks polygon vertices and
    finishes with double-click / Enter / "Save & Close". Saves
    ``{"<zone>": {"polygon": [[x, y], ...]}}`` in IMAGE-pixel coordinates to custom_zones.json —
    the shape ``_load_speaker_zone`` in plot_speaker_distance_phonotaxis.py expects. Any existing
    zones in the file are preserved (only the edited zone is replaced).
    """
    MAX_W, MAX_H = 1100, 800

    def __init__(self, root, image_path, save_path, zone_name="speaker", on_saved=None, log=None):
        self.save_path = save_path
        self.zone_name = zone_name
        self.on_saved = on_saved
        self.log = log or (lambda m: None)
        self.points = []          # canvas coords
        self._dot_ids = []

        img = Image.open(image_path)
        w, h = img.size
        self.scale = min(self.MAX_W / w, self.MAX_H / h, 1.0)
        disp = img if self.scale >= 1.0 else img.resize((int(w * self.scale), int(h * self.scale)), Image.LANCZOS)
        self.disp_w, self.disp_h = disp.size

        self.win = tk.Toplevel(root)
        self.win.title(f"Define '{zone_name}' zone — click vertices, double-click/Enter to finish")
        self._tkimg = ImageTk.PhotoImage(disp)
        self.canvas = tk.Canvas(self.win, width=self.disp_w, height=self.disp_h, cursor="crosshair")
        self.canvas.pack(side=tk.TOP)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg)

        bar = tk.Frame(self.win)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(bar, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(bar, text="Save & Close", command=self.save).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(bar, text="Cancel", command=self.win.destroy).pack(side=tk.LEFT, padx=4, pady=4)
        self.status = tk.Label(bar, text="0 points", fg="gray")
        self.status.pack(side=tk.RIGHT, padx=8)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Double-Button-1>", lambda e: self.save())
        self.win.bind("<Return>", lambda e: self.save())
        self.win.bind("<Escape>", lambda e: self.win.destroy())

    def on_click(self, event):
        x, y = float(event.x), float(event.y)
        self.points.append((x, y))
        r = 3
        self._dot_ids.append(self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ff3", outline="black"))
        if len(self.points) >= 2:
            x0, y0 = self.points[-2]
            self._dot_ids.append(self.canvas.create_line(x0, y0, x, y, fill="#ff8c00", width=2))
        self.status.config(text=f"{len(self.points)} points")

    def undo(self):
        if not self.points:
            return
        self.points.pop()
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg)
        self._dot_ids = []
        pts = self.points
        self.points = []
        for (x, y) in pts:
            self.on_click(type("E", (), {"x": x, "y": y}))

    def clear(self):
        self.points = []
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg)
        self.status.config(text="0 points")

    def save(self):
        if len(self.points) < 3:
            messagebox.showwarning("Speaker zone", "Need at least 3 points to define a polygon.")
            return
        # canvas -> image-pixel coords
        polygon = [[round(x / self.scale, 2), round(y / self.scale, 2)] for (x, y) in self.points]
        zones = {}
        try:
            if os.path.isfile(self.save_path):
                with open(self.save_path) as f:
                    zones = json.load(f) or {}
        except Exception:
            zones = {}
        zones[self.zone_name] = {"polygon": polygon}
        try:
            with open(self.save_path, "w") as f:
                json.dump(zones, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Speaker zone", f"Could not save:\n{exc}")
            return
        self.log(f"BuzzPhono: saved {len(polygon)}-point '{self.zone_name}' zone -> {self.save_path}")
        if self.on_saved:
            self.on_saved(self.save_path)
        self.win.destroy()


class ZoneClonerEditor:
    """Clone an existing zone's exact polygon (same size/shape) and let the user drag it to a new
    position, for defining a comparison region (e.g. a sugar feeder) that's directly comparable to
    the speaker zone because it's identical in size — only translated, never resized or rotated.

    Saves ``{"<zone>": {"polygon": [[x, y], ...]}}`` into the same custom_zones.json, merging with
    (not replacing) any other zones already there — same file shape SpeakerZoneEditor writes.
    """
    MAX_W, MAX_H = 1100, 800

    def __init__(self, root, image_path, save_path, base_polygon, zone_name, on_saved=None, log=None):
        self.save_path = save_path
        self.zone_name = zone_name
        self.on_saved = on_saved
        self.log = log or (lambda m: None)
        self._drag_last = None

        img = Image.open(image_path)
        w, h = img.size
        self.scale = min(self.MAX_W / w, self.MAX_H / h, 1.0)
        disp = img if self.scale >= 1.0 else img.resize((int(w * self.scale), int(h * self.scale)), Image.LANCZOS)
        self.disp_w, self.disp_h = disp.size

        # base_polygon is in IMAGE-pixel coords (as stored in custom_zones.json) -> canvas coords.
        self.points = [(x * self.scale, y * self.scale) for x, y in base_polygon]

        self.win = tk.Toplevel(root)
        self.win.title(f"Place '{zone_name}' zone — drag the shape, Save & Close when positioned")
        self._tkimg = ImageTk.PhotoImage(disp)
        self.canvas = tk.Canvas(self.win, width=self.disp_w, height=self.disp_h, cursor="fleur")
        self.canvas.pack(side=tk.TOP)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg)
        self.poly_id = self.canvas.create_polygon(self._flat_points(), outline="#ff8c00",
                                                  fill="#ff8c00", stipple="gray50", width=2)

        bar = tk.Frame(self.win)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="Save & Close", command=self.save).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(bar, text="Cancel", command=self.win.destroy).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Label(bar, text="Drag the shaded shape to reposition it (size/shape stay fixed).",
                 fg="gray").pack(side=tk.LEFT, padx=8)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.win.bind("<Return>", lambda e: self.save())
        self.win.bind("<Escape>", lambda e: self.win.destroy())

    def _flat_points(self):
        return [c for pt in self.points for c in pt]

    def on_press(self, event):
        self._drag_last = (event.x, event.y)

    def on_drag(self, event):
        if self._drag_last is None:
            return
        dx, dy = event.x - self._drag_last[0], event.y - self._drag_last[1]
        self.points = [(x + dx, y + dy) for x, y in self.points]
        self._drag_last = (event.x, event.y)
        self.canvas.coords(self.poly_id, *self._flat_points())

    def save(self):
        # canvas -> image-pixel coords, same convention SpeakerZoneEditor.save() uses.
        polygon = [[round(x / self.scale, 2), round(y / self.scale, 2)] for (x, y) in self.points]
        zones = {}
        try:
            if os.path.isfile(self.save_path):
                with open(self.save_path) as f:
                    zones = json.load(f) or {}
        except Exception:
            zones = {}
        zones[self.zone_name] = {"polygon": polygon}
        try:
            with open(self.save_path, "w") as f:
                json.dump(zones, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Comparison zone", f"Could not save:\n{exc}")
            return
        self.log(f"BuzzPhono: saved {len(polygon)}-point '{self.zone_name}' zone -> {self.save_path}")
        if self.on_saved:
            self.on_saved(self.save_path)
        self.win.destroy()
