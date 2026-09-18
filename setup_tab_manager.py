"""Activity Setup tab: cage-border drawing + reference-image extraction.

Modeled on buzzphono_tab_manager.py's left control column, minus all speaker/zone code (Activity
has no zone concept). Reuses the same threaded-image-step / draw-and-refresh / preview patterns,
kept as separate copies here (matches the existing buzzphono/buzzswarm duplication style) rather
than moving method bodies between files.

Exposes two entry points:
  - build_setup_section(parent): builds just the controls column into a caller-supplied frame
    (reused by the BuzzSwarm tab).
  - init_setup_tab(tab): builds the controls column + a right-hand preview panel, for the
    Activity "Setup" tab.
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from tooltip import add_tooltip
from scrollable_frame import make_scrollable_column
from image_preview import ImagePreview


class SetupTabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.tab = None
        self.exp_var = None
        self.status_var = None
        self.preview = None
        self.force_rerun_images_var = None

    # ---------------------------------------------------------------- UI build
    def build_setup_section(self, parent):
        """Build just the Reference-images + Cage-borders controls into `parent` (no header/
        experiment-label/refresh-button — callers that embed this into their own tab, e.g.
        BuzzSwarm, already build their own equivalent header). Returns `parent`."""
        images_frame = tk.LabelFrame(parent, text="Reference images")
        images_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        self.force_rerun_images_var = tk.BooleanVar(value=False)
        tk.Button(images_frame, text="Extract Images from Video",
                  command=lambda: self._run_image_step(
                      'get_images_from_video', "Extracting images from video...")).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=2)
        tk.Button(images_frame, text="Get Background from Images",
                  command=lambda: self._run_image_step(
                      'get_background_from_images', "Computing background from images...")).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=2)
        tk.Checkbutton(images_frame, text="Force re-run", variable=self.force_rerun_images_var
                       ).pack(side=tk.TOP, anchor="w", padx=4, pady=(0, 2))

        borders_frame = tk.LabelFrame(parent, text="Cage borders")
        borders_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        tk.Button(borders_frame, text="Show background with borders",
                  command=lambda: self._draw_borders_and_refresh(
                      lambda: self.experiment_manager.update_border_image())).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=2)
        tk.Button(borders_frame, text="Draw Cage Borders",
                  command=lambda: self._draw_borders_and_refresh(
                      lambda: self.experiment_manager.draw_cage_borders())).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=2)
        tk.Button(borders_frame, text="Reset Drawn Borders",
                  command=lambda: self._draw_borders_and_refresh(
                      lambda: self.experiment_manager.reset_drawn_borders())).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(6, 2))

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(parent, textvariable=self.status_var, fg="blue",
                 wraplength=280, justify="left").pack(side=tk.TOP, anchor="w", pady=(8, 0))
        return parent

    def init_setup_tab(self, tab: ttk.Frame):
        self.tab = tab
        controls_outer, controls = make_scrollable_column(tab)
        controls_outer.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(controls, text="Setup", font=("TkDefaultFont", 12, "bold")).pack(
            side=tk.TOP, anchor="w")
        tk.Label(controls, text="Extract reference images and draw cage borders.",
                 fg="gray", wraplength=280, justify="left").pack(side=tk.TOP, anchor="w", pady=(0, 8))

        self.exp_var = tk.StringVar(value=self._experiment_label())
        tk.Label(controls, textvariable=self.exp_var, wraplength=280, justify="left",
                 fg="#0a5").pack(side=tk.TOP, anchor="w", pady=(0, 6))
        tk.Button(controls, text="Refresh experiment",
                  command=self.refresh_experiment).pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        self.build_setup_section(controls)

        ttk.Separator(controls, orient="horizontal").pack(side=tk.TOP, fill=tk.X, pady=(10, 8))
        new_exp_btn = tk.Button(controls, text="Set up new experiment",
                                 command=self._start_new_experiment)
        new_exp_btn.pack(side=tk.TOP, fill=tk.X)
        add_tooltip(new_exp_btn, "Clear the source folder, dates and experiment fields on the "
                    "Experiment & Video tab so you can prepare the next experiment right away. "
                    "Doesn't affect the experiment you just created — it's already saved.")

        preview = tk.Frame(tab)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preview = ImagePreview(preview, placeholder="Draw borders or extract images to preview here.",
                                    log=lambda msg: self.log(f"Setup: {msg}"))

        self.refresh_experiment()

    # ---------------------------------------------------------------- helpers
    def _start_new_experiment(self):
        """End of the setup flow: clear the intake form and send the user back to Step A."""
        wiz = getattr(self.ui_manager, 'intake_wizard', None)
        if wiz is None:
            return
        wiz.reset_form()
        nb = getattr(self.ui_manager, 'setup_notebook', None)
        subtabs = getattr(self.ui_manager, 'setup_subtabs', None)
        if nb is not None and subtabs:
            nb.select(subtabs['intake'])

    def _current_experiment_dir(self):
        return getattr(self.experiment_manager, 'folder_analysis', None)

    def _experiment_label(self):
        exp = self._current_experiment_dir()
        if exp:
            return f"Experiment: {os.path.basename(os.path.normpath(exp))}"
        return "No experiment loaded (open one in the Intake Wizard first)."

    def refresh_experiment(self):
        if self.exp_var is not None:
            self.exp_var.set(self._experiment_label())
        exp = self._current_experiment_dir()
        if exp and os.path.isdir(exp):
            bg = self._find_background_image(exp)
            if bg:
                self.preview.show_path(bg)

    def _find_background_image(self, exp):
        """Reference frame to display: prefer background_with_borders.png, else background.png,
        else the first images_mortality/*.png."""
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

    def _run_image_step(self, method_name, log_msg):
        """Run get_images_from_video/get_background_from_images on a background thread, marshaling
        experiment_manager/experiment logging back to the main thread for the duration (Tk widget
        updates from a background thread aren't safe). Mirrors
        buzzphono_tab_manager.BuzzphonoTabManager._run_image_step."""
        exp = self._current_experiment_dir()
        if not exp or not os.path.isdir(exp):
            messagebox.showerror("Setup", "No experiment loaded. Open an experiment first.")
            return
        force = bool(self.force_rerun_images_var.get())
        self.log(f"Setup: {log_msg}")

        def safe_log(msg):
            self.root.after(0, lambda m=msg: self.log(m))

        def worker():
            em = self.experiment_manager
            saved_em_log = em.log
            saved_exp_log = getattr(em.experiment, 'log', None)
            em.log = safe_log
            try:
                em.experiment.log = safe_log
            except Exception:
                pass
            try:
                getattr(em, method_name)(force)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Setup", f"{log_msg.rstrip('.')} failed:\n{exc}"))
            finally:
                em.log = saved_em_log
                try:
                    em.experiment.log = saved_exp_log
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _draw_borders_and_refresh(self, draw_func):
        """Run a cage-border-drawing call (main thread — opens an interactive OpenCV window), then
        show the result in the right-hand preview panel. Mirrors
        buzzphono_tab_manager.BuzzphonoTabManager._draw_borders_and_refresh."""
        exp = self._current_experiment_dir()
        if not exp:
            messagebox.showerror("Setup", "No experiment loaded. Open an experiment first.")
            return
        try:
            draw_func()
        except Exception as exc:
            self.log(f"Setup: error drawing borders: {exc}")
            return
        border_png = os.path.join(exp, "background_with_borders.png")
        if not os.path.isfile(border_png):
            # The draw call already logged a specific reason (e.g. no reference images yet).
            return
        self.preview.show_path(border_png)
