# buzzsuite_app.py
import sys
import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from tooltip import add_tooltip
from ui_manager import UIManager
from experiment_manager import ExperimentManager
from video_manager import VideoManager
from image_manager import ImageManager
from state_manager import StateManager
from config_manager import ConfigManager
from plot_manager import PlotManager

PATH_TO_APP = os.path.dirname(__file__)
sys.path.append(PATH_TO_APP)

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BuzzSuite")

        # Set the application icon
        self._set_app_icon()

        #self._maximize_window()


        # Initialize ConfigManager
        self.config_manager = ConfigManager(os.path.join(PATH_TO_APP, "config.json"))
        self.config = self.config_manager.config

        # Initialize Managers
        self._initialize_managers()

        # Create a unified logging area in the main window
        self._setup_logging_area()

        # Persistent status bar (experiment / module / last action / spinner).
        self._setup_status_bar()

        # Show the welcome panel first; the notebook tabs are built once the user picks
        # which modules (Activity / Swarming / Phonotaxis) to work with (or opens a recent one).
        self._show_assay_selector()

    def _set_app_icon(self):
        logo_path = os.path.join(PATH_TO_APP, "app_logo.png")
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            logo_photo = ImageTk.PhotoImage(logo_image)
            self.root.iconphoto(False, logo_photo)

    def _initialize_managers(self):
        self.experiment_manager = ExperimentManager(self.log, self.config_manager)
        self.video_manager = VideoManager(self.log)
        self.image_manager = ImageManager(self.log)
        self.state_manager = StateManager()
        self.plot_manager = PlotManager(self.log)
        self.ui_manager = UIManager(self.root, self.experiment_manager, self.video_manager, self.image_manager, self.log, self.state_manager,self.plot_manager)
        
        self.experiment_manager.ui_manager = self.ui_manager

        # Set central log function to all managers
        self.experiment_manager.log = self.log
        self.video_manager.log = self.log
        self.image_manager.log = self.log
        self.ui_manager.log = self.log

    def _show_assay_selector(self):
        from assay_selector import AssaySelector
        recent = self.experiment_manager.get_recent_experiments()
        self._assay_selector = AssaySelector(self.root, self._on_launch,
                                             recent_experiments=recent)

    def _on_launch(self, open_recent_path=None, focus_intake=False):
        # Build all notebook tabs (Setup / Analysis / Plotting + Dashboard + H264 — no module
        # choice anymore), then load an experiment.
        self.ui_manager.init_tabs()
        if open_recent_path:
            self._open_experiment_path(open_recent_path)
        else:
            self._load_last_experiment()
        if focus_intake:
            self._focus_tab('setup_section')

    def _focus_tab(self, tab_key):
        try:
            tabs = getattr(self.ui_manager, 'tabs', {})
            if tab_key in tabs and self.ui_manager.notebook is not None:
                self.ui_manager.notebook.select(tabs[tab_key])
        except Exception:
            pass

    def _open_experiment_path(self, path):
        """Load a specific experiment file (.json or .pkl) with user-visible error handling."""
        from tkinter import messagebox
        try:
            if isinstance(path, str) and path.lower().endswith('.json'):
                self.experiment_manager.load_experiment_from_json(path)
            else:
                self.experiment_manager.load_experiment(path)
            self.ui_manager.update_ui_after_loading_experiment()
            self.ui_manager.set_status(experiment=self.experiment_manager.experiment_alias,
                                       action="Loaded experiment")
        except Exception as exc:
            self.log(f"Error loading experiment: {exc}")
            messagebox.showerror("Open experiment",
                                 f"Could not open the experiment:\n{path}\n\n{exc}")

    def _setup_logging_area(self):
        self.log_frame = tk.Frame(self.root)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.log_text = tk.Text(self.log_frame, height=8)
        self.log_text.pack(fill=tk.X, expand=True)

        self.last_progress_line_index = None

    def _setup_status_bar(self):
        """Persistent status bar: current experiment, active module, last action + spinner.
        Sits just above the log area. Updated via ui_manager.set_status(...)."""
        self._status_experiment = "No experiment"
        self._status_module = "—"
        self._status_action = "Ready"
        self._status_busy = False
        self._spinner_frames = ["◐", "◓", "◑", "◒"]
        self._spinner_i = 0

        bar = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar()
        self.spinner_var = tk.StringVar(value="•")
        tk.Label(bar, textvariable=self.spinner_var, width=2).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(bar, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=2)
        kill_btn = tk.Button(bar, text="Kill switch", fg="white", bg="#B00020",
                             activeforeground="white", activebackground="#8A0018",
                             command=self._on_kill_switch_clicked)
        kill_btn.pack(side=tk.RIGHT, padx=6, pady=1)
        add_tooltip(kill_btn, "Emergency stop: immediately kills every running process in every "
                    "tab (tracking, analysis, conversion), discarding any in-progress work, then "
                    "closes BuzzSuite and its terminal window.")
        self._render_status()

        # Expose a single entry point on the ui_manager for all tabs.
        self.ui_manager.set_status = self.set_status
        self.ui_manager.show_error = self.show_error

    def _render_status(self):
        self.status_var.set(
            f"Experiment: {self._status_experiment}    |    "
            f"Module: {self._status_module}    |    {self._status_action}")

    def _spin(self):
        if not self._status_busy:
            self.spinner_var.set("•")
            return
        self._spinner_i = (self._spinner_i + 1) % len(self._spinner_frames)
        self.spinner_var.set(self._spinner_frames[self._spinner_i])
        self.root.after(120, self._spin)

    def set_status(self, experiment=None, module=None, action=None, busy=None):
        """Update any subset of the status-bar fields. Main thread only."""
        if experiment is not None:
            self._status_experiment = experiment or "No experiment"
        if module is not None:
            self._status_module = module or "—"
        if action is not None:
            self._status_action = action
        if busy is not None and busy != self._status_busy:
            self._status_busy = bool(busy)
            if self._status_busy:
                self._spin()
            else:
                self.spinner_var.set("•")
        self._render_status()

    def show_error(self, title, message):
        """User-visible error dialog + log line (shared by tabs via ui_manager.show_error)."""
        from tkinter import messagebox
        self.log(f"{title}: {message}")
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass

    def log(self, message):
        """Log messages in the logging area. Every off-thread caller (Dashboard's background
        threads, Section 2's tracking queue thread, etc.) used to touch the log_text widget
        directly, which is unsafe -- Tk is not thread-safe and this could corrupt the widget or
        crash the app. Centralizing the re-dispatch here (rather than fixing every caller
        individually) makes it safe to call self.log(...) from anywhere."""
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, lambda m=message: self.log(m))
            return
        try:
            if "Progress" in message:
                # Determine the start of the last line
                last_line_index = self.log_text.index("end-1c linestart")
                # Delete the current text of the last line
                self.log_text.delete(last_line_index, 'end-1c')
                # Insert the new progress message at the start of the last line
                self.log_text.insert(last_line_index, message)

                self.last_progress_line_index = last_line_index
            else:
                self.log_text.insert(tk.END, message + "\n")
                self.last_progress_line_index = None

            self.log_text.see(tk.END)
            self.log_text.update_idletasks()  # Force the refresh of the UI
        except Exception:
            # The widget may already be destroyed (e.g. app closing while a background thread's
            # queued log call is still in flight) -- never let a log call crash the caller.
            pass

    def _load_last_experiment(self):
        if 'last_experiment' in self.config:
            from tkinter import messagebox
            last = self.config['last_experiment']
            try:
                # last_experiment is a JSON path for experiments made via the new flow; the old
                # load_experiment path does pickle.load and fails on JSON. Route by extension.
                if isinstance(last, str) and last.lower().endswith('.json'):
                    self.experiment_manager.load_experiment_from_json(last)
                else:
                    self.experiment_manager.load_experiment(last)
                self.ui_manager.update_ui_after_loading_experiment()
                self.ui_manager.set_status(experiment=self.experiment_manager.experiment_alias,
                                           action="Loaded last experiment")
            except Exception as exc:
                self.log(f"Error loading last experiment: {exc}")
                messagebox.showerror(
                    "Load last experiment",
                    f"Could not reopen your last experiment:\n{last}\n\n{exc}\n\n"
                    "It may have been moved, deleted, or the data drive isn't mounted. "
                    "Use the Intake Wizard to open or create one.")

    def on_closing(self):
        """Handle the window closing event."""
        try:
            self.video_manager.stop_and_release()
        except Exception:
            pass
        self.experiment_manager.stop_video_playback()
        self.root.destroy()

    def _on_kill_switch_clicked(self):
        """Emergency stop: confirm, then hand off to kill_switch.force_kill_everything(), which
        tears down this whole process tree (every tracking/analysis Pool, every ffmpeg conversion,
        the GUI itself) at once rather than trying to gracefully cancel each tab individually --
        see kill_switch.py's docstring for why. Does not return."""
        from tkinter import messagebox
        if not messagebox.askyesno(
                "Kill switch",
                "This immediately stops every running process (tracking, analysis, conversion) "
                "in every experiment, discards any in-progress work, and closes BuzzSuite and its "
                "terminal window.\n\nThis cannot be undone. Continue?",
                icon="warning", default="no"):
            return
        from kill_switch import force_kill_everything
        try:
            self.root.destroy()
        except Exception:
            pass
        force_kill_everything()
    #def _maximize_window(self):
    #    """Maximize the window to fit the screen."""
    #    self.root.state('zoomed')  # This will maximize the window for Windows and macOS