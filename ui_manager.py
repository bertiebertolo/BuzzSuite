from common_imports import *

from video_tab_manager import VideoTabManager
from batch_processing_tab_manager import BatchProcessingTabManager
from setup_tab_manager import SetupTabManager
from buzzswarm_tab_manager import BuzzswarmTabManager
from buzzphono_tab_manager import BuzzphonoTabManager
from h264_tab_manager import H264TabManager
from intake_wizard import IntakeWizardTabManager
from experiment_dashboard import ExperimentDashboard
from activity_plot_manager import ActivityPlotManager
from state_manager import StateManager

class UIManager:
    def __init__(self, root, experiment_manager, video_manager, image_manager, log_func, state_manager,plot_manager):
        self.root = root
        self.experiment_manager = experiment_manager
        self.video_manager = video_manager
        self.image_manager = image_manager
        self.log = log_func
        self.state_manager = state_manager
        self.plot_manager = plot_manager

        self.notebook = None
        self.setup_managers()

    # Default no-op status/error hooks; the app replaces these with the real status-bar
    # implementation in VideoAnalyzerApp._setup_status_bar. Defined here so tab managers can
    # always call ui_manager.set_status(...) / ui_manager.show_error(...) safely.
    def set_status(self, experiment=None, module=None, action=None, busy=None):
        pass

    def show_error(self, title, message):
        from tkinter import messagebox
        self.log(f"{title}: {message}")
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass

    def setup_managers(self):
        # All tab managers are constructed unconditionally (cheap, no I/O); init_tabs() then builds
        # every section into the notebook (no module choice — all sections are always present).
        #
        # ComparisonTabManager is deliberately NOT imported or constructed here. It has no tab and
        # no callers in the current three-section layout, but importing it pulls the whole
        # statsmodels + sklearn + statistical_analysis_manager + glmm_analysis_manager chain
        # (~250-300 ms, ~20% of cold start) into every launch for nothing. The module is kept on
        # disk: to reinstate a Compare surface, import it lazily inside init_tabs where the tab is
        # actually built, not at module scope here.
        # AnalysisTabManager is deliberately NOT imported or constructed here (GUI_FIX_PLAN.md
        # decision D4). It builds no tab and has no other caller (grep-confirmed) -- it exists
        # purely as export-seed code for the planned Export tab (see its own docstring / CLAUDE.md
        # Future work). The module is kept on disk; import it lazily where an Export tab actually
        # needs it, not at module scope here.
        self.video_tab_manager = VideoTabManager(self.root, self, self.state_manager)
        self.batch_processing_tab_manager = BatchProcessingTabManager(self.root, self, self.state_manager)
        self.setup_tab_manager = SetupTabManager(self.root, self, self.state_manager)
        self.buzzswarm_tab_manager = BuzzswarmTabManager(self.root, self, self.state_manager)
        self.buzzphono_tab_manager = BuzzphonoTabManager(self.root, self, self.state_manager)
        self.h264_tab_manager = H264TabManager(self.root, self, self.state_manager)
        # Always-present workflow tabs (independent of module selection).
        self.intake_wizard = IntakeWizardTabManager(self.root, self, self.state_manager)
        self.experiment_dashboard = ExperimentDashboard(self.root, self, self.state_manager)
        # Activity plotting (the "Activity (in-cage)" 3·Plotting sub-tab); also absorbs the former
        # standalone Compare-experiments tab. AnalysisTabManager (export-seed code, see above) is
        # no longer constructed here at all.
        self.activity_plot_manager = ActivityPlotManager(self.root, self, self.state_manager)

    def init_tabs(self):
        # No module choice anymore — every experiment walks the same three linear sections.
        # active_modules is kept (all three, always) so any remaining 'x' in active_modules
        # check elsewhere still passes.
        self.active_modules = {'activity', 'swarming', 'phonotaxis'}

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        self.initialize_tabs()

        # 1 · Setup — nested sub-notebook reusing both builders verbatim (no method bodies moved):
        # "Experiment & Video" is the Intake Wizard (Steps A-D); "Background & Cage Border" is the
        # existing Setup tab builder (cage border only — speaker zone lives in 3 · Plotting).
        self.intake_wizard.init_intake_tab(self.setup_subtabs['intake'])
        self.setup_tab_manager.init_setup_tab(self.setup_subtabs['background'])

        # 2 · Analysis — nested sub-notebook: track one experiment (scope/queue + raw video
        # playback, unchanged shape) vs. run several experiments at once (the relocated
        # Experiment Dashboard).
        self.batch_processing_tab_manager.init_tracking_tab(self.analysis_subtabs['single'])
        self.experiment_dashboard.init_dashboard_tab(self.analysis_subtabs['multi'])

        # 3 · Plotting — nested sub-notebook: one plot type per sub-tab.
        self.activity_plot_manager.init_activity_tab(self.plots_subtabs['activity'])
        self.buzzphono_tab_manager.init_buzzphono_tab(self.plots_subtabs['phonotaxis'])
        self.buzzswarm_tab_manager.init_buzzswarm_tab(self.plots_subtabs['swarming'])

        # H264 converter is always available.
        self.h264_tab_manager.init_h264_tab(self.tabs['h264_tab'])

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self.setup_notebook.bind("<<NotebookTabChanged>>", self._on_setup_subtab_change)
        self.analysis_notebook.bind("<<NotebookTabChanged>>", self._on_analysis_subtab_change)
        self.plots_notebook.bind("<<NotebookTabChanged>>", self._on_plots_subtab_change)

    def initialize_tabs(self):
        self.tabs = {}

        # 1 · Setup (nested sub-notebook)
        setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(setup_frame, text="1 · Setup")
        self.tabs['setup_section'] = setup_frame
        self.setup_notebook = ttk.Notebook(setup_frame)
        self.setup_notebook.pack(fill='both', expand=True)
        self.setup_subtabs = {}
        for key, title in [('intake', 'Experiment & Video'),
                           ('background', 'Background & Cage Border')]:
            f = ttk.Frame(self.setup_notebook)
            self.setup_notebook.add(f, text=title)
            self.setup_subtabs[key] = f

        # 2 · Analysis (nested sub-notebook)
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="2 · Analysis")
        self.tabs['analysis_section'] = analysis_frame
        self.analysis_notebook = ttk.Notebook(analysis_frame)
        self.analysis_notebook.pack(fill='both', expand=True)
        self.analysis_subtabs = {}
        for key, title in [('single', 'Single experiment'),
                           ('multi', 'Multiple experiments')]:
            f = ttk.Frame(self.analysis_notebook)
            self.analysis_notebook.add(f, text=title)
            self.analysis_subtabs[key] = f

        # 3 · Plotting (nested sub-notebook)
        plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(plots_frame, text="3 · Plotting")
        self.tabs['plots_section'] = plots_frame
        self.plots_notebook = ttk.Notebook(plots_frame)
        self.plots_notebook.pack(fill='both', expand=True)
        self.plots_subtabs = {}
        for key, title in [('activity', 'Activity (in-cage)'),
                           ('phonotaxis', 'Speaker custom zone (Phonotaxis)'),
                           ('swarming', 'Swarming (Aggregation)')]:
            f = ttk.Frame(self.plots_notebook)
            self.plots_notebook.add(f, text=title)
            self.plots_subtabs[key] = f

        # H264 converter is always available.
        for tab_key, tab_title in [('h264_tab', 'H264 → MP4 Converter')]:
            self.tabs[tab_key] = ttk.Frame(self.notebook)
            self.notebook.add(self.tabs[tab_key], text=tab_title)

    def update_ui_after_loading_experiment(self):
        # No module gating anymore — every experiment has all three sections, so always refresh
        # every tab's view of the loaded experiment.
        alias = getattr(self.experiment_manager, 'experiment_alias', None)
        self.set_status(experiment=alias, action="Experiment loaded")
        self.handle_errors(self.setup_tab_manager.refresh_experiment, "Error refreshing Setup tab")
        self.handle_errors(self.video_tab_manager.set_initial_video_list, "Error loading .mp4 videos")
        self.handle_errors(self.batch_processing_tab_manager.update_segment_listbox, "Error loading segment list")
        self.handle_errors(self.batch_processing_tab_manager.update_range_options, "Error loading date range")
        self.handle_errors(self.buzzswarm_tab_manager.refresh_experiment, "Error refreshing BuzzSwarm tab")
        self.handle_errors(self.buzzphono_tab_manager.refresh_experiment, "Error refreshing BuzzPhono tab")

    def handle_errors(self, func, error_message):
        try:
            func()
        except Exception as e:
            self.log(f"{error_message}: {e}")

    # NOTE (GUI_FIX_PLAN.md decision D4): create_logging_area/log/update_progress_line/
    # add_log_message used to live here but were dead code -- __init__ sets `self.log = log_func`
    # (VideoAnalyzerApp.log, which owns the real log_text widget), which as an instance attribute
    # permanently shadows the class method of the same name, so this class's own `log` (and the two
    # helpers only it called) could never run, and create_logging_area had zero callers anywhere.
    # Removed rather than fixed in place since VideoAnalyzerApp.log is the one real implementation.

    def on_tab_change(self, event):
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")
        self.log(f"Switched to tab: {tab_text}")
        # Keep the status bar's "Module" field in sync with the active tab.
        self.set_status(module=tab_text)

        # Stop ongoing video playback
        self.video_manager.is_playing = False
        if self.video_manager.cap:
            self.video_manager.cap.release()  # Properly release any loaded video

        # Reset video display: If you switch to a non-video tab, clear the label
        if self.video_manager.label:
            self.video_manager.label.config(image='')  # Clear any displayed image

        # Assign the appropriate display label/scrollbar (or refresh) for the active section.
        # "1 · Setup", "2 · Analysis" and "3 · Plotting" are all nested sub-notebooks, so their
        # wiring is decided by whichever sub-tab is currently showing (also bound to the
        # sub-notebooks' own <<NotebookTabChanged>> below, so switching sub-tabs without changing
        # the top-level tab still updates the display label).
        if tab_text == '1 · Setup':
            self._on_setup_subtab_change()
        elif tab_text == '2 · Analysis':
            self._on_analysis_subtab_change()
        elif tab_text == '3 · Plotting':
            self._on_plots_subtab_change()

    def _on_setup_subtab_change(self, event=None):
        sub = self.setup_notebook.select()
        if not sub:
            return
        sub_text = self.setup_notebook.tab(sub, "text")
        if sub_text == 'Background & Cage Border':
            self.handle_errors(self.setup_tab_manager.refresh_experiment, "Error refreshing Setup tab")

    def _on_analysis_subtab_change(self, event=None):
        sub = self.analysis_notebook.select()
        if not sub:
            return
        sub_text = self.analysis_notebook.tab(sub, "text")
        # Raw-video playback widgets (build_playback_section) live only in "Single experiment";
        # "Multiple experiments" (the dashboard) has no video display to wire.
        if sub_text == 'Single experiment':
            self.video_manager.set_display_label(self.video_tab_manager.video_label)
            self.video_manager.set_scrollbar(self.video_tab_manager.video_scrollbar)

    def _on_plots_subtab_change(self, event=None):
        # No per-sub-tab wiring needed anymore: the Activity plotting sub-tab is now the
        # picker-driven ActivityPlotManager (no embedded video playback — that lives only in
        # 2 · Analysis → Single experiment). Kept as a bound hook for future sub-tab needs.
        return
