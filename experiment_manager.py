import pickle
import os
import shutil
import tempfile
import yaml
from tkinter import filedialog
from tkinter import simpledialog
import sys
import time
import json
import random
import multiprocessing
from PIL import Image, ImageTk
import cv2
from threading import Thread

from buzzwatch_data_analysis.experiment_analysis import buzzwatch_experiment_analysis
from buzzwatch_data_analysis.misc_functions import create_folder
from buzzwatch_data_analysis.single_video_analysis import single_video_analysis
from tracking_overhaul import apply_tracking_overhaul
from logger import MultiLogger


class ExperimentManager:
    def __init__(self, log_func, config_manager=None):
        self.log = log_func
        self.experiment = None
        self.folder_analysis = None
        self.folder_videos = None
        self.experiment_alias = None
        self.settings = None
        # When True, run_tracking_analysis injects the adopted tracking overhaul (WS1+WS3+WS4+WS5)
        # into the in-memory settings for the run only (never written back to the YAML). Toggled
        # from the Batch Processing tab; honored by both the batch and single-video run paths.
        self.use_tracking_overhaul = True
        # Single source of truth (Finding 3): when the app passes its ConfigManager, every
        # GUI-process ExperimentManager (main app + the dashboard's helper instance) shares the
        # SAME in-memory config dict AND writes to the SAME file, so recent-experiments / settings
        # never drift between them. Worker processes and the CLI pass no manager and keep their own
        # reloaded copy at the default path.
        self.config_manager = config_manager
        if config_manager is not None:
            self.config_path = config_manager.config_path
            self.config = config_manager.config
        else:
            self.config_path = os.path.join(os.path.dirname(__file__), "config.json")
            self.config = self.load_config()
        self.settings_file = None
        self.cap = None
        self.total_frames = 0
        self.is_playing = False
        self.tracking_image_label = None
        self.video_label = None
        self.scrollbar = None
        self.mosquito_tracks = None

    def load_settings(self, settings_file):
        if not os.path.isfile(settings_file):
            error_msg = f"Settings file not found: {settings_file}"
            self.log(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Retry logic for potential file locking issues on Windows with multiprocessing
        # Increase retries significantly when multiple workers read the same file
        max_retries = 30
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                with open(settings_file, 'r') as file:
                    self.settings = yaml.safe_load(file)
                self.settings_file = settings_file
                self.log(f"Settings loaded from {settings_file}")
                return
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Use exponential backoff with randomized jitter to prevent thundering herd
                    # when multiple workers retry at same time
                    base_wait = 0.1 * (1.5 ** attempt)
                    base_wait = min(base_wait, 3.0)
                    # Add jitter: ±25% of base wait time
                    jitter = random.uniform(0.75, 1.25)
                    wait_time = base_wait * jitter
                    time.sleep(wait_time)
                    continue
                else:
                    # After all retries, raise with detailed error info
                    error_msg = f"Error loading settings from {settings_file} after {max_retries} attempts: {str(last_exception)}"
                    self.log(f"ERROR: {error_msg}")
                    raise ValueError(error_msg) from last_exception

    def load_experiment(self, exp_file=None,update_ui_func=None):
        if exp_file is None:
            exp_file = filedialog.askopenfilename(title="Select Experiment File", filetypes=[("Pickle files", "*.pkl")])
        if not exp_file:
            self.log("No experiment file selected.")
            return

        with open(exp_file, 'rb') as f:
            self.experiment = pickle.load(f)
            self.folder_analysis = getattr(self.experiment, 'folder_analysis', None)
            self.folder_videos = getattr(self.experiment, 'folder_videos', None)
            self.experiment_alias = self.experiment.experiment_alias
            self.settings_file = self.experiment.settings_file
            with open(self.settings_file, 'r') as file:
                self.settings = yaml.safe_load(file)
            self.experiment.settings = self.settings
            self.experiment.log = self.log
            #self.settings = self.experiment.settings
            
        
        self.config['last_experiment'] = exp_file
        self.save_config(self.config)
        self.record_recent_experiment(exp_file)
        try:
            self.video_files = sorted([os.path.join(self.folder_videos, f) for f in os.listdir(self.folder_videos)
                                       if f.endswith('.mp4') and not f.startswith('.')])
            if not self.video_files:
                self.log("No .mp4 files found in the selected directory.")
                return
        except Exception:
            self.log("video_folder found.")

        if update_ui_func:
            update_ui_func()

    def load_experiment_from_json(self, json_path, update_ui_func=None):
        """Load experiment details from a JSON file."""
        if not os.path.exists(json_path):
            error_msg = f"Experiment JSON file not found: {json_path}"
            self.log(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)

        with open(json_path, 'r') as json_file:
            experiment_details = json.load(json_file)

        # Load all parts of the experiment details
        self.root_folder_videos = experiment_details.get('root_folder_videos')
        self.root_folder_analysis = experiment_details.get('root_folder_analysis')
        self.experiment_name = experiment_details.get('experiment_name')
        self.cage_name = experiment_details.get('cage_name')
        self.batch_name = experiment_details.get('batch_name')
        self.settings_file = experiment_details.get('settings_file')  # Reference the copied file path

        # Reconstruct full paths using loaded data
        self.folder_videos = os.path.join(self.root_folder_videos, self.experiment_name, self.cage_name, self.batch_name)
        self.folder_analysis = os.path.join(self.root_folder_analysis, self.experiment_name, self.cage_name, self.batch_name)
        # An explicit folder_videos (e.g. intake wizard: raw videos outside the Recording tree)
        # overrides the reconstructed path so tracking finds the real .mp4 files.
        if experiment_details.get('folder_videos'):
            self.folder_videos = experiment_details['folder_videos']

        # Load settings from the copied file (skip if already pre-loaded)
        if not os.path.isfile(self.settings_file):
            error_msg = f"Settings file not found: {self.settings_file}"
            self.log(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Skip loading settings if they're already pre-loaded in worker process
        if self.settings is None:
            self.load_settings(self.settings_file)
        
        # Validate settings were loaded
        if self.settings is None:
            error_msg = f"Failed to load settings from {self.settings_file}"
            self.log(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        # Create the experiment object with loaded settings
        self.experiment_alias = experiment_details.get('experiment_alias', f"exp_{self.experiment_name}_{self.cage_name}_{self.batch_name}")
        self.experiment = buzzwatch_experiment_analysis(
            folder_analysis=self.folder_analysis,
            folder_videos=self.folder_videos,
            experiment_alias=self.experiment_alias,
            settings=self.settings,
            settings_file=self.settings_file,
            log_func=self.log,
            debug_mode=False
        )

        # Only save config update in main process (not in worker processes). The stated intent
        # here always existed, but the check that was supposed to enforce it -- `hasattr(self.config,
        # '__getitem__')` -- is true for any plain dict, main process or worker alike, so it was a
        # no-op: every multiprocessing.Pool worker (_run_batch_analysis constructs one
        # ExperimentManager PER SEGMENT, not once per worker process) was calling save_config +
        # record_recent_experiment on every single video it tracked. Observed live on a 30-worker,
        # 213-segment run: a wall of "Warning: Could not save config... [WinError 5] Access is
        # denied" -- up to 30 processes simultaneously racing os.replace() on the same config.json,
        # each retry costing up to ~1s of sleep backoff, for a write whose result (this experiment is
        # already the "last"/"recent" one -- the GUI/CLI main process set that before dispatching the
        # pool) no worker actually needed to make. multiprocessing.parent_process() is None only in
        # the original process that never forked/spawned from anything -- reliably distinguishes
        # "am I main" from "am I a spawned worker" (unlike the dict check above).
        try:
            if multiprocessing.parent_process() is None:
                self.config['last_experiment'] = json_path
                self.save_config(self.config)
                self.record_recent_experiment(json_path)
        except Exception as e:
            # Silently skip config save if it fails (worker processes may not have write access)
            pass
        
        self.log(f"Experiment details loaded from {json_path}")

        # Call the UI update function if provided to refresh relevant UI components
        if update_ui_func:
            update_ui_func()


    def initialize_new_experiment(self, root_folder_videos=None, root_folder_analysis=None,
                                  experiment_name=None, cage_name=None, batch_name=None,settings_file=None,
                                  update_ui_func=None, folder_videos=None):
        if not root_folder_videos or not root_folder_analysis:
            self.log("Root folders for videos and analysis must be provided.")
            return
        
        if not experiment_name or not cage_name or not batch_name:
            self.log("Experiment name, cage name, and batch name must be provided.")
            return
        
        if not settings_file or not os.path.isfile(settings_file):
            self.log("A valid settings file must be provided.")
            return

        # Raw videos usually live under the constructed Recording tree, but the intake wizard may
        # point at an existing source folder outside it — honor an explicit folder_videos so
        # tracking finds the real .mp4 files (persisted in the JSON below for later loads).
        folder_videos_path = folder_videos or os.path.join(root_folder_videos, experiment_name, cage_name, batch_name)
        folder_analysis_path = os.path.join(root_folder_analysis, experiment_name, cage_name, batch_name)

        create_folder(folder_analysis_path)
        self.ensure_experiment_subfolders(folder_analysis_path)


        settings_file_name = os.path.basename(settings_file)
        settings_file_destination = os.path.join(folder_analysis_path, settings_file_name)
        shutil.copyfile(settings_file, settings_file_destination)

        with open(settings_file_destination, 'r') as file:
            self.settings = yaml.safe_load(file)

        # The shared template ships with stale, non-empty border coordinates from an unrelated
        # camera setup (cage_border_points etc.). A brand-new experiment must start with no
        # borders drawn at all, so the first "Show background with borders" shows a plain
        # background instead of silently rendering someone else's old rectangle as real. Loading
        # an EXISTING experiment never goes through this function, so previously-drawn borders
        # are untouched.
        for key in ('cage_border_points', 'sugar_border_points', 'control_border_points',
                    'square_3_border_points', 'square_4_border_points', 'center_border_points'):
            self.settings[key] = []
        with open(settings_file_destination, 'w') as file:
            yaml.safe_dump(self.settings, file)

        experiment_alias = f"{cage_name}_{batch_name}"

        self.folder_videos = folder_videos_path
        self.folder_analysis = folder_analysis_path
        self.experiment_alias = experiment_alias

        self.experiment = buzzwatch_experiment_analysis(
            self.folder_analysis,
            self.folder_videos,
            self.experiment_alias,
            self.settings,
            settings_file_destination,
            log_func=self.log,
            debug_mode=False
        )

        # Save experiment details to JSON
        experiment_details = {
            'root_folder_videos': root_folder_videos,
            'root_folder_analysis': root_folder_analysis,
            'experiment_name': experiment_name,
            'cage_name': cage_name,
            'batch_name': batch_name,
            'experiment_alias': experiment_alias,  # persist so load reads it back verbatim (Finding 4)
            'folder_videos': folder_videos_path,  # explicit video location (honored on load)
            'settings_file': settings_file_destination  # Use the copied settings file
        }
        json_experiment_path = os.path.join(self.folder_analysis, f"experiment_{self.experiment_alias}.json")
        self.save_experiment_to_json(json_experiment_path, experiment_details)

        self.config['last_experiment'] = json_experiment_path
        self.save_config(self.config)
        self.record_recent_experiment(json_experiment_path)
        self.log(f"Experiment details saved to {json_experiment_path}")

        if update_ui_func:
            update_ui_func()

        self.log("New experiment initialized successfully.")


    def save_experiment_to_json(self, json_path, experiment_details):
        """Save experiment details to a JSON file."""
        with open(json_path, 'w') as json_file:
            json.dump(experiment_details, json_file, indent=4)
        self.log(f"Experiment details saved to {json_path}")

    # Canonical analysis subfolder layout (see CLAUDE.md data-folder structure). The tracker
    # otherwise creates several of these lazily on first use; making them all upfront gives a
    # complete, predictable experiment tree the moment an experiment is created/loaded.
    ANALYSIS_SUBFOLDERS = [
        "final_tracking_data", "temp_data", "individual_images", "images_mortality",
        "tracking_resting", "tracking_moving", "log_analysis", "test_tracking",
        "plots", os.path.join("plots", "activity_plots"), os.path.join("plots", "custom_zones"),
        os.path.join("plots", "buzzswarm"),
    ]

    def ensure_experiment_subfolders(self, folder_analysis=None):
        """Create all standard analysis subfolders under the experiment (idempotent)."""
        base = folder_analysis or self.folder_analysis
        if not base:
            return
        os.makedirs(base, exist_ok=True)
        for sub in self.ANALYSIS_SUBFOLDERS:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    def detect_completed_analyses(self, folder_analysis=None):
        """Report which analyses already have output for an experiment, so the UI can avoid
        re-running finished work. Returns {'activity','swarming','phonotaxis': bool}."""
        base = folder_analysis or self.folder_analysis
        done = {'activity': False, 'swarming': False, 'phonotaxis': False}
        if not base or not os.path.isdir(base):
            return done

        def _has_content(*parts):
            d = os.path.join(base, *parts)
            return os.path.isdir(d) and any(not f.startswith('.') for f in os.listdir(d))

        done['activity'] = _has_content('final_tracking_data')
        done['swarming'] = _has_content('plots', 'buzzswarm')
        done['phonotaxis'] = _has_content('plots', 'custom_zones')
        return done


    ## Sample functions to go inside ExperimentManager class
    def get_images_from_video(self, force_rerun):
        self.log("Extracting images from video...")
        self.experiment.extract_images_v2(force_rerun)
        
        # Update image listbox
        #self.update_image_listbox()
        self.log("Images extracted and listed.")

    def get_background_from_images(self, force_rerun):
        self.log("Computing background from images...")
        self.experiment.extract_average_background(force_rerun)
        #self.update_median_image_listbox()
        self.log("Background images computed and saved.")

    def draw_cage_borders(self):
        self.log("Drawing cage borders...")
        self.experiment.user_input_draw_borders_cage(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Cage borders drawn and saved.")

    def draw_center_corners(self):
        self.log("Adjusting center corners and regenerating side regions...")
        scale_factor = simpledialog.askfloat(
            "Adjust Center Square",
            "Enter center size scale (1.0 keeps the same size, <1 shrinks, >1 expands):",
            minvalue=0.1,
            maxvalue=2.0,
            initialvalue=0.85,
            parent=None,
        )
        if scale_factor is None:
            self.log("Center adjustment cancelled.")
            return

        self.experiment.resize_center_region(scale_factor=scale_factor)
        self.experiment.plot_all_borders()
        self.log(f"Center corners updated with scale {scale_factor:.2f} and side regions regenerated.")

    def reset_drawn_borders(self):
        self.log("Resetting drawn side/center borders...")
        self.experiment.reset_drawn_borders(keep_cage=True)
        self.experiment.plot_all_borders()
        self.log("Drawn side/center borders reset.")

    def draw_speaker_side_borders(self):
        self.log("Auto-generating side regions (speaker on side 1)...")
        self.experiment.user_input_draw_speaker_side_region(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Speaker-side regions auto-generated and saved.")

    # Legacy alias for backward compatibility with sugar feeder naming.
    draw_sugar_feeder_borders = draw_speaker_side_borders

    def draw_control_borders(self):
        self.log("Auto-generating side regions (speaker on side 2)...")
        self.experiment.user_input_draw_control_squares(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Speaker-side regions auto-generated and saved.")

    def draw_square_3(self):
        self.log("Auto-generating side regions (speaker on side 3)...")
        self.experiment.user_input_draw_control_squares_3(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Speaker-side regions auto-generated and saved.")

    def draw_square_4(self):
        self.log("Auto-generating side regions (speaker on side 4)...")
        self.experiment.user_input_draw_control_squares_4(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Speaker-side regions auto-generated and saved.")

    def update_border_image(self):
        self.log("Udpating backgrund with borders")
        self.experiment.plot_all_borders()

    def run_tracking_analysis(self,video_name):
        # Inject (or strip) the tracking overhaul into the in-memory settings just before the run,
        # based on the GUI toggle. apply_tracking_overhaul deep-copies, so the loaded settings/YAML
        # stay clean and toggling is stateless across runs. single_video_analysis reads
        # experiment.settings by reference, so updating it here is honored by the analysis.
        overhauled = apply_tracking_overhaul(self.settings, self.use_tracking_overhaul)
        self.settings = overhauled
        if self.experiment is not None:
            self.experiment.settings = overhauled
        self.log(f"Tracking overhaul {'ENABLED' if self.use_tracking_overhaul else 'disabled'} for {video_name}")
        self.experiment.run_single_video_analysis(video_name,debug_mode=1)

    def get_video_path(self, video_name):
        # Implement this method based on your application structure
        return os.path.join(self.folder_videos, video_name+".mp4")

    def detect_and_store_fps(self, video_name, persist=True):
        if not video_name:
            return 25.0

        if video_name.lower().endswith('.mp4'):
            video_path = os.path.join(self.folder_videos, video_name)
        else:
            video_path = self.get_video_path(video_name)
        fps = 25.0

        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                detected_fps = cap.get(cv2.CAP_PROP_FPS)
                if detected_fps and detected_fps > 0:
                    fps = float(detected_fps)
            cap.release()
        except Exception:
            fps = 25.0

        if self.settings is None and self.settings_file:
            try:
                self.load_settings(self.settings_file)
            except Exception:
                self.settings = {}

        if self.settings is None:
            self.settings = {}

        self.settings['fps'] = fps
        if self.experiment is not None:
            try:
                self.experiment.settings['fps'] = fps
            except Exception:
                pass

        # In batch the same settings file is shared by ~N workers; each writing it is a write storm
        # (lock contention + last-writer clobber) and the on-disk fps gets overwritten anyway. The
        # in-memory fps above is what the run actually uses, so batch callers pass persist=False.
        if persist and self.settings_file and os.path.isfile(self.settings_file):
            try:
                with open(self.settings_file, 'w') as file:
                    yaml.dump(self.settings, file)
            except Exception:
                pass

        return fps


    def load_tracking_data(self, tracking_file):
        tracking_file_path = os.path.join(self.folder_analysis, "final_tracking_data", tracking_file)
        if not os.path.exists(tracking_file_path):
            self.log(f"Tracking file {tracking_file_path} does not exist.")
            return
        with open(tracking_file_path, 'rb') as f:
            self.mosquito_tracks = pickle.load(f)


    def stop_video_playback(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None
    def load_config(self):
        if os.path.exists(self.config_path):
            # Retry logic for file locking issues
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    with open(self.config_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Only parse if file is not empty
                            return json.loads(content)
                except (json.JSONDecodeError, IOError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        # Return empty dict if file is corrupted or inaccessible after retries
                        print(f"Warning: Could not load config from {self.config_path} after {max_retries} attempts: {e}. Using empty config.")
                        return {}
        return {}

    def save_config(self, config):
        # Keep the shared ConfigManager's reference pointing at the persisted dict so the app UI
        # (which reads config_manager.config) always reflects live state — no drift (Finding 3).
        if self.config_manager is not None:
            self.config_manager.config = config
        try:
            # Retry logic for file locking issues
            max_retries = 5
            for attempt in range(max_retries):
                temp_path = None
                try:
                    # Write to temporary file first, then rename (atomic operation)
                    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.config_path) or ".")
                    with os.fdopen(temp_fd, 'w') as f:
                        json.dump(config, f)
                    # os.replace, not shutil.move: shutil.move's os.rename fails on Windows when
                    # the destination already exists (which it always does after the first save),
                    # so it silently fell back to a non-atomic copy+delete -- exactly the window
                    # where two concurrent ExperimentManager instances (e.g. two Dashboard jobs'
                    # record_recent_experiment calls) could interleave writes and corrupt
                    # config.json (observed: a stray trailing '}' -> JSONDecodeError on launch).
                    # os.replace is atomic on both Windows (MOVEFILE_REPLACE_EXISTING) and POSIX.
                    os.replace(temp_path, self.config_path)
                    return  # Success
                except (IOError, OSError) as e:
                    # os.replace can fail on Windows with a sharing violation if config.json is
                    # momentarily open elsewhere (e.g. a concurrent save_config()/load_config()
                    # from another Dashboard job thread) -- that failure leaves temp_path behind
                    # since mkstemp already created it. Without this cleanup, every retried/failed
                    # attempt orphaned one tmp<random> file in this directory forever (observed:
                    # 1000+ accumulated stray files, each a copy of a config.json write attempt).
                    if temp_path is not None:
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        raise
        except Exception as e:
            print(f"Warning: Could not save config to {self.config_path}: {e}")

    # --- Recent experiments (Step 7 welcome panel) --------------------------------------
    RECENT_LIMIT = 10

    def record_recent_experiment(self, path, alias=None):
        """Push ``path`` (a .json or .pkl experiment file) to the front of
        ``config['recent_experiments']`` (most-recent-first, deduped, capped at RECENT_LIMIT).
        Best-effort: never raises into the caller (worker processes may not have write access)."""
        try:
            if not path:
                return
            entry = {
                'path': path,
                'alias': alias or getattr(self, 'experiment_alias', None) or os.path.basename(path),
                'ts': time.time(),
            }
            recent = self.config.get('recent_experiments', [])
            if not isinstance(recent, list):
                recent = []
            # Drop any existing entry for the same path, then prepend.
            recent = [r for r in recent if isinstance(r, dict) and r.get('path') != path]
            recent.insert(0, entry)
            self.config['recent_experiments'] = recent[:self.RECENT_LIMIT]
            self.save_config(self.config)
        except Exception:
            pass

    def get_recent_experiments(self):
        """Return the list of recent-experiment dicts (most recent first), filtered to
        those whose file still exists. Never raises."""
        try:
            recent = self.config.get('recent_experiments', [])
            if not isinstance(recent, list):
                return []
            return [r for r in recent
                    if isinstance(r, dict) and r.get('path') and os.path.exists(r['path'])]
        except Exception:
            return []