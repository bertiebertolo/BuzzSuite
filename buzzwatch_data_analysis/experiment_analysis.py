########################## IMPORT ALL NECESSARY PACKAGES ####################

# import numpy as np
# import cv2
import os
import yaml
from os import listdir
from os.path import isfile, join
from buzzwatch_data_analysis.misc_functions import *
from buzzwatch_data_analysis.single_video_analysis import *
import pickle
import sys
import numpy as np
import re
import warnings
from logger import MultiLogger

# Suppress expected NumPy warnings from empty data windows
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

# Plot configuration constants
ROLLING_WINDOW_SECONDS = int(os.environ.get('BUZZSUITE_ROLLING_WINDOW_SECONDS', '1200'))  # Rolling window in seconds
PLOT_DPI_TRAJECTORY = int(os.environ.get('BUZZSUITE_PLOT_DPI_TRAJECTORY', '100'))  # DPI for trajectory plots
PLOT_FIGHEIGHT_TRAJECTORY = int(os.environ.get('BUZZSUITE_PLOT_FIGHEIGHT_TRAJECTORY', '5'))  # Figure height for trajectory plots
PLOT_FIGWIDTH_TRAJECTORY = int(os.environ.get('BUZZSUITE_PLOT_FIGWIDTH_TRAJECTORY', '30'))  # Figure width for trajectory plots

# Add the custom constructor to PyYAML Loader

# import matplotlib.pyplot as plt
# import time
# import pandas as pd
# #from functions_Tracking_V2 import*
# import sys
# from scipy.spatial import distance as dist
# import pickle
# from resting_obj_tracker import resting_tracker
# from moving_obj_tracker import moving_tracker
# import scipy.stats as scp
# import scipy.sparse.csgraph as graph
# import datetime as dt
# from path_dict import PathDict


##################################################################################################################################
########################## Class that manages all videos from a given experiment ####################
class buzzwatch_experiment_analysis:
    # Class variables

    def __init__(self,folder_analysis,folder_videos,experiment_alias,settings,settings_file,log_func=None,debug_mode=False):

        self.experiment_alias = experiment_alias
        self.folder_videos = folder_videos
        self.folder_analysis = folder_analysis
        #print(self.folder_analysis)
        self.background_path = folder_analysis+"/background_"+experiment_alias+".png"

        # Always defined (even empty) so callers that iterate list_video_name/list_videos_files
        # never hit an AttributeError -- folder_videos may not exist yet (e.g. an experiment just
        # created but not yet had its video converted/moved in).
        self.list_video_name = []
        self.list_videos_files = []
        try:
            files_video = [f for f in listdir(folder_videos) if isfile(join(folder_videos, f)) and f.endswith(".mp4")
                           and not f.startswith(".")]   # skip macOS "._" companions (exFAT drives)
            files_video.sort()

            self.list_video_name = [os.path.splitext(video_mp4)[0] for video_mp4 in files_video]
            self.list_videos_files = files_video
        except Exception:
            print("Warning no videos files in "+ folder_videos)
        self.folder_test_tracking = folder_analysis+"/test_tracking"
        create_folder(self.folder_test_tracking)
        self.folder_final = folder_analysis+"/final_tracking_data"
        create_folder(self.folder_final)
        self.folder_temp_data = folder_analysis+"/temp_data"
        create_folder(self.folder_temp_data)

        # Load settings: prefer passed settings, reload from file if needed
        self.settings_file = settings_file
        if settings is not None:
            self.settings = settings
        elif os.path.isfile(settings_file):
            with open(settings_file, 'r') as file:
                self.settings = yaml.safe_load(file)
        else:
            raise ValueError(f"No settings provided and settings file not found: {settings_file}")
        
        self.control_border_points = self.settings["control_border_points"]
        self.cage_border_points = self.settings["cage_border_points"]
        self.speaker_side_border_points = self.settings["sugar_border_points"]
        self.sugar_border_points = self.speaker_side_border_points
        if debug_mode:
            # Save the experiment_object
            exp_object_path = self.folder_temp_data+"/temp_data_"+self.experiment_alias+".pkl"
            with open(exp_object_path, 'wb') as f:
                pickle.dump(self,f)

        self.log = log_func  # Store the logging function
########################## Compute background or not ####################
    def add_background(self,force_to_redo):
        if force_to_redo == 1 or os.path.exists(self.background_path)==0:
            background= self.get_background()
            cv2.imwrite(self.background_path, background)

########################## Background from median frame of all videos ####################
    def get_background(self):
        print("Extracting the background image")
        images_to_av = []
        for k,video_file in enumerate(self.list_videos_files):
            progress_bar(k, len(self.list_videos_files), bar_length=20)
            if video_file.endswith('.mp4'):
                #start = video_file.find('Cage')
                video_path = self.folder_videos+video_file#[start::]
                print(video_path)
                cap = cv2.VideoCapture(video_path)
                suc,frame = cap.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                images_to_av.append(frame_gray)
                cap.release()
        median_frame = np.median(images_to_av, axis=0).astype(np.uint8)
        print("Background image saved")
        return median_frame

########################## Background from median frame of all videos ####################
    def run_video_analysis_test(self,video_idx,step_to_force_analyze):
        """ 
        This function analyzes a videos for testing the tracking performance
        before running everything on a cluster or workstation
        """
        video_name = os.path.splitext(self.list_videos_files[video_idx])[0]
        video_object_path = self.folder_temp_data+"/temp_data_"+video_name+".pkl"
        path_back = self.folder_analysis+"images_mortality/"+video_name+".png"
        #path_back = self.background_path
        
        if os.path.isfile(path_back): # If the video has a background image made already
        
            if os.path.isfile(video_object_path)==0 or step_to_force_analyze==0: # if video_object was not already created
                video_tracked = single_video_analysis(self,video_idx,debug_mode=1) # Initialize a video analysis object.
            else:
                print("Load video_obj of : "+video_name)
                with open(video_object_path, 'rb') as f:
                    video_tracked = pickle.load(f)

    ########       
            video_tracked.segment_resting_and_moving_objects(step_to_force_analyze,debug_mode=1) # Step 1
            video_tracked.track_resting_obj(step_to_force_analyze,debug_mode=1) #                  Step 2
            video_tracked.track_moving_obj(step_to_force_analyze,debug_mode=1) #                   Step 3
    ######## Loop over increasing values of time_window_search and max_distance #    Step 4
            max_distance_search = self.settings["assembly"]["max_distance_search"]
            time_window_search = self.settings["assembly"]["time_window_search"]
            MAX_NB_ASSEMBLING = 5
            for trial in np.arange(MAX_NB_ASSEMBLING):
                print("Tracking step :"+str(trial))
                video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)
                try:
                    video_tracked.assemble_resting_and_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)
                except Exception:
                    print("No resting tracks to match")
                try:
                    video_tracked.assemble_unmatched_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)
                except Exception:
                    print("No moving tracks to match")

                if trial == MAX_NB_ASSEMBLING-1: # Finished last round
                    video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)

                time_window_search = time_window_search*1.1
                max_distance_search = max_distance_search*1.2

    # ############
    # # #         # Assemble forward backward tracks #                                     Step 5
            video_tracked.assemble_tracks_ids(step_to_force_analyze,debug_mode=1)
            video_tracked.extract_complete_trajectories_from_video(debug_mode=1)

            video_tracked.display_video_with_tracking("forward",starting_frame=15000,time_btw_frames=0.005)

        else:
            print("background image missing")
        


########################### Compute activity and trajectories from the video. ###################
    def run_video_analysis_all(self,video_idx):
        """ 
        This function analyzes all videos from a given folder
        """
        # Logging files
        create_folder(self.folder_analysis+"log_analysis")
        video_name = os.path.splitext(self.list_videos_files[video_idx])[0]

        path_back = self.folder_analysis+"images_mortality/"+video_name+".png"
        #path_back = self.background_path
        if os.path.isfile(path_back): # If the video has a background image made already

            if os.path.isfile(self.folder_final+"/forward_mosq_tracks_"+video_name)==False:
                log_file = open(self.folder_analysis+"log_analysis/"+video_name+".log","w")
                old_stdout = sys.stdout
                sys.stdout = log_file

                step_to_force_analyze = 0
                MAX_NB_ASSEMBLING = 5
                video_tracked = single_video_analysis(self,video_idx,debug_mode=0) # Initialize a video analysis object.
        ########
                video_tracked.segment_resting_and_moving_objects(step_to_force_analyze,debug_mode=0) # Step 1
                video_tracked.track_resting_obj(step_to_force_analyze,debug_mode=0) #                  Step 2
                video_tracked.track_moving_obj(step_to_force_analyze,debug_mode=0) #                   Step 3
        ######## Loop over increasing values of time_window_search and max_distance #    Step 4
                max_distance_search = self.settings["assembly"]["max_distance_search"]
                time_window_search = self.settings["assembly"]["time_window_search"]
                for trial in np.arange(MAX_NB_ASSEMBLING):
                    print("Tracking step :"+str(trial))
                    video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)
                    try:
                        video_tracked.assemble_resting_and_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)
                    except Exception:
                        print("No resting tracks to match")
                    try:
                        video_tracked.assemble_unmatched_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)
                    except Exception:
                        print("No moving tracks to match")
                    
                    if trial == MAX_NB_ASSEMBLING-1: # Finished last round
                        video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)

                    time_window_search = time_window_search*1.1
                    max_distance_search = max_distance_search*1.2

        # ############
        # # #         # Assemble forward backward tracks #                                     Step 5
                video_tracked.assemble_tracks_ids(step_to_force_analyze,debug_mode=0)
                try:
                    video_tracked.extract_complete_trajectories_from_video(debug_mode=0)
                except Exception as error:
                # handle the exception
                    print("An exception occurred:", error)
                

                print("Finished analyzing "+video_name)

                sys.stdout = old_stdout
                log_file.close()
        else:
            print("background image missing")            


    def run_single_video_zone_traj_analysis(self, video_name,debug_mode):
            
        video_analysis = single_video_analysis(self, video_name, debug_mode=False)
        forward_tracks_path = os.path.join(self.folder_final, f"forward_mosq_tracks_{video_name}")

        with open(forward_tracks_path, 'rb') as f:
            video_analysis.mosquito_tracks = pickle.load(f)
    # Perform the analysis to extract flight metrics around resting points
        video_analysis.extract_flight_metrics_around_resting()
        video_analysis.save_tracking_results()
            



########################### Compute activity and trajectories from the video. ###################
    def run_single_video_analysis(self, video_name,debug_mode):
        """ 
        This function analyzes all videos from a given folder
        """
        # Logging files
        #create_folder(self.folder_analysis + "/log_analysis")

        path_back = os.path.join(self.folder_analysis, "images_mortality", f"{video_name}.png")
        #print(path_back)

        if os.path.isfile(path_back):  # If the video has a background image made already

            forward_tracks_path = os.path.join(self.folder_final, f"forward_mosq_tracks_{video_name}")
            if not os.path.isfile(forward_tracks_path):
                log_file_path = os.path.join(self.folder_analysis, "log_analysis", f"{video_name}.log")
                logger = MultiLogger(self.log, log_file_path)

                old_stdout = sys.stdout
                sys.stdout = logger
                # Everything below is wrapped so sys.stdout (and this segment's log FileHandler)
                # always gets restored/released, even if one of the calls below that ISN'T already
                # individually try/except-guarded raises (segment_resting_and_moving_objects,
                # track_resting_obj, track_moving_obj, clean_tracks, assemble_tracks_ids, or the
                # single_video_analysis(...) constructor itself). Every call in this function
                # already reassigns sys.stdout to its own fresh logger on entry (line above), which
                # papers over the leak for the *next* segment tracked by this worker -- but between
                # the crash and that next call, any unrelated print() on this worker process (from
                # _run_batch_analysis, or whatever runs next) would silently land in *this* failed
                # segment's log file instead of the real console/log, and the FileHandler this
                # MultiLogger opened for log_file_path would go unclosed. The try/finally does not
                # change what gets computed or swallow anything new -- an exception here still
                # propagates to _run_batch_analysis's existing handler exactly as before; it just
                # guarantees the resource cleanup runs first.
                try:
                    step_to_force_analyze = 0
                    MAX_NB_ASSEMBLING = 5
                    video_tracked = single_video_analysis(self, video_name, debug_mode)  # Initialize a video analysis object.

                    # Steps for video analysis
                    video_tracked.segment_resting_and_moving_objects(step_to_force_analyze, debug_mode)  # Step 1
                    video_tracked.track_resting_obj(step_to_force_analyze, debug_mode)  # Step 2
                    video_tracked.track_moving_obj(step_to_force_analyze, debug_mode)  # Step 3

                    # Loop over increasing values of time_window_search and max_distance # Step 4
                    max_distance_search = self.settings["assembly"]["max_distance_search"]
                    time_window_search = self.settings["assembly"]["time_window_search"]

                    for trial in np.arange(MAX_NB_ASSEMBLING):
                        print(f"Tracking step: {trial}")
                        video_tracked.clean_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)

                        try:
                            video_tracked.assemble_resting_and_moving_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)
                        except Exception:
                            print("No resting tracks to match")

                        try:
                            video_tracked.assemble_unmatched_moving_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)
                        except Exception:
                            print("No moving tracks to match")

                        if trial == MAX_NB_ASSEMBLING - 1:  # Finished last round
                            video_tracked.clean_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)

                        time_window_search *= 1.1
                        max_distance_search *= 1.2

                    # Assemble forward backward tracks # Step 5
                    video_tracked.assemble_tracks_ids(step_to_force_analyze, debug_mode)

                    try:
                        video_tracked.extract_complete_trajectories_from_video_V2(debug_mode)
                    except Exception as error:
                        print("An exception occurred completing tracks:", error)
                    try:
                        video_tracked.extract_mosquito_population_variables()
                    except Exception as error:
                        print("An exception occurred in computing population var (fraction flying etc):", error)
                    try:
                        video_tracked.extract_mosquito_individual_variables()
                    except Exception as error:
                        print("An exception occurred in computing individual variables (flight speed etc):", error)

                    try:
                        video_tracked.extract_mosquito_resting_variables()
                    except Exception as error:
                        print("An exception occurred in extracting resting variables results:", error)

                    try:
                        video_tracked.save_tracking_results()
                    except Exception as error:
                        print("An exception occurred in saving results:", error)

                        #video_tracked.extract_complete_trajectories_from_video(debug_mode=0)


                    #print(f"Finished analyzing {video_name}")
                finally:
                    # Reset stdout
                    sys.stdout = old_stdout

        else:
            print("background image missing")

### 
    def re_extract_single_video_analysis(self, video_name,debug_mode):
        """ 
        This re-extracts the population variable etc from the raw tracking data
        """
        # Logging files
        #create_folder(self.folder_analysis + "/log_analysis")

        forward_tracks_path = os.path.join(self.folder_final, f"forward_mosq_tracks_{video_name}")

        if os.path.isfile(forward_tracks_path):
            log_file_path = os.path.join(self.folder_analysis, "log_analysis", f"{video_name}.log")
            logger = MultiLogger(self.log, log_file_path)

            with open(forward_tracks_path, 'rb') as f:
                mosquito_tracks = pickle.load(f)

            old_stdout = sys.stdout
            sys.stdout = logger

            video_tracked = single_video_analysis(self, video_name, debug_mode)  # Initialize a video analysis object.

            video_tracked.mosquito_tracks = mosquito_tracks
            try:
                video_tracked.extract_mosquito_population_variables()
            except Exception as error:
                print("An exception occurred in computing population var (fraction flying etc):", error)
            try:
                video_tracked.extract_mosquito_individual_variables()
            except Exception as error:
                print("An exception occurred in computing individual variables (flight speed etc):", error)

            try:
                video_tracked.extract_mosquito_resting_variables()
            except Exception as error:
                print("An exception occurred in extracting resting variables results:", error)


            try:
                video_tracked.extract_flight_metrics_around_resting()
            except Exception as error:
                print("An exception occurred in extracting speaker-side stats:", error)
            
            try:
                video_tracked.save_tracking_results()
            except Exception as error:
                print("An exception occurred in saving results:", error)

                #video_tracked.extract_complete_trajectories_from_video(debug_mode=0)


            #print(f"Finished analyzing {video_name}")

            # Reset stdout
            sys.stdout = old_stdout




    def display_video_final_tracking(self,video_name,starting_frame,time_btw_frames):

        #video_name = os.path.splitext(self.list_videos_files[video_idx])[0] # Load video_name
        video_tracked = single_video_analysis(self,video_name,debug_mode=0) # Initialize a video analysis object.

        with open(self.folder_final+"/forward_mosq_tracks_"+video_name, 'rb') as f:
            mosquito_tracks = pickle.load(f)

        # Initialiaze video
        cap = cv2.VideoCapture(video_tracked.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        f_i = starting_frame

        while True:
            suc,frame = cap.read()
            time.sleep(time_btw_frames)

            if suc == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame.copy()

                if f_i > 1:
                    frame_idx = f_i

                    for k,id in enumerate(mosquito_tracks.objects.keys()):
                        t_start = mosquito_tracks.objects[id]["start"]
                        t_end = mosquito_tracks.objects[id]["end"]

                        if frame_idx > t_start and frame_idx < t_end:
                            t_relative = frame_idx - t_start+2

                            try:
                                centroid = mosquito_tracks.objects[id]["coordinates"][t_relative]
                                text = "id {}".format(id)
                                if mosquito_tracks.objects[id]["state"][t_relative]==0:
                                    color = (0, 0, 255)
                                else:
                                    color = (255, 0, 0)
                                cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, color, 1)
                                cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                            except Exception:
                                print("error display")


                cv2.imshow("frame",img)
                f_i += 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if f_i>n_frame-2:
                    break
        cap.release()
        cv2.destroyAllWindows()



########################### Compute activity and trajectories from the video. ###################
    def concatenate_flight_activity_(self):
        print("assembling flight activities of all videos")

        files_tracking = [f for f in listdir(self.folder_final) if isfile(join(self.folder_final, f)) and f.startswith("forward")]
        files_tracking.sort()

        for i,file_name in enumerate(files_tracking):
            if i <10000:
                progress_bar(i, len(files_tracking), bar_length=20)
                #print(file_name)
                with open(self.folder_final+"/"+file_name, 'rb') as f:
                    mosquito_tracks = pickle.load(f)

                    if 'fly' in locals():
                        fly = pd.concat([fly,mosquito_tracks.nb_mosquitos_flying])
                    else:
                        fly = mosquito_tracks.nb_mosquitos_flying

        self.global_flight_activity = fly
        exp_object_path = self.folder_temp_data+"/temp_data_"+self.experiment_alias+".pkl"
        with open(exp_object_path, 'wb') as f:
            pickle.dump(self,f)

########################### Compute activity and trajectories from the video. ###################
    def plot_flight_activity_(self):       
        
        fly = self.global_flight_activity 

        try:
            fps = float(self.settings.get("fps", 25.0))
        except Exception:
            fps = 25.0
        if fps <= 0:
            fps = 25.0

        fig, ax = plt.subplots(1, 1, dpi=PLOT_DPI_TRAJECTORY)
        fig.set_figheight(PLOT_FIGHEIGHT_TRAJECTORY)
        fig.set_figwidth(PLOT_FIGWIDTH_TRAJECTORY)

        ax.plot(fly.rolling(int(fps * ROLLING_WINDOW_SECONDS), min_periods=1).mean())
        ax.set_ylim([0, 0.2])

        plt.ioff()
        plt.savefig(self.folder_analysis+"plots_trajectories/__"+self.video_name+"_plot_ID_"+str(id)+'.png',bbox_inches='tight')

        #plt.show()

########################### Define the precise angle of the image ###################
    def user_input_draw_borders_cage(self, force_to_redo):
        if force_to_redo == 1:
            images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
            mortality_images = [f for f in os.listdir(images_mortality_folder)
                                if f.endswith('.png') and not f.startswith('.')]
            mortality_images.sort()
            if not mortality_images:
                print("No images found in images_mortality folder.")
                return

            background_image_path = os.path.join(images_mortality_folder, mortality_images[0])

            points = draw_parallelogram(background_image_path)
            self.settings["cage_border_points"] = self._normalize_quad_order(self._normalize_quad_points(points))

            if os.path.isfile(self.settings_file):
                with open(self.settings_file, 'w') as file:
                    yaml.dump(self.settings, file)

    def _save_settings_file(self):
        if os.path.isfile(self.settings_file):
            with open(self.settings_file, 'w') as file:
                yaml.dump(self.settings, file)

    def _normalize_quad_points(self, raw_points):
        points = []
        if raw_points is None:
            return points
        for p in raw_points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append([int(round(float(p[0]))), int(round(float(p[1])))])
        return points

    def _compute_default_center_from_cage(self, cage_arr, inset_ratio=0.45):
        centroid = np.mean(cage_arr, axis=0)
        inset = max(0.2, min(float(inset_ratio), 0.8))
        center_pts = []
        for p in cage_arr:
            q = p + inset * (centroid - p)
            center_pts.append([int(round(q[0])), int(round(q[1]))])
        return center_pts

    def _normalize_quad_order(self, points):
        if not points:
            # No border drawn yet (e.g. a fresh experiment, see Fix 2) — np.array([]) would be
            # 1-D and crash every downstream [:, 1]-style index. Nothing to order.
            return []
        arr = np.array(points, dtype=float)
        center = np.mean(arr, axis=0)
        angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = arr[order]
        tl_idx = np.argmin(ordered[:, 0] + ordered[:, 1])
        ordered = np.roll(ordered, -tl_idx, axis=0)
        return [[int(round(p[0])), int(round(p[1]))] for p in ordered]

    def _align_inner_quad_to_outer(self, outer_points, inner_points):
        if len(outer_points) != 4 or len(inner_points) != 4:
            return inner_points

        outer = np.array(outer_points, dtype=float)
        inner_base = np.array(self._normalize_quad_order(inner_points), dtype=float)
        best = inner_base
        best_score = float("inf")

        candidates = [inner_base, np.flip(inner_base, axis=0)]
        for candidate in candidates:
            for shift in range(4):
                aligned = np.roll(candidate, -shift, axis=0)
                score = float(np.sum(np.linalg.norm(outer - aligned, axis=1)))
                if score < best_score:
                    best_score = score
                    best = aligned

        return [[int(round(p[0])), int(round(p[1]))] for p in best]

    def _distance_point_to_segment(self, point, segment_start, segment_end):
        point = np.array(point, dtype=float)
        segment_start = np.array(segment_start, dtype=float)
        segment_end = np.array(segment_end, dtype=float)
        segment = segment_end - segment_start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - segment_start))
        projection = np.dot(point - segment_start, segment) / denom
        projection = max(0.0, min(1.0, float(projection)))
        closest = segment_start + projection * segment
        return float(np.linalg.norm(point - closest))

    def _annotate_polygon_lengths(self, image, points, color):
        if len(points) < 3:
            return
        for idx in range(len(points)):
            p1 = np.array(points[idx], dtype=float)
            p2 = np.array(points[(idx + 1) % len(points)], dtype=float)
            midpoint = ((p1 + p2) / 2.0).astype(int)
            length_px = float(np.linalg.norm(p2 - p1))
            cv2.putText(
                image,
                f"{length_px:.1f}px",
                (int(midpoint[0]) + 4, int(midpoint[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    def _annotate_corner_edge_distances(self, image, points, cage_points, color):
        if len(points) < 3 or len(cage_points) < 3:
            return
        for point in points:
            distances = [self._distance_point_to_segment(point, cage_points[i], cage_points[(i + 1) % len(cage_points)]) for i in range(len(cage_points))]
            min_distance = min(distances)
            point_xy = (int(round(point[0])) + 5, int(round(point[1])) + 14)
            cv2.putText(
                image,
                f"d={min_distance:.1f}px",
                point_xy,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    def resize_center_region(self, scale_factor=0.85):
        cage_points = self._normalize_quad_points(self.settings.get("cage_border_points"))
        if len(cage_points) != 4:
            print("Cannot resize center region: cage border must contain exactly 4 points.")
            return False

        center_points = self._normalize_quad_points(self.settings.get("center_border_points"))
        if len(center_points) != 4:
            center_points = self._compute_default_center_from_cage(np.array(cage_points, dtype=float), inset_ratio=0.45)

        center_arr = np.array(center_points, dtype=float)
        centroid = np.mean(center_arr, axis=0)
        scale_factor = max(0.1, min(float(scale_factor), 2.0))
        resized = centroid + (center_arr - centroid) * scale_factor
        resized_points = self._normalize_quad_order([[float(p[0]), float(p[1])] for p in resized])

        self.settings["center_border_points"] = resized_points
        speaker_side_index = int(self.settings.get("speaker_side_index", 0))
        if self._auto_generate_side_regions_from_cage(speaker_side_index=speaker_side_index):
            self._save_settings_file()
            return True
        return False

    def _auto_generate_side_regions_from_cage(self, speaker_side_index=0, inset_ratio=0.45):
        cage_points = self._normalize_quad_order(self._normalize_quad_points(self.settings.get("cage_border_points")))
        if len(cage_points) != 4:
            print("Cannot auto-generate side regions: cage border must contain exactly 4 points.")
            return False

        cage_arr = np.array(cage_points, dtype=float)
        center_points = self._normalize_quad_points(self.settings.get("center_border_points"))
        if len(center_points) != 4:
            center_points = self._compute_default_center_from_cage(cage_arr, inset_ratio=inset_ratio)
        center_points = self._align_inner_quad_to_outer(cage_points, center_points)
        center_arr = np.array(center_points, dtype=float)

        side_polygons = []
        for idx in range(4):
            p0 = cage_arr[idx]
            p1 = cage_arr[(idx + 1) % 4]
            q0 = center_arr[idx]
            q1 = center_arr[(idx + 1) % 4]
            poly = [
                [int(round(p0[0])), int(round(p0[1]))],
                [int(round(p1[0])), int(round(p1[1]))],
                [int(round(q1[0])), int(round(q1[1]))],
                [int(round(q0[0])), int(round(q0[1]))],
            ]
            side_polygons.append(poly)

        speaker_side_index = int(speaker_side_index) % 4
        # Reorder the geometric edges into the physical side order used by the UI:
        # side 1 = left, side 2 = bottom, side 3 = right, side 4 = top.
        physical_side_order = [3, 2, 1, 0]
        side_polygons = [side_polygons[idx] for idx in physical_side_order]
        ordered_sides = [side_polygons[(speaker_side_index + k) % 4] for k in range(4)]

        # Keep legacy keys so downstream code remains compatible.
        self.speaker_side_border_points = ordered_sides[0]
        self.sugar_border_points = self.speaker_side_border_points
        self.settings["sugar_border_points"] = self.speaker_side_border_points
        self.settings["control_border_points"] = ordered_sides[1]
        self.settings["square_3_border_points"] = ordered_sides[2]
        self.settings["square_4_border_points"] = ordered_sides[3]
        self.settings["center_border_points"] = center_points
        self.settings["auto_side_regions"] = True
        self.settings["speaker_side_index"] = speaker_side_index

        self._save_settings_file()
        return True


    def user_input_draw_speaker_side_region(self, force_to_redo):
        if force_to_redo == 1:
            if self._auto_generate_side_regions_from_cage(speaker_side_index=0):
                print("Auto-generated side regions with speaker on side 1.")

    # Legacy alias for backward compatibility with sugar feeder naming.
    user_input_draw_sugar_feeding = user_input_draw_speaker_side_region

    def user_input_draw_center_region(self, force_to_redo):
        if force_to_redo == 1:
            images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
            mortality_images = [f for f in os.listdir(images_mortality_folder)
                                if f.endswith('.png') and not f.startswith('.')]
            mortality_images.sort()
            if not mortality_images:
                print("No images found in images_mortality folder.")
                return

            background_image_path = os.path.join(images_mortality_folder, mortality_images[0])
            points = draw_parallelogram(background_image_path)
            center_points = self._normalize_quad_points(points)
            if len(center_points) != 4:
                print("Center region requires exactly 4 points.")
                return

            self.settings["center_border_points"] = self._normalize_quad_order(center_points)
            speaker_side_index = int(self.settings.get("speaker_side_index", 0))
            if self._auto_generate_side_regions_from_cage(speaker_side_index=speaker_side_index):
                print("Center corners saved and side regions regenerated.")


    def user_input_draw_control_squares(self, force_to_redo):
        if force_to_redo == 1:
            if self._auto_generate_side_regions_from_cage(speaker_side_index=1):
                print("Auto-generated side regions with speaker on side 2.")

    def user_input_draw_control_squares_3(self, force_to_redo):
        if force_to_redo == 1:
            if self._auto_generate_side_regions_from_cage(speaker_side_index=2):
                print("Auto-generated side regions with speaker on side 3.")

    def user_input_draw_control_squares_4(self, force_to_redo):
        if force_to_redo == 1:
            if self._auto_generate_side_regions_from_cage(speaker_side_index=3):
                print("Auto-generated side regions with speaker on side 4.")

    def reset_drawn_borders(self, keep_cage=True):
        reset_keys = [
            "sugar_border_points",
            "control_border_points",
            "square_3_border_points",
            "square_4_border_points",
            "center_border_points",
        ]
        if not keep_cage:
            reset_keys.append("cage_border_points")

        for key in reset_keys:
            self.settings[key] = []

        self.settings["auto_side_regions"] = False
        self.settings["speaker_side_index"] = 0
        self._save_settings_file()

        self.cage_border_points = self.settings.get("cage_border_points", [])
        self.speaker_side_border_points = self.settings.get("sugar_border_points", [])
        self.sugar_border_points = self.speaker_side_border_points
        self.control_border_points = self.settings.get("control_border_points", [])

                
########################### Define the precise angle of the image ###################
    def plot_all_borders(self):

        with open(self.settings_file, 'r') as file:
                settings = yaml.safe_load(file)
                self.settings = settings

        # These early-return diagnostics also go through self.log (not just print()), since a
        # bare print() is invisible in the GUI's log pane — without it, a user clicking "Show
        # background with borders" before any reference image exists just sees nothing happen,
        # with no clue why.
        log = self.log or print
        images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
        if not os.path.isdir(images_mortality_folder):
            log(f"Cannot show background with borders: folder not found: {images_mortality_folder}")
            return
        mortality_images = [f for f in os.listdir(images_mortality_folder)
                                if f.endswith('.png') and not f.startswith('.')]
        mortality_images.sort()
        if not mortality_images:
            log("Cannot show background with borders: no reference images in images_mortality/ "
                "yet — run \"Extract Images from Video\" then \"Get Background from Images\" first.")
            return

        # Use the first image as the background
        background_image_path = os.path.join(images_mortality_folder, mortality_images[0])

        def _normalize_points(raw_points):
            if raw_points is None:
                return []
            pts = []
            try:
                for p in raw_points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((int(p[0]), int(p[1])))
            except Exception:
                return []
            return pts

        def plot_polygon(color, points, image, thickness=2):
            if len(points) < 3:
                return False
            for i in range(len(points)):
                cv2.line(image, points[i], points[(i + 1) % len(points)], color, thickness)
            return True

        def annotate_polygon(points, text, color, y_offset=0):
            if len(points) < 3:
                return
            pts_arr = np.array(points, dtype=np.int32)
            center = np.mean(pts_arr, axis=0).astype(int)
            text_x = int(center[0]) - 40
            text_y = int(center[1]) + int(y_offset)
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        image = cv2.imread(background_image_path)
        if image is None:
            log(f"Cannot show background with borders: could not read {background_image_path}")
            return

        cage_points = _normalize_points(self._normalize_quad_order(self.settings.get("cage_border_points")))
        plot_polygon((0, 255, 0), cage_points, image, thickness=2)
        annotate_polygon(cage_points, "CAGE", (0, 255, 0), y_offset=-10)

        sugar_points = _normalize_points(self.settings.get("sugar_border_points"))
        control_points = _normalize_points(self.settings.get("control_border_points"))
        square_3_points = _normalize_points(self.settings.get("square_3_border_points"))
        square_4_points = _normalize_points(self.settings.get("square_4_border_points"))
        center_points = _normalize_points(self.settings.get("center_border_points"))

        # Speaker side is always stored in sugar_border_points for compatibility.
        if plot_polygon((255, 0, 0), sugar_points, image, thickness=3):
            annotate_polygon(sugar_points, "SPEAKER", (255, 0, 0))

        if plot_polygon((0, 0, 255), control_points, image, thickness=2):
            annotate_polygon(control_points, "CTRL 1", (0, 0, 255))
        if plot_polygon((0, 0, 255), square_3_points, image, thickness=2):
            annotate_polygon(square_3_points, "CTRL 2", (0, 0, 255))
        if plot_polygon((0, 0, 255), square_4_points, image, thickness=2):
            annotate_polygon(square_4_points, "CTRL 3", (0, 0, 255))
        if plot_polygon((0, 255, 255), center_points, image, thickness=2):
            annotate_polygon(center_points, "CENTER", (0, 255, 255))


            # Display the final image with the parallelogram
        cv2.imwrite(self.folder_analysis+"/background_with_borders.png",image)
        



########################### Define the precise angle of the image ###################
    def extract_images(self,BATCH_NB):
        print("Extracting one image per video")
        create_folder(self.folder_analysis + "individual_images_"+str(BATCH_NB)+"/")
        for k, video_file in enumerate(self.list_videos_files):
            progress_bar(k, len(self.list_videos_files), bar_length=20)
            if video_file.endswith('.mp4'):
                # Save each frame as a PNG file only if it doesn't exist already
                frame_filename = video_file.replace('.mp4', '.png')
                output_path = os.path.join(self.folder_analysis, "individual_images_"+str(BATCH_NB), frame_filename)
                if not os.path.exists(output_path):
                    #start = video_file.find('Cage')
                    video_path = self.folder_videos + video_file  # [start::]
                    cap = cv2.VideoCapture(video_path)
                    suc, frame = cap.read()
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cap.release()


                    
                    
                    cv2.imwrite(output_path, frame_gray)

        return
    
    def extract_images_v2(self,force_to_rerun):
        print("Extracting one image per video")
        create_folder(self.folder_analysis + "/individual_images/")
        for k, video_file in enumerate(self.list_videos_files):
            progress_bar(k, len(self.list_videos_files), bar_length=20)
            if video_file.endswith('.mp4'):
                # Save each frame as a PNG file only if it doesn't exist already
                frame_filename = video_file.replace('.mp4', '.png')
                output_path = os.path.join(self.folder_analysis, "individual_images", frame_filename)
                print(output_path)
                if not os.path.exists(output_path) or force_to_rerun==1 :
                    #start = video_file.find('Cage')
                    video_path = self.folder_videos + "/"+video_file  # [start::]
                    print(video_path)
                    # One bad file (truncated/corrupt .mp4 -- a real occurrence in raw Pi
                    # recordings) must not abort every video after it: list_videos_files is
                    # sorted, so an unguarded crash here silently left every later segment in the
                    # experiment without a reference image at all, with no error surfaced anywhere
                    # (the caller has no per-video granularity to report from). cap.release() also
                    # used to be unreached on a failed read, leaking the handle. Skip-and-continue,
                    # always releasing.
                    cap = cv2.VideoCapture(video_path)
                    try:
                        suc, frame = cap.read()
                        if not suc or frame is None:
                            print(f"WARNING: could not read a frame from {video_path} -- "
                                  "skipping (file may be corrupt/truncated); no reference image "
                                  "will be created for this segment.")
                            continue
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    finally:
                        cap.release()

                    cv2.imwrite(output_path, frame_gray)

        return  # No need to return images_to_av if not used elsewhere
    
    def polish_background(self, median_frame):
        """Apply bilateral filter to reduce noise while preserving edges"""
        # Bilateral filter: excellent for noise reduction while keeping edges sharp
        polished = cv2.bilateralFilter(median_frame, d=9, sigmaColor=75, sigmaSpace=75)
        return polished

    def _write_background_references(self, window_images, video_name, force_to_rerun):
        """Write the per-pixel median background ``<video>.png`` for one segment.

        Used by the moving/frame-difference path. The resting path's bright-percentile
        reference (WS1) is built *within-video* on demand in single_video_analysis
        (see _get_within_video_rest_reference) -- a cross-video median window cannot capture
        this video's exact mesh nor reliably exclude long-resting mosquitoes.
        """
        median_path = os.path.join(self.folder_analysis, "images_mortality", video_name + ".png")
        if not os.path.isfile(median_path) or force_to_rerun == 1:
            median_frame = np.median(np.asarray(window_images), axis=0).astype(np.uint8)
            cv2.imwrite(median_path, self.polish_background(median_frame))

    def extract_average_background(self,force_to_rerun):
                # Average images
        create_folder(self.folder_analysis + "/images_mortality/")

        

        logger = MultiLogger(self.log, None)
        old_stdout = sys.stdout
        sys.stdout = logger
        try:
            self._extract_average_background_impl(force_to_rerun)
        finally:
            # Must always run: an unguarded exception here (e.g. a corrupt individual_images/*.png,
            # see the filtering below) used to leave sys.stdout permanently pointed at this
            # MultiLogger -- every later bare print() anywhere in the process (this experiment's
            # remaining steps, or a totally unrelated one processed afterward) would silently
            # reroute through a stale closure over this call's self.log instead of the real
            # console/file.
            sys.stdout = old_stdout

    def _extract_average_background_impl(self, force_to_rerun):
        size_window = 15 ##### Can change - reduced from 50 to minimize blur while still filtering mosquitoes
        print("Computing the median frame of 30-frame moving window over time")

        image_folder = self.folder_analysis + "/individual_images/"
        names = sorted(f for f in os.listdir(image_folder) if f.endswith('.png') and not f.startswith('.'))
        all_images = []
        all_names = []
        for filename in names:
            img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                # A corrupt/truncated PNG (e.g. from an interrupted previous run) would otherwise
                # feed a None into the np.median() calls below, which crash the whole batch --
                # every other video's background, not just this one's, would go uncomputed. Skip
                # it and keep both lists in step (all_names is indexed in parallel with all_images
                # throughout this function).
                print(f"WARNING: could not read {filename} in {image_folder} -- skipping "
                      "(file may be corrupt/truncated); it won't contribute to any background.")
                continue
            all_images.append(img)
            all_names.append(filename[:-4])

        for s, image in enumerate(all_images):
            if  size_window < s < len(all_images) - size_window:
                video_name = all_names[s]  # You might need to adjust how you get the video name
                progress_bar(s, len(all_images), bar_length= size_window)
                self._write_background_references(all_images[s - size_window:s + size_window], video_name, force_to_rerun)
            elif len(all_images) -  size_window <= s < len(all_images):
                video_name = all_names[s]
                self._write_background_references(all_images[- size_window*2:-1], video_name, force_to_rerun)

            elif 0 <= s <=  size_window:
                video_name = all_names[s]
                self._write_background_references(all_images[0: size_window*2], video_name, force_to_rerun)

    # def extract_images(self):
    #     print("Extracting one image per video")

    #     images_to_av = []
    #     for k,video_file in enumerate(self.list_videos_files):
    #         progress_bar(k, len(self.list_videos_files), bar_length=20)
    #         if video_file.endswith('.mp4'):
    #             #start = video_file.find('Cage')
    #             video_path = self.folder_videos+video_file#[start::]
    #             cap = cv2.VideoCapture(video_path)
    #             suc,frame = cap.read()
    #             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             images_to_av.append(frame_gray)
    #             cap.release()

    #     return images_to_av
    

#################### Extract
    def extract_number_longterm_resting_objects(self,background_image,show_movie,draw_plot):  

        def get_datetime_from_video_name(video_name):
            # Shared parser handles both old and new naming conventions.
            return get_datetime_from_file_name(video_name)
        
        print("Counting the number of dead mosquitos")
        # Initialize variable
        number_of_dead = []
        time_vid = []

        background = cv2.imread(background_image)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        folder_images = self.folder_analysis+"images_mortality/"

        files_images = [f for f in listdir(folder_images) if isfile(join(folder_images, f)) and f.endswith(".png")
                        and not f.startswith(".")]
        files_images.sort()
        video_tracked = single_video_analysis(self,0,debug_mode=0)

        with open(self.folder_analysis+"all_names", 'rb') as f:
            all_names = pickle.load(f)

        for s, image_name in enumerate(files_images):    
            progress_bar(s, len(files_images), bar_length=20)
            image = cv2.imread(folder_images+image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            img = image.copy()

            frame_gray = cv2.subtract(background,image)
            frame_gray[frame_gray<0]=0
            centroids_still = video_tracked.get_centroids_still_objects(frame_gray,self.settings["seg_resting"])         
            nb_dead = 0   
            for centroid in centroids_still:
                if video_tracked.point_inside_cage(self.cage_border_points,centroid):
                    cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 10, (0, 0, 255), 1)
                    nb_dead +=1
            number_of_dead.append(nb_dead)
            #print(get_datetime_from_video_name(all_names[s]))
            
            time_vid.append(get_datetime_from_video_name(all_names[s]))

            if show_movie:
                cv2.imshow("Dead buddy",img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        cv2.destroyAllWindows()

            
        data = {'time': time_vid,
        'dead_count': number_of_dead}
        df = pd.DataFrame(data)

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        if draw_plot:
            plt.plot(df)
            plt.savefig(self.folder_analysis+"mortality.png",bbox_inches='tight')
            plt.show()
        with open(self.folder_analysis+self.experiment_alias+"_count_dead_mosquito", 'wb') as f:
            pickle.dump(df,f)


    def plot_sample_flight_trajectories_from_video(self,axes,mosquito_tracks):
        video_name = mosquito_tracks.video_name
        nb_plotted = 0
        try:
            fps = float(self.settings.get("fps", 25.0))
        except Exception:
            fps = 25.0
        if fps <= 0:
            fps = 25.0
        for k,id in enumerate(mosquito_tracks.objects.keys()):
            state_v = mosquito_tracks.objects[id]["state"]
            coord_v = mosquito_tracks.objects[id]["coordinates"]

            state_v = np.array([1-i for i in state_v])
            runs = zero_runs(state_v)
            nb_tracks = len(runs[:,0])

            x = [a for a,b in  coord_v]
            y = [b for a,b in  coord_v]

            if nb_tracks>0:
                for k in np.arange(nb_tracks):
                    t_i = runs[k,0]
                    t_f = runs[k,1]

                    d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
                    d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

                    d_x_2 = [np.square(z) for z in d_x]
                    d_y_2 = [np.square(z) for z in d_y]

                    dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]
                    if t_f - t_i > fps * 10:
                        #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])

                        if nb_plotted < 18 and np.mean(dist)>5:
                            axes[nb_plotted] = self.plot_flight_trajectory(x[t_i+1:t_f],y[t_i+1:t_f],axes[nb_plotted],video_name)
                            nb_plotted += 1
                            
        return axes

    
    def plot_flight_trajectory(self,x,y,ax,video_name):
        # Set plot and add background
        back_path = self.video_path = os.path.join(self.folder_analysis ,"images_mortality", video_name)+".png"
        im = plt.imread(back_path)
        ax.imshow(im,zorder=1,cmap = "gray")

        x_f = uniform_filter1d(x, size=5)
        y_f = uniform_filter1d(y, size=5)

        try:
            fps = float(self.settings.get("fps", 25.0))
        except Exception:
            fps = 25.0
        if fps <= 0:
            fps = 25.0

        c_f = np.arange(len(x_f))
        c_f = np.array([c_f[i] / fps for i in np.arange(len(c_f))])

        points = np.array([x_f, y_f]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(c_f.min(), c_f.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)

        # Set the values used for colormapping
        lc.set_array(c_f)
        lc.set_linewidth(1)
        line = ax.add_collection(lc)

        ax.set_xlim([0 ,im.shape[0]])
        ax.set_ylim([0 ,im.shape[1]])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.tight_layout()
        return ax