import cv2
from PIL import Image, ImageTk
import tkinter as tk
import os
import threading
import time
import pickle
import re
import math
from datetime import datetime, timedelta
from buzzwatch_data_analysis.experiment_analysis import buzzwatch_experiment_analysis
from buzzwatch_data_analysis.misc_functions import create_folder
from buzzwatch_data_analysis.single_video_analysis import single_video_analysis


class VideoManager:
    def __init__(self, log_func):
        self.log = log_func
        self.cap = None
        self.total_frames = 0
        self.is_playing = False
        self.label = None
        self.scrollbar = None
        self.display_tracking = False
        self.mosquito_tracks = None
        self.speed_filter_min = None
        self.speed_filter_max = None
        self._track_speed_cache = {}
        self._play_thread = None

    def load_video(self, video_path):
        self.stop_and_release()
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            self.log(f"Error opening video file: {video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames < 1:
            self.log(f"Error: No frames found in the video {video_path}")
            return False

        return True

    def set_display_label(self, label):
        self.label = label

    def set_scrollbar(self, scrollbar):
        self.scrollbar = scrollbar

    def set_display_tracking(self, display_tracking):
        self.display_tracking = display_tracking

    def load_tracking_data(self, tracking_file):
        with open(tracking_file, 'rb') as f:
            self.mosquito_tracks = pickle.load(f)
        self._track_speed_cache = {}

    def set_speed_filter(self, speed_min=None, max_speed=None):
        self.speed_filter_min = speed_min
        self.speed_filter_max = max_speed

    def _get_track_average_speed(self, track_id, data):
        if track_id in self._track_speed_cache:
            return self._track_speed_cache[track_id]

        coords = data.get("coordinates", []) or []
        if len(coords) < 2:
            self._track_speed_cache[track_id] = 0.0
            return 0.0

        dists = []
        for idx in range(1, len(coords)):
            p0 = coords[idx - 1]
            p1 = coords[idx]
            if p0 is None or p1 is None:
                continue
            try:
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
            except Exception:
                continue
            if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
                continue
            dists.append(math.hypot(x1 - x0, y1 - y0))

        avg_speed = float(sum(dists) / len(dists)) if dists else 0.0
        self._track_speed_cache[track_id] = avg_speed
        return avg_speed

    def play_video(self, time_btw_frames=0.04):
        if self.is_playing:
            self.log("Video is already playing.")
            return
        self.is_playing = True
        self._play_thread = threading.Thread(target=self._play_video, args=(time_btw_frames,), daemon=True)
        self._play_thread.start()

    def _play_video(self, time_btw_frames):
        while self.is_playing and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                break
            if self.display_tracking and self.mosquito_tracks:
                frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame = self.overlay_tracking(frame, frame_idx)
            frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._schedule_frame_update(frame, frame_index)
            time.sleep(time_btw_frames)

    def _schedule_frame_update(self, frame, frame_index):
        if self.label is None:
            return
        try:
            if not self.label.winfo_exists():
                return
        except Exception:
            return

        def _update_ui():
            try:
                if self.label is None or not self.label.winfo_exists():
                    return
                self._display_frame(frame)
                self.update_scrollbar(frame_index)
            except Exception:
                return

        try:
            self.label.after(0, _update_ui)
        except Exception:
            return

    def pause_video(self):
        self.is_playing = False

    def stop_and_release(self):
        self.is_playing = False
        if self._play_thread and self._play_thread.is_alive() and self._play_thread is not threading.current_thread():
            self._play_thread.join(timeout=0.5)
        self._play_thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.label is not None:
            try:
                self.label.config(image='')
            except Exception:
                pass
            try:
                self.label.imgtk = None
            except Exception:
                pass

    def show_frame(self, frame_index):
        if self.cap is None or not self.cap.isOpened():
            self.log("No video loaded or video capture is not opened.")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            if self.display_tracking and self.mosquito_tracks:
                frame = self.overlay_tracking(frame, frame_index)
            self._display_frame(frame)

    def _display_frame(self, frame):
        if not self.label or frame is None:
            self.log("Frame or video display label is not available.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_width = self.label.winfo_width()
        label_height = self.label.winfo_height()
        frame_rgb_resized = self.resize_image(frame_rgb, label_width, label_height)
        img = Image.fromarray(frame_rgb_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.config(image=imgtk)

    def resize_image(self, img, width, height):
        aspect_ratio = img.shape[1] / img.shape[0]
        if width / height > aspect_ratio:
            width = int(height * aspect_ratio)
        else:
            height = int(width / aspect_ratio)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def overlay_tracking(self, frame, frame_idx):
        # Drawing overlay logic goes here
        for id, data in self.mosquito_tracks.objects.items():
            if self.speed_filter_min is not None and self.speed_filter_max is not None:
                avg_speed = self._get_track_average_speed(id, data)
                if not (self.speed_filter_min <= avg_speed <= self.speed_filter_max):
                    continue
            t_start, t_end = data["start"], data["end"]
            if t_start <= frame_idx <= t_end:
                t_relative = frame_idx - t_start+2
                try:
                    centroid = data["coordinates"][t_relative]
                    state = data["state"][t_relative]
                    color = (0, 0, 255) if state == 0 else (255, 0, 0)
                    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, color, 1)
                except Exception as e:
                    continue
                    #print(e)
                    #self.log(f"Error: {e}")
        return frame

    def next_frame(self):
        if not self.is_playing and self.cap and self.cap.isOpened():
            current_frame = self.scrollbar.get()
            if current_frame < self.total_frames - 1:
                self.show_frame(current_frame + 1)
                self.update_scrollbar(current_frame + 1)

    def previous_frame(self):
        if not self.is_playing and self.cap and self.cap.isOpened():
            current_frame = self.scrollbar.get()
            if current_frame > 0:
                self.show_frame(current_frame - 1)
                self.update_scrollbar(current_frame - 1)

    def update_scrollbar(self, frame_index):
        if self.scrollbar is None:
            return
        try:
            self.scrollbar.set(frame_index)
        except Exception:
            pass
    
    def get_total_frames(self):
        return self.total_frames

    def get_datetime_from_file_name(self, video_name):
            try:
                name = video_name[:-4] if video_name.lower().endswith(".mp4") else video_name

                # Pattern A: ..._YYYYMMDD_HHMMSS (optional _vNN afterwards)
                match = re.search(r'(\d{8})_(\d{6})(?:_v(\d+))?(?:_|$)', name)
                if match:
                    date_part = match.group(1)
                    time_part = match.group(2)
                    video_number = int(match.group(3)) if match.group(3) else 1
                    base_time = datetime(
                        int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8]),
                        int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6])
                    )
                    return base_time + timedelta(minutes=20 * (video_number - 1))

                # Pattern B (legacy): ..._YYMMDD_<label>_HHMMSS_vNN
                match = re.search(r'(\d{6})_[^_]+_(\d{6})_v(\d+)(?:_|$)', name)
                if match:
                    date_part = match.group(1)
                    time_part = match.group(2)
                    video_number = int(match.group(3))
                    base_time = datetime(
                        2000 + int(date_part[:2]), int(date_part[2:4]), int(date_part[4:6]),
                        int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6])
                    )
                    return base_time + timedelta(minutes=20 * (video_number - 1))

                raise ValueError("No matching date/time pattern found")
            except Exception as e:
                print(f"Error parsing date and time from video name '{video_name}': {e}")
                return None