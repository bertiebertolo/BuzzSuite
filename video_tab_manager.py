import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from threading import Thread
import time

class VideoTabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.video_manager = ui_manager.video_manager
        self.log = ui_manager.log
        self.tab = None
        self.video_label = None
        self.video_start_time_label = None
        self.video_scrollbar = None
        self.video_listbox = None
        self.playing = False

    def build_playback_section(self, parent: tk.Frame):
        """Build the raw-video list + display + transport controls into `parent`. Reused by the
        Activity Analysis tab (raw playback folded into tracking selection) — same widgets/logic
        as create_video_inspection_tab, just no longer owning a top-level notebook tab."""
        self.create_video_inspection_tab(parent)
        self.video_manager.set_display_label(self.video_label)
        return parent

    def create_video_inspection_tab(self, video_tab: ttk.Frame):
        self.configure_video_grid(video_tab)
        self.create_video_list_frame(video_tab)
        self.create_video_display_frame(video_tab)
        self.create_controls_frame(video_tab)

    def configure_video_grid(self, video_tab):
        video_tab.grid_rowconfigure(0, weight=1)  # Main content
        video_tab.grid_rowconfigure(1, weight=0)  # Controls
        video_tab.grid_columnconfigure(0, weight=1, minsize=300)  # Left side (Video list)
        video_tab.grid_columnconfigure(1, weight=4)  # Right side (Video display)

    def create_video_list_frame(self, video_tab: ttk.Frame):
        list_frame = tk.Frame(video_tab)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        video_listbox_label = tk.Label(list_frame, text="Available Videos")
        video_listbox_label.grid(row=0, column=0, sticky="w")

        self.video_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, width=40)
        self.video_listbox.grid(row=1, column=0, sticky="nswe", pady=5)

        load_raw_video_button = tk.Button(list_frame, text="Load Raw Video", command=self.load_raw_video)
        load_raw_video_button.grid(row=2, column=0, pady=5)

    def create_video_display_frame(self, video_tab: ttk.Frame):
        video_display_frame = tk.Frame(video_tab)
        video_display_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        video_display_frame.grid_rowconfigure(0, weight=1)
        video_display_frame.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_display_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind('<Configure>', self.on_resize_video)

        self.video_start_time_label = tk.Label(video_display_frame, text="")
        self.video_start_time_label.grid(row=1, column=0, sticky="ew")

    def create_controls_frame(self, video_tab: ttk.Frame):
        controls_frame = tk.Frame(video_tab)
        controls_frame.grid(row=1, column=1, sticky="ew", padx=10, pady=10)

        previous_frame_button = tk.Button(controls_frame, text="<<", command=self.previous_frame)
        previous_frame_button.pack(side=tk.LEFT, padx=5)

        play_button = tk.Button(controls_frame, text="Play", command=self.play_video)
        play_button.pack(side=tk.LEFT, padx=5)

        pause_button = tk.Button(controls_frame, text="Pause", command=self.pause_video)
        pause_button.pack(side=tk.LEFT, padx=5)

        next_frame_button = tk.Button(controls_frame, text=">>", command=self.next_frame)
        next_frame_button.pack(side=tk.LEFT, padx=5)

        self.video_scrollbar = tk.Scale(controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_scroll)
        self.video_scrollbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)

        self.video_manager.set_scrollbar(self.video_scrollbar)

    def load_raw_video(self):
        selected_index = self.video_listbox.curselection()
        if not selected_index:
            self.log("No video selected from the list.")
            return

        video_path = os.path.join(self.experiment_manager.folder_videos, self.video_listbox.get(selected_index[0]))

        if not os.path.isfile(video_path):
            self.log(f"Error: Video file does not exist at the path {video_path}")
            return
        self.log(f"Loading raw video: {video_path}")
        if self.video_manager.load_video(video_path):
            self.state_manager.set('video_loaded', True)
            self.state_manager.set('current_video', video_path)
            self.video_manager.set_display_tracking(False)
            total_frames = self.video_manager.get_total_frames()
            self.video_scrollbar.configure(to=total_frames - 1)
            self.video_manager.show_frame(0)
            start_time = self.video_manager.get_datetime_from_file_name(self.video_listbox.get(selected_index[0]))
            self.set_video_start_time(start_time)
            self.update_scrollbar()

    def previous_frame(self):
        self.playing = False
        current_frame = int(self.video_scrollbar.get())
        new_frame = max(current_frame - 1, 0)
        self.video_scrollbar.set(new_frame)
        self.video_manager.previous_frame()

    def next_frame(self):
        self.playing = False
        current_frame = int(self.video_scrollbar.get())
        new_frame = min(current_frame + 1, self.video_scrollbar.cget("to"))
        self.video_scrollbar.set(new_frame)
        self.video_manager.next_frame()

    def play_video(self):
        self.playing = True
        self.video_manager.play_video()

    def pause_video(self):
        self.playing = False
        self.video_manager.pause_video()

    def on_scroll(self, value):
        frame_index = int(value)
        self.video_manager.show_frame(frame_index)

    def on_resize_video(self, event):
        if self.video_manager.cap and self.video_manager.cap.isOpened():
            frame_index = int(self.video_scrollbar.get())
            self.video_manager.show_frame(frame_index)

    def set_video_start_time(self, start_time):
        if start_time:
            formatted_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            self.video_start_time_label.config(text=f"Video Start Time: {formatted_time}")
        else:
            self.video_start_time_label.config(text="")

    def update_ui_after_loading_experiment(self):
        self.set_initial_video_list()

    def fill_video_listbox(self, video_files):
        self.video_listbox.delete(0, tk.END)
        for video in video_files:
            self.video_listbox.insert(tk.END, video)
        self.log("Loaded video list from video folder.")

    def set_initial_video_list(self):
        video_files = [f for f in os.listdir(self.experiment_manager.folder_videos)
                       if f.endswith('.mp4') and not f.startswith('.')]
        self.fill_video_listbox(video_files)

    def update_scrollbar(self):
        total_frames = self.video_manager.get_total_frames()
        self.video_scrollbar.configure(to=total_frames - 1)
        self.video_manager.show_frame(0)