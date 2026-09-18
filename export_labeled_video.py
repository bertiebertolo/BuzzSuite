#!/usr/bin/env python3
"""Standalone: export ONE labelled tracking video from N already-tracked segments, concatenated
in the given order. Read-only against the experiment folder and its Recording-tree videos --
writes only the requested output .mp4.

Overlays (reusing VideoManager.overlay_tracking's exact per-track drawing logic, so the dots
match what the app's own video playback would show):
  - flying (blue) / resting (red) circle per tracked mosquito, per frame
  - a wall-clock timestamp, taken from the segment's own tracking data (mt.time_stamp), not
    guessed from the filename
  - a SPEAKER: ON / SPEAKER: OFF indicator -- ON only when the minute-of-hour is in [40, 50),
    OFF for every other minute (a simpler, deliberately different rule than BuzzPhono's own
    stimulus-window plotting, which additionally distinguishes a separate 28-38 "OFF" band)

Usage:
    python export_labeled_video.py <folder_analysis> <video_name> [<video_name> ...] -o out.mp4

    <folder_analysis> is the experiment's Analysis folder (contains final_tracking_data/ and an
    experiment_*.json, which is read to resolve the matching Recording-tree video folder).
    <video_name> is a segment basename with no extension, e.g.
    CageAedesAegyptiF_250_test_20260710_120009_00052 -- must already have tracking output in
    final_tracking_data/ (i.e. tracking has finished for it).

Example (the 3 segments from today's Repulsion run, in order):
    python export_labeled_video.py \\
        /Volumes/Mosquito2/Buzzwatch/Analysis/Repulsion/Penzance/Aedes_aegypti_F_250hz_20260711_20260714 \\
        CageAedesAegyptiF_250_test_20260710_120009_00052 \\
        CageAedesAegyptiF_250_test_20260710_120009_00053 \\
        CageAedesAegyptiF_250_test_20260710_120009_00054 \\
        -o repulsion_00052_54_labeled.mp4
"""
import argparse
import json
import os
import pickle
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_processing_tab_manager  # noqa: F401 -- import side effect patches legacy numpy._core pickles
from video_manager import VideoManager

_ON_START_MIN = 40
_ON_END_MIN = 50


def _tracking_pkl_path(folder_analysis, video_name):
    return os.path.join(folder_analysis, "final_tracking_data", f"forward_mosq_tracks_{video_name}")


def _load_tracking(folder_analysis, video_name):
    path = _tracking_pkl_path(folder_analysis, video_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No tracking output for '{video_name}' yet (expected {path}). "
            "Tracking must finish for this segment before it can be exported.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_folder_videos(folder_analysis):
    for name in os.listdir(folder_analysis):
        if name.startswith("experiment_") and name.endswith(".json"):
            with open(os.path.join(folder_analysis, name)) as f:
                data = json.load(f)
            folder_videos = data.get("folder_videos")
            if folder_videos:
                return folder_videos
    raise FileNotFoundError(f"No experiment_*.json with a folder_videos entry found in {folder_analysis}")


def _speaker_on(ts):
    minute_of_hour = ts.minute + ts.second / 60.0
    return _ON_START_MIN <= minute_of_hour < _ON_END_MIN


def _draw_time_and_speaker_overlay(frame, ts):
    if ts is None:
        cv2.putText(frame, "time unknown", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        return
    cv2.putText(frame, ts.strftime("%Y-%m-%d %H:%M:%S"), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    is_on = _speaker_on(ts)
    label = "SPEAKER: ON" if is_on else "SPEAKER: OFF"
    color = (0, 255, 0) if is_on else (0, 0, 255)
    cv2.putText(frame, label, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def export_labeled_video(folder_analysis, video_names, output_path,
                          speed_min=None, speed_max=None, log=print):
    folder_videos = _resolve_folder_videos(folder_analysis)
    vm = VideoManager(log)
    if speed_min is not None and speed_max is not None:
        vm.set_speed_filter(speed_min, speed_max)

    writer = None
    total_written = 0
    try:
        for video_name in video_names:
            mp4_path = os.path.join(folder_videos, video_name + ".mp4")
            if not os.path.isfile(mp4_path):
                raise FileNotFoundError(f"Video file not found: {mp4_path}")

            vm.mosquito_tracks = _load_tracking(folder_analysis, video_name)
            vm._track_speed_cache = {}
            time_stamps = getattr(vm.mosquito_tracks, "time_stamp", None)

            cap = cv2.VideoCapture(mp4_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {mp4_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}")

            log(f"{video_name}: writing frames...")
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = vm.overlay_tracking(frame, frame_idx)
                ts = time_stamps[frame_idx] if time_stamps is not None and frame_idx < len(time_stamps) else None
                _draw_time_and_speaker_overlay(frame, ts)
                writer.write(frame)
                frame_idx += 1
                total_written += 1
            cap.release()
            log(f"{video_name}: {frame_idx} frames written.")
    finally:
        if writer is not None:
            writer.release()

    log(f"Done. {total_written} total frames -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export one labelled (flying/resting + time + speaker ON/OFF) tracking "
                     "video from N already-tracked segments, concatenated in order.")
    parser.add_argument("folder_analysis", help="Experiment's Analysis folder (contains final_tracking_data/)")
    parser.add_argument("videos", nargs="+", help="Segment basenames (no extension), in output order")
    parser.add_argument("-o", "--output", required=True, help="Output .mp4 path")
    parser.add_argument("--speed-min", type=float, default=None,
                         help="Optional: only draw tracks with average speed >= this (px/frame)")
    parser.add_argument("--speed-max", type=float, default=None,
                         help="Optional: only draw tracks with average speed <= this (px/frame)")
    args = parser.parse_args()
    export_labeled_video(args.folder_analysis, args.videos, args.output,
                          speed_min=args.speed_min, speed_max=args.speed_max)


if __name__ == "__main__":
    main()
