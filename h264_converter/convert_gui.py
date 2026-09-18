#!/usr/bin/env python3
"""
H264 -> MP4 batch converter (GUI).

Double-click "Convert H264 to MP4.bat" (or run this file) to open a small
window: pick an input folder of .h264 files and an output folder, click
Convert, and watch the progress bar.

Conversion is a lossless REMUX (no re-encode), so the output keeps the exact
same video specs. These recordings are split into segments where only the
FIRST segment carries the SPS/PPS stream headers; the rest start on an IDR
keyframe but have no headers, so they cannot be remuxed on their own. This app
extracts the SPS/PPS header blob from a header-bearing segment and prepends it
(via ffmpeg's "concat:" protocol) to every headerless segment before remuxing.

Requires: Python 3 (tkinter, standard) and ffmpeg/ffprobe on PATH (or placed
next to this file / in an "ffmpeg\\bin" subfolder).
"""
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# --------------------------------------------------------------------------
# Locate ffmpeg/ffprobe: next to this file, in ./ffmpeg/bin, or on PATH.
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_tool(name):
    exe = name + (".exe" if os.name == "nt" else "")
    plat_dir = "win" if os.name == "nt" else "mac"
    # Prefer the per-OS bundled binary (ffmpeg/bin/{win,mac}/), then a flat
    # ffmpeg/bin/, then next to this file, and finally PATH. The per-OS subfolder
    # is checked first because that is how the binaries are actually bundled.
    for cand in (os.path.join(APP_DIR, "ffmpeg", "bin", plat_dir, exe),
                 os.path.join(APP_DIR, "ffmpeg", "bin", exe),
                 os.path.join(APP_DIR, exe)):
        if os.path.isfile(cand):
            return cand
    return name  # fall back to PATH lookup


FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")

# On Windows, prevent subprocesses from flashing console windows.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Built-in fallback SPS/PPS header (Annex-B). Used ONLY when the chosen folder
# has no header-bearing segment and the user opts in via the "use default?"
# popup. Captured from this camera's recordings -- H.264, 960x720, 25 fps.
# Every headerless segment from the same camera config shares these exact
# parameter sets, so prepending this blob lets them be remuxed with no donor.
DEFAULT_HEADER = bytes.fromhex(
    "0000000127640028ac2b40780b742000000300200000065c14000f4240002f"
    "af37bdc03c489a800000000128ee025cb0"
)
DEFAULT_FPS = "25/1"
DEFAULT_DESC = "960x720, 25 fps"


# --------------------------------------------------------------------------
# H.264 stream inspection (unchanged logic from convert_h264_to_mp4.py).
# --------------------------------------------------------------------------
def nal_scan(data):
    """Yield (offset, nal_type) for each Annex-B NAL start code in data."""
    i, n = 0, len(data)
    while i < n - 4:
        if data[i] == 0 and data[i + 1] == 0 and (
            data[i + 2] == 1 or (data[i + 2] == 0 and data[i + 3] == 1)
        ):
            sc = 3 if data[i + 2] == 1 else 4
            yield i, data[i + sc] & 0x1F
            i += sc
        else:
            i += 1


def inspect(path, limit=2_000_000):
    """Return (has_sps_pps_header, header_blob_bytes_or_None)."""
    with open(path, "rb") as f:
        head = f.read(limit)
    first_vcl = None
    saw_sps = saw_pps = False
    for off, t in nal_scan(head):
        if t == 7:
            saw_sps = True
        elif t == 8:
            saw_pps = True
        elif t in (1, 5):  # first VCL (slice) NAL
            first_vcl = off
            break
    if saw_sps and saw_pps and first_vcl:
        return True, head[:first_vcl]
    return False, None


def find_header_blob(files):
    """Find a donor segment that carries SPS+PPS and return its header blob."""
    for path in files:
        ok, blob = inspect(path)
        if ok:
            return blob, path
    return None, None


def detect_fps(path, default="25"):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate", "-of",
             "default=nokey=1:noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=120,
            creationflags=CREATE_NO_WINDOW,
        ).stdout.strip()
        if "/" in out:
            num, den = out.split("/")
            if int(den) != 0 and int(num) != 0:
                return out  # ffmpeg accepts "25/1"
    except Exception:
        pass
    return default


def ffmpeg_available():
    try:
        subprocess.run([FFMPEG, "-version"], capture_output=True,
                       creationflags=CREATE_NO_WINDOW, timeout=30)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Drive-level lock: two conversions racing on the same physical disk cause seek
# contention that slows both down far more than running them one at a time (a lossless
# remux is I/O-bound, not CPU-bound, so parallelism on one disk only hurts). This makes
# BuzzSuite serialize conversions per physical drive, even across separate app instances.
# --------------------------------------------------------------------------
_LOCK_STALE_SECONDS = 8 * 60 * 60  # generously longer than any realistic conversion job


def _volume_root(path):
    """Return the filesystem root containing `path` -- the drive letter root on Windows,
    the mount point on macOS -- so the lock is scoped per physical drive on either OS."""
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


class _DriveLock(object):
    """A plain directory-creation mutex: os.mkdir is atomic on both Windows and macOS, so
    this needs no extra dependency. Scoped to `target_path`'s drive/volume root."""

    def __init__(self, target_path, status_cb=None):
        self.lock_dir = os.path.join(_volume_root(target_path), ".buzzsuite_convert.lock")
        self.status_cb = status_cb or (lambda m: None)

    def __enter__(self):
        announced = False
        while True:
            try:
                os.mkdir(self.lock_dir)
                return self
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self.lock_dir)
                except OSError:
                    age = 0
                if age > _LOCK_STALE_SECONDS:
                    try:
                        os.rmdir(self.lock_dir)
                    except OSError:
                        pass
                    continue
                if not announced:
                    self.status_cb("Waiting: another conversion is using this drive...")
                    announced = True
                time.sleep(2)

    def __exit__(self, *exc_info):
        try:
            os.rmdir(self.lock_dir)
        except OSError:
            pass
        return False


# --------------------------------------------------------------------------
# Conversion worker (runs in a background thread; talks to the GUI via a
# thread-safe queue of ("log"/"progress"/"status"/"done", payload) messages).
# --------------------------------------------------------------------------
def convert_worker(in_dir, out_dir, overwrite, blob, header_src, fps, msgq, files=None):
    """Remux .h264 files to .mp4. The SPS/PPS `blob` and `fps` are resolved on the
    GUI thread beforehand (from a header-bearing segment, or the built-in default
    the user opted into), so this worker just converts.

    `files`, if given, restricts conversion to exactly that list of .h264 paths
    (e.g. only the dates a caller selected); otherwise every .h264 in `in_dir` is
    converted (the standalone converter's whole-folder behaviour)."""
    def log(m):    msgq.put(("log", m))
    def status(m): msgq.put(("status", m))

    try:
        os.makedirs(out_dir, exist_ok=True)
        log(f"Output folder: {out_dir}")

        if files is None:
            files = sorted(
                os.path.join(in_dir, f) for f in os.listdir(in_dir)
                if f.lower().endswith(".h264") and not f.startswith(".")
            )
        else:
            files = sorted(files)
        if not files:
            log(f"No .h264 files found in {in_dir}")
            msgq.put(("done", (0, 0, 0, "No .h264 files found.")))
            return

        log(f"Found {len(files)} .h264 file(s).")
        log(f"Header blob ({len(blob)} bytes) from: {header_src}")
        log(f"Frame rate: {fps}")

        hdr = tempfile.NamedTemporaryFile(suffix=".h264", delete=False)
        hdr.write(blob)
        hdr.close()

        ok = skip = fail = 0
        total = len(files)
        msgq.put(("progress", (0, total)))
        try:
            with _DriveLock(out_dir, status_cb=status):
                for i, path in enumerate(files, 1):
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(out_dir, base + ".mp4")
                    tag = f"[{i}/{total}] {base}.mp4"
                    status(f"Converting {tag}")

                    if os.path.exists(out_path) and not overwrite:
                        log(f"SKIP (exists): {tag}")
                        skip += 1
                        msgq.put(("progress", (i, total)))
                        continue

                    has_hdr, _ = inspect(path)
                    src = path if has_hdr else f"concat:{hdr.name}|{path}"

                    cmd = [FFMPEG, "-nostdin", "-y", "-probesize", "100M",
                           "-analyzeduration", "100M", "-r", fps, "-i", src, "-c",
                           "copy", "-movflags", "+faststart", out_path]
                    log(f"CONVERT: {tag}" + ("" if has_hdr else "  (+header)"))

                    r = subprocess.run(cmd, capture_output=True, text=True,
                                       stdin=subprocess.DEVNULL,
                                       creationflags=CREATE_NO_WINDOW)
                    if r.returncode == 0 and os.path.exists(out_path) \
                            and os.path.getsize(out_path) > 0:
                        ok += 1
                    else:
                        fail += 1
                        log(f"  FAILED: {os.path.basename(path)}")
                        tail = r.stderr.strip().splitlines()
                        if tail:
                            log("    " + tail[-1])
                    msgq.put(("progress", (i, total)))
        finally:
            os.unlink(hdr.name)

        log("")
        log(f"Done. Converted: {ok}  Skipped: {skip}  Failed: {fail}")
        msgq.put(("done", (ok, skip, fail, None)))
    except Exception as e:
        log(f"ERROR: {e}")
        msgq.put(("done", (0, 0, 0, str(e))))


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.msgq = queue.Queue()
        self.worker = None

        # When embedded as a notebook tab, `root` is a ttk.Frame (no title/geometry/minsize).
        # Only configure the top-level window chrome when we actually own a Tk/Toplevel window.
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("H264 -> MP4 Converter")
            root.geometry("640x470")
            root.minsize(560, 430)

        pad = {"padx": 12, "pady": 4}

        # Input folder
        tk.Label(root, text="Input folder (contains .h264 files):").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad)
        self.in_var = tk.StringVar()
        tk.Entry(root, textvariable=self.in_var).grid(
            row=1, column=0, sticky="we", padx=(12, 4), pady=2)
        tk.Button(root, text="Browse...", width=11,
                  command=self.browse_in).grid(row=1, column=1, padx=(4, 12), pady=2)

        # Output folder
        tk.Label(root, text="Output folder (.mp4 files go here):").grid(
            row=2, column=0, columnspan=2, sticky="w", **pad)
        self.out_var = tk.StringVar()
        tk.Entry(root, textvariable=self.out_var).grid(
            row=3, column=0, sticky="we", padx=(12, 4), pady=2)
        tk.Button(root, text="Browse...", width=11,
                  command=self.browse_out).grid(row=3, column=1, padx=(4, 12), pady=2)

        # Overwrite + Convert
        self.overwrite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="Overwrite existing .mp4 files",
                       variable=self.overwrite_var).grid(
            row=4, column=0, sticky="w", padx=12, pady=(8, 2))

        self.convert_btn = tk.Button(root, text="Convert", width=14, height=2,
                                     command=self.start_convert)
        self.convert_btn.grid(row=5, column=0, sticky="w", padx=12, pady=6)

        # Progress + status
        self.progress = ttk.Progressbar(root, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=2, sticky="we",
                           padx=12, pady=(8, 2))
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status_var, anchor="w").grid(
            row=7, column=0, columnspan=2, sticky="we", padx=12, pady=2)

        # Log
        frame = tk.Frame(root)
        frame.grid(row=8, column=0, columnspan=2, sticky="nsew",
                   padx=12, pady=(4, 12))
        self.log = tk.Text(frame, height=8, wrap="word", state="disabled",
                           bg="white")
        sb = tk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(8, weight=1)

        if not ffmpeg_available():
            self.status_var.set("WARNING: ffmpeg not found - see README.txt")
            self._log("ffmpeg/ffprobe were not found. Install ffmpeg and add it "
                      "to PATH, or place ffmpeg.exe + ffprobe.exe next to this "
                      "app (or in an 'ffmpeg\\bin' subfolder).")

        self.root.after(100, self.pump_queue)

    # -- logging / queue --
    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def pump_queue(self):
        try:
            while True:
                kind, payload = self.msgq.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    done, total = payload
                    self.progress["maximum"] = total
                    self.progress["value"] = done
                elif kind == "done":
                    ok, skip, fail, err = payload
                    self.convert_btn.configure(state="normal")
                    if err:
                        self.status_var.set(err)
                        messagebox.showerror("Conversion stopped", err)
                    else:
                        self.status_var.set(
                            f"Done. Converted: {ok}  Skipped: {skip}  Failed: {fail}")
                        box = messagebox.showwarning if fail else messagebox.showinfo
                        box("Conversion complete",
                            f"Converted: {ok}\nSkipped: {skip}\nFailed: {fail}\n\n"
                            f"Output: {self.out_var.get()}")
        except queue.Empty:
            pass
        self.root.after(100, self.pump_queue)

    # -- browse --
    def browse_in(self):
        d = filedialog.askdirectory(title="Select folder containing .h264 files",
                                    initialdir=self.in_var.get() or APP_DIR)
        if d:
            d = os.path.normpath(d)
            self.in_var.set(d)
            if not self.out_var.get():
                self.out_var.set(os.path.join(os.path.dirname(d),
                                              os.path.basename(d) + "mp4"))

    def browse_out(self):
        d = filedialog.askdirectory(title="Select output folder for .mp4 files",
                                    initialdir=self.out_var.get() or APP_DIR)
        if d:
            self.out_var.set(os.path.normpath(d))

    # -- convert --
    def start_convert(self):
        if self.worker and self.worker.is_alive():
            return
        in_dir = self.in_var.get().strip()
        if not in_dir or not os.path.isdir(in_dir):
            messagebox.showwarning("Invalid input", "Please select a valid input folder.")
            return
        out_dir = self.out_var.get().strip()
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(in_dir),
                                   os.path.basename(in_dir) + "mp4")
            self.out_var.set(out_dir)
        if not ffmpeg_available():
            messagebox.showerror(
                "ffmpeg not found",
                "ffmpeg.exe was not found.\n\nInstall ffmpeg and add it to PATH, "
                "or place ffmpeg.exe + ffprobe.exe next to this app (or in an "
                "'ffmpeg\\bin' subfolder).")
            return

        # Resolve the SPS/PPS header up front (so we can ask the user about a
        # fallback before the conversion thread starts).
        files = sorted(
            os.path.join(in_dir, f) for f in os.listdir(in_dir)
            if f.lower().endswith(".h264") and not f.startswith(".")
        )
        if not files:
            messagebox.showwarning(
                "No .h264 files", f"No .h264 files were found in:\n\n{in_dir}")
            return

        self.status_var.set("Scanning for SPS/PPS stream headers...")
        self.root.update_idletasks()
        blob, donor = find_header_blob(files)
        if blob is None:
            use_default = messagebox.askyesno(
                "No stream header found",
                "No segment with SPS/PPS stream headers was found in this "
                "folder, so the video's resolution/format can't be read from "
                "the files themselves.\n\n"
                f"Use the built-in default header ({DEFAULT_DESC})?\n\n"
                "Click Yes to convert using the default header, or No to "
                "cancel.")
            if not use_default:
                self.status_var.set("Cancelled - no stream header found.")
                return
            blob, header_src, fps = DEFAULT_HEADER, f"built-in default ({DEFAULT_DESC})", DEFAULT_FPS
        else:
            header_src, fps = os.path.basename(donor), detect_fps(donor)

        self.log.configure(state="normal"); self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        self.convert_btn.configure(state="disabled")
        self.worker = threading.Thread(
            target=convert_worker,
            args=(in_dir, out_dir, self.overwrite_var.get(),
                  blob, header_src, fps, self.msgq),
            daemon=True)
        self.worker.start()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
