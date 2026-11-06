"""Intake wizard: raw video -> ready-to-run experiment, embedded as the "Experiment & Video"
sub-tab of the app's 1 · Setup section.

Four steps, always visible:
  A  Source video: browse a folder, scan .h264/.mp4, parse {Cage}_{YYYYMMDD}_{HHMMSS}_{NNNNN}.
  B  Dates: pick which recording dates to include.
  C  Experiment setup: Experiment / Incubator / Species / Sex / Cage name -> build the Analysis
     tree under the resolved data root and write experiment_{alias}.json (via ExperimentManager).
     Creates/opens the experiment's Recording/ folder but does not populate it yet.
  D  Convert / move into experiment: now that the experiment (and its Recording/ folder) exists,
     lossless .h264 -> .mp4 remux (reuses h264_converter.convert_gui) writes straight into it, and
     any segment that was already .mp4 (already converted in an older session, or a raw .mp4
     recording that never needed conversion) is moved straight in too (a same-drive rename, not a
     copy, so no duplicate data is left behind). Runs directly against the experiment's own video
     folder, not a separate copy step — requires Step C to have run first.

`reset_form()` clears the form (source folder, dates, Step C fields) so the next experiment can be
prepared right away, without restarting the app — the just-created experiment stays saved and
reachable from the recents list / Experiment Dashboard. It's triggered from the "Set up new
experiment" button at the end of the Background & Cage Border sub-tab (setup_tab_manager.py),
not from this tab — that button only makes sense once an experiment is fully set up (video in,
background extracted, borders drawn), which this tab alone can't guarantee.

Tracking (2 · Analysis), speaker-zone drawing and plots (3 · Plotting) live in their own
sections — this tab only gets video in and the experiment folder created.

No hardcoded data paths: the root comes from path_utils.resolve_data_root(), which auto-detects
the usual mount points but can be overridden — Step C has a "Browse…" next to the data-root label
for picking a different drive/folder (e.g. one that mounts somewhere unexpected); a 'Buzzwatch'
subfolder is created there if missing, and that path is saved to config.json as
`data_root_override` so it persists across restarts. "Auto-detect" clears it.
Everything else is user selection. Long work (conversion) runs on a worker thread and reports via
a queue polled with after(), so the Tk main thread stays responsive.
"""
import os
import re
import shutil
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta

import path_utils
from tooltip import add_tooltip

# {Cage}_{YYYYMMDD}_{HHMMSS}_{NNNNN} with a video extension
_SEG_RE = re.compile(r'^(?P<cage>.+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<seg>\d{5})\.(?P<ext>h264|mp4)$',
                     re.IGNORECASE)

# Each segment is a fixed 20-minute chunk (see CLAUDE.md's data-folder-structure notes).
_SEGMENT_MINUTES = 20

# Mirrors assay_selector.AssaySelector.MODULES' keys/labels. Duplicated (not imported) to keep
# the welcome screen and this tab independent modules with no cross-import.
_MODULE_LABELS = {
    'activity': 'Activity',
    'swarming': 'BuzzSwarm (Aggregation)',
    'phonotaxis': 'BuzzPhono (Phonotaxis)',
}


def _segment_actual_dates(date_str, time_str, seg_str):
    """The real calendar date(s) (00:00-00:00) this segment's recording interval touches.

    {YYYYMMDD}_{HHMMSS} in the filename is fixed at the recording SESSION's start and shared by
    every segment in that session, even one running past midnight — e.g. a session starting at
    12:00 has segment 00000 at ~12:00 that day but segment 00036 at ~00:00 the *next* day. Since
    segments are a fixed 20-minute grid anchored to the session's (arbitrary) start time, not to
    midnight, a calendar-day boundary almost never falls exactly between two segments — it falls
    *inside* one. That segment is returned as touching BOTH calendar dates, so selecting either
    neighbouring day pulls it in too: ticking one full day yields ~72 segments starting that day
    plus (usually) one extra carried over from the tail of the previous day, so the day's footage
    has no gap at 00:00. Returns a set of 1 or 2 "YYYYMMDD" strings."""
    start = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S") + timedelta(minutes=_SEGMENT_MINUTES * int(seg_str))
    end = start + timedelta(minutes=_SEGMENT_MINUTES, seconds=-1)   # inclusive end, avoids a
    return {start.strftime("%Y%m%d"), end.strftime("%Y%m%d")}       # spurious 3rd day at exact midnight


class IntakeWizardTabManager:
    ASSAY_TYPES = ["Phonotaxis", "Lightctrl", "Nosound"]

    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.tab = None
        self.msgq = queue.Queue()
        self.scanned = []          # list of dicts: {name, path, cage, date, time, seg, ext}
        self._converting = False
        self._last_auto_expname = None

    # ---------------------------------------------------------------- UI build
    def init_intake_tab(self, tab: ttk.Frame):
        self.tab = tab
        outer = tk.Frame(tab)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        tk.Label(outer, text="Intake wizard — raw video to ready experiment",
                 font=("TkDefaultFont", 13, "bold")).pack(anchor="w")

        # ---- Step A: source video ----
        a = tk.LabelFrame(outer, text="A · Source video folder")
        a.pack(fill=tk.X, pady=6)
        self.src_var = tk.StringVar()
        row = tk.Frame(a); row.pack(fill=tk.X, padx=6, pady=4)
        self.src_entry = tk.Entry(row, textvariable=self.src_var)
        src_entry = self.src_entry
        src_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        add_tooltip(src_entry, "Folder containing the raw camera recordings (.h264/.mp4). "
                    "Point this at the SD-card or download folder for this run, then press Scan.")
        tk.Button(row, text="Browse…", command=self.browse_source).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="Scan", command=self.scan_source).pack(side=tk.LEFT)
        self.scan_status = tk.Label(a, text="No folder scanned yet.", fg="gray")
        self.scan_status.pack(anchor="w", padx=6, pady=(0, 4))

        # ---- Step B: dates ----
        b = tk.LabelFrame(outer, text="B · Recording dates to include")
        b.pack(fill=tk.X, pady=6)
        self.date_vars = {}     # date str "YYYYMMDD" -> tk.BooleanVar, rebuilt on each Scan
        self.dates_frame = tk.Frame(b)
        self.dates_frame.pack(fill=tk.X, padx=6, pady=4, anchor="w")

        # ---- Step C: experiment setup ----
        # Runs BEFORE convert/move (Step D) on purpose: it creates the experiment's Recording/
        # folder, which Step D then writes/copies straight into instead of converting into the raw
        # source folder and copying out afterward.
        step_c = tk.LabelFrame(outer, text="C · Experiment setup")
        step_c.pack(fill=tk.X, pady=6)
        grid = tk.Frame(step_c); grid.pack(fill=tk.X, padx=6, pady=4)
        assay_lbl = tk.Label(grid, text="Experiment")
        assay_lbl.grid(row=0, column=0, sticky="w")
        self.assay_var = tk.StringVar()
        assay_entry = tk.Entry(grid, textvariable=self.assay_var, width=24)
        assay_entry.grid(row=0, column=1, sticky="w", padx=6, pady=2)
        _assay_help = ("The behavioural assay this recording belongs to. Sets the "
                       "Analysis/<Experiment>/ subfolder the experiment is filed under. Type "
                       "your own, or a common value: " + ", ".join(self.ASSAY_TYPES) + ".")
        add_tooltip(assay_lbl, _assay_help)
        add_tooltip(assay_entry, _assay_help)
        loc_lbl = tk.Label(grid, text="Incubator")
        loc_lbl.grid(row=1, column=0, sticky="w")
        self.location_var = tk.StringVar()
        loc_combo = ttk.Combobox(grid, textvariable=self.location_var, width=21,
                                 values=["Cakung", "Bangkok", "Penzance"])
        loc_combo.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        _loc_help = ("Which incubator/site this was recorded in. Pick a preset or type your "
                     "own. Groups experiments recorded at the same place under "
                     "Analysis/<Experiment>/<Incubator>/.")
        add_tooltip(loc_lbl, _loc_help)
        add_tooltip(loc_combo, _loc_help)
        species_lbl = tk.Label(grid, text="Species")
        species_lbl.grid(row=2, column=0, sticky="w")
        self.species_var = tk.StringVar()
        species_entry = tk.Entry(grid, textvariable=self.species_var, width=24)
        species_entry.grid(row=2, column=1, sticky="w", padx=6, pady=2)
        _species_help = ("Mosquito species/strain (e.g. AedesAegyptiLiv475M). Feeds the "
                         "auto-suggested cage name below.")
        add_tooltip(species_lbl, _species_help)
        add_tooltip(species_entry, _species_help)
        sex_lbl = tk.Label(grid, text="Sex")
        sex_lbl.grid(row=3, column=0, sticky="w")
        self.sex_var = tk.StringVar()
        sex_combo = ttk.Combobox(grid, textvariable=self.sex_var, width=21,
                                 values=["F", "M"])
        sex_combo.grid(row=3, column=1, sticky="w", padx=6, pady=2)
        _sex_help = ("Sex of the mosquitoes in this experiment. Pick a preset or type your own. "
                    "Feeds the auto-suggested cage name below.")
        add_tooltip(sex_lbl, _sex_help)
        add_tooltip(sex_combo, _sex_help)
        name_lbl = tk.Label(grid, text="Cage name")
        name_lbl.grid(row=4, column=0, sticky="w")
        self.expname_var = tk.StringVar()
        name_entry = tk.Entry(grid, textvariable=self.expname_var, width=32)
        name_entry.grid(row=4, column=1, sticky="w", padx=6, pady=2)
        _name_help = ("Unique folder name for this cage/experiment. Auto-suggested from Species "
                      "+ Sex + the dates ticked in Step B — edit it freely if you want something "
                      "else. Becomes the experiment directory holding tracking data and plots.")
        add_tooltip(name_lbl, _name_help)
        add_tooltip(name_entry, _name_help)
        dataroot_row = tk.Frame(step_c)
        dataroot_row.pack(fill=tk.X, padx=6, anchor="w")
        self.dataroot_label = tk.Label(dataroot_row, text=self._dataroot_text(), fg="gray")
        self.dataroot_label.pack(side=tk.LEFT)
        browse_root_btn = tk.Button(dataroot_row, text="Browse…", command=self._browse_data_root)
        browse_root_btn.pack(side=tk.LEFT, padx=(8, 0))
        add_tooltip(browse_root_btn, "Pick a drive or folder to hold the data — e.g. a different "
                    "drive than the one auto-detected above. A 'Buzzwatch' folder (with the "
                    "Recording/ and Analysis/ trees) is created there if it doesn't exist yet. "
                    "Remembered for next time.")
        reset_root_btn = tk.Button(dataroot_row, text="Auto-detect", command=self._reset_data_root)
        reset_root_btn.pack(side=tk.LEFT, padx=(4, 0))
        add_tooltip(reset_root_btn, "Forget the manually-picked data root and go back to "
                    "auto-detecting it (E:\\Buzzwatch / /Volumes/Mosquito2/Buzzwatch).")
        tk.Button(step_c, text="Create / open experiment", command=self.create_experiment).pack(anchor="w", padx=6, pady=4)

        # Keep the Cage-name suggestion in sync with Species/Sex/selected dates (each date
        # checkbox also calls _maybe_autofill_expname directly), without clobbering a name the
        # user has typed themselves.
        self.species_var.trace_add('write', lambda *_a: self._maybe_autofill_expname())
        self.sex_var.trace_add('write', lambda *_a: self._maybe_autofill_expname())

        # ---- Step D: convert / move into the experiment ----
        step_d = tk.LabelFrame(outer, text="D · Convert / move video into the experiment")
        step_d.pack(fill=tk.X, pady=6)
        crow = tk.Frame(step_d); crow.pack(fill=tk.X, padx=6, pady=4)
        convert_btn = tk.Button(crow, text="Convert / move selected dates", command=self.convert_selected)
        convert_btn.pack(side=tk.LEFT)
        add_tooltip(convert_btn, "Needs the experiment created/opened above (Step C) first. "
                    ".h264 segments are losslessly remuxed straight into the experiment's video "
                    "folder; segments that are already .mp4 (already converted earlier, or a raw "
                    ".mp4 recording that never needed conversion) are moved straight in too (a "
                    "same-drive rename, no duplicate left behind). Already-present segments are "
                    "skipped.")
        self.convert_progress = ttk.Progressbar(crow, mode="determinate", length=240)
        self.convert_progress.pack(side=tk.LEFT, padx=8)
        self.convert_status = tk.Label(step_d, text="", fg="gray")
        self.convert_status.pack(anchor="w", padx=6, pady=(0, 4))

        self.root.after(150, self._pump_queue)

    def reset_form(self):
        """Clear Steps A-D back to a blank state so the next experiment can be prepared
        immediately, without restarting the app. Does not touch any already-created experiment."""
        self.src_var.set("")
        self.scanned = []
        for w in self.dates_frame.winfo_children():
            w.destroy()
        self.date_vars = {}
        self.scan_status.config(text="No folder scanned yet.", fg="gray")
        self.assay_var.set("")
        self.location_var.set("")
        self.species_var.set("")
        self.sex_var.set("")
        self.expname_var.set("")
        self._last_auto_expname = None
        self.convert_progress['value'] = 0
        self.convert_status.config(text="")
        self.src_entry.focus_set()
        self.log("Intake: form cleared — ready to set up a new experiment.")

    # ---------------------------------------------------------------- Step A
    def browse_source(self):
        path = filedialog.askdirectory(title="Select the raw recording folder")
        if path:
            self.src_var.set(path)

    def scan_source(self):
        folder = self.src_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Intake", "Please choose a valid source folder.")
            return
        self.scanned = []
        for name in sorted(os.listdir(folder)):
            if name.startswith('.'):
                continue
            m = _SEG_RE.match(name)
            if not m:
                continue
            g = m.groupdict()
            self.scanned.append({
                'name': name, 'path': os.path.join(folder, name),
                'cage': g['cage'], 'date': g['date'], 'time': g['time'],
                'seg': g['seg'], 'ext': g['ext'].lower(),
                'actual_dates': _segment_actual_dates(g['date'], g['time'], g['seg']),
            })
        for w in self.dates_frame.winfo_children():
            w.destroy()
        self.date_vars = {}
        if not self.scanned:
            self.scan_status.config(
                text="No recordings matching {Cage}_YYYYMMDD_HHMMSS_NNNNN(.h264/.mp4) were found "
                     "here — is this the right folder?", fg="red")
            return
        dates = sorted(set().union(*(f['actual_dates'] for f in self.scanned)))
        for i, dt in enumerate(dates):
            var = tk.BooleanVar(value=True)   # default: all dates ticked
            self.date_vars[dt] = var
            label = f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]}"
            chk = tk.Checkbutton(self.dates_frame, text=label, variable=var,
                                 command=self._maybe_autofill_expname)
            chk.grid(row=i // 6, column=i % 6, sticky="w", padx=4, pady=2)
        n_h264 = sum(1 for f in self.scanned if f['ext'] == 'h264')
        n_mp4 = sum(1 for f in self.scanned if f['ext'] == 'mp4')
        self.scan_status.config(
            text=f"{len(self.scanned)} segments across {len(dates)} date(s): {n_h264} .h264, {n_mp4} .mp4.",
            fg="gray")
        self._maybe_autofill_expname()

    def _selected_dates(self):
        return {dt for dt, var in self.date_vars.items() if var.get()}

    def _suggested_expname(self):
        """Build a cage-name suggestion from Species + Sex + the dates ticked in Step B."""
        species = self.species_var.get().strip().replace(' ', '')
        sex = self.sex_var.get().strip().replace(' ', '')
        dates = sorted(self._selected_dates())
        parts = [p for p in (species, sex) if p]
        if dates:
            parts.append(dates[0] if len(dates) == 1 else f"{dates[0]}_{dates[-1]}")
        return "_".join(parts)

    def _maybe_autofill_expname(self):
        """Refresh Cage name from the suggestion, unless the user has typed their own value."""
        suggestion = self._suggested_expname()
        if not suggestion:
            return
        current = self.expname_var.get().strip()
        if not current or current == self._last_auto_expname:
            self.expname_var.set(suggestion)
            self._last_auto_expname = suggestion

    def _donor_candidates(self, to_convert):
        """Scanned .h264 segments that could donate SPS/PPS headers to `to_convert`.

        Only the first segment of a recording normally carries them, so a date selection that
        excludes it has no donor among its own files even though one sits right there in the
        source folder. Restricted to the same recording session (cage + session start) as the
        segments being converted: a different session may have been recorded at another
        resolution, and prepending its headers would silently produce a misparsed video.
        Lowest segment index first, since that is where the headers live.
        """
        sessions = set((f['cage'], f['date'], f['time']) for f in to_convert)
        same_session = [f for f in self.scanned
                        if f['ext'] == 'h264' and (f['cage'], f['date'], f['time']) in sessions]
        same_session.sort(key=lambda f: f['seg'])
        return [f['path'] for f in same_session]

    # ---------------------------------------------------------------- Step D
    def convert_selected(self):
        """Convert/move the selected dates straight into the experiment's own video folder.
        Requires Step C (Create/open experiment) to have run first, so the destination exists.

        .h264 segments are remuxed directly into it; segments that are already .mp4 (either a raw
        .mp4 recording that never needed conversion, or a leftover from an older session that
        converted into the source folder) are moved in as-is (same-drive rename, no ffmpeg, no
        duplicate data left at the source). Segments already present in the destination are
        skipped either way."""
        if self._converting:
            self.log("Intake: a conversion is already running.")
            return
        dates = self._selected_dates()
        src = self.src_var.get().strip()
        if not src or not dates:
            messagebox.showerror("Intake", "Scan a folder and select at least one date first.")
            return
        recording_dir = getattr(self.experiment_manager, 'folder_videos', None)
        if not recording_dir or not getattr(self.experiment_manager, 'folder_analysis', None):
            messagebox.showerror("Intake", "Create/open the experiment first (Step C above), "
                                  "then convert/move its video here.")
            return
        os.makedirs(recording_dir, exist_ok=True)

        in_scope = [f for f in self.scanned if f['actual_dates'] & dates]
        seen_bases = set()
        to_convert = []          # .h264 entries with no .mp4 anywhere yet -> need ffmpeg
        to_copy = []             # (src_path, dest_path) already .mp4 -> moved in place (rename)
        already = 0
        for f in in_scope:
            base = os.path.splitext(f['name'])[0]
            if base in seen_bases:
                continue
            seen_bases.add(base)
            dest = os.path.join(recording_dir, base + ".mp4")
            if os.path.isfile(dest):
                already += 1
                continue
            mp4_in_src = os.path.join(src, base + ".mp4")
            if os.path.isfile(mp4_in_src):
                to_copy.append((mp4_in_src, dest))
            elif f['ext'] == 'h264':
                to_convert.append(f)

        if already:
            self.log(f"Intake: {already} segment(s) already present in {recording_dir} — skipping.")

        copied = 0
        for mp4_path, dest in to_copy:
            shutil.move(mp4_path, dest)
            copied += 1
        if copied:
            self.log(f"Intake: moved {copied} already-.mp4 segment(s) into {recording_dir}.")

        if not to_convert:
            if copied or already:
                if copied:
                    self._reload_experiment()  # so list_video_name/list_videos_files see the new files
                self.convert_status.config(text="All videos are converted/moved.", fg="gray")
                messagebox.showinfo("Intake", "All videos are converted/moved into:\n"
                                     f"{recording_dir}\n\n" + self._counts_line(copied, 0, already))
            else:
                self.convert_status.config(text="Nothing to convert/move for the selected dates.",
                                           fg="gray")
            return

        # Remembered here (not available inside _pump_queue's own scope) so the "all done" popup
        # fired once the background thread finishes can report the full picture, not just this
        # round's ffmpeg conversion count.
        self._convert_recording_dir = recording_dir
        self._convert_copied = copied
        self._convert_already = already
        self._convert_ok = self._convert_fail = 0

        out_dir = recording_dir
        self._converting = True
        self._convert_error = False
        self.convert_status.config(text="Preparing conversion…", fg="gray")
        self.convert_progress['value'] = 0

        def work():
            try:
                from h264_converter import convert_gui
                if not convert_gui.ffmpeg_available():
                    self.msgq.put(("cerror",
                                   "ffmpeg was not found on your PATH or bundled with the app, so "
                                   ".h264 files cannot be converted.\n\n"
                                   "Install ffmpeg (e.g. 'brew install ffmpeg' on macOS) or add it "
                                   "to your PATH, then try again."))
                    return
                files = [f['path'] for f in to_convert]
                blob, header_src = convert_gui.find_header_blob(files)
                if blob is None:
                    # Only the first segment of a recording normally carries the SPS/PPS stream
                    # headers, so a date selection that leaves it out has no donor among its own
                    # files. Borrow one from elsewhere in the same session rather than failing.
                    blob, header_src = convert_gui.find_header_blob(
                        self._donor_candidates(to_convert))
                if blob is None:
                    self.msgq.put(("cerror",
                                   "None of the recordings in this folder carry the SPS/PPS stream "
                                   "headers needed to rebuild the video, so these .h264 segments "
                                   "cannot be converted.\n\nThose headers normally live in the "
                                   "first segment of the recording (..._00000.h264) \u2014 check "
                                   "that it was copied across together with the rest."))
                    return
                fps = convert_gui.detect_fps(header_src or files[0])
                convert_gui.convert_worker(src, out_dir, False, blob, header_src, fps, self.msgq,
                                            files=files)
            except Exception as exc:
                self.msgq.put(("cerror", f"Conversion failed:\n{exc}"))
            finally:
                self.msgq.put(("cdone", None))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _counts_line(copied, converted, already):
        """One human-readable clause per nonzero category, e.g. '3 converted, 2 moved.'"""
        parts = []
        if converted:
            parts.append(f"{converted} converted from .h264")
        if copied:
            parts.append(f"{copied} moved (already .mp4)")
        if already:
            parts.append(f"{already} already there from before")
        return (", ".join(parts) + ".") if parts else "Nothing to do."

    def _pump_queue(self):
        try:
            while True:
                kind, payload = self.msgq.get_nowait()
                if kind == "progress":
                    try:
                        done, total = payload
                        self.convert_progress['maximum'] = max(1, total)
                        self.convert_progress['value'] = done
                        self.convert_status.config(text=f"Converting… {done}/{total}")
                    except Exception:
                        pass
                elif kind in ("log", "status", "cstatus"):
                    self.convert_status.config(text=str(payload), fg="gray")
                    self.log(f"Intake/convert: {payload}")
                elif kind == "cerror":
                    # Hard failure (ffmpeg missing / conversion crash): show a modal dialog on the
                    # main thread, not just a status line a non-expert might miss.
                    self._convert_error = True
                    self.convert_status.config(text=str(payload).splitlines()[0], fg="red")
                    self.log(f"Intake/convert ERROR: {payload}")
                    messagebox.showerror("H264 → MP4 conversion", str(payload))
                elif kind == "done":
                    # From convert_worker itself: (ok, skip, fail, error_msg).
                    try:
                        ok, _skip, fail, _err = payload
                        self._convert_ok, self._convert_fail = ok, fail
                    except Exception:
                        pass
                elif kind == "cdone":
                    # From work()'s finally — fires exactly once per Convert/move click, whether it
                    # succeeded, partially failed, or errored out; this is where the "all done"
                    # popup belongs.
                    self._converting = False
                    if getattr(self, "_convert_error", False):
                        self.convert_status.config(text="Conversion stopped (see the error above).",
                                                   fg="red")
                    else:
                        if getattr(self, "_convert_ok", 0) or getattr(self, "_convert_copied", 0):
                            self._reload_experiment()  # pick up the newly-added video files
                        if getattr(self, "_convert_fail", 0):
                            self.convert_status.config(
                                text=f"{self._convert_fail} segment(s) failed to convert.", fg="red")
                            messagebox.showwarning(
                                "Intake", f"{self._convert_fail} segment(s) failed to convert — see the "
                                f"log for details.\n{self._convert_ok} converted successfully.")
                        else:
                            self.convert_status.config(text="All videos are converted/moved.", fg="gray")
                            messagebox.showinfo(
                                "Intake", "All videos are converted/moved into:\n"
                                f"{getattr(self, '_convert_recording_dir', '')}\n\n" +
                                self._counts_line(getattr(self, '_convert_copied', 0),
                                                  getattr(self, '_convert_ok', 0),
                                                  getattr(self, '_convert_already', 0)))
        except queue.Empty:
            pass
        self.root.after(150, self._pump_queue)

    def _reload_experiment(self):
        """Refresh the loaded experiment after Step D adds video, so list_video_name /
        list_videos_files (scanned once at construction, in buzzwatch_experiment_analysis.__init__)
        reflect what's now on disk — without this, "Extract Images from Video" etc. would keep
        using the stale, empty-at-creation-time listing until the user manually reopens."""
        folder = getattr(self.experiment_manager, 'folder_analysis', None)
        alias = getattr(self.experiment_manager, 'experiment_alias', None)
        if not folder or not alias:
            return
        json_path = os.path.join(folder, f"experiment_{alias}.json")
        if not os.path.isfile(json_path):
            return
        update_ui = getattr(self.ui_manager, 'update_ui_after_loading_experiment', None)
        try:
            self.experiment_manager.load_experiment_from_json(json_path, update_ui_func=update_ui)
        except Exception as exc:
            self.log(f"Intake: could not refresh experiment after convert/move: {exc}")

    # ---------------------------------------------------------------- Step C
    def _data_root_override(self):
        cfg = getattr(self.experiment_manager, 'config', None)
        return cfg.get('data_root_override') if isinstance(cfg, dict) else None

    def _set_data_root_override(self, path):
        """Persist (or clear, if path is falsy) the manually-picked data root to config.json so
        it survives app restarts, then refresh the Step C label."""
        cfg = getattr(self.experiment_manager, 'config', None)
        if isinstance(cfg, dict):
            if path:
                cfg['data_root_override'] = path
            else:
                cfg.pop('data_root_override', None)
            self.experiment_manager.save_config(cfg)
        if self.dataroot_label is not None:
            self.dataroot_label.config(text=self._dataroot_text())

    def _browse_data_root(self):
        """Ask for the drive/folder that should HOLD the Buzzwatch data folder, not the Buzzwatch
        folder itself — a 'Buzzwatch' subfolder (with empty Recording/ and Analysis/ trees, same
        shape as the auto-detected E:\\Buzzwatch) is created there if missing, and that becomes
        the override."""
        current = path_utils.resolve_data_root(self._data_root_override())
        initial = os.path.dirname(current) if current else ""
        if not initial or not os.path.isdir(initial):
            initial = os.path.expanduser("~")
        picked = filedialog.askdirectory(
            title="Select the drive/folder where the Buzzwatch data folder should be created",
            initialdir=initial)
        if not picked:
            return
        buzzwatch_dir = os.path.join(picked, "Buzzwatch")
        try:
            os.makedirs(os.path.join(buzzwatch_dir, "Recording"), exist_ok=True)
            os.makedirs(os.path.join(buzzwatch_dir, "Analysis"), exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Intake", f"Could not create the Buzzwatch folder at:\n{buzzwatch_dir}\n\n{exc}")
            return
        self._set_data_root_override(buzzwatch_dir)
        self.log(f"Intake: data root set to {buzzwatch_dir}")

    def _reset_data_root(self):
        self._set_data_root_override(None)
        self.log("Intake: data root reset to auto-detect.")

    def _dataroot_text(self):
        root = path_utils.resolve_data_root(self._data_root_override())
        return f"Data root: {root}  →  Analysis/{{Experiment}}/{{Incubator}}/{{CageName}}"

    def create_experiment(self):
        """Create (or open) the experiment's folder tree only — this no longer copies/converts any
        video. That now happens in Step D (convert_selected), run afterward against this
        experiment's own folder_videos, once it exists."""
        assay = self.assay_var.get().strip()
        location = self.location_var.get().strip()
        expname = self.expname_var.get().strip()
        if not (assay and location and expname):
            messagebox.showerror("Intake", "Experiment, incubator and cage name are required.")
            return
        data_root = path_utils.resolve_data_root(self._data_root_override())
        if not data_root:
            messagebox.showerror("Intake", "Could not resolve the data root (is the drive mounted?).")
            return
        settings_yml = os.path.join(path_utils.get_package_root(), "buzzwatch_track_settings.yml")
        if not os.path.isfile(settings_yml):
            messagebox.showerror("Intake", f"Default settings file missing:\n{settings_yml}")
            return
        update_ui = getattr(self.ui_manager, 'update_ui_after_loading_experiment', None)
        # ExperimentManager builds root/{name}/{cage}/{batch}; map -> Analysis/{Assay}/{Location}/{Experiment}
        # (alias = {cage}_{batch} = {Location}_{Experiment}).
        recording_dir = os.path.join(data_root, "Recording", assay, location, expname)
        analysis_dir = os.path.join(data_root, "Analysis", assay, location, expname)
        existing_json = os.path.join(analysis_dir, f"experiment_{location}_{expname}.json")
        # Duplicate-name guard: re-running Create over an existing experiment would overwrite its
        # per-experiment settings YAML (cage borders + custom config) with the defaults. Offer to
        # OPEN the existing experiment instead (non-destructive), or cancel so the user can rename.
        if os.path.isfile(existing_json):
            if not messagebox.askyesno(
                    "Experiment already exists",
                    f"An experiment named '{expname}' already exists at:\n{analysis_dir}\n\n"
                    "Open it? This keeps its existing tracking data, cage borders and settings.\n\n"
                    "Choose No to cancel and give this experiment a different name instead."):
                self.log("Intake: create cancelled — name already exists (rename to make a new one).")
                return
            try:
                self.experiment_manager.load_experiment_from_json(existing_json, update_ui_func=update_ui)
                self.log(f"Intake: opened existing experiment at {self.experiment_manager.folder_analysis}")
                self._pretick_completed()
                messagebox.showinfo("Intake", f"Experiment opened:\n{self.experiment_manager.folder_analysis}\n\n"
                                     "Use Step D below to convert/move its video in.")
            except Exception as exc:
                messagebox.showerror("Intake", f"Could not open the existing experiment:\n{exc}")
            return
        try:
            self.experiment_manager.initialize_new_experiment(
                root_folder_videos=os.path.join(data_root, "Recording"),
                root_folder_analysis=os.path.join(data_root, "Analysis"),
                experiment_name=assay,
                cage_name=location,
                batch_name=expname,
                settings_file=settings_yml,
                update_ui_func=update_ui,
                folder_videos=recording_dir,
            )
            self.log(f"Intake: experiment ready at {self.experiment_manager.folder_analysis}")
            self._pretick_completed()
            messagebox.showinfo("Intake", f"Experiment created:\n{self.experiment_manager.folder_analysis}\n\n"
                                 "Use Step D below to convert/move its video in.")
        except Exception as exc:
            messagebox.showerror("Intake", f"Could not create the experiment:\n{exc}")

    def _pretick_completed(self):
        """Informational only: note which analyses already have output for this experiment, so
        the user knows what 2 · Analysis / 3 · Plotting will redo vs. skip. No longer tied to a
        single "active module" — every experiment can run all three now."""
        folder = getattr(self.experiment_manager, 'folder_analysis', None)
        if not folder:
            return
        done = self.experiment_manager.detect_completed_analyses(folder)
        done_labels = [_MODULE_LABELS.get(key, key) for key, is_done in done.items() if is_done]
        if done_labels:
            self.log("Intake: already has output for " + ", ".join(done_labels) +
                     " — re-running it in 2 · Analysis / 3 · Plotting will redo it.")
