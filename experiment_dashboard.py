"""Multi-experiment dashboard (Step 4).

A persistent panel that runs several experiments' tracking at once, each fully independent: one
``threading.Thread`` per experiment wrapping its own ``multiprocessing.Pool``. Every row is a
card (alias + step text + total progress bar + Start/Cancel); staging (Add to dashboard) does not
start tracking -- press Start on a row or "Start all queued" (see _start_job).

A "Live file progress" panel on the right shows, per running experiment, every segment currently
being tracked with its own progress bar -- parsed from the same log_analysis/<segment>.log files
single_video_analysis already writes (read-only; no change to tracking numerics or log format).
This exists because on_video_done only fires when a whole segment finishes (20-45+ min each per
observed logs), so without it there is a long silent gap between "started" and the first visible
update that reads as "stuck".

Tracking-only: this Dashboard runs tracking (via the shared ``batch_runner`` engine, the same one
Section 2's queue uses) and, right after, the same auto-concatenation into ``analyzed_data.pkl``
Section 2 does. BuzzSwarm and BuzzPhono are not tracking -- they're plotting/analysis that reads
the tracking files this Dashboard produces, exactly like Activity Plotting does; generating their
output lives entirely in their own Plotting sub-tabs, not here.

All work is off the Tk main thread; UI updates are marshalled back with ``after()``.
"""
import os
import re
import statistics
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from tooltip import add_tooltip

_PROGRESS_RE = re.compile(r'Progress:\s*\[[^\]]*\]\s*(\d+)%')
_FILE_POLL_MS = 1500
# A running segment is flagged once it has taken this many times the typical (median) time of the
# segments already finished in the same run -- and at least _SLOW_MIN_SECONDS. Nothing is skipped
# automatically: some dense segments legitimately take hours and are wanted.
_SLOW_FACTOR = 3.0
_SLOW_MIN_SECONDS = 600


def _theme_color(widget, mac_name, fallback):
    """A macOS semantic colour, which follows Light/Dark mode, where Tk knows it; otherwise
    `fallback` (Windows/Linux, where the app is always light so a fixed colour stays readable). The
    fixed dark greys used before were unreadable on a dark macOS window."""
    try:
        widget.winfo_rgb(mac_name)
        return mac_name
    except tk.TclError:
        return fallback


def _friendly_stage(message):
    """Log line -> short stage label. Segmentation logs its reference-building steps last, which
    read as a confusing, truncated 'Built within-video restin…' for the whole segmentation stage."""
    if message.startswith(("Start running segmentation", "Built within-video")):
        return "segmenting"
    return message


def _format_recent(entry):
    """Mirrors AssaySelector._format_recent's label so recents look the same everywhere."""
    alias = entry.get('alias') or os.path.basename(os.path.dirname(entry.get('path', '')))
    ts = entry.get('ts')
    if ts:
        try:
            import time as _time
            return f"{alias}   —   last opened {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(ts))}"
        except Exception:
            pass
    return alias


def _short_segment_label(video_name):
    """'CageAedesAegyptiF_250_test_20260713_120010_00013' -> '20260713_120010_00013' -- drops the
    cage-name prefix (shared by every segment in a job, so it adds no information here) and keeps
    the session date/time/index, which is what actually distinguishes rows in the side panel."""
    parts = video_name.split('_')
    return '_'.join(parts[-3:]) if len(parts) >= 3 else video_name


def _read_log_progress(log_path, since=None):
    """Best-effort parse of the tail of a segment's log_analysis/<name>.log: returns
    (stage_text, percent). (None, 0) means the worker hasn't picked this segment up yet (no log
    file written, or -- with ``since`` -- only an older run's log, since logs are appended to).
    Read-only -- this is the same file single_video_analysis already writes via
    misc_functions.progress_bar; nothing here changes what gets logged or computed."""
    try:
        if since is not None and os.path.getmtime(log_path) < since:
            return None, 0
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return None, 0
    if not lines:
        return "starting…", 0
    percent = None
    stage = ""
    for raw in reversed(lines):
        msg = raw.split(' - ', 1)[-1].strip() if ' - ' in raw else raw.strip()
        if not msg:
            continue
        m = _PROGRESS_RE.search(msg)
        if m:
            if percent is None:
                percent = int(m.group(1))
            continue
        stage = msg
        break
    return (_friendly_stage(stage) or "starting…"), (percent or 0)


class ExperimentDashboard:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.log = ui_manager.log
        self.tab = None
        self.rows_frame = None
        self.file_progress_frame = None
        self.jobs = {}   # alias -> job dict
        self._picker_paths = {}   # combobox label -> experiment_*.json path
        self._poll_scheduled = False

    # ---------------------------------------------------------------- UI
    def init_dashboard_tab(self, tab: ttk.Frame):
        self.tab = tab
        tk.Label(tab, text="Experiment dashboard — track several experiments at once, independently and cancellable",
                 font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 4))

        add_frame = tk.LabelFrame(tab, text="Add an experiment to track alongside the others", padx=10, pady=8)
        add_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        pick_row = tk.Frame(add_frame); pick_row.pack(fill=tk.X, pady=2)
        tk.Label(pick_row, text="Experiment:").pack(side=tk.LEFT)
        self.picker_var = tk.StringVar()
        self.picker_combo = ttk.Combobox(pick_row, textvariable=self.picker_var, width=46, state="readonly")
        self.picker_combo.pack(side=tk.LEFT, padx=(4, 4))
        add_tooltip(self.picker_combo, "Pick any of your last 10 opened experiments.")
        refresh_btn = tk.Button(pick_row, text="Refresh", command=self._refresh_recents)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(refresh_btn, "Reload this list (in case another tab just opened a new experiment).")
        browse_btn = tk.Button(pick_row, text="Browse…", command=self._browse_and_stage)
        browse_btn.pack(side=tk.LEFT)
        add_tooltip(browse_btn, "Pick an experiment_*.json that isn't in the recents list above.")

        btn_row = tk.Frame(add_frame); btn_row.pack(fill=tk.X, pady=(6, 0))
        add_btn = tk.Button(btn_row, text="Add to dashboard", command=self._on_add_clicked)
        add_btn.pack(side=tk.LEFT)
        add_tooltip(add_btn, "Stage the selected experiment as a row below. It does NOT start "
                             "tracking yet -- press \"Start\" on its row (or \"Start all queued\") "
                             "when you're ready. BuzzSwarm/BuzzPhono plots are generated from the "
                             "3 · Plotting tab, not here.")
        start_all_btn = tk.Button(btn_row, text="Start all queued", command=self._start_all_queued)
        start_all_btn.pack(side=tk.LEFT, padx=(6, 0))
        add_tooltip(start_all_btn, "Start tracking every staged row below that hasn't been "
                                   "started yet, all at once.")

        tk.Label(btn_row, text="Worker processes:").pack(side=tk.LEFT, padx=(16, 4))
        from batch_processing_tab_manager import _default_batch_worker_count
        em = getattr(self.ui_manager, 'experiment_manager', None)
        default_workers = _default_batch_worker_count(getattr(em, 'config', None))
        self.worker_count_var = tk.IntVar(value=default_workers)
        worker_spinbox = tk.Spinbox(btn_row, from_=1, to=32, textvariable=self.worker_count_var, width=5)
        worker_spinbox.pack(side=tk.LEFT)
        add_tooltip(worker_spinbox,
                    "How many videos to track in parallel (one CPU process each). More = faster but "
                    "heavier on RAM/CPU. A good default is (number of CPU cores − 1). Applies to "
                    "each experiment started from here after this point (not already-running ones).")

        self._refresh_recents()

        main_area = tk.Frame(tab)
        main_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 8))

        left = tk.Frame(main_area)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="Experiments", font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill=tk.X)
        self.rows_frame = tk.Frame(left)
        self.rows_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        right = tk.LabelFrame(main_area, text="Live file progress", padx=6, pady=6)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.configure(width=430)
        right.pack_propagate(False)
        add_tooltip(right, "Every segment currently being tracked for a running experiment, with "
                           "its own progress bar (parsed from that segment's log). A segment's "
                           "pipeline has several stages (segmentation, resting/moving tracking, "
                           "assembly), each restarting its own 0-100% bar, so the % shown is "
                           "progress within its CURRENT stage, not the whole segment.\n\n"
                           "A segment taking far longer than the others is marked in orange; "
                           "\"Skip\" leaves it out so the rest of the experiment can finish.")

        canvas = tk.Canvas(right, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
        self.file_progress_frame = tk.Frame(canvas)
        self.file_progress_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.file_progress_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        if not self._poll_scheduled:
            self._poll_scheduled = True
            self.root.after(_FILE_POLL_MS, self._poll_file_progress)

    # ---------------------------------------------------------------- add-experiment picker
    def _refresh_recents(self):
        """(Re)populate the picker from the app's shared recent-experiments list (config.json).
        Preserves the current selection if it's still present; a Browse…-added entry that isn't
        in recents survives a refresh too since it's merged back in below."""
        em = getattr(self.ui_manager, 'experiment_manager', None)
        recent = em.get_recent_experiments() if em else []
        keep_label = self.picker_var.get()
        keep_path = self._picker_paths.get(keep_label)

        self._picker_paths = {}
        values = []
        for r in recent:
            path = r.get('path')
            if not path:
                continue
            label = _format_recent(r)
            self._picker_paths[label] = path
            values.append(label)

        if keep_path and keep_label not in self._picker_paths:
            values.insert(0, keep_label)
            self._picker_paths[keep_label] = keep_path

        self.picker_combo['values'] = values
        if keep_label in self._picker_paths:
            self.picker_var.set(keep_label)
        elif values:
            self.picker_var.set(values[0])
        else:
            self.picker_var.set('')

    def _browse_and_stage(self):
        path = filedialog.askopenfilename(
            title="Select experiment JSON",
            # Extension-only patterns: macOS's native open panel takes file extensions, not
            # globs, so a prefixed pattern like "experiment_*.json" makes Tk hand it a nil and
            # the whole process aborts (NSInvalidArgumentException). Don't narrow this again.
            filetypes=[("Experiment JSON", "*.json")])
        if not path:
            return
        alias = os.path.basename(os.path.dirname(path))
        label = f"{alias}   —   browsed"
        values = list(self.picker_combo['values'])
        if label not in values:
            values.insert(0, label)
            self.picker_combo['values'] = values
        self._picker_paths[label] = path
        self.picker_var.set(label)

    def _on_add_clicked(self):
        path = self._picker_paths.get(self.picker_var.get())
        if not path:
            self.log("Dashboard: pick an experiment (recents dropdown or Browse…) first.")
            return
        folder = os.path.dirname(path)
        self.start_experiment(folder)
        em = getattr(self.ui_manager, 'experiment_manager', None)
        if em is not None:
            em.record_recent_experiment(path, alias=os.path.basename(folder))

    # ---------------------------------------------------------------- stage / launch
    def start_experiment(self, folder_analysis):
        """Stage ``folder_analysis`` as a row. Despite the name (kept for the one external
        caller, ``_on_add_clicked``), this no longer starts tracking -- it only adds the row;
        tracking begins when the user presses that row's "Start" button or "Start all queued"
        (see _start_job). Staging-not-running-immediately was a deliberate ask after a session
        where the dashboard's "queued" label never changed for the ~20-45 min a segment actually
        takes to finish (on_video_done only fires on full-segment completion), which read as
        "stuck" and led to repeated relaunches that threw away real in-progress work each time."""
        if not folder_analysis or not os.path.isdir(folder_analysis):
            self.log("Dashboard: invalid experiment folder.")
            return
        alias = os.path.basename(os.path.normpath(folder_analysis))
        existing = self.jobs.get(alias)
        if existing is not None:
            thread = existing.get('thread')
            if thread is not None and thread.is_alive():
                self.log(f"Dashboard: '{alias}' is already running.")
            else:
                self.log(f"Dashboard: '{alias}' is already staged — press Start on its row.")
            return

        step_var = tk.StringVar(value="queued (not started)")
        progress_text_var = tk.StringVar(value="")
        job = {'alias': alias, 'folder': folder_analysis,
               'cancel': None, 'pool': None, 'step_var': step_var,
               'progress_text_var': progress_text_var, 'thread': None,
               'pending_videos': None, 'file_rows': {}}
        self._build_row(job)
        self.jobs[alias] = job
        self.log(f"Dashboard: staged '{alias}' — press Start when ready.")

    def _build_row(self, job):
        row = tk.Frame(self.rows_frame, relief=tk.GROOVE, borderwidth=1)
        row.pack(fill=tk.X, pady=4, padx=2)
        job['row'] = row

        top = tk.Frame(row); top.pack(fill=tk.X, padx=8, pady=(6, 2))
        job['top'] = top
        tk.Label(top, text=job['alias'], font=("TkDefaultFont", 10, "bold"),
                 anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        stop_btn = tk.Button(top, text="Remove", command=lambda: self._cancel(job))
        stop_btn.pack(side=tk.RIGHT)
        add_tooltip(stop_btn, "Before starting: remove this row. While tracking: stop. Segments "
                              "that already finished are kept and the data is built from them.")
        job['stop_btn'] = stop_btn
        start_btn = tk.Button(top, text="Start", command=lambda: self._start_job(job))
        start_btn.pack(side=tk.RIGHT, padx=(0, 6))
        job['start_btn'] = start_btn

        bottom = tk.Frame(row); bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(bottom, textvariable=job['step_var'], anchor="w",
                 fg=_theme_color(bottom, 'systemSecondaryLabelColor', "#444444"),
                 font=("TkDefaultFont", 9)).pack(side=tk.TOP, fill=tk.X)
        prog_row = tk.Frame(bottom); prog_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        prog = ttk.Progressbar(prog_row, mode="determinate", maximum=100)
        prog.pack(side=tk.LEFT, fill=tk.X, expand=True)
        job['prog'] = prog
        tk.Label(prog_row, textvariable=job['progress_text_var'], width=14,
                 anchor="e").pack(side=tk.LEFT, padx=(8, 0))

    def _start_all_queued(self):
        started = 0
        for job in list(self.jobs.values()):
            if job.get('thread') is None:
                self._start_job(job)
                started += 1
        if not started:
            self.log("Dashboard: nothing staged to start.")

    def _start_job(self, job):
        thread = job.get('thread')
        if thread is not None and thread.is_alive():
            return  # already running -- ignore a stale/duplicate click
        if thread is not None:
            self._reset_finished_job(job)   # "Start again" on a finished row
        # Snapshot the "Worker processes" spinbox now, on the main thread (this is a button-click
        # handler) -- _run_activity/run_tracking_job_for_folder run on a background Thread and must
        # not touch a Tk IntVar themselves. Mirrors Section 2's identical snapshot-before-dispatch
        # pattern (batch_processing_tab_manager.py::_run_queue).
        try:
            requested_workers = int(self.worker_count_var.get())
            if requested_workers < 1:
                requested_workers = None
        except Exception:
            requested_workers = None
        job['requested_workers'] = requested_workers
        cancel_event = threading.Event()
        job['cancel'] = cancel_event
        job['started_at'] = time.time()
        job['step_var'].set("starting…")
        start_btn = job.pop('start_btn', None)
        if start_btn is not None:
            start_btn.destroy()
        job['stop_btn'].configure(text="Stop", state=tk.NORMAL)

        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        job['thread'] = t
        t.start()
        self.log(f"Dashboard: started tracking '{job['alias']}'")

    def _cancel(self, job):
        if job.get('thread') is None:
            # Never started -- nothing to terminate, just drop the staged row.
            job['row'].destroy()
            section = job.get('file_section')
            if section is not None:
                section.destroy()
            self.jobs.pop(job['alias'], None)
            self.log(f"Dashboard: removed staged '{job['alias']}'.")
            return
        if not job['thread'].is_alive():
            return  # already finished
        if not messagebox.askyesno(
                "Stop tracking",
                f"Stop tracking '{job['alias']}'?\n\n"
                "Segments that already finished are kept, and the data is built from them.\n"
                "Unfinished segments are tracked if you press \"Start again\" later."):
            return
        job['cancel'].set()
        self._set_step(job, "stopping…")
        pool = job.get('pool')
        if pool is not None:
            try:
                pool.terminate()
            except Exception:
                pass

    # ---------------------------------------------------------------- worker
    def _run_job(self, job):
        try:
            summary = self._run_activity(job)
            prefix = "stopped" if job['cancel'].is_set() else "done"
            self._set_step(job, f"{prefix} — {summary}" if summary else prefix)
        except Exception as exc:
            self._set_step(job, "error")
            self.log(f"Dashboard '{job['alias']}': {exc}")
        finally:
            self.root.after(0, lambda: self._on_job_finished(job))

    def _on_job_finished(self, job):
        """Main thread. Disable Stop and offer "Start again", which tracks whatever is still
        untracked (after a Stop, or a segment you un-skipped); finished and skipped segments are
        not redone."""
        job['stop_btn'].configure(state=tk.DISABLED)
        if job.get('start_btn') is None:
            btn = tk.Button(job['top'], text="Start again", command=lambda: self._start_job(job))
            btn.pack(side=tk.RIGHT, padx=(0, 6))
            add_tooltip(btn, "Track this experiment's remaining untracked segments (not the "
                             "finished or skipped ones), then rebuild its data.")
            job['start_btn'] = btn

    def _reset_finished_job(self, job):
        section = job.pop('file_section', None)
        if section is not None:
            section.destroy()
        job['file_rows'] = {}
        job['pending_videos'] = None
        job['_file_progress_settled'] = False
        job['thread'] = None

    @staticmethod
    def _data_summary(folder, concat_status, failed):
        """One line for the row once a run ends: what the experiment's data now contains."""
        from batch_processing_tab_manager import load_excluded_segments
        skipped = load_excluded_segments(folder)
        prefix = "forward_mosq_tracks_"
        try:
            n_in_data = sum(1 for f in os.listdir(os.path.join(folder, "final_tracking_data"))
                            if f.startswith(prefix) and f[len(prefix):] not in skipped)
        except OSError:
            n_in_data = 0
        n_skipped = len(skipped)
        if concat_status in ('saved', 'up_to_date'):
            parts = [f"data built from {n_in_data} tracked segment(s)"]
        elif concat_status == 'no_data':
            parts = ["no tracked segments yet, so no data"]
        else:
            parts = ["data NOT built (see the log)"]
        if n_skipped:
            parts.append(f"{n_skipped} skipped")
        if failed:
            parts.append(f"{len(failed)} failed (see the log)")
        return ", ".join(parts)

    def _run_activity(self, job):
        """Cancellable multiprocessing.Pool over the experiment's pending videos, via the shared
        batch_runner engine (also used by Section 2's tracking queue) — one execution path."""
        from batch_runner import run_tracking_job_for_folder
        folder = job['folder']
        cfg_mgr = getattr(getattr(self.ui_manager, 'experiment_manager', None), 'config_manager', None)

        def on_step(text):
            self._set_step(job, text)

        def on_pending(video_names):
            job['pending_videos'] = list(video_names)
            self.root.after(0, lambda: self._build_file_section(job))

        failed = []

        def on_video_done(result, idx, total):
            self._set_step(job, f"tracking {idx}/{total} segment(s) done")
            self._set_progress(job, idx / total * 100, idx, total)
            ok = isinstance(result, dict) and result.get('ok')
            if ok and result.get('excluded'):
                self.log(f"Dashboard '{job['alias']}': skipped {result.get('video_name', '?')}")
            if not ok:
                failed.append(result.get('video_name', '?') if isinstance(result, dict) else '?')
                name = result.get('video_name', '?') if isinstance(result, dict) else '?'
                error = str(result.get('error', '')).replace("\n", " ")[:180] if isinstance(result, dict) else ''
                self.log(f"Dashboard '{job['alias']}': ✗ {name} — {error}")

        def on_pool(pool):
            job['pool'] = pool

        result = run_tracking_job_for_folder(
            folder, cfg_mgr, video_names=None,
            on_step=on_step, on_video_done=on_video_done, on_pool=on_pool,
            on_pending=on_pending, cancel=job['cancel'],
            requested_workers=job.get('requested_workers'))

        if result['total']:
            self._set_progress(job, result['done'] / result['total'] * 100,
                               result['done'], result['total'])

        if result['total'] == 0 and not result['skipped_no_background']:
            self._set_step(job, "nothing left to track")
        elif result['total'] == 0 and result['skipped_no_background']:
            self.log(f"Dashboard '{job['alias']}': none of the pending segment(s) have a "
                     "background image (images_mortality/). Run the Setup tab's \"Extract "
                     "Images from Video\" / \"Get Background from Images\" on this experiment "
                     "first, then start it again.")
        elif result['skipped_no_background']:
            self.log(f"Dashboard '{job['alias']}': {len(result['skipped_no_background'])} pending "
                     f"segment(s) had no background image and were skipped; tracked "
                     f"{result['done']}/{result['total']} ready one(s). Run the Setup tab's "
                     "image/background extraction to include the rest.")

        # Keep analyzed_data.pkl (incl. total_pixels_moved) in sync with final_tracking_data/ right
        # after tracking finishes -- same auto-concatenation the Section 2 tracking queue does, so
        # there is no manual "Concatenate and Save Data" step. Own try/except: a concatenation
        # failure must not propagate past _run_activity and crash _run_job.
        from batch_processing_tab_manager import concatenate_and_save_experiment_data
        self._set_step(job, "concatenating…")
        config = getattr(getattr(self.ui_manager, 'experiment_manager', None), 'config', None)
        concat_status = 'failed'
        try:
            concat_status = concatenate_and_save_experiment_data(
                folder, config=config,
                log_fn=lambda msg, a=job['alias']: self.log(f"Dashboard '{a}': {msg}"))
        except Exception as exc:
            self.log(f"Dashboard '{job['alias']}': concatenation failed: {exc}")
        return self._data_summary(folder, concat_status, failed)

    # ---------------------------------------------------------------- live per-file progress
    def _build_file_section(self, job):
        """Called once per job, right after the pending-video list is known (on_pending) -- builds
        the widgets once; _poll_file_progress only updates their values afterwards (no
        rebuild/flicker per tick)."""
        if job.get('file_section') is not None or self.file_progress_frame is None:
            return
        video_names = job.get('pending_videos') or []
        if not video_names:
            return

        sec = tk.Frame(self.file_progress_frame, relief=tk.GROOVE, borderwidth=1)
        sec.pack(fill=tk.X, pady=(0, 6), padx=2)
        job['file_section'] = sec

        muted = _theme_color(sec, 'systemSecondaryLabelColor', "#666666")
        warn = _theme_color(sec, 'systemSystemOrangeColor', "#c75b00")
        tk.Label(sec, text=job['alias'], font=("TkDefaultFont", 9, "bold"),
                 anchor="w").pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(sec, textvariable=job['progress_text_var'], anchor="w",
                 font=("TkDefaultFont", 8), fg=muted).pack(fill=tk.X, padx=6)

        from batch_processing_tab_manager import load_excluded_segments
        already_skipped = load_excluded_segments(job['folder'])
        holder = tk.Frame(sec)
        holder.pack(fill=tk.X, padx=6, pady=(2, 6))
        for name in video_names:
            r = tk.Frame(holder); r.pack(fill=tk.X, pady=1)
            tk.Label(r, text=_short_segment_label(name), width=17, anchor="w",
                     font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
            bar = ttk.Progressbar(r, mode="determinate", length=90, maximum=100)
            bar.pack(side=tk.LEFT, padx=(2, 4))
            skipped = name in already_skipped
            skip_btn = tk.Button(r, text="Undo" if skipped else "Skip", font=("TkDefaultFont", 8),
                                 command=lambda n=name: self._toggle_skip(job, n))
            skip_btn.pack(side=tk.RIGHT)
            add_tooltip(skip_btn, "Leave this segment out: it won't be tracked and won't be in "
                                  "this experiment's data, so the rest can finish. Typical for "
                                  "the first segment of a recording, while the camera is still "
                                  "focusing. \"Undo\" takes it back.")
            stage_var = tk.StringVar(value="skipped" if skipped else "queued")
            stage_lbl = tk.Label(r, textvariable=stage_var, anchor="w",
                                 font=("TkDefaultFont", 8), fg=muted)
            stage_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            job['file_rows'][name] = {'bar': bar, 'stage_var': stage_var, 'stage_lbl': stage_lbl,
                                      'skip_btn': skip_btn, 'muted': muted, 'warn': warn,
                                      '_color': muted, '_skipped': skipped}

    def _toggle_skip(self, job, name):
        """Main thread. Skip / un-skip one segment of a running (or finished) experiment."""
        widgets = job['file_rows'].get(name)
        if widgets is None or widgets.get('_done'):
            return
        skipping = not widgets.get('_skipped')
        if skipping and not messagebox.askyesno(
                "Skip segment",
                f"Skip {_short_segment_label(name)}?\n\n"
                "It won't be tracked and won't be included in this experiment's data. The other "
                "segments carry on, and the data is built from them when they finish."):
            return
        from batch_processing_tab_manager import set_segment_excluded
        try:
            set_segment_excluded(job['folder'], name, excluded=skipping)
        except OSError as exc:
            messagebox.showerror("Skip segment", f"Could not save this change:\n{exc}")
            return
        widgets['_skipped'] = skipping
        widgets['skip_btn'].configure(text="Undo" if skipping else "Skip")
        widgets['stage_var'].set("skipped" if skipping else "queued")
        if widgets['_color'] != widgets['muted']:
            widgets['stage_lbl'].configure(fg=widgets['muted'])
            widgets['_color'] = widgets['muted']
        thread = job.get('thread')
        running = thread is not None and thread.is_alive()
        if skipping:
            self.log(f"Dashboard '{job['alias']}': skipping {name}.")
        elif running:
            self.log(f"Dashboard '{job['alias']}': {name} will be tracked after all.")
        else:
            self.log(f"Dashboard '{job['alias']}': {name} un-skipped — press \"Start again\" "
                     "to track it.")

    def _poll_file_progress(self):
        """Windows note: re-setting a themed ttk.Progressbar's 'value' -- even to a number it
        already has -- repaints it via the native "vista" theme (Tk_GetPixmap ->
        CreateDIBSection), and that native draw call leaks one GDI handle per repaint on this
        Tk/Windows combo. A big job (100+ segment rows) polled every _FILE_POLL_MS for its full,
        possibly multi-hour, runtime does that enough times to exhaust the process's ~10k GDI
        handle quota, which surfaces as an unrelated-looking "Tk_GetPixmap: Error from
        CreateDIBSection - Not enough memory resources" dialog. Two guards below keep the redraw
        count bounded: only touch a widget when its displayed value/text actually changes, and
        stop touching a job's rows at all once its worker thread has finished (previously nothing
        reset job['thread'] to None on completion, so finished jobs' rows were re-painted forever
        for the rest of the app session)."""
        for job in list(self.jobs.values()):
            thread = job.get('thread')
            rows = job.get('file_rows')
            if thread is None or not rows or job.get('_file_progress_settled'):
                continue
            folder = job['folder']
            final_dir = os.path.join(folder, "final_tracking_data")
            log_dir = os.path.join(folder, "log_analysis")
            now = time.time()
            durations = [w['_t_done'] - w['_t_start'] for w in rows.values()
                         if w.get('_t_done') and w.get('_t_start')]
            typical = statistics.median(durations) if durations else None
            for name, widgets in rows.items():
                if widgets.get('_done') or widgets.get('_skipped'):
                    continue
                if os.path.isfile(os.path.join(final_dir, f"forward_mosq_tracks_{name}")):
                    widgets['_done'] = True
                    if widgets.get('_t_start'):
                        widgets['_t_done'] = now
                    widgets['skip_btn'].configure(state=tk.DISABLED)
                    if widgets['bar']['value'] != 100:
                        widgets['bar']['value'] = 100
                    if widgets['stage_var'].get() != "done":
                        widgets['stage_var'].set("done")
                    if widgets['_color'] != widgets['muted']:
                        widgets['stage_lbl'].configure(fg=widgets['muted'])
                        widgets['_color'] = widgets['muted']
                    continue
                stage, percent = _read_log_progress(os.path.join(log_dir, f"{name}.log"),
                                                    since=job.get('started_at'))
                if stage is not None and not widgets.get('_t_start'):
                    widgets['_t_start'] = now
                slow = (typical is not None and widgets.get('_t_start') is not None and
                        now - widgets['_t_start'] > max(_SLOW_FACTOR * typical, _SLOW_MIN_SECONDS))
                if stage is None:
                    new_stage = "queued"
                elif slow:
                    new_stage = "much slower than others"
                else:
                    new_stage = stage[:42]
                new_percent = 0 if stage is None else percent
                if widgets['bar']['value'] != new_percent:
                    widgets['bar']['value'] = new_percent
                if widgets['stage_var'].get() != new_stage:
                    widgets['stage_var'].set(new_stage)
                color = widgets['warn'] if slow else widgets['muted']
                if widgets['_color'] != color:
                    widgets['stage_lbl'].configure(fg=color)
                    widgets['_color'] = color
            if not thread.is_alive():
                job['_file_progress_settled'] = True
        self.root.after(_FILE_POLL_MS, self._poll_file_progress)

    # ---- thread-safe UI updates (marshalled to the main thread) ----
    def _set_step(self, job, text):
        self.root.after(0, lambda: job['step_var'].set(text))

    def _set_progress(self, job, value, done=None, total=None):
        def _update():
            job['prog'].configure(value=value)
            pv = job.get('progress_text_var')
            if pv is None:
                return
            if done is not None and total:
                pv.set(f"{done}/{total} ({value:.0f}%)")
            else:
                pv.set(f"{value:.0f}%")
        self.root.after(0, _update)
