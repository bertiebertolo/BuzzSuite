"""BuzzSuite's Phonotaxis tab: a launcher for BuzzPhono, which runs in its own environment.

All phonotaxis analysis now lives in the BuzzPhono package (buzzphono_env). This tab opens the
BuzzPhono app in the browser, or runs the main analyses on an experiment in the background and
follows its progress (the run's status.json), with a Cancel button. It never imports BuzzPhono.
"""
import os
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

import buzzphono_bridge as bb
from tooltip import add_tooltip

DEFAULT_ANALYSES = "resting_fraction,attraction_index,psth,zt_profile,reliability"


class BuzzphonoLauncherTab(object):
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.log = getattr(ui_manager, "log", print)
        self.proc = None
        self.run_dir = None
        self.cancelling = False
        self.app_proc = None

    # ---------------------------------------------------------------- UI
    def init_buzzphono_tab(self, tab):
        frame = tk.Frame(tab)
        frame.pack(side=tk.TOP, anchor="nw", padx=12, pady=12, fill=tk.X)
        tk.Label(frame, text="BuzzPhono (Phonotaxis)", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        tk.Label(frame, wraplength=520, justify="left", fg="gray",
                 text="Phonotaxis analysis runs in BuzzPhono, which has its own app. Open the app for "
                      "calibration, checks, every analysis and interactive results, or run the main "
                      "analyses on an experiment from here.").pack(anchor="w", pady=(0, 10))
        b = tk.Button(frame, text="Open the BuzzPhono app", command=self.open_app, width=30)
        b.pack(anchor="w", pady=(0, 12))
        add_tooltip(b, "Starts BuzzPhono's app and opens it in your web browser (it stays on this computer).")

        box = tk.LabelFrame(frame, text="Analyse an experiment")
        box.pack(fill=tk.X, pady=(0, 8))
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=6, pady=4)
        self.exp_var = tk.StringVar()
        tk.Entry(row, textvariable=self.exp_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="Browse...", command=self.browse).pack(side=tk.LEFT, padx=4)
        row2 = tk.Frame(box)
        row2.pack(fill=tk.X, padx=6, pady=4)
        tk.Label(row2, text="Analyses:").pack(side=tk.LEFT)
        self.analyses_var = tk.StringVar(value=DEFAULT_ANALYSES)
        e = tk.Entry(row2, textvariable=self.analyses_var, width=60)
        e.pack(side=tk.LEFT, padx=4)
        add_tooltip(e, "Comma-separated, or 'all'. resting_fraction is the headline ON vs OFF result.")
        row3 = tk.Frame(box)
        row3.pack(fill=tk.X, padx=6, pady=4)
        self.run_btn = tk.Button(row3, text="Analyse", command=self.analyse, width=14)
        self.run_btn.pack(side=tk.LEFT)
        self.cancel_btn = tk.Button(row3, text="Cancel", command=self.cancel, state=tk.DISABLED, width=10)
        self.cancel_btn.pack(side=tk.LEFT, padx=4)
        self.report_btn = tk.Button(row3, text="Open report", command=self.open_report, state=tk.DISABLED, width=12)
        self.report_btn.pack(side=tk.LEFT, padx=4)
        self.progress = ttk.Progressbar(box, maximum=1.0, length=420)
        self.progress.pack(fill=tk.X, padx=6, pady=(4, 2))
        self.status_var = tk.StringVar(value="")
        tk.Label(box, textvariable=self.status_var, wraplength=520, justify="left").pack(anchor="w", padx=6, pady=(0, 6))
        self.refresh_experiment()

    def refresh_experiment(self):
        em = getattr(self.ui_manager, "experiment_manager", None)
        folder = getattr(em, "folder_analysis", None) if em is not None else None
        if folder and os.path.isdir(folder) and not self.exp_var.get():
            self.exp_var.set(folder)

    # ---------------------------------------------------------------- actions
    def browse(self):
        d = filedialog.askdirectory(title="Experiment analysis folder (holds tracking_resting/)")
        if d:
            self.exp_var.set(d)

    def open_app(self):
        if self.app_proc is not None and self.app_proc.poll() is None:
            webbrowser.open("http://localhost:8501")
            return
        try:
            log = os.path.join(os.path.expanduser("~"), ".buzzphono_app.log")
            self.app_proc = bb.start(["app"], log)
            self.log("BuzzPhono app starting (log: %s)" % log)
        except bb.BuzzPhonoNotFound as exc:
            messagebox.showerror("BuzzPhono not found", str(exc))

    def analyse(self):
        exp = self.exp_var.get().strip()
        if not exp or not os.path.isdir(exp):
            messagebox.showerror("BuzzPhono", "Choose the experiment's analysis folder first.")
            return
        stamp = time.strftime("%Y%m%d-%H%M%S") + "-%03d" % (int(time.time() * 1000) % 1000)
        run_dir = os.path.join(exp, "plots", "buzzphono", stamp + "-buzzsuite")
        args = ["analyse", exp, "-a", self.analyses_var.get().strip() or DEFAULT_ANALYSES, "--run-dir", run_dir, "-q"]
        try:
            bb.buzzphono_command()  # find BuzzPhono before creating anything
            # Mark the folder as a BuzzPhono run's before process.log goes in: BuzzPhono refuses a
            # non-empty results folder it cannot prove is its own.
            bb.claim_run_dir(run_dir)
            self.proc = bb.start(args, os.path.join(run_dir, "process.log"))
        except bb.BuzzPhonoNotFound as exc:
            messagebox.showerror("BuzzPhono not found", str(exc))
            return
        except OSError as exc:
            messagebox.showerror("BuzzPhono", "Could not create the results folder %s:\n%s" % (run_dir, exc))
            return
        self.run_dir = run_dir
        self.cancelling = False
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.report_btn.config(state=tk.DISABLED)
        self.status_var.set("Starting...")
        self.root.after(1000, self._poll)

    def cancel(self):
        if self.run_dir:
            self.cancelling = True
            try:
                with open(os.path.join(self.run_dir, bb.CANCEL), "w") as fh:
                    fh.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
            except OSError:
                pass  # the folder is gone: the kill below still stops the run
            self.status_var.set("Cancelling...")
            self.root.after(10000, self._kill_if_alive)

    def _kill_if_alive(self):
        if self.proc is not None and self.proc.poll() is None:
            if os.name == "nt":
                import subprocess
                subprocess.call(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"])
            else:
                self.proc.kill()

    def open_report(self):
        if self.run_dir and os.path.isfile(os.path.join(self.run_dir, "report.html")):
            webbrowser.open("file:///" + os.path.join(self.run_dir, "report.html").replace("\\", "/"))

    def _poll(self):
        status = bb.read_status(self.run_dir) or {}
        state = status.get("state", "starting")
        self.progress["value"] = float(status.get("progress", 0.0))
        self.status_var.set(status.get("message", "Starting..."))
        alive = self.proc is not None and self.proc.poll() is None
        if state in ("running", "starting") and alive:
            self.root.after(1000, self._poll)
            return
        if state in ("running", "starting"):
            # The process is gone but never recorded an end (killed after Cancel, crashed, out of memory):
            # one last read in case it finished between the two checks above.
            status = bb.read_status(self.run_dir) or status
            state = status.get("state", state)
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        if state == "done":
            self.status_var.set("Done: %s" % self.run_dir)
            self.report_btn.config(state=tk.NORMAL)
            self.log("BuzzPhono done -> %s" % self.run_dir)
        elif state == "cancelled":
            self.status_var.set("Cancelled; partial results were deleted.")
        elif state == "running" and self.cancelling:
            self.status_var.set("Cancelled: BuzzPhono was stopped before it could tidy up, so partial results "
                                "may remain in %s." % self.run_dir)
        elif state == "running":
            self.status_var.set("Failed: BuzzPhono stopped unexpectedly (crashed or ran out of memory); "
                                "see process.log in %s" % self.run_dir)
        else:
            msg = status.get("error") or "BuzzPhono stopped; see process.log in %s" % self.run_dir
            self.status_var.set("Failed: %s" % msg)
            if status.get("hint"):
                self.status_var.set(self.status_var.get() + "\n" + status["hint"])
