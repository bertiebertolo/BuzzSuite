"""Startup welcome panel for BuzzSuite (workflow redesign).

Shown before any experiment is loaded. There is no module choice here anymore — every
experiment walks the same three linear sections (1 · Setup, 2 · Analysis, 3 · Plotting), and
Plotting lets you choose activity / speaker-zone / swarming plots per run. This panel just lets
the user:
  * start a **New experiment** (jumps straight to the 1 · Setup section), or
  * re-open one of the last 10 experiments from a recents list, or
  * Browse for an experiment_*.json that isn't in the recents list.

On continue it calls ``on_continue(open_recent_path=None, focus_intake=False)`` and removes
itself, at which point the app builds all notebook tabs (Setup/Analysis/Plotting, Dashboard,
H264 converter — always, regardless of any prior selection).
"""
import os
import tkinter as tk
from tkinter import filedialog

from tooltip import add_tooltip


class AssaySelector:
    # One-line descriptions of the three analyses, shown as informational text only.
    MODULES = [
        ('activity',   'Activity',    'Track flying / resting activity from video.'),
        ('swarming',   'BuzzSwarm (Aggregation)', 'Sholl-style r50 spatial aggregation.'),
        ('phonotaxis', 'BuzzPhono (Phonotaxis)',  'Speaker-zone resting fraction.'),
    ]

    def __init__(self, root, on_continue, recent_experiments=None):
        self.root = root
        self.on_continue = on_continue
        self.recent = list(recent_experiments or [])

        self.frame = tk.Frame(root)
        self.frame.pack(fill='both', expand=True)

        tk.Label(self.frame, text="BuzzSuite",
                 font=("TkDefaultFont", 22, "bold")).pack(pady=(34, 2))
        tk.Label(self.frame, text="Mosquito behavioural-assay analysis suite",
                 fg="gray").pack(pady=(0, 18))

        body = tk.Frame(self.frame)
        body.pack(fill='both', expand=True, padx=30)

        # ---- Left: start a new experiment -------------------------------------------------
        left = tk.LabelFrame(body, text="Start", padx=12, pady=10)
        left.pack(side='left', fill='both', expand=True, padx=(0, 12))
        tk.Label(left, text="Every experiment walks the same three sections — Setup, Analysis,\n"
                            "Plotting — and Plotting covers all three analyses below:",
                 fg="gray", justify="left").pack(anchor='w')
        for _key, label, desc in self.MODULES:
            row = tk.Frame(left)
            row.pack(fill='x', anchor='w', pady=3)
            tk.Label(row, text="•  " + label, font=("TkDefaultFont", 10, "bold")).pack(side='left')
            tk.Label(row, text="  " + desc, fg="gray").pack(side='left')

        new_btn = tk.Button(left, text="New experiment  →", command=self._new_experiment,
                            height=2)
        new_btn.pack(fill='x', pady=(14, 2))
        add_tooltip(new_btn, "Open the app on the 1 · Setup section to set up a brand-new "
                             "experiment (create/open, convert/move video in, extract background, "
                             "draw the cage border).")
        cont_btn = tk.Button(self.frame, text="Continue  →", command=self._continue,
                             width=18, height=2)

        # ---- Right: recent experiments ------------------------------------------------
        right = tk.LabelFrame(body, text="Or re-open a recent experiment", padx=12, pady=10)
        right.pack(side='left', fill='both', expand=True)
        if self.recent:
            self.listbox = tk.Listbox(right, height=8, width=38, activestyle='dotbox')
            for r in self.recent:
                self.listbox.insert('end', self._format_recent(r))
            self.listbox.pack(fill='both', expand=True)
            self.listbox.selection_set(0)
            self.listbox.bind("<Double-Button-1>", lambda _e: self._open_selected())
            open_btn = tk.Button(right, text="Open selected", command=self._open_selected)
            open_btn.pack(fill='x', pady=(8, 0))
            add_tooltip(open_btn, "Load the highlighted experiment.")
        else:
            self.listbox = None
            tk.Label(right, text="No recent experiments yet.\nStart a new one on the left.",
                     fg="gray", justify="left").pack(anchor='w', pady=20)

        browse_btn = tk.Button(right, text="Browse for experiment…", command=self._browse_for_experiment)
        browse_btn.pack(fill='x', pady=(4, 0))
        add_tooltip(browse_btn, "Open an experiment_*.json that isn't in the recents list above.")

        cont_btn.pack(pady=(18, 26))
        add_tooltip(cont_btn, "Build the app and load your last experiment (if any).")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _format_recent(entry):
        alias = entry.get('alias') or os.path.basename(entry.get('path', ''))
        ts = entry.get('ts')
        if ts:
            try:
                import time as _time
                return f"{alias}   —   last opened {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(ts))}"
            except Exception:
                pass
        return alias

    # ------------------------------------------------------------------ actions
    def _continue(self):
        self.frame.destroy()
        self.on_continue()

    def _new_experiment(self):
        # Same as continue, but signal the app to land on the 1 · Setup section.
        self.frame.destroy()
        self.on_continue(open_recent_path=None, focus_intake=True)

    def _browse_for_experiment(self):
        # Opens an experiment that isn't in the recents list.
        path = filedialog.askopenfilename(
            title="Select experiment JSON",
            # Extension-only patterns: macOS's native open panel takes file extensions, not
            # globs, so a prefixed pattern like "experiment_*.json" makes Tk hand it a nil and
            # the whole process aborts (NSInvalidArgumentException). Don't narrow this again.
            filetypes=[("Experiment JSON", "*.json")])
        if not path:
            return
        self.frame.destroy()
        self.on_continue(open_recent_path=path)

    def _open_selected(self):
        if not self.listbox:
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        entry = self.recent[sel[0]]
        self.frame.destroy()
        self.on_continue(open_recent_path=entry.get('path'))
