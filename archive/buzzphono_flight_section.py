"""BuzzPhono tab: "Flight phonotaxis" section (controls + inline results).

Runs ``buzzphono_legacy.flight_phonotaxis.run_flight_phonotaxis`` on the experiment selected in the BuzzPhono
tab's list: does the tone make the mosquitoes fly more (arousal), and does their flight go to the speaker
wall more than to the other three walls? Uses the tab's ON/OFF stimulus windows. The run happens on a
worker thread (which itself uses a process pool for the per-chunk reading) under a cancellable progress
dialog; the worker never touches Tk, it reports through a queue that the main thread drains every 100 ms
(same pattern as the BuzzSwarm tab). Results: a key-results table and the figures, drawn inline with
FigureCanvasTkAgg; PNGs, CSVs and a README are written to <exp>/plots/flight_phonotaxis/.

Heavy imports (pandas, matplotlib, the analysis package) happen only when Run is clicked / results shown.
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

from tooltip import add_tooltip

TITLE = "BuzzPhono flight phonotaxis"


def _fmt(v, fmt="%.3g"):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v) if v not in (None, "") else ""
    if f != f:          # NaN
        return ""
    return fmt % f


class FlightPhonotaxisSection:
    CONFIG_KEY = 'buzzphono_flight_params'

    def __init__(self, tab_manager):
        self.tm = tab_manager
        self.root = tab_manager.root
        self.ui_manager = tab_manager.ui_manager
        self.log = tab_manager.log
        self._running = False
        self._progress = None
        self._result = None
        self._canvas = None
        self._fig = None
        self.notebook = None
        self.results_frame = None

    # ================================================================ controls
    def build_controls(self, parent):
        saved = self._load_saved()
        frame = tk.LabelFrame(parent, text="Flight phonotaxis (selected experiment)")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 8))
        tk.Label(frame, text="Arousal and where the flight goes: speaker wall vs the other three walls. "
                             "Uses the stimulus windows above. Needs a speaker zone plus three control zones.",
                 fg="gray", wraplength=270, justify="left").grid(row=0, column=0, columnspan=4, sticky="w",
                                                                  padx=4, pady=(2, 4))
        r = 1
        self.from_var = tk.StringVar(value="")
        self.to_var = tk.StringVar(value="")
        tk.Label(frame, text="From").grid(row=r, column=0, sticky="w", padx=4)
        e_from = tk.Entry(frame, textvariable=self.from_var, width=17)
        e_from.grid(row=r, column=1, columnspan=3, sticky="w", padx=2, pady=1)
        add_tooltip(e_from, "Start of the analysis window, e.g. 2026-10-11 or 2026-10-11 05:00. Leave both "
                            "From and To blank to use all recorded (tracked) data.")
        r += 1
        tk.Label(frame, text="To").grid(row=r, column=0, sticky="w", padx=4)
        e_to = tk.Entry(frame, textvariable=self.to_var, width=17)
        e_to.grid(row=r, column=1, columnspan=3, sticky="w", padx=2, pady=1)
        add_tooltip(e_to, "End of the analysis window (exclusive): 2026-10-13 means up to midnight at the "
                          "start of 13 Oct. Blank = to the end of the recorded data.")
        r += 1
        self.light_start_var = tk.StringVar(value=str(saved.get('light_start', 5)))
        self.light_end_var = tk.StringVar(value=str(saved.get('light_end', 17)))
        tk.Label(frame, text="Light hours").grid(row=r, column=0, sticky="w", padx=4)
        s1 = tk.Spinbox(frame, from_=0, to=23, width=4, textvariable=self.light_start_var)
        s1.grid(row=r, column=1, sticky="w", padx=2, pady=1)
        tk.Label(frame, text="to").grid(row=r, column=2)
        s2 = tk.Spinbox(frame, from_=1, to=24, width=4, textvariable=self.light_end_var)
        s2.grid(row=r, column=3, sticky="w", padx=2, pady=1)
        for w in (s1, s2):
            add_tooltip(w, "Clock hours when the cage lights are on (start to end). Results are split into light "
                           "and dark phases, and dawn / day / dusk / night. Typical: 5 to 17 (lights on 05:00, "
                           "off 17:00).")
        r += 1
        self.mult_var = tk.StringVar(value=str(saved.get('flyer_mult', 1.0)))
        tk.Label(frame, text="Flyer threshold x").grid(row=r, column=0, sticky="w", padx=4)
        e_m = tk.Entry(frame, textvariable=self.mult_var, width=6)
        e_m.grid(row=r, column=1, sticky="w", padx=2, pady=1)
        add_tooltip(e_m, "Which moving tracks count as real flight: at least 10 points and a mean step of at "
                         "least 4 px/frame, scaled to the camera (cage width / 780 px), times this multiplier. "
                         "1.0 = the validated rule. Try 0.5 and 2 to check that the results do not depend on it; "
                         "other values are saved in a flyer_x<value> subfolder and compared in "
                         "sensitivity_flyer_threshold.csv.")
        r += 1
        self.median_var = tk.StringVar(value=str(saved.get('min_median', 0.0)))
        tk.Label(frame, text="Min. sustained movement\n(median px/frame)", justify="left").grid(
            row=r, column=0, sticky="w", padx=4)
        e_med = tk.Entry(frame, textvariable=self.median_var, width=6)
        e_med.grid(row=r, column=1, sticky="w", padx=2, pady=1)
        add_tooltip(e_med, "Minimum sustained movement (median px/frame, scaled to the camera). In a crowded cage "
                           "many 'flying' tracks are really stationary mosquitoes with one or two big jumps from "
                           "linking to a neighbour; their median step is ~0. Tracks with a lower median step are "
                           "dropped from flyers, flight activity and the trajectory / distance metrics. 0 = off "
                           "(the setting of the published Penzance and Aedes results); 3 is the reference value "
                           "for dense cages such as Bangkok. Results go to a flyer_med<value> subfolder.")
        r += 1
        self.radius_var = tk.StringVar(value=str(saved.get('radius', 0.1)))
        tk.Label(frame, text="Radius (cage widths)").grid(row=r, column=0, sticky="w", padx=4)
        e_r = tk.Entry(frame, textvariable=self.radius_var, width=6)
        e_r.grid(row=r, column=1, sticky="w", padx=2, pady=1)
        add_tooltip(e_r, "Neighbourhood around each zone, as a fraction of the cage width: a flyer this close "
                         "to the speaker zone counts as 'near the speaker'. Half and double this radius are also "
                         "reported in the CSVs. Typical: 0.1.")
        r += 1
        from buzzphono_legacy.flight_phonotaxis import default_workers
        n_cpu = os.cpu_count() or 2
        self.workers_var = tk.StringVar(value=str(saved.get('workers', default_workers())))
        tk.Label(frame, text="Workers").grid(row=r, column=0, sticky="w", padx=4)
        s_w = tk.Spinbox(frame, from_=1, to=max(1, n_cpu), width=4, textvariable=self.workers_var)
        s_w.grid(row=r, column=1, sticky="w", padx=2, pady=1)
        add_tooltip(s_w, "Parallel processes reading the tracking files (each needs up to ~1 GB of RAM). "
                         "More is faster until the disk is the bottleneck. Default: cores - 2, at most 8. "
                         "Already-read chunks are cached, so a re-run is quick.")
        r += 1
        self.zones_var = tk.StringVar(value="")
        tk.Label(frame, text="Zones file").grid(row=r, column=0, sticky="w", padx=4)
        e_z = tk.Entry(frame, textvariable=self.zones_var, width=12)
        e_z.grid(row=r, column=1, columnspan=2, sticky="we", padx=2, pady=1)
        b_z = tk.Button(frame, text="…", width=2, command=self._browse_zones)
        b_z.grid(row=r, column=3, sticky="w", padx=2)
        for w in (e_z, b_z):
            add_tooltip(w, "Optional. Blank = the experiment's custom_zones.json (needs 'speaker' plus exactly "
                           "three other zones, one per wall), else plots/flight_phonotaxis/custom_zones_flight.json "
                           "if present. Pick a file to use other zones.")
        r += 1
        run_btn = tk.Button(frame, text="Run flight phonotaxis", command=self.run)
        run_btn.grid(row=r, column=0, columnspan=4, sticky="we", padx=4, pady=(6, 2))
        add_tooltip(run_btn, "Analyse the experiment selected in the list above. Reads tracking_moving, "
                             "tracking_resting and final_tracking_data (never changes them) and writes figures, "
                             "CSVs and a README to plots/flight_phonotaxis/.")
        r += 1
        open_btn = tk.Button(frame, text="Open output folder", command=self._open_output)
        open_btn.grid(row=r, column=0, columnspan=4, sticky="we", padx=4, pady=(0, 4))
        add_tooltip(open_btn, "Open plots/flight_phonotaxis/ of the selected experiment.")
        r += 1
        self.status_var = tk.StringVar(value="")
        tk.Label(frame, textvariable=self.status_var, fg="blue", wraplength=270,
                 justify="left").grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 4))

    # ================================================================ results panel
    def build_results(self, notebook):
        self.notebook = notebook
        frame = tk.Frame(notebook)
        notebook.add(frame, text="Flight phonotaxis results")
        self.results_frame = frame
        self.summary_var = tk.StringVar(value="Select an experiment and click 'Run flight phonotaxis'.")
        tk.Label(frame, textvariable=self.summary_var, justify="left", anchor="w",
                 wraplength=900).pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 4))
        tbl = tk.Frame(frame)
        tbl.pack(side=tk.TOP, fill=tk.X, padx=6)
        cols = ("metric", "phase", "off", "on", "change", "fold", "dz", "n", "p")
        heads = ("Metric", "Phase", "OFF", "ON", "Change", "x-fold", "dz", "ON>OFF", "p (ON>OFF)")
        widths = (430, 50, 70, 70, 75, 55, 55, 115, 75)
        self.tree = ttk.Treeview(tbl, columns=cols, show="headings", height=9)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w" if c == "metric" else "center", stretch=(c == "metric"))
        sb = ttk.Scrollbar(tbl, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        add_tooltip(self.tree, "ON vs OFF per hour-cycle. 'Speaker - ...' rows test the speaker's ON-OFF change "
                               "minus that wall's change (> 0 = the tone favours the speaker). dz = mean / SD of the "
                               "per-cycle differences; p = one-sided Wilcoxon signed-rank. Moving toward the "
                               "speaker always means moving away from the opposite wall, so judge the speaker "
                               "against the top and bottom walls too.")
        row = tk.Frame(frame)
        row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(row, text="Figure:").pack(side=tk.LEFT)
        from buzzphono_legacy.flight_phonotaxis.figures import FIGURES
        self._fig_kinds = [k for k, _t, _f in FIGURES]
        self.fig_var = tk.StringVar(value=FIGURES[0][1])
        self.fig_combo = ttk.Combobox(row, textvariable=self.fig_var, state="readonly", width=40,
                                      values=[t for _k, t, _f in FIGURES])
        self.fig_combo.pack(side=tk.LEFT, padx=4)
        self.fig_combo.bind("<<ComboboxSelected>>", lambda e: self._draw_figure())
        add_tooltip(self.fig_combo, "Figures of the last run (the same ones are saved as PNG files).")
        self.canvas_host = tk.Frame(frame)
        self.canvas_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))
        self._raise_preview_on_show()

    def _raise_preview_on_show(self):
        """The tab's other actions (zone previews, BuzzPhono plots) draw into the 'Plot preview' sub-tab
        (index 0): bring it to the front whenever they do, so they are not hidden behind these results."""
        pv = getattr(self.tm, "preview", None)
        if pv is None:
            return
        for name in ("show_path", "show_image"):
            orig = getattr(pv, name, None)
            if orig is None:
                continue

            def wrapped(*a, **k):
                fn = k.pop("_orig")
                try:
                    self.notebook.select(0)
                except tk.TclError:
                    pass
                return fn(*a, **k)
            setattr(pv, name, lambda *a, _o=orig, **k: wrapped(*a, _orig=_o, **k))

    # ================================================================ run
    def run(self):
        if self._running:
            self.log("BuzzPhono: a flight-phonotaxis run is already in progress.")
            return
        exp = self.tm._selected_experiment_dir()
        if not exp:
            messagebox.showerror(TITLE, "Add an experiment and select it in the Experiments list first.")
            return
        try:
            from buzzphono_legacy.flight_phonotaxis import FlightParams, FlightPhonotaxisError
            light = (int(self.light_start_var.get()), int(self.light_end_var.get()))
            mult = float(self.mult_var.get())
            min_median = float(self.median_var.get())
            radius = float(self.radius_var.get())
            workers = max(1, int(self.workers_var.get()))
            on = (self.tm._int_or(self.tm.on_start_var, 40), self.tm._int_or(self.tm.on_end_var, 50))
            off = (self.tm._int_or(self.tm.off_start_var, 28), self.tm._int_or(self.tm.off_end_var, 38))
            FlightParams(on=on, off=off, light_hours=light, flyer_mult=mult, min_median=min_median,
                         radius=radius)   # validates
        except FlightPhonotaxisError as exc:
            messagebox.showerror(TITLE, str(exc))
            return
        except (TypeError, ValueError) as exc:
            messagebox.showerror(TITLE, "Check the flight-phonotaxis settings: they must be numbers (%s)." % exc)
            return
        kwargs = dict(zones_file=self.zones_var.get().strip() or None, start=self.from_var.get().strip() or None,
                      end=self.to_var.get().strip() or None, on=on, off=off, light_hours=light, flyer_mult=mult,
                      min_median=min_median, radius=radius, workers=workers)
        self._save(light, mult, radius, workers, min_median)
        alias = os.path.basename(os.path.normpath(exp))
        self._dispatch(exp, alias, kwargs)

    def _dispatch(self, exp, alias, kwargs):
        from buzzphono_legacy.flight_phonotaxis import Cancelled, FlightPhonotaxisError, run_flight_phonotaxis
        self._running = True
        cancel = threading.Event()
        events = queue.Queue()
        describe = "flight phonotaxis on %s" % alias
        self.status_var.set("Running %s…" % describe)
        self.ui_manager.set_status(action="BuzzPhono: %s…" % describe, busy=True)
        self.log("BuzzPhono: running %s…" % describe)
        try:
            from progress_dialog import ProgressDialog
            self._progress = ProgressDialog(self.root, title="BuzzPhono (Phonotaxis)",
                                            message="Running %s…" % describe, cancelable=True)
        except Exception:  # noqa: BLE001
            self._progress = None

        def worker():
            try:
                result = run_flight_phonotaxis(
                    exp, progress=lambda f, m: events.put(('progress', (f, m))), cancel=cancel,
                    log=lambda m: events.put(('log', m)), **kwargs)
            except Cancelled:
                events.put(('cancelled', None))
            except FlightPhonotaxisError as exc:
                events.put(('error', (str(exc), None)))
            except Exception as exc:  # noqa: BLE001
                events.put(('error', ("%s: %s" % (type(exc).__name__, exc), traceback.format_exc())))
            else:
                events.put(('done', result))

        threading.Thread(target=worker, daemon=True).start()
        self._poll(describe, events, cancel)

    def _poll(self, describe, events, cancel):
        if self._progress is not None and self._progress.cancelled:
            cancel.set()
        progress = None
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == 'progress':
                progress = payload
                continue
            if kind == 'log':
                self.log("BuzzPhono: %s" % payload)
                continue
            if kind == 'done':
                self._on_done(describe, payload)
            elif kind == 'cancelled':
                self._finish("Cancelled.", "%s cancelled" % describe)
                self.log("BuzzPhono: %s cancelled (finished chunk caches are kept)." % describe)
            else:
                msg, tb = payload
                self._finish("Error in %s." % describe, "%s failed" % describe)
                if tb:
                    self.log("BuzzPhono: %s failed:\n%s" % (describe, tb))
                messagebox.showerror(TITLE, "Flight phonotaxis failed:\n%s" % msg)
            return
        if progress is not None and self._progress is not None:
            self._progress.update(*progress)
        self.root.after(100, lambda: self._poll(describe, events, cancel))

    def _finish(self, status, action):
        self._running = False
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self.status_var.set(status)
        self.ui_manager.set_status(action="BuzzPhono: %s" % action, busy=False)

    def _on_done(self, describe, result):
        self._result = result
        ck = result["clock"]
        warn = ""
        if not ck["fitted"]:
            warn = "  Not enough tone-onset bursts: using the nominal clock."
        if result["problems"]:
            warn += "  %d chunk(s) unreadable (see log)." % len(result["problems"])
        self._finish("Done: %d hour-cycles -> %s%s" % (result["n_cycles"], result["out_dir"], warn),
                     "%s done" % describe)
        self.log("BuzzPhono: %s finished in %.0f s -> %s" % (describe, result.get("elapsed_s", 0), result["out_dir"]))
        self._show_result(result)
        if self.notebook is not None and self.results_frame is not None:
            try:
                self.notebook.select(self.results_frame)
            except tk.TclError:
                pass

    # ================================================================ results display
    def _show_result(self, result):
        ck = result["clock"]
        p = result["params"]
        if ck["fitted"]:
            clock_txt = "clock calibrated on %d tone-onset bursts (drift %+.1f s/day, residual SD %.1f s)" % (
                ck["n_kept"], ck["b"], ck["resid_sd"])
        else:
            clock_txt = "not enough tone-onset bursts (%d found), using the nominal clock" % ck["n_bursts"]
        if p.get("min_median", 0) > 0:
            clock_txt += "   |   sustained-movement filter: median >= %g px/frame" % p["min_median"]
        roles = result["role_zones"]
        self.summary_var.set(
            "%s   |   %s to %s   |   %d hour-cycles (%d light, %d dark)   |   ON %d-%d, OFF %d-%d   |   %s   |   "
            "zones: speaker = %s, opposite wall = %s, top = %s, bottom = %s   |   output: %s"
            % (result["label"], result["window"][0], result["window"][1], result["n_cycles"],
               result["n_cycles_light"], result["n_cycles_dark"], p["on"][0], p["on"][1], p["off"][0], p["off"][1],
               clock_txt, roles[0], roles[1], roles[2], roles[3], result["out_dir"]))
        self.tree.delete(*self.tree.get_children())
        from buzzphono_legacy.flight_phonotaxis.report import pfmt
        last_section = None
        for r in result["key_results"].itertuples():
            if r.section != last_section:
                self.tree.insert("", tk.END, values=("— %s —" % r.section, "", "", "", "", "", "", "", ""))
                last_section = r.section
            if r.kind == "pct":
                vals = (r.metric, r.phase, _fmt(r.off, "%.1f%%"), _fmt(r.on, "%.1f%%"), _fmt(r.change, "%+.1f"),
                        _fmt(r.fold, "%.2f"), "", r.n_on_gt_off, "")
            else:
                vals = (r.metric, r.phase, _fmt(r.off), _fmt(r.on), _fmt(r.change, "%+.3g"),
                        _fmt(r.fold, "%.2f"), _fmt(r.dz, "%+.2f"), r.n_on_gt_off, pfmt(r.p_on_gt_off))
            self.tree.insert("", tk.END, values=vals)
        self._draw_figure()

    def _draw_figure(self):
        if self._result is None:
            return
        try:
            idx = list(self.fig_combo['values']).index(self.fig_var.get())
        except ValueError:
            idx = 0
        kind = self._fig_kinds[idx]
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from buzzphono_legacy.flight_phonotaxis.figures import build_figure
            self.canvas_host.update_idletasks()
            dpi = 80
            w = max(6.0, self.canvas_host.winfo_width() / float(dpi))
            h = max(3.5, self.canvas_host.winfo_height() / float(dpi))
            fig = build_figure(kind, self._result, scale=0.75, dpi=dpi, figsize=(w, h))
        except Exception as exc:  # noqa: BLE001
            self.log("BuzzPhono: could not draw the flight-phonotaxis figure: %s" % exc)
            return
        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self.canvas_host)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ================================================================ helpers
    def _browse_zones(self):
        exp = self.tm._selected_experiment_dir()
        path = filedialog.askopenfilename(title="Zones file (speaker + 3 control zones)",
                                          initialdir=exp or None, filetypes=[("Zones JSON", "*.json")])
        if path:
            self.zones_var.set(path)

    def _open_output(self):
        exp = self.tm._selected_experiment_dir()
        if not exp:
            messagebox.showerror(TITLE, "Select an experiment in the Experiments list first.")
            return
        d = os.path.join(exp, "plots", "flight_phonotaxis")
        if not os.path.isdir(d):
            messagebox.showerror(TITLE, "No flight-phonotaxis output yet for this experiment:\n%s" % d)
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(TITLE, "Could not open %s:\n%s" % (d, exc))

    def _load_saved(self):
        try:
            saved = self.tm.experiment_manager.config.get(self.CONFIG_KEY, {})
            if isinstance(saved, dict):
                return saved
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _save(self, light, mult, radius, workers, min_median=0.0):
        try:
            em = self.tm.experiment_manager
            em.config[self.CONFIG_KEY] = {'light_start': light[0], 'light_end': light[1], 'flyer_mult': mult,
                                          'min_median': min_median, 'radius': radius, 'workers': workers}
            em.save_config(em.config)
        except Exception as exc:  # noqa: BLE001
            self.log("BuzzPhono: could not save flight-phonotaxis settings: %s" % exc)
