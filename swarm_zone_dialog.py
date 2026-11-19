"""Swarm-zone picker for the Swarming tab: shows a camera frame of the experiment with its cage
outline and lets the user set the swarm-zone radius (pixels) with a slider, a number box or by
dragging on the image. Different cameras need different radii: 100 px suits the old 720p camera,
a higher-resolution camera (e.g. Camera Module 3) needs a larger one for the same physical zone.

The radius is saved per experiment (clock_analysis.save_swarm_radius) and used by the next
Run analysis. Day caches do not depend on it, so a rerun only recomputes the metrics.
"""
import tkinter as tk

import numpy as np

MAX_CANVAS_W = 960
MAX_CANVAS_H = 640
OLD_DEFAULT_PX = 100
_AREA_SAMPLES = 40000


class SwarmZoneDialog:
    """Modal Toplevel. on_save(radius_px) is called after the radius has been written to disk."""

    def __init__(self, parent, experiment_dir, title, radius_px, on_save=None, log=None):
        from buzzswarm import clock_analysis as ca
        self._ca = ca
        self.experiment_dir = experiment_dir
        self.on_save = on_save
        self.log = log or (lambda msg: None)
        # raises FileNotFoundError / ValueError when the cage border is missing (caller reports it)
        centroid, polygon = ca.cage_geometry(experiment_dir)
        self.centroid = (float(centroid[0]), float(centroid[1]))
        self.polygon = np.asarray(polygon, dtype=float)
        self._r_inside = self._inside_distances()
        dists = np.hypot(self.polygon[:, 0] - self.centroid[0], self.polygon[:, 1] - self.centroid[1])
        self.max_radius = int(np.ceil(dists.max()))

        self.top = tk.Toplevel(parent)
        self.top.title(f"Swarm zone - {title}")
        self.top.transient(parent)
        self._photo = None
        self._img_size = None
        self._scale = 1.0
        self._build_canvas()

        start = int(round(radius_px)) if radius_px else OLD_DEFAULT_PX
        self.radius_var = tk.IntVar(value=max(1, min(start, self.max_radius)))
        self._build_controls()
        self.radius_var.trace_add('write', lambda *_: self._redraw_zone())
        self._redraw_zone()
        self.top.protocol("WM_DELETE_WINDOW", self.top.destroy)
        self.top.grab_set()
        self.top.focus_set()

    # ------------------------------------------------------------------ geometry
    def _inside_distances(self):
        """Distances from the centroid of uniform points inside the cage (fixed seed), so the share
        of cage area inside any radius is one comparison."""
        from matplotlib.path import Path
        rng = np.random.default_rng(0)
        lo, hi = self.polygon.min(axis=0), self.polygon.max(axis=0)
        path = Path(self.polygon)
        pts = np.empty((0, 2))
        for _ in range(50):
            cand = rng.uniform(lo, hi, size=(_AREA_SAMPLES, 2))
            pts = np.vstack([pts, cand[path.contains_points(cand)]])
            if len(pts) >= _AREA_SAMPLES:
                break
        return np.hypot(pts[:, 0] - self.centroid[0], pts[:, 1] - self.centroid[1])

    def _area_share(self, radius):
        if not len(self._r_inside):
            return float('nan')
        return float(np.mean(self._r_inside < radius))

    # ------------------------------------------------------------------ UI
    def _build_canvas(self):
        path = self._ca.cage_reference_image(self.experiment_dir)
        img = None
        if path:
            try:
                from PIL import Image, ImageTk
                with Image.open(path) as im:
                    img = im.convert('RGB')
                self._img_size = img.size
            except Exception as exc:   # unreadable frame: draw the outline on a blank canvas
                self.log(f"BuzzSwarm: could not open {path}: {exc}")
                img = None
        if self._img_size is None:
            hi = self.polygon.max(axis=0)
            self._img_size = (int(hi[0]) + 20, int(hi[1]) + 20)
        w, h = self._img_size
        self._scale = min(MAX_CANVAS_W / float(w), MAX_CANVAS_H / float(h), 1.0)
        cw, ch = int(round(w * self._scale)), int(round(h * self._scale))
        self.canvas = tk.Canvas(self.top, width=cw, height=ch, bg="black", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(side=tk.TOP, padx=8, pady=(8, 4))
        if img is not None:
            from PIL import Image, ImageTk
            resample = getattr(Image, 'LANCZOS', Image.BICUBIC)
            self._photo = ImageTk.PhotoImage(img.resize((cw, ch), resample) if self._scale < 1.0 else img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.create_text(cw // 2, 14, fill="gray", text="(no camera frame found: outline only)")
        s = self._scale
        flat = [v * s for xy in self.polygon for v in xy]
        self.canvas.create_polygon(*flat, outline="yellow", fill="", width=2)
        cx, cy = self.centroid[0] * s, self.centroid[1] * s
        self.canvas.create_line(cx - 8, cy, cx + 8, cy, fill="yellow", width=2)
        self.canvas.create_line(cx, cy - 8, cx, cy + 8, fill="yellow", width=2)
        r_old = OLD_DEFAULT_PX * s
        self.canvas.create_oval(cx - r_old, cy - r_old, cx + r_old, cy + r_old, outline="gray70",
                                dash=(4, 4), width=1)
        self.canvas.bind("<Button-1>", self._on_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

    def _build_controls(self):
        frame = tk.Frame(self.top)
        frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Label(frame, text="Swarm radius (px):").pack(side=tk.LEFT)
        tk.Scale(frame, from_=1, to=self.max_radius, orient=tk.HORIZONTAL, variable=self.radius_var,
                 showvalue=False, length=360).pack(side=tk.LEFT, padx=(6, 6))
        tk.Spinbox(frame, from_=1, to=self.max_radius, textvariable=self.radius_var, width=6).pack(
            side=tk.LEFT)
        self.info_var = tk.StringVar()
        tk.Label(self.top, textvariable=self.info_var, justify="left", anchor="w").pack(
            side=tk.TOP, fill=tk.X, padx=8)
        tk.Label(self.top, fg="gray", justify="left", anchor="w", wraplength=MAX_CANVAS_W,
                 text="Drag on the image or use the slider. Red = swarm zone, dashed grey = 100 px "
                      "(the old camera's default), yellow = cage outline and centre. Saved for this "
                      "experiment; click Run analysis afterwards (cached days rerun in seconds)."
                 ).pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 4))
        buttons = tk.Frame(self.top)
        buttons.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(buttons, text="Save", width=10, command=self._on_save).pack(side=tk.RIGHT)
        tk.Button(buttons, text="Cancel", width=10, command=self.top.destroy).pack(side=tk.RIGHT, padx=(0, 6))
        tk.Button(buttons, text="Reset to 100 px",
                  command=lambda: self.radius_var.set(min(OLD_DEFAULT_PX, self.max_radius))).pack(side=tk.LEFT)

    def _radius(self):
        try:
            r = int(self.radius_var.get())
        except (tk.TclError, ValueError):
            return None
        return r if 1 <= r <= self.max_radius else None

    def _redraw_zone(self):
        self.canvas.delete("zone")
        r = self._radius()
        if r is None:
            self.info_var.set(f"Type a whole number of pixels between 1 and {self.max_radius}.")
            return
        s = self._scale
        cx, cy, rr = self.centroid[0] * s, self.centroid[1] * s, r * s
        self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline="red", width=2, tags="zone")
        w, h = self._img_size
        self.info_var.set(f"Radius {r} px  ·  {100 * self._area_share(r):.1f}% of the cage area  ·  "
                          f"frame {w} x {h} px")

    def _on_drag(self, event):
        x, y = event.x / self._scale, event.y / self._scale
        r = int(round(np.hypot(x - self.centroid[0], y - self.centroid[1])))
        self.radius_var.set(max(1, min(r, self.max_radius)))

    def _on_save(self):
        r = self._radius()
        if r is None:
            return
        try:
            self._ca.save_swarm_radius(self.experiment_dir, r)
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Swarm zone", f"Could not save the swarm zone:\n{exc}", parent=self.top)
            return
        self.top.destroy()
        if self.on_save:
            self.on_save(r)
