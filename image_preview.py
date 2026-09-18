"""Shared responsive plot preview for the "3 · Plotting" sub-tabs.

A Label that scales the shown plot to fill whatever space its containing frame actually has
(never upscaled past the image's native resolution, to avoid blurring a matplotlib PNG), and
re-renders on resize. Replaces each tab's own fixed-900x640-cap _show_image, which left a
blank margin around the plot that grew with window size instead of using it.
"""
import tkinter as tk
from PIL import Image, ImageTk


class ImagePreview:
    def __init__(self, parent, placeholder="", log=None):
        self.label = tk.Label(parent, text=placeholder, anchor="center")
        self.label.pack(fill=tk.BOTH, expand=True)
        self._log = log or (lambda msg: None)
        self._source = None   # last-shown path (str) or PIL.Image, re-rendered on resize
        self._photo = None    # keep a reference so Tk does not GC the PhotoImage
        self._resize_job = None
        self.label.bind("<Configure>", self._on_configure)

    def show_path(self, path):
        self._source = path
        self._render()

    def show_image(self, img):
        self._source = img
        self._render()

    def _on_configure(self, _event):
        if self._source is None:
            return
        if self._resize_job is not None:
            self.label.after_cancel(self._resize_job)
        self._resize_job = self.label.after(80, self._render)

    def _render(self):
        self._resize_job = None
        if self._source is None:
            return
        try:
            img = Image.open(self._source) if isinstance(self._source, str) else self._source
            w_avail, h_avail = self.label.winfo_width(), self.label.winfo_height()
            max_w = (w_avail - 16) if w_avail > 20 else 900
            max_h = (h_avail - 16) if h_avail > 20 else 640
            w, h = img.size
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.label.configure(image=self._photo, text="")
        except Exception as exc:
            self._log(f"Could not display preview: {exc}")
