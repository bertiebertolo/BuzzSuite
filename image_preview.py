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
        self._anim = None     # an animated GIF being played (open PIL image, current frame)
        self._anim_job = None
        self.label.bind("<Configure>", self._on_configure)

    def show_path(self, path):
        self._stop_animation()
        self._source = path
        try:
            img = Image.open(path)
            if getattr(img, "n_frames", 1) > 1:
                self._anim = img
            else:
                img.close()
        except Exception:
            pass
        self._render()
        if self._anim is not None:
            self._anim_job = self.label.after(self._frame_ms(), self._next_frame)

    def show_image(self, img):
        self._stop_animation()
        self._source = img
        self._render()

    def _frame_ms(self):
        return max(20, int(self._anim.info.get("duration") or 100))

    def _next_frame(self):
        """Advance an animated GIF one frame (seeking forward is cheap; back to 0 at the end)."""
        self._anim_job = None
        if self._anim is None:
            return
        try:
            nxt = self._anim.tell() + 1
            self._anim.seek(nxt if nxt < self._anim.n_frames else 0)
        except EOFError:
            self._anim.seek(0)
        except Exception:
            self._stop_animation()
            return
        self._render()
        if self._anim is not None:
            self._anim_job = self.label.after(self._frame_ms(), self._next_frame)

    def _stop_animation(self):
        if self._anim_job is not None:
            try:
                self.label.after_cancel(self._anim_job)
            except Exception:
                pass
            self._anim_job = None
        if self._anim is not None:
            try:
                self._anim.close()
            except Exception:
                pass
            self._anim = None

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
            if self._anim is not None:
                img = self._anim.convert("RGB")             # the current frame
            else:
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
