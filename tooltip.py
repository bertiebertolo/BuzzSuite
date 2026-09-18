"""Lightweight hover tooltip for Tkinter widgets (BuzzSuite Step 7 inline help).

Pure-Tk, main-thread only, no external dependencies. Use ``add_tooltip(widget, text)``
to attach a plain-English description that appears after a short hover delay and
disappears on leave / click. Safe on any Tk widget (Entry, Spinbox, Checkbutton, Label…).
"""
import tkinter as tk


class ToolTip:
    """Attach a hover tooltip to a single widget.

    The tip is a borderless Toplevel shown near the pointer after ``delay`` ms of hover
    and hidden on <Leave> / <ButtonPress>. One instance per widget.
    """

    def __init__(self, widget, text, delay=500, wraplength=280):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None or not self.text:
            return
        # Position just below/right of the widget.
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry("+%d+%d" % (x, y))
        try:
            # Keep it above; harmless if the platform ignores it.
            self._tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        label = tk.Label(
            self._tip, text=self.text, justify="left",
            background="#ffffe0", foreground="#000000",
            relief="solid", borderwidth=1,
            wraplength=self.wraplength, padx=6, pady=4,
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def add_tooltip(widget, text, delay=500, wraplength=280):
    """Attach a hover tooltip to ``widget`` and return the ToolTip (kept alive by the binding)."""
    return ToolTip(widget, text, delay=delay, wraplength=wraplength)
