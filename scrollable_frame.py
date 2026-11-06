"""Shared vertical-scroll wrapper for a tab's left-hand controls column.

Tkinter has no built-in scrollable Frame. Every "3 · Plotting" sub-tab stacks enough
LabelFrames in its controls column (picker, experiment list, parameters, run buttons) to
overflow a short window with no indication more controls exist below; this wraps that column
in a Canvas + Scrollbar so every control stays reachable regardless of window height.
"""
import sys
import tkinter as tk


def make_scrollable_column(parent, width=300):
    """Returns (outer, inner). Pack ``outer`` where a plain controls Frame used to go (same
    side/fill/pad as before); build all controls into ``inner`` exactly as before. Scrolls via
    the scrollbar, mouse wheel (while hovered), or trackpad."""
    outer = tk.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0, width=width)
    scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas)

    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _wheel(event):
        if sys.platform == "darwin":
            canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_wheel(_e=None):
        canvas.bind_all("<MouseWheel>", _wheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    def _unbind_wheel(_e=None):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _bind_wheel)
    canvas.bind("<Leave>", _unbind_wheel)

    return outer, inner
