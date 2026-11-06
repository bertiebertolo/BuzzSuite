"""Modal progress dialog with percentage + estimated-time-remaining (BuzzSuite Step 7).

Tk is single-threaded: create/update/close this dialog only from the main thread. For a
background worker, marshal updates via ``root.after(0, lambda: dlg.update(frac, msg))``.

Typical use (indeterminate work whose fraction you can estimate):

    dlg = ProgressDialog(root, title="BuzzSwarm", message="Running Sholl…")
    ...                          # on the worker, call root.after(0, ...) to update
    dlg.update(0.4, "Binning flights…")
    dlg.close()

If you cannot compute a fraction, call ``dlg.pulse()`` to run an indeterminate bar.
"""
import time
import tkinter as tk
from tkinter import ttk


class ProgressDialog:
    def __init__(self, root, title="Working…", message="Please wait…", cancelable=False):
        self.root = root
        self._start = time.time()
        self._cancelled = False
        self._indeterminate = False

        self.top = tk.Toplevel(root)
        self.top.title(title)
        self.top.transient(root)
        self.top.resizable(False, False)
        # Block interaction with the main window while the dialog is up.
        try:
            self.top.grab_set()
        except Exception:
            pass

        frm = tk.Frame(self.top, padx=18, pady=14)
        frm.pack(fill="both", expand=True)

        self.msg_var = tk.StringVar(value=message)
        tk.Label(frm, textvariable=self.msg_var, anchor="w",
                 wraplength=340, justify="left").pack(fill="x")

        self.bar = ttk.Progressbar(frm, orient="horizontal", length=340,
                                   mode="determinate", maximum=100.0)
        self.bar.pack(fill="x", pady=(10, 6))

        self.eta_var = tk.StringVar(value="")
        tk.Label(frm, textvariable=self.eta_var, anchor="w", fg="gray").pack(fill="x")

        if cancelable:
            tk.Button(frm, text="Cancel", command=self._on_cancel).pack(pady=(8, 0))
            self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)
        else:
            # Ignore the window-close button; the owner controls the lifecycle.
            self.top.protocol("WM_DELETE_WINDOW", lambda: None)

        self._center()
        self.top.update_idletasks()

    # ------------------------------------------------------------------ public
    def update(self, fraction, message=None):
        """Set the bar to ``fraction`` in [0, 1] and refresh the ETA. Main thread only."""
        if self._indeterminate:
            self._indeterminate = False
            self.bar.stop()
            self.bar.configure(mode="determinate")
        frac = max(0.0, min(1.0, float(fraction)))
        self.bar["value"] = frac * 100.0
        if message is not None:
            self.msg_var.set(message)
        self.eta_var.set(self._eta_text(frac))
        self._pump()

    def pulse(self, message=None):
        """Switch to an indeterminate marching bar (unknown fraction). Main thread only."""
        if not self._indeterminate:
            self._indeterminate = True
            self.bar.configure(mode="indeterminate")
            self.bar.start(60)
        if message is not None:
            self.msg_var.set(message)
        self.eta_var.set("Elapsed %s" % self._fmt(time.time() - self._start))
        self._pump()

    def set_message(self, message):
        self.msg_var.set(message)
        self._pump()

    @property
    def cancelled(self):
        return self._cancelled

    def close(self):
        try:
            if self._indeterminate:
                self.bar.stop()
            self.top.grab_release()
        except Exception:
            pass
        try:
            self.top.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------ internal
    def _on_cancel(self):
        self._cancelled = True
        self.set_message("Cancelling…")

    def _eta_text(self, frac):
        elapsed = time.time() - self._start
        if frac <= 0.001:
            return "Estimating…  (elapsed %s)" % self._fmt(elapsed)
        if frac >= 0.999:
            return "Finishing…  (elapsed %s)" % self._fmt(elapsed)
        remaining = elapsed * (1.0 - frac) / frac
        return "%.0f%%   ~%s remaining   (elapsed %s)" % (
            frac * 100.0, self._fmt(remaining), self._fmt(elapsed))

    @staticmethod
    def _fmt(seconds):
        seconds = int(max(0, seconds))
        if seconds < 60:
            return "%ds" % seconds
        if seconds < 3600:
            return "%dm %02ds" % (seconds // 60, seconds % 60)
        return "%dh %02dm" % (seconds // 3600, (seconds % 3600) // 60)

    def _center(self):
        try:
            self.top.update_idletasks()
            rw = self.root.winfo_rootx(); rh = self.root.winfo_rooty()
            rW = self.root.winfo_width(); rH = self.root.winfo_height()
            w = self.top.winfo_reqwidth(); h = self.top.winfo_reqheight()
            x = rw + max(0, (rW - w) // 2)
            y = rh + max(0, (rH - h) // 3)
            self.top.wm_geometry("+%d+%d" % (x, y))
        except Exception:
            pass

    def _pump(self):
        try:
            self.top.update_idletasks()
        except Exception:
            pass
