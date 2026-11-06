"""Check that this machine's buzzsuite_env can run BuzzSuite (run after setup_env.bat, or any time).

    python check_install.py

Read-only: writes nothing, opens no window. Checks
  1. package versions against the ones tracking was validated on (environment.yml),
  2. numpy/matplotlib are the PyPI wheels and 2-D matrix multiply + a shaded plot work (the
     conda-forge builds crash under Windows Memory Integrity + CET; DEVLOG 2026-09-16) -- run in a
     child process, because that crash kills the process without a Python error,
  3. every BuzzSuite module imports and the full window builds (hidden),
  4. ffmpeg and the data root are found.
Also prints OpenCV's IPP code path: tracking output is bit-identical to the validation machine only
when this matches (IPP picks CPU-specific code, AVX2 = "l9" there).
"""
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# (import name, distribution name, validated version)
EXPECTED = [
    ("numpy", "numpy", "1.24.4"), ("scipy", "scipy", "1.10.1"), ("pandas", "pandas", "2.0.3"),
    ("matplotlib", "matplotlib", "3.7.3"), ("cv2", "opencv-python", "4.8.0.76"),
    ("sklearn", "scikit-learn", "1.3.2"), ("statsmodels", "statsmodels", "0.14.1"),
    ("seaborn", "seaborn", "0.13.2"), ("PIL", "Pillow", "10.0.0"), ("yaml", "PyYAML", "6.0.3"),
    ("tkcalendar", "tkcalendar", "1.6.1"), ("psutil", "psutil", "6.0.0"),
]
PIP_WHEELS = ("numpy", "matplotlib", "kiwisolver", "contourpy")
PLOT_PROBE = (
    "import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, io\n"
    "np.random.rand(3, 3) @ np.random.rand(3, 3)\n"
    "f, ax = plt.subplots(); ax.axvspan(0, 1); f.savefig(io.BytesIO(), format='png'); print('ok')\n"
)

problems = []


def report(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        problems.append(msg)


def installer_of(dist):
    import glob
    for site in sys.path:
        for info in glob.glob(os.path.join(site, dist + "-*.dist-info")):
            try:
                with open(os.path.join(info, "INSTALLER")) as f:
                    return f.read().strip()
            except OSError:
                return "?"
    return None


def main():
    print("Python", sys.version.split()[0], "at", sys.executable)
    report(sys.version_info[:2] == (3, 8), "Python 3.8")

    print("\nPackages")
    from importlib import metadata
    for mod, dist, want in EXPECTED:
        try:
            importlib.import_module(mod)
            have = metadata.version(dist)
        except Exception as exc:
            report(False, "%s: not importable (%s)" % (dist, exc))
            continue
        report(have == want, "%s %s (validated: %s)" % (dist, have, want))
    for dist in PIP_WHEELS:
        inst = installer_of(dist)
        report(inst == "pip", "%s installed by %s (needs the PyPI wheel; see environment.yml)" % (dist, inst))

    print("\nPlotting (child process)")
    r = subprocess.run([sys.executable, "-c", PLOT_PROBE], capture_output=True, text=True, timeout=120)
    report(r.returncode == 0 and "ok" in r.stdout,
           "matrix multiply + shaded plot (exit code %s)" % r.returncode)

    print("\nOpenCV")
    import cv2
    ipp = cv2.ipp.getIppVersion() if cv2.ipp.useIPP() else "IPP off"
    print("  info  IPP:", ipp, "(validation machine: AVX2 (l9))")

    print("\nBuzzSuite")
    sys.path.insert(0, HERE)
    os.chdir(HERE)
    try:
        import tkinter as tk
        from tkinter import messagebox
        import experiment_manager
        shown = []
        for name in ("showerror", "showwarning", "showinfo"):
            setattr(messagebox, name, lambda *a, **k: shown.append(a[:2]))
        # Opening the window reloads the last experiment, which would update config.json.
        experiment_manager.ExperimentManager.save_config = lambda self, *a, **k: None
        experiment_manager.ExperimentManager.record_recent_experiment = lambda self, *a, **k: None
        from buzzsuite_app import VideoAnalyzerApp
        root = tk.Tk()
        root.withdraw()
        errors = []
        root.report_callback_exception = lambda *exc: errors.append(exc[1])
        app = VideoAnalyzerApp(root)
        app._on_launch()
        for _ in range(20):
            root.update()
        root.destroy()
        report(not errors, "main window builds (all tabs)")
        for title_msg in shown:
            print("  info  dialog on startup:", " | ".join(str(x) for x in title_msg)[:200])
    except Exception as exc:
        report(False, "main window builds: %r" % (exc,))

    sys.path.insert(0, os.path.join(HERE, "h264_converter"))
    from convert_gui import FFMPEG
    try:
        ok = subprocess.run([FFMPEG, "-version"], capture_output=True, timeout=30).returncode == 0
    except OSError:
        ok = False
    report(ok, "ffmpeg: %s" % FFMPEG)

    import json
    from path_utils import resolve_data_root
    override = None
    try:
        with open(os.path.join(HERE, "config.json")) as f:
            override = json.load(f).get("data_root_override")
    except (OSError, ValueError):
        pass
    root_dir = resolve_data_root(override)
    found = bool(root_dir) and os.path.isdir(root_dir)
    print("  %s  data root: %s" % ("ok  " if found else "info", root_dir if found else
          "not found (set it with Browse... in the Setup tab; experiments open by browsing to "
          "their experiment_*.json)"))

    print("\n" + ("All checks passed." if not problems else "%d problem(s):\n  - " % len(problems)
                  + "\n  - ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
