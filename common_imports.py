import os
import tkinter as tk
from tkinter import ttk, filedialog, Toplevel
from PIL import Image, ImageTk
import json
import sys
import threading
import time
import pickle
import re
from datetime import datetime, timedelta,time
import pandas as pd
import numpy as np
from tkcalendar import DateEntry

import matplotlib
_HEADLESS_WORKER = os.environ.get("BUZZSUITE_HEADLESS_WORKER", "0") == "1"
# Always 'Agg', including the main GUI process (not just multiprocessing workers). Every plot in
# this app -- Activity (activity_plot_manager.py), BuzzSwarm/BuzzPhono (their engine modules'
# own matplotlib.use('Agg') calls, previously redundant with this one) -- renders to a PNG via
# fig.savefig() and displays that PNG as a static image (image_preview.py); nothing anywhere in
# the reachable UI calls FigureCanvasTkAgg or plt.show() (verified: grepped the whole tree --
# the few plt.show() call sites left are in comparison_tab_manager.py/statistical_analysis_
# manager.py/glmm_analysis_manager.py/plot_manager.py, none of which setup_managers() in
# ui_manager.py constructs since the 2026-07 reorg -- dead code, not reachable). So 'TkAgg' was
# never actually needed here, and choosing it had a real cost: BuzzSwarm/BuzzPhono's own
# `matplotlib.use('Agg')` module-level calls only execute once (Python caches imports), and since
# those modules are only ever first-imported lazily inside a background worker thread
# (buzzswarm_tab_manager.py/buzzphono_tab_manager.py's `_dispatch`/worker()), that first call was
# switching the process's *global* backend away from 'TkAgg' from a non-main thread. matplotlib's
# switch_backend() closes every still-open figure manager from the *previous* backend first, and
# Activity plotting's _on_plot_clicked keeps its most recent TkAgg-backed figure alive
# (self._last_fig, only closed on the *next* plot) -- so if the user had plotted anything in
# Activity before their first BuzzSwarm/BuzzPhono run, that close() call tore down a live
# Tk-attached figure manager from a background thread instead of the main thread: a classic
# Tcl/Tk cross-thread hazard that hangs rather than raising, which read as "gets most of the way
# through a BuzzSwarm/BuzzPhono run, then the whole app just stops responding." Setting 'Agg' once
# here, on the main thread, before any tab or background worker ever touches matplotlib, makes
# every later matplotlib.use('Agg') call (including the engine modules' own) a same-backend no-op
# that never invokes switch_backend()/close() at all -- the hazard is now structurally impossible,
# not just less likely.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FigureCanvasTkAgg = None  # unused anywhere in this app (see above) -- kept as a name in case any
                          # caller still imports it from here; do not resurrect the TkAgg branch
                          # without re-auditing the freeze this fixed.
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import seaborn as sns