"""H264 -> MP4 converter tab.

Embeds the standalone converter (``h264_converter/convert_gui.py``) as a notebook
tab instead of a separate window. The converter's ``App`` builds its widgets onto
whatever container it is handed and only touches window chrome (title/geometry)
when it owns a real Tk/Toplevel window, so passing it the tab frame is enough.

All conversion work runs on a background thread inside ``convert_worker`` and talks
back to the UI through a thread-safe queue polled via ``after()`` — no analysis runs
on the Tk main thread.
"""
import tkinter as tk
from tkinter import ttk

from h264_converter import convert_gui


class H264TabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.tab = None
        self.app = None

    def init_h264_tab(self, tab: ttk.Frame):
        self.tab = tab
        # convert_gui.App builds its widgets directly onto the container it is given
        # and skips window-chrome calls when that container is a Frame, so it drops
        # straight into the tab.
        self.app = convert_gui.App(tab)
