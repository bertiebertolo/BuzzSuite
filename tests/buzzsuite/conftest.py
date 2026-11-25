"""BuzzSuite app tests (Tk app, BuzzSwarm, yolo_export). They run in the app's own environment
(buzzsuite_env / buzzwatch_env, Python 3.8+), which has no pytest: run them with

    python tests/buzzsuite/run_all.py

They are plain test_* functions with no fixtures, so pytest collects them too where it is installed.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
