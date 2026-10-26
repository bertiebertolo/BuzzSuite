#!/usr/bin/env python
"""Batch-prepare experiments for tracking: reference images + background images for every selected
experiment (unattended, in parallel), THEN one interactive session drawing each experiment's cage
border (4 corners), one window after another. Stops there -- tracking stays a separate step
(run_flare_pipeline.py, which already skips finished work).

Replaces opening every experiment in the GUI Setup tab and clicking Extract Images -> Get Background
-> Draw Cage Borders by hand. Both prep steps call the same idempotent ExperimentManager methods the
Setup tab calls (existing outputs are skipped), and the border is saved exactly the way
user_input_draw_borders_cage + plot_all_borders save it, so re-running this script after an
interruption just picks up whatever is still missing.

Any experiment whose folder name contains INVESTIGATE (case-insensitive) is set aside and never
touched -- that is the convention for experiments with a known problem still being looked into.

Usage
-----
    python prepare_experiments.py --root F:\\Buzzwatch --dry-run     # status table only, no changes
    python prepare_experiments.py --root F:\\Buzzwatch               # prep everything, then draw
    python prepare_experiments.py --root F:\\Buzzwatch --only Lutzia # name-substring filter
    python prepare_experiments.py F:\\Buzzwatch\\Analysis\\Flare\\Bangkok\\Lutzia_F_20260906_20260909
    python prepare_experiments.py --root F:\\Buzzwatch --draw-only --redraw   # redo borders

Drawing window keys
-------------------
    left-click        place a corner (4 needed; the outline appears after the 4th)
    Enter / Space     accept and save (only once exactly 4 corners are placed)
    r / Backspace     clear the corners and start over
    s                 skip this experiment (left undrawn -- re-run later to draw it)
    q / Esc           stop the drawing session (everything already accepted stays saved)
"""
import argparse
import concurrent.futures
import glob
import json
import os
import sys
import threading
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from path_utils import resolve_data_root

SET_ASIDE_MARKER = "INVESTIGATE"
LOG_DIR = os.path.join(APP_DIR, "logs")
_log_path = None
_log_lock = threading.Lock()


def _log(msg, log_path=None):
    """Timestamped line to the console and (if set) the run's log file. Child prep processes pass
    the parent's log_path explicitly, since module globals aren't shared across spawned processes."""
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    path = log_path or _log_path
    with _log_lock:
        print(line, flush=True)
        if path:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass  # never let a logging failure take down the run


# ---------------------------------------------------------------------------------------------
# Selection + status (cheap: reads the JSON/YAML and lists folders, never loads the experiment)
# ---------------------------------------------------------------------------------------------

def _find_json(path):
    """Accept an experiment folder or an experiment_*.json path; return the JSON path or None."""
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return os.path.abspath(path)
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "experiment_*.json")))
        if found:
            return os.path.abspath(found[0])
    return None


def _select(args):
    if args.paths:
        jsons = []
        for p in args.paths:
            jp = _find_json(p)
            if jp is None:
                _log("SKIP (no experiment_*.json found): %s" % p)
            else:
                jsons.append(jp)
    else:
        data_root = resolve_data_root(args.root or os.environ.get("BUZZSUITE_DATA_ROOT"))
        if not data_root:
            _log("Could not resolve a data root -- pass --root (e.g. --root F:\\Buzzwatch).")
            return []
        pattern = os.path.join(data_root, "Analysis", args.assay or "*", "*", "*", "experiment_*.json")
        jsons = sorted(glob.glob(pattern))
        if not jsons:
            _log("No experiment_*.json found under %s" % pattern)
    names = lambda jp: os.path.basename(os.path.dirname(jp)).lower()
    if args.only:
        jsons = [jp for jp in jsons if args.only.lower() in names(jp)]
    if args.skip:
        jsons = [jp for jp in jsons if args.skip.lower() not in names(jp)]
    return jsons


def _pngs(folder):
    if not os.path.isdir(folder):
        return set()
    return set(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(".png"))


def _status(jp):
    """Dict describing what this experiment has and needs. Mirrors load_experiment_from_json's
    path reconstruction (explicit folder_videos wins over the rebuilt Recording path)."""
    import yaml
    exp_dir = os.path.dirname(jp)
    st = {"json": jp, "dir": exp_dir, "name": os.path.basename(exp_dir), "error": None,
          "set_aside": SET_ASIDE_MARKER.lower() in os.path.basename(exp_dir).lower(),
          "videos": 0, "images": 0, "backgrounds": 0, "border": False,
          "need_images": False, "need_bg": False, "has_video_folder": False}
    if st["set_aside"]:
        return st
    try:
        with open(jp, "r") as f:
            d = json.load(f)
        folder_videos = d.get("folder_videos") or os.path.join(
            d.get("root_folder_videos") or "", d.get("experiment_name") or "",
            d.get("cage_name") or "", d.get("batch_name") or "")
        st["has_video_folder"] = os.path.isdir(folder_videos)
        videos = set()
        if st["has_video_folder"]:
            videos = set(os.path.splitext(f)[0] for f in os.listdir(folder_videos)
                         if f.lower().endswith(".mp4"))
        images = _pngs(os.path.join(exp_dir, "individual_images"))
        backgrounds = _pngs(os.path.join(exp_dir, "images_mortality"))
        st.update(videos=len(videos), images=len(images), backgrounds=len(backgrounds),
                  need_images=bool(videos - images), need_bg=bool(images - backgrounds))
        settings_file = d.get("settings_file")
        if settings_file and os.path.isfile(settings_file):
            with open(settings_file, "r") as f:
                settings = yaml.safe_load(f) or {}
            st["border"] = len(settings.get("cage_border_points") or []) == 4
        else:
            st["error"] = "settings file not found: %s" % settings_file
    except Exception as exc:
        st["error"] = "could not read experiment: %s" % exc
    return st


def _needs_prep(st):
    return (not st["set_aside"] and not st["error"] and st["has_video_folder"]
            and (st["need_images"] or st["need_bg"]))


def _needs_draw(st, redraw):
    return (not st["set_aside"] and not st["error"] and st["backgrounds"] > 0
            and (redraw or not st["border"]))


def _print_table(statuses, redraw):
    _log("%-55s %7s %7s %7s %7s  %s" % ("Experiment", "videos", "images", "bg", "border", "action"))
    for st in statuses:
        rel = os.path.relpath(st["dir"], os.path.dirname(os.path.dirname(os.path.dirname(st["dir"]))))
        if st["set_aside"]:
            action = "set aside (INVESTIGATE)"
        elif st["error"]:
            action = "ERROR: " + st["error"]
        else:
            parts = []
            if _needs_prep(st):
                parts.append("prep")
            elif not st["has_video_folder"]:
                parts.append("no Recording folder")
            elif st["videos"] == 0:
                parts.append("no .mp4 videos yet")
            if (redraw or not st["border"]) and (st["backgrounds"] > 0 or _needs_prep(st)):
                parts.append("draw")
            action = " + ".join(parts) if parts else "ready"
        _log("%-55s %7d %7d %7d %7s  %s" % (rel, st["videos"], st["images"], st["backgrounds"],
                                           "yes" if st["border"] else "-", action))


# ---------------------------------------------------------------------------------------------
# Phase 1: prep (one spawned process per experiment)
# ---------------------------------------------------------------------------------------------

def _prep_one(jp, log_path):
    """Top-level (spawn-safe) worker: reference images, then background images, for one experiment.
    Runs in its own process because extract_average_background swaps the global sys.stdout."""
    name = os.path.basename(os.path.dirname(jp))
    log = lambda msg: _log("[%s] %s" % (name, msg), log_path)
    t0 = time.time()
    try:
        from experiment_manager import ExperimentManager
        em = ExperimentManager(log)
        em.load_experiment_from_json(jp)
        em.get_images_from_video(force_rerun=0)
        em.get_background_from_images(force_rerun=0)
        return jp, True, "done in %.0fs" % (time.time() - t0)
    except Exception:
        return jp, False, traceback.format_exc()


def _run_prep(todo, jobs):
    if not todo:
        _log("Prep: nothing to do.")
        return {}
    jobs = max(1, min(jobs, len(todo)))
    _log("Prep: %d experiment(s), %d at a time." % (len(todo), jobs))
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = [ex.submit(_prep_one, st["json"], _log_path) for st in todo]
        for fut in concurrent.futures.as_completed(futures):
            try:
                jp, ok, msg = fut.result()
            except Exception:
                continue  # a crashed worker process; reported as missing below
            results[jp] = (ok, msg)
            name = os.path.basename(os.path.dirname(jp))
            _log("Prep %s: %s" % ("OK  " if ok else "FAILED", name + ("  " + msg if ok else ":\n" + msg)))
    for st in todo:
        if st["json"] not in results:
            results[st["json"]] = (False, "worker process crashed")
            _log("Prep FAILED: %s (worker process crashed)" % st["name"])
    return results


# ---------------------------------------------------------------------------------------------
# Phase 2: draw (main process, one OpenCV window per experiment)
# ---------------------------------------------------------------------------------------------

class QuadDrawer(object):
    """Click/key state for drawing one 4-corner cage border. Kept free of any window calls so it
    can be exercised headlessly; draw_cage_border() below wires it to an OpenCV window."""

    ENTER_KEYS = (13, 10, 32)
    RESET_KEYS = (ord("r"), ord("R"), 8)
    SKIP_KEYS = (ord("s"), ord("S"))
    QUIT_KEYS = (ord("q"), ord("Q"), 27)

    def __init__(self, image, title, existing=None):
        self.image = image
        self.title = title
        self.existing = [tuple(int(v) for v in p[:2]) for p in (existing or [])]
        self.points = []
        # Enlarges markers/header text to stay readable when the window shows the image downscaled
        self.ui_scale = 1.0

    def on_click(self, x, y):
        if len(self.points) < 4:
            self.points.append((int(x), int(y)))

    def on_key(self, key):
        """Return 'accept', 'skip', 'quit' or None (keep going)."""
        if key in self.ENTER_KEYS:
            return "accept" if len(self.points) == 4 else None
        if key in self.RESET_KEYS:
            self.points = []
        elif key in self.SKIP_KEYS:
            return "skip"
        elif key in self.QUIT_KEYS:
            return "quit"
        return None

    def render(self):
        import cv2
        img = self.image.copy()
        s = self.ui_scale

        def px(v):
            return max(1, int(round(v * s)))

        if len(self.existing) == 4:
            for i in range(4):
                cv2.line(img, self.existing[i], self.existing[(i + 1) % 4], (0, 165, 255), px(1), cv2.LINE_AA)
        for p in self.points:
            cv2.circle(img, p, px(4), (0, 255, 0), -1)
        for a, b in zip(self.points, self.points[1:]):
            cv2.line(img, a, b, (0, 255, 0), px(2), cv2.LINE_AA)
        if len(self.points) == 4:
            cv2.line(img, self.points[3], self.points[0], (0, 255, 0), px(2), cv2.LINE_AA)
        n = len(self.points)
        hint = ("Enter/Space = save   r = redo   s = skip   q = quit" if n == 4 else
                "click corner %d of 4   r = redo   s = skip   q = quit" % (n + 1))
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], px(52)), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.putText(img, self.title, (px(8), px(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), px(1), cv2.LINE_AA)
        cv2.putText(img, hint, (px(8), px(43)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 255, 255), px(1), cv2.LINE_AA)
        return img


def draw_cage_border(image_path, title, existing=None):
    """Open a window on image_path; return ('accept', points) / ('skip', None) / ('quit', None).
    Closing the window with its X counts as skip."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        _log("Cannot read %s -- skipping." % image_path)
        return "skip", None
    drawer = QuadDrawer(image, title, existing)
    win = "BuzzSuite - draw cage border"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawer.on_click(x, y)

    from buzzwatch_data_analysis.misc_functions import fit_cv_window_to_screen
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    # First imshow before sizing, so the window isn't snapped back to the full image size
    cv2.imshow(win, drawer.render())
    scale = fit_cv_window_to_screen(win, image.shape[1], image.shape[0])
    drawer.ui_scale = 1.0 / scale
    try:
        while True:
            cv2.imshow(win, drawer.render())
            key = cv2.waitKey(30)
            if key != -1:
                action = drawer.on_key(key & 0xFF)
                if action == "accept":
                    return "accept", list(drawer.points)
                if action in ("skip", "quit"):
                    return action, None
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return "skip", None
    finally:
        cv2.destroyWindow(win)
        cv2.waitKey(1)


def save_cage_border(exp, points):
    """Save exactly as buzzwatch_experiment_analysis.user_input_draw_borders_cage does, then
    regenerate background_with_borders.png the same way the GUI's Draw Cage Borders button does."""
    exp.settings["cage_border_points"] = exp._normalize_quad_order(exp._normalize_quad_points(points))
    exp._save_settings_file()
    exp.plot_all_borders()


def _run_draw(queue):
    if not queue:
        _log("Draw: nothing to draw.")
        return 0, 0, 0
    from experiment_manager import ExperimentManager
    _log("Draw: %d experiment(s). A window opens for each -- see the keys in the window header."
         % len(queue))
    drawn = skipped = 0
    for i, st in enumerate(queue, 1):
        mort = os.path.join(st["dir"], "images_mortality")
        images = sorted(f for f in os.listdir(mort) if f.endswith(".png"))
        if not images:
            _log("[%d/%d] %s: no background image -- skipping." % (i, len(queue), st["name"]))
            skipped += 1
            continue
        try:
            em = ExperimentManager(_log)
            em.load_experiment_from_json(st["json"])
            exp = em.experiment
        except Exception:
            _log("[%d/%d] %s: could not load:\n%s" % (i, len(queue), st["name"], traceback.format_exc()))
            skipped += 1
            continue
        existing = exp.settings.get("cage_border_points") or []
        title = "[%d/%d] %s" % (i, len(queue), os.path.relpath(
            st["dir"], os.path.dirname(os.path.dirname(os.path.dirname(st["dir"])))))
        action, points = draw_cage_border(os.path.join(mort, images[0]), title, existing)
        if action == "accept":
            try:
                save_cage_border(exp, points)
                drawn += 1
                _log("%s: saved cage border %s" % (title, exp.settings["cage_border_points"]))
            except Exception:
                skipped += 1
                _log("%s: SAVE FAILED:\n%s" % (title, traceback.format_exc()))
        elif action == "skip":
            skipped += 1
            _log("%s: skipped" % title)
        else:
            _log("%s: drawing session stopped by user." % title)
            return drawn, skipped, len(queue) - i + 1
    return drawn, skipped, 0


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    global _log_path
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*",
                   help="Experiment folders or experiment_*.json files. Omit to scan the data root.")
    p.add_argument("--root", default=None,
                   help="Data root (e.g. F:\\Buzzwatch). Default: $BUZZSUITE_DATA_ROOT or auto-detect.")
    p.add_argument("--assay", default=None, help="Only this assay folder under Analysis\\ (e.g. Flare).")
    p.add_argument("--only", default=None, help="Only experiments whose folder name contains this.")
    p.add_argument("--skip", default=None, help="Skip experiments whose folder name contains this.")
    p.add_argument("--redraw", action="store_true",
                   help="Also offer drawing for experiments that already have a cage border "
                        "(the current one is shown in orange for reference).")
    p.add_argument("--prep-only", action="store_true", help="Only extract images/backgrounds.")
    p.add_argument("--draw-only", action="store_true", help="Only draw borders.")
    p.add_argument("--jobs", type=int, default=4, help="Experiments prepped in parallel (default 4).")
    p.add_argument("--dry-run", action="store_true", help="Print the status table and exit.")
    args = p.parse_args(argv)

    if not args.dry_run:
        os.makedirs(LOG_DIR, exist_ok=True)
        _log_path = os.path.join(LOG_DIR, "prepare_experiments_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
        _log("Logging to %s" % _log_path)

    jsons = _select(args)
    if not jsons:
        return 1
    statuses = [_status(jp) for jp in jsons]
    _print_table(statuses, args.redraw)
    if args.dry_run:
        _log("Dry run -- nothing done.")
        return 0

    prep_failed = []
    if not args.draw_only:
        prep_results = _run_prep([st for st in statuses if _needs_prep(st)], args.jobs)
        prep_failed = [jp for jp, (ok, _) in prep_results.items() if not ok]

    remaining = 0
    if not args.prep_only:
        statuses = [_status(jp) for jp in jsons]  # re-read: prep just created the backgrounds
        drawn, skipped, remaining = _run_draw([st for st in statuses if _needs_draw(st, args.redraw)])
        _log("Draw summary: %d drawn, %d skipped, %d not reached." % (drawn, skipped, remaining))

    if prep_failed:
        _log("%d experiment(s) failed prep -- see the FAILED lines above:" % len(prep_failed))
        for jp in prep_failed:
            _log("   " + os.path.dirname(jp))
    still = [st["name"] for st in (_status(jp) for jp in jsons)
             if not st["set_aside"] and not st["error"] and not st["border"]]
    if still:
        _log("Still without a cage border (re-run to finish): " + ", ".join(still))
    return 1 if prep_failed else 0


if __name__ == "__main__":
    sys.exit(main())
