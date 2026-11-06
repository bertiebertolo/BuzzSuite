#!/usr/bin/env python3
"""Collect what BuzzPhono needs from a BuzzWatch recording Pi into one zip file.

Run it on the Pi after (or during) an experiment. It needs only Python 3.9+ and its standard
library::

    python3 pi_collect_logs.py
    python3 pi_collect_logs.py --since 2026-10-01 --out ~/buzzphono_pi_logs.zip

The zip holds:

* the sound service's log (``journalctl -u buzzwatch-sound``): when each tone window opened and
  closed, and any failures; plus the recorder's and the light's logs;
* every file in the recorder's log folder (``[paths] logs_dir`` in config.ini);
* ``config.ini`` itself;
* the clock: ``timedatectl`` status and the time-synchronisation history (a Pi has no real-time
  clock, so this shows when its clock was set or jumped);
* the Pi's identity: host name, OS release, uptime.

Copy the zip next to the experiment folder on the analysis PC; ``buzzphono validate
--playback-log <zip>`` and ``buzzphono analyse`` read it directly. ``--self-test`` checks the script
itself (it runs anywhere, with fake files).
"""

import argparse
import configparser
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile

SERVICES = ("buzzwatch-sound", "buzzwatch-record", "buzzwatch-light")
CONFIG_CANDIDATES = (
    "~/Picamera2/config.ini",
    "~/pirecorder/config.ini",
    "~/BuzzWatch/config.ini",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"),
)


def run(cmd, timeout=120):
    """(output text, problem or None). A missing program is a problem, not a crash."""
    if shutil.which(cmd[0]) is None:
        return "", "%s is not installed on this computer" % cmd[0]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return "", "%s failed: %s" % (" ".join(cmd), exc)
    if res.returncode != 0 and not res.stdout:
        return "", "%s exited with code %d: %s" % (" ".join(cmd), res.returncode, res.stderr.strip()[:300])
    return res.stdout, None


def find_config(explicit=None):
    candidates = [explicit] if explicit else list(CONFIG_CANDIDATES)
    for c in candidates:
        if c and os.path.isfile(os.path.expanduser(c)):
            return os.path.expanduser(c)
    return None


def collect(out_path, config_path=None, since=None, extra_log_dirs=()):
    """Write the zip. Returns the manifest (what was collected and what could not be)."""
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "collected": [],
        "problems": [],
    }
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        def add_text(name, text):
            zf.writestr(name, text)
            manifest["collected"].append(name)

        cfg = find_config(config_path)
        logs_dirs = list(extra_log_dirs)
        if cfg:
            zf.write(cfg, "config.ini")
            manifest["collected"].append("config.ini")
            parser = configparser.ConfigParser(interpolation=None)
            try:
                parser.read(cfg)
                logs = parser.get("paths", "logs_dir", fallback=None)
                if logs:
                    logs_dirs.append(logs)
            except configparser.Error as exc:
                manifest["problems"].append("config.ini could not be parsed: %s" % exc)
        else:
            manifest["problems"].append("config.ini not found (looked in %s)" % ", ".join(CONFIG_CANDIDATES))
        for unit in SERVICES:
            cmd = ["journalctl", "-u", unit, "-o", "short-iso", "--no-pager"]
            if since:
                cmd += ["--since", since]
            text, problem = run(cmd)
            if problem:
                manifest["problems"].append(problem)
            else:
                add_text("journal_%s.txt" % unit, text)
        for folder in logs_dirs:
            folder = os.path.expanduser(folder)
            if not os.path.isdir(folder):
                manifest["problems"].append("log folder not found: %s" % folder)
                continue
            for name in sorted(os.listdir(folder)):
                path = os.path.join(folder, name)
                if os.path.isfile(path) and (name.endswith((".log", ".txt"))):
                    zf.write(path, "logs/" + name)
                    manifest["collected"].append("logs/" + name)
        for name, cmd in (
            ("timedatectl.txt", ["timedatectl", "status"]),
            ("timesync_status.txt", ["timedatectl", "timesync-status"]),
            (
                "journal_timesyncd.txt",
                ["journalctl", "-u", "systemd-timesyncd", "-o", "short-iso", "--no-pager"],
            ),
        ):
            text, problem = run(cmd)
            if problem:
                manifest["problems"].append(problem)
            else:
                add_text(name, text)
        ident = {"hostname": socket.gethostname(), "platform": platform.platform()}
        if os.path.isfile("/etc/os-release"):
            with open("/etc/os-release", encoding="utf-8", errors="replace") as fh:
                ident["os_release"] = fh.read()
        if os.path.isfile("/proc/uptime"):
            with open("/proc/uptime", encoding="utf-8") as fh:
                ident["uptime_s"] = fh.read().split()[0]
        add_text("identity.json", json.dumps(ident, indent=1))
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))
    return manifest


def self_test():
    """Exercise the collector with fake files (no Pi needed). Returns 0 when everything works."""
    work = tempfile.mkdtemp(prefix="pi_collect_selftest_")
    try:
        logs = os.path.join(work, "logs")
        os.makedirs(logs)
        with open(os.path.join(logs, "sound.log"), "w", encoding="utf-8") as fh:
            fh.write(
                "2026-10-10 00:40:00 [INFO] Window open - playing 510 Hz tone\n"
                "2026-10-10 00:50:00 [INFO] Window closed - playback stopped\n"
            )
        cfg = os.path.join(work, "config.ini")
        with open(cfg, "w", encoding="utf-8") as fh:
            fh.write("[recording]\nlabel = Test\n[paths]\nlogs_dir = %s\n" % logs)
        out = os.path.join(work, "out.zip")
        manifest = collect(out, config_path=cfg)
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            sound = zf.read("logs/sound.log").decode("utf-8")
        ok = {
            "config.ini",
            "logs/sound.log",
            "identity.json",
            "manifest.json",
        } <= names and "Window open" in sound
        print(
            "self-test %s: %d entries, %d problems noted (missing programs are expected off a Pi)"
            % ("passed" if ok else "FAILED", len(names), len(manifest["problems"]))
        )
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect the logs and settings BuzzPhono needs from this Pi.")
    p.add_argument(
        "--out", default=None, help="zip file to write (default: ~/buzzphono_pi_logs_<host>_<date>.zip)"
    )
    p.add_argument(
        "--config", default=None, help="path of config.ini (default: searched in the usual places)"
    )
    p.add_argument("--since", default=None, help="only journal entries since this date, e.g. 2026-10-01")
    p.add_argument("--self-test", action="store_true", help="check this script with fake files and exit")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    out = args.out or os.path.expanduser(
        "~/buzzphono_pi_logs_%s_%s.zip" % (socket.gethostname(), dt.date.today().isoformat())
    )
    manifest = collect(os.path.expanduser(out), config_path=args.config, since=args.since)
    print("Wrote %s with %d files." % (out, len(manifest["collected"])))
    for problem in manifest["problems"]:
        print("  note: " + problem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
