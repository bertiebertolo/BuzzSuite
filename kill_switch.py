"""Emergency stop: immediately terminate every process this app has spawned (tracking/
concatenation multiprocessing.Pool workers, ffmpeg.exe conversions, any other child process)
and then this process itself, closing the GUI and -- when launched via run_buzzsuite.bat's
double-click path -- the wrapping terminal window too (the batch script has nothing left to run
once this process is gone, so its console closes on its own).

Deliberately does not try to individually track and gracefully cancel every Pool/thread across
every tab manager (dashboard jobs, the batch queue, concatenation, h264 conversion, ...) -- that
registry doesn't exist today and would be easy to leave a gap in. Instead this kills the whole OS
process tree at once, which is complete by construction: every multiprocessing worker and every
subprocess (ffmpeg included) is a descendant of this process, so a tree-kill reaches all of them
regardless of which code path spawned them.
"""
import os
import platform
import signal
import subprocess


def force_kill_everything():
    """Never returns normally -- this process is gone by the time it would."""
    system = platform.system()
    pid = os.getpid()
    if system == "Windows":
        try:
            # /T kills the whole process tree (this process + every descendant: Pool workers,
            # ffmpeg.exe, etc.) in one shot, run with no console flashing on screen. Block on it
            # (subprocess.run, not Popen) rather than firing-and-forgetting: taskkill needs this
            # PID and its children to still be alive when it actually queries the process tree, so
            # racing ahead and exiting first (before taskkill has run its query) can leave
            # non-Pool-managed children (e.g. a raw ffmpeg subprocess) orphaned -- confirmed by an
            # empirical test, see DEVLOG.md. Since taskkill's target list includes this PID's own
            # process, in practice this process gets torn down mid-wait and never returns from the
            # call at all; if it somehow does return, fall through to the explicit os._exit below.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception:
            pass
    else:
        try:
            # POSIX: killing our own process group takes every child down with us in one call,
            # no external tool needed. Multiprocessing workers inherit our process group. SIGKILL
            # can't be caught, so on success this line ends this process too -- nothing after it
            # runs.
            os.killpg(os.getpgid(0), signal.SIGKILL)
        except Exception:
            pass
    # Fallback / belt-and-braces: only reached if the platform-specific path above raised.
    # os._exit skips atexit/finally handlers on purpose -- this is a kill switch, not a clean
    # shutdown, and waiting on any of those is exactly what it's meant to avoid.
    os._exit(1)
