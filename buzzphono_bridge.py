"""Bridge from BuzzSuite (Python 3.8, buzzsuite_env) to BuzzPhono (its own environment, buzzphono_env).

BuzzPhono runs in a separate Python, so BuzzSuite never imports it: it starts
``<buzzphono python> -m buzzphono ...`` as a child process with a clean environment. The
interpreter is found, in order, from the BUZZPHONO_PYTHON environment variable, the
``buzzphono_python`` key of BuzzSuite's config.json, the usual conda locations, and finally
``conda run -n buzzphono_env``.
"""
import json
import os
import shutil
import subprocess
import sys
import time

ENV_NAME = "buzzphono_env"
_STRIP = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE",
          "CONDA_SHLVL", "CONDA_PROMPT_MODIFIER", "_CE_CONDA", "_CE_M")


class BuzzPhonoNotFound(RuntimeError):
    pass


def _config_value():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "config.json"), "r") as fh:
            return json.load(fh).get("buzzphono_python")
    except (OSError, ValueError, AttributeError):
        return None


def _candidates():
    exe = "python.exe" if os.name == "nt" else os.path.join("bin", "python")
    roots = []
    for base in (os.environ.get("CONDA_EXE"), sys.executable):
        if base:
            # .../anaconda3/envs/<env>/python.exe or .../anaconda3/Scripts/conda.exe -> .../anaconda3
            p = os.path.dirname(os.path.abspath(base))
            for _ in range(3):
                if os.path.isdir(os.path.join(p, "envs")):
                    roots.append(p)
                    break
                p = os.path.dirname(p)
    home = os.path.expanduser("~")
    roots += [os.path.join(home, d) for d in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", "opt/anaconda3")]
    roots += ["/opt/anaconda3", "/opt/miniconda3", "/opt/homebrew/Caskroom/miniforge/base"]
    seen = []
    for r in roots:
        cand = os.path.join(r, "envs", ENV_NAME, exe)
        if cand not in seen:
            seen.append(cand)
    return seen


def buzzphono_command():
    """The command prefix that runs BuzzPhono, e.g. ['C:/.../python.exe', '-m', 'buzzphono']."""
    for value in (os.environ.get("BUZZPHONO_PYTHON"), _config_value()):
        if value:
            if not os.path.isfile(value):
                raise BuzzPhonoNotFound("BuzzPhono's Python is set to %s, but that file does not exist." % value)
            return [value, "-m", "buzzphono"]
    for cand in _candidates():
        if os.path.isfile(cand):
            return [cand, "-m", "buzzphono"]
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        return [conda, "run", "--no-capture-output", "-n", ENV_NAME, "python", "-m", "buzzphono"]
    raise BuzzPhonoNotFound(
        "BuzzPhono is not installed: no '%s' conda environment was found.\n"
        "Install it with: conda env create -f environment-buzzphono.yml\n"
        "or set BUZZPHONO_PYTHON to its python executable." % ENV_NAME)


def clean_env():
    env = dict(os.environ)
    for k in _STRIP:
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    return env


def run(args, log=None, cwd=None):
    """Run ``buzzphono <args>`` to completion, passing each output line to ``log``; return the exit code."""
    cmd = buzzphono_command() + list(args)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=clean_env(), cwd=cwd,
                            universal_newlines=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if log is not None:
            log(line.rstrip())
    return proc.wait()


def start(args, log_path, cwd=None):
    """Start ``buzzphono <args>`` in the background (output to ``log_path``); return the Popen."""
    cmd = buzzphono_command() + list(args)
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with open(log_path, "ab") as fh:  # the child keeps its own handle
        return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=clean_env(), cwd=cwd,
                                creationflags=flags)


# BuzzPhono's run-folder protocol (buzzphono/jobs.py), mirrored here because BuzzSuite never imports it.
RUN_MARKER = ".buzzphono_run"   # jobs.MARKER: only a folder holding it counts as a BuzzPhono run's
STATUS = "status.json"
CANCEL = "CANCEL"


def claim_run_dir(path):
    """Create a new, empty run folder and mark it as a BuzzPhono run's (as jobs.claim_run_dir does), so
    the run accepts it although the launcher has already put process.log there."""
    os.makedirs(path)  # never an existing folder: earlier results are not reused
    with open(os.path.join(path, RUN_MARKER), "w", encoding="utf-8") as fh:
        fh.write("BuzzPhono run folder, created %s by BuzzSuite (process %d)\n"
                 % (time.strftime("%Y-%m-%dT%H:%M:%S"), os.getpid()))
    return path


def read_status(run_dir):
    """The run's status.json as a dict, or None while it does not exist (or is being replaced)."""
    try:
        with open(os.path.join(run_dir, STATUS), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_settings(path, experiment, on=(40, 50), off=(28, 38), start=None, end=None, analyses=None,
                   light_hours=None, workers=None, extra=None):
    """A BuzzPhono settings file for one experiment (JSON is valid YAML)."""
    exp = {"path": os.path.abspath(experiment)}
    if start:
        exp["start"] = str(start)
    if end:
        exp["end"] = str(end)
    data = {"experiments": [exp], "stimulus": {"on": list(on), "off": list(off)},
            "analyses": list(analyses or ["resting_fraction"])}
    if light_hours is not None:
        data["light"] = {"lights_on": "%02d:00" % light_hours[0], "lights_off": "%02d:00" % light_hours[1]}
    if workers:
        data["workers"] = int(workers)
    data.update(extra or {})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return path
