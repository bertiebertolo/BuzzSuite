"""BuzzSuite finds BuzzPhono's environment wherever conda keeps it, and says plainly how to install it when
it is missing (it used to fall back to `conda run`, which failed with "EnvironmentLocationNotFound")."""
import contextlib
import json
import os
import shutil
import stat
import tempfile

import buzzphono_bridge as bb


@contextlib.contextmanager
def _fake_conda(output):
    """A `conda` that prints ``output`` for any command, used as CONDA_EXE, with no other way to find
    BuzzPhono (no BUZZPHONO_PYTHON, no config value, no environment in the usual places)."""
    tmp = tempfile.mkdtemp()
    saved = {k: os.environ.get(k) for k in ("CONDA_EXE", "BUZZPHONO_PYTHON")}
    saved_fns = bb._config_value, bb._candidates
    try:
        out_file = os.path.join(tmp, "out.txt")
        with open(out_file, "w") as fh:
            fh.write(output)
        if os.name == "nt":
            conda = os.path.join(tmp, "conda.bat")
            with open(conda, "w") as fh:
                fh.write('@type "%s"\n' % out_file)
        else:
            conda = os.path.join(tmp, "conda")
            with open(conda, "w") as fh:
                fh.write('#!/bin/sh\ncat "%s"\n' % out_file)
            os.chmod(conda, os.stat(conda).st_mode | stat.S_IEXEC)
        os.environ["CONDA_EXE"] = conda
        os.environ.pop("BUZZPHONO_PYTHON", None)
        bb._config_value = lambda: None
        bb._candidates = lambda: []
        yield tmp, conda
    finally:
        bb._config_value, bb._candidates = saved_fns
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def _env(root, name):
    prefix = os.path.join(root, name)
    exe = os.path.join(prefix, "python.exe" if os.name == "nt" else os.path.join("bin", "python"))
    os.makedirs(os.path.dirname(exe))
    open(exe, "w").close()
    return prefix, exe


def test_missing_environment_says_how_to_install_it():
    with _fake_conda(json.dumps({"envs": ["/nowhere/anaconda3"]})):
        try:
            bb.buzzphono_command()
        except bb.BuzzPhonoNotFound as exc:
            msg = str(exc)
        else:
            raise AssertionError("expected BuzzPhonoNotFound")
    assert "not installed" in msg and "setup_env.bat" in msg and "environment-buzzphono.yml" in msg, msg


def test_environment_in_another_envs_folder_is_used_directly():
    with _fake_conda("") as (tmp, _):
        prefix, exe = _env(os.path.join(tmp, "elsewhere", "envs"), bb.ENV_NAME)
        _env(os.path.join(tmp, "elsewhere", "envs"), "buzzphono_env_old")
        with open(os.path.join(tmp, "out.txt"), "w") as fh:
            json.dump({"envs": [os.path.join(tmp, "base"), prefix + "_old", prefix]}, fh)
        assert bb.buzzphono_command() == [exe, "-m", "buzzphono"]


def test_conda_that_cannot_list_falls_back_to_conda_run():
    with _fake_conda("this is not json") as (_, conda):
        cmd = bb.buzzphono_command()
    assert cmd[:2] == [conda, "run"] and bb.ENV_NAME in cmd, cmd
