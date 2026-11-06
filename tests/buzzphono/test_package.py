"""Package-level guarantees: version, CLI exit codes, and the core/UI separation."""

from __future__ import annotations

import json
import pkgutil
import subprocess
import sys

import buzzphono
from buzzphono import cli
from buzzphono.errors import BuzzPhonoError, CalibrationError, Cancelled, ConfigError, DataError

UI_MODULES = ("typer", "streamlit", "plotly", "rich")


def test_version_is_a_string() -> None:
    assert isinstance(buzzphono.__version__, str)
    assert buzzphono.__version__[0].isdigit()


def test_cli_version_exits_zero(capsys) -> None:
    assert cli.run(["--version"]) == 0
    assert buzzphono.__version__ in capsys.readouterr().out


def test_cli_unknown_command_is_a_usage_error() -> None:
    assert cli.run(["no-such-command"]) == 2


def test_error_exit_codes() -> None:
    assert ConfigError("x").exit_code == 3
    assert DataError("x").exit_code == 4
    assert CalibrationError("x").exit_code == 5
    assert Cancelled().exit_code == 130
    err = BuzzPhonoError("Something is wrong.", hint="Do this.")
    assert str(err) == "Something is wrong.\nHint: Do this."


def _core_modules() -> list[str]:
    names = []
    for info in pkgutil.walk_packages(buzzphono.__path__, prefix="buzzphono."):
        if info.name in ("buzzphono.cli", "buzzphono.__main__") or info.name.startswith("buzzphono.app"):
            continue
        names.append(info.name)
    return names


def test_core_never_imports_a_ui_library() -> None:
    """Analysis code must run without the CLI/app stack (and stay importable on a bare install)."""
    code = (
        "import importlib, json, sys\n"
        f"for name in {_core_modules()!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(json.dumps(sorted(m for m in {UI_MODULES!r} if m in sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
