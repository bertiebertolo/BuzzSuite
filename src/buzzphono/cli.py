"""Command-line interface: ``buzzphono <command>``.

The commands are thin wrappers: they parse options, call into the buzzphono package, print
progress and map errors to exit codes (see :mod:`buzzphono.errors`). No analysis lives here.
"""

from __future__ import annotations

import contextlib
import sys

import typer

from buzzphono import __version__
from buzzphono.errors import BuzzPhonoError

app = typer.Typer(
    name="buzzphono",
    help="BuzzPhono: phonotaxis analysis of BuzzWatch mosquito tracking data.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"buzzphono {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """BuzzPhono: phonotaxis analysis of BuzzWatch mosquito tracking data."""


@app.command()
def version() -> None:
    """Print the BuzzPhono version."""
    typer.echo(f"buzzphono {__version__}")


def _safe_streams() -> None:
    """Never crash on a character the console cannot show (e.g. a Japanese-locale Windows console)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def run(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code (used by :func:`main` and the tests).

    Typer handles help and usage errors itself (exit code 2). BuzzPhono's own errors are printed
    in plain English and mapped to their exit codes; long-running commands turn Ctrl-C into
    :class:`buzzphono.errors.Cancelled` (exit code 130).
    """
    _safe_streams()
    try:
        app(args=argv, prog_name="buzzphono")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    except BuzzPhonoError as exc:
        typer.echo(f"Error: {exc}", err=True)
        return exc.exit_code
    return 0


def main() -> None:
    """Console-script entry point (``buzzphono``)."""
    raise SystemExit(run())
