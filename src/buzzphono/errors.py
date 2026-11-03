"""Errors BuzzPhono raises on purpose.

Every error meant for a person carries a plain-English message (what is wrong and where) and an
optional hint (what to do about it). The command line turns each class into an exit code:

====  =====================================================
0     success
1     unexpected internal error (a bug: please report it)
2     command-line usage error (unknown option, bad value)
3     configuration or validation problem
4     data problem (missing, corrupt or unusable tracking data)
5     calibration problem (missing, or does not match the data)
130   cancelled by the user
====  =====================================================
"""

from __future__ import annotations


class BuzzPhonoError(Exception):
    """Base class for errors written for a person to read."""

    exit_code = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ConfigError(BuzzPhonoError):
    """A configuration value, file or path is wrong."""

    exit_code = 3


class ValidationFailed(BuzzPhonoError):
    """`validate` (or the check before a run) found a blocking problem."""

    exit_code = 3


class DataError(BuzzPhonoError):
    """Tracking data is missing, corrupt or cannot be used for the requested analysis."""

    exit_code = 4


class CalibrationError(BuzzPhonoError):
    """The calibration is missing or does not match the recording it is applied to."""

    exit_code = 5


class Cancelled(BuzzPhonoError):
    """The user cancelled a running job."""

    exit_code = 130

    def __init__(self, message: str = "Cancelled.", *, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
