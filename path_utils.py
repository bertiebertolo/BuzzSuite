"""Shared cross-platform path resolution helpers for BuzzSuite.

Different machines this app runs on can mount the shared data drive at different
paths (e.g. a drive letter on Windows vs a mount point on macOS). Route every
data-root lookup through this module instead of hardcoding a drive path so the
app behaves the same everywhere. Set the `BUZZSUITE_DATA_ROOT` environment
variable to point at your own data drive without editing any code.
"""
import os

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_first_existing(candidates, default=None):
    """Return the first path in `candidates` that exists on disk.

    Falls back to `default` (or the first candidate) if none exist, so callers
    always get a usable path to report in an error message rather than None.
    """
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    if default is not None:
        return default
    return candidates[0] if candidates else None


def resolve_data_root(override=None):
    """Resolve the root of the shared `Buzzwatch` data drive (Recording/Analysis trees).

    `override`, when given and existing on disk, wins outright over the auto-detected
    candidates below — this is how a user-browsed-and-saved data root (intake_wizard.py's
    "Browse…" next to the data-root label) takes precedence over the machine's usual mount.
    """
    if override and os.path.isdir(override):
        return override
    env_root = os.environ.get('BUZZSUITE_DATA_ROOT')
    # A `Buzzwatch` folder sitting next to wherever BuzzSuite itself is running from -- lets a
    # copy of the app placed on the same removable drive as the data find it regardless of what
    # drive letter/mount point that drive gets assigned on a given machine.
    sibling = os.path.join(os.path.dirname(PACKAGE_ROOT), 'Buzzwatch')
    candidates = [
        env_root,
        sibling,
        r'E:\Buzzwatch',
        '/Volumes/Mosquito2/Buzzwatch',
        '/Volumes/Mosquito2',
    ]
    return resolve_first_existing(candidates)


def get_package_root():
    """Return the absolute path to the BuzzSuite/ package directory."""
    return PACKAGE_ROOT


def newest_mtime(directory, prefix=None):
    """Return the newest mtime among non-dot files in `directory` (optionally filtered to names
    starting with `prefix`), or 0.0 if the directory is missing/empty. Mirrors
    buzzswarm/fru2_zt_normalized_analysis.py's `_newest_tracking_mtime` so both the BuzzSwarm cache
    and the Activity concatenation staleness check (batch_processing_tab_manager.py) use the same
    invalidation rule."""
    newest = 0.0
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if name.startswith('.'):
                continue
            if prefix and not name.startswith(prefix):
                continue
            try:
                newest = max(newest, os.path.getmtime(os.path.join(directory, name)))
            except OSError:
                pass
    return newest
