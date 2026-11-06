"""tools/pi_collect_logs.py runs on the Pi; check it works anywhere and that BuzzPhono reads its zip."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from buzzphono.io.playback_log import read_playback_log

TOOL = Path(__file__).resolve().parents[3] / "tools" / "pi_collect_logs.py"


def test_self_test_passes() -> None:
    res = subprocess.run(
        [sys.executable, str(TOOL), "--self-test"], capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "self-test passed" in res.stdout


def test_collected_zip_is_readable_by_buzzphono(tmp_path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "sound.log").write_text(
        "2026-10-10 00:40:00 [INFO] Window open - playing 510 Hz tone\n"
        "2026-10-10 00:50:00 [INFO] Window closed - playback stopped\n"
    )
    cfg = tmp_path / "config.ini"
    cfg.write_text(f"[paths]\nlogs_dir = {logs}\n")
    out = tmp_path / "pi.zip"
    res = subprocess.run(
        [sys.executable, str(TOOL), "--config", str(cfg), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    with zipfile.ZipFile(out) as zf:
        assert "logs/sound.log" in zf.namelist() and "config.ini" in zf.namelist()
    log = read_playback_log(out)
    assert len(log.intervals) == 1 and log.intervals["end_reason"].iloc[0] == "closed"
