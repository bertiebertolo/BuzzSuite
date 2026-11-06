"""The Pi's config.ini and its sound-playback log."""

from __future__ import annotations

import zipfile
from datetime import datetime

import pytest

from buzzphono.errors import ConfigError, DataError
from buzzphono.io.pi_config import cross_check, parse_windows, read_pi_config
from buzzphono.io.playback_log import read_playback_log, window_status

GOOD_INI = """
[recording]
output_dir = /home/penzancepi/pirecorder/recordings
label = TbambusaM-510Hz-1212LD-5am5pm
segment_duration = 1200
width = 1920
height = 1080
fps = 25

[sound]
enabled = true
mode = tone
frequency_hz = 510
sound_file = /home/penzancepi/pirecorder/sounds/stimulus.wav
windows = 40-50
active_from = 00:00
active_until = 24:00

[light]
ramp_up_start = 05:00
ramp_up_end = 05:30
ramp_down_start = 16:30
ramp_down_end = 17:00

[paths]
logs_dir = /home/penzancepi/pirecorder/logs
"""


def _write(tmp_path, text, name="config.ini"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_good_config_parses_cleanly(tmp_path) -> None:
    cfg = read_pi_config(_write(tmp_path, GOOD_INI))
    assert cfg.checks == []
    assert (cfg.width, cfg.height, cfg.fps) == (1920, 1080, 25.0)
    assert cfg.windows == [(40, 50)] and cfg.frequency_hz == 510.0 and cfg.sound_enabled is True
    assert cfg.ramp_up_start_min == 300 and cfg.ramp_down_end_min == 17 * 60 and cfg.active_until_min == 1440


def test_home_folder_typo_is_named(tmp_path) -> None:
    text = GOOD_INI.replace("sound_file = /home/penzancepi/", "sound_file = /home/penzance/")
    cfg = read_pi_config(_write(tmp_path, text))
    (check,) = cfg.checks
    assert check.field == "[sound] sound_file"
    assert check.severity == "warning"  # unused while mode = tone
    assert (
        "/home/penzance/" in check.message
        and "/home/penzancepi/" in check.message
        and "typo" in check.message
    )
    cfg = read_pi_config(_write(tmp_path, text.replace("mode = tone", "mode = file")))
    assert [c.severity for c in cfg.checks] == ["error"]


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("width = 1920", "width = wide", "[recording] width"),
        ("fps = 25", "fps = 900", "[recording] fps"),
        ("windows = 40-50", "windows = 40-70", "[sound] windows"),
        ("active_from = 00:00", "active_from = 7.30", "[sound] active_from"),
        ("enabled = true", "enabled = maybe", "[sound] enabled"),
        ("mode = tone", "mode = song", "[sound] mode"),
        ("ramp_up_end = 05:30", "ramp_up_end = 04:30", "[light] ramp_up_end"),
    ],
)
def test_each_bad_field_is_reported_by_name(tmp_path, old, new, field) -> None:
    cfg = read_pi_config(_write(tmp_path, GOOD_INI.replace(old, new)))
    assert field in [c.field for c in cfg.errors], cfg.checks


def test_missing_and_unreadable_files(tmp_path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        read_pi_config(tmp_path / "missing.ini")
    with pytest.raises(ConfigError, match=r"not a valid config\.ini"):
        read_pi_config(_write(tmp_path, "no section header here\nx = 1\n"))


def test_parse_windows() -> None:
    assert parse_windows("10-20, 40-50") == [(10, 20), (40, 50)]
    with pytest.raises(ValueError):
        parse_windows("40")


def test_cross_check_against_the_recording(tmp_path) -> None:
    cfg = read_pi_config(_write(tmp_path, GOOD_INI))
    assert (
        cross_check(
            cfg,
            frame_size=(1920, 1080),
            labels={"TbambusaM-510Hz-1212LD-5am5pm"},
            on_window=(40, 50),
            frequency_hz=510,
            fps=25.0,
        )
        == []
    )
    problems = cross_check(
        cfg, frame_size=(1280, 720), labels={"CageAedes"}, on_window=(30, 40), frequency_hz=475, fps=30.0
    )
    assert {c.field for c in problems} == {
        "[recording] width/height",
        "[recording] label",
        "[sound] windows",
        "[sound] frequency_hz",
        "[recording] fps",
    }


@pytest.mark.realdata
def test_the_real_pi_config_typo_is_flagged(require_real_data) -> None:
    cfg = read_pi_config(require_real_data("pi_config"))
    flagged = [c for c in cfg.checks if c.field == "[sound] sound_file"]
    assert flagged and "/home/penzance/" in flagged[0].message


LOG = """\
2026-10-10 00:39:58 [INFO] Sound schedule running: 510 Hz tone on plughw:CARD=CODEC,DEV=0, minutes 40-50 of every hour
2026-10-10 00:40:00 [INFO] Window open - playing 510 Hz tone at 80% volume
2026-10-10 00:50:00 [INFO] Window closed - playback stopped
Oct 10 01:40:00 penzancepi python3[812]: 2026-10-10 01:40:00 [ERROR] Could not start playback: [Errno 2] No such file (retry in 10s)
2026-10-10T02:40:01+0100 penzancepi python3[812]: 2026-10-10 02:40:01 [INFO] Window open - playing 510 Hz tone at 80% volume
2026-10-10 02:44:00 [ERROR] Playback stopped unexpectedly. Retrying in 10s
2026-10-10 03:40:00 [INFO] Window open - playing 510 Hz tone at 80% volume
2026-10-10 03:50:00 [INFO] Window closed - playback stopped
2026-10-10 04:00:00 [INFO] Sound schedule stopped
"""  # noqa: E501


def test_playback_log_events_and_window_status(tmp_path) -> None:
    log = read_playback_log(_write(tmp_path, LOG, "sound.log"))
    assert log.stimuli == ["510 Hz tone at 80% volume"]
    assert len(log.intervals) == 3
    assert list(log.intervals["end_reason"]) == ["closed", "interrupted", "closed"]

    def status(h: int) -> str:
        return window_status(log, datetime(2026, 10, 10, h, 40), datetime(2026, 10, 10, h, 50))[0]

    assert status(0) == "played"
    assert status(1) == "failed"
    assert status(2) == "partly played"
    assert status(3) == "played"
    assert (
        window_status(log, datetime(2026, 10, 11, 0, 40), datetime(2026, 10, 11, 0, 50))[0]
        == "not covered by the log"
    )


def test_playback_log_from_a_zip_and_duplicates(tmp_path) -> None:
    z = tmp_path / "pi_logs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("logs/sound.log", LOG)
        zf.writestr("journal_buzzwatch-sound.txt", LOG)  # same events twice: counted once
        zf.writestr("config.ini", GOOD_INI)
    log = read_playback_log(z)
    assert len(log.intervals) == 3 and len(log.sources) == 2


def test_playback_log_without_events(tmp_path) -> None:
    with pytest.raises(DataError, match="No playback events"):
        read_playback_log(_write(tmp_path, "nothing useful\n", "sound.log"))
