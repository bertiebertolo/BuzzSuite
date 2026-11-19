"""Control zones made from the speaker zone: one copy per other wall, same size, against its wall."""

from __future__ import annotations

import numpy as np
import pytest

from buzzphono.calibration import Calibration, Zone, wall_controls
from buzzphono.errors import CalibrationError
from buzzphono.geometry import area, as_polygon, centroid

CAGE = [(100.0, 50.0), (900.0, 50.0), (900.0, 550.0), (100.0, 550.0)]  # 800 x 500, TL TR BR BL
SPEAKER_LEFT = [(100.0, 250.0), (140.0, 250.0), (140.0, 350.0), (100.0, 350.0)]  # 40 deep, 100 long


def by_name(zones):
    return {z.name: as_polygon(z.polygon) for z in zones}


def test_left_speaker_gives_three_controls_on_the_other_walls() -> None:
    zones = by_name(wall_controls(CAGE, SPEAKER_LEFT))
    assert set(zones) == {"control_right", "control_top", "control_bottom"}
    for poly in zones.values():
        assert area(poly) == pytest.approx(area(as_polygon(SPEAKER_LEFT)))
    right, top, bottom = zones["control_right"], zones["control_top"], zones["control_bottom"]
    # each copy touches its own wall and reaches 40 px into the cage
    assert right[:, 0].max() == pytest.approx(900) and right[:, 0].min() == pytest.approx(860)
    assert top[:, 1].min() == pytest.approx(50) and top[:, 1].max() == pytest.approx(90)
    assert bottom[:, 1].max() == pytest.approx(550) and bottom[:, 1].min() == pytest.approx(510)
    # centred speaker -> centred controls (turned: 100 px along the top/bottom walls)
    assert centroid(right)[1] == pytest.approx(300)
    assert centroid(top)[0] == pytest.approx(500) and np.ptp(top[:, 0]) == pytest.approx(100)


def test_speaker_wall_is_the_nearest_one() -> None:
    top_speaker = [(450.0, 52.0), (550.0, 52.0), (550.0, 92.0), (450.0, 92.0)]
    assert set(by_name(wall_controls(CAGE, top_speaker))) == {
        "control_left",
        "control_right",
        "control_bottom",
    }


def test_controls_make_a_valid_calibration_and_clip_to_the_frame() -> None:
    long_speaker = [(100.0, 100.0), (160.0, 100.0), (160.0, 500.0), (100.0, 500.0)]  # 400 px along
    zones = wall_controls(CAGE, long_speaker, frame=(1000, 600))
    for z in zones:
        a = np.asarray(z.polygon)
        assert (
            (a[:, 0] >= 0).all()
            and (a[:, 0] <= 1000).all()
            and (a[:, 1] >= 0).all()
            and (a[:, 1] <= 600).all()
        )

    cal = Calibration(
        frame_width_px=1000,
        frame_height_px=600,
        cage=CAGE,
        zones=[Zone(name="speaker", role="speaker", polygon=long_speaker), *zones],
    )
    assert len(cal.controls) == 3


def test_needs_four_cage_corners() -> None:
    with pytest.raises(CalibrationError):
        wall_controls(CAGE[:3], SPEAKER_LEFT)
