from __future__ import annotations

import json

import numpy as np
import pytest

from buzzphono import calibration as cal
from buzzphono.config import LightSettings, Window, build_config, load_config
from buzzphono.errors import CalibrationError, ConfigError


def test_defaults_are_the_published_protocol(tmp_path) -> None:
    c = build_config({"experiments": [{"path": str(tmp_path)}]})
    assert c.stimulus.on.as_tuple() == (40, 50) and c.stimulus.off.as_tuple() == (28, 38)
    assert c.metric.colony_size == 30 and c.clock.mode == "corrected" and c.stats.block_hours == 12
    assert c.analyses == ["resting_fraction"]


@pytest.mark.parametrize("value", ["40-50", [40, 50], {"start_min": 40, "end_min": 50}])
def test_window_forms(value) -> None:
    assert Window.model_validate(value).as_tuple() == (40, 50)


def test_typo_is_named_with_a_suggestion(tmp_path) -> None:
    with pytest.raises(ConfigError) as err:
        build_config({"experiments": [{"path": str(tmp_path)}], "metric": {"colony_sise": 30}})
    text = str(err.value)
    assert "metric.colony_sise" in text and "Did you mean 'colony_size'?" in text


def test_bad_values_say_what_was_given(tmp_path) -> None:
    with pytest.raises(ConfigError) as err:
        build_config(
            {
                "experiments": [{"path": str(tmp_path / "missing")}],
                "stimulus": {"on": "50-40"},
                "metric": {"colony_size": 0},
            }
        )
    text = str(err.value)
    assert "folder not found" in text
    assert "stimulus.on" in text and "must end after it starts" in text
    assert "metric.colony_size" in text and "(you gave 0)" in text


def test_overlapping_windows_refused(tmp_path) -> None:
    with pytest.raises(ConfigError, match="overlap"):
        build_config({"experiments": [{"path": str(tmp_path)}], "stimulus": {"off": "35-45"}})


def test_load_yaml_with_relative_paths(tmp_path) -> None:
    exp = tmp_path / "exp"
    exp.mkdir()
    cfg = tmp_path / "settings.yml"
    cfg.write_text("experiments:\n  - path: exp\n    frequency_hz: 475\nanalyses: [resting_fraction, psth]\n")
    c = load_config(cfg)
    assert c.experiments[0].path == exp.resolve() and c.analyses == ["resting_fraction", "psth"]
    bad = tmp_path / "bad.yml"
    bad.write_text("experiments: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yml")
    (tmp_path / "list.yml").write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="must contain settings"):
        load_config(tmp_path / "list.yml")


def test_phases_from_the_light_schedule() -> None:
    ph = LightSettings().phases()
    assert ph[5] == "dawn" and ph[6] == "day" and ph[15] == "day" and ph[16] == "dusk"
    assert ph[17] == "night" and ph[4] == "night"
    shifted = LightSettings(lights_on="07:00", lights_off="19:30").phases()
    assert shifted[7] == "dawn" and shifted[18] == "dusk" and shifted[19] == "night"


def test_experiment_window_must_be_ordered(tmp_path) -> None:
    with pytest.raises(ConfigError, match="must be after start"):
        build_config({"experiments": [{"path": str(tmp_path), "start": "2026-06-05", "end": "2026-06-04"}]})


CAGE = [(100.0, 50.0), (500.0, 50.0), (520.0, 330.0), (80.0, 330.0)]


def _cal(**kw) -> cal.Calibration:
    base: dict[str, object] = {
        "frame_width_px": 640,
        "frame_height_px": 360,
        "cage": CAGE,
        "zones": [
            cal.Zone(
                name="speaker", role="speaker", polygon=[(110, 150), (150, 150), (150, 230), (110, 230)]
            ),
            cal.Zone(name="right", role="control", polygon=[(450, 150), (490, 150), (490, 230), (450, 230)]),
        ],
    }
    base.update(kw)
    return cal.Calibration(**base)


def test_calibration_validation() -> None:
    c = _cal()
    assert c.speaker.name == "speaker" and [z.name for z in c.controls] == ["right"]
    with pytest.raises(ValueError, match="exactly one zone with role 'speaker'"):
        _cal(zones=[cal.Zone(name="a", role="control", polygon=[(1, 1), (2, 1), (2, 2)])])
    with pytest.raises(ValueError, match="outside the 640×360 frame"):
        _cal(cage=[(100.0, 50.0), (700.0, 50.0), (520.0, 330.0), (80.0, 330.0)])
    with pytest.raises(CalibrationError, match="No zone named"):
        c.polygon("nope")


def test_homography_maps_corners_to_millimetres() -> None:
    c = _cal(physical=cal.PhysicalSize(cage_width_mm=300, cage_height_mm=200))
    mm = c.to_mm(np.array(CAGE))
    assert np.allclose(mm, [[0, 0], [300, 0], [300, 200], [0, 200]], atol=1e-9)
    with pytest.raises(CalibrationError, match="no physical cage size"):
        _cal().to_mm(np.array([[1.0, 1.0]]))


def test_save_load_round_trip(tmp_path) -> None:
    c = _cal(notes="x")
    path = cal.save_calibration(c, tmp_path / cal.CALIBRATION_FILE)
    assert cal.load_calibration(path) == c
    path.write_text("{bad json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        cal.load_calibration(path)
    path.write_text(json.dumps({"frame_width_px": 640}))
    with pytest.raises(ConfigError, match="is required"):
        cal.load_calibration(path)
    with pytest.raises(CalibrationError, match="No calibration"):
        cal.load_calibration(tmp_path / "none.json")


def _experiment_with_images(tmp_path, size=(640, 360)):
    from PIL import Image

    exp = tmp_path / "exp"
    (exp / "individual_images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = (rng.random((size[1], size[0])) * 255).astype(np.uint8)
    Image.fromarray(img).save(exp / "individual_images" / "Cage_20260604_120009_00000.png")
    (exp / "custom_zones.json").write_text(
        json.dumps(
            {
                "speaker": {"polygon": [[110, 150], [150, 150], [150, 230], [110, 230]]},
                "right_zone": {"polygon": [[450, 150], [490, 150], [490, 230], [450, 230]]},
            }
        )
    )
    (exp / "buzzwatch_track_settings.yml").write_text(
        "cage_border_points:\n- [100, 50]\n- [500, 50]\n- [520, 330]\n- [80, 330]\n"
    )
    return exp


def test_import_existing_zones(tmp_path) -> None:
    exp = _experiment_with_images(tmp_path)
    c = cal.import_existing(exp)
    assert c.frame_size == (640, 360) and c.speaker.polygon[0] == (110.0, 150.0)
    assert [z.role for z in c.zones] == ["speaker", "control"]
    assert (
        c.reference_image == "individual_images/Cage_20260604_120009_00000.png"
        and len(c.reference_sha256) == 64
    )


def test_import_needs_its_inputs(tmp_path) -> None:
    exp = _experiment_with_images(tmp_path)
    (exp / "buzzwatch_track_settings.yml").write_text("fps: 25\n")
    with pytest.raises(CalibrationError, match="cage_border_points"):
        cal.import_existing(exp)
    (exp / "custom_zones.json").unlink()
    with pytest.raises(CalibrationError, match="No zone file"):
        cal.import_existing(exp)
    with pytest.raises(CalibrationError, match="no reference frame"):
        cal.import_existing(tmp_path / "empty")


def test_iou_and_phase_shift() -> None:
    a = np.array(CAGE)
    assert cal.polygon_iou(a, a, (640, 360)) == pytest.approx(1.0)
    assert cal.polygon_iou(a, a + np.array([600, 0]), (1400, 360)) == 0.0
    rng = np.random.default_rng(3)
    ref = rng.random((180, 320))
    from scipy.ndimage import gaussian_filter, shift

    ref = gaussian_filter(ref, 1.5) * 255
    moved = shift(ref, (4, -7), order=1, mode="nearest")
    dx, dy, peak = cal.phase_shift(ref, moved)
    assert (round(dx), round(dy)) == (-7, 4) and peak > 0.05
