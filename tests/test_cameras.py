"""Camera identity and discovery: nothing here opens a camera or needs the SDK.

Discovery is deliberately the dependency-free half of :mod:`openpi_control.cameras`
(it reads udev symlink names and nothing else), which is exactly what makes it
testable against a fake ``/dev/v4l/by-id`` -- and what lets ``doctor`` report an
unplugged camera on a box with no RealSense SDK installed.
"""

from __future__ import annotations

import numpy as np
import pytest

# The fake bus lives in its own module so test_cli can build one too, without
# importing another test module for it.
from tests_helpers_cameras import fake_by_id

from openpi_control.cameras import (
    DEFAULT_COLOR_INDEX,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    RigCamera,
    _rotate,
    discover,
    parse_camera_overrides,
    present_serials,
)
from openpi_control.exceptions import ConfigurationError


def camera(name, serial, **kwargs):
    return RigCamera(name=name, serial=serial, label=name.title(), **kwargs)


def test_present_serials_keys_every_node_by_serial_and_index(tmp_path) -> None:
    by_id = fake_by_id(tmp_path, ["254623070531", "254623070863"])

    nodes = present_serials(by_id)

    # Both cameras, all six nodes each -- the caller picks the colour one.
    assert len(nodes) == 12
    assert ("254623070531", DEFAULT_COLOR_INDEX) in nodes
    assert nodes[("254623070863", 0)].endswith("video-index0")


def test_present_serials_is_empty_when_udev_published_nothing(tmp_path) -> None:
    # No by-id directory means no cameras, not a crash: `doctor` has to be able
    # to run on a box with nothing plugged in.
    assert present_serials(tmp_path / "does-not-exist") == {}


def test_discovery_matches_each_camera_to_its_colour_node(tmp_path) -> None:
    by_id = fake_by_id(tmp_path, ["254623070531", "254623070863"])
    cameras = [camera("top", "254623070531"), camera("left_wrist", "254623070863")]

    result = discover(cameras, by_id_dir=by_id)

    assert result.complete
    assert set(result.matched) == {"top", "left_wrist"}
    assert result.matched["top"].device.endswith(f"video-index{DEFAULT_COLOR_INDEX}")
    assert not result.matched["top"].overridden


def test_discovery_reports_a_camera_that_is_not_on_the_bus(tmp_path) -> None:
    # The serial is carried into the report because that is the number an
    # operator needs to go looking for the camera that fell off.
    by_id = fake_by_id(tmp_path, ["254623070531"])

    result = discover(
        [camera("top", "254623070531"), camera("right_wrist", "254623070417")],
        by_id_dir=by_id,
    )

    assert not result.complete
    assert result.missing == {"right_wrist": "254623070417"}
    assert set(result.matched) == {"top"}


def test_discovery_reports_a_camera_nobody_claimed(tmp_path) -> None:
    # This is the informative half of "the top view is missing": a camera was
    # swapped, and the new serial has to reach the rig definition.
    by_id = fake_by_id(tmp_path, ["254623070531", "999999999999"])

    result = discover([camera("top", "254623070531")], by_id_dir=by_id)

    assert result.unclaimed == ("999999999999",)


def test_a_camera_on_a_different_colour_node_is_not_matched(tmp_path) -> None:
    # A model whose colour stream is not on index 4 must say so; silently
    # matching some other node would record depth as if it were colour.
    by_id = fake_by_id(tmp_path, ["254623070531"], indices=[0, 1])

    assert discover([camera("top", "254623070531")], by_id_dir=by_id).missing
    assert discover(
        [camera("top", "254623070531", color_index=1)], by_id_dir=by_id
    ).complete


def test_an_override_wins_over_discovery(tmp_path) -> None:
    by_id = fake_by_id(tmp_path, ["254623070531"])
    pinned = tmp_path / "video99"
    pinned.touch()

    result = discover(
        [camera("top", "254623070531")],
        overrides={"top": str(pinned)},
        by_id_dir=by_id,
    )

    assert result.matched["top"].device == str(pinned)
    assert result.matched["top"].overridden


def test_an_override_pointing_at_nothing_is_reported_not_ignored(tmp_path) -> None:
    # Falling back to the discovered device would hand the operator a camera
    # they did not ask for, which is worse than saying the pin is broken.
    by_id = fake_by_id(tmp_path, ["254623070531"])

    result = discover(
        [camera("top", "254623070531")],
        overrides={"top": str(tmp_path / "absent")},
        by_id_dir=by_id,
    )

    assert result.missing == {"top": "254623070531"}
    assert not result.matched


def test_an_override_for_an_unknown_camera_names_the_ones_that_exist(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="top"):
        discover(
            [camera("top", "254623070531")],
            overrides={"tpo": "/dev/video4"},
            by_id_dir=fake_by_id(tmp_path, []),
        )


def test_a_found_camera_carries_the_serial_and_capture_settings(tmp_path) -> None:
    # The SDK addresses a device by serial, so a spec that lost it cannot be
    # opened at all -- the device path alone is not enough.
    by_id = fake_by_id(tmp_path, ["254623070531"])
    declared = camera("top", "254623070531", rotate=180, fps=15)

    spec = discover([declared], by_id_dir=by_id).matched["top"].spec()

    assert spec.serial == "254623070531"
    assert (spec.name, spec.rotate, spec.fps) == ("top", 180, 15)
    assert (spec.width, spec.height) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert spec.fps != DEFAULT_FPS  # the declared override survived


def test_a_camera_rejects_a_serial_that_is_not_one() -> None:
    # A /dev path or an SDK serial pasted into the serial field would fail much
    # later, as "camera not on the bus", with no hint that the value is wrong.
    with pytest.raises(ConfigurationError, match="digits"):
        camera("top", "/dev/video4")


def test_a_camera_rejects_a_rotation_it_cannot_apply() -> None:
    with pytest.raises(ConfigurationError, match="rotate=45"):
        camera("top", "254623070531", rotate=45)


@pytest.mark.parametrize("bad", [{"width": 0}, {"height": -1}, {"fps": 0}])
def test_a_camera_rejects_a_nonsense_capture_size(bad) -> None:
    with pytest.raises(ConfigurationError, match="positive"):
        camera("top", "254623070531", **bad)


@pytest.mark.parametrize(
    ("rotate", "shape"),
    [(90, (848, 480, 3)), (180, (480, 848, 3)), (270, (848, 480, 3))],
)
def test_rotation_transposes_the_frame_and_keeps_it_contiguous(rotate, shape) -> None:
    # Video encoders want contiguous memory, and np.rot90 returns a strided
    # view, so a rotated frame that skipped the copy would fail deep in ffmpeg.
    frame = np.zeros((480, 848, 3), dtype=np.uint8)

    turned = _rotate(frame, rotate)

    assert turned.shape == shape
    assert turned.flags["C_CONTIGUOUS"]


def test_rotation_by_90_and_270_go_opposite_ways() -> None:
    # A wrist camera mounted sideways is fixed by exactly one of these; if both
    # did the same thing the flag would be a coin toss.
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[0, 0] = 255  # top-left corner

    clockwise = _rotate(frame, 90)
    counter = _rotate(frame, 270)

    assert clockwise[0, -1].tolist() == [255] * 3  # corner moved to top-right
    assert counter[-1, 0].tolist() == [255] * 3  # ... and to bottom-left


def test_camera_overrides_parse_like_interface_overrides() -> None:
    assert parse_camera_overrides(["top=/dev/video4", "left_wrist=/dev/video10"]) == {
        "top": "/dev/video4",
        "left_wrist": "/dev/video10",
    }
    assert parse_camera_overrides(None) == {}


@pytest.mark.parametrize("bad", ["top", "=/dev/video4", "top="])
def test_a_malformed_camera_override_is_refused(bad) -> None:
    with pytest.raises(ConfigurationError, match="NAME=DEVICE"):
        parse_camera_overrides([bad])


def test_a_serialless_by_id_name_is_not_read_as_a_camera(tmp_path) -> None:
    """A D435 with no USB serial must not be parsed into a camera called "435".

    Its udev name ends in the model number where a D405's ends in the serial
    (``..._Depth_Camera_435_..._Depth_Camera_435-video-index0``), and a lazy
    pattern reads that as serial ``435`` -- inventing a camera that is not there
    and, worse, claiming to have found the one that is.
    """
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    for index in (0, 1, 2, 3):
        (
            by_id
            / (
                "usb-Intel_R__RealSense_TM__Depth_Camera_435_"
                f"Intel_R__RealSense_TM__Depth_Camera_435-video-index{index}"
            )
        ).touch()

    assert present_serials(by_id) == {}


def test_a_camera_udev_cannot_name_is_found_through_the_sdk(tmp_path, monkeypatch) -> None:
    """Discovery falls back to the SDK for a camera with no by-id entry.

    Device paths are diagnostics here -- streams are opened by SDK serial -- so
    a camera the SDK can address is present even when udev cannot name it.
    """
    from openpi_control import cameras as cameras_mod

    by_id = fake_by_id(tmp_path, ["254623070863"])
    # The real system directory is what enables the SDK fallback; a redirected
    # bus is taken literally, so point BY_ID_DIR at the fake and pass no dir.
    monkeypatch.setattr(cameras_mod, "BY_ID_DIR", by_id)
    monkeypatch.setattr(cameras_mod, "SYSTEM_BY_ID_DIR", by_id)
    monkeypatch.setattr(
        cameras_mod, "sdk_present_asic_serials", lambda: {"348523020354": "243622071623"}
    )

    result = discover([camera("top", "348523020354"), camera("left_wrist", "254623070863")])

    assert result.complete
    assert result.matched["top"].device == "sdk:243622071623"
    assert result.matched["left_wrist"].device.endswith("video-index4")
    assert result.unclaimed == ()


def test_a_redirected_bus_is_taken_literally(tmp_path, monkeypatch) -> None:
    """Pointing discovery at a directory means that directory is the whole bus.

    Otherwise whatever happens to be plugged into the machine running the tests
    decides whether a camera is "missing".
    """
    from openpi_control import cameras as cameras_mod

    def _boom():  # pragma: no cover - must never be called
        raise AssertionError("the SDK must not be consulted behind a redirected bus")

    monkeypatch.setattr(cameras_mod, "sdk_present_asic_serials", _boom)

    result = discover([camera("top", "348523020354")], by_id_dir=fake_by_id(tmp_path, []))

    assert result.missing == {"top": "348523020354"}
