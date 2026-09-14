"""Native calibration units, persistence, and profile propagation."""

import json

import numpy as np
import pytest

from openpi_control import cli, safety
from openpi_control.exceptions import ConfigurationError
from openpi_control.rigs import resolve_rig
from openpi_control.so101_calibration import (
    TICK,
    apply_profiles,
    build_profile,
    load_profile,
    verify_firmware,
)


def samples():
    rig = resolve_rig("so101")
    low, high = safety.joint_limits(rig)["right"]
    # A measured mechanical midpoint shifted relative to factory firmware zero.
    center = np.array([1900.0, 2000.0, 1850.0, 2100.0, 2048.0])
    width = (high - low) / TICK
    minimum = np.r_[center - width / 2, 1500.0]
    maximum = np.r_[center + width / 2, 2500.0]
    home = np.r_[center + np.array([20.0, -50.0, 70.0, 10.0, 30.0]), 2000.0]
    return minimum, maximum, home


def test_offsets_home_and_gripper_use_native_units():
    low, high, home = samples()
    profile = build_profile(low, high, home, [0] * 6)
    for i, joint in enumerate(profile["arm_instance"]["joints"]):
        servo = joint["servos"][0]
        q = (home[i] - 2048) * TICK - servo["zero_pos"]
        assert q == pytest.approx(servo["home_pos"])
        assert servo["pos_min"] <= q <= servo["pos_max"]
    grip = profile["effector_instance"]["joints"][0]["servos"][0]
    assert (low[5] - 2048) * TICK - grip["zero_pos"] == pytest.approx(0.0)
    assert (high[5] - 2048) * TICK - grip["zero_pos"] == pytest.approx(grip["pos_max"])
    assert grip["home_pos"] / grip["pos_max"] == pytest.approx(0.5)


def test_bad_sweeps_and_home_are_rejected():
    low, high, home = samples()
    with pytest.raises(ConfigurationError, match="travel"):
        build_profile(low, low + 10, low + 5, [0] * 6)
    with pytest.raises(ConfigurationError, match="home pose"):
        build_profile(low, high, high + 10, [0] * 6)
    with pytest.raises(ConfigurationError, match="finite"):
        build_profile([float("nan")] * 6, high, home, [0] * 6)


def test_profile_propagates_offsets_home_and_tighter_limits(tmp_path):
    low, high, home = samples()
    # Narrow the measured range to verify all consumers see the intersection.
    low[:5] += 30
    high[:5] -= 30
    profile = build_profile(low, high, home, [0] * 6)
    path = tmp_path / "right.json"
    path.write_text(json.dumps(profile))
    rig = apply_profiles(resolve_rig("so101"), {"right": str(path)})
    config = rig.arms[0].arm_config()
    assert config.instance_config.name == "right.arm.json"
    assert config.effector_instance_config.name == "right.gripper.json"
    lower, upper = safety.joint_limits(rig)["right"]
    for i, joint in enumerate(profile["arm_instance"]["joints"]):
        servo = joint["servos"][0]
        assert lower[i] == pytest.approx(servo["pos_min"])
        assert upper[i] == pytest.approx(servo["pos_max"])
    # Embedded native configs cannot override computed calibration limits.
    profile["arm_instance"]["joints"][0]["servos"][0]["pos_max"] = 999
    path.write_text(json.dumps(profile))
    assert load_profile(path)["arm_instance"]["joints"][0]["servos"][0]["pos_max"] < 3.2


def test_firmware_changes_rejected_before_native_start(monkeypatch, tmp_path):
    from openpi_control import so101_calibration as cal

    low, high, home = samples()
    path = tmp_path / "right.json"
    path.write_text(json.dumps(build_profile(low, high, home, [0] * 6)))
    rig = apply_profiles(resolve_rig("so101"), {"right": str(path)})

    class Bus:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(cal, "_open_bus", lambda *args, **kwargs: Bus())
    monkeypatch.setattr(cal, "_read", lambda *args: 1)
    with pytest.raises(ConfigurationError, match="homing offset changed"):
        verify_firmware(rig)


def test_calibration_cli_is_available_without_lerobot(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path / "logs"))
    captured = {}
    monkeypatch.setattr(
        cli, "_command_calibrate_so101", lambda args, log: captured.update(args=args) or 0
    )
    assert (
        cli.main(
            [
                "calibrate-so101",
                "--interface",
                "/dev/ttyACM0",
                "--output",
                str(tmp_path / "right.json"),
            ]
        )
        == 0
    )
    assert captured["args"].interface == "/dev/ttyACM0"


def test_calibrated_home_and_limits_reach_quest_runtime(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace

    from openpi_control import so101_calibration as cal
    from openpi_control import teleop_runtime

    low, high, home = samples()
    path = tmp_path / "right.json"
    profile = build_profile(low, high, home, [0] * 6)
    path.write_text(json.dumps(profile))
    rig = apply_profiles(resolve_rig("so101"), {"right": str(path)})
    monkeypatch.setattr(cal, "verify_firmware", lambda rig: None)
    live = SimpleNamespace(name="right", rig_arm=rig.arms[0], arm=object())
    monkeypatch.setattr(cli, "power_up", lambda *args, **kwargs: (object(), [live]))
    monkeypatch.setattr(cli, "settle_arm_states", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "power_down", lambda *args, **kwargs: 0)
    captured = {}

    class Source:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def describe(self):
            return "test"

        def close(self):
            pass

    monkeypatch.setattr(teleop_runtime, "QuestTeleopSource", Source)
    stop = threading.Event()
    stop.set()
    assert teleop_runtime.run_teleop(rig, stop=stop) == 0
    config = captured["config_overrides"]
    expected = [j["servos"][0]["home_pos"] for j in profile["arm_instance"]["joints"]]
    np.testing.assert_allclose(config["rest_qpos_right"], expected)
    assert len(config["joint_limits_right"]) == 5


def test_encoder_centering_backs_up_before_writes_and_verifies_torque_off(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from openpi_control import so101_calibration as cal

    output = tmp_path / "right.json"
    completed = []
    disabled = []

    def read(bus, sid, address, length=2):
        if address == 56:
            return 2048
        if address == 40:
            return 0
        return sid + (100 if sid in completed else 0)

    def set_zero(bus, sid):
        backups = list(tmp_path.glob("*.firmware-before-*.json"))
        assert len(backups) == 1
        assert json.loads(backups[0].read_text())["motors"]["wrist_roll"]["homing_offset_raw"] == 5
        completed.append(sid)

    monkeypatch.setattr(cal, "_read", read)
    monkeypatch.setattr(cal.ft_serial, "set_zero", set_zero)
    monkeypatch.setattr(
        cal.ft_serial,
        "torque_enable",
        lambda bus, sid, value: disabled.append((sid, value)) or True,
    )
    offsets = cal.center_encoders(SimpleNamespace(port="/dev/fake"), output)
    assert offsets == list(range(101, 107))
    assert disabled == [(i, False) for i in range(1, 7)]


def test_encoder_centering_stops_on_failed_acknowledgement(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from openpi_control import so101_calibration as cal

    calls = []

    def read(bus, sid, address, length=2):
        return 2048 if address == 56 else 0

    def set_zero(bus, sid):
        calls.append(sid)
        return "no acknowledgement" if sid == 2 else None

    monkeypatch.setattr(cal, "_read", read)
    monkeypatch.setattr(cal.ft_serial, "set_zero", set_zero)
    monkeypatch.setattr(cal.ft_serial, "torque_enable", lambda *a: True)
    with pytest.raises(ConfigurationError, match="some offsets may have changed"):
        cal.center_encoders(SimpleNamespace(port="/dev/fake"), tmp_path / "right.json")
    assert calls == [1, 2]


def test_centered_wizard_skips_roll_wrap_and_saves_post_centering_offsets(monkeypatch, tmp_path):
    from openpi_control import so101_calibration as cal

    low, high, home = samples()
    # Roll crosses the encoder boundary while other joints complete a valid sweep.
    low[4], high[4], home[4] = 4090, 4, 2048
    middle = (low + high) / 2
    middle[4] = 4090
    readings = iter([home.copy(), home.copy(), low, middle, high])
    stable = iter([home, low, high, home])  # centering, closed, open, home

    class Bus:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(cal, "_open_bus", lambda *a: Bus())
    monkeypatch.setattr(cal, "_positions", lambda *a: next(readings))
    monkeypatch.setattr(cal, "_stable_positions", lambda *a, **kw: next(stable))
    monkeypatch.setattr(cal, "_read", lambda *a: 0)
    monkeypatch.setattr(cal.ft_serial, "torque_enable", lambda *a: True)
    monkeypatch.setattr(cal, "center_encoders", lambda *a: [123] * 6)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    checks = iter([([], [], []), ([], [], []), ([], [], []), ([True], [], [])])
    monkeypatch.setattr(cal.select, "select", lambda *a: next(checks))
    monkeypatch.setattr(cal.time, "sleep", lambda *a: None)
    path = tmp_path / "right.json"
    assert cal.run_calibration("/dev/fake", path, center=True) == 0
    profile = cal.load_profile(path)
    roll = profile["motors"]["wrist_roll"]
    assert (roll["range_min"], roll["range_max"]) == (0, 4095)
    assert roll["homing_offset_raw"] == 123


@pytest.mark.parametrize("reversed_gripper", [False, True])
def test_gripper_endpoint_direction_roundtrip(tmp_path, reversed_gripper):
    low, high, home = samples()
    closed = high[5] if reversed_gripper else low[5]
    opened = low[5] if reversed_gripper else high[5]
    profile = build_profile(low, high, home, [0] * 6, gripper_closed=closed)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    servo = load_profile(path)["effector_instance"]["joints"][0]["servos"][0]
    for raw, expected in [(closed, 0), (opened, servo["pos_max"])]:
        q = ((raw - 2048) * TICK - servo["zero_pos"]) * servo["dir_invert"]
        assert q == pytest.approx(expected)
        assert (q * servo["dir_invert"] + servo["zero_pos"]) / TICK + 2048 == pytest.approx(raw)


@pytest.mark.parametrize("opened", [2941, 2589, 2812])
def test_hand_captured_endpoints_need_not_match_sweep(opened):
    from openpi_control.so101_calibration import _gripper_range

    assert _gripper_range(1376, opened, 1497) == (1376, opened)
    assert _gripper_range(opened, 1376, 1497) == (1376, opened)


def test_endpoint_capture_rejects_same_pose():
    from openpi_control.so101_calibration import _gripper_range

    with pytest.raises(ConfigurationError, match="travel"):
        _gripper_range(1376, 1380, 1497)


def test_gripper_capture_ignores_arm_motion_and_retries_gripper_motion(monkeypatch):
    from openpi_control import so101_calibration as cal

    frames = [np.array([i * 50] * 5 + [1000 + i * 20]) for i in range(10)]
    frames += [np.array([i * 50] * 5 + [1000 + i % 3]) for i in range(10)]
    readings = iter(frames)
    prompts = []
    monkeypatch.setattr(cal, "_positions", lambda bus: next(readings))
    monkeypatch.setattr(cal.time, "sleep", lambda *a: None)
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt))
    assert cal._stable_positions(None, gripper_only=True)[5] == 1001
    assert len(prompts) == 1
