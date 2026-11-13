import dataclasses
import json
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from fake_arm_backend import FakeArmBackend

from openpi_control import cli
from openpi_control.exceptions import ConfigurationError, StaleStateError
from openpi_control.record import EpisodeEvent, MemorySink
from openpi_control.teaching import (
    FrictionJoint,
    NextStateActionSink,
    TeachingProfile,
    TeachingSource,
    example_config,
    load_profiles,
)
from openpi_control.types import ArmMode


class TeachingArm:
    def __init__(self):
        self.capabilities = SimpleNamespace(supports_teaching=True)
        self.mode = ArmMode.HOLD
        self.holds = self.renewals = 0
        self.fresh = True
        self.fail_enter = False

    def configure_teaching(self, profile):
        self.profile = profile

    def enter_teaching(self):
        if self.fail_enter:
            raise RuntimeError("enter failed")
        self.mode = ArmMode.TEACHING

    def renew_teaching(self):
        self.renewals += 1

    def hold(self):
        self.holds += 1
        self.mode = ArmMode.HOLD

    def state(self):
        backend = FakeArmBackend()
        backend.connected, backend._has_effector = True, True
        return dataclasses.replace(
            backend.latest_state(),
            mode=self.mode,
            monotonic_timestamp=time.monotonic() - (0 if self.fresh else 1),
        )


def make_source(events, *, record=True):
    arms = {name: TeachingArm() for name in ("left", "right")}
    profile = TeachingProfile(tuple(FrictionJoint() for _ in range(7)))
    events = deque(events)
    source = TeachingSource(arms, dict.fromkeys(arms, profile), events.popleft, allow_record=record)
    return source, arms


def poll(source, arms):
    return source.poll({name: arm.state() for name, arm in arms.items()})


def test_profiles_zero_assistance_by_default_and_explicit_per_arm(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(example_config()))
    profiles = load_profiles(path, ("left", "right"))
    assert len(profiles["left"].wire_values()) == 7
    assert profiles["left"].joints[6].torque(1) == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coulomb_nm": -1},
        {"transition_rad_s": 0},
        {"coulomb_nm": float("nan")},
        {"damping_nm_s": -1},
        {"velocity_limit_rad_s": 4},
        {"compensation_limit_nm": 3},
        {"coulomb_nm": 0.1, "compensation_limit_nm": 0},
        {"damping_nm_s": True},
    ],
)
def test_invalid_profiles_rejected(kwargs):
    with pytest.raises(ConfigurationError):
        TeachingProfile(tuple(FrictionJoint(**kwargs) for _ in range(7)))


def test_gripper_has_lower_assistance_cap():
    with pytest.raises(ConfigurationError):
        TeachingProfile((*[FrictionJoint()] * 6, FrictionJoint(compensation_limit_nm=0.6)))


def test_smooth_friction_is_odd_zero_at_rest_and_capped():
    joint = FrictionJoint(
        coulomb_nm=0.1, viscous_nm_s=0.1, compensation_limit_nm=0.2, damping_nm_s=0
    )
    assert joint.torque(0) == 0
    assert joint.torque(-0.1) == -joint.torque(0.1)
    assert joint.torque(100) == 0.2
    assert abs(joint.torque(1e-6)) < 1e-4


def test_teaching_source_never_issues_position_targets_and_holds_before_save():
    source, arms = make_source(["r", "", "s"])
    first = poll(source, arms)
    assert first.event is EpisodeEvent.START and not first.targets
    assert not poll(source, arms).targets
    assert poll(source, arms).event is EpisodeEvent.SAVE
    assert all(a.mode is ArmMode.HOLD and a.holds == 1 and a.renewals == 2 for a in arms.values())


@pytest.mark.parametrize("fault", ["stale", "native_hold", "partial_enter"])
def test_failures_hold_both_arms(fault):
    source, arms = make_source(["t", ""])
    if fault == "partial_enter":
        arms["right"].fail_enter = True
    else:
        poll(source, arms)
        if fault == "stale":
            arms["right"].fresh = False
        else:
            arms["right"].mode = ArmMode.HOLD
    with pytest.raises((RuntimeError, StaleStateError)):
        poll(source, arms)
    assert all(a.mode is ArmMode.HOLD and a.holds >= 1 for a in arms.values())


def test_old_native_runtime_refused_before_configuration():
    arm = TeachingArm()
    arm.capabilities.supports_teaching = False
    with pytest.raises(ConfigurationError, match="rebuild"):
        TeachingSource({"left": arm}, {}, lambda: "t")
    assert not hasattr(arm, "profile")


def frame(q, timestamp):
    return {
        "observation.state": np.array([q], np.float32),
        "action": np.array([q]),
        "observation.capture_monotonic_s": np.array([timestamp]),
        "observation.images.top": np.zeros((2, 2, 3), np.uint8),
        "task": "pick",
    }


def test_measured_next_action_labels_keep_image_at_previous_time_and_drop_tail():
    raw = MemorySink()
    sink = NextStateActionSink(raw)
    first = frame(0, 10)
    sink.add_frame(first)
    first["observation.images.top"][:] = 255
    sink.add_frame(frame(0.2, 10.04))
    sink.add_frame(frame(0.3, 10.08))
    sink.save_episode()
    assert len(raw.episodes[0]) == 2
    assert raw.episodes[0][0]["action"].tolist() == pytest.approx([0.2])
    assert raw.episodes[0][0]["observation.state"].tolist() == [0]
    assert raw.episodes[0][0]["action.dt_s"].tolist() == pytest.approx([0.04])
    assert not raw.episodes[0][0]["observation.images.top"].any()


def test_discard_and_short_episode_do_not_leak_next_actions():
    raw = MemorySink()
    sink = NextStateActionSink(raw)
    sink.add_frame(frame(1, 1))
    sink.discard_episode()
    sink.add_frame(frame(2, 2))
    sink.save_episode()
    assert raw.num_episodes == 0
    sink.add_frame(frame(3, 3))
    sink.add_frame(frame(4, 4))
    sink.save_episode()
    assert raw.episodes[0][0]["observation.state"].tolist() == [3]


def test_capture_time_must_increase():
    sink = NextStateActionSink(MemorySink())
    sink.add_frame(frame(1, 1))
    with pytest.raises(ConfigurationError, match="increase"):
        sink.add_frame(frame(2, 1))


def test_write_profile_never_opens_motor_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("KARMA_LOG_DIR", str(tmp_path / "logs"))

    def forbidden(*a, **k):
        pytest.fail("profile generation must not energize anything")

    monkeypatch.setattr(cli, "power_up", forbidden)
    path = tmp_path / "teaching.json"
    assert cli.main(["teach", "--write-config", str(path)]) == 0
    assert load_profiles(path, ("left", "right"))


def test_teaching_mode_abi_is_append_only():
    assert list(ArmMode).index(ArmMode.HOLD) == 2
    assert list(ArmMode).index(ArmMode.TEACHING) == 9
