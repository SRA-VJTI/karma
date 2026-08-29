"""The vr-teleop-kit bridge, without vr-teleop-kit, a headset, or a relay.

The adapter itself is small, but it carries the two things in this pipeline most
worth pinning down and least testable against real hardware: the gripper
inversion between the two projects, and the button edge detection that decides
whether a held button restarts an episode thirty times a second.
"""

from __future__ import annotations

import time

import pytest

from openpi_control.exceptions import ConfigurationError
from openpi_control.record import (
    DATASET_GRIPPER_CLOSED,
    DATASET_GRIPPER_OPEN,
    NATIVE_GRIPPER_CLOSED,
    NATIVE_GRIPPER_OPEN,
    EpisodeEvent,
)
from openpi_control.teleop_vr import VR_ARM_DOFS, QuestTeleopSource
from openpi_control.types import (
    ArmMode,
    ArmRole,
    ArmState,
    EffectorState,
    JointState,
)


class FakeTeleoperator:
    """Stands in for vr-teleop-kit's BiQuestTeleoperator.

    Faithful on the three things the bridge actually touches: the ``.pos``-keyed
    bimanual action dict, the two buttons reported as *levels*, and
    ``seed_qpos_from_obs``.
    """

    def __init__(self, *, joint: float = 0.5, gripper: float = DATASET_GRIPPER_OPEN):
        self.joint = joint
        self.gripper = gripper
        self.start_pressed = False
        self.save_pressed = False
        self.seeds: list[dict[str, float]] = []
        self.disconnects = 0
        self.action_calls = 0

    def get_action(self) -> dict[str, float]:
        self.action_calls += 1
        action: dict[str, float] = {}
        # Always bimanual, whether or not the session holds both arms.
        for hand in ("left", "right"):
            for index in range(VR_ARM_DOFS):
                action[f"{hand}_joint_{index + 1}.pos"] = self.joint
            action[f"{hand}_gripper.pos"] = self.gripper
        return action

    def is_pause_pressed(self) -> bool:
        return self.start_pressed

    def is_reverse_pressed(self) -> bool:
        return self.save_pressed

    def seed_qpos_from_obs(self, observation: dict[str, float]) -> None:
        self.seeds.append(dict(observation))

    def disconnect(self) -> None:
        self.disconnects += 1


def source(arm_names=("left", "right"), **kwargs) -> tuple[QuestTeleopSource, FakeTeleoperator]:
    teleop = FakeTeleoperator(**kwargs)
    return QuestTeleopSource(arm_names, teleoperator=teleop), teleop


def state(position: float = 0.25, effector: float | None = NATIVE_GRIPPER_OPEN) -> ArmState:
    return ArmState(
        name="fake",
        role=ArmRole.FOLLOWER,
        joints=JointState(
            names=tuple(f"joint_{i + 1}" for i in range(VR_ARM_DOFS)),
            position_rad=[position] * VR_ARM_DOFS,
            velocity_rad_s=[0.0] * VR_ARM_DOFS,
            effort_nm=[0.0] * VR_ARM_DOFS,
            temperature_c=[25.0] * VR_ARM_DOFS,
            current_a=[0.0] * VR_ARM_DOFS,
        ),
        effector=EffectorState(position=effector) if effector is not None else None,
        monotonic_timestamp=time.monotonic(),
        wall_timestamp=0.0,
        sequence=1,
        mode=ArmMode.HOLD,
    )


# --------------------------------------------------------------------------- #
# the gripper inversion
# --------------------------------------------------------------------------- #


def test_a_released_trigger_becomes_an_open_gripper() -> None:
    # The trigger speaks LeRobot's convention (0 = open); this package's
    # effector is the opposite. Getting this backwards means the trigger opens
    # the gripper instead of closing it.
    vr, _ = source(gripper=DATASET_GRIPPER_OPEN)

    step = vr.poll({"left": state(), "right": state()})

    assert step.targets["left"].effector == pytest.approx(NATIVE_GRIPPER_OPEN)


def test_a_squeezed_trigger_becomes_a_closed_gripper() -> None:
    vr, _ = source(gripper=DATASET_GRIPPER_CLOSED)

    step = vr.poll({"left": state(), "right": state()})

    assert step.targets["left"].effector == pytest.approx(NATIVE_GRIPPER_CLOSED)


def test_a_partial_trigger_squeeze_stays_continuous() -> None:
    vr, _ = source(gripper=0.4)

    step = vr.poll({"left": state(), "right": state()})

    # Dataset/trigger 0.4 closed maps to native 0.6 open; it is not treated as
    # a binary button press.
    assert step.targets["left"].effector == pytest.approx(0.6)


def test_the_seed_converts_the_gripper_the_other_way() -> None:
    # seed_qpos_from_obs speaks the teleoperator's convention, so a fully open
    # gripper (native 1.0) has to arrive there as 0.0.
    vr, teleop = source()

    vr.poll({"left": state(effector=NATIVE_GRIPPER_OPEN), "right": state()})

    assert teleop.seeds[0]["left_gripper.pos"] == pytest.approx(DATASET_GRIPPER_OPEN)


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #


def test_the_first_poll_seeds_the_solver_from_the_measured_pose() -> None:
    # Unseeded, the teleoperator starts at its rest pose, so the first command
    # would be a full-speed move from wherever the arm actually is.
    vr, teleop = source()

    vr.poll({"left": state(position=0.42), "right": state(position=0.42)})

    assert len(teleop.seeds) == 1
    assert teleop.seeds[0]["left_joint_1.pos"] == pytest.approx(0.42)
    assert teleop.seeds[0]["right_joint_6.pos"] == pytest.approx(0.42)


def test_seeding_happens_once_not_every_tick() -> None:
    # Re-seeding mid-session would force-disengage the clutch every tick and
    # the operator could never drive the arm anywhere.
    vr, teleop = source()

    for _ in range(5):
        vr.poll({"left": state(), "right": state()})

    assert len(teleop.seeds) == 1


def test_nothing_is_commanded_before_any_arm_has_reported() -> None:
    # Commanding nothing leaves the arms holding, which is the only safe thing
    # to do without knowing where they are.
    vr, teleop = source()

    step = vr.poll({"left": None, "right": None})

    assert step.targets == {}
    assert step.event is EpisodeEvent.NONE
    assert teleop.action_calls == 0  # the solver was never asked


def test_a_late_arm_still_gets_seeded() -> None:
    vr, teleop = source()

    vr.poll({"left": None, "right": None})
    step = vr.poll({"left": state(), "right": state()})

    assert len(teleop.seeds) == 1
    assert step.targets


def test_a_bimanual_seed_waits_for_both_selected_arm_states() -> None:
    vr, teleop = source()

    step = vr.poll({"left": state(), "right": None})

    assert step.targets == {}
    assert teleop.seeds == []

    step = vr.poll({"left": state(), "right": state()})
    assert step.targets
    assert len(teleop.seeds) == 1


# --------------------------------------------------------------------------- #
# buttons
# --------------------------------------------------------------------------- #


def test_holding_the_start_button_starts_one_episode_not_thirty() -> None:
    # Both buttons are reported as levels. Without edge detection, a held
    # button would discard and restart the take on every single tick.
    vr, teleop = source()
    states = {"left": state(), "right": state()}
    vr.poll(states)  # seed

    teleop.start_pressed = True
    events = [vr.poll(states).event for _ in range(4)]

    assert events == [EpisodeEvent.START] + [EpisodeEvent.NONE] * 3


def test_releasing_and_pressing_again_starts_another_episode() -> None:
    vr, teleop = source()
    states = {"left": state(), "right": state()}
    vr.poll(states)

    teleop.start_pressed = True
    assert vr.poll(states).event is EpisodeEvent.START
    teleop.start_pressed = False
    assert vr.poll(states).event is EpisodeEvent.NONE
    teleop.start_pressed = True
    assert vr.poll(states).event is EpisodeEvent.START


def test_the_save_button_saves() -> None:
    vr, teleop = source()
    states = {"left": state(), "right": state()}
    vr.poll(states)

    teleop.save_pressed = True

    assert vr.poll(states).event is EpisodeEvent.SAVE


def test_start_wins_when_both_buttons_are_down() -> None:
    # Arbitrary but fixed: an ambiguous double-press must not depend on
    # iteration order, or the operator sees a coin toss.
    vr, teleop = source()
    states = {"left": state(), "right": state()}
    vr.poll(states)

    teleop.start_pressed = teleop.save_pressed = True

    assert vr.poll(states).event is EpisodeEvent.START


# --------------------------------------------------------------------------- #
# arm filtering
# --------------------------------------------------------------------------- #


def test_a_one_armed_session_ignores_the_other_arm_s_targets() -> None:
    # The teleoperator is always bimanual; a target for an arm this session does
    # not hold would be rejected by the record loop.
    vr, _ = source(arm_names=("right",))

    step = vr.poll({"right": state()})

    assert set(step.targets) == {"right"}
    assert len(step.targets["right"].position_rad) == VR_ARM_DOFS


def test_an_arm_the_teleoperator_cannot_drive_is_refused_up_front() -> None:
    # Better at construction than as a silently unmoving arm mid-session.
    with pytest.raises(ConfigurationError, match="drives arms named left, right"):
        source(arm_names=("gantry",))


def test_a_partial_action_is_an_error_not_a_half_pose() -> None:
    class Truncated(FakeTeleoperator):
        def get_action(self):
            action = super().get_action()
            del action["left_joint_3.pos"]
            return action

    vr = QuestTeleopSource(("left",), teleoperator=Truncated())

    # The error names the key that was missing, which is what makes a protocol
    # change diagnosable rather than just "the count was wrong".
    with pytest.raises(ConfigurationError, match="no left_joint_3.pos"):
        vr.poll({"left": state()})


# --------------------------------------------------------------------------- #
# teardown
# --------------------------------------------------------------------------- #


def test_closing_releases_the_relay_connection() -> None:
    vr, teleop = source()

    vr.close()

    assert teleop.disconnects == 1


def test_a_failure_on_disconnect_does_not_mask_the_session_result() -> None:
    class Stuck(FakeTeleoperator):
        def disconnect(self):
            raise RuntimeError("websocket already gone")

    QuestTeleopSource(("left",), teleoperator=Stuck()).close()  # must not raise


def test_the_description_names_the_relay_and_the_arms() -> None:
    vr, _ = source(arm_names=("left",))

    assert "left" in vr.describe()
    assert "ws://" in vr.describe()
