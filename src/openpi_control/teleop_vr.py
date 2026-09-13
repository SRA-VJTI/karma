"""Drive this cell's arms from a Meta Quest.

The vendored :mod:`vr_teleop_kit` package solves the hard half of VR
teleoperation: the WebXR relay, clutch-relative pose mapping, and a damped IK
solver selected for YAM or SO101. This module is the boundary between that
package and :mod:`openpi_control`: it turns the Quest action stream into the
native stack's :class:`~openpi_control.record.TeleopSource` protocol, so the
headset drives arms through ``pi_control_node`` rather than through the kit's
optional i2rt example driver.

    openpi relay         <--WebXR--  Quest
             |
        BiQuestTeleoperator  (pose mapping + IK)
             |  joint targets
        QuestTeleopSource    (this module)
             |  PositionCommand
        FollowerArm -> pi_control_node -> CAN/serial -> YAM/SO101

The YAM MJCF is still supplied by the i2rt model tree because it is a robot
model asset, not a Python dependency. Set ``YAM_XML`` or place an i2rt checkout
at ``./i2rt``. ``openpi teleop`` starts the vendored relay and can establish the
Quest USB tunnel itself. SO101 loads its packaged URDF and needs no i2rt model.

Three things that will otherwise cost you a session
---------------------------------------------------

**The gripper polarity is inverted between the two projects.** The Quest trigger
speaks LeRobot's convention (0 open, 1 closed); this package's effector speaks
the opposite (1 open, 0 closed). :func:`~openpi_control.record.to_native_gripper`
is the one place that flips, and a mistake here means the trigger opens the
gripper.

**The teleoperator must be seeded from the arms' real pose before it commands
anything.** It starts at its configured rest pose, so an unseeded first command
is a full-speed move from wherever the arm actually is to wherever the IK thinks
it is. :meth:`QuestTeleopSource.poll` seeds itself on its first call for exactly
this reason.

**It is always bimanual.** ``BiQuestTeleoperator`` emits ``left_*`` and
``right_*`` keys whether or not both arms exist, so targets for arms this session
does not hold are dropped rather than passed on to a session that would reject
them.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .exceptions import ConfigurationError
from .record import (
    ArmTarget,
    EpisodeEvent,
    TeleopStep,
    to_dataset_gripper,
    to_native_gripper,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from .types import ArmState

# The relay's default WebSocket endpoint, matching vr-teleop-kit's own default.
DEFAULT_WS_URL = "ws://127.0.0.1:8443/ws"

# The teleoperator's arm keys. Fixed in vr-teleop-kit, and they happen to match
# this package's `yam_bimanual` arm names -- which is why the bridge can map by
# name instead of by position.
VR_HANDS = ("left", "right")

# Joints per arm in the teleoperator's action dict. A YAM has six; the gripper
# is reported separately as `<hand>_gripper.pos`.
VR_ARM_DOFS = 6


class QuestTeleopSource:
    """A :class:`~openpi_control.record.TeleopSource` fed by a Quest headset.

    Owns the ``BiQuestTeleoperator`` connection for the life of a session:
    constructing this opens the WebSocket to the relay, and :meth:`close`
    is what releases it.
    """

    def __init__(
        self,
        arm_names: Sequence[str],
        *,
        ws_url: str = DEFAULT_WS_URL,
        robot_model: str = "Yam",
        kit_path: Path | None = None,
        model_path: str | None = None,
        connect_timeout_s: float = 5.0,
        config_overrides: Mapping[str, object] | None = None,
        teleoperator: object | None = None,
        emit_episode_events: bool = True,
    ) -> None:
        """``teleoperator`` injects an already-built (or stand-in) teleoperator.

        That seam exists because the adapter carries the gripper inversion and
        the button edge detection -- the two things here most worth testing, and
        the two least testable against a real headset.

        ``emit_episode_events=False`` leaves B and Y alone. A DAgger session
        spends the same two buttons on taking control and giving it back, and
        two edge detectors reading one button would have a single press both
        start an intervention and restart the episode.
        """
        if robot_model not in ("Yam", "SO101"):
            raise ConfigurationError(f"unsupported Quest robot model: {robot_model}")
        if robot_model == "SO101" and model_path:
            raise ConfigurationError("SO101 uses its packaged URDF; --yam-xml is YAM-only")
        self.robot_model = robot_model
        self.arm_dofs = 5 if robot_model == "SO101" else VR_ARM_DOFS
        self.arm_names = tuple(arm_names)
        unknown = set(self.arm_names).difference(VR_HANDS)
        if unknown:
            raise ConfigurationError(
                f"the Quest teleoperator drives arms named {', '.join(VR_HANDS)}; "
                f"this rig has {', '.join(sorted(unknown))}. Rename the rig's arms "
                "or add a mapping before recording."
            )

        self._ws_url = ws_url
        if teleoperator is not None:
            self._teleop = teleoperator
        else:
            teleop_module = _import_vr_kit(kit_path)
            config_kwargs: dict[str, object] = dict(config_overrides or {})
            config_kwargs.update(
                {
                    "ws_url": ws_url,
                    "robot_model": robot_model,
                    "connect_timeout_s": connect_timeout_s,
                }
            )
            if model_path:
                config_kwargs["model_path"] = model_path

            self._teleop = teleop_module.BiQuestTeleoperator(
                teleop_module.BiQuestTeleoperatorConfig(**config_kwargs)
            )
            try:
                self._teleop.connect()
            except Exception as err:
                # The overwhelmingly common cause is that the relay is not
                # running, and a bare timeout traceback does not say so.
                raise ConfigurationError(
                    f"cannot reach the VR relay at {ws_url}: {err}. Start it with "
                    "`openpi relay`, or point --vr-url at wherever it is listening."
                ) from err
        self._seeded = False
        self._emit_episode_events = bool(emit_episode_events)
        # Both buttons are reported as levels, so the bridge does its own edge
        # detection: holding a button must not restart an episode every tick.
        self._last_start = False
        self._last_save = False

    def describe(self) -> str:
        return (
            f"Quest teleoperator via {self._ws_url} "
            f"({self.robot_model}; arms: {', '.join(self.arm_names)})"
        )

    def seed_from(
        self,
        states: Mapping[str, ArmState | None],
        *,
        effector: Mapping[str, float] | None = None,
    ) -> bool:
        """Anchor the IK at the arms' measured pose.

        Returns False when no arm has reported yet, so the caller can wait
        rather than seed the solver with zeros -- which would make the first
        command a move to the folded park pose.

        ``effector`` overrides the measured gripper with what was last
        *commanded*, in native units. It is what a mid-episode handoff needs: a
        gripper holding something reads short of the value that closed it --
        that is what holding looks like -- so seeding the operator's trigger
        from the measurement hands them a grip that is already giving way, and
        the object drops on the first tick they take over. The joints still
        come from the measurement, because that is genuinely where the arm is.
        """
        observation: dict[str, float] = {}
        missing = []
        for name in self.arm_names:
            state = states.get(name)
            if state is None:
                missing.append(name)
                continue
            positions = state.joints.position_rad
            if len(positions) != self.arm_dofs:
                raise ConfigurationError(
                    f"{name}: {self.robot_model} expects {self.arm_dofs} measured joints, "
                    f"got {len(positions)}"
                )
            for index in range(self.arm_dofs):
                observation[f"{name}_joint_{index + 1}.pos"] = float(positions[index])
            commanded = None if effector is None else effector.get(name)
            held: float | None = None
            if commanded is not None:
                held = float(commanded)
            elif state.effector is not None:
                held = float(state.effector.position)
            if held is not None:
                # Going the other way from poll(): the teleoperator speaks the
                # dataset convention, and this reading is native. Numerically
                # the same flip, but naming the direction is the only thing
                # keeping either call site readable.
                observation[f"{name}_gripper.pos"] = to_dataset_gripper(held)
        # The solver starts at its configured rest pose. Seeding from only one
        # side of a bimanual cell would silently leave the other side at rest,
        # so wait until every selected follower has published a real pose.
        if missing or not observation:
            return False
        try:
            self._teleop.seed_qpos_from_obs(observation)
        except ValueError as err:
            raise ConfigurationError(f"cannot initialize Quest teleop: {err}") from err
        self._seeded = True
        return True

    def poll(self, states: Mapping[str, ArmState | None]) -> TeleopStep:
        if not self._seeded and not self.seed_from(states):
            # No arm has published yet. Commanding nothing holds the arms where
            # they are, which is the only safe thing to do without knowing
            # where that is.
            return TeleopStep()

        event = self._read_event() if self._emit_episode_events else EpisodeEvent.NONE
        action = self._teleop.get_action()
        self._publish_effector_feedback(action, states)
        targets: dict[str, ArmTarget] = {}
        for name in self.arm_names:
            try:
                joints = tuple(
                    float(action[f"{name}_joint_{index + 1}.pos"]) for index in range(self.arm_dofs)
                )
            except KeyError as err:
                # A partial action is a protocol change, not a transient: better
                # to name the missing key than to command an arm from half a pose.
                raise ConfigurationError(
                    f"the Quest teleoperator returned no {err.args[0]} for arm "
                    f"{name!r}; it should emit {self.arm_dofs} joints per arm"
                ) from err
            gripper = action.get(f"{name}_gripper.pos")
            targets[name] = ArmTarget(
                position_rad=joints,
                effector=None if gripper is None else to_native_gripper(gripper),
            )
        return TeleopStep(targets=targets, event=event)

    def _publish_effector_feedback(
        self, action: Mapping[str, object], states: Mapping[str, ArmState | None]
    ) -> None:
        """Feed native gripper effort back to the vendored Quest teleoperator.

        The kit already mixes this signal into its controller haptics. Keeping
        the conversion here means both ``record`` and the direct ``teleop``
        command get force feedback without knowing anything about WebXR.
        """
        send_feedback = getattr(self._teleop, "send_feedback", None)
        if not callable(send_feedback):
            return
        torques: dict[str, float] = {}
        for name in self.arm_names:
            state = states.get(name)
            if state is None or state.effector is None:
                continue
            torques[f"{name}_gripper.torque"] = float(state.effector.effort_nm)
            key = f"{name}_gripper.pos"
            if key in action:
                torques[key] = float(action[key])
        if torques:
            send_feedback({"torques": torques})

    def right_b(self) -> bool:
        """Level of the right controller's B button.

        Named for the hardware rather than for a meaning, because it has three
        of them: vr-teleop-kit calls it pause/resume, recording calls it start,
        and DAgger calls it take-over. Reported as a level -- the kit's reader
        thread keeps it current whether or not anyone is calling
        :meth:`get_action`, which is what lets a consumer that is *not* driving
        the arms still see the press. Edge detection belongs to the caller.
        """
        return bool(self._teleop.is_pause_pressed())

    def left_y(self) -> bool:
        """Level of the left controller's Y button. See :meth:`right_b`."""
        return bool(self._teleop.is_reverse_pressed())

    def _read_event(self) -> EpisodeEvent:
        """Map the two controller buttons to an episode event, on their edges.

        Right B starts a take (and restarts an open one); left Y saves it. That
        is vr-teleop-kit's binding, kept identical so muscle memory carries over
        between the two recorders.
        """
        start = self.right_b()
        save = self.left_y()
        rising_start, rising_save = start and not self._last_start, save and not self._last_save
        self._last_start, self._last_save = start, save
        if rising_start:
            return EpisodeEvent.START
        if rising_save:
            return EpisodeEvent.SAVE
        return EpisodeEvent.NONE

    def close(self) -> None:
        try:
            self._teleop.disconnect()
        except (Exception, KeyboardInterrupt):  # teardown must reach native power-down
            pass


def _import_vr_kit(kit_path: Path | None):  # noqa: ANN202 - the module, Any by design
    """Import the vendored teleoperator, with a legacy checkout escape hatch.

    ``kit_path`` remains accepted for users migrating from the previous sibling
    checkout workflow. New sessions import the in-repository package directly.
    """
    if kit_path is not None:
        source = kit_path / "src"
        root = source if source.is_dir() else kit_path
        if not root.is_dir():
            raise ConfigurationError(f"no vr-teleop-kit checkout at {kit_path}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from vr_teleop_kit.lerobot import bi_quest_teleop
    except ImportError as err:
        # "vr_teleop_kit is missing" and "vr_teleop_kit is here but mujoco is
        # not" have completely different fixes, and conflating them sends the
        # operator hunting for a checkout they already have.
        if (err.name or "").startswith("vr_teleop_kit"):
            raise ConfigurationError(
                "VR teleoperation is not importable. Install the VR extra with "
                "`uv sync`, then retry."
            ) from err
        raise ConfigurationError(
            f"the VR teleoperator needs {err.name!r}, which is not installed. "
            "Install the VR extra with `uv sync`."
        ) from err
    return bi_quest_teleop
