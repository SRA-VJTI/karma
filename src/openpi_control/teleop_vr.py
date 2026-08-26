"""Driving this cell's arms from a Meta Quest, via ``vr-teleop-kit``.

``vr-teleop-kit`` already solves the hard half of VR teleoperation -- a WebXR
relay the headset talks to, clutch-relative pose mapping, and a damped IK solver
tuned for the YAM's wrist. None of that is reimplemented here. This module is
the adapter: it turns that project's ``BiQuestTeleoperator`` into a
:class:`~openpi_control.record.TeleopSource`, so the headset drives arms through
``openpi_control``'s own native stack rather than through the i2rt driver.

    vr-teleop-kit relay  <--WebXR--  Quest
             |
        BiQuestTeleoperator  (pose mapping + IK)
             |  joint targets
        QuestTeleopSource    (this module)
             |  PositionCommand
        FollowerArm -> pi_control_node -> CAN -> YAM

Setup, once
-----------

``vr-teleop-kit`` is not a dependency of this package -- it is a sibling
checkout, because it carries its own relay, its own web assets, and an i2rt
clone for the YAM MJCF the IK needs. Point ``--vr-kit`` at it (or install it
into this environment), then run its relay next to this command::

    vr-teleop-relay                 # in the vr-teleop-kit checkout
    openpi-control record --vr ...  # here

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
        kit_path: Path | None = None,
        model_path: str | None = None,
        connect_timeout_s: float = 5.0,
        teleoperator: object | None = None,
    ) -> None:
        """``teleoperator`` injects an already-built (or stand-in) teleoperator.

        That seam exists because the adapter carries the gripper inversion and
        the button edge detection -- the two things here most worth testing, and
        the two least testable against a real headset.
        """
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
            config_kwargs: dict[str, object] = {
                "ws_url": ws_url,
                "connect_timeout_s": connect_timeout_s,
            }
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
                    "`vr-teleop-relay` in the vr-teleop-kit checkout, or point "
                    "--vr-url at wherever it is listening."
                ) from err
        self._seeded = False
        # Both buttons are reported as levels, so the bridge does its own edge
        # detection: holding a button must not restart an episode every tick.
        self._last_start = False
        self._last_save = False

    def describe(self) -> str:
        return f"Quest teleoperator via {self._ws_url} (arms: {', '.join(self.arm_names)})"

    def seed_from(self, states: Mapping[str, ArmState | None]) -> bool:
        """Anchor the IK at the arms' measured pose.

        Returns False when no arm has reported yet, so the caller can wait
        rather than seed the solver with zeros -- which would make the first
        command a move to the folded park pose.
        """
        observation: dict[str, float] = {}
        for name in self.arm_names:
            state = states.get(name)
            if state is None:
                continue
            positions = state.joints.position_rad
            for index in range(min(VR_ARM_DOFS, len(positions))):
                observation[f"{name}_joint_{index + 1}.pos"] = float(positions[index])
            if state.effector is not None:
                # Going the other way from poll(): the teleoperator speaks the
                # dataset convention, and this reading is native. Numerically
                # the same flip, but naming the direction is the only thing
                # keeping either call site readable.
                observation[f"{name}_gripper.pos"] = to_dataset_gripper(
                    state.effector.position
                )
        if not observation:
            return False
        self._teleop.seed_qpos_from_obs(observation)
        self._seeded = True
        return True

    def poll(self, states: Mapping[str, ArmState | None]) -> TeleopStep:
        if not self._seeded and not self.seed_from(states):
            # No arm has published yet. Commanding nothing holds the arms where
            # they are, which is the only safe thing to do without knowing
            # where that is.
            return TeleopStep()

        event = self._read_event()
        action = self._teleop.get_action()
        targets: dict[str, ArmTarget] = {}
        for name in self.arm_names:
            try:
                joints = tuple(
                    float(action[f"{name}_joint_{index + 1}.pos"])
                    for index in range(VR_ARM_DOFS)
                )
            except KeyError as err:
                # A partial action is a protocol change, not a transient: better
                # to name the missing key than to command an arm from half a pose.
                raise ConfigurationError(
                    f"the Quest teleoperator returned no {err.args[0]} for arm "
                    f"{name!r}; it should emit {VR_ARM_DOFS} joints per arm"
                ) from err
            gripper = action.get(f"{name}_gripper.pos")
            targets[name] = ArmTarget(
                position_rad=joints,
                effector=None if gripper is None else to_native_gripper(gripper),
            )
        return TeleopStep(targets=targets, event=event)

    def _read_event(self) -> EpisodeEvent:
        """Map the two controller buttons to an episode event, on their edges.

        Right B starts a take (and restarts an open one); left Y saves it. That
        is vr-teleop-kit's binding, kept identical so muscle memory carries over
        between the two recorders.
        """
        start = bool(self._teleop.is_pause_pressed())
        save = bool(self._teleop.is_reverse_pressed())
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
        except Exception:  # noqa: BLE001 - teardown must not mask a session result
            pass


def _import_vr_kit(kit_path: Path | None):  # noqa: ANN202 - the module, Any by design
    """Import ``vr_teleop_kit.lerobot.bi_quest_teleop``, from a checkout if given.

    ``vr-teleop-kit`` is a sibling project rather than a dependency, so a path
    to its checkout is prepended to ``sys.path`` when one is given. The import
    error is worth catching because the fix ("point --vr-kit at the checkout")
    is not obvious from a bare ModuleNotFoundError.
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
                "VR teleoperation needs vr-teleop-kit, which is not importable: "
                "pass --vr-kit /path/to/vr-teleop-kit, or install it into this "
                f"environment. ({err})"
            ) from err
        raise ConfigurationError(
            f"vr-teleop-kit was found, but it needs {err.name!r}, which is not "
            "installed here. Install the VR extra (uv sync --extra vr), or run "
            f"from an environment that has vr-teleop-kit's dependencies. ({err})"
        ) from err
    return bi_quest_teleop
