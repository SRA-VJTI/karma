"""A stand-in ArmBackend for tests that need a session but no hardware.

Used by the ``live`` tests in test_cli.py and test_viz.py. Faithful on the
points the power lifecycle depends on: ``connect`` is what energizes,
``close(move_to_ready=...)`` is what parks, and ``latest_state`` never blocks.
"""

from __future__ import annotations

import time

from openpi_control.types import (
    ArmCapabilities,
    ArmMode,
    ArmRole,
    ArmState,
    EffectorState,
    JointState,
    PositionCommand,
)


class FakeArmBackend:
    """ArmBackend stand-in that records the power lifecycle. No bus, no node.

    Faithful on the points ``live`` depends on: connect() is what energizes,
    close(move_to_ready=...) is what parks, and latest_state() never blocks.
    """

    def __init__(
        self,
        *,
        dof: int = 6,
        supports_move_to_ready: bool = True,
        supports_gravity_compensation: bool = True,
        connect_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.dof = dof
        self._supports_move_to_ready = supports_move_to_ready
        self._supports_gravity_compensation = supports_gravity_compensation
        self._connect_error = connect_error
        self._close_error = close_error
        self.connected = False
        self.connects = 0
        self.closes: list[bool] = []
        self.floats = 0
        self.holds = 0
        self.position = 0.0
        # Effector state is published only when the ArmConfig names one, the
        # same way the node does -- viser_control refuses to arm an arm that
        # claims a gripper and never says where it is.
        self.effector_position = 0.0
        self._has_effector = False
        self.commands: list[PositionCommand] = []
        # Knobs the browser-control tests use to make an arm go quiet or stale
        # without a bus: state_age_s backdates the published timestamp,
        # publishing=False stops state arriving at all.
        self.state_age_s = 0.0
        self.publishing = True

    # -- lifecycle ------------------------------------------------------- #

    def configure_pair(self, *, follower_state_topic: str) -> None:
        del follower_state_topic

    def connect(self, config, role, topics):
        del topics
        if self._connect_error is not None:
            raise self._connect_error
        self.connects += 1
        self.connected = True
        self.role = role
        self._has_effector = config.effector_model is not None
        return ArmCapabilities(
            protocol_version=(1, 0),
            model=config.model,
            joint_names=tuple(f"joint_{i + 1}" for i in range(self.dof)),
            has_effector=config.effector_model is not None,
            supports_direct_commands=True,
            supports_live_input=True,
            supports_gravity_compensation=self._supports_gravity_compensation,
            supports_force_feedback=False,
            supports_move_to_ready=self._supports_move_to_ready,
        )

    def _is_connected(self) -> bool:
        return self.connected

    def close(self, *, move_to_ready: bool = False) -> None:
        self.closes.append(move_to_ready)
        self.connected = False
        if self._close_error is not None:
            raise self._close_error

    # -- state and modes ------------------------------------------------- #

    def latest_state(self):
        if not self.connected or not self.publishing:
            return None
        joints = JointState(
            names=tuple(f"joint_{i + 1}" for i in range(self.dof)),
            position_rad=[self.position] * self.dof,
            velocity_rad_s=[0.0] * self.dof,
            effort_nm=[0.0] * self.dof,
            temperature_c=[25.0] * self.dof,
            current_a=[0.0] * self.dof,
        )
        return ArmState(
            name="fake",
            role=ArmRole.FOLLOWER,
            joints=joints,
            effector=EffectorState(position=self.effector_position) if self._has_effector else None,
            monotonic_timestamp=time.monotonic() - self.state_age_s,
            wall_timestamp=0.0,
            sequence=1,
            mode=ArmMode.HOLD,
        )

    def read_state(self, timeout_s=None):
        del timeout_s
        return self.latest_state()

    def pause_live_input(self, paused: bool) -> None:
        del paused

    def enter_gravity_float(self, drift_abort_rad=None) -> None:
        del drift_abort_rad
        self.floats += 1

    def hold(self) -> None:
        self.holds += 1

    def command(self, command: PositionCommand, *, live: bool = False) -> None:
        del live
        self.commands.append(command)

    def set_mode(self, mode) -> None:
        del mode
