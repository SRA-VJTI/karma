"""Native hardware loop for Quest teleoperation."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import ConfigurationError
from .inference import GripperWatch
from .safety import (
    StateGuard,
    joint_limits,
    report_gripper_start,
    validate_motion_parameters,
    warn_stalled_grippers,
)
from .teleop_vr import DEFAULT_WS_URL, QuestTeleopSource
from .types import PositionCommand

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

    from .backend import ArmBackend
    from .rigs import Rig, RigArm


def _sleep_until(stop: threading.Event, deadline: float, period: float) -> float:
    """Wait for one loop period and resynchronise after an overrun."""
    deadline += period
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        stop.wait(remaining)
        return deadline
    return time.perf_counter()


def run_teleop(
    rig: Rig,
    *,
    ws_url: str = DEFAULT_WS_URL,
    model_path: str | None = None,
    rate_hz: float = 200.0,
    park: bool = True,
    config_overrides: Mapping[str, object] | None = None,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
    stop: threading.Event | None = None,
    max_state_age_s: float = 0.25,
) -> int:
    """Drive the selected follower arms from the vendored Quest teleoperator.

    The function owns the entire native lifecycle. A direct command therefore
    cannot return while its native processes remain energized, and a stale arm
    state stops sending commands until the state stream recovers.
    """
    validate_motion_parameters(rate_hz=rate_hz, max_state_age_s=max_state_age_s)
    if rate_hz <= 0:
        raise ConfigurationError("--rate must be positive")
    if max_state_age_s <= 0:
        raise ConfigurationError("max_state_age_s must be positive")
    models = {arm.model for arm in rig.arms}
    if len(models) != 1 or not models.issubset({"Yam", "SO101"}):
        raise ConfigurationError("Quest teleoperation needs an all-YAM or all-SO101 rig")
    robot_model = next(iter(models))
    if robot_model == "SO101" and model_path:
        raise ConfigurationError("SO101 uses its packaged URDF; --yam-xml is YAM-only")
    if any(not arm.is_follower for arm in rig.arms):
        raise ConfigurationError("Quest teleoperation needs follower arms only")
    if any(arm.name not in {"left", "right"} for arm in rig.arms):
        raise ConfigurationError("Quest teleoperation requires left/right arm names")

    # Imported lazily to keep doctor/zero usable on installations without the
    # VR optional dependencies.
    from .cli import power_down, power_up, settle_arm_states

    stop = stop if stop is not None else threading.Event()
    limits = joint_limits(rig)
    guard = StateGuard(rig.names, max_age_s=max_state_age_s)
    gripper = GripperWatch()
    stalled_grippers: set[str] = set()
    session = None
    live_arms = []
    source: QuestTeleopSource | None = None
    try:
        session, live_arms = power_up(rig, float_mode=False, backend_factory=backend_factory)
        arms = {entry.name: entry.arm for entry in live_arms if entry.rig_arm.is_follower}
        if set(arms) != set(rig.names):
            raise ConfigurationError(
                "Quest teleoperation needs follower arms only; leaders cannot accept "
                "direct joint targets"
            )
        report_gripper_start(settle_arm_states(live_arms, max_age_s=max_state_age_s))
        source = QuestTeleopSource(
            tuple(arms),
            ws_url=ws_url,
            robot_model=robot_model,
            model_path=model_path,
            config_overrides=config_overrides,
        )
        print(f"  teleop   {source.describe()}")
        print(f"  control  {rate_hz:g} Hz — grip to clutch, trigger to control the gripper")
        print("  ctrl-c to " + ("park at home_pos and " if park else "") + "power down")

        period = 1.0 / rate_hz
        next_tick = time.perf_counter()
        stale_names: tuple[str, ...] | None = None
        while not stop.is_set():
            states = guard.check({name: arm.latest_state for name, arm in arms.items()})
            if states is None:
                stale: list[str] = []
                for name, arm in arms.items():
                    state = getattr(arm, "latest_state", None)
                    if state is None or not state.is_fresh(max_state_age_s):
                        stale.append(name)
                current_stale = tuple(stale)
                if current_stale != stale_names:
                    print(
                        "  state   waiting for fresh arm state: " + ", ".join(current_stale),
                        flush=True,
                    )
                    stale_names = current_stale
                next_tick = _sleep_until(stop, next_tick, period)
                continue

            if stale_names is not None:
                print("  state   arm state recovered", flush=True)
                stale_names = None

            step = source.poll(states)  # type: ignore[arg-type]
            commands = {}
            for name, target in step.targets.items():
                arm = arms.get(name)
                if arm is None:
                    raise ConfigurationError(f"Quest teleoperator commanded unknown arm {name!r}")
                positions = np.asarray(target.position_rad, dtype=np.float64)
                lower, upper = limits[name]
                if positions.shape != lower.shape or not np.all(np.isfinite(positions)):
                    raise ConfigurationError(f"invalid joint targets for {name}")
                commands[name] = PositionCommand(
                    position_rad=np.clip(positions, lower, upper),
                    effector=target.effector,
                )
            # Validate every arm before issuing any of this tick's commands.
            for name, command in commands.items():
                arms[name].command(command)
            gripper.observe(commands, states)
            warn_stalled_grippers(gripper, stalled_grippers)
            next_tick = _sleep_until(stop, next_tick, period)
    except KeyboardInterrupt:
        print()
    finally:
        try:
            if source is not None:
                source.close()
        finally:
            failures = power_down(session, live_arms, park=park) if session is not None else 0
    return 1 if failures else 0
