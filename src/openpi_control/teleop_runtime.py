"""Native hardware loop for Quest teleoperation."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import ConfigurationError
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


def _fresh_states(arms: Mapping[str, object], max_age_s: float) -> dict[str, object] | None:
    states: dict[str, object] = {}
    for name, arm in arms.items():
        state = getattr(arm, "latest_state", None)
        if state is None or not state.is_fresh(max_age_s):
            return None
        states[name] = state
    return states


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
    if rate_hz <= 0:
        raise ConfigurationError("--rate must be positive")
    if max_state_age_s <= 0:
        raise ConfigurationError("max_state_age_s must be positive")
    if any(arm.model != "Yam" for arm in rig.arms):
        raise ConfigurationError(
            "Quest teleoperation currently supports the packaged Yam model only; "
            "the vendored IK is YAM-specific"
        )
    if any(arm.name not in {"left", "right"} for arm in rig.arms):
        raise ConfigurationError("Quest teleoperation requires left/right arm names")

    # Imported lazily to keep doctor/zero usable on installations without the
    # VR optional dependencies.
    from .cli import power_down, power_up

    stop = stop if stop is not None else threading.Event()
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
        source = QuestTeleopSource(
            tuple(arms),
            ws_url=ws_url,
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
            states = _fresh_states(arms, max_state_age_s)
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
            for name, target in step.targets.items():
                arm = arms.get(name)
                if arm is None:
                    raise ConfigurationError(f"Quest teleoperator commanded unknown arm {name!r}")
                arm.command(
                    PositionCommand(
                        position_rad=np.asarray(target.position_rad, dtype=np.float64),
                        effector=target.effector,
                    )
                )
            next_tick = _sleep_until(stop, next_tick, period)
    except KeyboardInterrupt:
        print()
    finally:
        if source is not None:
            source.close()
        if session is not None:
            failures = power_down(session, live_arms, park=park)
        else:
            failures = 0
    return 1 if failures else 0
