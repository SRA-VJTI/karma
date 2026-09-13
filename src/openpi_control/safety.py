"""Shared Python-side runtime checks; native fault protection remains active."""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping

import numpy as np

from .config import resolve_model_assets
from .exceptions import ConfigurationError
from .inference import GripperWatch, InferenceError
from .rigs import Rig
from .types import ArmState

MAX_STATE_AGE_S = 0.25
MAX_STALE_S = 1.0


class StateGuard:
    """Pause on brief state gaps and fail after one second of silence."""

    def __init__(self, names, *, max_age_s=MAX_STATE_AGE_S, max_stale_s=MAX_STALE_S):
        if not np.isfinite(max_age_s) or not 0 < max_age_s < max_stale_s:
            raise ConfigurationError("state freshness must be positive and below the stale timeout")
        if not np.isfinite(max_stale_s):
            raise ConfigurationError("state stale timeout must be finite")
        self.names = tuple(names)
        self.max_age_s = max_age_s
        self.max_stale_s = max_stale_s
        self.missing_since: dict[str, float] = {}

    def check(self, states: Mapping[str, ArmState | None]) -> dict[str, ArmState] | None:
        fresh = {}
        now = time.monotonic()
        for name in self.names:
            state = states.get(name)
            if state is not None and state.is_fresh(self.max_age_s):
                self.missing_since.pop(name, None)
                if not np.all(np.isfinite(state.joints.position_rad)):
                    raise InferenceError(f"{name} state contains non-finite joint positions")
                fresh[name] = state
                continue
            since = self.missing_since.setdefault(name, now)
            age = max(now - since, state.age_s if state is not None else 0.0)
            if age >= self.max_stale_s:
                detail = "missing" if state is None else f"{state.age_s * 1e3:.0f} ms old"
                raise InferenceError(
                    f"{name} state is {detail} (silent past the {self.max_stale_s:g}s limit)"
                )
        return fresh if len(fresh) == len(self.names) else None


def joint_limits(rig: Rig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read chain-ordered URDF limits without loading visualization or meshes."""
    result = {}
    for arm in rig.arms:
        path = resolve_model_assets(arm.model).urdf
        if path is None:
            raise ConfigurationError(f"{arm.model} has no URDF joint limits")
        root = ET.parse(path).getroot()
        joints = root.findall("joint")
        children = {j.find("child").attrib["link"] for j in joints}
        roots = [
            link.attrib["name"]
            for link in root.findall("link")
            if link.attrib["name"] not in children
        ]
        if len(roots) != 1:
            raise ConfigurationError(f"{path} must have one root link")
        bounds = []

        def visit(link, joints=joints, bounds=bounds):
            for joint in joints:
                if joint.find("parent").attrib["link"] != link:
                    continue
                if joint.attrib["type"] != "fixed":
                    limit = joint.find("limit")
                    if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                        raise ConfigurationError(f"missing bounds for {joint.attrib['name']}")
                    bounds.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
                visit(joint.find("child").attrib["link"])

        visit(roots[0])
        values = np.asarray(bounds)
        if (
            not len(bounds)
            or not np.all(np.isfinite(values))
            or np.any(values[:, 0] >= values[:, 1])
        ):
            raise ConfigurationError(f"invalid joint limits in {path}")
        result[arm.name] = (values[:, 0], values[:, 1])
    return result


def report_gripper_start(states: Mapping[str, ArmState]) -> None:
    readings = ", ".join(
        f"{name} {state.effector.position:.3f}"
        for name, state in sorted(states.items())
        if state.effector is not None
    )
    if readings:
        print(f"  gripper  measured now: {readings} (1.0 = open) — check the jaws agree")


def warn_stalled_grippers(watch: GripperWatch, reported: set[str]) -> None:
    for name in watch.stalled():
        if name in reported:
            continue
        reported.add(name)
        print(
            f"  gripper  {name} is NOT TRACKING: commanded across "
            f"{watch.command_travel:.2f} of its range and the measured position has not moved. "
            "Check the startup calibration and whether the jaws were blocked.",
            file=sys.stderr,
            flush=True,
        )


def validate_motion_parameters(**values: float) -> None:
    """Reject unusable motion settings before opening hardware."""
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ConfigurationError(f"{name} must be finite and positive")
