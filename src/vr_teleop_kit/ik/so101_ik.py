"""Bounded differential IK for the packaged SO101 five-joint chain.

Read the same URDF as the native driver, including its calibrated joint frames
and end_link TCP. A weighted pose solve favours translation over orientation:
a five-axis arm cannot satisfy every six-dimensional controller pose. No meshes
or external robot checkout are needed for kinematics.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..core.pose_mapping import mat_to_quat, quat_conj, quat_mul, quat_to_rotvec

logger = logging.getLogger(__name__)

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


class SO101IKSolver:
    """One damped, joint-limited step per tick, seeded in native radians."""

    def __init__(self, *, max_dq_per_joint=None):
        from openpi_control import models

        path = Path(next(iter(models.__path__))) / "arms/SO101/SO101.urdf"
        root = ET.parse(path).getroot()
        config = json.loads(path.with_suffix(".json").read_text())
        self.seed_margin = np.array(
            [
                joint["pos_error_margin"]
                for joint in sorted(config["joints"], key=lambda item: item["joint_id"])
            ],
            dtype=float,
        )
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        self._chain = []
        limits = []
        for name in (*JOINT_NAMES, "end_link_joint"):
            joint = joints[name]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (
                _rotation([0, 0, 1], rpy[2])
                @ _rotation([0, 1, 0], rpy[1])
                @ _rotation([1, 0, 0], rpy[0])
            )
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            self._chain.append((transform, axis))
            if name in JOINT_NAMES:
                limit = joint.find("limit")
                limits.append((float(limit.get("lower")), float(limit.get("upper"))))
        self.lower, self.upper = np.array(limits).T
        self.max_dq_per_joint = np.asarray(
            [0.02] * 5 if max_dq_per_joint is None else max_dq_per_joint, dtype=float
        )
        if self.max_dq_per_joint.shape != (5,) or not np.all(
            np.isfinite(self.max_dq_per_joint) & (self.max_dq_per_joint > 0)
        ):
            raise ValueError("SO101 max_dq_per_joint must contain five finite positive values")
        self.last_limit_pressure = 0.0
        self.last_pos_err_norm = 0.0

    def _kinematics(self, qpos):
        q = np.asarray(qpos, dtype=float)[:5]
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            raise ValueError("SO101 joint seed must contain five finite angles")
        transform = np.eye(4)
        origins, axes = [], []
        for i, (offset, axis) in enumerate(self._chain):
            transform = transform @ offset
            if i < 5:
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ axis)
                rotation = np.eye(4)
                rotation[:3, :3] = _rotation(axis, q[i])
                transform = transform @ rotation
        pos = transform[:3, 3]
        jac = np.vstack((np.cross(axes, pos - np.array(origins)).T, np.array(axes).T))
        return pos.copy(), mat_to_quat(transform[:3, :3]), jac

    def fk(self, qpos):
        pos, quat, _ = self._kinematics(qpos)
        return pos, quat

    def j4_anchor_xpos(self):
        # SO101 targets end_link directly; it has no YAM wrist pivot.
        return None

    def validate_seed(self, qpos):
        """Match native tracking tolerance, seeding the bounded commanded pose.

        Firmware feedback may sit beyond a command limit under load or near a
        stop. The native driver already clips goals while allowing the model's
        pos_error_margin on measurements. Apply that same convention here;
        do not reinterpret firmware zeros or expand commanded joint limits.
        """
        q = np.asarray(qpos, dtype=float)[:5]
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            raise ValueError("SO101 joint seed must contain five finite angles")
        bad = np.flatnonzero(
            (q < self.lower - self.seed_margin - 1e-6) | (q > self.upper + self.seed_margin + 1e-6)
        )
        if bad.size:
            details = "; ".join(
                f"{JOINT_NAMES[i]}={np.degrees(q[i]):.2f} deg "
                f"(allowed {np.degrees(self.lower[i]):.2f}.."
                f"{np.degrees(self.upper[i]):.2f} deg, "
                f"tracking tolerance {np.degrees(self.seed_margin[i]):.2f} deg)"
                for i in bad
            )
            raise ValueError(
                "SO101 starting pose exceeds native tracking tolerance: "
                + details
                + ". With torque off, move the arm away from its folded/stopped pose "
                "into the allowed range and retry. If the readings disagree with the "
                "physical pose, check calibration; do not zero at an arbitrary pose."
            )
        bounded = np.clip(q, self.lower, self.upper)
        if np.any(np.abs(q - bounded) > 1e-6):
            logger.info(
                "SO101 seed within native tracking tolerance; using bounded joint target "
                "(max difference %.2f deg). Controller motion remains clutch-relative.",
                np.degrees(np.max(np.abs(q - bounded))),
            )
        return bounded

    def solve(self, target_pos, target_quat, qpos):
        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = np.asarray(target_quat, dtype=float)
        if (
            target_pos.shape != (3,)
            or target_quat.shape != (4,)
            or not np.all(np.isfinite(target_pos))
            or not np.all(np.isfinite(target_quat))
            or np.linalg.norm(target_quat) < 1e-9
        ):
            raise ValueError("SO101 target must be a finite position and nonzero quaternion")
        q = self.validate_seed(qpos)
        pos, quat, jac = self._kinematics(q)
        rotation_error = quat_to_rotvec(
            quat_mul(target_quat / np.linalg.norm(target_quat), quat_conj(quat))
        )
        error = np.r_[target_pos - pos, rotation_error]
        self.last_pos_err_norm = float(np.linalg.norm(error[:3]))
        # Metres versus radians: 0.04 m/rad makes TCP position the dominant task.
        weights = np.array([1, 1, 1, 0.04, 0.04, 0.04])
        weighted_jac = weights[:, None] * jac
        dq = np.linalg.solve(
            weighted_jac.T @ weighted_jac + 0.015**2 * np.eye(5),
            weighted_jac.T @ (weights * error),
        )
        # Scale the whole vector to preserve the Cartesian step direction.
        dq /= max(1.0, float(np.max(np.abs(dq) / self.max_dq_per_joint)))
        result = np.clip(q + dq, self.lower, self.upper)
        self.last_limit_pressure = float(np.max(np.abs(q + dq - result)))
        return result
