"""Calibrated camera extrinsics used by the bimanual scene visualizer.

The supplied calibration script defines a transform from the top camera frame
into the midpoint frame between the two arm bases. Keeping the matrix here
makes that convention explicit and lets Viser show the calibrated camera frame
without changing the image bytes sent to the policy server.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MatrixTuple = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _matrix_tuple(matrix: np.ndarray) -> _MatrixTuple:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4):
        raise ValueError(f"camera extrinsics must have shape (4, 4), got {values.shape}")
    rows = tuple(tuple(float(value) for value in row) for row in values)
    return rows  # type: ignore[return-value]


def _rotation_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to Viser's ``wxyz`` quaternion order."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")

    # This branch-stable conversion avoids pulling scipy into the core package.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * float(np.sqrt(trace + 1.0))
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ]
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
        quaternion = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
        quaternion = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return tuple(float(value) for value in quaternion)


@dataclass(frozen=True, slots=True)
class CameraExtrinsic:
    """A camera-to-scene rigid transform and its calibrated parent frame."""

    name: str
    parent_frame: str
    transform_from_camera: _MatrixTuple

    @classmethod
    def from_matrix(
        cls, name: str, parent_frame: str, transform: np.ndarray
    ) -> CameraExtrinsic:
        values = np.asarray(transform, dtype=np.float64)
        if values.shape != (4, 4) or not np.all(np.isfinite(values)):
            raise ValueError("camera extrinsic must be a finite 4x4 matrix")
        if not np.allclose(values[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("camera extrinsic must have a homogeneous bottom row")
        rotation = values[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("camera extrinsic rotation must be orthonormal")
        return cls(name, parent_frame, _matrix_tuple(values))

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.transform_from_camera, dtype=np.float64)

    @property
    def inverse_matrix(self) -> np.ndarray:
        transform = self.matrix
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = transform[:3, :3].T
        inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
        return inverse

    @property
    def position(self) -> np.ndarray:
        return self.matrix[:3, 3].copy()

    @property
    def rotation_wxyz(self) -> tuple[float, float, float, float]:
        return _rotation_to_wxyz(self.matrix[:3, :3])

    @property
    def rpy_xyz_deg(self) -> np.ndarray:
        rotation = self.matrix[:3, :3]
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(
            -rotation[2, 0],
            np.hypot(rotation[2, 1], rotation[2, 2]),
        )
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.degrees([roll, pitch, yaw])

    @property
    def optical_axis_table_intersection(self) -> np.ndarray:
        """Intersect the camera's +Z axis with the parent-frame z=0 plane."""
        transform = self.matrix
        position = transform[:3, 3]
        camera_z = transform[:3, 2]
        distance = -position[2] / camera_z[2]
        return position + distance * camera_z

    @property
    def optical_axis_tilt_from_down_deg(self) -> float:
        camera_z = self.matrix[:3, 2]
        return float(
            np.degrees(
                np.arccos(np.clip(np.dot(camera_z, [0.0, 0.0, -1.0]), -1.0, 1.0))
            )
        )


# Generated from the supplied camera-pose-bimanual.py calibration. It maps
# points in the camera frame into the bimanual midpoint frame, in metres.
T_MIDPOINT_FROM_TOP_CAMERA = np.array(
    [
        [0.999986716, -0.004210873, -0.002972762, 0.013927329],
        [-0.003082845, -0.950802809, 0.309781399, -0.007471385],
        [-0.004130961, -0.309768119, -0.950803159, 0.889581466],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

YAM_TOP_CAMERA_EXTRINSIC = CameraExtrinsic.from_matrix(
    "top", "midpoint", T_MIDPOINT_FROM_TOP_CAMERA
)

# The inverse printed by the calibration script: it maps midpoint-frame
# points into the top-camera frame.
T_TOP_CAMERA_FROM_MIDPOINT = YAM_TOP_CAMERA_EXTRINSIC.inverse_matrix
