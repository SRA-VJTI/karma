"""Tests for the supplied bimanual camera calibration."""

from __future__ import annotations

import numpy as np
import pytest

from openpi_control.camera_poses import (
    T_TOP_CAMERA_FROM_MIDPOINT,
    YAM_TOP_CAMERA_EXTRINSIC,
)
from openpi_control.rigs import resolve_rig


def test_top_camera_calibration_matches_supplied_pose_script() -> None:
    pose = YAM_TOP_CAMERA_EXTRINSIC
    np.testing.assert_allclose(pose.position, [0.013927329, -0.007471385, 0.889581466])
    np.testing.assert_allclose(
        pose.rpy_xyz_deg,
        [-161.954583897, 0.236687304, -0.176635794],
    )
    np.testing.assert_allclose(
        pose.optical_axis_table_intersection,
        [0.011145982, 0.282363360, 0.0],
        atol=1e-8,
    )
    assert pose.optical_axis_tilt_from_down_deg == pytest.approx(18.046916632, abs=1e-8)


def test_top_camera_transform_inverts_to_midpoint_from_camera() -> None:
    pose = YAM_TOP_CAMERA_EXTRINSIC
    np.testing.assert_allclose(pose.matrix @ pose.inverse_matrix, np.eye(4), atol=1e-8)
    np.testing.assert_allclose(T_TOP_CAMERA_FROM_MIDPOINT, pose.inverse_matrix)


def test_packaged_rig_carries_only_the_supplied_top_calibration() -> None:
    rig = resolve_rig("yam_bimanual")
    assert rig.cameras[0].extrinsic is YAM_TOP_CAMERA_EXTRINSIC
    assert rig.cameras[1].extrinsic is None
    assert rig.cameras[2].extrinsic is None
