# Acknowledgements

Thank you to everyone who contributed to the foundations of openpi-control.
Many people developed examples of underlying drivers, built reference
implementations, and created a rich codebase for this ecosystem to carry
forward.

Chino Chen, Chris Chen, Dillion Jiang, David Lan, Larry Tian, Kyle Vedder,
Kai Wang, Nara Won, Maki Xu, Josh Zhang, Shenzhi Zhu

Any errors or omissions are ours alone — corrections welcome by pull request.

## Upstream projects

Karma builds on two repositories:

- [openpi-basic-control](https://github.com/Physical-Intelligence/openpi-basic-control)
  by Physical Intelligence — the robot control stack this repository grew out
  of: the `openpi_control` package layout, the native CAN/FeeTech control node,
  arm and effector model configurations, the servo drivers, safety checks, and
  the MolmoAct2 inference client and LeRobot recording paths.
- [vr-teleop-kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit)
  by Dream Machines — the Meta Quest teleoperation runtime vendored under
  `src/vr_teleop_kit/` (details below).

Thank you to both teams.

## Robot models

- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) by The Robot Studio —
  the SO101 arm description. The packaged `SO101.urdf` is their
  `Simulation/SO101/so101_new_calib.urdf` (generated with onshape-to-robot),
  with the gripper frame renamed to `end_link` for the FK target; the SO101
  IK, calibration limits and browser view all derive from it.
- [i2rt](https://github.com/i2rt-robotics/i2rt) by I2RT Robotics — the YAM arm
  description: the packaged YAM URDF is their `yam.urdf` with the wrist link
  renamed, the Quest IK loads their `yam.xml` and `linear_4310` gripper MJCF
  from a local clone, and `karma-viz --fetch-meshes` downloads their meshes
  (MIT) with provenance recorded next to them.

## Vendored Quest teleoperation runtime

The WebXR relay, Quest web client, clutch-relative pose mapping, YAM inverse
kinematics, and LeRobot adapters under `src/vr_teleop_kit/` are vendored from
[vr-teleop-kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit),
Copyright 2026 Dream Machines, under the Apache License 2.0. The original
license text is preserved at `licenses/vr-teleop-kit-LICENSE`.
