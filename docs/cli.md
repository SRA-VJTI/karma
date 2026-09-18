# Karma commands

The executable is `karma` (`uv run karma`). Every subcommand has `--help` with
its complete flag set; this page is the map.

## Main workflows

| Command | Purpose |
| --- | --- |
| `teleop` | Drive YAM or SO101 followers from a Meta Quest. `--record` writes demonstrations as a LeRobot v3 dataset. |
| `inference` (`infer`) | Run a MolmoAct2 checkpoint continuously with one `--instruction`, executing action chunks with bounded steps. |
| `rollout` | Timed policy episodes; the task is asked at the start of each episode; every attempt is recorded. |
| `hitl` | DAgger: the policy drives until Right B hands control to the Quest operator; Left Y tap returns to the policy, hold ends the attempt; episodes are reviewed, labelled and saved with intervention labels. |

## Maintenance and diagnostics

| Command | Purpose |
| --- | --- |
| `doctor` | Read-only preflight for an arm or a rig: model assets, bus presence, servo registry, cameras. Opens no bus unless `--probe`. |
| `cameras` | Resolve a rig's cameras to devices and report presence. `--probe` opens each one and measures the delivered rate; `--snapshot DIR` writes a frame per view. |
| `calibrate-so101` | SO101 calibration wizard: centre encoders (`--center-encoders`), record travel, gripper endpoints and home. Writes `calibration/NAME.json`. |
| `so101-policy-frame` | Capture LeRobot's zero and rotated poses and write a `--policy-frame` file mapping Karma's radians onto a checkpoint trained on LeRobot degrees. Work in progress; see [inference](inference.md#policy-frames). |
| `zero` | Write the current pose as each servo's firmware zero. YAM only, at the mechanical reference pose; `--dry-run` previews. Never use on SO101 (use the wizard). |
| `live` | Energize a rig, mirror it in the browser (Viser), then park and power down. `--float` for gravity float, `--control` for the browser panel. |
| `relay` | Run the Quest WebXR relay alone, without arms. |
| `record`, `collect` | Lower-level teleop recording entry points; `teleop --record` is the supported path. |

## Shared flags

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--rig NAME` | all | `yam_bimanual`, `so101`, `so101_bimanual` |
| `--interface ARM=BUS` | all | Move an arm to another bus/port, e.g. `right=/dev/ttyACM0`, `left=can2` |
| `--calibration ARM=FILE` | SO101 | Per-arm calibration profile from `calibrate-so101` |
| `--camera-serial ROLE=SERIAL` | camera workflows | Declare a RealSense for a role (`top`, `left_wrist`, `right_wrist`, `side`) |
| `--camera ROLE=/dev/videoN` | camera workflows | Pin a RealSense to a device, or — for a role with no RealSense serial — declare a USB webcam captured through OpenCV |
| `--server URL` | policy | MolmoAct2 server; `/act` is appended |
| `--norm-tag KEY` | policy | Checkpoint normalization key the server must report (required for SO101) |
| `--policy-frame FILE` | policy | State/action frame conversion for checkpoints not trained in Karma's radians |
| `--speed F` | policy | Play each chunk at this fraction of the training rate |
| `--max-step-rad R` | policy | Bound on one joint's change per control tick |
| `--skip-preflight` | hardware | Energize without `doctor` checks (runtime checks stay on) |
| `--open-quest` | teleop/hitl | Open the WebXR page in the Quest browser over ADB |

Policy commands check the server's health contract (`norm_tag`, `state_dim`,
`num_cameras`) before any motor is energized, and required cameras must deliver
a frame first. See [safety](safety.md).
