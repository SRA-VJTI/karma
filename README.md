# Karma

A **bimanual YAM-first** Linux robot workstation stack, with SO101 support: Meta Quest VR teleoperation,
MolmoAct2 policy inference, autonomous rollout recording, and human-in-the-loop
DAgger collection. The Quest relay, browser application, IK, native motor runtime,
and LeRobot v3 writer integration live in this repository.

The primary setup is two YAM arms, an overhead RealSense D435, and one D405
on each wrist. All four workflows use `uv run karma`: `teleop`, `inference`,
`rollout`, and `hitl`. YAM has six arm joints plus a gripper per side, giving
fourteen state/action values for the bimanual policy.

SO101 is the second supported setup, with five arm joints plus a gripper per
side. Its setup and examples follow the YAM walkthrough below.

## Install once

Use Ubuntu Linux, USB 3 for RealSense, and Python 3.12 or 3.13. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then from this checkout:

```bash
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
UV_HTTP_TIMEOUT=180 uv sync --locked
uv run karma --help
```

The first sync downloads the complete workstation runtime, including LeRobot,
PyTorch, camera SDK and VR packages, and builds the native executable. Later
commands reuse it. No sibling VR repository or `PYTHONPATH` setup is required.
Dependencies still require an initial download; model weights and a GPU policy
server are separate from the robot workstation. Model visualization dependencies are included in the workstation install.

For USB access, add your user to `dialout` and `video`, then log out and back in:

```bash
sudo usermod -aG dialout,video "$USER"
sudo apt-get install adb
sudo install -m 0644 scripts/udev/51-meta-quest.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

Enable Quest developer mode and USB debugging, reconnect it, and accept the
headset's authorization prompt. `adb devices` must report `device`.

## 1. Bimanual YAM setup

### Connect and check the arms

The default rig is `yam_bimanual`: **left on can0, right on can1**, each with an
E_Yam gripper. Check adapter names, power the arms, and bring up the CAN buses:

```bash
ip -brief link show type can
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
uv run karma doctor --rig yam_bimanual
```

If your adapters have different names, use `--interface left=YOUR_LEFT_BUS`
and `--interface right=YOUR_RIGHT_BUS` in the workflow commands below.

### Verify calibration and home

YAM uses firmware motor zeros and native instance configurations. It does not
use SO101 calibration JSON. Keep existing correct firmware zeros. If a motor
needs zeroing, support the arm at the manufacturer's mechanical reference pose
and inspect the plan before writing:

```bash
uv run karma zero --model Yam --interface can0 --effector E_Yam --dry-run
# Only with the left arm at the correct mechanical reference:
uv run karma zero --model Yam --interface can0 --effector E_Yam
```

Repeat for `can1` only if the right arm needs zeroing. Do not zero at an arbitrary
home pose. Per-joint `home_pos` in
[`Yam_01.json`](src/openpi_control/models/arms/Yam/Yam_01.json) defines native
parking; the gripper uses
[`E_Yam_01.json`](src/openpi_control/models/effectors/E_Yam/E_Yam_01.json).
Verify these targets before testing automatic home. See [YAM calibration](docs/yam-setup.md).

### Configure the three RealSense views

| Role | Camera | Placement |
| --- | --- | --- |
| `top` | RealSense D435 | Fixed overhead, covering both arms' shared workspace |
| `left_wrist` | RealSense D405 | Left wrist, looking at the grasp area |
| `right_wrist` | RealSense D405 | Right wrist, looking at the grasp area |

Use USB 3 and keep mounts/framing consistent with training. List SDK serials:

```bash
uv run python -c 'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
```

In the commands below, replace `TOP_SERIAL`, `LEFT_WRIST_SERIAL`, and
`RIGHT_WRIST_SERIAL` with your devices. Probe capture before recording or inference:

```bash
uv run karma cameras --rig yam_bimanual \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL --probe
```

See [camera configuration](docs/cameras.md) for capture defaults and troubleshooting.

### YAM teleop and demonstration collection

```bash
uv run karma teleop --rig yam_bimanual \
  --interface left=can0 --interface right=can1 --open-quest
```

Enter immersive VR, hold grip to clutch, and use the trigger for the gripper.
Ctrl+C parks then disables torque; use `--no-park` for an initial check if home
has not yet been verified. Record demonstrations through the same main command:

```bash
uv run karma teleop --record --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --repo-id local/yam-demo --task "fold the towel" --fps 30 --open-quest
```

### YAM MolmoAct2 inference

Run the matching YAM model server on your GPU host first. The client expects
14-dimensional actions and the `yam_dual_molmoact2` normalization contract.
Replace `GPU_HOST` with the server address; this command connects to the server,
it does not launch or download the GPU model.

```bash
uv run karma inference --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_HOST:8202 --norm-tag yam_dual_molmoact2 \
  --instruction "fold the towel"
```

### YAM rollout recording

```bash
uv run karma rollout --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_HOST:8202 --norm-tag yam_dual_molmoact2 \
  --repo-id local/yam-rollouts --episodes 10 --episode-seconds 30 --fps 30
```

### YAM HITL / DAgger

```bash
uv run karma hitl --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_HOST:8202 --norm-tag yam_dual_molmoact2 \
  --repo-id local/yam-dagger --episodes 10 --episode-seconds 30 --fps 30 --open-quest
```

Rollout/HITL ask for the task per episode. Right B takes control; tap Left Y to
hand back to the policy, or hold it to end the attempt. Follow the terminal
review prompts to accept, label or discard. Demonstrations, rollouts and HITL
all use [LeRobot v3 recording](docs/recording.md).

## 2. SO100 family / SO101 setup

The supplied model and calibration are for **SO101**. Original SO100 hardware
may differ in gearing/geometry and has not been validated by these SO101 checks.
SO101 has five arm joints plus a gripper: six policy values for one arm, twelve
for two. Start with [SO101 calibration](docs/so101-teleop.md).

Each SO101 arm needs its own profile. Example left calibration:

```bash
uv run karma calibrate-so101 --interface /dev/ttyACM1 \
  --output calibration/so101-left.json --center-encoders
```

Do the same for right on `/dev/ttyACM0`, saving `so101-right.json`. Follow the
physical midpoint, travel, closed/open gripper, and home prompts. Back up the
`calibration/` directory yourself; it is intentionally ignored by Git.

### SO101 workflows

Every SO101 command below uses the **same** physical arm/profile mapping.
Only the top camera is declared by default; wrist views are optional.
Replace `TOP_SERIAL` with your camera's serial. For policy workflows, replace
`CHECKPOINT_NORM_TAG` with your SO101 checkpoint's actual normalization key.

#### SO101 teleop

```bash
uv run karma teleop --rig so101_bimanual \
  --interface left=/dev/ttyACM1 --interface right=/dev/ttyACM0 \
  --calibration left=calibration/so101-left.json \
  --calibration right=calibration/so101-right.json \
  --rate 50 --open-quest
```

Enter immersive VR; grip clutches motion, trigger controls the gripper. Ctrl+C
parks at the saved home before disabling torque. To collect demonstrations, use
`karma teleop --record` with the same rig/interface/calibration flags and
`--camera-serial top=TOP_SERIAL --repo-id local/demo --task "pick up the object"
--fps 30 --open-quest`. Recording has its own options:
`uv run karma teleop --record --help`.

#### SO101 inference

```bash
uv run karma inference --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --camera-serial top=TOP_SERIAL --norm-tag CHECKPOINT_NORM_TAG \
  --server http://GPU_HOST:8202 --instruction "pick up the object"
```

#### SO101 rollout

```bash
uv run karma rollout --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --camera-serial top=TOP_SERIAL --norm-tag CHECKPOINT_NORM_TAG \
  --server http://GPU_HOST:8202 --repo-id local/so101-rollouts \
  --episodes 10 --episode-seconds 30 --fps 30
```

#### SO101 HITL / DAgger

```bash
uv run karma hitl --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --camera-serial top=TOP_SERIAL --norm-tag CHECKPOINT_NORM_TAG \
  --server http://GPU_HOST:8202 --repo-id local/so101-dagger \
  --episodes 10 --episode-seconds 30 --fps 30 --open-quest
```

Rollout/HITL ask for the task per episode. In HITL, right B takes control;
left Y taps back to the policy or holds to end the attempt. Follow the printed
review controls to save, mark success/failure, or discard.

## Policy and data contract

Read [MolmoAct2 integration](docs/inference.md) before enabling policy control.
A checkpoint must match the arm count, joint ordering, camera views, units and
gripper polarity. The YAM checkpoint is not a pretrained SO101 policy.
[Recording and DAgger](docs/recording.md) documents the LeRobot v3 feature schema
and intervention labels. This repository collects DAgger data; model retraining
runs in your MolmoAct2 training environment.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/openpi_control/` | Karma CLI, hardware sessions, policy client, recording and calibration |
| `src/vr_teleop_kit/` | Bundled Quest WebXR client, relay and IK |
| `native/pi_control/` | C++ CAN/FeeTech control and safety |
| `src/openpi_control/models/` | YAM/SO101 models and effectors |
| `calibration/` | Local per-arm profiles and firmware backups (ignored) |
| `scripts/` | Pinned native dependency build and USB setup |
| `docs/` | Hardware, camera, policy and data setup |
| `tests/` | Hardware-free tests and native checks |

Logs default to `logs/`; override with `KARMA_LOG_DIR`. Run tests with
`uv run --extra dev pytest`. Physical calibration, camera placement and a
checkpoint trained for your rig still need validation on your workstation.
