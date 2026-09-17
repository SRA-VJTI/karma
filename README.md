# Karma

Karma is the robot-side workstation stack for **bimanual YAM** and **SO100/SO101**
arms: Meta Quest VR teleoperation, MolmoAct2 policy inference, autonomous
rollout recording, and human-in-the-loop (HITL / DAgger) data collection.
The Quest relay and WebXR client, the IK, the native motor runtime, the camera
capture layer and the LeRobot v3 dataset writer all live in this repository.

Everything runs through one executable, `uv run karma`, with four main workflows:

| Workflow | Command | What it does |
| --- | --- | --- |
| Teleop | `karma teleop` | Drive the arms from a Quest; add `--record` to write demonstrations |
| Inference | `karma inference` | Run a MolmoAct2 checkpoint continuously with one instruction |
| Rollout | `karma rollout` | Timed policy episodes, task asked per episode, saved as a dataset |
| HITL | `karma hitl` | Policy drives, you take over from the Quest when needed; DAgger labels |

The model itself runs elsewhere: a MolmoAct2 **server** on a GPU host, which
Karma talks to over HTTP. See [Model servers](docs/model-servers.md).

## Supported hardware

| Rig | Arms | Bus | State/action width | Cameras |
| --- | --- | --- | --- | --- |
| `yam_bimanual` | two YAM, E_Yam grippers | SocketCAN (`can0`, `can1`) | 14 (6 joints + gripper per arm) | `top`, `left_wrist`, `right_wrist` |
| `so101` | one SO101 (right) | USB serial (`/dev/ttyACM0`) | 6 (5 joints + gripper) | `top` plus optional `side`, `right_wrist` |
| `so101_bimanual` | two SO101 | USB serial | 12 | `top` plus optional wrists |

Cameras can be **Intel RealSense** (addressed by serial, captured through the
SDK) or **plain USB/UVC webcams** (addressed by `/dev/videoN`, captured through
OpenCV). Both go through the same reader interface, so every workflow accepts
either. See [Cameras](docs/cameras.md).

SO100 hardware runs with the SO101 model and calibration; original SO100
gearing/geometry has not been validated by the SO101 checks.

## Install once

Ubuntu Linux, USB 3 for RealSense, Python 3.12 or 3.13. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
UV_HTTP_TIMEOUT=180 uv sync --locked
uv run karma --help
```

The first sync downloads the complete runtime (LeRobot, PyTorch CPU wheels,
camera SDK, VR packages) and builds the native `pi_control_node`. No sibling
repository or `PYTHONPATH` is needed. If your shell exports a ROS `PYTHONPATH`,
run Karma with `env -u PYTHONPATH uv run karma ...`.

USB access and the Quest:

```bash
sudo usermod -aG dialout,video "$USER"       # log out and back in afterwards
sudo apt-get install adb
sudo install -m 0644 scripts/udev/51-meta-quest.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

Enable developer mode and USB debugging on the Quest, reconnect it, accept the
headset's prompt; `adb devices` must report `device`.

## 1. YAM (bimanual or single arm)

Full walkthrough with CAN pinning, zeros and troubleshooting: [YAM setup](docs/yam-setup.md).

### Once: i2rt files, CAN buses, meshes

Quest teleop's IK loads the YAM MJCF from an [i2rt](https://github.com/i2rt-robotics/i2rt)
checkout (not vendored); clone it inside the Karma checkout, where it is found
automatically and git-ignored. Fetch the browser meshes once too:

```bash
git clone https://github.com/i2rt-robotics/i2rt.git
uv run karma-viz --fetch-meshes --model Yam
```

Each arm is one USB‑CAN adapter. Bring the buses up at 1 Mbit/s and check the arms:

```bash
ls -l /sys/class/net/can*
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
uv run karma doctor --rig yam_bimanual
```

`can0`/`can1` follow USB enumeration order, so either pin names with a udev rule
(see the setup doc) or pass `--interface left=BUS --interface right=BUS` on every
command. YAM uses firmware motor zeros and the packaged instance configuration;
there is no calibration JSON. Only write zeros at the manufacturer's mechanical
zero pose (`karma zero ... --dry-run` first).

### Cameras

| Role | Camera | Placement |
| --- | --- | --- |
| `top` | RealSense D435 | Fixed overhead, both arms' shared workspace |
| `left_wrist` | RealSense D405 | Left wrist, looking at the grasp |
| `right_wrist` | RealSense D405 | Right wrist, looking at the grasp |

```bash
uv run python -c 'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
uv run karma cameras --rig yam_bimanual \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL --probe
```

### Teleop: both arms or one

```bash
# both arms
uv run karma teleop --rig yam_bimanual \
  --interface left=can0 --interface right=can1 --open-quest

# one arm: same rig, one Quest hand, only that bus is opened
uv run karma teleop --rig yam_bimanual --arm right --interface right=can0 --open-quest
uv run karma teleop --rig yam_bimanual --arm left  --interface left=can0  --open-quest
```

In VR: hold grip to clutch, trigger for the gripper, thumbstick click returns to
rest. Ctrl+C parks then de-energizes (`--no-park` skips the parking move).

### Demonstrations

```bash
uv run karma teleop --record --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --repo-id local/yam-demo --task "fold the towel" --fps 30 --open-quest
```

Single-arm recording: add `--arm right` and declare only `top` and `right_wrist`.

### Policy: inference, rollout, HITL

Start the YAM MolmoAct2 server on the GPU host first (port 8202, norm tag
`yam_dual_molmoact2`; see [Model servers](docs/model-servers.md)). The client
checks the server's health contract before energizing anything. The bimanual
checkpoint needs both arms and all three views.

```bash
# Continuous run, one instruction
uv run karma inference --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_HOST:8202 --norm-tag yam_dual_molmoact2 \
  --instruction "fold the towel"

# Timed episodes, task asked per episode, saved as a LeRobot v3 dataset
uv run karma rollout --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_HOST:8202 --norm-tag yam_dual_molmoact2 \
  --repo-id local/yam-rollouts --episodes 10 --episode-seconds 30 --fps 30

# DAgger: policy drives, Right B takes over, Left Y tap hands back / hold ends
uv run karma hitl --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_HOST:8202 --norm-tag yam_dual_molmoact2 \
  --repo-id local/yam-dagger --episodes 10 --episode-seconds 30 --fps 30 --open-quest
```

## 2. SO100 / SO101

### Browser view

Every SO101 command that opens Viser draws the real arm once its meshes are
cached (otherwise a skeleton and a `visual meshes not cached` warning):

```bash
uv run karma-viz --fetch-meshes --model SO101
uv run karma-viz --rig so101          # hardware-free viewer with sliders
```

### Calibrate each arm

Every SO101 arm gets its own profile from the calibration wizard (torque off,
mid-travel centring, full sweep, closed/open gripper, home pose):

```bash
uv run karma calibrate-so101 --interface /dev/ttyACM0 \
  --output calibration/so101-right.json --center-encoders
```

Repeat per arm. The `calibration/` directory is ignored by Git — back it up
yourself. Details in [SO101 setup](docs/so101-teleop.md).

### Cameras: RealSense or webcams

Only `top` is declared by default; add views by role. The SO100/SO101 MolmoAct2
checkpoint uses `top` and `side`. With RealSense, pass serials; with USB webcams,
pass device paths (find the capture node of each with `ls -l /dev/v4l/by-path/`):

```bash
# RealSense
uv run karma cameras --rig so101 --camera-serial top=TOP_SERIAL --camera-serial side=SIDE_SERIAL --probe
# USB webcams (OpenCV)
uv run karma cameras --rig so101 --camera top=/dev/video5 --camera side=/dev/video7 --probe --snapshot /tmp/cams
```

`--probe` opens each camera, measures the delivered rate and (with `--snapshot`)
writes a frame per view so you can check framing.

### Teleop and demonstrations

```bash
uv run karma teleop --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --rate 50 --open-quest

uv run karma teleop --record --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --camera top=/dev/video5 --camera side=/dev/video7 \
  --repo-id local/so101-demo --task "pick up the object" --fps 30 --open-quest
```

Bimanual: `--rig so101_bimanual --interface left=/dev/ttyACM1 --interface right=/dev/ttyACM0`
with a `--calibration` per arm.

### Policy frame for the public checkpoint (work in progress)

Karma's policy wire is calibrated **radians** with a mid-travel zero and a
0=open gripper. The public `allenai/MolmoAct2-SO100_101` checkpoint was trained
on LeRobot **v1** datasets: **degrees**, zero at LeRobot's "arm straight out"
pose, gripper 0–100. Feeding it radians, or executing its degrees as radians,
drives the arm into its limits. The `--policy-frame` file bridges the two and is
captured once per arm with two reference poses:

```bash
uv run karma so101-policy-frame --interface /dev/ttyACM0 \
  --calibration calibration/so101-right.json \
  --output calibration/so101-right.policy-frame.json
```

**Status: work in progress.** The capture works and the maths is tested against
LeRobot's own calibration formula, but the result depends on how precisely the
two poses are held; in practice the wrist-roll offset needed a manual correction
against the checkpoint's rest state. Importing a LeRobot calibration file
instead of posing is planned. Read [Inference › Policy frames](docs/inference.md#policy-frames)
before running the policy, and verify the mapped rest pose against the
checkpoint's before energizing.

### Policy: inference, rollout, HITL

Start the SO100 MolmoAct2 server on the GPU host (port 8203, norm tag
`so100_so101_molmoact2`, two cameras `top_cam`/`side_cam`; see
[Model servers](docs/model-servers.md)).

```bash
# Continuous run, one instruction
uv run karma inference --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --policy-frame calibration/so101-right.policy-frame.json \
  --camera top=/dev/video5 --camera side=/dev/video7 \
  --server http://GPU_HOST:8203 --norm-tag so100_so101_molmoact2 \
  --speed 0.5 --instruction "pick up the object"

# Timed episodes, task asked per episode, recorded
uv run karma rollout --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --policy-frame calibration/so101-right.policy-frame.json \
  --camera top=/dev/video5 --camera side=/dev/video7 \
  --server http://GPU_HOST:8203 --norm-tag so100_so101_molmoact2 \
  --repo-id local/so101-rollouts --episodes 5 --episode-seconds 30 --fps 30 --speed 0.5

# DAgger with Quest takeover
uv run karma hitl --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --policy-frame calibration/so101-right.policy-frame.json \
  --camera top=/dev/video5 --camera side=/dev/video7 \
  --server http://GPU_HOST:8203 --norm-tag so100_so101_molmoact2 \
  --repo-id local/so101-dagger --episodes 10 --episode-seconds 30 --fps 30 \
  --speed 0.5 --open-quest
```

For a checkpoint you trained on Karma's own recordings, drop `--policy-frame`
and use that checkpoint's norm tag. `--speed 0.5` plays each chunk at half the
training rate for first runs; `--max-step-rad` bounds any single tick.

## Policy and data contract

A checkpoint must match the rig's arm count, joint order, camera roles, units
and gripper polarity; the YAM checkpoint is not an SO101 policy and vice versa.
[Inference](docs/inference.md) documents the HTTP contract, the health check
and the policy-frame mechanism; [Model servers](docs/model-servers.md) what each
server must expose; [Recording](docs/recording.md) the LeRobot v3 schema and
[DAgger](docs/dagger.md) the intervention workflow. Karma collects data; training
runs in your MolmoAct2 environment.

## Documentation

| Document | Covers |
| --- | --- |
| [Commands](docs/cli.md) | Every `karma` subcommand and the shared flags |
| [Cameras](docs/cameras.md) | RealSense and OpenCV capture, roles, probing, troubleshooting |
| [Inference](docs/inference.md) | Policy contract, HTTP payload, health check, policy frames |
| [Model servers](docs/model-servers.md) | The YAM and SO100 MolmoAct2 servers Karma connects to |
| [YAM setup](docs/yam-setup.md) | CAN buses, zeros, home, teaching handle |
| [SO101 setup](docs/so101-teleop.md) | Calibration wizard, teleop, IK tuning, Quest troubleshooting |
| [Recording](docs/recording.md) | LeRobot v3 dataset schema |
| [DAgger](docs/dagger.md) | HITL workflow and intervention labels |
| [Safety](docs/safety.md) | Preflight and runtime checks |
| [Viser](docs/viser.md) | Browser visualization |

## Acknowledgements

Karma builds on [openpi-basic-control](https://github.com/Physical-Intelligence/openpi-basic-control)
by Physical Intelligence (the control stack, native runtime, models and policy
client this repository grew out of) and on
[vr-teleop-kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit) by
Dream Machines (the Quest teleoperation runtime vendored under
`src/vr_teleop_kit/`, Apache 2.0). The SO101 arm description is
[SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) by The Robot Studio;
the YAM models, MJCF and meshes come from [i2rt](https://github.com/i2rt-robotics/i2rt).
See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `src/openpi_control/` | Karma CLI, hardware sessions, policy client, cameras, recording, calibration |
| `src/vr_teleop_kit/` | Bundled Quest WebXR client, relay and IK |
| `native/pi_control/` | C++ CAN/FeeTech control and safety |
| `src/openpi_control/models/` | YAM and SO101 arm/effector models |
| `calibration/` | Local per-arm profiles, policy frames and firmware backups (ignored) |
| `scripts/` | Native dependency build and USB setup |
| `docs/` | Hardware, camera, policy and data documentation |
| `tests/` | Hardware-free tests |

Logs go to `logs/` (`KARMA_LOG_DIR` overrides). Run tests with
`uv run --extra dev pytest`. Physical calibration, camera placement and a
checkpoint trained for your rig still need validation on your workstation.
