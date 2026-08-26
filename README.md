# OpenPI Control for Karma

Robot-cell control for the Karma bimanual YAM platform: native arm control,
Meta Quest VR teleoperation, LeRobot v3 demonstration collection, MolmoAct2
inference, rollout recording, RealSense cameras, and live Viser visualization.
Each arm runs one `pi_control_node` process and communicates with the Python
operator process over ZeroMQ.

> [!CAUTION]
> `teleop`, `collect`, `infer`, and `rollout` energize physical robot arms.
> Run them in the foreground with the workspace clear. Their normal Ctrl-C
> paths park the selected arms at `home_pos` and then de-energize them; do not
> use `--no-park` unless stopping in place is intentional and safe.

| Arm | Joints | Bus | Follower effector | Leader effector |
| --- | --- | --- | --- | --- |
| `Yam` | 6 | SocketCAN | `E_Yam` | `E_Yam_Handle` |
| `ARX_X5` | 6 | SocketCAN | `E_ARX` | `E_ARX_ENC` |
| `ARX_L5` | 6 | SocketCAN | `E_ARX` | `E_ARX_ENC` |
| `ARX_ENC` | 6 | SocketCAN | — (leader only, read-only) | `E_ARX_ENC` |
| `FR3` | 7 | Franka controller | `Robotiq` | — |
| `SO101` | 5 | USB serial | `E_SO101` | — |
| `Trossen_wai_ctrl` | 6 | Ethernet controller | `E_Trossen_ctrl` | — |

```python
from openpi_control import ArmConfig, ArmSession, PositionCommand, SocketCanConnection

with ArmSession() as session:
    follower = session.add_follower(
        ArmConfig(
            "right_follower",
            "Yam",
            SocketCanConnection("can_follower_r"),
            effector_model="E_Yam",
            follower_gravity_compensation=True,
        )
    )
    session.connect()
    follower.command(PositionCommand([0, 0, 0, 0, 0, 0], 1.0))
```

FR3 uses the same `ArmSession`, `FollowerArm`, and `PositionCommand` API. Its
connection selects one of the two real Robotiq Modbus transports:

```python
from openpi_control import FR3Connection, RobotiqConnection

config = ArmConfig(
    "follower",
    "FR3",
    FR3Connection("192.168.1.10"),
    effector_model="Robotiq",
    effector_connection=RobotiqConnection.rtu("/dev/serial/by-id/usb-robotiq"),
    # Or: RobotiqConnection.tcp("192.168.1.11", port=502),
)
```

See [docs/fr3.md](docs/fr3.md) for firmware, networking, controller, and
hardware-validation details.

## Installing

```bash
uv sync
```

That is the whole install for an operator box: cameras, MolmoAct2 inference,
the Viser scene, VR teleop, LeRobot v3 collection, and the test tools all arrive
together. `uv sync` is an *exact* sync -- it uninstalls whatever the command
did not ask for -- so
naming one extra at a time (`uv sync --extra cameras`) used to strip the extras
a live cell had just installed. The `cell` dependency-group in `pyproject.toml`
is a default group, so it is always included and no later `--extra` can drop it.

LeRobot pulls in torch, so the first sync is larger than a control-only install.
It is included deliberately: `uv run openpi collect ...` works as written and
does not discover a missing dataset writer after a hardware session begins.

## Robot-cell quick start

All normal operator workflows use the unified `openpi` entry point:

```bash
uv run openpi --help
uv run openpi <command> --help
```

The commands below assume this robot box's persistent SocketCAN names:

```bash
ip -brief link show type can
```

Expected:

```text
can_right  UP  <NOARP,UP,LOWER_UP,ECHO>
can_left   UP  <NOARP,UP,LOWER_UP,ECHO>
```

Run `uv sync` once after pulling code or changing dependencies. Do not prefix
every operator command with another `uv sync`; `uv run` uses the synchronized
environment directly.

### 1. Meta Quest teleoperation

Bimanual VR teleoperation:

```bash
uv run openpi teleop --arm both --open-quest \
    --interface left=can_left \
    --interface right=can_right
```

Single-arm VR teleoperation:

```bash
uv run openpi teleop --arm left --open-quest \
    --interface left=can_left

uv run openpi teleop --arm right --open-quest \
    --interface right=can_right
```

With the default USB transport, that one command starts the in-process WebXR
relay, creates `adb reverse tcp:8443 tcp:8443`, and opens the relay page in the
Quest browser. Do not run a separate relay or `adb reverse` for this path.
Use `--adb-serial SERIAL` only when `adb devices` shows more than one Quest.

Quest controls:

| Control | Action |
| --- | --- |
| Grip | Clutch that controller to its arm; releasing grip freezes the arm target |
| Trigger | Close/open that arm's gripper |
| A / X | Precision mode while held |
| Thumbstick click | Smoothly return that arm to its configured rest pose |
| Ctrl-C in the terminal | Park both selected arms and de-energize |

### 2. Record a teleoperated LeRobot v3 dataset

Bimanual collection with the top camera and both wrist cameras:

```bash
uv run openpi collect --arm both --open-quest \
    --interface left=can_left \
    --interface right=can_right \
    --format lerobot-v3 \
    --repo-id SRA-VJTI/fold-pink-towel-v1 \
    --task "Fold the pink towel in half" \
    --num-episodes 20
```

Use a concrete `--task`; it is stored on every frame. Never leave example text
such as `Describe the manipulation task` in a training dataset. Unless
`--root` is supplied, LeRobot writes under
`~/.cache/huggingface/lerobot/<repo-id>`. To choose the location explicitly:

```bash
uv run openpi collect --arm both --open-quest \
    --interface left=can_left \
    --interface right=can_right \
    --repo-id SRA-VJTI/fold-pink-towel-v1 \
    --root ~/openpi-data/datasets/fold-pink-towel-v1 \
    --task "Fold the pink towel in half" \
    --num-episodes 20
```

Single-arm collection automatically keeps the top camera and only the selected
arm's wrist camera:

```bash
uv run openpi collect --arm left --open-quest \
    --interface left=can_left \
    --repo-id SRA-VJTI/left-arm-task-v1 \
    --task "Place the object in the tray" \
    --num-episodes 20

uv run openpi collect --arm right --open-quest \
    --interface right=can_right \
    --repo-id SRA-VJTI/right-arm-task-v1 \
    --task "Place the object in the tray" \
    --num-episodes 20
```

Open <http://localhost:8080> during collection for measured-arm and live-camera
Viser visualization. The collector shares each RealSense reader between the
dataset, Viser, and Quest; it does not open the cameras three times.

| In VR | Episode action |
| --- | --- |
| Right B | Start recording; while recording, discard that take and restart it |
| Left Y | Save the current episode |
| Ctrl-C | Discard an unsaved open take, finalize saved episodes, park, and de-energize |

The session exits automatically after `--num-episodes` successful saves. Add
`--push-to-hub --private` to upload only after the dataset is finalized and the
arms are down. Use a fresh `--repo-id`/`--root` when starting a separate
dataset.

### 3. Run MolmoAct2 inference

Start the MolmoAct2 HTTP server from its checkout on the GPU machine:

```bash
uv run python host_server_yam.py \
    --host 0.0.0.0 \
    --port 4090 \
    --cuda-graph
```

Then run the robot-side client on the Karma box:

```bash
uv run openpi infer \
    --server http://192.168.0.107:4090 \
    --instruction "Fold the pink towel in half" \
    --interface left=can_left \
    --interface right=can_right \
    --speed 0.5 \
    --port 8080
```

`infer` opens all three cameras, powers both arms, sends observations in
`top, left_wrist, right_wrist` order, executes bounded action chunks, and
serves Viser at <http://localhost:8080>. The Viser camera tiles are the exact
policy-input observations, not independent preview frames. Ctrl-C or an
inference/hardware error parks and de-energizes both arms.

The server URL may be `host:port` or a full URL; `/act` is appended
automatically. `--speed 0.5` follows the same planned path over twice the
control ticks and is the normal conservative starting point on this cell.

### 4. Record MolmoAct2 policy rollouts

Use `rollout`, not `collect`, when MolmoAct2 drives the arms and each policy
attempt should become a labeled LeRobot v3 episode:

```bash
uv run openpi rollout \
    --repo-id SRA-VJTI/molmo-fold-pink-towel-rollouts-v1 \
    --root ~/openpi-data/rollouts/molmo-fold-pink-towel-v1 \
    --episodes 3 \
    --episode-seconds 120 \
    --server http://192.168.0.107:4090 \
    --interface left=can_left \
    --interface right=can_right \
    --speed 0.5 \
    --port 8080
```

The first attempt starts by asking for its prompt. Before every later attempt,
the command asks you to reset the scene and then enter that episode's prompt.
Afterward it asks `Episode N successful? [y/n]:`. Every frame stores the prompt
in LeRobot's `task` field, while
`openpi_control_rollouts.json` stores the success label, interruption status,
and saved dataset episode index.

Ctrl-C during an active rollout saves the captured prefix as a partial episode,
parks both arms, and continues to the result prompt; it differs deliberately
from `collect`, where Ctrl-C discards an unfinished demonstration. A second
Ctrl-C at an interactive prompt exits the multi-episode run. Viser is available
at <http://localhost:8080> throughout each attempt.

### Workflow summary

| Goal | Command | Driver | Cameras | Dataset |
| --- | --- | --- | --- | --- |
| Move arms from Quest | `uv run openpi teleop` | Human/Quest | Quest preview | No |
| Record demonstrations | `uv run openpi collect` | Human/Quest | Top + selected wrist camera(s) | LeRobot v3 |
| Run a policy | `uv run openpi infer` | MolmoAct2 | Top + both wrists | No |
| Record policy attempts | `uv run openpi rollout` | MolmoAct2 | Top + both wrists | LeRobot v3 + rollout manifest |

All four hardware workflows preflight before energizing and own their complete
shutdown path. Keep them in the foreground and use Ctrl-C rather than killing
their native child processes directly.

## Operator CLI

`openpi` preflights an arm, sets its servo zeros, and brings a rig up
and back down. Every run logs to `~/openpi-data/logs/runtime/`.

```bash
uv run openpi doctor --model Yam --interface can_right --effector E_Yam
uv run openpi zero   --model Yam --interface can_right --dry-run
uv run openpi doctor --rig yam_bimanual \
    --interface-override left=can_left \
    --interface-override right=can_right
uv run openpi live --rig yam_bimanual \
    --interface left=can_left --interface right=can_right
```

`doctor` is read-only and opens no bus unless given `--probe`; it checks the
packaged assets, that every `servo_model` has a driver, and that the interface
is present, up, and at the bit rate the model wants. `zero` writes the arm's
current pose as each servo's firmware zero, so it confirms first. See
[docs/cli.md](docs/cli.md).

## Rigs, and turning them on and off

A rig names a whole cell — which arms it has, the bus each sits on, and where
their bases sit relative to each other — so the CLI and the visualizer mean the
same thing by `left`. `yam_bimanual` is two YAM followers with `E_Yam` grippers,
with packaged defaults of left on `can0` and right on `can1`. This cell uses
persistent aliases instead, so its operator commands pass
`--interface left=can_left --interface right=can_right`.

`openpi live` is the one command here that energizes an arm, and it owns
the whole arc:

```bash
uv run openpi live --rig yam_bimanual \
    --interface left=can_left --interface right=can_right
uv run openpi live --rig yam_bimanual --float \
    --interface left=can_left --interface right=can_right
uv run openpi live --rig yam_bimanual --only left --no-viz \
    --interface left=can_left
```

Preflight runs first and nothing is energized unless every arm passes. Both
followers then come up *holding* — `--float` is the opt-in for a backdrivable
arm. ctrl-c parks each arm at the `home_pos` in its instance JSON before cutting
torque, so an arm is never dropped from wherever it stood.

There is no separate `up` and `down`: `pi_control_node` holds a liveness pipe to
the process that spawned it, so an arm cannot stay energized after the command
returns. One foreground process owns the lifecycle, and ctrl-c is the way out.

## Running MolmoAct2 on the YAM hardware

The robot-side inference command uses the MolmoAct2 `/act` HTTP contract. Start
the GPU inference server separately, then run the client on the robot box:

```bash
uv run openpi infer \
    --server http://192.168.0.107:4090 \
    --instruction "Fold the pink towel in half" \
    --interface left=can_left \
    --interface right=can_right \
    --speed 0.5 \
    --port 8080
```

It opens the three YAM cameras, powers both arms, executes whole action chunks
with bounded interpolation, and serves a Viser page showing measured poses,
camera previews, and translucent predicted end-effector trails. The command
parks both arms at `home_pos` on Ctrl-C or an inference/hardware error.

The wire defaults match the reference MolmoAct2 deployment -- CUDA-graph
requests, JPEG frame transport, a keep-alive connection, full 30-action chunks.
How those chunks are executed deliberately does not: `--reach-actions`,
`--prefetch`, and `--reset-start-pose` switch the reference runtime on piece by
piece. See [docs/inference.md](docs/inference.md) for the wire ordering and why.

### Recording policy rollouts

Sync the operator environment once, then use `rollout` for timed episodes. The
command asks for a new prompt before every episode and asks `y/n` after each
episode, including a Ctrl-C-interrupted partial episode. Ctrl-C stops only the
current episode: its captured frames are saved, both arms park at `home_pos`,
and the next episode starts after you reset the same towel.

```bash
uv run openpi rollout \
    --repo-id SRA-VJTI/molmo-fold-pink-towel-rollouts-v1 \
    --root ~/openpi-data/rollouts/molmo-fold-pink-towel-v1 \
    --episodes 3 \
    --episode-seconds 120 \
    --server http://192.168.0.107:4090 \
    --interface left=can_left \
    --interface right=can_right \
    --speed 0.5 \
    --port 8080
```

LeRobot v3 data is written under `--root`. The same directory receives
`openpi_control_rollouts.json`, which stores the per-attempt prompt, whether
the attempt was interrupted, and its `y/n` label. The dataset's `task` field
also contains the prompt on every frame.

## Cameras

Three RealSense D405s watch the bimanual cell: one overhead, one per wrist. They
are part of the rig, pinned by serial number — a `/dev/videoN` is not a camera,
it changes with boot order and which port you used.

```bash
uv run openpi cameras                # what is plugged in, and where
uv run openpi cameras --probe        # open each one and measure it
```

A wrist camera names the arm it rides on, so `--only right` narrows to `top` and
`right_wrist` without anyone special-casing it. Discovery itself reads udev and
nothing else, so `doctor --rig` can report an unplugged camera on a box with no
SDK installed.

Two defaults are measurements rather than taste: 848x480 (the D405's native
mode — 640x480 makes the firmware rescale and three cameras drop to 15-20 fps),
and capture through `pyrealsense2` rather than OpenCV (whose V4L2 path tops out
near 10-13 fps on the same node that `v4l2-ctl` streams at 30). See
[docs/cameras.md](docs/cameras.md).

## Collecting data

`openpi collect` is the integrated Meta Quest collection command. It starts the
relay and USB tunnel, teleoperates the selected arm(s), writes LeRobot v3 —
parquet for state/action and one MP4 per camera — and serves Viser at
<http://localhost:8080> with measured arm poses and the live frames being
recorded.

```bash
uv run openpi collect --arm both --open-quest \
    --interface left=can_left \
    --interface right=can_right \
    --format lerobot-v3 \
    --repo-id SRA-VJTI/fold-pink-towel-v1 \
    --task "Fold the pink towel in half" \
    --num-episodes 20
```

`--arm both` records `top`, `left_wrist`, and `right_wrist`. A single-arm
collection keeps the head/top view and only that arm's wrist camera:

```bash
uv run openpi collect --arm right --open-quest \
    --interface right=can_right \
    --repo-id SRA-VJTI/right-arm-pick-v1 \
    --task "Pick up the object"
```

One RealSense capture is shared by the dataset writer, Viser, and Quest WebRTC,
so all three camera consumers can run together without fighting over the device.
Use `--no-viz` only when the browser preview is not wanted. `record` remains as
the lower-level/legacy command for an independently managed relay.

The complete Quest stack is vendored in this repository: the WebXR relay,
Quest page, clutch-relative pose mapping, YAM inverse kinematics, haptics, and
camera publisher. The only external runtime asset is the i2rt YAM MJCF; set
`YAM_XML` or place the i2rt checkout at `./i2rt`.

For direct hardware teleoperation, one command starts the relay, establishes
the Quest USB tunnel, powers the native arms, and parks them on exit:

```bash
uv run openpi teleop --arm both --open-quest \
    --interface left=can_left \
    --interface right=can_right

uv run openpi teleop --arm right --open-quest \
    --interface right=can_right
```

Use `--quest-transport lan` with TLS certificates for a Wi-Fi Quest session.
Use `openpi relay` when the relay should remain running independently. The
Quest controls are: grip to clutch, trigger to control the gripper, A/X for
precision, thumbstick click to return one arm to rest, and B/Y for recording
signals. Right B starts an episode and left Y saves it when using `collect` or
`record`.

The legacy `--vr-kit` option remains accepted for migration, but normal runs
import the in-repository `vr_teleop_kit` package directly.

`--teleop hold --dry-run` rehearses a whole session — arms up, cameras open,
nothing written — so the pipeline can be checked without a headset.

Note that the gripper convention is **inverted** between this package (`1.0` =
open) and LeRobot (`0.0` = open); recorded datasets use LeRobot's, and
`record.to_native_gripper` is the one place that converts back. That and the
other three ways a dataset comes out quietly wrong are in
[docs/recording.md](docs/recording.md).

## Visualizing an arm

`openpi_control.viz` serves any packaged model in the browser with
[viser](https://viser.studio). It opens no bus and starts no native node, so it
runs with the hardware down or absent.

```bash
uv run openpi-control-viz --fetch-meshes --model Yam     # once, needs network
uv run openpi-control-viz --model Yam --effector E_Yam   # http://localhost:8080
```

The wheel ships each URDF but not its meshes — the URDFs are here for the
gravity-compensation model, which needs link inertias and joint origins and
never needs geometry. `--fetch-meshes` caches I2RT's YAM meshes (MIT) under
`~/openpi-data/meshes/`, after which every run renders the real arm offline with
no flags. Without them you get a kinematic skeleton built from the joint
origins, which needs no assets at all.

GUI sliders drive the joints. To follow a live arm instead, hand it poses from
your own session — the visualizer only draws, so hardware stays your call:

```python
from openpi_control.viz import ArmVisualizer

viz = ArmVisualizer("Yam", effector_model="E_Yam")
viz.update(follower.read_state().joints.position_rad)
```

`ArmSceneVisualizer` puts several arms in one scene, each with its own base
pose. Pass it a packaged rig to draw a whole cell:

```bash
uv run openpi-control-viz --rig yam_bimanual   # both arms, sliders, no bus
```

To mirror two *live* arms instead of sliders, use `openpi live` — it
owns the power-on and power-off that a live view implies. See
[docs/viser.md](docs/viser.md).

## Documentation

| Doc | Covers |
| --- | --- |
| [docs/cli.md](docs/cli.md) | `doctor` checks, `zero` safeguards, rigs, `live` power on/off |
| [docs/cameras.md](docs/cameras.md) | camera identity, discovery, the two D405 serials, capture rates |
| [docs/recording.md](docs/recording.md) | LeRobot datasets, VR teleop, gripper polarity, episode boundaries |
| [docs/inference.md](docs/inference.md) | MolmoAct2 HTTP inference, action chunks, and hardware execution |
| [docs/viser.md](docs/viser.md) | render modes, mesh sourcing, rigs, joint ordering |
| [docs/fr3.md](docs/fr3.md) | FR3 firmware, networking, controller, validation |
| [docs/yam_teaching_handle.md](docs/yam_teaching_handle.md) | YAM handle CAN protocol and trigger calibration |

## Building from source

Building requires CMake and a C++17 compiler. The dependency builder pins and
builds Pinocchio, ZeroMQ, cppzmq, Trossen, libfranka 0.21.3, and libmodbus;
the resulting libfranka and libmodbus archives are linked into
`pi_control_node`. It has been tested on Ubuntu 22.04 and 24.04.

```bash
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
uv build --wheel
```

The wheel is written to `dist/`.
