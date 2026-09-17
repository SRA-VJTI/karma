# YAM setup

One YAM arm is six DM CAN motors plus an E_Yam (linear_4310) gripper on one
USB‑CAN adapter. The packaged rig `yam_bimanual` is **left on `can0`, right on
`can1`**; a single arm is the same rig restricted with `--arm`. Everything below
was done on the reference cell in this order.

## 1. Karma and the i2rt checkout

Install Karma as in the [README](../README.md#install-once). The native motor
runtime, the URDF and the IK are in this repository; two things come from
I2RT's own repository:

- **The YAM MJCF for Quest teleop.** The VR IK loads `yam.xml` plus the
  `linear_4310` gripper MJCF from an i2rt clone. They are deliberately not
  vendored. Clone i2rt *inside* the Karma checkout, where it is picked up
  automatically (the `i2rt/` directory is git-ignored), or point `YAM_XML` /
  `--yam-xml` at it anywhere else:

  ```bash
  git clone https://github.com/i2rt-robotics/i2rt.git    # in the karma checkout
  # or, for a clone elsewhere:
  export YAM_XML=/path/to/i2rt/i2rt/robot_models/arm/yam/yam.xml
  ```

  Teleop and HITL need this; inference and rollout do not. Without it teleop
  stops at startup with "No YAM model configured".

- **The visual meshes for the browser view.** Downloaded once into
  `meshes/Yam/` (next to `logs/`, git-ignored) from the pinned i2rt revision:

  ```bash
  uv run karma-viz --fetch-meshes --model Yam
  ```

  Without them every command prints `[WARN] visual meshes not cached` and draws
  a skeleton; nothing else changes.

You do not need i2rt's Python package installed for Karma — only its files.
Its motor tools are still useful for firmware-level jobs (see zeros below).

## 2. CAN buses

Each adapter is a CANable/candlelight device that appears as a SocketCAN
interface. Bring them up at 1 Mbit/s:

```bash
ls -l /sys/class/net/can*                      # which adapters are detected
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
ip -brief link show type can                   # both should say UP
```

A bus that stops answering (`RTNETLINK answers: Device or resource busy`,
timeouts in `doctor`) is fixed by taking it down and up again, or replugging
the adapter; i2rt ships `scripts/reset_all_can.sh` that does exactly that for
every `can*` interface.

### Which adapter is which

`can0`/`can1` are assigned in USB enumeration order, so after a reboot or a
replug the left and right arm can swap. Either pass the buses explicitly on
every command (`--interface left=can1 --interface right=can0`) or pin names
with udev, as i2rt's hardware guide recommends. Plug the adapters in one at a
time and read each serial:

```bash
udevadm info -a -p /sys/class/net/can0 | grep -i serial
```

Then `/etc/udev/rules.d/90-can.rules`, one line per adapter (names must start
with `can` and be at most 13 characters):

```
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="LEFT_ADAPTER_SERIAL",  NAME="can_left"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="RIGHT_ADAPTER_SERIAL", NAME="can_right"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Replug, check `ip link show`, and use `--interface left=can_left --interface right=can_right`
from then on. To bring the buses up automatically at boot, i2rt's
`sudo sh devices/install_devices.sh` installs a udev rule that runs
`ip link set ... up` for every `can*` interface; a systemd-networkd `.network`
file with `[CAN] BitRate=1M` does the same.

### Check the arms

```bash
uv run karma doctor --rig yam_bimanual
uv run karma doctor --rig yam_bimanual --interface-override right=can_right --probe   # also listen on the bus
```

`doctor` is read-only. It checks the model assets, that each bus exists and is
up, the servo registry, and (with `--probe`) that motors answer. Never run two
robot processes on one bus.

## 3. Zeros and home

YAM uses **firmware motor zeros** and the packaged instance configuration —
there is no calibration JSON as for SO101. `src/openpi_control/models/arms/Yam/Yam_01.json`
stores per-servo `zero_pos` and `home_pos` (radians, relative to the firmware
zero); `home_pos` is where Ctrl+C parks. The gripper instance is
`models/effectors/E_Yam/E_Yam_01.json`; native gripper position is 1=open.

Zeros only need writing after a motor was replaced or the arm disassembled, and
only with the arm held at the manufacturer's mechanical zero pose (folded park
pose, all joints at 0) — a chosen rest pose is not the zero pose. Preview, then
write:

```bash
uv run karma zero --model Yam --interface can0 --effector E_Yam --dry-run
uv run karma zero --model Yam --interface can0 --effector E_Yam        # arm at the mechanical zero
```

`--joint N` zeros a single motor. This is the same operation as i2rt's
`python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id N`, run
from their checkout; use whichever you have at hand, not both.

The DM motors have a 400 ms safety timeout from the factory: no command for
400 ms puts a motor in damping mode. Karma's native node commands at 200 Hz, so
leave the timeout enabled; i2rt's `set_timeout.py` can disable it for their
gravity-compensation examples, which is not needed here and removes a safety
net.

## 4. Cameras

The YAM checkpoint uses three RealSense views:

| Role | Camera | Placement |
| --- | --- | --- |
| `top` | RealSense D435 | Fixed overhead, covering both arms' shared workspace |
| `left_wrist` | RealSense D405 | Left wrist, looking at the grasp |
| `right_wrist` | RealSense D405 | Right wrist, looking at the grasp |

Cameras are identified by serial, so `/dev/video*` numbering does not matter.
Use USB 3 ports (a D405 on USB 2 negotiates a lower rate) and rigid mounts:

```bash
uv run python -c 'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
uv run karma cameras --rig yam_bimanual \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL --probe --snapshot /tmp/cams
```

The same three `--camera-serial` flags go on `teleop --record`, `inference`,
`rollout` and `hitl`. A single-arm rig accepts `top` and its own wrist only.
Plain USB webcams work too (`--camera ROLE=/dev/videoN`) — see
[cameras](cameras.md) — but the public YAM checkpoint was trained on the three
RealSense views above.

## 5. Teleop

Quest on USB with developer mode and USB debugging on (`adb devices` → `device`).

Both arms:

```bash
uv run karma teleop --rig yam_bimanual \
  --interface left=can0 --interface right=can1 --open-quest
```

One arm — same rig, one hand. Only that arm's bus is opened and only that Quest
controller drives:

```bash
# right arm on can0
uv run karma teleop --rig yam_bimanual --arm right --interface right=can0 --open-quest
# left arm on can0
uv run karma teleop --rig yam_bimanual --arm left --interface left=can0 --open-quest
```

In VR: hold **grip** to clutch and move the controller; release to hold the
target while repositioning your hand. **Trigger** closes the gripper. The
thumbstick click returns to the rest pose (`--rest-pose-left/right`
override the six joint angles). Ctrl+C parks at `home_pos` and de-energizes;
`--no-park` de-energizes in place, useful before home has been verified.
`--rate` is the native command/IK rate (200 Hz default). The optional passive
teaching handle is documented in [yam_teaching_handle.md](yam_teaching_handle.md).

Recording adds the cameras and a dataset id:

```bash
uv run karma teleop --record --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --repo-id local/yam-demo --task "fold the towel" --fps 30 --open-quest
```

`--arm right --camera-serial top=... --camera-serial right_wrist=...` records a
single-arm dataset (7 state values).

## 6. Policy

Start the YAM MolmoAct2 server (port 8202, norm tag `yam_dual_molmoact2`; see
[model servers](model-servers.md)), then `inference`, `rollout` or `hitl` with
the same bus and camera flags — commands in the [README](../README.md#policy-inference-rollout-hitl).
The bimanual checkpoint needs both arms and all three views; it is not a
single-arm policy.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `doctor` says a bus does not exist | Adapter not detected or not up: `ls /sys/class/net/can*`, `ip link set canN up type can bitrate 1000000` |
| Left and right swapped after reboot | Enumeration order changed: pin names with the udev rule above or pass `--interface` |
| "No YAM model configured" at teleop start | i2rt clone missing: clone into the checkout or set `YAM_XML` |
| `[WARN] visual meshes not cached` | Run `uv run karma-viz --fetch-meshes --model Yam` once |
| Arm goes limp mid-run | Motor safety timeout after a command gap: check the bus (`reset_all_can.sh`), USB cable, and that nothing else uses the bus |
| Gripper reads inverted | Native gripper is 1=open; the dataset convention is 0=open — do not "fix" one to match the other, the recorder converts |
