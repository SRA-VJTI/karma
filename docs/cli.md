# Operator CLI

`openpi-control` covers the jobs around making an arm usable: check that a setup
is sane, set each servo's firmware zero, and bring a whole rig up and back down.

| Command | Touches hardware? |
| --- | --- |
| `doctor` | no, unless given `--probe` (which only ever reads) |
| `zero` | writes servo firmware; confirms first |
| `live` | **energizes the arms**, and owns putting them down again |
| `infer` | **energizes the bimanual YAM**, executes MolmoAct2 chunks, and owns putting it down again |
| `rollout` | **energizes the bimanual YAM**, records LeRobot v3 episodes, and owns putting it down again |
| `cameras` | no, unless given `--probe` (which opens camera streams, not the bus) |
| `record` | **energizes the arms** and teleoperates them, like `live` |

Every command attaches `runlog.setup_run_logging`, so each run leaves a trace
under `~/openpi-data/logs/runtime/<command>.log`, including native crashes.

## Common arguments

`doctor` and `zero` take the same three:

| Argument | Meaning |
| --- | --- |
| `--model` | arm model, e.g. `Yam` (required) |
| `--interface` | CAN interface name, `/dev` path, or controller IPv4 (required) |
| `--effector` | effector model, e.g. `E_Yam` or `E_Yam_Handle` (optional) |

`--interface` picks the transport by its own shape: a `/dev` path is a serial
bus, an IPv4 literal is an Ethernet controller, anything else is a SocketCAN
interface name. `doctor` cross-checks that against the transport the model's
servos actually need.

Naming the effector matters for `zero`: it adds the effector's own servos to the
plan, so `--effector E_Yam` includes the gripper motor (joint 7) and
`--effector E_Yam_Handle` includes the handle encoder as a skipped read-only
entry.

`live` addresses arms by rig instead — see [Rigs](#rigs) below.

## doctor

Read-only preflight. Opens no bus unless you ask it to.

```bash
uv run openpi-control doctor --model Yam --interface can_follower_r --effector E_Yam
```

```
doctor: Yam on can_follower_r
  [OK  ] packaged assets            Yam.json, Yam_01.json, E_Yam.json
  [OK  ] urdf                       Yam.urdf
  [OK  ] servo registry             7 servos, all models known
  [OK  ] bus type                   can
  [OK  ] connection                 SocketCanConnection
  [OK  ] interface can_follower_r   present
  [OK  ] link state                 up
  [OK  ] bitrate                    1000000 (model wants 1000000)
  [OK  ] arm config                 valid
  [OK  ] visual meshes              /home/you/openpi-data/meshes/Yam
```

What each check is for:

| Check | Catches |
| --- | --- |
| packaged assets | a model or instance JSON that does not resolve |
| urdf | a model with no URDF, which gravity compensation needs |
| servo registry | a `servo_model` in the catalog with no driver — config drift |
| bus type | servos spread across two transports |
| connection | an interface string that implies the wrong transport |
| interface | the adapter is not plugged in, or is named differently |
| link state | the interface exists but is down (`ip link set <if> up`) |
| bitrate | the bus is up at the wrong speed — YAM wants 1 Mbit |
| arm config | a model/connection/effector combination `ArmConfig` rejects |
| visual meshes | the viser view will fall back to a skeleton |

Exit status is 1 if any check failed, 0 otherwise, so it drops into a script.

`--rig` checks every arm of a rig in one pass, with the same checks, and stays
just as read-only — this is how you confirm a bimanual cell is ready without
energizing anything:

```bash
uv run openpi-control doctor --rig yam_bimanual
```

```
doctor: rig yam_bimanual — two YAM followers, each with an E_Yam gripper

left (Yam on can0):
  [OK  ] packaged assets            Yam.json, Yam_01.json, E_Yam.json
  ...
right (Yam on can1):
  ...

2 arms, 0 failed, 2 warned
```

`--rig` already names every arm and its bus, so it refuses to be combined with
`--model`/`--interface`; use `--interface-override left=can2` to move one arm.

`--probe` additionally opens the bus and listens. It only ever reads, never
sends. Silence is reported as a warning, not a failure: DM servos answer
requests rather than broadcasting, so a quiet idle bus is normal.

## zero

Writes the arm's **current physical pose** into each servo as its firmware zero,
through the per-family drivers in `openpi_control.servos`.

```bash
uv run openpi-control zero --model Yam --interface can_follower_r --effector E_Yam
```

Move the arm to its intended zero pose and support it first. The command lists
what it will touch and then requires you to type `zero` — `y` and `yes` are
deliberately not accepted:

```
About to write a new firmware zero on Yam via can_follower_r:
  joint 1 (arm)            DM J4340                 servo id 1
  ...
  joint 7 (E_Yam)          DM J4310                 servo id 7

The arm's CURRENT physical pose becomes zero for every servo listed.
Move it to the intended zero pose and support it before continuing.

  joint 7 (E_Yam) is the GRIPPER. Its zero must be the FULLY CLOSED stop.
  ...
Type 'zero' to proceed:
```

> **The gripper's zero is measured, not written.** `E_Yam` ships
> `needs_calibration`, so the node finds the gripper's two mechanical stops at
> every startup and normalizes to those (see `docs/inference.md`). Zeroing
> joint 7 only shifts the frame that calibration then anchors, so it is rarely
> the thing you want; if you do it, do it with the jaws shut.

| Flag | Meaning |
| --- | --- |
| `--joint N` | zero only joint id `N` |
| `--dry-run` | print the plan and exit; opens no bus |
| `--yes` | skip the prompt (for scripted rigs) |

Behaviour worth knowing:

- **Read-only encoders are skipped, not failed.** The YAM teaching handle's
  passive encoder (`CAN Passive Encoder`, id `0x50E`) has its zero fixed in
  hardware, so `zero` reports it as skipped. Same for the `ARX_ENC` leader arm,
  where every joint is read-only and there is nothing to write at all.
- **The interface is checked before you are prompted**, so a missing adapter
  fails without putting a destructive question on screen.
- **Whole-arm controllers zero in one transaction.** Trossen's Ethernet
  controller writes every joint's zero with a single EEPROM write, so it is
  called once rather than per joint.
- Servos are zeroed in ascending joint id order.
- Exit status is 1 if any servo did not acknowledge.

## Which servos does an arm have?

`zero --dry-run` answers that without touching anything:

```bash
uv run openpi-control zero --model Yam --effector E_Yam_Handle \
  --interface can_leader_r --dry-run
```

```
dry run: would zero via can on can_leader_r
  joint 1 (arm)            DM J4340                 id 1      zero
  ...
  joint 7 (E_Yam_Handle)   CAN Passive Encoder      id 1294   read-only, skipped
```

## Rigs

A rig names a whole cell — which arms it has, the bus each sits on, and where
their bases sit relative to each other — so `live`, `doctor --rig`, and the
visualizer all mean the same thing by `left`. They live in
`openpi_control.rigs` and are pure configuration: resolving one opens no bus.

```bash
uv run openpi-control live --list          # the rig and its arms, then exit
uv run openpi-control-viz --list-rigs      # same rigs, from the viz side
```

| Rig | Arms | Cameras |
| --- | --- | --- |
| `yam_bimanual` | `left` (Yam + E_Yam, can0), `right` (Yam + E_Yam, can1) | `top`, `left_wrist`, `right_wrist` (D405s) |

Two flags adapt a packaged rig to the cell in front of you, on any command that
takes a rig:

| Flag | Meaning |
| --- | --- |
| `--interface ARM=IFACE` | move one arm to another bus, e.g. `left=can2` |
| `--only ARM` | bring up just one arm of the rig (repeatable) |
| `--camera NAME=DEVICE` | pin one camera to an explicit device, e.g. `top=/dev/video4` |

`--only` narrows cameras along with arms: a wrist camera names the arm it rides
on, so `--only right` leaves you with `top` and `right_wrist`.

An override naming an arm the rig does not have is an error rather than a
no-op — a typo that silently left the arm on its default bus is the one failure
here worth being loud about. (On `doctor` the flag is spelled
`--interface-override`, because `--interface` there already means a single arm's
bus.)

## infer

`infer` is the hardware-side MolmoAct2 client for `yam_bimanual`. It requires
both YAM followers and all three trained camera views. Start the GPU policy
server separately, then run:

```bash
uv sync
uv run openpi-control infer \
    --server http://192.168.0.107:4090 \
    --instruction "pick up the object"
```

The client sends `top`, `left_wrist`, and `right_wrist` as the model's
`top_cam`, `left_cam`, and `right_cam`, with a 14-value left/right state. The
every action of every returned chunk is executed, one bounded command per
control tick. Viser shows measured arm poses, translucent predicted
end-effector trails, and a "Policy input" panel holding the three frames the
model was actually handed, at capture resolution.

The wire defaults match the reference deployment; the three flags that change
how the chunk is *executed* are opt-in, because matching the reference there
behaved worse on this rig. See [inference.md](inference.md).

| Flag | Meaning |
| --- | --- |
| `--server` | MolmoAct server URL or host:port; `/act` is appended |
| `--instruction` | language instruction sent with every observation (required) |
| `--request-timeout` | HTTP timeout in seconds (default 60) |
| `--num-steps` | model denoising steps requested from the server (default 10) |
| `--no-cuda-graph` | stop requesting CUDA graph inference (~20x slower) |
| `--raw-frames` / `--jpeg-quality` | frame transport; JPEG q95 by default |
| `--reach-actions` | walk to every action in sub-steps (the reference runtime) |
| `--prefetch` | infer the next chunk while this one is still executing |
| `--prefetch-margin-s` | margin added to measured latency when prefetching |
| `--reset-start-pose` | ramp to the training start pose before the first chunk |
| `--control-rate` | nominal action rate in Hz, used to time the prefetch |
| `--max-step-rad` | maximum joint movement per commanded step (default 0.10) |
| `--max-effector-step` | maximum normalized gripper movement per commanded step |
| `--no-park` | de-energize in place instead of parking at `home_pos` |
| `--no-viz` | run the client without starting Viser |
| `--camera NAME=DEVICE` | pin a camera to an explicit device |
| `--interface ARM=IFACE` | move an arm to another CAN interface |
| `--skip-preflight` | energize without doctor checks |

Inference and stale-state failures stop commanding before the normal park and
de-energize sequence runs. See [inference.md](inference.md) for full wire
ordering and setup.

## rollout

`rollout` records timed, interactive MolmoAct2 policy attempts as LeRobot v3
episodes. It asks for a new language prompt before every attempt, waits for
the same towel to be reset, and asks for a `y/n` success label only after the
arms have safely parked. Ctrl-C interrupts the current attempt, saves all
frames captured so far as a partial episode, parks at `home_pos`, and continues
to the next prompt.

```bash
uv sync --extra lerobot
uv run openpi-control rollout \
    --repo-id Dimios45/openpi-fold-towel-rollout-ablation \
    --root ~/openpi-data/rollouts/fold-towel \
    --episodes 3 --episode-seconds 120 \
    --server http://192.168.0.107:4090 \
    --interface left=can_left --interface right=can_right \
    --speed 0.5 --port 8080
```

The dataset directory also contains `openpi_control_rollouts.json`, which
keeps the per-attempt prompt, saved episode index, interruption flag, and
success label. The prompt is additionally written to every frame's LeRobot
`task` field. Viser is available at `http://<robot-box>:8080` during the run.

| Flag | Meaning |
| --- | --- |
| `--episodes` | number of attempts (default 3) |
| `--episode-seconds` | maximum duration per attempt (default 120) |
| `--speed` | action playback speed (default 0.5) |
| `--chunk-size` | use a prefix of each returned action chunk, 1–30 |
| `--no-prefetch` | infer synchronously between chunks |
| `--no-reset-pause` | skip the manual reset prompt between attempts |
| `--no-viz` | disable Viser |
| `--interface ARM=IFACE` | override an arm's CAN interface |
| `--camera NAME=DEVICE` | pin a camera to a device |
| `--skip-preflight` | energize without doctor checks |

## live

Energizes a rig, mirrors it in the browser, and parks it on the way out.

```bash
uv run openpi-control live --rig yam_bimanual
```

```
live: rig yam_bimanual — two YAM followers, each with an E_Yam gripper
  left     Yam    can0     E_Yam    follower
  right    Yam    can1     E_Yam    follower

preflight left: all checks pass
preflight right: all checks pass

  left     Yam on can0 — energized, holding
  right    Yam on can1 — energized, holding
  cameras  top, left_wrist, right_wrist — live in the browser
  viser    http://localhost:8080
ctrl-c to park at home_pos and power down
```

| Flag | Meaning |
| --- | --- |
| `--rig` | which rig (default `yam_bimanual`) |
| `--only ARM` | bring up one arm of the rig instead of all of them |
| `--interface ARM=IFACE` | move one arm to another bus |
| `--float` | gravity float instead of holding: the arms become backdrivable |
| `--no-park` | de-energize where the arm stands instead of parking it first |
| `--no-viz` | energize and hold without serving the browser view |
| `--control` | add the per-arm browser control panel (see below) |
| `--no-cameras` | skip the camera tiles, leaving the cameras free for another process |
| `--camera NAME=DEVICE` | pin one camera to an explicit device |
| `--port` | viser HTTP port (default 8080) |
| `--mesh-dir` | directory holding the URDF's meshes |
| `--skip-preflight` | energize without running the doctor checks first |
| `--list` | describe the rig and exit, touching nothing |

### The lifecycle is the command

`pi_control_node` holds a liveness pipe to the Python process that spawned it,
so it exits when that process does. An arm therefore **cannot stay energized
after the command that powered it on returns** — there is no `up` to run now and
`down` to run later. One foreground process owns the whole arc, and ctrl-c is
the way out.

### Powering on

Connecting *is* the power-on: the native node comes up holding whatever pose it
found, so both followers go stiff. `--float` swaps that for the gravity
feed-forward, leaving the arms backdrivable so you can pose them by hand and
watch the browser follow. That stays opt-in because a compliant arm sags if its
`torq_rescale` is untuned.

Preflight runs first, and **nothing is energized unless every arm passes**. A
rig brought half up is the bad case: the arm that did connect is stiff, and the
session that could have put it down is already unwinding.

### Powering off

ctrl-c parks each arm at the `home_pos` in its instance JSON and only then cuts
torque, riding the native `MOVE_TO_READY_AND_SHUTDOWN` path — so the node exits
at the end of that move and the arm is never dropped from wherever it happened
to be standing.

- **`--no-park` drops the arm where it stands.** Support it first.
- A node that does not advertise `CAP_MOVE_TO_READY` is closed in place, and
  says so rather than pretending it parked.
- A park that fails is reported, and the remaining arms are still put down —
  one bad park must not abandon an energized arm.
- Exit status is 1 if any arm failed to park or close.

### What the browser shows

The scene is driven by the hardware. Poses are pushed at 30 Hz from each arm's
newest published state; an arm that goes briefly quiet keeps its last pose on
screen rather than tearing down a session that is holding two energized arms.
Without `--control` the page is a view and nothing more — no GUI sliders, so
nothing on it can move an arm.

The rig's cameras appear alongside it, one tile per camera, under **Cameras**
— so "is the right wrist actually pointing at the thing" is answered on the
same page that shows the pose, without a headset and without a second tool.
See [Cameras in the browser](#cameras-in-the-browser) below.

`--no-viz` skips the browser entirely, for when you just want both arms up and
holding.

### Cameras in the browser

Tiles are previews, and priced like previews: each frame is subsampled to 400 px
wide and pushed at 10 Hz, not the 30 Hz the poses go out at. Three 848x480
streams pushed whole on the mirror clock would be ~35 MB/s of websocket to
answer a question a thumbnail answers just as well. They ride the same clock as
the poses rather than a thread of their own, so images and poses share one
socket instead of racing for it.

What a tile does *not* do is hide a problem:

- A camera that is not on the bus is named on stdout and gets no tile. `live`
  exists to drive arms — an unplugged wrist camera costs you a tile, not the
  session. (`record` is the opposite: there, all-or-none, because a dataset
  with a view silently missing is a corrupt dataset.)
- A camera another process is already streaming is named the same way. **Only
  one process can hold a camera**, so `live` and `record` cannot both preview
  the same cell, and neither can two `live` sessions.
- A camera that stops delivering holds its last image rather than re-encoding
  it ten times a second.

`--no-cameras` skips them, which is how you leave the cameras free for a
recorder or a policy while still driving the arms from the browser.

Cameras follow `--only` the way they do everywhere else: `--only right` brings
up one arm with the overhead and right-wrist views, and drops the left wrist
camera along with the arm it rides on.

### Driving the arms from the browser

`--control` adds a control panel per arm, on the one page, in the one world
frame the rig places them in:

```bash
uv run openpi-control live --rig yam_bimanual --control
```

```
  left     Yam on can0 — energized, holding
  right    Yam on can1 — energized, holding
  control  left, right — disarmed; arm each one in the browser
  viser    http://localhost:8080
```

Sliders and a live hardware feed fight over the pose only if you let them both
own it. Here they do not:

- **The render always follows the hardware**, armed or not. The screen is a
  measurement, never a wish.
- **The sliders are the target.** While an arm is disarmed they are slaved to
  its measured pose, so they can never fall behind an arm someone pushed by
  hand; while it is armed you own them and nothing writes back.

#### Arming

Each arm has its own gate, so you can drive one side of a bimanual cell while
the other just holds. Tick **Pose on screen matches the arm**, then **Arm**.

The checkbox is the part no machine can do — look at the actual arm. Everything
it would otherwise be asserting is checked before the arm engages, and a failed
check refuses with the reason on the page rather than arming anyway:

| Refused when | Why it matters |
| --- | --- |
| no state has arrived yet | there is no measured pose to start from |
| the state is stale | commanding an arm you cannot currently see |
| the render is >0.05 rad off the measured pose | the screen is not the arm; arming would jump it |
| the arm is outside its URDF joint limits | the render clamps, so the first command would snap it back |
| a configured effector publishes no state | the gripper would be commanded to a guess |
| the node takes no direct commands | nothing to drive |

Arming calls `hold()` first, at the pose the arm is already standing in — a
no-op move, and the thing that re-engages position control after `--float`.

#### While armed

Targets step toward the sliders at **0.5 rad/s per joint**, so a full-throw
drag is a move you can watch and interrupt rather than a single-tick lurch.
Commands are clamped to the URDF joint limits. The gripper slider is normalized
to [0, 1] — the node owns what that means in millimetres for this effector.

#### Disarming

**Disarm** on an arm, **Disarm all** in the Safety folder at the top, or any of
the automatic trips:

- the arm's state goes stale or stops arriving,
- the node rejects a command,
- **the browser disconnects** — an unattended tab must not leave an arm
  tracking a target nobody has hold of.

A disarmed arm goes back to its resting mode: holding, or gravity float if the
session was started with `--float`. Disarming also clears the confirmation
checkbox, so re-arming after a trip means looking at the arm again rather than
clicking one button. ctrl-c disarms everything before the park runs, so a
panel can never be pushing targets at an arm that is parking.

#### Not there yet

- **No IK.** i2rt's interface can drag a 6-DOF gizmo on the end effector; this
  is per-joint only. Adding it needs a solver in the tree (`pyroki` or
  mujoco + `mink`) — a real dependency decision, not an afternoon.
- **No self-collision check.** i2rt forward-simulates the candidate pose in a
  scratch MuJoCo state and blocks the command on a link-link penetration. The
  rate limit means you can stop a bad move here, but nothing refuses it for you
  — the two arms of a bimanual cell can be driven into each other.
- **No target ghost.** The screen shows where the arm *is*; the difference
  between that and where the sliders are pointing is not drawn.

## cameras

Resolves the rig's cameras to device paths and says which ones are on the bus.
Read-only; `--probe` additionally opens each stream and measures it.

```bash
uv run openpi-control cameras                             # discovery only
uv run openpi-control cameras --probe --snapshot /tmp/c   # open, measure, save a frame
uv run openpi-control cameras --only right                # top + right wrist
```

Discovery reads udev symlink names and nothing else, so it works without the
`cameras` extra installed — that is what lets `doctor --rig` report an unplugged
camera on a machine that cannot open one. `--probe` needs the extra.

A missing camera is a warning, not a failure: no camera is needed to drive an
arm. Exit status is 1 only if a probe actually failed.

Cameras are pinned by serial number rather than by `/dev` path, and the serial in
the rig is the *ASIC* serial — the one in `/dev/v4l/by-id` — not the different
number `pyrealsense2` reports. See [docs/cameras.md](cameras.md) for that
distinction, for why the default mode is 848x480, and for why capture goes
through the RealSense SDK instead of OpenCV.

## record

Teleoperates a rig from a Quest headset and writes the episodes as a LeRobot
dataset. Energizes the arms and owns putting them down again, exactly like
`live`.

```bash
uv sync --extra lerobot   # torch, on top of the default set
vr-teleop-relay                                    # in the vr-teleop-kit checkout
uv run openpi-control record --repo-id you/task \
    --task "fold the towel" --num-episodes 20 --vr-kit ~/vr-teleop-kit
```

Right B starts (or redoes) an episode; left Y saves it. No headset to hand?
`--teleop hold --dry-run` runs the whole session with the arms stationary, which
checks the cameras, the schema, and the loop rate without collecting data.

Preflight is stricter than `doctor`'s: a declared camera that is not on the bus
is fatal, because recording with a view silently absent yields a dataset that is
wrong rather than a cell that is merely unchecked.

Two details that will cost you data if you do not know them — the gripper
polarity is **inverted** between this package (1.0 open) and LeRobot (0.0 open),
and ctrl-c during an open episode discards it. Both, plus the teardown ordering
and the stall handling, are in [docs/recording.md](recording.md).
