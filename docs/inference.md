# MolmoAct2 integration

Karma is the robot-side client. A MolmoAct2 model server runs on a GPU host;
Karma POSTs observations to `http://HOST:PORT/act` and executes the action chunk
it gets back. Checkpoint training and loading stay in the model environment —
see [Model servers](model-servers.md) for what each server exposes.

## Contract per rig

| Rig | State/action width | Joint order per arm | Views | Gripper on the wire |
| --- | --- | --- | --- | --- |
| `yam_bimanual` | 14 | 6 joint radians, gripper (left then right) | `top`, `left_wrist`, `right_wrist` | 1=open |
| `so101` | 6 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper | `top`; `side` and `right_wrist` optional | 0=open |
| `so101_bimanual` | 12 | same SO101 block, left then right | `top`; wrists optional | 0=open |

Karma's wire is **absolute calibrated radians** (mid-travel zero for SO101,
firmware zero for YAM) and absolute joint targets, not deltas. The SO101 wire
uses the dataset gripper convention (0=open); the original YAM HTTP contract
keeps 1=open and converts in its server adapter. Native hardware always reports
1=open. Karma never pads missing joints or duplicates a missing camera to satisfy
a checkpoint from another rig.

## Health check

Before energizing, the client `GET`s the `/act` URL and validates the reply:

```json
{"status": "ok", "norm_tag": "so100_so101_molmoact2", "state_dim": 6, "num_cameras": 2}
```

`norm_tag` must equal `--norm-tag`, `state_dim` the rig's width and
`num_cameras` the number of declared views. A server that omits the fields is
accepted only for the default YAM contract. SO101 always requires `--norm-tag`
because there is no universal SO101 key; take it from the checkpoint's
`norm_stats.json`.

## Request and response

Each POST carries:

| Field | Content |
| --- | --- |
| `state` | float32 vector of the rig's width, in the policy frame (see below) |
| `instruction` | the task string |
| `top_cam`, `left_cam`, `right_cam`, `side_cam` | one per declared role: JPEG bytes as a `json_numpy` array by default, or an H×W×3 RGB array with `--raw-frames` |
| `timestamp`, `num_steps`, `normalization_tag`, `enable_cuda_graph` | bookkeeping and solver settings |

The response is `{"actions": <N × width array>}` in the same codec, optionally
with `dt_ms`. Empty, wrong-width or non-finite chunks are rejected.

Execution is bounded: each control tick (30 Hz by default) moves at most
`--max-step-rad` per joint towards the current action, within the calibrated
joint limits. `--speed` stretches a chunk over more ticks (0.5 = half the
training rate). The next chunk is requested while the current one plays
(`--no-prefetch` disables that).

## Policy frames

A checkpoint trained on Karma's own recordings speaks Karma's wire directly.
A checkpoint trained on data from another stack does not, and units are only
part of it: the zero pose and joint directions differ too. `--policy-frame FILE`
applies a per-joint affine map at the HTTP boundary only:

```
policy = wire * scale + offset      # state, before each request
wire   = (policy - offset) / scale  # actions, before execution
```

Bounds, clamps, datasets and everything else stay in Karma's radians. A
six-entry frame is tiled across both arms of `so101_bimanual`.

### The public SO100/SO101 checkpoint (work in progress)

`allenai/MolmoAct2-SO100_101` (norm tag `so100_so101_molmoact2`) was trained on
LeRobot **v1** community datasets. Its `norm_stats.json` shows the frame: joints
in degrees with the rest pose near `[0, 189, 181, 61, -4]`, gripper 0 (closed)
to about 45 (open), i.e. LeRobot v1's calibration:

- **zero position** — arm straight out horizontally, gripper up and closed: every joint 0°;
- **rotated position** — every joint a quarter turn from zero: +90°;
- gripper `LINEAR`: 0 at the zero pose, 100 at the rotated (open) pose.

Because both Karma's radians and LeRobot's degrees are affine in raw encoder
steps, two captured poses plus the calibration profile determine the map
exactly. `karma so101-policy-frame` disables torque, asks for the two poses
(reference photos: `media/so100/follower_zero.webp` and `follower_rotated.webp`
in the LeRobot repository at commit `42bf1e8`), and writes the frame:

```bash
uv run karma so101-policy-frame --interface /dev/ttyACM0 \
  --calibration calibration/so101-right.json \
  --output calibration/so101-right.policy-frame.json
```

**Status.** The maths is tested against LeRobot's own calibration formula for
random poses, both gripper directions and flipped joint directions. What is
not yet robust is the capture itself: the result is only as good as the two
hand-held poses, and a joint held a quarter turn off in one pose lands that
joint's offset ~90° off. On the reference arm the wrist-roll offset had to be
corrected by hand so that the physical rest pose read the checkpoint's rest
state. Planned: derive the frame from a LeRobot calibration file instead of
posing (LeRobot v2 files target a different, mid-travel-zero frame and would
need translating through the firmware homing offsets as well), and a one-joint-
at-a-time capture. Until then, **verify before energizing**: map the resting
arm through the frame and compare with the checkpoint's rest state above;
wrist roll outside the checkpoint's −66…+43° range means the model will pin
that joint at its clip.

## Running it

```bash
uv run karma inference --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --policy-frame calibration/so101-right.policy-frame.json \
  --camera top=/dev/video5 --camera side=/dev/video7 \
  --server http://GPU_HOST:8203 --norm-tag so100_so101_molmoact2 \
  --speed 0.5 --instruction "pick up the object"
```

`rollout` and `hitl` take the same policy flags and ask for the task per
episode. The legacy YAM `--reach-actions` and demonstration start-pose ramp do
not apply to SO101. Keep the control cadence, camera framing, units and
normalization used in training. The tests exercise both arm layouts with
simulated states and HTTP responses; they do not establish that a checkpoint
performs a physical task.
