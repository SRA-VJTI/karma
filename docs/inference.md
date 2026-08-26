# MolmoAct2 inference

The `infer` command runs the robot-side half of the MolmoAct2 bimanual YAM
loop. It owns the two YAM followers, the three RealSense cameras, the Viser
page, and action-chunk execution. The GPU model stays in the separate HTTP
server from the MolmoAct2 example.

The **wire protocol and its defaults** are the reference deployment's, down to
the CUDA-graph request and the JPEG frame transport;
`openpi_control/inference.py` names the measurement behind each one.

How the chunk is **executed** is not, by default. The reference deployment
walks to every action as fast as its sub-steps allow, which was tried here and
behaved worse — this package drives the arms through the native node rather
than i2rt's direct CAN writes. The default is one bounded command per control
tick. `--reach-actions` and `--reset-start-pose` turn each piece of the
reference runtime back on, one at a time, so the two can be A/B'd on hardware.
Prefetching is the one piece that is on by default — it changes *when* the next
call is made, not what the policy is asked to do or how the chunk is executed.

## Install and start

On the robot computer:

```bash
uv sync
```

On the GPU computer, start the reference server — with CUDA graphs, which is
not optional in any practical sense (see below):

```bash
uv run python host_server_yam.py --host 0.0.0.0 --port 4090 --cuda-graph
```

Then run on the robot computer:

```bash
uv run openpi-control infer \
    --server http://192.168.0.107:4090 \
    --instruction "pick up the object"
```

The server URL may be a full URL, an address such as
`192.168.0.107:4090`, or a hostname. `/act` is appended automatically.

Use a wired link if there is one. The payload is ~0.2 MB per request over
ethernet; the same session over Wi-Fi spends more time in transport than on the
GPU.

## Runtime behavior

The command requires the packaged `yam_bimanual` rig: left and right YAM
followers plus the `top`, `left_wrist`, and `right_wrist` cameras. Observations
are sent in the checkpoint's order:

```text
top_cam, left_cam, right_cam
state = [left joint 1..6, left gripper,
         right joint 1..6, right gripper]
```

The server returns an absolute action chunk with one seven-value action per
arm. The native gripper convention is used directly: `1.0` means open.

A run is, in order:

1. **Health check.** `GET /act` before anything is energized. The payload's
   `norm_tag`, `state_dim`, and `num_cameras` are checked, not just its status
   — a DROID server answers the same route and differs only in those fields.
2. **Chunks, in full.** Every action the server returns is executed. The
   checkpoint plans 30 (`max_action_horizon`) and expects all of it to run, and
   a chunk of any other length is called out in the log.
3. **Bounded execution.** One command per control tick, each within
   `--max-step-rad` of the previously commanded target.

Viser keeps the measured hardware pose as the solid arm and draws the active
prediction as translucent end-effector trails. The trail is built once per
inference response and anchored at the pose that response was predicted from; a
marker slides along it to show how far into the chunk the arms are. The whole
trail stays up until the next response replaces it, and disappears during
teardown.

Building it is ~60 ms of forward kinematics and mesh uploads, which is why it
happens on the chunk boundary (just after a round trip that already cost
hundreds of ms) and never on a control tick. Advancing the marker costs
~0.03 ms.

Two variants of this were tried on hardware and are worth not repeating:

* **Rebuilding the overlay per tick** — 60 mesh removes, 62 forward-kinematics
  evaluations and 60 mesh adds — costs 61 ms against the 33 ms tick budget at
  the default 30 Hz. The loop could not hold its period, so the arms were
  commanded at roughly half the intended rate, the prefetch fired against a
  period the loop was not achieving, and the overlay strobed as it was removed
  and re-added.
* **Retiring each segment as its action executed** drains the trail to nothing
  over a chunk — 60 segments to 0 in 1.0 s at 30 Hz — and leaves the scene
  empty through every inference round trip. On screen that reads as no overlay
  at all.

So the trail is immutable for the life of a chunk and progress is a transform
update. Anything that needs to change mid-chunk needs to be that cheap.

The scene includes the calibrated top-camera frame from the bimanual midpoint
calibration; that is visualization metadata only, and wrist-camera poses stay
absent until calibrated.

### The "Policy input" panel

The camera tiles on the `infer` page are not a live preview. They are the three
frames the policy was handed, decoded back from the wire, at capture resolution
(848x480 on this cell) — so what is on the page is the model's own picture,
compression artifacts included, not a picture taken beside it. They update once
per inference rather than on a preview clock, which is why they hold still
between chunks: that is one observation, held for as long as the policy is
acting on it.

Measured on this cell at q95, the wire form is 11-12x smaller than the raw
frame with a mean per-pixel difference of ~1.3/255, so the tiles are
effectively what the camera saw. With `--raw-frames` the tiles are served
lossless, because the policy's input is then lossless too.

For reference, the reference deployment's own comments describe 360x640 frames,
so this cell feeds the model ~1.8x the pixels it does. The processor resizes
internally and the aspect ratio is the same 16:9 either way, but it is worth
knowing when comparing payload sizes or latency against theirs. Capture
resolution is a rig-level default (`openpi_control/cameras.py`); there is no
`infer` flag for it.

## Recording timed policy rollouts

Use `rollout` when the goal is a dataset of policy attempts rather than an
open-ended controller run. LeRobot is optional because it brings in torch:

```bash
uv sync
uv run openpi-control rollout \
    --repo-id Dimios45/openpi-fold-towel-rollout-ablation \
    --root ~/openpi-data/rollouts/fold-towel \
    --episodes 3 \
    --episode-seconds 120 \
    --server http://192.168.0.107:4090 \
    --interface left=can_left \
    --interface right=can_right \
    --speed 0.5 \
    --port 8080
```

The command starts Viser on port 8080, opens the three policy cameras, and
parks both arms at `home_pos` after every attempt. Before each attempt it asks:

```text
Reset the same towel to its starting pose, then press Enter to continue:
Prompt for episode 1/3:
```

The reset prompt is skipped for episode 1. The language prompt is always asked
again, so every recorded episode can have its own instruction. At the duration
limit the episode is saved and the arms park. Ctrl-C during an episode saves
the frames captured so far as a partial episode, then parks; it does not delete
those frames or terminate the whole multi-episode run. After parking, the
terminal asks `Episode N successful? [y/n]:` for both complete and partial
attempts. A second Ctrl-C at the prompt exits the run.

Each saved frame has the prompt in LeRobot's `task` field. The `--root`
directory also contains `openpi_control_rollouts.json`, a small sidecar with
the attempt number, prompt, saved dataset episode index, `success` label, and
whether the attempt was interrupted. This is where the y/n result lives; it is
not a model input feature and does not alter the LeRobot v3 schema.

Useful rollout controls:

| Flag | Meaning |
| --- | --- |
| `--episodes` | number of episode attempts (default 3) |
| `--episode-seconds` | maximum time per attempt (default 120) |
| `--speed` | action-chunk playback speed; default 0.5 |
| `--chunk-size` | execute only the first 1–30 actions returned per policy call; default full chunk |
| `--no-prefetch` | remove overlap between inference and chunk execution |
| `--no-viz` | record without starting Viser |
| `--no-reset-pause` | do not wait for the manual towel reset between attempts |
| `--interface ARM=IFACE` | map the packaged rig to this cell's CAN names |

For this cell the persistent SocketCAN names are `can_left` and `can_right`,
so keep the two `--interface` overrides unless the packaged rig defaults have
been changed locally. Open Viser at `http://<robot-box>:8080` while the rollout
is running; the page shows measured arms, the current predicted chunk, and the
policy-input camera frames.

Each chunk logs its size, the round trip split into GPU and transport time, the
payload size, how many actions the safety clamp has filed down, how far behind
its targets the hardware ran (`lag`, rad), and the gripper as
`grip l<commanded>/<measured> r<commanded>/<measured>`. A rising clamp count
means the policy is asking for jumps the limit is refusing; a large `lag`
beside a low clamp count means the opposite — the limit let the plan through
and the arms did not keep up. See *When the arms are too reactive* below.

Inference errors and stale hardware state stop the loop, then the normal
shutdown path parks at each arm's `home_pos` and de-energizes the native nodes.
Ctrl-C has the same shutdown behavior.

## The gripper

`E_Yam` ships `needs_calibration`, so the node measures the gripper's two
mechanical stops at every startup and normalizes `[0, 1]` to what it finds,
rather than to a configured stroke.

It has to. The DM4310 reports a **single-turn** angle (period 6.283 rad) and
i2rt measures this gripper's stroke at **6.57 rad** — longer than one turn — so
the open and closed ends alias onto nearly the same reading and no configured
range can tell them apart. `Servo::initialize_position_wrap()` resolved the
turn once, from wherever the jaws happened to be at startup: boot closed and
the guess was right, boot open and the node believed the gripper was shut for
the whole session, published `~0.002` to the policy, and mapped every command
in `[0, 1]` into the stop the jaws were already resting against. The arms
tracked normally and only the gripper was inert, which is what made it hard to
see. Widening the configured range is not an option either — `pi_servo.cpp`
refuses to start unless `raw_range < position_wrap_period`.

So the servo now accumulates turns continuously as they happen
(`Servo::accumulate_position`, i2rt's `dm_driver.py` model), and
`DeviceEffector::calibrate_gripper_limits` nudges the jaws into each stop with
a 0.2 Nm probe and takes those two readings as normalized 0.0 and 1.0 — i2rt's
`detect_gripper_limits`, which is why `linear_4310.yml` ships
`gripper_limits: null, needs_calibration: true`. Which stop is *closed* stays
`open_at_min`: a fact about how the gripper is built, not about this boot.

The startup log names what it found:

```text
E_Yam_01: gripper calibrated: closed=0.004 rad, open=6.558 rad, stroke=6.554 rad
```

A probe that cannot reach both stops (jaws blocked, or something between them)
measures too short a stroke, says so, and keeps the configured range rather
than installing a bad one.

### Reading it during a run

Nothing on the Viser page shows the gripper's *pose*: the packaged YAM URDF has
six actuated joints and the jaws are baked into `link_6`'s mesh, so the render
draws the same gripper at every position — and it reads as closed. The page
carries a **Gripper** panel with the two numbers that do mean something
(commanded and measured, normalized, `1.0` open), the run prints both grippers
before it starts, and each chunk line carries
`grip l<commanded>/<measured> r<commanded>/<measured>`.

A gripper whose measured value stays frozen while the commanded one moves is
called out once per arm as `NOT TRACKING`.

The commanded gripper carries across chunk boundaries rather than re-seeding
from the measurement. A gripper holding an object always reads short of what it
was commanded — that is what holding looks like — so re-seeding would hand back
a slice of the grip on every chunk and re-close it on the ticks that follow.
The joints do re-seed from measurement, every chunk, unless `--carry-targets`
says otherwise.

## When the arms are too reactive

A run that lunges at things, arrives past them, or changes its mind once a
second is not being caused by the safety clamp. The clamp only ever *removes*
motion. What it can do — and this is the failure that looks like rushing — is
file each action down and then move on to the next one without ever
re-attempting the one it left short, so a chunk whose per-action steps exceed
`--max-step-rad` is executed as a shrunken copy of itself. The arm starts the
reach and abandons it, which on hardware reads as *it went for it and missed*.
Tightening the clamp makes that worse, not better.

There are four separate things that can be wrong, and the chunk log tells them
apart:

| What you see | What the log says | The lever |
| --- | --- | --- |
| The arm starts a reach and stops short of it | `clamped` high | Raise `--max-step-rad`, or lower `--speed` so the plan asks for less per tick |
| The arm is smooth but always trailing the plan | `lag` large, `clamped` low | `--speed` below 1.0 — the hardware cannot follow at 1x and no clamp setting changes that |
| A jerk once per chunk, roughly once a second | neither, it is a boundary | `--carry-targets` |
| It keeps re-deciding and never finishes an approach | neither | `--speed` below 1.0 (a longer chunk is re-planned less often), or `--no-prefetch` for a fresher observation |

### `--speed`

`--speed 0.5` plays each chunk over twice as many control ticks, linearly
interpolated between the policy's own actions. The path, its shape, and its
endpoint are untouched — only the time spent on it changes. That is what makes
it different from a tighter clamp, which shortens the path, and from a lower
`--control-rate`, which slows the arms *and* the rate commands reach the node.

It also holds the policy back, which is the second half of what it is for. A
30-action chunk at 30 Hz lasts 1.0 s; at `--speed 0.5` it lasts 2.0 s, so the
policy re-plans half as often and an observation gets acted on to its
conclusion instead of being overtaken by the next one.

What it costs is freshness, the same currency prefetching spends: the
observation behind a chunk is older by the end of a chunk that lasts twice as
long. That is the balance. Slow enough that the arms arrive where the plan
says, and no slower.

### `--carry-targets`

By default the joint targets are pulled back to the measured pose at the start
of every chunk. An arm running behind its target therefore has the target
re-seeded to the lag and then re-accelerates: a sawtooth in commanded velocity
at every chunk boundary, once a second, which is the jerk an operator feels.
`--carry-targets` keeps the executor's own integrator instead, so the command
stream is continuous across the boundary — the same argument the gripper
already wins, for the same reason.

It is off by default because re-seeding is also what bounds how far the
commanded target can march past an arm that has stopped against something. With
`--carry-targets` the command keeps walking toward the plan while the arm is
held; the clamp still bounds the rate, but not the total.

### Finding the balance

One flag at a time, and read `clamped` and `lag` rather than watching the arms
— every combination looks plausible from across the room.

1. Run the default and note the two numbers.
2. If `clamped` is more than a few per cent of the ticks, that is the plan
   being refused: raise `--max-step-rad` before anything else, and remember
   that the count counts the gripper too (raise `--max-effector-step` and see
   whether it falls).
3. With `clamped` low, `lag` is the answer to "is the plan being executed".
   Halve `--speed` and it should fall roughly with it. If it does not, the lag
   is not a speed problem — check the arm is not fighting gravity or an
   obstruction.
4. Add `--carry-targets` last, and only for the boundary jerk.

## The reference runtime, opt-in

Each of these is one-to-one with the reference deployment and off by default,
because switching them all on at once made the arms behave worse on this rig.
Turn them on one at a time.

| Flag | What it does |
| --- | --- |
| `--reach-actions` | Clamp each action against the *measured* pose (`0.3` rad there, so pass `--max-step-rad 0.3` too) and walk to it in ~0.01 rad sub-steps 1 ms apart, so the arm arrives at every action instead of approaching it. |
| `--reset-start-pose` | Ramp both arms to the pose the training demonstrations begin from before the first observation. |

`--no-prefetch` goes the other way: it turns off the one piece of the reference
runtime that *is* on by default. See below for why.

### Why prefetching is on

Inferring between chunks means the arms hold their last commanded target for a
whole round trip at every chunk boundary. On this rig that is 0.25 s against a
1.0 s chunk — a quarter of the run standing still, and a hitch in the motion
once a second that is plainly visible on the hardware. Prefetching fires the
next request once the motion still queued is shorter than the measured latency
plus `--prefetch-margin-s`, so the chunk lands as the current one drains and
the arms never stop.

Only one request is ever in flight: the server serializes `predict_action`
behind a lock, so a second concurrent call would queue rather than overlap. The
win is hiding latency behind motion, not parallelism.

What it costs is freshness — the observation the policy acts on is one round
trip old. That is the trade `--prefetch-margin-s` tunes: too small and the
queue runs dry before the chunk lands (the stall comes back), too large and the
policy is handed a staler picture than it needs. `--no-prefetch` removes the
staleness and pays the stall, which is the right way to tell the two apart when
a run looks like it is reacting late.

### Why the start pose matters

The reference deployment begins from the median first frame of the training episodes, gripper
open. The tempting alternative — the checkpoint's own `norm_stats`
`state_stats.q50` — is the median over *every* frame of every episode, which is
a mid-task pose with the cloth already in hand. Starting there hands the policy
a state saying "arms raised, mid-fold" while the cameras show an untouched
table, a combination that never occurs in training. The policy then runs on
proprioception alone and mills aimlessly.

### Why CUDA graphs are on by default

Graph capture in the action expert took inference from 9–16 s to ~0.5 s with
real 360x640 frames. The flag is sent on every request rather than left to the
server's launch options, so the fast path does not depend on how the operator
started the server — and so a client cannot silently *override* a server that
was launched with `--cuda-graph`. `--no-cuda-graph` exists to reproduce the
slow baseline; expect requests near the timeout if you use it.

### Why frames go out as JPEG

Three raw 360x640 frames are ~2.8 MB of base64 per request, which dominates the
round trip on anything short of gigabit ethernet. JPEG q95 cuts that ~25x and
measured the same wall-clock as q85, so fidelity is kept. `json_numpy` base64s
any array, so the encoded frame rides the same payload key as a 1-D `uint8`
array with no format negotiation. `--raw-frames` sends raw frames, for a server
too old to decode them or to rule out compression as a variable.

A server too old to decode them is recognised by what its `_to_pil` says, and
the two generations are one clause apart:

```
raw-only:  image must be HxWx3, got shape (271814,)
current:   image must be HxWx3 (raw) or 1-D uint8 (encoded), got shape (360, 640, 4)
```

Only the second advertises the encoded form, so only its *absence* drops the
run to raw frames. Matching `HxWx3` alone reads a current server's complaint
about a genuinely malformed frame — RGBA after a camera reconfigure, a 2-D
grayscale one — as an out-of-date server, which costs the rest of the episode
3.7x the latency for a fault raw frames share, and points the operator at the
wrong box. When the message matches neither, the server's own words are
reported rather than a guess.

Fixing a raw-only server means adding the encoded-frame branch to its `_to_pil`
in place, keeping the `1-D uint8 (encoded)` clause in the message. Do not copy
`~/molmoact2/examples/yam/host_server_yam.py` over a deployment to do it: the
serving boxes run a fork that is *newer* in other respects (`--revision`
pinning, a request `seed`, load and GPU telemetry), and replacing the file
regresses all of it to fix twelve lines. `scripts/patch_server_to_pil.py` adds
just those twelve lines to a checkout, in place and idempotently.

Whether the box you are pointed at needs it is one request, no hardware:

```bash
uv run python scripts/probe_encoded_frames.py 192.168.0.107:4090
```

## Deviations from the reference deployment

| Here | Reference | Why |
| --- | --- | --- |
| One bounded command per control tick, clamped against the previous target at `0.10` rad, and the gripper at `0.30` of its stroke | Walks to every action, clamped against the measured pose at `0.30` rad | Measured worse on this rig; available as `--reach-actions`. The gripper's looser clamp is below. |
| Inference starts from where the arms stand | Ramps to the training start pose | Same; available as `--reset-start-pose`. |
| `--request-timeout` defaults to 60 s | no timeout | A client that can block forever holds torque on two energized arms. 60 s clears a cold or non-CUDA-graph call. |
| Joint targets are clipped to the URDF's limits | not clipped | This package knows them; clipping an in-range action is a no-op. |
| The gripper target is clipped into `[0, 1]` | not clipped | `PositionCommand` requires a normalized effector. |
| A server that cannot decode JPEG drops the run to raw frames, once and loudly | fails the request | Losing an energized run to a payload format is a poor trade. |

## Useful limits

```bash
uv run openpi-control infer \
    --instruction "pick up the object" \
    --max-step-rad 0.15 \
    --max-effector-step 0.15
```

Lowering `--max-step-rad` makes a first run gentler, at the cost of the policy
converging on each action more slowly. It does not change how many actions run:
the whole chunk always does. It is the wrong knob for an arm that is *rushing*
— see *When the arms are too reactive* — because a clamp that bites shortens
the path rather than slowing it down.

`--max-effector-step` defaults to `0.30` rather than sharing the joints' `0.10`,
because the gripper is not a joint travelling through space — it is a jaw that
has to finish closing before the arm moves on. At `0.10` a full open-to-close
takes ten ticks, a third of a second at 30 Hz, and the plan has already moved
the hand on while the jaw is still travelling. This checkpoint commands
full-stroke swings constantly, and on a fold run the effector rather than the
arms was what most of the clamp count was refusing. `0.30` reaches either stop
in four ticks and still refuses a single action that would slam the jaw.

That also makes the clamp count in the chunk log harder to attribute, since it
counts an action the limit touched *anywhere* among its 14 values. To find out
which, raise `--max-effector-step` and watch whether the count falls: if it
does, it was the gripper, not the arms.

Use `--interface left=can2 --interface right=can3` when the rig's CAN aliases
are different. Use `--camera NAME=DEVICE` only when discovery by the rig's
serial number is not suitable.
