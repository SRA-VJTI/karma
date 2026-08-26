# Recording datasets

`openpi-control record` teleoperates a rig and writes what happens to a
[LeRobot](https://github.com/huggingface/lerobot) dataset — parquet for state and
action, one mp4 per camera. Episode control is in the headset, so a session is
`B … do the task … Y`, repeat.

LeRobot writes v3.0 datasets here (`CODEBASE_VERSION v3.0`); the `lerobot`
extra needs **Python 3.12+**, so on 3.11 it resolves to nothing and `record`
says so rather than failing obscurely.

```bash
uv sync --extra lerobot   # torch, on top of the default set
vr-teleop-relay                        # in the vr-teleop-kit checkout

uv run openpi-control record \
    --repo-id you/yam-fold-towel \
    --task "fold the towel in half" \
    --num-episodes 20 \
    --vr-kit ~/vr-teleop-kit
```

| In VR | Does |
| --- | --- |
| grip (squeeze) | drive that arm; trigger is the gripper |
| **right B** | start an episode — or throw the current one away and restart it |
| **left Y** | save the episode |

The session ends itself after `--num-episodes` saves, or on ctrl-c.

## Where the pieces live

VR teleoperation is not reimplemented here. [`vr-teleop-kit`](https://github.com/Dream-Machines-Robotics/vr-teleop-kit)
already owns the WebXR relay the headset talks to, the clutch-relative pose
mapping, and a damped IK solver tuned for the YAM's wrist.
`openpi_control.teleop_vr` is the adapter that turns its `BiQuestTeleoperator`
into a teleop source, so the headset drives arms through this package's native
stack:

```
Quest  --WebXR-->  vr-teleop-kit relay
                          |
                   BiQuestTeleoperator     pose mapping + IK
                          |  joint targets
                   QuestTeleopSource       openpi_control.teleop_vr
                          |  PositionCommand
                   FollowerArm  ->  pi_control_node  ->  CAN  ->  YAM
```

`vr-teleop-kit` is a sibling checkout rather than a dependency — it carries its
own relay, web assets, and the i2rt clone holding the YAM MJCF the IK loads.
Point `--vr-kit` at it, or install it into this environment and drop the flag.

## Rehearsing without a headset

`--teleop hold` records one episode with the arms deliberately stationary. It
collects no useful data; it is how you check everything *around* the teleop —
that the arms come up, that all three cameras land in the dataset with the right
shape and the right colours, that the loop holds the frame rate:

```bash
uv run openpi-control record --teleop hold --hold-seconds 5 --dry-run
```

`--dry-run` runs the entire session, arms included, and writes nothing to disk.

## Recording at the maximum rate

The ceiling is the cameras, and it is **90 Hz**:

```bash
uv run openpi-control record --fps 90 \
    --repo-id you/yam-fold-towel --task "fold the towel in half" \
    --vr-kit ~/vr-teleop-kit
```

`--fps` sets the loop rate, the dataset rate, **and** the camera rate together.
That last one matters: cameras pinned at 30 while the loop ran at 90 would write
each frame three times and call it data. Use `--camera-fps` only if you
deliberately want them different.

Measured on this cell, all three D405s running at once:

| Rate | Cameras (3× 848x480) | Loop, p95 tick | Budget |
| --- | --- | --- | --- |
| 30 Hz | 30.0 / 30.0 / 29.5 | 34.3 ms | 33.3 ms |
| 60 Hz | 60.0 / 60.0 / 59.8 | 16.8 ms | 16.7 ms |
| 90 Hz | 90.0 / 90.0 / 90.0 | 11.3 ms | 11.1 ms |

90 Hz is the D405's fastest colour mode at any resolution up to 848x480 (above
that, 1280x720 caps at 30). At 90 Hz the three streams push ~220 MB/s over the
shared USB 3 uplink with no dropped frames. The arms are not the limit — the
native node publishes at 200 Hz.

The default stays 30, because the cost is real: **~5 GB per hour** of recorded
episode at 90 Hz with three cameras (AV1, ~1.4 MB/s), three times the 30 Hz
figure. Recording fast and subsampling later is lossless; the reverse is not.
Pick deliberately.

Three things make 90 Hz possible rather than merely requested:

- **Frames are encoded during capture, not inside `save_episode`.** LeRobot's
  default stages every frame as a PNG and encodes the lot when the episode is
  saved. At three cameras and 90 Hz that is 270 PNG encodes a second (~37 ms
  each on a noisy 848x480 frame), several GB of temporary files per take, and a
  `save_episode` that blocks the control loop — with the arms holding their last
  command — for tens of seconds. `LeRobotSink` passes `streaming_encoding=True`,
  which encodes as frames arrive. Measured on a 20 s take at 90 Hz with three
  cameras: **1779 frames at 88.9 Hz, `save_episode` blocked 0.45 s, 28 MB, no
  staged files.** The cost is a bounded encoder queue: if it fills, LeRobot drops
  a frame with a warning rather than blocking. That is the right trade for a
  teleop loop — a frozen loop is worse than a dropped frame — but it is real, so
  check the frame count a session reports.

- **Cameras are opened as `rgb8`, not `bgr8`.** LeRobot stores RGB, and flipping
  channels in numpy costs 1.37 ms per 848x480 frame against 0.03 ms for a plain
  copy — 4.1 ms of an 11.1 ms tick with three cameras, over a third of the
  budget spent swapping bytes. Asking the SDK for RGB moves that into its native
  conversion, where it is free. `record.needs_rgb_conversion` is what decides,
  and it defaults to converting for any camera that does not declare a format.
- **The mode is checked before the arms are energized.** `record` refuses a rate
  the camera does not offer, and says which ones it has:

  ```
  [FAIL] mode top   45 fps at 848x480 is not offered; this camera does 5, 15, 30, 60, 90
  ```

## What gets recorded

One row per tick, at `--fps` (default 30):

| Feature | Shape | Contents |
| --- | --- | --- |
| `observation.state` | `(7 × arms,)` | measured: `<arm>_joint_1..6`, `<arm>_gripper` |
| `action` | `(7 × arms,)` | commanded, same layout |
| `observation.images.<camera>` | `(H, W, 3)` | `top`, `left_wrist`, `right_wrist` |

Both arms give a 14-dim vector; `--only right` gives 7-dim, which is a strict
prefix of the bimanual layout rather than a different schema. Cameras are typed
`video`, so an episode is one mp4 per camera instead of a directory of PNGs —
about two orders of magnitude on disk.

Anything the teleop did not command records as *where the arm already is*, not as
zero. A zero would read as a command to fold the arm up.

### The gripper is inverted between the two worlds

This is the single most dangerous detail in the pipeline, so it is worth stating
plainly:

| | Open | Closed |
| --- | --- | --- |
| `openpi-control` (`EffectorState`, `PositionCommand`) | **1.0** | 0.0 |
| LeRobot, the Quest trigger, recorded `*_gripper` columns | **0.0** | 1.0 |

Recorded columns use the LeRobot convention, because the point of writing a
LeRobotDataset is that LeRobot and openpi tooling can read it — and because the
datasets this cell already produced use it. One collection with two polarities in
it would be worse than either choice.

`record.to_native_gripper` / `record.to_dataset_gripper` are the only two places
that flip, and they are named for their direction so a bare `1.0 - x` never has
to be interpreted. **When you deploy a policy trained on this data, convert its
gripper output back to native before commanding an arm.**

## Three ways a dataset comes out quietly wrong

The recorder refuses all three rather than producing a file that looks fine.

**A stalled arm.** When CAN frames stop arriving, the native node keeps
republishing the arm's last cached pose. Recording that teaches a policy the arm
was motionless while the operator was moving it. A tick where any arm's state is
older than 100 ms is skipped — not commanded, not written — and half a second of
that discards the open episode:

```
  episode discarded: left stalled for 15 ticks
```

A momentary stall costs one frame and the take continues.

**Frames in the wrong channel order.** LeRobot treats a three-channel frame as
RGB. `record` therefore opens the cameras as `rgb8` and stores their frames
verbatim; a camera handing over BGR (the default everywhere else here, because
that is what OpenCV wants) is converted instead. The decision is per camera, from
its declared `pixel_format`, and an undeclared format is assumed BGR — guessing
"already RGB" would silently store every dataset with red and blue swapped.

**A discarded take leaking into the next one.** Throwing a take away has to
clear the buffer *and* anything already written for it, or the next `save`
silently includes both. On lerobot 3.0 that is one `clear_episode_buffer` call —
verified by recording a take, discarding it, recording a second, and reading the
dataset back to confirm only the second one's frames are there and no orphaned
files remain. It was not always one call: older versions left per-frame PNGs on
disk for `video` features, which is worth knowing if you ever pin an older
lerobot.

## Episode boundaries

Events apply before the frame is written, so a take runs from the **B** tick
through the tick before its **Y**. The tick carrying the save is the operator
reaching for a button rather than doing the task. This matches `vr-teleop-kit`'s
recorder, so muscle memory and frame counts carry over between the two.

Teleop keeps driving the arms between episodes on purpose — resetting the scene
with the arm is most of what happens between takes, and an operator should not
have to think about whether the arm is live. Only frames inside an episode are
written.

Two more rules worth knowing:

- **B while recording** discards and immediately restarts. A botched take is
  redone with one press rather than stop-then-start.
- **Ctrl-c during an open episode discards it.** A partial take stops mid-motion.
  Press **Y** first if you want it.

## Teardown order

The order is deliberate, and it is why the whole session lives inside one
foreground command:

1. The dataset is **finalized on disk before the arms are parked**, so an
   interrupted shutdown still leaves a complete, loadable dataset.
2. The arms are parked at `home_pos` and de-energized.
3. `--push-to-hub` uploads **after** the arms are down — a few hundred megabytes
   of video takes minutes, and there is no reason to hold motors energized for
   it. A failed upload is reported and the local copy kept for a manual retry.

Cameras are opened *before* any motor is energized, so discovering that one is
held by another process does not happen with two arms live.

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--repo-id` | — | required unless `--dry-run`; the owner must match your `hf auth login` account or `--push-to-hub` 403s |
| `--task` | `teleop` | stored on every frame. **Set it** — relabelling means rewriting the dataset |
| `--num-episodes` | `0` | end after this many *saved* episodes; 0 runs until ctrl-c |
| `--fps` | `30` | dataset, control-loop, **and** camera rate. 90 is the ceiling here |
| `--camera-fps` | `--fps` | run the cameras at a different rate from the loop |
| `--teleop vr\|hold` | `vr` | `hold` is the headset-free pipeline check |
| `--only ARM` | — | one arm, which also drops the other wrist camera |
| `--no-cameras` | off | state and action only |
| `--dry-run` | off | full session, nothing written |
| `--vr-kit PATH` | — | a `vr-teleop-kit` checkout |
| `--vr-url` | `ws://127.0.0.1:8443/ws` | the relay |
| `--yam-xml` | — | YAM MJCF the IK loads, if it is not found automatically |
| `--push-to-hub` / `--private` | off | upload when the arms are down |
| `--no-park` | off | de-energize in place instead of parking at `home_pos` |

Preflight runs first and nothing is energized unless every arm *and every
declared camera* passes. A missing camera is fatal here, unlike in `doctor`:
recording with a view silently absent produces a dataset that is wrong, not a
cell that is merely unchecked.

Exit status is 1 if no episode was saved, so a wrapper script cannot mistake an
empty session for a successful one.

## One camera, one consumer

A camera streams to one process at a time. **Do not enable the Quest camera
stream while recording** — the relay would grab the same RealSense and the
recorder's open fails. If you re-run `record` immediately after a session and see
`Device or resource busy`, that is the kernel still holding the v4l2 node; the
reader retries for a few seconds before giving up, so it usually resolves itself.

## Using the loop yourself

The record loop takes three seams, which is what makes a session testable without
hardware, a headset, or `lerobot` installed:

```python
from openpi_control.record import (
    ArmTarget, EpisodeEvent, MemorySink, TeleopStep, record_session,
)

class Circles:
    """A TeleopSource is just poll() -> targets + an occasional event."""

    def describe(self): return "circles"
    def close(self): pass

    def poll(self, states):
        # states: arm name -> ArmState | None
        return TeleopStep(
            targets={"left": ArmTarget(position_rad=(0.0,) * 6, effector=1.0)},
            event=EpisodeEvent.NONE,
        )

sink = MemorySink()
result = record_session(arms=arms, source=Circles(), sink=sink, task="demo", fps=30)
print(result.summary())
```

`arms` is anything with `latest_state` and `command` — a `FollowerArm` from a
live session, or a stand-in backend. Swap `MemorySink` for `LeRobotSink` to write
a real dataset, and see `openpi_control.record.ScriptedSource` for the smallest
complete source.
