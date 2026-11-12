# Flash-ALOHA on the default bimanual YAM rig

This draft implements YAM kinesthetic teaching, explicit gripper backdriving,
next-state demonstration recording, a task-specific ACT training/server path,
shared operator masking and overlapping action chunks. It also includes
[parametric scissor-style grip CAD](../hardware/scissor/README.md).
Physical hardware validation, per-arm friction calibration, mounting fit and
training on real demonstrations are still operator work. No ARX gains or
checkpoints are treated as YAM-compatible.

## Target and approach

Use Karma's existing `yam_bimanual` rig unchanged:

| Role | Hardware | Interface / view |
| --- | --- | --- |
| left | YAM, six joints + E_Yam linear gripper | `can0` |
| right | YAM, six joints + E_Yam linear gripper | `can1` |
| top | overhead camera | `top` |
| wrists | one camera per wrist | `left_wrist`, `right_wrist` |

The [Flash-ALOHA reference](https://github.com/JohnYao93/flash-aloha) uses direct
hand-guided demonstrations on backdrivable arms. Its current hardware path uses
the ARX SDK; do not run its `L5` tuners against a YAM or copy its calibrated gains.
Keep Karma's native YAM driver, motor units, watchdogs and dataset conventions.

The goal of "zero friction" here is low-resistance kinesthetic teaching through
gravity and calibrated friction feed-forward, not a guarantee of literally zero
friction. The [hardware tutorial](https://docs.google.com/document/d/1GAxYQq6ppRNMVVDRZRWVP7z5HXIXlONxjdPNFYmqPC0/edit)
also makes gripper backdrivability part of the mechanical design. Its printed
handles and adapters are not verified to fit E_Yam. Measure the YAM jaw mounting,
clearance, travel and cable strain before making adapters; add the measured handle
and camera payload to the gravity model.

## 1. Capture readiness without enabling motors

Install on Ubuntu using the [standard build instructions](../README.md#install-once).
Identify the physical adapter for each arm; `can0`/`can1` can swap after replugging.
Use [persistent names](yam-setup.md#which-adapter-is-which) when available.
With other robot processes stopped, configure the identified buses:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
ip -details -statistics link show can0
ip -details -statistics link show can1
uv run karma doctor --rig yam_bimanual --report logs/yam-bringup.json
```

For pinned names, add `--interface-override left=can_left` and
`--interface-override right=can_right` to `doctor`. The other rig workflows use
`--interface left=can_left --interface right=can_right`.

The report records the resolved arm interfaces, model and effector names, checks,
failure/warning counts, platform and UTC capture time. `physical_validation` is
always `not_performed`; a green static report is not a motor or calibration test.
Exit status is 1 for failed checks and 2 for invalid configuration/report write
errors. Missing cameras remain warnings in `doctor`.

New checks reject a shared interface between selected arms (including serial
symlink aliases), and detect CAN controller `BUS-OFF`, stopped/sleeping states,
and error warning/passive states even when the link is administratively up.
Unavailable controller status is reported as a warning. Bus ownership is also
checked immediately before constructing motor backends, including when preflight
is skipped. This does not detect another independently running process.

Optional passive observation:

```bash
uv run karma doctor --rig yam_bimanual --probe --report logs/yam-listen.json
```

`--probe` listens only: it sends no motor query or enable command. Silence is
normal for idle DM motors, and traffic from one servo does not establish that
all seven motors are responding. Duplicate bus assignments suppress this probe.

## 2. Operator check: one arm, then both

These commands **energize motors**. Support the arm for shutdown, verify access to
the hardware stop, and start with one arm and a clear workspace. Keep the factory
motor timeout and native protection enabled. No firmware zero write is part of
this procedure; zeroing requires the manufacturer's mechanical reference pose.

First check holding, state freshness, motor faults and gripper feedback:

```bash
uv run karma live --rig yam_bimanual --only left --no-cameras --no-park
```

Repeat with `--only right`. `--no-park` avoids an unverified automatic home move;
Ctrl+C de-energizes in place, so support the arm. Verify all six joint signs and
the gripper's 1=open native convention from measured state. Record anomalies and
the current commit (`git rev-parse HEAD`) with the doctor report.

After verifying holding and the model/payload, test the existing **bounded gravity
calibration float** on one supported arm:

```bash
uv run karma live --rig yam_bimanual --only left --float --no-cameras --no-park
```

This is a small-displacement calibration check. The native drift guard returns
the arm to HOLD when displacement exceeds its threshold. Do not disable that
guard to obtain a teaching mode. It does not establish that the E_Yam gripper is
backdrivable, and it does not apply calibrated friction compensation. Record
drift direction at several poses; validate gravity before tuning friction.
Repeat for the right arm, then test both arms holding together.

Bring-up evidence to attach to the draft PR: doctor JSON, commit, native logs,
left/right bus identity, payload/mount details, motor fault observations, joint
sign/zero observations and gravity-float drift. Camera serials are local setup
details; the packaged serials are placeholders.

## 3. Kinesthetic teaching and recording

Rebuild the native runtime from this branch. An old binary reports no teaching
capability and is rejected before entering teaching. The normal YAM six joints
and E_Yam gripper each use a measured-velocity friction profile:

```text
assist = clamp(coulomb * tanh(velocity / transition) + viscous * velocity,
               -compensation_limit, +compensation_limit)
torque = model_gravity + assist - damping * velocity
```

The gripper uses zero model gravity. Values are joint-frame quantities before
Karma's existing `torq_rescale`; the code converts DM raw velocity direction
correctly, including the inverted gripper. Native torque/thermal/communication
protection remains active. Default profiles have **zero friction assistance**,
0.1 damping and a 0.5 rad/s velocity guard. They do not claim calibrated zero
friction. First validate gravity/payload, then tune each joint and each physical
arm independently at low speed. Increase assistance only from measured resistance;
check release stability at several poses. Configuration is immutable while active.
The accepted assistance caps are 2 Nm per arm joint and 0.5 Nm for the gripper,
not suggested tuning values.

Generate a local profile (will not overwrite an existing file), then practice:

```bash
uv run karma teach --write-config yam-teaching.json
uv run karma teach --rig yam_bimanual --arm left --config yam-teaching.json
# Repeat right, then both:
uv run karma teach --rig yam_bimanual --arm both --config yam-teaching.json
```

Startup energizes and holds using Karma's normal startup, including gripper
calibration; keep fingers clear until the console says ready. Type a key then
Enter: `t` practice, `r` start/restart recording, `s` hold and save, `h`/`d` hold
and discard, `q` hold and exit. Ctrl+C/exit de-energizes in place without parking;
support the arms. No position targets are sent while teaching.

A 250 ms lease is renewed by nonblocking sends from the recording/control loop,
not the background heartbeat. Native teaching mode is reported repeatedly on the
bounded status channel; renewal does not wait for per-frame acknowledgements. Missed renewal, unknown/>100 ms joint feedback age, nonfinite state,
joint limit violation or excessive velocity exits teaching to HOLD. HOLD rebases
position targets to the measured pose. One native node protects each arm and its
gripper; Python holds both on a detected fault. This is not a hardwired
cross-arm emergency stop. An individual child gripper guard is observed by its
parent on the next native tick. Validate stop/release behavior physically before
normal use; retain the rig's independent hardware stop.

After fitting the grip prototypes and validating both arms, record all cameras:

```bash
uv run karma teach --config yam-teaching.json --record \
  --repo-id local/yam-cup --root data/yam-cup --task "pick up the cup" --fps 30 \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_SERIAL --camera-serial right_wrist=RIGHT_SERIAL
```

For UVC cameras use `--camera top=/dev/videoN` etc. Camera identity overrides are
required for the packaged placeholder serials. Raw RGB recordings remain intact.
The shared recorder samples the most recent frames; this does not establish
hardware-synchronized camera exposures.

Each observation at t is labeled with **measured state at t+1**, not an unissued
PD command. Grippers use dataset 0=open; the native/wire convention is 1=open.
The final unpaired frame is dropped on save (the console's capture count includes
that dropped frame). Single-frame takes produce no saved episode. Boundaries
never share pending frames. `meta/teaching.json` records profiles, ordering and
alignment; `action.dt_s` records actual intervals. The trainer rejects sampled
intervals outside 0.5–1.5 nominal periods; inspect timing before collecting at
higher rates. Save is performed after HOLD, before slow video finalization.

## 4. Train and serve ACT

Copy [yam-act.json](../examples/flash/yam-act.json) and set the exact task, fps,
image size and operator masks for this rig. Each mask is a normalized RGB-frame
rectangle `[x0,y0,x1,y1]`; empty lists apply no masks. Inspect camera snapshots to
choose rectangles that exclude hands/handles without hiding the task object.
Masking is fixed geometry, not automatic operator segmentation. The same function
masks, resizes and applies ImageNet RGB normalization in training and serving.

Train on a GPU machine using Karma's pinned LeRobot 0.6.1 and an appropriate CUDA
PyTorch installation. The workstation's default `uv` configuration uses CPU
PyTorch wheels; it does not provision a training GPU. A first training run also
downloads torchvision ResNet18 ImageNet initialization. No cloud service is
created by these commands.

```bash
python -m openpi_control.flash_train \
  --repo-id local/yam-cup --root data/yam-cup \
  --config examples/flash/yam-act.json --output checkpoints/yam-cup \
  --steps 20000 --batch-size 8 --device cuda
python -m openpi_control.flash_serve \
  --checkpoint checkpoints/yam-cup --device cuda --host 127.0.0.1 --port 8202
```

The trainer checks bimanual next-state metadata, uses action delta timestamps,
normalizes the 14 state/action dimensions with saved dataset statistics, and
trains LeRobot ACT. Its output includes weights, ACT config, `normalization.npz`,
`flash.json` and training metadata. ACT is task-specific: the instruction must
exactly match its one training task. This is an initial fixed-step trainer; it
has no held-out success estimator or automatic checkpoint selection. Evaluate
recorded predictions and task success before choosing a checkpoint for use.

Serving defaults to loopback. Use an SSH tunnel for a remote GPU, for example
`ssh -L 8202:127.0.0.1:8202 GPU_HOST`, then on the robot workstation:

```bash
uv run karma infer --rig yam_bimanual --flash-policy \
  --server http://127.0.0.1:8202 --instruction "pick up the cup" \
  --control-rate 30 --request-timeout 2 --speed 0.25 --no-park \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_SERIAL --camera-serial right_wrist=RIGHT_SERIAL
```

Check `/act` health before powering up: exact YAM dimension, camera ordering,
polarity, task and fps must agree. State/action order is `[left q1..q6, left
gripper, right q1..q6, right gripper]`, radians plus normalized grippers. Serving
converts native grippers to dataset polarity and predicted actions back exactly
once. It clips predicted grippers to [0,1].

The client retains the unexecuted suffix, blends the next plan's first overlap
with cubic old-plan decay, and executes only the configured prefix. Unlike the
reference's cubic implementation, the last overlap sample lands exactly on the
new plan. Zeros are valid targets. Karma's bounded executor continues to enforce
position-step limits; blending is not an acceleration/jerk guarantee.

This first integration uses **synchronous inference** and holds the last position
while waiting for the next chunk. Prefetch is disabled for this path so overlap
indices have explicit meaning. It rejects reach-actions, carry-targets and the
MolmoAct-specific reset pose. Slow rollout and executor clamping further change
wall-clock timing. It is not a validated near-human-speed reproduction of
Flash-ALOHA; benchmark latency and physical tracking before increasing speed.
No trained YAM policy or task-success claim is included.

Reference inspection: Karma `66bcd5ed1caa3f21d0f5bf61cd57d052b62ddad2`,
Flash-ALOHA `4172a8ed8cadb6eeb4a59285826e83cb00a2f01c`.
