# Flash-ALOHA on the default bimanual YAM rig

This is the hardware bring-up stage of the port. The readiness checks and JSON
report below are implemented. A Flash-style kinesthetic teaching mode, calibrated
friction compensation, and Flash policy execution are **not implemented by this
change**. Hardware validation is pending an operator run on the Linux workstation.

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

## 3. Next implementation: kinesthetic teaching and recording

The required native change is a dedicated teaching mode, not a wrapper that
repeatedly re-enters calibration float. Integration points:

- `native/pi_control/src/pi_device_arm.cpp`: gravity and per-joint bounded
  friction compensation in the native loop. Use per-arm calibration, a smooth
  sign transition/deadband near zero velocity, torque caps and retained feedback
  age/thermal/communication protection. Validate release stability before higher
  speed sweeps; overcompensation can drive motion without a hand on the arm.
- `native/pi_control/src/pi_device_effector.cpp`: explicit backdrivable E_Yam
  teaching behavior and an intentional transition back to position control.
  The passive `E_Yam_Handle` is a different device; replacing the configured
  E_Yam with that model does not make the follower gripper backdrivable.
- `types.py`, `protocol.py`, `native.py`, `arms.py`: explicit teaching/hold
  transitions and capability reporting. Rebase position targets to measured
  positions when leaving teaching so an old command cannot pull the arm away.
- `cli.py` and the recording source: a dedicated kinesthetic workflow, initially
  with explicit episode start/stop controls. Preserve synchronized raw camera
  frames and measured trajectories; define action alignment and dataset metadata
  before training. A measured teaching trajectory is not an issued PD target.

First validate one arm at low speed and no cameras, including loss of feedback,
process exit, mode transitions and release stability. Then add both grippers,
both arms, and synchronized camera recording. A foot pedal can start/end takes;
it is not a substitute for the hardware stop. Quest teleop remains available for
comparison, but is not required for hand-guided teaching.

## 4. Then: policy pipeline

Keep a checkpoint contract for `[left q1..q6, left gripper, right q1..q6,
right gripper]`, radians, camera roles, control/data rates and gripper polarity.
Karma's native/wire gripper is 1=open while its dataset gripper is 0=open; convert
exactly once at the relevant boundary. Match training and inference image
preprocessing and operator masks; preserve original recordings and calibrate
mask geometry for this YAM rig. Reusing ARX masks or checkpoints is not validated.

Implement the Flash action-smoothing/chunk-transition path with replay tests for
continuity, timing, stale responses, bounded velocity/acceleration and stopping.
Benchmark policy latency against the chosen action rate before enabling faster
motions. First run an offline recorded-episode contract check, then a bounded
low-speed rollout. Karma's existing MolmoAct2 endpoint is a useful separate
baseline, not evidence that a Flash policy has been ported.

Reference inspection: Karma `66bcd5ed1caa3f21d0f5bf61cd57d052b62ddad2`,
Flash-ALOHA `4172a8ed8cadb6eeb4a59285826e83cb00a2f01c`.
