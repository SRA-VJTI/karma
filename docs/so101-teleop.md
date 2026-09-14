# SO101 Meta Quest teleoperation

The `so101` rig drives one SO101 follower with the **right Quest controller**.
`so101_bimanual` drives two followers with their matching controllers. Both use
the existing native FeeTech driver and `E_SO101` gripper; no leader arm or i2rt
checkout is needed. Inference, rollout, HITL, and collection remain separate work.

## Setup

Install/build the checkout using the [main setup instructions](../README.md).
For this teleop-only environment, install the VR extra without the default
`cell` group (which also downloads LeRobot, PyArrow, PyTorch, and CUDA):

```bash
UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 uv sync --locked --no-default-groups --extra vr
```

Keep `--no-default-groups --extra vr` on subsequent `uv run` commands, as shown
below. Plain `uv run` would restore the full `cell` environment. For a wheel
installation, install `openpi-control[vr]`. The native `pi_control_node` must also be built/available,
as for the existing arms. Install ADB and enable developer mode / USB debugging
on the Quest for the USB connection.

Connect the SO101 follower's powered serial adapter. Its five arm servos use
IDs 1–5 and its gripper uses ID 6, at 1,000,000 baud. Replace `/dev/ttyACM0`
below with your adapter path (prefer `/dev/serial/by-id/...` when available).

```bash
uv run karma doctor --rig so101 --interface-override right=/dev/ttyACM0
```

The packaged geometry is `SO101.urdf`, derived from `so101_new_calib.urdf`, and
the IK target is its `end_link` frame. Native joint readings and commands are
**radians**, ordered shoulder pan, shoulder lift, elbow flex, wrist flex, wrist
roll. The gripper is separate. This runtime uses native servo firmware zeros
and instance configuration; it does not import LeRobot calibration JSON or
normalized joint observations. Verify your arm's measured zero pose and joint
directions agree with the packaged model before Cartesian teleoperation.
Do not run `openpi zero` at an arbitrary pose: it writes the current pose as the
servo zero. See [zeroing instructions](cli.md#zero) if calibration is needed.

## Calibrate this arm and record home

Close teleop first, leave the arm powered, and support it. Run:

```bash
uv run karma calibrate-so101 \
  --interface /dev/ttyACM0 --output calibration/so101-right.json --center-encoders
```

The wizard disables torque and asks you to place each joint near the middle of
its travel, with the gripper halfway open and wrist roll at its intended middle
orientation. It backs up the existing firmware registers beside the output file
before setting encoder midpoints. This prevents the encoder boundary from falling
inside a joint's working travel. Older calibration profiles become stale.

Next, move the shoulder, elbow, wrist flex, and gripper through their full travel
by hand without forcing the stops. **Do not sweep wrist roll**: the wizard uses
its full single-turn encoder range, as LeRobot does. Fully open and close the
gripper. Press Enter after the sweep, capture the fully closed and fully open gripper when prompted,
then position the arm and gripper at your desired home and capture it. The closed-jaw
check is separate from home: home may have an open gripper. A failed closed-jaw
check lets you retry while keeping the recorded sweep. Closed/open captures define
the gripper range, with the sweep used only to check plausible travel. Gripper
capture ignores other joints moving; an unsteady capture retries without losing
the sweep. Torque
remains off afterwards; the wizard does not command the arm to move.

This creates a **native SO101 profile**, not a drop-in LeRobot calibration file.
It records the new firmware homing offsets and preserves the native 4096-count
radian scale. For stock SO101 geometry, measured mechanical range midpoints are
aligned to the packaged model's range midpoints. The gripper direction is inferred from the captured endpoints and applied through
the native servo direction setting, preserving 0 = closed and 1 = open.
Incomplete travel or a wrap in a non-roll joint is rejected. If a joint still wraps, repeat
centering with that joint closer to the middle of its mechanical travel.

Omit `--center-encoders` only if the encoders are already centered and you want
to preserve firmware offsets. That mode writes no EEPROM and requires manually
recording all six ranges, including wrist roll, without crossing the encoder
boundary. The profile is saved only after the complete calibration succeeds.

Start teleop with that profile:

```bash
uv run karma teleop \
  --rig so101 --interface right=/dev/ttyACM0 \
  --calibration right=calibration/so101-right.json --rate 50 --open-quest
```

The profile generates adjacent `.arm.json` and `.gripper.json` native instances;
packaged robot files stay unchanged. Offsets apply in the native driver, and
measured limits are intersected with the model limits for both IK and command
validation. The saved home is used for Ctrl+C parking and, unless overridden,
thumbstick return-to-rest. Home is a joint target relative to calibration;
it never changes the calibrated zero. A fresh calibration is required if a
servo's firmware homing offset changes. Keep the profile for this physical arm;
the local `calibration/` folder is ignored by Git. Existing output files are
never overwritten by the wizard.

## Run

With the Quest connected over USB and USB debugging authorized:

```bash
uv run karma teleop --rig so101 \
  --interface right=/dev/ttyACM0 --rate 50 --open-quest
```

The command starts the relay and USB tunnel. Enter immersive VR in the Quest
browser. Hold the right grip button to clutch, then move the controller. Release
the grip to hold the arm's last target while repositioning your hand. The trigger
controls gripper closure while clutched; A enables precision scaling. Tracking
staleness pauses movement and recovery re-anchors the controller pose.

The initial IK pose comes from measured arm state, so no rest-pose move is issued
by the Quest bridge on startup. Thumbstick click explicitly requests a timed
return to the configured rest pose (default five zeros). Set it to an appropriate
pose for your mounting, for example `--rest-pose-right 0,0,0,0,0`. This is separate
from shutdown parking at native `home_pos`. Ctrl+C returns the arm to that
configured home pose, then de-energizes it. Let parking finish; a second Ctrl+C
interrupts parking. Add `--no-park` only to power down in place.

For two SO101 followers:

```bash
uv run karma teleop --rig so101_bimanual \
  --interface left=/dev/ttyACM0 --interface right=/dev/ttyACM1 \
  --rate 50 --open-quest
```

Use `--arm left` or `--arm right` to select one side of that rig. Existing LAN/TLS
and external-relay options also work. `--yam-xml` is rejected for SO101.

## Motion and tuning

SO101 has five arm joints, so it cannot match every position and orientation
combination from a six-dimensional controller pose. The solver favours tool
position and approximates orientation, using the complete arm Jacobian rather
than YAM's separate wrist solve. Defaults are 0.5 translation scale, 5 cm target
reach limit, and 0.02 rad per-joint step cap. At the suggested 50 Hz this cap is
1 rad/s; it is a per-tick cap, so increasing the rate also increases that bound.
Native joint limits and state checks remain active. The solver does not perform
collision avoidance.

`--max-dq-pos` controls the first three joint caps and `--max-dq-rot` the last two.
The YAM-specific damping/wrist CLI knobs do not tune the SO101 solver. Browser
settings are stored separately for SO101 and YAM; saved SO101 slider values take
precedence over CLI values on connection. Reset settings in the browser to use
the current session's defaults. Torque haptics reuse the native effort estimate;
they are not a calibrated force measurement.

The automated tests cover the URDF geometry against MuJoCo, IK convergence and
bounds, five-joint seeding, clutch/release, stale tracking recovery, gripper
polarity, and CLI profiles. Physical teleoperation still needs validation on
your connected and calibrated SO101.

## Quest reports `no permissions` in ADB

On Ubuntu, install the packaged Meta/Oculus USB rule:

```bash
sudo install -m 0644 scripts/udev/51-meta-quest.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

Unplug and reconnect the Quest, then run `adb devices -l`. If it now reports
`unauthorized`, put on the headset and accept the USB debugging prompt. Once
it reports `device`, rerun teleop; the command creates `adb reverse` itself.

## Starting pose and calibration

Quest movement is relative to the pose captured at each grip press. Robot joint
commands are still in native radians because the motor controller and URDF need
a shared joint coordinate system. A LeRobot/Hugging Face calibration file is
not used, and moving the arm to a new starting pose does not redefine its zeros.

At startup, IK follows the native SO101 convention: measured joints may exceed
command limits by the packaged `pos_error_margin` (0.1 rad / 5.73 degrees), but
the initialized command stays within the limits. This handles tracking error
near a folded pose without rejecting every small overrun. It matches the native
FeeTech driver's goal clipping; it does not expand the arm's commanded workspace.
Grip captures this bounded starting pose, so pressing grip adds no motion by
itself. Subsequent IK steps are capped as described above.

A measurement beyond that native tolerance still stops startup with joint names
and values. After de-energizing, check the physical pose and calibration. Do not
zero at an arbitrary pose to dismiss the error.
