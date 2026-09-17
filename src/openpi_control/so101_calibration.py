"""Manual SO101 travel calibration and home capture, with optional encoder centering.

Offsets align the midpoint of each measured mechanical range with the midpoint
of the stock URDF range. Radians retain the native 4096-count encoder scale.
This is a native profile, not a LeRobot normalized-action calibration file.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import math
import os
import select
import subprocess
import sys
import termios
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import serial

from .config import resolve_model_assets
from .exceptions import ConfigurationError
from .servos import ft_serial

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TICK = 2 * math.pi / 4096


@contextmanager
def _open_bus(interface):
    """Exclude native control and other serial clients before any bus traffic."""
    try:
        with serial.Serial(interface, 1000000, timeout=0.2, exclusive=True) as bus:
            fcntl.ioctl(bus.fileno(), termios.TIOCEXCL)
            try:
                users = subprocess.run(["fuser", str(interface)], capture_output=True, text=True)
                if users.returncode not in (0, 1):
                    raise ConfigurationError(
                        "cannot check serial port ownership; stop robot programs"
                    )
                others = {int(pid) for pid in users.stdout.split() if pid.isdigit()} - {os.getpid()}
                if others:
                    raise ConfigurationError(
                        f"serial port is already in use by processes {sorted(others)}"
                    )
                yield bus
            finally:
                fcntl.ioctl(bus.fileno(), termios.TIOCNXCL)
    except OSError as err:
        raise ConfigurationError(f"cannot use calibration serial port {interface}: {err}") from err


def _read(bus, servo_id, address, length=2, *, attempts=5):
    # A single dropped reply on a shared half-duplex bus is routine, not a
    # fault; only a servo that stays silent is worth aborting a wizard over.
    for attempt in range(attempts):
        value = ft_serial.ft_read(bus, servo_id, address, length)
        if value is not None:
            return int.from_bytes(value, "little")
        time.sleep(0.02 * (attempt + 1))
    raise ConfigurationError(
        f"cannot read servo {servo_id} register {address} after {attempts} attempts"
    )


def _positions(bus):
    values = np.array([_read(bus, i, 56) for i in range(1, 7)], dtype=int)
    if np.any(values < 0) or np.any(values > 4095):
        raise ConfigurationError("expected single-turn SO101 positions in 0..4095")
    return values


def _stable_positions(bus, *, gripper_only=False):
    while True:
        samples = []
        for _ in range(10):
            samples.append(_positions(bus))
            time.sleep(0.03)
        samples = np.array(samples)
        # Ignore isolated encoder outliers; assess only the joint being captured.
        spread = np.percentile(samples, 90, axis=0) - np.percentile(samples, 10, axis=0)
        checked = spread[5:] if gripper_only else spread
        if np.all(checked <= 24):
            return np.median(samples, axis=0)
        print("Movement detected during capture. Recorded travel is retained.")
        input(
            "Hold the gripper steady and press Enter to retry: "
            if gripper_only
            else "Support the arm steadily and press Enter to retry: "
        )


def _gripper_range(closed, opened, swept_span):
    span = abs(opened - closed)
    if span < 100 or not 0.7 * swept_span <= span <= 1.3 * swept_span:
        raise ConfigurationError(
            f"captured gripper travel is {span:.0f} counts (sweep {swept_span:.0f}); "
            "capture fully closed and fully open again"
        )
    # Explicit endpoints define the usable range; the sweep is a plausibility check.
    return min(closed, opened), max(closed, opened)


def build_profile(minimum, maximum, home, homing_offsets, *, gripper_closed=None):
    """Produce native instances from raw ranges; pure and independently testable."""
    arrays = [np.asarray(value, dtype=float) for value in (minimum, maximum, home)]
    if any(a.shape != (6,) or not np.all(np.isfinite(a)) for a in arrays):
        raise ConfigurationError("calibration requires six finite positions per sample")
    low, high, home = arrays
    if np.any(low < 0) or np.any(high > 4095) or np.any(low >= high):
        raise ConfigurationError("invalid encoder ranges; expected increasing bounds in 0..4095")
    if np.any(home < low) or np.any(home > high):
        raise ConfigurationError("home pose must lie inside the recorded travel ranges")
    if len(homing_offsets) != 6:
        raise ConfigurationError("need firmware homing-offset readings for all six motors")
    closed = low[5] if gripper_closed is None else float(gripper_closed)
    if not np.isfinite(closed) or min(abs(closed - low[5]), abs(closed - high[5])) > 20:
        raise ConfigurationError("closed gripper must match a recorded travel endpoint")
    closed_at_high = abs(closed - high[5]) < abs(closed - low[5])
    assets = resolve_model_assets("SO101", effector_model="E_SO101")
    model = json.loads(assets.model_config.read_text())
    arm = json.loads(assets.instance_config.read_text())
    effector = json.loads(assets.effector_instance_config.read_text())
    motors = {}
    for i, name in enumerate(NAMES):
        direction = 1
        if i < 5:
            servo = model["joints"][i]["servos"][0]
            model_low, model_high = servo["pos_min"], servo["pos_max"]
            span = (high[i] - low[i]) * TICK
            if not 0.8 * (model_high - model_low) <= span <= 1.25 * (model_high - model_low):
                raise ConfigurationError(
                    f"{name}: measured travel {math.degrees(span):.1f} deg does not match "
                    "the stock SO101 range; sweep its full travel without forcing stops. "
                    "If the encoder wraps, a firmware midpoint calibration is needed first."
                )
            # Align mechanical midpoint to the model midpoint, preserving scale.
            center = (low[i] + high[i]) / 2
            zero = (center - 2048) * TICK - (model_low + model_high) / 2
            q_low = max(model_low, (low[i] - 2048) * TICK - zero)
            q_high = min(model_high, (high[i] - 2048) * TICK - zero)
            joint = arm["joints"][i]
        else:
            if high[i] - low[i] < 100:
                raise ConfigurationError(
                    "gripper travel is too small; record fully closed and open"
                )
            direction = -1 if closed_at_high else 1
            zero = ((high[i] if closed_at_high else low[i]) - 2048) * TICK
            q_low, q_high = 0.0, (high[i] - low[i]) * TICK
            joint = effector["joints"][0]
        home_q = ((home[i] - 2048) * TICK - zero) * direction
        if not q_low <= home_q <= q_high:
            raise ConfigurationError(f"{name}: choose a home pose within the model's joint limits")
        joint["servos"][0].update(
            dir_invert=direction,
            zero_pos=float(zero),
            home_pos=float(home_q),
            pos_min=float(q_low),
            pos_max=float(q_high),
        )
        motors[name] = {
            "id": i + 1,
            "range_min": float(low[i]),
            "range_max": float(high[i]),
            "home_steps": float(home[i]),
            "homing_offset_raw": int(homing_offsets[i]),
        }
    motors["gripper"]["closed_steps"] = float(high[5] if closed_at_high else low[5])
    return {
        "version": 1,
        "model": "SO101",
        "motors": motors,
        "arm_instance": arm,
        "effector_instance": effector,
    }


def load_profile(path):
    try:
        profile = json.loads(Path(path).read_text())
        if profile["version"] != 1 or profile["model"] != "SO101":
            raise ValueError("unsupported profile version/model")
        records = [profile["motors"][name] for name in NAMES]
        if [r["id"] for r in records] != list(range(1, 7)):
            raise ValueError("expected motor IDs 1..6")
        # Regenerate instances from measurements, rather than trusting arbitrary
        # edited offsets/limits embedded in the file.
        return build_profile(
            [r["range_min"] for r in records],
            [r["range_max"] for r in records],
            [r["home_steps"] for r in records],
            [r["homing_offset_raw"] for r in records],
            gripper_closed=records[5].get("closed_steps", records[5]["range_min"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise ConfigurationError(f"invalid SO101 calibration {path}: {err}") from err


def apply_profiles(rig, paths):
    unknown = set(paths) - set(rig.names)
    if unknown:
        raise ConfigurationError(f"calibration names unknown arms: {sorted(unknown)}")
    arms = []
    for arm in rig.arms:
        if arm.name not in paths:
            arms.append(arm)
            continue
        if arm.model != "SO101":
            raise ConfigurationError("--calibration currently supports SO101 only")
        path = Path(paths[arm.name]).expanduser().resolve()
        profile = load_profile(path)
        # Generated artifacts live next to the profile, outside packaged models.
        arm_file = path.with_name(path.stem + ".arm.json")
        effector_file = path.with_name(path.stem + ".gripper.json")
        arm_file.write_text(json.dumps(profile["arm_instance"], indent=2) + "\n")
        effector_file.write_text(json.dumps(profile["effector_instance"], indent=2) + "\n")
        arms.append(
            dataclasses.replace(
                arm,
                instance_config=arm_file,
                effector_instance_config=effector_file,
                calibration_file=path,
            )
        )
    return dataclasses.replace(rig, arms=tuple(arms))


def verify_firmware(rig):
    for arm in rig.arms:
        if arm.calibration_file is None:
            continue
        profile = load_profile(arm.calibration_file)
        with _open_bus(arm.interface) as bus:
            for name, motor in profile["motors"].items():
                if _read(bus, motor["id"], 31) != motor["homing_offset_raw"]:
                    raise ConfigurationError(
                        f"{arm.name}/{name}: firmware homing offset changed since calibration; "
                        "recalibrate before running teleop"
                    )


def center_encoders(bus, output):
    """Save the prior registers, then use the driver's acknowledged middle calibration."""
    output = Path(output)
    before = {
        "interface": str(bus.port),
        "motors": {
            name: {
                "id": i + 1,
                "homing_offset_raw": _read(bus, i + 1, 31),
                "min_position_limit_raw": _read(bus, i + 1, 9),
                "max_position_limit_raw": _read(bus, i + 1, 11),
            }
            for i, name in enumerate(NAMES)
        },
    }
    backup = output.with_name(f"{output.stem}.firmware-before-{time.time_ns()}.json")
    backup.parent.mkdir(parents=True, exist_ok=True)
    with backup.open("x") as file:
        json.dump(before, file, indent=2)
        file.write("\n")
    print(f"Previous firmware registers saved to {backup}", flush=True)
    for i, name in enumerate(NAMES, 1):
        error = ft_serial.set_zero(bus, i)
        # Calibrate-middle is a special torque-register command. Explicitly
        # keep torque off even on firmware variants with different readback.
        disabled = ft_serial.torque_enable(bus, i, False)
        if error or not disabled or _read(bus, i, 40, 1) != 0:
            raise ConfigurationError(
                f"{name}: encoder centering failed ({error or 'torque-off verification'}). "
                f"Calibration stopped; some offsets may have changed. Backup: {backup}"
            )
        position = _read(bus, i, 56)
        if abs(position - 2048) > 20:
            raise ConfigurationError(
                f"{name}: expected centered reading near 2048, got {position}. "
                f"Do not run teleop with an older profile. Backup: {backup}"
            )
    return [_read(bus, i, 31) for i in range(1, 7)]


POLICY_FRAME_FORMAT = "karma_policy_frame.v1"
_LEROBOT_QUARTER_TURN_STEPS = 1024
_DEG_PER_STEP = 360.0 / 4096


def build_policy_frame(profile, zero_steps, rotated_steps):
    """Affine map from Karma's SO101 policy wire to the LeRobot v1 degree frame.

    The SO100/SO101 MolmoAct2 checkpoints were trained on LeRobot v1 datasets:
    joints in degrees, zero at the "zero position" (arm straight out
    horizontally, gripper up and closed), each joint +90 at the "rotated
    position"; the gripper is 0 closed and 100 at its rotated (open) position.
    Karma's wire is calibrated radians with a mid-range zero and the gripper as
    0=open/1=closed. Both are affine in raw encoder steps, so two captured
    poses and the calibration profile fix ``policy = scale * wire + offset``
    per joint exactly as LeRobot's own calibration would have.
    """
    zero = np.asarray(zero_steps, dtype=float)
    rotated = np.asarray(rotated_steps, dtype=float)
    if zero.shape != (6,) or rotated.shape != (6,):
        raise ConfigurationError("policy frame needs six positions per captured pose")
    if np.any(np.abs(rotated - zero) < 200):
        still = [n for n, z, r in zip(NAMES, zero, rotated, strict=True) if abs(r - z) < 200]
        raise ConfigurationError(
            "these joints barely moved between the zero and rotated poses: "
            + ", ".join(still)
            + "; every joint (and the gripper) must turn about a quarter turn"
        )
    scale, offset = [], []
    for i in range(6):
        if i < 5:
            servo = profile["arm_instance"]["joints"][i]["servos"][0]
            # Karma: rad = ((steps - 2048) * TICK - zero_pos) * dir_invert
            k_a = servo["dir_invert"] * TICK
            k_b = (-2048 * TICK - servo["zero_pos"]) * servo["dir_invert"]
            # LeRobot: drive so that rotated - zero is +90 deg; homing so that
            # the rotated pose reads exactly 90 (as run_arm_manual_calibration).
            drive = 1.0 if rotated[i] > zero[i] else -1.0
            l_a = drive * _DEG_PER_STEP
            l_b = 90.0 - l_a * rotated[i]
        else:
            motor = profile["motors"]["gripper"]
            span = motor["range_max"] - motor["range_min"]
            closed = motor["closed_steps"]
            direction = 1.0 if closed == motor["range_min"] else -1.0
            # Karma wire gripper: 0 open .. 1 closed, over the calibrated travel.
            k_a = -direction / span
            k_b = 1.0 + direction * closed / span
            # LeRobot LINEAR: 0 at the zero (closed) pose, 100 at the rotated one.
            l_a = 100.0 / (rotated[i] - zero[i])
            l_b = -l_a * zero[i]
        # Eliminate steps: policy = l_a/k_a * wire + (l_b - l_a/k_a * k_b).
        a = l_a / k_a
        scale.append(float(a))
        offset.append(float(l_b - a * k_b))
    return {
        "format": POLICY_FRAME_FORMAT,
        "model": "SO101",
        "target": "lerobot_v1_degrees",
        "names": list(NAMES),
        "scale": scale,
        "offset": offset,
        "zero_steps": zero.tolist(),
        "rotated_steps": rotated.tolist(),
    }


def run_policy_frame_capture(interface, calibration, output):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ConfigurationError(f"{output} already exists; choose a new output to preserve it")
    profile = load_profile(calibration)
    print("SO101 policy frame: maps Karma's calibrated radians onto the LeRobot v1")
    print("degree frame the SO100/SO101 MolmoAct2 checkpoints were trained in.")
    print("Stop all robot programs. Support the arm; torque will be disabled.")
    print("Reference photos: https://github.com/huggingface/lerobot/tree/"
          "42bf1e8b9df3d38b3898b24e46b5e0386910c466/media/so100 (follower_zero, follower_rotated)")
    input("Press Enter when the arm is supported and ready: ")
    with _open_bus(interface) as bus:
        _positions(bus)
        for i in range(1, 7):
            if not ft_serial.torque_enable(bus, i, False) or _read(bus, i, 40, 1) != 0:
                raise ConfigurationError(f"could not disable torque on servo {i}")
        print("Torque off.")
        print("ZERO pose: whole arm straight out horizontally from the base, forearm in")
        print("line with the upper arm, wrist straight, gripper jaws pointing UP and CLOSED,")
        print("wrist roll and base pan centred. Hold it steady.")
        input("Press Enter to capture the zero pose: ")
        zero = _stable_positions(bus)
        print("ROTATED pose: turn every joint a quarter turn from zero, as in the photo:")
        print("pan 90 deg to the left (seen from above), upper arm straight UP, forearm")
        print("horizontal forward, gripper pointing DOWN, wrist roll 90 deg, jaws OPEN.")
        input("Press Enter to capture the rotated pose: ")
        rotated = _stable_positions(bus)
    frame = build_policy_frame(profile, zero, rotated)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frame, indent=2) + "\n")
    print(f"wrote {output}")
    for name, a, b in zip(NAMES, frame["scale"], frame["offset"], strict=True):
        print(f"  {name:<14} policy = {a:+.3f} * wire {b:+.2f}")
    print("Pass it to inference/rollout/hitl with --policy-frame.")
    return 0


def load_policy_frame(path):
    try:
        data = json.loads(Path(path).read_text())
        if data["format"] != POLICY_FRAME_FORMAT:
            raise ValueError("unsupported policy frame format")
        scale = [float(v) for v in data["scale"]]
        offset = [float(v) for v in data["offset"]]
        if len(scale) != 6 or len(offset) != 6 or any(v == 0 for v in scale):
            raise ValueError("expected six non-zero scales and six offsets")
        if not all(math.isfinite(v) for v in scale + offset):
            raise ValueError("non-finite values")
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise ConfigurationError(f"invalid policy frame {path}: {err}") from err
    return tuple(scale), tuple(offset)


def run_calibration(interface, output, *, center=False):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ConfigurationError(f"{output} already exists; choose a new output to preserve it")
    print("SO101 calibration: stock follower, IDs 1..6, 1 Mbps.")
    print("Stop all robot programs. Support the arm; torque will be disabled.")
    if center:
        print("Encoder centering will change firmware homing offsets; old profiles become stale.")
        print("Existing homing/limit registers will be backed up before any EEPROM write.")
    else:
        print("No firmware zeros or EEPROM limits will be written.")
    input("Press Enter when the arm is supported and ready: ")
    with _open_bus(interface) as bus:
        # Identify/read every motor before changing even torque state.
        offsets = [_read(bus, i, 31) for i in range(1, 7)]
        _positions(bus)
        for i in range(1, 7):
            if not ft_serial.torque_enable(bus, i, False) or _read(bus, i, 40, 1) != 0:
                raise ConfigurationError(f"could not disable torque on servo {i}")
        if center:
            print("Torque off. Put each joint near the middle of its travel, including")
            print("the gripper halfway open. Align wrist roll to its intended middle orientation.")
            input("Keep the arm supported. Press Enter to set firmware encoder midpoints: ")
            _stable_positions(bus)
            offsets = center_encoders(bus, output)
            print("Encoders centered. Sweep shoulder, elbow, wrist FLEX and gripper by hand.")
            print("Do NOT sweep wrist ROLL: its calibrated single-turn range is already known.")
        else:
            print("Torque off. Move all SIX joints through their full travel by hand.")
            print("Include wrist roll and fully closed/open gripper.")
        print("Do not force the stops. Gripper must be moved fully closed and fully open.")
        print("Press Enter after completing the sweep.")
        previous = _positions(bus)
        low, high = previous.copy(), previous.copy()
        if center:
            low[4], high[4] = 0, 4095
        last_print = 0.0
        while not select.select([sys.stdin], [], [], 0)[0]:
            current = _positions(bus)
            wrapped = np.abs(current - previous) > 2048
            if center:
                wrapped[4] = False
            if np.any(wrapped):
                raise ConfigurationError(
                    "encoder wrapped during sweep in "
                    + ", ".join(np.array(NAMES)[wrapped])
                    + "; rerun with --center-encoders and position each joint near its "
                    "middle before the centering step"
                )
            low, high = np.minimum(low, current), np.maximum(high, current)
            previous = current
            if time.monotonic() - last_print > 1:
                print(
                    "  travel counts: "
                    + " ".join(
                        f"{n}={hi - lo}" for n, lo, hi in zip(NAMES, low, high, strict=True)
                    ),
                    flush=True,
                )
                last_print = time.monotonic()
            time.sleep(0.03)
        input()
        # Check sweep completeness before asking the user for a home pose.
        build_profile(low, high, (low + high) / 2, offsets)
        print("Travel recorded. Next is the CLOSED-JAW check, not home capture.")
        print("Your saved home can have the gripper open; choose it after this check.")
        while True:
            input("Gently close only the gripper fully (jaws together), then press Enter: ")
            closed = _stable_positions(bus, gripper_only=True)[5]
            input("Now fully OPEN the gripper, then press Enter: ")
            opened = _stable_positions(bus, gripper_only=True)[5]
            try:
                low[5], high[5] = _gripper_range(closed, opened, high[5] - low[5])
                break
            except ConfigurationError as err:
                print(f"{err}. Travel is retained.")
        print("Place the arm AND gripper in your desired home pose; support it steadily.")
        while True:
            input("Press Enter to capture home: ")
            home = _stable_positions(bus)
            # A repeated endpoint capture can differ by a few encoder counts.
            # Snap only small measurement differences, never widen travel limits.
            near_range = (home >= low - 24) & (home <= high + 24)
            home = np.where(near_range, np.clip(home, low, high), home)
            try:
                profile = build_profile(low, high, home, offsets, gripper_closed=closed)
                break
            except ConfigurationError as err:
                print(f"{err}. Travel is retained; adjust home and retry.")
        profile["interface_at_capture"] = interface
        print(
            "Home joint angles (deg):",
            [
                round(math.degrees(j["servos"][0]["home_pos"]), 2)
                for j in profile["arm_instance"]["joints"]
            ],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally overwriting an existing profile.
        with output.open("x") as file:
            json.dump(profile, file, indent=2)
            file.write("\n")
    print(f"Saved {output}. Torque remains off.")
    print(f"Use --calibration ARM={output} with karma teleop; replace ARM with left or right.")
    return 0
