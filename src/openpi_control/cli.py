"""Operator CLI: preflight, maintain, drive, and infer on connected arms.

Commands aimed at the jobs around making an arm usable:

``doctor``
    Read-only checks: do the packaged assets resolve, is every servo model
    known to the driver registry, does the bus interface exist and is it up at
    the baud rate the model wants. Static by default; ``--probe`` additionally
    opens the bus and listens.

``zero``
    Write the arm's current physical pose into each servo's firmware as its
    zero reference, via the per-family drivers in :mod:`openpi_control.servos`.
    Destructive and pose-dependent, so it confirms before touching anything.

``live``
    Bring a whole rig up, mirror it in the browser -- arms and cameras on the
    one page -- then park it and put it back down. This is the one command here
    that energizes an arm, and it owns that lifecycle start to finish:
    ``pi_control_node`` dies with its parent process, so an arm cannot outlive
    the command that powered it on.

``cameras``
    Resolve the rig's cameras to device paths and say which ones are actually
    on the bus. Read-only; ``--probe`` additionally opens each stream and
    grabs a frame, and ``--snapshot`` writes those frames out so you can check
    where a wrist camera is pointing without a headset.

``infer``
    Run the bimanual YAM MolmoAct2 HTTP client, execute bounded action chunks,
    and show measured poses plus predicted end-effector trails in Viser.

``rollout``
    Run timed, interactive MolmoAct2 episodes, save every completed or
    Ctrl-C-interrupted episode as LeRobot v3, and collect a terminal
    success/failure label for each attempt.

All of them attach :func:`openpi_control.runlog.setup_run_logging`, so every run
leaves a trace under ``~/openpi-data/logs/runtime/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from . import cameras as cameras_mod
from . import meshes, runlog
from .config import (
    SUPPORTED_EFFECTORS,
    SUPPORTED_MODELS,
    ArmConfig,
    connection_for_interface,
    resolve_model_assets,
)
from .exceptions import ConfigurationError, PiControlError
from .inference import (
    DEFAULT_CHUNK_SPEED,
    DEFAULT_MAX_EFFECTOR_STEP,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_MOLMOACT_JPEG_QUALITY,
    DEFAULT_MOLMOACT_NUM_STEPS,
    DEFAULT_PREFETCH_MARGIN_S,
    DEFAULT_REQUEST_TIMEOUT_S,
    MOLMOACT_ACTION_HORIZON,
    SUB_STEP_PERIOD_S,
    BoundedChunkExecutor,
    ChunkPrefetcher,
    EncodedFramesUnsupported,
    GripperWatch,
    InferenceError,
    MolmoActClient,
    ReachingChunkExecutor,
    build_observation,
    command_lag,
    served_frames,
    split_chunk,
    start_pose_plan,
    time_scale,
)
from .inference_record import InferenceRolloutSource
from .rigs import Rig, RigArm, resolve_rig, rig_names
from .servos import SERVO_ZERO_DRIVERS, buses

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from .arms import FollowerArm, LeaderArm
    from .backend import ArmBackend
    from .session import ArmSession
    from .types import PositionCommand

_OK = "ok"
_WARN = "warn"
_FAIL = "fail"

_MARKS = {_OK: "OK  ", _WARN: "WARN", _FAIL: "FAIL"}

# How long --probe listens for traffic before giving up. Silence is not proof of
# a fault (DM servos answer requests rather than broadcasting), so this only
# ever reports what it saw.
_PROBE_WINDOW_S = 1.5

# How often ``live`` pushes each arm's newest pose into the browser. The native
# node publishes faster than this; the browser has no use for more.
_MIRROR_RATE_HZ = 30.0

# How long a camera gets to deliver its first frame before ``--probe`` calls it
# dead; how long the stream is then left alone to settle; and how long the
# probe counts frames to measure the real rate. A D405 needs a few hundred ms
# after the first frame before it is streaming at rate, and measuring across
# that ramp reports ~10 fps for a camera that is in fact doing 28.
_CAMERA_WARMUP_S = 5.0
_CAMERA_SETTLE_S = 1.0
_CAMERA_COUNT_S = 2.0

# Fraction of the requested rate a camera has to hit before the probe stops
# complaining. Loose on purpose: v4l2 delivers 27-28 of a requested 30 on a
# healthy bus, while a camera sharing a USB 2 hub with two others lands far
# below this.
_CAMERA_FPS_TOLERANCE = 0.8


@dataclass(frozen=True, slots=True)
class ServoEntry:
    """One servo declared by an arm or effector model catalog."""

    source: str  # "arm" or the effector model name
    joint_id: int
    servo_model: str
    servo_id: int
    driver: ModuleType | None

    @property
    def read_only(self) -> bool:
        """True for encoders whose zero is fixed in hardware — nothing to write."""
        return self.driver is None

    def label(self) -> str:
        return f"joint {self.joint_id} ({self.source})"


@dataclass(frozen=True, slots=True)
class CheckResult:
    status: str
    name: str
    detail: str

    def render(self) -> str:
        return f"  [{_MARKS[self.status]}] {self.name:<26} {self.detail}"


def servo_entries(config_path: Path, source: str) -> list[ServoEntry]:
    """Every servo a device catalog declares, in joint order."""
    data = json.loads(config_path.read_text())
    entries: list[ServoEntry] = []
    for index, joint in enumerate(data.get("joints", []), start=1):
        joint_id = int(joint.get("joint_id", index))
        for servo in joint.get("servos", []):
            servo_model = str(servo.get("servo_model", ""))
            if not servo_model:
                raise ConfigurationError(
                    f"{config_path.name} joint {joint_id} has a servo with no servo_model"
                )
            if servo_model not in SERVO_ZERO_DRIVERS:
                raise ConfigurationError(
                    f"servo model {servo_model!r} ({config_path.name} joint {joint_id}) is not in "
                    "the servo registry (openpi_control/servos): add it to SERVO_ZERO_DRIVERS"
                )
            entries.append(
                ServoEntry(
                    source=source,
                    joint_id=joint_id,
                    servo_model=servo_model,
                    servo_id=int(servo.get("servo_id", joint_id)),
                    driver=SERVO_ZERO_DRIVERS[servo_model],
                )
            )
    return entries


def build_plan(model: str, effector_model: str | None = None) -> tuple[ServoEntry, ...]:
    """The servos of an arm plus its effector, ordered by joint id."""
    assets = resolve_model_assets(model, effector_model=effector_model)
    entries = servo_entries(assets.model_config, "arm")
    if assets.effector_model_config is not None and effector_model is not None:
        entries += servo_entries(assets.effector_model_config, effector_model)
    if not entries:
        raise ConfigurationError(f"model {model!r} declares no servos")
    return tuple(sorted(entries, key=lambda entry: entry.joint_id))


def plan_port_type(plan: tuple[ServoEntry, ...]) -> str | None:
    """The single bus type the writable servos need, or None when all are read-only."""
    port_types = {entry.driver.PORT_TYPE for entry in plan if entry.driver is not None}
    if not port_types:
        return None
    if len(port_types) > 1:
        raise ConfigurationError(
            "this arm's servos span more than one bus type "
            f"({', '.join(sorted(port_types))}); zero them one bus at a time"
        )
    return port_types.pop()


def _can_sysfs(interface: str, leaf: str) -> str | None:
    path = pathlib.Path(f"/sys/class/net/{interface}/{leaf}")
    try:
        return path.read_text().strip()
    except OSError:
        return None


def can_bitrate(interface: str) -> int | None:
    """The configured bitrate of a SocketCAN interface, or None if unknown.

    Two sources, because neither is universal. ``/sys/class/net/<if>/
    can_bittiming/bitrate`` is the cheap one, but it does not exist on every
    kernel -- it is absent on 6.8 here, which made this check warn "interface
    does not report one" on every single run of a correctly configured 1 Mbit
    bus. A preflight that always warns is one nobody reads, so fall back to
    ``ip``, which reads the same value over netlink and is where ``ip -d link
    show`` gets it from.
    """
    from_sysfs = _can_sysfs(interface, "can_bittiming/bitrate")
    if from_sysfs:
        try:
            return int(from_sysfs)
        except ValueError:
            pass
    try:
        completed = subprocess.run(
            ["ip", "-d", "-json", "link", "show", interface],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        links = json.loads(completed.stdout)
        bitrate = links[0]["linkinfo"]["info_data"]["bittiming"]["bitrate"]
    except (json.JSONDecodeError, LookupError, TypeError):
        return None
    return int(bitrate) if bitrate else None


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def run_doctor(
    model: str,
    interface: str,
    *,
    effector_model: str | None = None,
    probe: bool = False,
) -> list[CheckResult]:
    """Collect the preflight checks. Opens no bus unless ``probe`` is set."""
    results: list[CheckResult] = []

    try:
        assets = resolve_model_assets(model, effector_model=effector_model)
    except ConfigurationError as err:
        return [CheckResult(_FAIL, "packaged assets", str(err))]
    results.append(
        CheckResult(
            _OK,
            "packaged assets",
            f"{assets.model_config.name}, {assets.instance_config.name}"
            + (f", {assets.effector_model_config.name}" if assets.effector_model_config else ""),
        )
    )
    results.append(
        CheckResult(
            _OK if assets.urdf is not None else _WARN,
            "urdf",
            assets.urdf.name
            if assets.urdf is not None
            else "none packaged — gravity compensation needs one (FR3 uses its controller)",
        )
    )

    try:
        plan = build_plan(model, effector_model)
    except ConfigurationError as err:
        results.append(CheckResult(_FAIL, "servo registry", str(err)))
        return results

    writable = [entry for entry in plan if not entry.read_only]
    read_only = [entry for entry in plan if entry.read_only]
    detail = f"{len(plan)} servos, all models known"
    if read_only:
        detail += f" ({len(read_only)} read-only: " + ", ".join(
            f"joint {entry.joint_id}" for entry in read_only
        ) + ")"
    results.append(CheckResult(_OK, "servo registry", detail))

    try:
        port_type = plan_port_type(plan)
    except ConfigurationError as err:
        results.append(CheckResult(_FAIL, "bus type", str(err)))
        return results
    if port_type is None:
        results.append(
            CheckResult(_WARN, "bus type", "every servo is read-only; nothing to zero")
        )
        return results
    results.append(CheckResult(_OK, "bus type", port_type))

    # The connection type the interface string implies must match what the
    # servos need, or the arm is configured against the wrong transport.
    connection = connection_for_interface(interface)
    results.append(CheckResult(_OK, "connection", type(connection).__name__))

    problem = buses.check_interface(port_type, interface)
    results.append(
        CheckResult(
            _FAIL if problem else _OK,
            f"interface {interface}",
            problem or "present",
        )
    )

    if port_type == buses.PORT_TYPE_CAN and problem is None:
        state = _can_sysfs(interface, "operstate")
        results.append(
            CheckResult(
                _OK if state == "up" else _FAIL,
                "link state",
                state or "unknown",
            )
        )
        want = _catalog_baudrate(assets.model_config)
        have = can_bitrate(interface)
        if want is None:
            results.append(CheckResult(_WARN, "bitrate", "model declares no catalog baudrate"))
        elif have is None:
            results.append(
                CheckResult(_WARN, "bitrate", f"model wants {want}; interface does not report one")
            )
        else:
            results.append(
                CheckResult(
                    _OK if have == want else _FAIL,
                    "bitrate",
                    f"{have} (model wants {want})",
                )
            )

    if port_type == buses.PORT_TYPE_SERIAL:
        want = _catalog_baudrate(assets.model_config)
        results.append(
            CheckResult(
                _OK if want else _FAIL,
                "baudrate",
                str(want) if want else "serial buses need a catalog baudrate",
            )
        )

    # An ArmConfig is what a session would be handed; building one here catches
    # model/connection mismatches before any hardware is involved.
    try:
        ArmConfig("preflight", model, connection, effector_model=effector_model)
        results.append(CheckResult(_OK, "arm config", "valid"))
    except ConfigurationError as err:
        results.append(CheckResult(_WARN, "arm config", str(err)))

    cached = meshes.cached_mesh_dir(model)
    results.append(
        CheckResult(
            _OK if cached else _WARN,
            "visual meshes",
            str(cached)
            if cached
            else f"not cached — openpi-control-viz --fetch-meshes --model {model}",
        )
    )

    if probe:
        results.append(_probe_bus(port_type, interface, writable, assets.model_config))
    return results


def _catalog_baudrate(model_config: Path) -> int | None:
    data = json.loads(model_config.read_text())
    value = data.get("catalog", {}).get("baudrate")
    return int(value) if isinstance(value, int) and value > 0 else None


def _probe_bus(
    port_type: str, interface: str, writable: list[ServoEntry], model_config: Path
) -> CheckResult:
    """Open the bus and listen. Reads only — never sends."""
    expected = tuple(entry.servo_id for entry in writable)
    try:
        with buses.open_bus(
            port_type, interface, baudrate=_catalog_baudrate(model_config)
        ) as bus:
            if port_type != buses.PORT_TYPE_CAN:
                return CheckResult(_OK, "bus probe", f"{port_type} session opened")
            answered = buses.recv_from(bus, expected, _PROBE_WINDOW_S)
    except Exception as err:  # noqa: BLE001 - surface any driver error as a check
        return CheckResult(_FAIL, "bus probe", f"{type(err).__name__}: {err}")
    if answered:
        return CheckResult(_OK, "bus probe", "traffic seen from a configured servo id")
    # DM servos answer requests rather than broadcasting, so quiet is normal on
    # an idle bus. Reporting this as a failure would cry wolf.
    return CheckResult(
        _WARN,
        "bus probe",
        f"opened, no traffic in {_PROBE_WINDOW_S}s — normal on an idle bus",
    )


# --------------------------------------------------------------------------- #
# cameras
# --------------------------------------------------------------------------- #


def run_camera_checks(
    rig: Rig,
    *,
    overrides: dict[str, str] | None = None,
    required: bool = False,
) -> list[CheckResult]:
    """One check per rig camera: did its serial resolve to a device path?

    A camera that is not on the bus is a warning, not a failure, because none
    of them are needed to drive an arm -- ``doctor`` should not refuse to
    green-light a bimanual cell because someone unplugged a wrist camera. The
    dataset recorder passes ``required=True``, where a missing camera really is
    fatal: it would otherwise write an episode with a view silently absent.
    """
    mark = _FAIL if required else _WARN
    results: list[CheckResult] = []
    if not rig.cameras:
        return [CheckResult(_OK, "cameras", "this rig declares none")]

    result = cameras_mod.discover(rig.cameras, overrides=overrides)
    for camera in rig.cameras:
        found = result.matched.get(camera.name)
        if found is None:
            results.append(
                CheckResult(
                    mark,
                    f"camera {camera.name}",
                    f"serial {camera.serial} not on the bus"
                    if camera.name not in (overrides or {})
                    else f"pinned device {(overrides or {})[camera.name]} does not exist",
                )
            )
            continue
        detail = f"{camera.width}x{camera.height}@{camera.fps}"
        if camera.rotate:
            detail += f" rot{camera.rotate}"
        detail += f" — {found.device}"
        if found.overridden:
            detail += " (pinned)"
        results.append(CheckResult(_OK, f"camera {camera.name}", detail))

    if result.unclaimed:
        # A camera on the bus that no rig claims is usually the interesting
        # half of "the top view is missing": it was swapped, and the new serial
        # needs to go into the rig.
        results.append(
            CheckResult(
                _WARN,
                "unclaimed cameras",
                f"serial(s) {', '.join(result.unclaimed)} present but in no rig camera — "
                "add them to YAM_BIMANUAL_CAMERA_SERIALS in rigs.py",
            )
        )
    return results


def check_camera_modes(rig: Rig) -> list[CheckResult]:
    """Can each camera actually deliver the mode the run is asking for?

    Kept out of :func:`run_camera_checks` because answering it needs the
    RealSense SDK, and ``doctor`` must keep working on a box without it. This
    only enumerates profiles -- it opens no stream -- so it is cheap enough to
    sit in a preflight, and it turns "you asked for 45 fps" into the list of
    rates the device has rather than a bare ioctl failure once the arms are
    already energized.
    """
    results: list[CheckResult] = []
    for camera in rig.cameras:
        label, size = f"mode {camera.name}", f"{camera.width}x{camera.height}"
        try:
            modes = cameras_mod.supported_color_modes(camera.serial)
        except ConfigurationError as err:
            # Not connected, or no SDK. Either way the presence checks already
            # said so; repeating it as a failure here would double-count.
            results.append(CheckResult(_WARN, label, str(err)))
            continue
        rates = modes.get((camera.width, camera.height))
        if rates is None:
            offered = ", ".join(f"{w}x{h}" for w, h in modes) or "nothing"
            detail = f"{size} is not offered; this camera has {offered}"
        elif camera.fps not in rates:
            detail = (
                f"{camera.fps} fps at {size} is not offered; this camera does "
                f"{', '.join(str(rate) for rate in rates)}"
            )
        else:
            results.append(
                CheckResult(
                    _OK,
                    label,
                    f"{size}@{camera.fps} {camera.pixel_format} "
                    f"(offers {', '.join(str(rate) for rate in rates)})",
                )
            )
            continue
        results.append(CheckResult(_FAIL, label, detail))
    return results


def probe_cameras(
    rig: Rig,
    *,
    overrides: dict[str, str] | None = None,
    snapshot_dir: Path | None = None,
    warmup_s: float = _CAMERA_WARMUP_S,
    settle_s: float = _CAMERA_SETTLE_S,
    count_s: float = _CAMERA_COUNT_S,
) -> list[CheckResult]:
    """Open each resolvable camera, grab frames, and report what came out.

    Opened one at a time on purpose. A v4l2 device opens once, so the useful
    error here is "this camera is held by something else", and holding all three
    at once would let the first camera's success hide the second's failure
    behind a shared USB bandwidth complaint.
    """
    results: list[CheckResult] = []
    discovery = cameras_mod.discover(rig.cameras, overrides=overrides)
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    for camera in rig.cameras:
        found = discovery.matched.get(camera.name)
        if found is None:
            results.append(
                CheckResult(_WARN, f"probe {camera.name}", "no device; nothing to open")
            )
            continue
        try:
            with cameras_mod.CameraReader(found.spec()) as reader:
                frame = reader.wait_for_frame(warmup_s)
                if frame is None:
                    results.append(
                        CheckResult(
                            _FAIL,
                            f"probe {camera.name}",
                            f"opened but delivered no frame in {warmup_s:g}s",
                        )
                    )
                    continue
                time.sleep(settle_s)
                measured = reader.measure_fps(count_s)
                fourcc, _, _, _ = reader.negotiated
                # Snapshot the freshest frame, not the warm-up one. `or` is not
                # an option here: a numpy array has no truth value.
                newest = reader.latest()
                if newest is not None:
                    frame = newest
                height, width = frame.shape[:2]
                detail = (
                    f"{width}x{height} {fourcc}, {measured:.0f} fps (asked {camera.fps})"
                )
                status = (
                    _OK if measured >= camera.fps * _CAMERA_FPS_TOLERANCE else _WARN
                )
                if snapshot_dir is not None:
                    path = snapshot_dir / f"{camera.name}.png"
                    cameras_mod.write_snapshot(path, frame)
                    detail += f" -> {path}"
                results.append(CheckResult(status, f"probe {camera.name}", detail))
        except Exception as err:  # noqa: BLE001 - surface any driver error as a check
            results.append(
                CheckResult(_FAIL, f"probe {camera.name}", f"{type(err).__name__}: {err}")
            )
    return results


# --------------------------------------------------------------------------- #
# zero
# --------------------------------------------------------------------------- #


def zero_arm(
    plan: tuple[ServoEntry, ...],
    port_type: str,
    interface: str,
    *,
    baudrate: int | None = None,
) -> list[tuple[ServoEntry, str | None]]:
    """Zero every writable servo in ``plan``; returns (entry, error or None) pairs.

    Read-only encoders are skipped rather than reported as failures, per the
    servo registry's contract. Whole-arm controllers zero every joint in one
    transaction, so they are called once.
    """
    outcomes: list[tuple[ServoEntry, str | None]] = []
    writable = [entry for entry in plan if not entry.read_only]
    if not writable:
        return outcomes

    whole_arm = [entry for entry in writable if getattr(entry.driver, "WHOLE_ARM_ZERO", False)]
    with buses.open_bus(port_type, interface, baudrate=baudrate) as bus:
        if whole_arm:
            driver = whole_arm[0].driver
            assert driver is not None
            error = driver.set_zero_whole_arm(bus)
            return [(entry, error) for entry in whole_arm]
        for entry in writable:
            driver = entry.driver
            assert driver is not None
            outcomes.append((entry, driver.set_zero(bus, entry.servo_id)))
    return outcomes


def _confirm(plan: tuple[ServoEntry, ...], model: str, interface: str) -> bool:
    writable = [entry for entry in plan if not entry.read_only]
    print(f"About to write a new firmware zero on {model} via {interface}:")
    for entry in writable:
        print(f"  {entry.label():<24} {entry.servo_model:<24} servo id {entry.servo_id}")
    skipped = [entry for entry in plan if entry.read_only]
    for entry in skipped:
        print(f"  {entry.label():<24} {entry.servo_model:<24} read-only, skipped")
    print()
    print("The arm's CURRENT physical pose becomes zero for every servo listed.")
    print("Move it to the intended zero pose and support it before continuing.")
    for entry in writable:
        if entry.source == "arm":
            continue
        print()
        print(f"  {entry.label()} is the GRIPPER, and its zero is not a pose you pick:")
        print("  the stops are measured at every startup (E_Yam needs_calibration), so")
        print("  this write only shifts the frame that calibration then anchors. Zero it")
        print("  with the jaws shut if you zero it at all.")
    reply = input("Type 'zero' to proceed: ").strip()
    return reply == "zero"


# --------------------------------------------------------------------------- #
# live
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LiveArm:
    """One energized arm, paired with the rig entry that placed it."""

    rig_arm: RigArm
    arm: FollowerArm | LeaderArm

    @property
    def name(self) -> str:
        return self.rig_arm.name


def preflight_rig(rig: Rig) -> tuple[int, list[tuple[str, list[CheckResult]]]]:
    """Run the doctor checks for every arm in a rig. Returns (failures, per-arm).

    Nothing is energized until every arm passes, so a rig with one unplugged
    adapter does not come half up: bringing the good arm to a hold and then
    aborting would leave it stiff with no session left to put it down.
    """
    failures = 0
    reports: list[tuple[str, list[CheckResult]]] = []
    for rig_arm in rig.arms:
        results = run_doctor(
            rig_arm.model, rig_arm.interface, effector_model=rig_arm.effector_model
        )
        failures += sum(1 for result in results if result.status == _FAIL)
        reports.append((rig_arm.name, results))
    return failures, reports


def power_up(
    rig: Rig,
    *,
    float_mode: bool = False,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
) -> tuple[ArmSession, list[LiveArm]]:
    """Start one session and energize every arm in the rig.

    Connecting *is* the power-on: the native node comes up holding whatever pose
    it found, so a follower goes stiff here. ``float_mode`` swaps that for the
    gravity feed-forward, which leaves the arm backdrivable -- an explicit
    choice, never the default, because a compliant arm sags if its torq_rescale
    is untuned. Leaders float either way; that is what ``ArmSession.connect``
    already does for them.

    ``backend_factory`` exists so this can be driven against a stand-in backend
    in tests without a bus or a native node.
    """
    from .session import ArmSession

    session = ArmSession()
    entries: list[LiveArm] = []
    for rig_arm in rig.arms:
        config = rig_arm.arm_config()
        backend = backend_factory(rig_arm) if backend_factory is not None else None
        arm: FollowerArm | LeaderArm = (
            session.add_follower(config, backend=backend)
            if rig_arm.is_follower
            else session.add_leader(config, backend=backend)
        )
        entries.append(LiveArm(rig_arm, arm))

    try:
        session.connect()
        if float_mode:
            for entry in entries:
                if not entry.rig_arm.is_follower:
                    continue
                if not entry.arm.capabilities.supports_gravity_compensation:
                    print(f"  {entry.name}: node has no gravity float; left holding")
                    continue
                entry.arm.enter_gravity_compensation()
    except BaseException:
        # A partial bring-up must not survive: whatever did connect is already
        # energized, and this session is the only thing that can put it down.
        session.close()
        raise
    return session, entries


def power_down(session: ArmSession, live_arms: list[LiveArm], *, park: bool = True) -> int:
    """Park each arm at its home pose, then de-energize. Returns the failure count.

    Parking rides the native MOVE_TO_READY_AND_SHUTDOWN path, so the arm drives
    to the ``home_pos`` in its instance JSON and the node exits at the end of
    that move -- it is never dropped from wherever it happened to be standing.
    A node that does not advertise CAP_MOVE_TO_READY is closed in place, said
    out loud rather than silently.

    The park is a real move and takes seconds; each one reports how long it
    took, so an arm on its way to home_pos never reads as a hung shutdown. A
    ctrl-c during it is honored -- it just cannot be allowed to skip the
    de-energize below, which is the only thing that retires the nodes.
    """
    failures = 0
    if park:
        try:
            for entry in live_arms:
                try:
                    supported = entry.arm.capabilities.supports_move_to_ready
                except Exception as err:  # noqa: BLE001 - keep putting the rest down
                    failures += 1
                    print(f"  {entry.name}: cannot check the ready move: {err}")
                    continue
                if not supported:
                    print(f"  {entry.name}: node has no ready move; closing in place")
                    continue
                print(f"  parking {entry.name} -> home_pos ...", end="", flush=True)
                started = time.monotonic()
                try:
                    entry.arm.close(move_to_ready=True)
                except Exception as err:  # noqa: BLE001 - keep putting the rest down
                    failures += 1
                    print(f" FAILED: {type(err).__name__}: {err}")
                else:
                    print(f" done ({time.monotonic() - started:.1f}s)")
        except KeyboardInterrupt:
            # A second ctrl-c aborts the remaining parks, but the arms that are
            # still energized are this process's to put down: fall through to
            # session.close() instead of letting the interrupt unwind past it.
            failures += 1
            print("\n  park interrupted; de-energizing where the arms stand")

    # Always reached, parked or not: session.close() is what retires the nodes
    # and the sockets, and it is idempotent for an arm already closed above.
    try:
        session.close()
    except Exception as err:  # noqa: BLE001 - report, do not mask the shutdown
        failures += 1
        print(f"  power-down error: {type(err).__name__}: {err}")
    print("de-energized, with errors above" if failures else "de-energized")
    return failures


def open_preview_cameras(
    rig: Rig, *, overrides: dict[str, str] | None = None
) -> dict[str, cameras_mod.CameraReader]:
    """Open every camera of ``rig`` that can be opened, and say what was not.

    Deliberately not the recorder's all-or-none open: a dataset with a view
    silently missing is a corrupt dataset, but ``live`` exists to drive arms,
    and refusing to energize them because someone unplugged a wrist camera --
    or because another process is already streaming it -- would be the wrong
    trade. Every camera that opens is previewed; every one that does not is
    named.

    Opened one at a time for the same reason ``--probe`` does it: the useful
    error is "this camera is held by something else", and it has to name the
    camera it is actually about. Identical failures are collapsed into one
    line, because the common one -- no SDK installed -- is the same sentence
    for all three.
    """
    discovery = cameras_mod.discover(rig.cameras, overrides=overrides)
    for name, serial in discovery.missing.items():
        pinned = (overrides or {}).get(name)
        why = (
            f"pinned device {pinned} does not exist"
            if pinned
            else f"serial {serial} not on the bus"
        )
        print(f"  camera   {name:<12} not previewing — {why}")

    readers: dict[str, cameras_mod.CameraReader] = {}
    problems: dict[str, list[str]] = {}
    for name, found in discovery.matched.items():
        try:
            readers[name] = cameras_mod.CameraReader(found.spec())
        except (ConfigurationError, OSError) as err:
            problems.setdefault(str(err), []).append(name)
    for message, names in problems.items():
        print(f"  camera   {', '.join(names)} not previewing — {message}")
    return readers


def mirror(
    scene: object | None,
    live_arms: list[LiveArm],
    *,
    stop: threading.Event,
    rate_hz: float = _MIRROR_RATE_HZ,
    control: object | None = None,
    cameras: object | None = None,
) -> None:
    """Pump each arm's newest pose into the scene until ``stop`` is set.

    Reads ``latest_state`` rather than ``read_state`` on purpose: it does not
    block and does not raise on a briefly silent arm. A stale pose left on
    screen is the right failure mode here -- raising would tear down a session
    that is holding two energized arms.

    A ``control`` panel is stepped on this same clock rather than on a thread
    of its own, so the pose that is drawn and the pose that is commanded are
    always one tick apart at most, and the two can never interleave. A
    ``cameras`` panel rides the same clock for the same reason -- it throttles
    itself down to a preview rate internally, so the images and the poses share
    one websocket instead of racing for it.
    """
    period = 1.0 / rate_hz
    while not stop.is_set():
        if scene is not None:
            for entry in live_arms:
                state = entry.arm.latest_state
                if state is not None:
                    scene.update(entry.name, state.joints.position_rad)  # type: ignore[attr-defined]
        if control is not None:
            control.step(period)  # type: ignore[attr-defined]
        if cameras is not None:
            cameras.step(period)  # type: ignore[attr-defined]
        stop.wait(period)


def run_live(
    rig: Rig,
    *,
    float_mode: bool = False,
    park: bool = True,
    visualize: bool = True,
    control: bool = False,
    camera_preview: bool = True,
    camera_overrides: dict[str, str] | None = None,
    port: int = 8080,
    mesh_dir: Path | None = None,
    rate_hz: float = _MIRROR_RATE_HZ,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Power the rig up, mirror it, then park it and power it down.

    The whole lifecycle lives inside this call because it has to: the native
    node holds a liveness pipe to this process, so returning from here is what
    de-energizes the arms. There is no way to leave them up for a later command.
    """
    stop = stop if stop is not None else threading.Event()
    if control and not visualize:
        raise ConfigurationError("browser control needs the browser view; drop --no-viz")
    scene = None
    if visualize:
        # Imported here, not at module scope: doctor and zero must keep working
        # without the optional viz extra installed.
        from .viz import ArmSceneVisualizer

        scene = ArmSceneVisualizer.from_rig(rig, mesh_dir=mesh_dir, port=port)
        # Deliberately no add_gui(): viz's own sliders drive the render, and the
        # render here belongs to the hardware. Browser control is a different
        # surface -- see viser_control, where the sliders are targets instead.

    # Cameras before the motors, as in ``record``: opening them is what finds
    # out a camera is held by another process, and finding that out with two
    # arms already energized is worse. Unlike ``record`` it is not fatal here.
    camera_readers: dict[str, cameras_mod.CameraReader] = {}
    if scene is not None and camera_preview and rig.cameras:
        camera_readers = open_preview_cameras(rig, overrides=camera_overrides)

    mode = "gravity float (backdrivable)" if float_mode else "holding"
    try:
        session, live_arms = power_up(rig, float_mode=float_mode, backend_factory=backend_factory)
    except BaseException:
        # The capture threads are already running; nothing below will reach the
        # finally that would have stopped them.
        cameras_mod.close_readers(camera_readers)
        raise
    panel = None
    camera_panel = None
    failures = 0
    try:
        for entry in live_arms:
            print(
                f"  {entry.name:<8} {entry.rig_arm.model} on {entry.rig_arm.interface}"
                f" — energized, {mode}"
            )
        if control and scene is not None:
            from .viser_control import RigControlPanel

            followers = {
                entry.name: entry.arm for entry in live_arms if entry.rig_arm.is_follower
            }
            panel = RigControlPanel(scene, followers, float_mode=float_mode)  # type: ignore[arg-type]
            print(f"  control  {', '.join(followers)} — disarmed; arm each one in the browser")
        if camera_readers and scene is not None:
            from .viz import CameraPanel

            camera_panel = CameraPanel(scene.server, camera_readers)
            print(f"  cameras  {', '.join(camera_readers)} — live in the browser")
        if scene is not None:
            print(f"  viser    {scene.url}")
        print(f"ctrl-c to {'park at home_pos and ' if park else ''}power down")
        mirror(
            scene,
            live_arms,
            stop=stop,
            rate_hz=rate_hz,
            control=panel,
            cameras=camera_panel,
        )
    except KeyboardInterrupt:
        print()
    finally:
        # Disarm before anything else: the park below moves the arms, and it
        # must not race a panel that is still pushing targets at them.
        if panel is not None:
            panel.disarm_all("session ending")
        # Released before the park below, not after: parking takes seconds, and
        # a camera left held for them is a camera the next command cannot open.
        cameras_mod.close_readers(camera_readers)
        if scene is not None:
            scene.server.stop()
        failures = power_down(session, live_arms, park=park)
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# MolmoAct2 inference
# --------------------------------------------------------------------------- #


def open_inference_cameras(
    rig: Rig,
    *,
    overrides: dict[str, str] | None = None,
    warmup_s: float = _CAMERA_WARMUP_S,
) -> dict[str, cameras_mod.CameraReader]:
    """Open all three trained MolmoAct2 views, or none of them."""
    expected = {"top", "left_wrist", "right_wrist"}
    actual = set(rig.camera_names)
    if actual != expected:
        raise ConfigurationError(
            "MolmoAct2 bimanual inference needs cameras top, left_wrist, and right_wrist; "
            f"rig {rig.name!r} declares {', '.join(rig.camera_names) or 'none'}"
        )
    discovery = cameras_mod.discover(rig.cameras, overrides=overrides)
    if discovery.missing:
        details = ", ".join(
            f"{name} (serial {serial})" for name, serial in discovery.missing.items()
        )
        raise ConfigurationError(f"inference cameras missing: {details}")
    specs = tuple(found.spec() for found in discovery.matched.values())
    readers = cameras_mod.open_readers(specs)
    try:
        for name, reader in readers.items():
            if reader.wait_for_frame(warmup_s) is None:
                raise ConfigurationError(
                    f"inference camera {name!r} opened but delivered no frame in {warmup_s:g}s"
                )
    except BaseException:
        cameras_mod.close_readers(readers)
        raise
    return readers


def _inference_states(live_arms: list[LiveArm], *, max_age_s: float) -> dict[str, object]:
    """Return the latest states after the observation builder has checked them."""
    states: dict[str, object] = {}
    for entry in live_arms:
        state = entry.arm.latest_state
        if state is None or not state.is_fresh(max_age_s):
            age = "missing" if state is None else f"{state.age_s * 1e3:.0f} ms old"
            raise InferenceError(f"{entry.name} state is not fresh ({age})")
        states[entry.name] = state
    return states


def run_infer(
    rig: Rig,
    *,
    instruction: str,
    server: str | None = None,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    num_steps: int = DEFAULT_MOLMOACT_NUM_STEPS,
    enable_cuda_graph: bool = True,
    jpeg_quality: int = DEFAULT_MOLMOACT_JPEG_QUALITY,
    control_rate_hz: float = _MIRROR_RATE_HZ,
    speed: float = DEFAULT_CHUNK_SPEED,
    max_step_rad: float = DEFAULT_MAX_STEP_RAD,
    max_effector_step: float = DEFAULT_MAX_EFFECTOR_STEP,
    carry_targets: bool = False,
    reach_actions: bool = False,
    prefetch: bool = True,
    prefetch_margin_s: float = DEFAULT_PREFETCH_MARGIN_S,
    reset_start_pose: bool = False,
    park: bool = True,
    visualize: bool = True,
    camera_overrides: dict[str, str] | None = None,
    port: int = 8080,
    mesh_dir: Path | None = None,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
    stop: threading.Event | None = None,
    policy: MolmoActClient | None = None,
) -> int:
    """Run MolmoAct2 closed-loop inference against the bimanual YAM rig.

    The wire defaults are the reference deployment's. ``reach_actions`` and
    ``reset_start_pose`` are off by default because switching every execution
    flag on at once behaved worse on this rig; see ``docs/inference.md``.

    ``prefetch`` is on, and is the exception because it does not change what the
    policy is asked to do or how a chunk is executed -- only *when* the next
    call is made. Inferring between chunks leaves the arms holding still for a
    whole round trip at every chunk boundary (0.25 s against a 1.0 s chunk on
    this rig: a quarter of the run standing still, and a visible hitch in the
    motion every second). Prefetching hides the call under the motion already
    queued. The cost is that the observation is one round trip old, which is
    what ``prefetch_margin_s`` trades against running the queue dry.

    ``speed`` and ``carry_targets`` are the two knobs for a run that is too
    reactive rather than too slow. ``speed`` below 1.0 plays each chunk over
    proportionally more ticks -- the same path, walked gently, and re-planned
    less often because the chunk lasts longer. ``carry_targets`` stops the
    joint targets being pulled back to the measured pose at each chunk
    boundary. Neither changes what the policy is asked for; both change how
    hard the answer is driven, so A/B them one at a time.
    """
    if tuple(rig.names) != ("left", "right"):
        raise ConfigurationError(
            "MolmoAct2 bimanual inference requires the packaged left/right YAM rig"
        )
    if not instruction.strip():
        raise ConfigurationError("inference instruction must not be empty")
    if control_rate_hz <= 0:
        raise ConfigurationError("--control-rate must be positive")
    if speed <= 0:
        raise ConfigurationError("--speed must be positive")
    if prefetch_margin_s < 0:
        raise ConfigurationError("--prefetch-margin-s must not be negative")
    stop = stop if stop is not None else threading.Event()
    # num_steps and the CUDA-graph request live on the client rather than on
    # each call, so a prefetched chunk is asked for exactly the same way as a
    # synchronous one.
    client = policy or MolmoActClient(
        server,
        timeout_s=request_timeout_s,
        num_steps=num_steps,
        jpeg_quality=jpeg_quality,
        enable_cuda_graph=enable_cuda_graph,
    )
    client.health()

    # RGB is part of the model contract. The existing live command keeps BGR
    # for its cheap browser preview; inference requests RGB from the SDK so it
    # does not pay a channel swap on every HTTP request.
    capture_rig = rig.with_camera_capture(pixel_format="rgb8")
    scene = None
    camera_readers: dict[str, cameras_mod.CameraReader] = {}
    session = None
    live_arms: list[LiveArm] = []
    prefetcher: ChunkPrefetcher | None = None
    failures = 0
    runtime_error: Exception | None = None
    try:
        if visualize:
            from .viz import ArmSceneVisualizer

            scene = ArmSceneVisualizer.from_rig(rig, mesh_dir=mesh_dir, port=port)
        camera_readers = open_inference_cameras(capture_rig, overrides=camera_overrides)
        session, live_arms = power_up(
            rig, float_mode=False, backend_factory=backend_factory
        )
        arm_map = {entry.name: entry.arm for entry in live_arms}
        limits = {
            name: (
                np.array([spec.lower for spec in scene[name].joint_specs], dtype=np.float64)
                if scene is not None
                else np.full(6, -np.inf),
                np.array([spec.upper for spec in scene[name].joint_specs], dtype=np.float64)
                if scene is not None
                else np.full(6, np.inf),
            )
            for name in ("left", "right")
        }
        executor: BoundedChunkExecutor | ReachingChunkExecutor = (
            ReachingChunkExecutor(
                max_joint_step_rad=max_step_rad,
                max_effector_step=max_effector_step,
            )
            if reach_actions
            else BoundedChunkExecutor(
                max_step_rad=max_step_rad,
                max_effector_step=max_effector_step,
                carry_targets=carry_targets,
            )
        )
        if prefetch:
            prefetcher = ChunkPrefetcher(client)
        gripper = GripperWatch()
        stalled_grippers: set[str] = set()
        clamped_before = 0
        last_chunk_len = MOLMOACT_ACTION_HORIZON
        last_lag = 0.0
        camera_panel = None
        gripper_panel = None
        if scene is not None:
            from .viz import CameraPanel, GripperPanel

            # The render has no gripper joint to move (the YAM URDF's six
            # joints stop at the wrist), so the page states the gripper in
            # numbers instead of implying it with a mesh that never changes.
            gripper_panel = GripperPanel(scene.server, ("left", "right"))

            # The tiles are the policy's own input, decoded back off the wire
            # and pushed once per inference -- not a preview taken beside it --
            # so they are served at capture resolution and never on a clock.
            camera_panel = CameraPanel(
                scene.server, camera_readers, folder="Policy input", max_width=None
            )
            print(f"  viser    {scene.url}")
        print(f"  inference {client.url} — {instruction}")
        # Said before anything moves, because the operator can see the jaws and
        # the number in the same glance. The Viser render cannot help here: the
        # packaged YAM URDF has six actuated joints and the gripper is baked
        # into link_6's mesh, so the render shows the same jaws whatever the
        # gripper is doing. A reading that disagrees with the hardware in front
        # of you means the gripper servo is zeroed at the wrong stop.
        opening = _inference_states(live_arms, max_age_s=0.25)
        readings = ", ".join(
            f"{name} {state.effector.position:.3f}"  # type: ignore[union-attr]
            for name, state in sorted(opening.items())
            if state.effector is not None  # type: ignore[union-attr]
        )
        print(f"  gripper  measured now: {readings} (1.0 = open) — check the jaws agree")
        runtime = [
            label
            for label, enabled in (
                (f"speed {speed:g}x", speed != 1.0),
                ("carry-targets", carry_targets),
                ("reach-actions", reach_actions),
                ("no-prefetch", not prefetch),
                ("reset-start-pose", reset_start_pose),
                ("raw-frames", jpeg_quality <= 0),
                ("no-cuda-graph", not enable_cuda_graph),
            )
            if enabled
        ]
        if runtime:
            print(f"  runtime  {', '.join(runtime)}")
        print("ctrl-c to park at home_pos and power down")

        period = 1.0 / control_rate_hz

        def walk(rows: list[dict[str, PositionCommand]]) -> None:
            """Command an interpolated ramp, one row per ``SUB_STEP_PERIOD_S``."""
            for row in rows:
                if stop.is_set():
                    return
                for name, command in row.items():
                    arm_map[name].command(command)
                stop.wait(SUB_STEP_PERIOD_S)

        def request(observation: object) -> np.ndarray:
            if prefetcher is not None:
                return prefetcher.take(observation, instruction)  # type: ignore[arg-type]
            return client.infer(observation, instruction)  # type: ignore[arg-type]

        def infer_chunk(observation: object) -> np.ndarray:
            """One chunk, dropping to raw frames if the server cannot decode JPEG.

            Said once and loudly, then the run continues: losing two energized
            arms mid-task to a payload format is a poor trade for a server that
            predates the encoded-frame branch in its ``_to_pil``.
            """
            try:
                return request(observation)
            except EncodedFramesUnsupported as err:
                if int(getattr(client, "jpeg_quality", 0)) <= 0:
                    raise
                print(f"  frames   {err}", file=sys.stderr)
                print(
                    "  frames   sending raw frames for the rest of the run", file=sys.stderr
                )
                client.jpeg_quality = 0
                if prefetcher is not None:
                    prefetcher.drop()
                return request(observation)

        if reset_start_pose:
            states = _inference_states(live_arms, max_age_s=0.25)
            plan = start_pose_plan(states)  # type: ignore[arg-type]
            print(f"  reset    ramping to the training start pose ({len(plan)} steps)")
            walk(plan)

        while not stop.is_set():
            observation = build_observation(arm_map, camera_readers)
            states = _inference_states(live_arms, max_age_s=0.25)
            if scene is not None:
                for name, state in states.items():
                    scene.update(name, state.joints.position_rad)  # type: ignore[union-attr]
            if isinstance(executor, BoundedChunkExecutor):
                executor.reset(states)  # type: ignore[arg-type]
            actions = infer_chunk(observation)
            # The overlay draws the policy's own plan, so it is built from the
            # chunk rather than from the resampled ticks; the marker below is
            # scaled back into action space to follow it.
            plan = time_scale(actions, speed)
            if scene is not None:
                scene.update_chunk(split_chunk(actions))
            if camera_panel is not None:
                camera_panel.push(served_frames(client))
            latency = getattr(client, "last_latency", None) or {}
            reading = gripper.render()
            clamped_now = executor.clamped
            print(
                f"  chunk    {len(actions)} actions"
                + (f" over {len(plan)} ticks" if len(plan) != len(actions) else "")
                + "  "
                f"{latency.get('round_trip_s', 0.0):.2f}s "
                f"(gpu {latency.get('gpu_s', 0.0):.2f} "
                f"+ wire {latency.get('transport_s', 0.0):.2f})  "
                f"{latency.get('payload_mb', 0.0):.2f} MB  "
                # Per chunk first, running total second. The total on its own
                # only ever says how long the run has been going: it is the
                # rate that says whether the policy is being slowed down, and
                # making the reader subtract two log lines to find it meant
                # nobody did. Both counts describe the chunk just executed --
                # this line is printed before the new one runs, same as `grip`.
                f"clamped +{executor.clamped - clamped_before}/{last_chunk_len} "
                f"({executor.clamped})  "
                # The other half of the same question. `clamped` says how much
                # of the plan the limit refused; `lag` says how much of what
                # was *not* refused the arms failed to reach. A low clamp count
                # beside a large lag is hardware that cannot keep up with the
                # plan, and no clamp setting fixes that -- `--speed` does.
                f"lag {last_lag:.3f}"
                + (f"  grip {reading}" if reading else ""),
                flush=True,
            )
            clamped_before = clamped_now
            # Ticks, not actions: the clamp is counted once per command issued,
            # so the denominator has to be what the chunk was executed as.
            last_chunk_len = len(plan)
            lag = 0.0
            # Said once per arm and then not repeated: a gripper that is inert
            # stays inert, and a warning on every chunk would bury the run.
            for name in gripper.stalled():
                if name in stalled_grippers:
                    continue
                stalled_grippers.add(name)
                print(
                    f"  gripper  {name} is NOT TRACKING: commanded across "
                    f"{gripper.command_travel:.2f} of its range and the measured "
                    f"position has not moved once. Check the startup calibration "
                    f"line in the node log — a stroke it could not measure means "
                    f"the jaws were blocked when it probed, and the run is using "
                    f"the configured range instead.",
                    file=sys.stderr,
                    flush=True,
                )
            if gripper_panel is not None:
                gripper_panel.update(
                    gripper.commanded, gripper.measured, stalled=sorted(stalled_grippers)
                )

            for index, action in enumerate(plan):
                if stop.is_set():
                    break
                tick_start = time.monotonic()
                states = _inference_states(live_arms, max_age_s=0.25)
                if isinstance(executor, ReachingChunkExecutor):
                    # The reference runtime: walk to the action so the arm
                    # arrives at it, which paces itself and usually outruns the
                    # nominal tick below.
                    rows = executor.plan(action, states, limits=limits)  # type: ignore[arg-type]
                    walk(rows)
                    issued = rows[-1] if rows else None
                else:
                    issued = executor.step(action, limits=limits)
                    for name, command in issued.items():
                        arm_map[name].command(command)
                if issued is not None:
                    gripper.observe(issued, states)  # type: ignore[arg-type]
                    # Against the state read at the top of this tick, so it is
                    # the gap the arms were still carrying when this command
                    # went out: how far behind the plan the hardware is running.
                    lag = max(lag, command_lag(issued, states))  # type: ignore[arg-type]
                if scene is not None:
                    for name, state in states.items():
                        scene.update(name, state.joints.position_rad)  # type: ignore[union-attr]
                    # Advance the overlay, do not rebuild it: this runs every
                    # control tick, and a rebuild costs 61 ms against a 33 ms
                    # period at the default 30 Hz.
                    scene.set_chunk_progress(int(round((index + 1) * speed)))
                if prefetcher is not None and not prefetcher.busy:
                    # Fire once the motion still queued is shorter than the call
                    # takes, so the next chunk lands just as this one runs out.
                    queued_s = (len(plan) - index - 1) * period
                    if queued_s <= prefetcher.latency_s + prefetch_margin_s:
                        prefetcher.submit(
                            build_observation(arm_map, camera_readers), instruction
                        )
                remaining = period - (time.monotonic() - tick_start)
                if remaining > 0:
                    stop.wait(remaining)
            last_lag = lag
    except KeyboardInterrupt:
        print()
    except (InferenceError, ConfigurationError, PiControlError) as err:
        runtime_error = err
        print(f"inference stopped: {type(err).__name__}: {err}", file=sys.stderr)
    finally:
        if prefetcher is not None:
            prefetcher.close()
        client.close()
        if scene is not None:
            scene.clear_chunk()
        cameras_mod.close_readers(camera_readers)
        if scene is not None:
            scene.server.stop()
        if session is not None:
            failures = power_down(session, live_arms, park=park)
    if runtime_error is not None:
        return 1
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #


def run_rollout(
    rig: Rig,
    *,
    repo_id: str,
    episodes: int = 1,
    episode_seconds: float = 120.0,
    fps: int = 30,
    root: Path | None = None,
    server: str | None = None,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    num_steps: int = DEFAULT_MOLMOACT_NUM_STEPS,
    enable_cuda_graph: bool = True,
    jpeg_quality: int = DEFAULT_MOLMOACT_JPEG_QUALITY,
    speed: float = 0.5,
    chunk_size: int | None = None,
    max_step_rad: float = DEFAULT_MAX_STEP_RAD,
    max_effector_step: float = DEFAULT_MAX_EFFECTOR_STEP,
    carry_targets: bool = False,
    prefetch: bool = True,
    prefetch_margin_s: float = DEFAULT_PREFETCH_MARGIN_S,
    visualize: bool = True,
    camera_overrides: dict[str, str] | None = None,
    port: int = 8080,
    mesh_dir: Path | None = None,
    wait_between_episodes: bool = True,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Record interactive, timed MolmoAct episodes as a LeRobot v3 dataset.

    Each episode gets its own policy prompt. A completed episode is saved by
    ``LeRobotSink`` before the arms are parked; a Ctrl-C saves the frames
    already captured in the open episode and still runs the same safe park path
    before the next prompt. Success/failure labels are written to a sidecar
    manifest after parking because asking for a label must never hold the
    motors energized.
    """
    from . import record as record_mod

    if tuple(rig.names) != ("left", "right"):
        raise ConfigurationError(
            "policy rollouts require the packaged bimanual left/right YAM rig"
        )
    if episodes <= 0:
        raise ConfigurationError("--episodes must be positive")
    if episode_seconds <= 0:
        raise ConfigurationError("--episode-seconds must be positive")
    if fps <= 0:
        raise ConfigurationError("--fps must be positive")
    if not repo_id.strip():
        raise ConfigurationError("--repo-id must not be empty")
    if speed <= 0:
        raise ConfigurationError("--speed must be positive")
    if chunk_size is not None and chunk_size <= 0:
        raise ConfigurationError("--chunk-size must be positive")
    if chunk_size is not None and chunk_size > MOLMOACT_ACTION_HORIZON:
        raise ConfigurationError(
            f"--chunk-size cannot exceed the model horizon ({MOLMOACT_ACTION_HORIZON})"
        )

    client = MolmoActClient(
        server,
        timeout_s=request_timeout_s,
        num_steps=num_steps,
        jpeg_quality=jpeg_quality,
        enable_cuda_graph=enable_cuda_graph,
    )
    client.health()
    capture_rig = rig.with_camera_capture(fps=fps, pixel_format="rgb8")
    cameras: dict[str, cameras_mod.CameraReader] = {}
    scene = None
    camera_panel = None
    sink: object | None = None
    manifest: dict[str, object] = {
        "format": "openpi-control.inference-rollout-v1",
        "dataset_format": "LeRobot v3.0",
        "repo_id": repo_id,
        "instruction_is_per_episode": True,
        "speed": speed,
        "chunk_size": (
            chunk_size if chunk_size is not None else MOLMOACT_ACTION_HORIZON
        ),
        "episode_seconds": episode_seconds,
        "fps": fps,
        "partial_episodes_saved_on_ctrl_c": True,
        "episodes": [],
    }
    manifest_path: Path | None = None
    status = 0
    try:
        cameras = open_inference_cameras(capture_rig, overrides=camera_overrides)
        shapes = record_mod.camera_shapes(cameras)
        state_names = record_mod.arm_feature_names(
            ["left", "right"], {"left": 6, "right": 6}
        )
        features = record_mod.build_features(state_names, shapes)
        sink = record_mod.LeRobotSink(
            repo_id=repo_id,
            fps=fps,
            features=features,
            robot_type=record_mod.rig_robot_type(rig),
            root=root,
            image_writer_threads=4 * len(cameras),
        )
        manifest_path = sink.root / "openpi_control_rollouts.json"
        _write_rollout_manifest(manifest_path, manifest)

        if visualize:
            from .viz import ArmSceneVisualizer, CameraPanel

            scene = ArmSceneVisualizer.from_rig(rig, mesh_dir=mesh_dir, port=port)
            camera_panel = CameraPanel(
                scene.server, cameras, folder="Policy input", max_width=None
            )
            print(f"  viser    {scene.url}")

        for episode_index in range(1, episodes + 1):
            if episode_index > 1 and wait_between_episodes:
                input_fn(
                    "Reset the same towel to its starting pose, then press Enter to continue: "
                )
            prompt = _rollout_prompt(input_fn, episode_index, episodes)
            print(f"\nEpisode {episode_index}/{episodes}: {prompt!r}")

            session = None
            live_arms: list[LiveArm] = []
            source: InferenceRolloutSource | None = None
            saved_before = getattr(sink, "num_episodes")
            episode_result = None
            park_failures = 0
            try:
                session, live_arms = power_up(
                    rig, float_mode=False, backend_factory=backend_factory
                )
                arm_map = {entry.name: entry.arm for entry in live_arms}
                limits = {
                    name: (
                        np.array(
                            [spec.lower for spec in scene[name].joint_specs], dtype=np.float64
                        )
                        if scene is not None
                        else np.full(6, -np.inf),
                        np.array(
                            [spec.upper for spec in scene[name].joint_specs], dtype=np.float64
                        )
                        if scene is not None
                        else np.full(6, np.inf),
                    )
                    for name in ("left", "right")
                }

                def on_chunk(actions: np.ndarray) -> None:
                    if scene is not None:
                        scene.update_chunk(split_chunk(actions))
                    if camera_panel is not None:
                        camera_panel.push(served_frames(client))

                def on_tick(
                    states: Mapping[str, object],
                    commands: Mapping[str, PositionCommand],
                    consumed: int,
                ) -> None:
                    if scene is not None:
                        for name, state in states.items():
                            scene.update(name, state.joints.position_rad)  # type: ignore[union-attr]
                        scene.set_chunk_progress(consumed)

                source = InferenceRolloutSource(
                    arms=arm_map,
                    readers=cameras,
                    client=client,
                    instruction=prompt,
                    episode_seconds=episode_seconds,
                    fps=fps,
                    speed=speed,
                    chunk_size=chunk_size,
                    max_step_rad=max_step_rad,
                    max_effector_step=max_effector_step,
                    limits=limits,
                    prefetch=prefetch,
                    prefetch_margin_s=prefetch_margin_s,
                    carry_targets=carry_targets,
                    on_chunk=on_chunk,
                    on_tick=on_tick,
                )
                print(f"  rollout  {source.describe()}")
                episode_result = record_mod.record_session(
                    arms=arm_map,
                    source=source,
                    sink=sink,  # type: ignore[arg-type]
                    task=prompt,
                    fps=fps,
                    cameras=cameras,
                    num_episodes=0,
                    finalize=False,
                    save_on_interrupt=True,
                )
            finally:
                if source is not None:
                    source.close()
                if scene is not None:
                    scene.clear_chunk()
                if session is not None:
                    park_failures = power_down(session, live_arms, park=True)

            if park_failures:
                print(
                    f"  {park_failures} arm(s) failed to park; stopping before the "
                    "next episode",
                    file=sys.stderr,
                )
                status = 1
                break

            saved = getattr(sink, "num_episodes") > saved_before
            aborted = episode_result is not None and episode_result.ended_by == "interrupted"
            if aborted:
                if saved:
                    print("  episode interrupted; partial frames were saved")
                else:
                    print("  episode interrupted before a frame was captured")
            success = _rollout_yes_no(
                input_fn, f"Episode {episode_index} successful? [y/n]: "
            )
            if not saved:
                print("  no LeRobot episode was written for this attempt")
            entries = manifest["episodes"]
            assert isinstance(entries, list)
            entries.append(
                {
                    "attempt": episode_index,
                    "episode_index": saved_before if saved else None,
                    "prompt": prompt,
                    "success": success,
                    "label": "y" if success else "n",
                    "saved": saved,
                    "aborted": aborted,
                }
            )
            if manifest_path is not None:
                _write_rollout_manifest(manifest_path, manifest)
    except KeyboardInterrupt:
        print("\nrollout session stopped; hardware shutdown completed")
        status = 1
    except (ConfigurationError, InferenceError, PiControlError) as err:
        print(f"rollout stopped: {type(err).__name__}: {err}", file=sys.stderr)
        status = 1
    finally:
        if sink is not None:
            sink.finalize()  # type: ignore[attr-defined]
        client.close()
        cameras_mod.close_readers(cameras)
        if scene is not None:
            scene.server.stop()
    return status


def _rollout_prompt(input_fn: Callable[[str], str], index: int, total: int) -> str:
    while True:
        prompt = input_fn(f"Prompt for episode {index}/{total}: ").strip()
        if prompt:
            return prompt
        print("The prompt cannot be empty.", file=sys.stderr)


def _rollout_yes_no(input_fn: Callable[[str], str], message: str) -> bool:
    while True:
        value = input_fn(message).strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.", file=sys.stderr)


def _write_rollout_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #


def run_record(
    rig: Rig,
    *,
    task: str,
    repo_id: str | None,
    teleop: str = "vr",
    fps: int = 30,
    num_episodes: int = 0,
    hold_duration_s: float = 10.0,
    root: Path | None = None,
    dry_run: bool = False,
    park: bool = True,
    push_to_hub: bool = False,
    private: bool = False,
    camera_overrides: dict[str, str] | None = None,
    vr_url: str | None = None,
    vr_kit: Path | None = None,
    yam_xml: str | None = None,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Record teleoperated episodes from a rig into a LeRobot dataset.

    The whole session lives inside this call for the same reason ``live`` does:
    the native node dies with its parent, so returning from here is what
    de-energizes the arms. The order of the teardown matters and is deliberate
    -- the dataset is finalized on disk *before* the arms are parked, so an
    interrupted shutdown still leaves a complete, loadable dataset; and an
    upload happens *after* the arms are down, because a few hundred megabytes of
    video takes minutes and there is no reason to hold motors energized for it.
    """
    from . import record as record_mod

    stop = stop if stop is not None else threading.Event()
    cameras: dict[str, object] = {}

    # Cameras first: they need no motors, and finding out now that a camera is
    # held by another process beats finding out with two arms energized.
    if rig.cameras:
        discovery = cameras_mod.discover(rig.cameras, overrides=camera_overrides)
        if not discovery.complete:
            missing = ", ".join(
                f"{name} (serial {serial})" for name, serial in discovery.missing.items()
            )
            print(f"error: camera(s) not on the bus: {missing}", file=sys.stderr)
            return 1
        cameras = dict(cameras_mod.open_readers(discovery.specs()))
        print(f"  cameras  {', '.join(cameras)}")

    session = None
    live_arms: list[LiveArm] = []
    source = None
    status = 0
    try:
        shapes = record_mod.camera_shapes(cameras)

        session, live_arms = power_up(rig, float_mode=False, backend_factory=backend_factory)
        # Followers only. A leader has no `command`, so handing one to the record
        # loop would fail mid-session on a rig that has one -- and its pose is
        # not what the dataset is about anyway.
        arms = {entry.name: entry.arm for entry in live_arms if entry.rig_arm.is_follower}
        if not arms:
            raise ConfigurationError(
                f"rig {rig.name!r} has no follower arms to record from"
            )
        for entry in live_arms:
            print(
                f"  {entry.name:<8} {entry.rig_arm.model} on {entry.rig_arm.interface}"
                " — energized, holding"
            )

        dofs = {name: arm.capabilities.dof for name, arm in arms.items()}
        state_names = record_mod.arm_feature_names(list(arms), dofs)
        features = record_mod.build_features(state_names, shapes)

        source = _build_teleop_source(
            teleop,
            dofs=dofs,
            hold_duration_s=hold_duration_s,
            vr_url=vr_url,
            vr_kit=vr_kit,
            yam_xml=yam_xml,
        )
        print(f"  teleop   {source.describe()}")

        if dry_run:
            sink: object = record_mod.MemorySink()
            print("  dataset  --dry-run: nothing is written to disk")
        else:
            assert repo_id is not None  # guaranteed by the caller
            sink = record_mod.LeRobotSink(
                repo_id=repo_id,
                fps=fps,
                features=features,
                robot_type=record_mod.rig_robot_type(rig),
                root=root,
                image_writer_threads=4 * len(cameras),
            )
            print(f"  dataset  {repo_id} at {sink.root}")
        print(f"  schema   {len(state_names)}-dim state/action, {len(shapes)} camera(s)")
        print(f"  task     {task!r}")
        print()
        if teleop == "vr":
            print("  right B: start (or redo) an episode    left Y: save it")
        print(f"  ctrl-c to {'park and ' if park else ''}power down")
        print()

        result = record_mod.record_session(
            arms=arms,
            source=source,
            sink=sink,  # type: ignore[arg-type]
            cameras=cameras,
            task=task,
            fps=fps,
            num_episodes=num_episodes,
            stop=stop,
        )
        print(f"\n{result.summary()}")
        if result.episodes == 0:
            status = 1
            print("no episode was saved", file=sys.stderr)
    finally:
        if source is not None:
            source.close()
        cameras_mod.close_readers(cameras)  # type: ignore[arg-type]
        if session is not None:
            failures = power_down(session, live_arms, park=park)
            status = status or (1 if failures else 0)

    if push_to_hub and not dry_run and status == 0:
        print(f"pushing {repo_id} to the Hub ({'private' if private else 'public'}) ...")
        try:
            sink.push_to_hub(private=private)  # type: ignore[union-attr]
        except Exception as err:  # noqa: BLE001 - the dataset is already safe on disk
            print(f"push failed: {type(err).__name__}: {err}", file=sys.stderr)
            print("the local dataset is complete; retry the upload by hand", file=sys.stderr)
            status = 1
        else:
            print(f"pushed: https://huggingface.co/datasets/{repo_id}")
    return status


def _build_teleop_source(
    teleop: str,
    *,
    dofs: dict[str, int],
    hold_duration_s: float,
    vr_url: str | None,
    vr_kit: Path | None,
    yam_xml: str | None,
):  # noqa: ANN202 - a TeleopSource
    from . import record as record_mod

    if teleop == "hold":
        return record_mod.HoldSource(dofs, duration_s=hold_duration_s)
    if teleop == "vr":
        from .teleop_vr import DEFAULT_WS_URL, QuestTeleopSource

        return QuestTeleopSource(
            list(dofs),
            ws_url=vr_url or DEFAULT_WS_URL,
            kit_path=vr_kit,
            model_path=yam_xml,
        )
    raise ConfigurationError(f"unknown teleop source {teleop!r}; use 'vr' or 'hold'")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _add_common(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--model", required=required, help=f"one of: {', '.join(SUPPORTED_MODELS)}"
    )
    parser.add_argument(
        "--interface",
        required=required,
        help="CAN interface name, /dev path, or controller IPv4 address",
    )
    parser.add_argument(
        "--effector", default=None, help=f"one of: {', '.join(SUPPORTED_EFFECTORS)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openpi-control", description="Preflight and maintenance for a connected arm."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="read-only preflight checks for an arm or a rig")
    _add_common(doctor, required=False)
    doctor.add_argument(
        "--rig",
        default=None,
        help=f"check every arm of a packaged rig instead of one arm: {', '.join(rig_names())}",
    )
    doctor.add_argument(
        "--interface-override",
        action="append",
        metavar="ARM=IFACE",
        help="with --rig: check one arm on a different bus, e.g. left=can2 (repeatable)",
    )
    doctor.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="with --rig: pin one camera to an explicit device (repeatable)",
    )
    doctor.add_argument(
        "--probe",
        action="store_true",
        help="also open the bus and listen for traffic (reads only, never sends)",
    )

    zero = sub.add_parser("zero", help="write the current pose as each servo's firmware zero")
    _add_common(zero)
    zero.add_argument(
        "--joint",
        type=int,
        default=None,
        help="zero only this joint id instead of every joint",
    )
    zero.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit without opening the bus"
    )
    zero.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    live = sub.add_parser(
        "live", help="energize a rig, mirror it in the browser, then park it and power down"
    )
    live.add_argument(
        "--rig", default="yam_bimanual", help=f"one of: {', '.join(rig_names())}"
    )
    live.add_argument(
        "--only",
        action="append",
        metavar="ARM",
        help="bring up only this arm of the rig, e.g. --only left (repeatable)",
    )
    live.add_argument(
        "--interface",
        action="append",
        metavar="ARM=IFACE",
        help="move one arm to a different bus, e.g. --interface left=can2 (repeatable)",
    )
    live.add_argument(
        "--float",
        dest="float_mode",
        action="store_true",
        help="gravity float instead of holding: the arms become backdrivable",
    )
    live.add_argument(
        "--no-park",
        dest="park",
        action="store_false",
        help="de-energize where the arm stands instead of parking it at home_pos first",
    )
    live.add_argument(
        "--no-viz",
        dest="visualize",
        action="store_false",
        help="energize and hold without serving the browser view",
    )
    live.add_argument(
        "--control",
        action="store_true",
        help="add per-arm browser control: confirm the pose, Arm, then drive with sliders",
    )
    live.add_argument(
        "--no-cameras",
        dest="camera_preview",
        action="store_false",
        help="skip the camera tiles, leaving the rig's cameras free for another process",
    )
    live.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device, e.g. top=/dev/video4 (repeatable)",
    )
    live.add_argument("--port", type=int, default=8080, help="viser HTTP port")
    live.add_argument(
        "--mesh-dir", type=Path, default=None, help="directory holding the URDF's meshes"
    )
    live.add_argument(
        "--skip-preflight",
        action="store_true",
        help="energize without running the doctor checks first",
    )
    live.add_argument("--list", action="store_true", help="describe the rig and exit")

    infer = sub.add_parser(
        "infer",
        help="run bimanual YAM MolmoAct2 inference, execute chunks, and visualize them",
    )
    infer.add_argument(
        "--rig", default="yam_bimanual", help="the trained bimanual YAM rig"
    )
    infer.add_argument(
        "--interface",
        action="append",
        metavar="ARM=IFACE",
        help="move one arm to a different bus, e.g. --interface left=can2 (repeatable)",
    )
    infer.add_argument(
        "--server",
        default="http://127.0.0.1:8202",
        help="MolmoAct server URL or host:port; /act is appended automatically",
    )
    infer.add_argument(
        "--instruction", required=True, help="language instruction sent with every observation"
    )
    infer.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP inference timeout in seconds (default: 60)",
    )
    infer.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_MOLMOACT_NUM_STEPS,
        help="MolmoAct2 denoising steps requested from the server",
    )
    infer.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="stop requesting CUDA graph inference from the server (~20x slower)",
    )
    infer.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_MOLMOACT_JPEG_QUALITY,
        help="JPEG quality for frames on the wire; 0 sends them raw (default: 95)",
    )
    infer.add_argument(
        "--raw-frames",
        dest="jpeg_quality",
        action="store_const",
        const=0,
        help="send raw HxWx3 frames, for a server that cannot decode encoded ones",
    )
    infer.add_argument(
        "--control-rate",
        type=float,
        default=_MIRROR_RATE_HZ,
        help="nominal action rate in Hz, also used to time the prefetch (default: 30)",
    )
    infer.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_CHUNK_SPEED,
        help="play each chunk at this fraction of the training action rate "
        "(0.5 = half speed, same path over twice the ticks; default: 1.0)",
    )
    infer.add_argument(
        "--max-step-rad",
        type=float,
        default=DEFAULT_MAX_STEP_RAD,
        help="maximum joint movement per executed action tick",
    )
    infer.add_argument(
        "--max-effector-step",
        type=float,
        default=DEFAULT_MAX_EFFECTOR_STEP,
        help="maximum normalized gripper movement per executed action tick "
        "(default: 0.30, a full stroke in four ticks)",
    )
    infer.add_argument(
        "--carry-targets",
        action="store_true",
        help="carry commanded joint targets across chunk boundaries instead of "
        "re-seeding them from the measured pose, as the gripper already does",
    )
    infer.add_argument(
        "--reach-actions",
        action="store_true",
        help="walk to every action in sub-steps, clamped against the measured pose",
    )
    infer.add_argument(
        "--no-prefetch",
        dest="prefetch",
        action="store_false",
        help="infer between chunks instead of during them, so the arms hold "
        "still for a round trip at every chunk boundary",
    )
    infer.add_argument(
        "--prefetch-margin-s",
        type=float,
        default=DEFAULT_PREFETCH_MARGIN_S,
        help="margin added to the measured latency when timing a prefetch",
    )
    infer.add_argument(
        "--reset-start-pose",
        action="store_true",
        help="ramp to the training start pose before the first observation",
    )
    infer.add_argument(
        "--no-park",
        dest="park",
        action="store_false",
        help="de-energize where the arms stand instead of parking at home_pos",
    )
    infer.add_argument(
        "--no-viz",
        dest="visualize",
        action="store_false",
        help="run inference and hardware without starting Viser",
    )
    infer.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device (repeatable)",
    )
    infer.add_argument("--port", type=int, default=8080, help="viser HTTP port")
    infer.add_argument(
        "--mesh-dir", type=Path, default=None, help="directory holding the URDF's meshes"
    )
    infer.add_argument(
        "--skip-preflight",
        action="store_true",
        help="energize without running the doctor checks first",
    )

    rollout = sub.add_parser(
        "rollout",
        help="record interactive MolmoAct episodes as a LeRobot v3 dataset",
    )
    rollout.add_argument(
        "--rig", default="yam_bimanual", help="the trained bimanual YAM rig"
    )
    rollout.add_argument(
        "--interface",
        action="append",
        metavar="ARM=IFACE",
        help="move an arm to a different CAN interface, e.g. left=can_left",
    )
    rollout.add_argument(
        "--repo-id",
        required=True,
        help="local/Hub dataset id, e.g. Dimios45/openpi-fold-towel-rollout-ablation",
    )
    rollout.add_argument(
        "--root", type=Path, default=None, help="local LeRobot dataset directory"
    )
    rollout.add_argument(
        "--episodes", type=int, default=3, help="number of rollout attempts (default: 3)"
    )
    rollout.add_argument(
        "--episode-seconds",
        type=float,
        default=120.0,
        help="duration of each completed episode (default: 120)",
    )
    rollout.add_argument(
        "--fps", type=int, default=30, help="recording and action-loop rate (default: 30)"
    )
    rollout.add_argument(
        "--server",
        default="http://127.0.0.1:8202",
        help="MolmoAct server URL or host:port; /act is appended automatically",
    )
    rollout.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP inference timeout in seconds (default: 60)",
    )
    rollout.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_MOLMOACT_NUM_STEPS,
        help="MolmoAct2 denoising steps requested from the server",
    )
    rollout.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="stop requesting CUDA graph inference from the server",
    )
    rollout.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_MOLMOACT_JPEG_QUALITY,
        help="JPEG quality for policy frames; 0 sends raw frames",
    )
    rollout.add_argument(
        "--raw-frames",
        dest="jpeg_quality",
        action="store_const",
        const=0,
        help="send raw HxWx3 frames instead of JPEG-encoded frames",
    )
    rollout.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="play each policy chunk at this fraction of training speed (default: 0.5)",
    )
    rollout.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="execute only this prefix of each returned chunk (1-30; default: full chunk)",
    )
    rollout.add_argument(
        "--max-step-rad",
        type=float,
        default=DEFAULT_MAX_STEP_RAD,
        help="maximum joint movement per commanded step",
    )
    rollout.add_argument(
        "--max-effector-step",
        type=float,
        default=DEFAULT_MAX_EFFECTOR_STEP,
        help="maximum normalized gripper movement per commanded step",
    )
    rollout.add_argument(
        "--carry-targets",
        action="store_true",
        help="carry commanded targets across policy chunk boundaries",
    )
    rollout.add_argument(
        "--no-prefetch",
        dest="prefetch",
        action="store_false",
        help="infer between chunks instead of during them",
    )
    rollout.add_argument(
        "--prefetch-margin-s",
        type=float,
        default=DEFAULT_PREFETCH_MARGIN_S,
        help="extra margin used when scheduling prefetched chunks",
    )
    rollout.add_argument(
        "--no-reset-pause",
        dest="wait_between_episodes",
        action="store_false",
        help="do not wait for manual towel reset between episodes",
    )
    rollout.add_argument(
        "--no-viz",
        dest="visualize",
        action="store_false",
        help="record inference without starting Viser",
    )
    rollout.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device",
    )
    rollout.add_argument("--port", type=int, default=8080, help="viser HTTP port")
    rollout.add_argument(
        "--mesh-dir", type=Path, default=None, help="directory holding the URDF meshes"
    )
    rollout.add_argument(
        "--skip-preflight",
        action="store_true",
        help="energize without running doctor checks first",
    )

    cams = sub.add_parser(
        "cameras", help="resolve a rig's cameras to device paths and say which are present"
    )
    cams.add_argument("--rig", default="yam_bimanual", help=f"one of: {', '.join(rig_names())}")
    cams.add_argument(
        "--only",
        action="append",
        metavar="ARM",
        help="narrow to one arm of the rig, which drops the other arm's wrist camera",
    )
    cams.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device, e.g. top=/dev/video4 (repeatable)",
    )
    cams.add_argument(
        "--probe",
        action="store_true",
        help="also open each stream and grab a frame (needs the cameras extra)",
    )
    cams.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        metavar="DIR",
        help="with --probe: write each grabbed frame to DIR/<name>.png",
    )

    rec = sub.add_parser(
        "record", help="teleoperate a rig and record the episodes as a LeRobot dataset"
    )
    rec.add_argument("--rig", default="yam_bimanual", help=f"one of: {', '.join(rig_names())}")
    rec.add_argument(
        "--only",
        action="append",
        metavar="ARM",
        help="record one arm of the rig, which also drops the other wrist camera",
    )
    rec.add_argument(
        "--interface",
        action="append",
        metavar="ARM=IFACE",
        help="move one arm to a different bus, e.g. --interface left=can2 (repeatable)",
    )
    rec.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device (repeatable)",
    )
    rec.add_argument(
        "--repo-id",
        default=None,
        help="dataset id, e.g. you/yam-fold-towel. Required unless --dry-run",
    )
    rec.add_argument(
        "--task",
        default=None,
        help="the language instruction stored on every frame; relabelling later "
        "means rewriting the dataset, so set it",
    )
    rec.add_argument(
        "--fps",
        type=int,
        default=30,
        help="dataset, control-loop, and camera rate. The D405s here sustain 90 "
        "at 848x480 with all three running; the arms publish at 200",
    )
    rec.add_argument(
        "--camera-fps",
        type=int,
        default=None,
        help="run the cameras at a different rate from the loop (default: --fps)",
    )
    rec.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="end the session after this many saved episodes (0 = until ctrl-c)",
    )
    rec.add_argument("--root", type=Path, default=None, help="local dataset root")
    rec.add_argument(
        "--teleop",
        choices=("vr", "hold"),
        default="vr",
        help="vr: drive from a Quest via vr-teleop-kit. hold: arms stay still for "
        "--hold-seconds, to check the pipeline without a headset",
    )
    rec.add_argument(
        "--hold-seconds",
        type=float,
        default=10.0,
        help="episode length for --teleop hold",
    )
    rec.add_argument("--vr-url", default=None, help="relay WebSocket URL")
    rec.add_argument(
        "--vr-kit", type=Path, default=None, help="path to a vr-teleop-kit checkout"
    )
    rec.add_argument(
        "--yam-xml", default=None, help="YAM MJCF the VR inverse kinematics loads"
    )
    rec.add_argument(
        "--no-cameras",
        dest="cameras_enabled",
        action="store_false",
        help="record state and action only",
    )
    rec.add_argument(
        "--dry-run",
        action="store_true",
        help="run the whole session, including the arms, but write nothing to disk",
    )
    rec.add_argument(
        "--no-park",
        dest="park",
        action="store_false",
        help="de-energize where the arms stand instead of parking them first",
    )
    rec.add_argument(
        "--skip-preflight", action="store_true", help="record without the doctor checks"
    )
    rec.add_argument(
        "--push-to-hub", action="store_true", help="upload once the arms are down"
    )
    rec.add_argument("--private", action="store_true", help="with --push-to-hub, keep it private")

    args = parser.parse_args(argv)
    log_path = runlog.setup_run_logging(args.command)

    commands = {
        "doctor": _command_doctor,
        "zero": _command_zero,
        "live": _command_live,
        "infer": _command_infer,
        "rollout": _command_rollout,
        "cameras": _command_cameras,
        "record": _command_record,
    }
    try:
        return commands[args.command](args, log_path)
    except (ConfigurationError, InferenceError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


def _command_doctor(args: argparse.Namespace, log_path: Path) -> int:
    if args.rig is not None:
        if args.model is not None or args.interface is not None:
            raise ConfigurationError(
                "--rig already names every arm and its bus; drop --model/--interface "
                "(use --interface-override ARM=IFACE to move one)"
            )
        return _doctor_rig(args, log_path)
    if args.model is None or args.interface is None:
        raise ConfigurationError("doctor needs either --rig, or both --model and --interface")

    print(f"doctor: {args.model} on {args.interface}")
    results = run_doctor(
        args.model, args.interface, effector_model=args.effector, probe=args.probe
    )
    for result in results:
        print(result.render())
    failures = sum(1 for result in results if result.status == _FAIL)
    warnings = sum(1 for result in results if result.status == _WARN)
    print(f"\n{len(results)} checks, {failures} failed, {warnings} warned")
    print(f"log: {log_path}")
    return 1 if failures else 0


def _doctor_rig(args: argparse.Namespace, log_path: Path) -> int:
    """Every arm of a rig, checked with the same code ``live`` preflights with."""
    rig = resolve_rig(args.rig).with_interfaces(
        _parse_interface_overrides(args.interface_override)
    )
    print(f"doctor: rig {rig.name} — {rig.description}")
    failures = warnings = 0
    for rig_arm in rig.arms:
        results = run_doctor(
            rig_arm.model,
            rig_arm.interface,
            effector_model=rig_arm.effector_model,
            probe=args.probe,
        )
        failures += sum(1 for result in results if result.status == _FAIL)
        warnings += sum(1 for result in results if result.status == _WARN)
        print(f"\n{rig_arm.name} ({rig_arm.model} on {rig_arm.interface}):")
        for result in results:
            print(result.render())

    # Cameras belong to the rig, not to an arm, so they are checked once here
    # rather than repeated under every arm.
    camera_results = run_camera_checks(
        rig, overrides=cameras_mod.parse_camera_overrides(args.camera)
    )
    failures += sum(1 for result in camera_results if result.status == _FAIL)
    warnings += sum(1 for result in camera_results if result.status == _WARN)
    print(f"\ncameras ({len(rig.cameras)} declared):")
    for result in camera_results:
        print(result.render())

    print(
        f"\n{len(rig.arms)} arms, {len(rig.cameras)} cameras, "
        f"{failures} failed, {warnings} warned"
    )
    print(f"log: {log_path}")
    return 1 if failures else 0


def _command_zero(args: argparse.Namespace, log_path: Path) -> int:
    full_plan = build_plan(args.model, args.effector)
    plan = full_plan
    if args.joint is not None:
        plan = tuple(entry for entry in full_plan if entry.joint_id == args.joint)
        if not plan:
            known = ", ".join(str(entry.joint_id) for entry in full_plan)
            raise ConfigurationError(
                f"{args.model} has no joint {args.joint}; joints are {known}"
            )

    port_type = plan_port_type(plan)
    if port_type is None:
        print("every servo in this plan is read-only; nothing to zero")
        return 0

    if args.dry_run:
        print(f"dry run: would zero via {port_type} on {args.interface}")
        for entry in plan:
            state = "read-only, skipped" if entry.read_only else "zero"
            print(f"  {entry.label():<24} {entry.servo_model:<24} id {entry.servo_id:<6} {state}")
        return 0

    problem = buses.check_interface(port_type, args.interface)
    if problem:
        raise ConfigurationError(problem)

    if not args.yes and not _confirm(plan, args.model, args.interface):
        print("aborted; nothing was written")
        return 1

    assets = resolve_model_assets(args.model, effector_model=args.effector)
    outcomes = zero_arm(
        plan,
        port_type,
        args.interface,
        baudrate=_catalog_baudrate(assets.model_config),
    )
    failed = 0
    for entry, error in outcomes:
        if error is None:
            print(f"  {entry.label():<24} zeroed")
        else:
            failed += 1
            print(f"  {entry.label():<24} FAILED: {error}")
    for entry in plan:
        if entry.read_only:
            print(f"  {entry.label():<24} skipped (read-only)")
    print(f"\n{len(outcomes) - failed} zeroed, {failed} failed")
    print(f"log: {log_path}")
    return 1 if failed else 0


def _parse_interface_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values or []:
        arm, _, interface = value.partition("=")
        if not arm or not interface:
            raise ConfigurationError(
                f"--interface takes ARM=IFACE (e.g. left=can2), got {value!r}"
            )
        overrides[arm] = interface
    return overrides


def _command_live(args: argparse.Namespace, log_path: Path) -> int:
    rig = resolve_rig(args.rig).with_interfaces(_parse_interface_overrides(args.interface))
    if args.only:
        rig = rig.subset(args.only)

    print(f"live: rig {rig.name} — {rig.description}")
    for rig_arm in rig.arms:
        effector = rig_arm.effector_model or "no effector"
        print(
            f"  {rig_arm.name:<8} {rig_arm.model:<6} {rig_arm.interface:<8} "
            f"{effector:<8} {rig_arm.role}"
        )
    if args.list:
        return 0

    if not args.skip_preflight:
        failures, reports = preflight_rig(rig)
        for name, results in reports:
            problems = [r for r in results if r.status != _OK]
            summary = "all checks pass" if not problems else f"{len(problems)} to look at"
            print(f"\npreflight {name}: {summary}")
            for result in problems:
                print(result.render())
        if failures:
            print(
                f"\n{failures} failed check(s); nothing was energized. "
                "Fix them, or re-run with --skip-preflight.",
                file=sys.stderr,
            )
            return 1

    print()
    status = run_live(
        rig,
        float_mode=args.float_mode,
        park=args.park,
        visualize=args.visualize,
        control=args.control,
        camera_preview=args.camera_preview,
        camera_overrides=cameras_mod.parse_camera_overrides(args.camera),
        port=args.port,
        mesh_dir=args.mesh_dir,
    )
    print(f"log: {log_path}")
    return status


def _command_infer(args: argparse.Namespace, log_path: Path) -> int:
    """Preflight and run the hardware-coupled MolmoAct2 loop."""
    rig = resolve_rig(args.rig).with_interfaces(_parse_interface_overrides(args.interface))
    if rig.names != ("left", "right"):
        raise ConfigurationError(
            "infer currently supports only the packaged bimanual left/right YAM rig"
        )

    if not args.skip_preflight:
        failures, reports = preflight_rig(rig)
        for name, results in reports:
            problems = [result for result in results if result.status != _OK]
            summary = "all checks pass" if not problems else f"{len(problems)} to look at"
            print(f"\npreflight {name}: {summary}")
            for result in problems:
                print(result.render())
        if failures:
            print(
                f"\n{failures} failed check(s); nothing was energized. "
                "Fix them, or re-run with --skip-preflight.",
                file=sys.stderr,
            )
            return 1

    status = run_infer(
        rig,
        instruction=args.instruction,
        server=args.server,
        request_timeout_s=args.request_timeout,
        num_steps=args.num_steps,
        enable_cuda_graph=args.cuda_graph,
        jpeg_quality=args.jpeg_quality,
        control_rate_hz=args.control_rate,
        max_step_rad=args.max_step_rad,
        max_effector_step=args.max_effector_step,
        reach_actions=args.reach_actions,
        speed=args.speed,
        carry_targets=args.carry_targets,
        prefetch=args.prefetch,
        prefetch_margin_s=args.prefetch_margin_s,
        reset_start_pose=args.reset_start_pose,
        park=args.park,
        visualize=args.visualize,
        camera_overrides=cameras_mod.parse_camera_overrides(args.camera),
        port=args.port,
        mesh_dir=args.mesh_dir,
    )
    print(f"log: {log_path}")
    return status


def _command_rollout(args: argparse.Namespace, log_path: Path) -> int:
    """Run interactive, labeled policy rollouts into a LeRobot dataset."""
    rig = resolve_rig(args.rig).with_interfaces(_parse_interface_overrides(args.interface))
    if rig.names != ("left", "right"):
        raise ConfigurationError(
            "rollout currently supports only the packaged bimanual left/right YAM rig"
        )

    if not args.skip_preflight:
        failures, reports = preflight_rig(rig)
        for name, results in reports:
            problems = [result for result in results if result.status != _OK]
            summary = "all checks pass" if not problems else f"{len(problems)} to look at"
            print(f"\npreflight {name}: {summary}")
            for result in problems:
                print(result.render())
        if failures:
            print(
                f"\n{failures} failed check(s); nothing was energized. "
                "Fix them, or re-run with --skip-preflight.",
                file=sys.stderr,
            )
            return 1

    status = run_rollout(
        rig,
        repo_id=args.repo_id,
        episodes=args.episodes,
        episode_seconds=args.episode_seconds,
        fps=args.fps,
        root=args.root,
        server=args.server,
        request_timeout_s=args.request_timeout,
        num_steps=args.num_steps,
        enable_cuda_graph=args.cuda_graph,
        jpeg_quality=args.jpeg_quality,
        speed=args.speed,
        chunk_size=args.chunk_size,
        max_step_rad=args.max_step_rad,
        max_effector_step=args.max_effector_step,
        carry_targets=args.carry_targets,
        prefetch=args.prefetch,
        prefetch_margin_s=args.prefetch_margin_s,
        visualize=args.visualize,
        camera_overrides=cameras_mod.parse_camera_overrides(args.camera),
        port=args.port,
        mesh_dir=args.mesh_dir,
        wait_between_episodes=args.wait_between_episodes,
    )
    print(f"log: {log_path}")
    return status


def _command_cameras(args: argparse.Namespace, log_path: Path) -> int:
    if args.snapshot is not None and not args.probe:
        raise ConfigurationError(
            "--snapshot needs --probe: a frame has to be grabbed before it can be written"
        )
    rig = resolve_rig(args.rig)
    if args.only:
        rig = rig.subset(args.only)
    overrides = cameras_mod.parse_camera_overrides(args.camera)

    print(f"cameras: rig {rig.name}")
    results = run_camera_checks(rig, overrides=overrides)
    for result in results:
        print(result.render())

    if args.probe:
        print()
        probed = probe_cameras(rig, overrides=overrides, snapshot_dir=args.snapshot)
        for result in probed:
            print(result.render())
        results += probed

    failures = sum(1 for result in results if result.status == _FAIL)
    warnings = sum(1 for result in results if result.status == _WARN)
    print(f"\n{len(results)} checks, {failures} failed, {warnings} warned")
    print(f"log: {log_path}")
    return 1 if failures else 0


def _command_record(args: argparse.Namespace, log_path: Path) -> int:
    if not args.dry_run and not args.repo_id:
        raise ConfigurationError(
            "record needs --repo-id (e.g. you/yam-fold-towel), or --dry-run to "
            "rehearse the session without writing a dataset"
        )
    if args.fps <= 0:
        raise ConfigurationError(f"--fps must be positive, got {args.fps}")

    overrides = cameras_mod.parse_camera_overrides(args.camera)
    rig = resolve_rig(args.rig).with_interfaces(_parse_interface_overrides(args.interface))
    if args.only:
        rig = rig.subset(args.only)
    if not args.cameras_enabled:
        rig = rig.without_cameras()
    # The run's capture settings, applied once here so the preflight validates
    # exactly the mode the recorder will open. Cameras run at the record rate:
    # a loop at 90 Hz reading 30 fps cameras would write each frame three times
    # and call it data. RGB straight from the SDK, because converting in numpy
    # costs 4.1 ms of an 11.1 ms tick with three cameras (see record.to_rgb).
    rig = rig.with_camera_capture(
        fps=args.camera_fps or args.fps, pixel_format="rgb8"
    )

    # A dataset carries its task string on every frame, and relabelling means
    # rewriting the dataset, so an unset --task is worth a word rather than a
    # silent default that ends up in a thousand episodes.
    task = args.task or "teleop"
    if not args.task:
        print("warning: no --task given; every frame will say 'teleop'", file=sys.stderr)

    print(f"record: rig {rig.name} — {rig.description}")
    for rig_arm in rig.arms:
        print(f"  {rig_arm.name:<8} {rig_arm.model:<6} {rig_arm.interface}")

    if not args.skip_preflight:
        failures, reports = preflight_rig(rig)
        camera_results = run_camera_checks(rig, overrides=overrides, required=True)
        # Asking for a rate the camera does not have should fail here, not after
        # two arms are energized and the first pipeline refuses to start.
        camera_results += check_camera_modes(rig)
        failures += sum(1 for result in camera_results if result.status == _FAIL)
        for name, results in reports:
            problems = [r for r in results if r.status != _OK]
            if problems:
                print(f"\npreflight {name}:")
                for result in problems:
                    print(result.render())
        camera_problems = [r for r in camera_results if r.status != _OK]
        if camera_problems:
            print("\npreflight cameras:")
            for result in camera_problems:
                print(result.render())
        if failures:
            print(
                f"\n{failures} failed check(s); nothing was energized. "
                "Fix them, or re-run with --skip-preflight.",
                file=sys.stderr,
            )
            return 1

    print()
    status = run_record(
        rig,
        task=task,
        repo_id=args.repo_id,
        teleop=args.teleop,
        fps=args.fps,
        num_episodes=args.num_episodes,
        hold_duration_s=args.hold_seconds,
        root=args.root,
        dry_run=args.dry_run,
        park=args.park,
        push_to_hub=args.push_to_hub,
        private=args.private,
        camera_overrides=overrides,
        vr_url=args.vr_url,
        vr_kit=args.vr_kit,
        yam_xml=args.yam_xml,
    )
    print(f"log: {log_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
