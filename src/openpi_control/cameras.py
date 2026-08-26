"""RealSense colour streams: which camera is which, and how to read one.

A camera's *identity* is its serial number -- that is the only thing about a
RealSense that survives a reboot, a replug, or being moved to another USB
port. Its ``/dev/video*` number does not, and even the stable-looking
``/dev/v4l/by-id`` path only tells you a serial, never whether that camera is
looking down at the table or riding on the right wrist. That last fact is a
property of the cell, so it lives in the rig definition next to the CAN
interfaces (see :mod:`openpi_control.rigs`), and this module is what turns it
into a device path you can open.

Discovery is pure filesystem enumeration: glob ``/dev/v4l/by-id``, pull the
serial out of each entry's name, keep the colour node. No ``pyrealsense2``, no
``v4l2-ctl``, nothing that has to be installed for ``doctor`` to tell you a
camera is unplugged:

    from openpi_control.cameras import discover
    from openpi_control.rigs import resolve_rig

    for found in discover(resolve_rig("yam_bimanual").cameras).matched.values():
        print(found.camera.name, found.device)

:class:`CameraReader` is the other half -- an actual capture thread -- and it
goes through the RealSense SDK rather than OpenCV, because on this cell OpenCV's
V4L2 path delivers 10-13 fps where the SDK delivers 30 (see the class docstring).
Both halves are keyed off the same serial, but note that a D405 answers to two
different numbers -- see :func:`sdk_serial_for_asic`. The SDK is imported lazily,
so discovery, ``doctor``, and everything above stay usable without the
``cameras`` extra installed.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .camera_poses import CameraExtrinsic
from .exceptions import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

# Where udev publishes its stable per-device symlinks. Overridable in tests and
# on the odd system that mounts devtmpfs somewhere else.
#
# Redirecting this (or passing ``by_id_dir``) means "this directory is the whole
# bus", and discovery honours that literally: the SDK fallback for cameras udev
# cannot name is only consulted when the real system directory is in play.
# Otherwise whatever happens to be plugged into the machine running the tests
# would leak into their results.
SYSTEM_BY_ID_DIR = Path("/dev/v4l/by-id")
BY_ID_DIR = SYSTEM_BY_ID_DIR

# A D405 publishes six v4l2 nodes; index 4 is the colour stream. Other
# RealSense models enumerate differently, so a camera that needs another node
# says so with its own ``color_index``.
DEFAULT_COLOR_INDEX = 4

# by-id entries look like:
#   usb-Intel_R__RealSense_TM__Depth_Camera_405_..._254623070531-video-index4
# Capture the serial (the digits before ``-video-index``) and the node number.
#
# At least six digits, because a camera whose USB descriptor carries no serial
# at all gets a by-id name that ends in its model number instead --
# ``..._Depth_Camera_435_..._Depth_Camera_435-video-index0`` -- and a lazy
# pattern happily reads that ``435`` as the serial. That invents a camera which
# is not there and hides the one that is (see `sdk_present_asic_serials`).
_BY_ID_RE = re.compile(r"RealSense.*?_(\d{6,})-video-index(\d+)$")

# Capture defaults. 848x480 is the D405's native colour mode, and that is why
# it is the default rather than the rounder-looking 640x480: asking for 640x480
# makes the firmware rescale, and three cameras doing that concurrently drop to
# 15-20 fps where all three hold a full 30 at 848x480. Crop or resize
# downstream if a policy wants a different aspect -- it is much cheaper there
# than in the camera.
DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30

# Rotations a capture thread can apply. Anything else is a typo, not a request.
VALID_ROTATIONS = (0, 90, 180, 270)

# Pixel formats a reader can be asked for. The SDK converts in native code as
# part of the frame it hands over, so asking for the layout you actually want is
# free -- see `record.to_rgb` for what doing it afterwards costs. bgr8 is the
# default because OpenCV's imwrite and the browser preview expect it; a dataset
# recorder asks for rgb8 and then needs no conversion at all.
VALID_PIXEL_FORMATS = ("bgr8", "rgb8")
DEFAULT_PIXEL_FORMAT = "bgr8"


# How long to keep retrying a camera that reports itself busy, and how often.
# The kernel holds a v4l2 node for a moment after a stream stops, so two
# back-to-back runs would otherwise collide with each other rather than with a
# real second consumer.
BUSY_RETRY_S = 3.0
BUSY_RETRY_INTERVAL_S = 0.25


@dataclass(frozen=True, slots=True)
class RigCamera:
    """One camera of a rig: what it is, where it looks, and what it rides on.

    ``name`` is the key everything downstream uses -- the observation key in a
    recorded dataset, the column in ``openpi-control cameras``, the argument to
    ``--camera NAME=DEVICE``. Renaming one rewrites datasets, so treat it as
    part of the rig's contract.

    ``arm`` ties a wrist camera to the arm it is bolted to, which is what lets
    ``--only right`` drop the left wrist camera without anyone special-casing
    it. A camera that watches the whole cell rather than one arm leaves it
    ``None``.

    ``rotate`` is applied in the capture thread, so every consumer -- browser,
    recorder, policy -- sees the same corrected frame. A wrist camera mounted
    sideways is a mechanical fact; fixing it once here beats fixing it in three
    places later.

    ``extrinsic`` is optional calibration metadata for scene visualization. It
    does not alter the policy image or camera discovery.
    """

    name: str
    serial: str
    label: str
    arm: str | None = None
    rotate: int = 0
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    color_index: int = DEFAULT_COLOR_INDEX
    extrinsic: CameraExtrinsic | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("a rig camera needs a name")
        if not self.serial.isdigit():
            raise ConfigurationError(
                f"camera {self.name!r} has serial {self.serial!r}; RealSense serials are "
                "digits, exactly as they appear in the /dev/v4l/by-id path"
            )
        if self.rotate not in VALID_ROTATIONS:
            raise ConfigurationError(
                f"camera {self.name!r} asks for rotate={self.rotate}; "
                f"valid rotations are {', '.join(str(r) for r in VALID_ROTATIONS)}"
            )
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ConfigurationError(
                f"camera {self.name!r} needs a positive width, height, and fps"
            )
        if self.pixel_format not in VALID_PIXEL_FORMATS:
            raise ConfigurationError(
                f"camera {self.name!r} asks for pixel format {self.pixel_format!r}; "
                f"supported: {', '.join(VALID_PIXEL_FORMATS)}"
            )


@dataclass(frozen=True, slots=True)
class FoundCamera:
    """A rig camera matched to a device path that exists right now."""

    camera: RigCamera
    device: str
    #: True when an operator pinned the path with ``--camera NAME=DEVICE``
    #: instead of it being found by serial.
    overridden: bool = False

    def spec(self) -> CameraSpec:
        return CameraSpec(
            name=self.camera.name,
            label=self.camera.label,
            serial=self.camera.serial,
            device=self.device,
            width=self.camera.width,
            height=self.camera.height,
            fps=self.camera.fps,
            rotate=self.camera.rotate,
            pixel_format=self.camera.pixel_format,
        )


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """What the bus looks like, relative to what the rig asked for."""

    #: camera name -> the device it resolved to, in rig order.
    matched: dict[str, FoundCamera] = field(default_factory=dict)
    #: camera name -> serial, for rig cameras with nothing on the bus.
    missing: dict[str, str] = field(default_factory=dict)
    #: serials present on the bus that no rig camera claims.
    unclaimed: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing

    def specs(self) -> tuple[CameraSpec, ...]:
        return tuple(found.spec() for found in self.matched.values())


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """Everything :class:`CameraReader` needs to open one stream.

    ``serial`` is what actually selects the camera -- the SDK addresses devices,
    not device files. ``device`` is carried along for diagnostics: it is the
    v4l2 node the presence check found, and it is what an operator sees in
    ``dmesg`` or hands to ``v4l2-ctl``.
    """

    name: str
    label: str
    serial: str
    device: str
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    rotate: int = 0
    pixel_format: str = DEFAULT_PIXEL_FORMAT

    def __post_init__(self) -> None:
        # RigCamera validates the same fields, but a spec can also be built by
        # hand -- and this is the type CameraReader consumes, so an unchecked
        # value here surfaces as an SDK AttributeError or a KeyError deep in the
        # capture thread rather than as a configuration error.
        if self.pixel_format not in VALID_PIXEL_FORMATS:
            raise ConfigurationError(
                f"camera {self.name!r} asks for pixel format {self.pixel_format!r}; "
                f"supported: {', '.join(VALID_PIXEL_FORMATS)}"
            )
        if self.rotate not in VALID_ROTATIONS:
            raise ConfigurationError(
                f"camera {self.name!r} asks for rotate={self.rotate}; valid: "
                f"{', '.join(str(rotation) for rotation in VALID_ROTATIONS)}"
            )
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ConfigurationError(
                f"camera {self.name!r} needs a positive width, height, and fps"
            )


def present_serials(by_id_dir: Path | None = None) -> dict[tuple[str, int], str]:
    """Every RealSense v4l2 node udev is currently publishing.

    Keyed by ``(serial, node_index)`` rather than by serial alone because the
    node number is how a colour stream is told apart from the depth and IR
    streams of the same physical camera.

    ``by_id_dir`` defaults to :data:`BY_ID_DIR`, resolved per call rather than
    bound at import, so a test can point the whole module at a fake bus.
    """
    root = BY_ID_DIR if by_id_dir is None else by_id_dir
    found: dict[tuple[str, int], str] = {}
    try:
        entries = sorted(root.iterdir())
    except OSError:
        # No by-id directory at all: no cameras, or a system that does not run
        # udev. Either way the answer is "nothing found", not an error -- the
        # caller reports it as a missing camera with the serial it wanted.
        return found
    for entry in entries:
        match = _BY_ID_RE.search(entry.name)
        if match is not None:
            found[(match.group(1), int(match.group(2)))] = str(entry)
    return found


def sdk_present_asic_serials() -> dict[str, str]:
    """``{asic_serial: sdk_serial}`` for every RealSense the SDK can see.

    The second source of truth for "is this camera plugged in". udev is the
    first and the cheaper one, but it can only answer for a camera whose USB
    descriptor carries a serial, and not every RealSense does: this cell's D435
    publishes its nodes as ``..._Depth_Camera_435-video-index0`` with no serial
    anywhere in the name, and its colour node gets no by-id entry at all. The
    SDK still addresses it perfectly well, which is the only thing that has to
    work -- device paths here are diagnostics, never how a stream is opened.

    Returns ``{}`` when the SDK is not installed, so discovery keeps working on
    a machine that only has udev.
    """
    try:
        rs = _require_realsense()
    except ConfigurationError:
        return {}
    found: dict[str, str] = {}
    try:
        devices = rs.context().query_devices()
    except Exception:  # noqa: BLE001 - a wedged SDK must not fail discovery
        return {}
    for device in devices:
        asic = rs.camera_info.asic_serial_number
        serial = rs.camera_info.serial_number
        if not device.supports(asic) or not device.supports(serial):
            continue
        found[str(device.get_info(asic))] = str(device.get_info(serial))
    return found


def discover(
    cameras: Iterable[RigCamera],
    *,
    overrides: Mapping[str, str] | None = None,
    by_id_dir: Path | None = None,
) -> DiscoveryResult:
    """Resolve each rig camera to a device path, by serial.

    ``overrides`` pins a camera to an explicit path, the camera equivalent of
    ``--interface left=can2``: it wins over discovery, and a path that is not
    there is reported missing rather than silently ignored, because an operator
    who names a device meant that one.
    """
    cameras = tuple(cameras)
    overrides = dict(overrides or {})
    unknown = set(overrides).difference(camera.name for camera in cameras)
    if unknown:
        known = ", ".join(camera.name for camera in cameras) or "none"
        raise ConfigurationError(
            f"no camera named {', '.join(sorted(unknown))} in this rig; it has: {known}"
        )

    nodes = present_serials(by_id_dir)
    # Only consulted for cameras udev could not name, so the common path stays
    # a directory listing and no rig pays for an SDK query it does not need.
    root = BY_ID_DIR if by_id_dir is None else by_id_dir
    sdk_serials: dict[str, str] | None = None if root == SYSTEM_BY_ID_DIR else {}
    matched: dict[str, FoundCamera] = {}
    missing: dict[str, str] = {}
    for camera in cameras:
        pinned = overrides.get(camera.name)
        if pinned is not None:
            if Path(pinned).exists():
                matched[camera.name] = FoundCamera(camera, pinned, overridden=True)
            else:
                missing[camera.name] = camera.serial
            continue
        device = nodes.get((camera.serial, camera.color_index))
        if device is None:
            if sdk_serials is None:
                sdk_serials = sdk_present_asic_serials()
            sdk_serial = sdk_serials.get(camera.serial)
            if sdk_serial is None:
                missing[camera.name] = camera.serial
            else:
                matched[camera.name] = FoundCamera(camera, f"sdk:{sdk_serial}")
        else:
            matched[camera.name] = FoundCamera(camera, device)

    claimed = {camera.serial for camera in cameras}
    present = {serial for serial, _ in nodes}
    if sdk_serials is None and not claimed.issubset(present):
        sdk_serials = sdk_present_asic_serials()
    present |= set(sdk_serials or {})
    unclaimed = tuple(sorted(present - claimed))
    return DiscoveryResult(matched=matched, missing=missing, unclaimed=unclaimed)


class CameraReader:
    """A background grabber holding the newest frame from one camera.

    Capture goes through ``pyrealsense2`` rather than OpenCV's V4L2 path, and
    that is a measured decision, not a preference. On this cell's D405s, reading
    the colour node through ``cv2.VideoCapture`` tops out around 10-13 fps at
    848x480 -- ``v4l2-ctl`` streams the same node at 30, so the ceiling is in
    OpenCV's UVC consumer, not the camera or the bus. Through the SDK all three
    cameras hold 30 fps at once.

    Latest-frame-wins on purpose: a control loop or a dataset recorder wants the
    freshest image at the moment it asks, never a queue of stale ones that grows
    whenever the consumer falls behind.

    A camera can only be streamed by one process at a time, so exactly one
    consumer may hold a given camera -- a recorder and a browser preview cannot
    both have it.
    """

    def __init__(self, spec: CameraSpec) -> None:
        rs = _require_realsense()
        self.spec = spec
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frames_read = 0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._started = threading.Event()

        config = rs.config()
        # The SDK addresses devices by its own ``serial_number``, which is not
        # the serial in the udev path -- that one is the ASIC serial. Resolving
        # here keeps the rig declaring the number an operator can actually read
        # off ``/dev/v4l/by-id`` (and out of vr-teleop-kit's cams.env).
        config.enable_device(sdk_serial_for_asic(spec.serial))
        config.enable_stream(
            rs.stream.color,
            spec.width,
            spec.height,
            {"bgr8": rs.format.bgr8, "rgb8": rs.format.rgb8}[spec.pixel_format],
            spec.fps,
        )
        self._pipeline = rs.pipeline()
        self._profile = self._start(config, spec)

        self._thread = threading.Thread(
            target=self._loop, name=f"camera-{spec.name}", daemon=True
        )
        self._thread.start()

    def _start(self, config, spec: CameraSpec):  # noqa: ANN001, ANN202 - SDK types
        """Start the pipeline, retrying briefly while the device is still busy.

        Closing a stream does not free the v4l2 node instantly -- the kernel
        holds it for a moment after the SDK lets go. Without this, re-running a
        command right after the previous one exits fails with "Device or
        resource busy", which looks like a hardware fault and is not one. A
        camera genuinely held by another process still fails, just a second
        later.
        """
        deadline = time.monotonic() + BUSY_RETRY_S
        while True:
            try:
                return self._pipeline.start(config)
            except RuntimeError as err:
                busy = "busy" in str(err).lower()
                if not busy or time.monotonic() >= deadline:
                    hint = (
                        " — another process is streaming it (a camera opens once)"
                        if busy
                        else ""
                    )
                    raise ConfigurationError(
                        f"cannot start camera {spec.name!r} (serial {spec.serial}) at "
                        f"{spec.width}x{spec.height}@{spec.fps}: {err}{hint}"
                    ) from err
                time.sleep(BUSY_RETRY_INTERVAL_S)

    @property
    def pixel_format(self) -> str:
        """The channel order :meth:`latest` hands back. Read it, do not assume:
        converting an already-RGB frame swaps the channels back the wrong way."""
        return self.spec.pixel_format

    @property
    def frames_read(self) -> int:
        with self._lock:
            return self._frames_read

    @property
    def negotiated(self) -> tuple[str, int, int, int]:
        """``(format, width, height, fps)`` as the SDK actually started it.

        Requested capture parameters are a request: the SDK picks the nearest
        profile it can serve. Reporting what it settled on is the difference
        between "the camera is slow" and "the camera never agreed to 30 fps".
        """
        stream = self._profile.get_stream(_require_realsense().stream.color)
        video = stream.as_video_stream_profile()
        return (
            str(stream.format()).rsplit(".", 1)[-1],
            video.width(),
            video.height(),
            video.fps(),
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                # A timeout is not fatal: USB hiccups, and the consumer keeps
                # the last good frame. A camera that is genuinely gone shows up
                # as a stale `latest()`, which the recorder checks for.
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            # asanyarray wraps SDK-owned memory that is recycled the moment
            # this frame is released, so the copy is not optional. _rotate
            # already copies, which is why it is the only branch that does not.
            view = np.asanyarray(color.get_data()).reshape(
                color.get_height(), color.get_width(), 3
            )
            frame = _rotate(view, self.spec.rotate) if self.spec.rotate else view.copy()
            with self._lock:
                self._frame = frame
                self._frames_read += 1
            self._started.set()

    def latest(self) -> np.ndarray | None:
        """The newest frame, in :attr:`pixel_format`, or None before the first."""
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout_s: float = 5.0) -> np.ndarray | None:
        """Block until a frame lands, or ``timeout_s`` passes.

        A camera needs a moment after the pipeline starts before the first
        frame appears, and a recorder that began writing before then would put
        a hole at the front of every episode.
        """
        self._started.wait(timeout_s)
        return self.latest()

    def measure_fps(self, window_s: float = 2.0) -> float:
        """Frames per second actually delivered over ``window_s``.

        Counts from now, so let the stream settle first: a window that starts
        at the very first frame measures the ramp-up and under-reports a
        healthy camera by a factor of two.
        """
        before = self.frames_read
        time.sleep(window_s)
        return (self.frames_read - before) / window_s

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._pipeline.stop()
        except RuntimeError:
            # Already stopped, or the device went away underneath us. Either
            # way there is nothing left to release.
            pass

    def __enter__(self) -> CameraReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def open_readers(specs: Iterable[CameraSpec]) -> dict[str, CameraReader]:
    """Open every spec, or none of them.

    Half-open camera sets are the kind of thing that produces a dataset with
    one camera silently absent, so a failure here releases what was already
    opened and re-raises.
    """
    readers: dict[str, CameraReader] = {}
    try:
        for spec in specs:
            readers[spec.name] = CameraReader(spec)
    except BaseException:
        for reader in readers.values():
            reader.stop()
        raise
    return readers


def close_readers(readers: Mapping[str, CameraReader]) -> None:
    """Stop every reader, even if one of them raises on the way down."""
    for reader in readers.values():
        try:
            reader.stop()
        except Exception:  # noqa: BLE001 - a stuck camera must not keep the rest open
            pass


def write_snapshot(path: Path, frame: np.ndarray) -> None:
    """Write one BGR frame to ``path`` as an image.

    Exists so that "show me what the right wrist is looking at" does not oblige
    the caller to import OpenCV itself just to encode a PNG.
    """
    if not _require_cv2().imwrite(str(path), frame):
        raise ConfigurationError(f"could not write a snapshot to {path}")


def _rotate(frame: np.ndarray, rotate: int) -> np.ndarray:
    """Rotate a frame clockwise by 90/180/270 degrees.

    numpy rather than ``cv2.rotate`` so that capture needs no OpenCV at all --
    only writing a snapshot does. ``np.rot90`` returns a view with permuted
    strides, and video encoders want contiguous memory, hence the copy.
    """
    turns = {90: -1, 180: 2, 270: 1}[rotate]
    return np.ascontiguousarray(np.rot90(frame, k=turns))


def sdk_serial_for_asic(asic_serial: str) -> str:
    """Translate a udev/ASIC serial into the serial the RealSense SDK uses.

    A D405 answers to two numbers. ``/dev/v4l/by-id`` (and the USB descriptor,
    and vr-teleop-kit's ``cams.env``) carry the *ASIC* serial -- e.g.
    ``254623070531``. The SDK's own ``serial_number`` is a different value --
    e.g. ``352122273221`` -- and that is the one ``enable_device`` wants. Rigs
    declare the ASIC serial because that is the one an operator can actually
    look up without the SDK installed; this bridges the two.
    """
    rs = _require_realsense()
    available: list[str] = []
    for device in rs.context().query_devices():
        info = rs.camera_info.asic_serial_number
        if not device.supports(info):
            continue
        found = str(device.get_info(info))
        available.append(found)
        if found == asic_serial:
            return str(device.get_info(rs.camera_info.serial_number))
    raise ConfigurationError(
        f"no RealSense with ASIC serial {asic_serial} is connected; the SDK sees "
        f"{', '.join(available) if available else 'none'}"
    )


def supported_color_modes(serial: str) -> dict[tuple[int, int], tuple[int, ...]]:
    """``{(width, height): (fps, ...)}`` the camera's colour stream offers.

    Enumeration only -- it opens no stream, so this is cheap enough to run in a
    preflight. Used to answer "you asked for 45 fps" with the rates the device
    actually has instead of letting the SDK fail with a bare ioctl error.
    """
    rs = _require_realsense()
    target = sdk_serial_for_asic(serial)
    modes: dict[tuple[int, int], set[int]] = {}
    for device in rs.context().query_devices():
        if str(device.get_info(rs.camera_info.serial_number)) != target:
            continue
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue
                video = profile.as_video_stream_profile()
                modes.setdefault((video.width(), video.height()), set()).add(video.fps())
    return {size: tuple(sorted(rates)) for size, rates in sorted(modes.items())}


def _require_realsense():  # noqa: ANN202 - the pyrealsense2 module, Any by design
    """The RealSense SDK, or a message that says how to get it."""
    try:
        import pyrealsense2 as rs
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            "reading a camera needs the RealSense SDK, which is not installed: "
            "uv sync --extra cameras"
        ) from err
    return rs


def _require_cv2():  # noqa: ANN202 - the cv2 module, typed as Any by design
    """OpenCV, or a message that says how to get it.

    Kept behind a function so ``doctor``, ``zero``, and camera discovery all
    work on a box with no OpenCV installed -- the parts of this module that
    only read udev have no business requiring a capture library.
    """
    try:
        import cv2
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            "reading a camera needs OpenCV, which is not installed: "
            "uv sync --extra cameras"
        ) from err
    return cv2


def parse_camera_overrides(values: Iterable[str] | None) -> dict[str, str]:
    """``["top=/dev/video4"]`` -> ``{"top": "/dev/video4"}``.

    Mirrors ``--interface ARM=IFACE`` so the two override flags behave the same
    way, typos included: a malformed pair fails loudly instead of leaving the
    camera on whatever discovery happened to find.
    """
    overrides: dict[str, str] = {}
    for value in values or []:
        name, _, device = value.partition("=")
        if not name or not device:
            raise ConfigurationError(
                f"--camera takes NAME=DEVICE (e.g. top=/dev/video4), got {value!r}"
            )
        overrides[name] = device
    return overrides
