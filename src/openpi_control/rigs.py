"""Packaged multi-arm rigs: which arms a cell has, on which bus, and where.

An :class:`~openpi_control.config.ArmConfig` describes one arm. A rig names the
whole cell -- both arms of a bimanual YAM, the CAN interface each sits on, where
their bases sit relative to each other, and which cameras watch it -- so the
operator CLI, the visualizer, and the dataset recorder agree on what ``left``
and ``right`` mean instead of each growing its own pile of flags.

Rigs are pure configuration. Resolving one opens no bus, starts no
``pi_control_node``, and reads nothing but the packaged model JSON:

    from openpi_control.rigs import resolve_rig

    rig = resolve_rig("yam_bimanual")
    for arm in rig.followers:
        print(arm.name, arm.interface)

Interfaces are named per rig because that is how a cell is wired, but they are
the one field that genuinely varies between two otherwise identical rigs, so
:meth:`Rig.with_interfaces` overrides them without redefining the rig.

Cameras are named the same way, by the one thing about them that is stable --
their serial number. Which arm a wrist camera rides on is part of that
declaration, so narrowing a rig to one arm takes its wrist camera with it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .camera_poses import YAM_TOP_CAMERA_EXTRINSIC
from .cameras import RigCamera
from .config import (
    SUPPORTED_EFFECTORS,
    SUPPORTED_MODELS,
    ArmConfig,
    connection_for_interface,
)
from .exceptions import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

ROLE_FOLLOWER = "follower"
ROLE_LEADER = "leader"
ROLES = (ROLE_FOLLOWER, ROLE_LEADER)

# Base separation of the two YAM arms in the bimanual scene, in metres. This is
# a layout default for rendering and for telling the arms apart on screen, not a
# measured calibration of anyone's table -- nothing commands a pose from it. A
# cell that is built to a different width overrides the base positions.
YAM_BIMANUAL_SEPARATION_M = 0.61

# Serial numbers of the RealSenses on the bimanual cell. A serial is the only
# handle on a camera that survives a replug, so it -- not a /dev path -- is what
# the rig pins. Swap a camera, edit the serial here; nothing else in the tree
# needs to know.
#
# These are ASIC serials, the number in the /dev/v4l/by-id path, because that is
# the one an operator can read off the bus without the SDK installed.
# `cameras.sdk_serial_for_asic` bridges to the SDK's own serial when a stream is
# actually opened.
#
# The wrists are D405s. The top is a D435 and differs in two ways that the rig
# has to carry: it publishes no serial in its USB descriptor at all (so udev
# cannot name it and discovery falls back to the SDK), and its colour sensor has
# no 848x480 mode, so it captures at 640x480 -- see YAM_TOP_CAPTURE below.
YAM_BIMANUAL_CAMERA_SERIALS = {
    "top": "348523020354",
    "left_wrist": "254623070863",
    "right_wrist": "254623070417",
}

# 640x480 is a USB 2.0 fallback, not this camera's real capability, and it
# should go away rather than be preserved.
#
# Enumerated on 2026-08-23 the top D435 offered only 424x240, 640x480,
# 1280x720@15 and 1920x1080@8 -- the reduced set a D435 falls back to when it
# negotiates USB 2.0. On USB 3 it also offers 848x480@30 and 640x360@30, and
# 848x480 is both the rig default and 16:9. That matters beyond frame rate:
# MolmoAct2's training frames are 640x360, so every other view this cell feeds
# the policy is 16:9 and 640x480 is the one that is 4:3.
#
# So: re-seat this camera on a USB 3 port, re-run `openpi-control cameras`, and
# if 848x480@30 is in `cameras.supported_color_modes("348523020354")`, delete
# this override and let the top camera take the rig default like the wrists.
YAM_TOP_CAPTURE = {"width": 640, "height": 480}


@dataclass(frozen=True, slots=True)
class RigArm:
    """One arm of a rig: its model, its bus, its role, and its base pose.

    ``base_position`` and ``base_rotation_wxyz`` place the arm in the shared
    scene frame. They are consumed by the visualizer only; the native node takes
    its own base orientation from the instance JSON's ``base_rpy``.
    """

    name: str
    model: str
    interface: str
    role: str = ROLE_FOLLOWER
    effector_model: str | None = None
    instance_config: Path | None = None
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ConfigurationError(
                f"arm {self.name!r} has role {self.role!r}; expected one of {', '.join(ROLES)}"
            )
        if self.model not in SUPPORTED_MODELS:
            raise ConfigurationError(
                f"arm {self.name!r} uses unknown model {self.model!r}; "
                f"supported: {', '.join(SUPPORTED_MODELS)}"
            )
        if self.effector_model is not None and self.effector_model not in SUPPORTED_EFFECTORS:
            raise ConfigurationError(
                f"arm {self.name!r} uses unknown effector {self.effector_model!r}; "
                f"supported: {', '.join(SUPPORTED_EFFECTORS)}"
            )

    @property
    def is_follower(self) -> bool:
        return self.role == ROLE_FOLLOWER

    def arm_config(self) -> ArmConfig:
        """The ArmConfig a session would be handed for this arm.

        The transport is inferred from the interface string exactly as the
        single-arm CLI does it, so a rig cannot drift from ``doctor``'s view of
        the same interface.
        """
        return ArmConfig(
            self.name,
            self.model,
            connection_for_interface(self.interface),
            effector_model=self.effector_model,
            instance_config=self.instance_config,
        )

    def with_interface(self, interface: str) -> RigArm:
        return dataclasses.replace(self, interface=interface)


@dataclass(frozen=True, slots=True)
class Rig:
    """A named set of arms that are operated together."""

    name: str
    description: str
    arms: tuple[RigArm, ...] = field(default_factory=tuple)
    cameras: tuple[RigCamera, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ConfigurationError(f"rig {self.name!r} declares no arms")
        seen: set[str] = set()
        for arm in self.arms:
            if arm.name in seen:
                raise ConfigurationError(
                    f"rig {self.name!r} declares two arms named {arm.name!r}"
                )
            seen.add(arm.name)
        camera_names: set[str] = set()
        camera_serials: dict[str, str] = {}
        for camera in self.cameras:
            if camera.name in camera_names:
                raise ConfigurationError(
                    f"rig {self.name!r} declares two cameras named {camera.name!r}"
                )
            camera_names.add(camera.name)
            # Two cameras on one serial means a copy-paste in the rig, and it
            # would silently record the same view twice under two keys.
            if camera.serial in camera_serials:
                raise ConfigurationError(
                    f"rig {self.name!r} gives serial {camera.serial} to both "
                    f"{camera_serials[camera.serial]!r} and {camera.name!r}"
                )
            camera_serials[camera.serial] = camera.name
            if camera.arm is not None and camera.arm not in seen:
                raise ConfigurationError(
                    f"rig {self.name!r} mounts camera {camera.name!r} on arm "
                    f"{camera.arm!r}, which it does not have; it holds: "
                    f"{', '.join(arm.name for arm in self.arms)}"
                )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(arm.name for arm in self.arms)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)

    @property
    def followers(self) -> tuple[RigArm, ...]:
        return tuple(arm for arm in self.arms if arm.role == ROLE_FOLLOWER)

    @property
    def leaders(self) -> tuple[RigArm, ...]:
        return tuple(arm for arm in self.arms if arm.role == ROLE_LEADER)

    def __getitem__(self, name: str) -> RigArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ConfigurationError(
            f"rig {self.name!r} has no arm {name!r}; it holds: {', '.join(self.names)}"
        )

    def with_interfaces(self, overrides: Mapping[str, str]) -> Rig:
        """Copy of this rig with some arms moved to different interfaces.

        Every key must name an arm in the rig: a typo in ``--interface`` should
        fail loudly rather than silently leave the arm on its default bus.
        """
        unknown = set(overrides).difference(self.names)
        if unknown:
            raise ConfigurationError(
                f"rig {self.name!r} has no arm named {', '.join(sorted(unknown))}; "
                f"it holds: {', '.join(self.names)}"
            )
        return dataclasses.replace(
            self,
            arms=tuple(
                arm.with_interface(overrides[arm.name]) if arm.name in overrides else arm
                for arm in self.arms
            ),
        )

    def subset(self, names: Iterable[str]) -> Rig:
        """The same rig narrowed to some of its arms, keeping their order.

        Used by ``--only`` to bring up one arm of a bimanual cell without
        inventing a second single-arm rig that would then drift from this one.
        """
        wanted = list(dict.fromkeys(names))
        for name in wanted:
            self[name]  # raises with the rig's arm list when the name is wrong
        if not wanted:
            raise ConfigurationError(f"no arms selected from rig {self.name!r}")
        keep = set(wanted)
        return dataclasses.replace(
            self,
            arms=tuple(arm for arm in self.arms if arm.name in keep),
            # A wrist camera goes wherever its arm goes: recording a left-wrist
            # view of an arm that is not powered would put a frozen image in
            # every frame of the dataset.
            cameras=tuple(
                camera
                for camera in self.cameras
                if camera.arm is None or camera.arm in keep
            ),
        )

    def without_cameras(self) -> Rig:
        """The same rig with no cameras, for a state-only run."""
        return dataclasses.replace(self, cameras=())

    def with_camera_capture(
        self, *, fps: int | None = None, pixel_format: str | None = None
    ) -> Rig:
        """Copy of this rig with different capture settings on every camera.

        Rate and pixel format are properties of a *run*, not of the cell: a
        recorder wants the highest rate the cameras will give and RGB straight
        from the SDK, while a browser preview wants neither. The rig keeps the
        defaults; a command overrides them here rather than each camera being
        redeclared.
        """
        changes: dict[str, object] = {}
        if fps is not None:
            changes["fps"] = fps
        if pixel_format is not None:
            changes["pixel_format"] = pixel_format
        if not changes:
            return self
        return dataclasses.replace(
            self,
            cameras=tuple(
                dataclasses.replace(camera, **changes) for camera in self.cameras
            ),
        )


def _yam_bimanual() -> Rig:
    """Two YAM followers, each with its own gripper, on adjacent CAN buses.

    +Y is the cell's left, matching the arms' own base frames, so the scene
    reads the way an operator standing behind the cell sees it.

    Three RealSense D405s watch it: one overhead, one on each wrist. The two
    wrist cameras name their arm, so ``--only right`` brings up one arm with the
    top and right-wrist views and nothing else.
    """
    half = YAM_BIMANUAL_SEPARATION_M / 2.0
    return Rig(
        name="yam_bimanual",
        description="two YAM followers, each with an E_Yam gripper, and three D405s",
        arms=(
            RigArm(
                name="left",
                model="Yam",
                interface="can0",
                effector_model="E_Yam",
                base_position=(0.0, half, 0.0),
            ),
            RigArm(
                name="right",
                model="Yam",
                interface="can1",
                effector_model="E_Yam",
                base_position=(0.0, -half, 0.0),
            ),
        ),
        cameras=(
            RigCamera(
                name="top",
                serial=YAM_BIMANUAL_CAMERA_SERIALS["top"],
                label="Top-down",
                extrinsic=YAM_TOP_CAMERA_EXTRINSIC,
                **YAM_TOP_CAPTURE,
            ),
            RigCamera(
                name="left_wrist",
                serial=YAM_BIMANUAL_CAMERA_SERIALS["left_wrist"],
                label="Left wrist",
                arm="left",
            ),
            RigCamera(
                name="right_wrist",
                serial=YAM_BIMANUAL_CAMERA_SERIALS["right_wrist"],
                label="Right wrist",
                arm="right",
            ),
        ),
    )


PACKAGED_RIGS: dict[str, Rig] = {rig.name: rig for rig in (_yam_bimanual(),)}


def rig_names() -> tuple[str, ...]:
    return tuple(PACKAGED_RIGS)


def resolve_rig(name: str) -> Rig:
    try:
        return PACKAGED_RIGS[name]
    except KeyError:
        raise ConfigurationError(
            f"unknown rig {name!r}; packaged rigs are: {', '.join(rig_names())}"
        ) from None
