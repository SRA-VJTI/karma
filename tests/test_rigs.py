"""Packaged rigs: pure configuration, so none of this touches a bus."""

from __future__ import annotations

import pytest

from openpi_control.cameras import RigCamera
from openpi_control.config import SocketCanConnection
from openpi_control.exceptions import ConfigurationError
from openpi_control.rigs import (
    PACKAGED_RIGS,
    ROLE_FOLLOWER,
    ROLE_LEADER,
    YAM_BIMANUAL_SEPARATION_M,
    Rig,
    RigArm,
    resolve_rig,
    rig_names,
)


def test_every_packaged_rig_resolves_and_builds_arm_configs() -> None:
    # A rig that names a model or effector the catalog does not have would only
    # fail at bring-up, with two arms already half energized.
    assert rig_names()
    for name in rig_names():
        rig = resolve_rig(name)
        assert rig.name == name
        assert rig.description
        for arm in rig.arms:
            config = arm.arm_config()
            assert config.name == arm.name
            assert config.model == arm.model


def test_unknown_rig_names_the_ones_that_exist() -> None:
    with pytest.raises(ConfigurationError, match="yam_bimanual"):
        resolve_rig("yam_trimanual")


def test_yam_bimanual_is_two_followers_on_adjacent_can_buses() -> None:
    rig = resolve_rig("yam_bimanual")

    assert rig.names == ("left", "right")
    assert [arm.interface for arm in rig.arms] == ["can0", "can1"]
    assert {arm.effector_model for arm in rig.arms} == {"E_Yam"}
    assert len(rig.followers) == 2
    assert rig.leaders == ()
    for arm in rig.arms:
        assert isinstance(arm.arm_config().connection, SocketCanConnection)


def test_yam_bimanual_bases_are_separated_and_centred() -> None:
    # The two arms must not render on top of each other, and +Y is the cell's
    # left so the scene reads the way the operator sees it.
    left, right = resolve_rig("yam_bimanual").arms
    assert left.base_position[1] > 0 > right.base_position[1]
    assert left.base_position[1] - right.base_position[1] == pytest.approx(
        YAM_BIMANUAL_SEPARATION_M
    )
    assert left.base_position[0] == right.base_position[0] == 0.0


def test_rig_rejects_an_unknown_model_or_effector() -> None:
    with pytest.raises(ConfigurationError, match="unknown model"):
        RigArm(name="a", model="Yam2", interface="can0")
    with pytest.raises(ConfigurationError, match="unknown effector"):
        RigArm(name="a", model="Yam", interface="can0", effector_model="E_Nope")


def test_rig_rejects_an_unknown_role_and_duplicate_arm_names() -> None:
    with pytest.raises(ConfigurationError, match="role"):
        RigArm(name="a", model="Yam", interface="can0", role="observer")
    twin = RigArm(name="a", model="Yam", interface="can0")
    with pytest.raises(ConfigurationError, match="two arms named"):
        Rig(name="dup", description="", arms=(twin, twin))
    with pytest.raises(ConfigurationError, match="no arms"):
        Rig(name="empty", description="", arms=())


def test_interface_overrides_move_one_arm_and_leave_the_other() -> None:
    rig = resolve_rig("yam_bimanual").with_interfaces({"right": "can5"})

    assert [arm.interface for arm in rig.arms] == ["can0", "can5"]
    # The original is untouched: rigs are frozen, overrides return a copy.
    assert [arm.interface for arm in resolve_rig("yam_bimanual").arms] == ["can0", "can1"]


def test_an_override_for_an_arm_that_is_not_in_the_rig_is_an_error() -> None:
    # Silently ignoring a typo would leave the arm on its default bus, which is
    # the one case where being quiet is worse than failing.
    with pytest.raises(ConfigurationError, match="no arm named"):
        resolve_rig("yam_bimanual").with_interfaces({"middle": "can9"})


def test_subset_keeps_rig_order_and_rejects_unknown_names() -> None:
    rig = resolve_rig("yam_bimanual")

    assert rig.subset(["right"]).names == ("right",)
    assert rig.subset(["right", "left"]).names == ("left", "right")
    assert rig.subset(["left", "left"]).names == ("left",)
    with pytest.raises(ConfigurationError, match="has no arm"):
        rig.subset(["middle"])


def test_rig_lookup_by_name_reports_what_it_holds() -> None:
    rig = resolve_rig("yam_bimanual")

    assert rig["left"].interface == "can0"
    with pytest.raises(ConfigurationError, match="left, right"):
        rig["middle"]


def test_roles_split_followers_from_leaders() -> None:
    rig = Rig(
        name="pair",
        description="one of each",
        arms=(
            RigArm(name="lead", model="Yam", interface="can0", role=ROLE_LEADER),
            RigArm(name="follow", model="Yam", interface="can1", role=ROLE_FOLLOWER),
        ),
    )

    assert rig.leaders[0].name == "lead"
    assert rig.followers[0].name == "follow"
    assert rig.followers[0].is_follower and not rig.leaders[0].is_follower


def test_the_registry_is_keyed_by_each_rig_s_own_name() -> None:
    for key, rig in PACKAGED_RIGS.items():
        assert key == rig.name


# --------------------------------------------------------------------------- #
# cameras
# --------------------------------------------------------------------------- #


def test_yam_bimanual_declares_the_cell_s_three_cameras() -> None:
    # The serials are a site fact: these are the D405s bolted to this cell, and
    # they are what pins each view to a role. A wrong one here records the
    # right-wrist view under the left-wrist key for every episode ever taken.
    rig = resolve_rig("yam_bimanual")

    assert rig.camera_names == ("top", "left_wrist", "right_wrist")
    assert {camera.name: camera.serial for camera in rig.cameras} == {
        "top": "348523020354",  # D435; the D405 it replaced was 254623070531
        "left_wrist": "254623070863",
        "right_wrist": "254623070417",
    }
    # The top camera watches the cell; each wrist camera rides on an arm, which
    # is what makes `--only` able to drop it.
    assert [camera.arm for camera in rig.cameras] == [None, "left", "right"]


def test_narrowing_a_rig_takes_the_other_arm_s_wrist_camera_with_it() -> None:
    # Recording a left-wrist view of an arm that was never powered would put a
    # frozen frame in every step of the dataset.
    rig = resolve_rig("yam_bimanual").subset(["right"])

    assert rig.names == ("right",)
    assert rig.camera_names == ("top", "right_wrist")


def test_a_state_only_run_can_drop_every_camera() -> None:
    rig = resolve_rig("yam_bimanual").without_cameras()

    assert rig.cameras == ()
    assert rig.names == ("left", "right")  # the arms are untouched


def test_moving_an_arm_to_another_bus_keeps_its_camera() -> None:
    # An interface override is about wiring, not about which views exist.
    rig = resolve_rig("yam_bimanual").with_interfaces({"left": "can2"})

    assert rig["left"].interface == "can2"
    assert rig.camera_names == ("top", "left_wrist", "right_wrist")


def test_a_rig_refuses_two_cameras_with_the_same_name() -> None:
    with pytest.raises(ConfigurationError, match="two cameras named"):
        Rig(
            name="dup",
            description="",
            arms=(RigArm(name="only", model="Yam", interface="can0"),),
            cameras=(
                RigCamera(name="top", serial="1", label="A"),
                RigCamera(name="top", serial="2", label="B"),
            ),
        )


def test_a_rig_refuses_two_cameras_on_one_serial() -> None:
    # Copy-pasting a serial would silently record one physical view twice under
    # two different keys, which looks like two cameras until you watch the mp4s.
    with pytest.raises(ConfigurationError, match="both"):
        Rig(
            name="dup",
            description="",
            arms=(RigArm(name="only", model="Yam", interface="can0"),),
            cameras=(
                RigCamera(name="top", serial="254623070531", label="A"),
                RigCamera(name="wrist", serial="254623070531", label="B"),
            ),
        )


def test_a_camera_cannot_ride_on_an_arm_the_rig_does_not_have() -> None:
    # Otherwise the camera simply never gets dropped by `subset`, and the typo
    # only shows up as a stale view in a recorded dataset.
    with pytest.raises(ConfigurationError, match="which it does not have"):
        Rig(
            name="typo",
            description="",
            arms=(RigArm(name="left", model="Yam", interface="can0"),),
            cameras=(
                RigCamera(name="wrist", serial="1", label="W", arm="lfet"),
            ),
        )


def test_camera_capture_settings_are_a_property_of_the_run() -> None:
    # Rate and pixel format belong to a run, not to the cell: a recorder wants
    # the fastest the cameras will go and RGB straight from the SDK, a browser
    # preview wants neither.
    rig = resolve_rig("yam_bimanual")
    assert {c.fps for c in rig.cameras} == {30}
    assert {c.pixel_format for c in rig.cameras} == {"bgr8"}

    fast = rig.with_camera_capture(fps=90, pixel_format="rgb8")

    assert {c.fps for c in fast.cameras} == {90}
    assert {c.pixel_format for c in fast.cameras} == {"rgb8"}
    # Serials, roles, and arm bindings are untouched.
    assert fast.camera_names == rig.camera_names
    assert [c.arm for c in fast.cameras] == [c.arm for c in rig.cameras]
    # And the rig itself is unchanged -- these are copies.
    assert {c.fps for c in rig.cameras} == {30}


def test_with_camera_capture_can_change_one_setting_alone() -> None:
    rig = resolve_rig("yam_bimanual").with_camera_capture(fps=60)

    assert {c.fps for c in rig.cameras} == {60}
    assert {c.pixel_format for c in rig.cameras} == {"bgr8"}
    assert rig.with_camera_capture() is not None  # no-op is allowed


def test_an_unsupported_pixel_format_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="pixel format"):
        resolve_rig("yam_bimanual").with_camera_capture(pixel_format="yuyv")
