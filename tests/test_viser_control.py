"""Browser control of a live rig: the safety gate, the rate limit, the trips.

Everything here runs against FakeArmBackend, so the arms are real objects with
a real capability handshake and no bus behind them.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from fake_arm_backend import FakeArmBackend

from openpi_control import cli
from openpi_control.exceptions import AlignmentError, CommandRejectedError, ConfigurationError
from openpi_control.rigs import resolve_rig

pytest.importorskip("viser", reason="needs the 'viz' extra")
pytest.importorskip("yourdfpy", reason="needs the 'viz' extra")

from openpi_control import viz  # noqa: E402
from openpi_control.viser_control import (  # noqa: E402
    ArmControlPanel,
    ControlLimits,
    RigControlPanel,
)

# Ports are per-test so a leaked server cannot wedge the next case.
_PORT = iter(range(8700, 8900))

DT = 0.05


@pytest.fixture(autouse=True)
def isolate_mesh_cache(monkeypatch):
    """Render a skeleton regardless of what the developer has fetched."""
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: None)


@pytest.fixture
def cell():
    """A powered-up yam_bimanual, its scene, and the fake backends behind it."""
    created = []

    def build(**backend_kwargs):
        rig = resolve_rig("yam_bimanual")
        backends: dict[str, FakeArmBackend] = {}

        def factory(rig_arm):
            backend = FakeArmBackend(dof=6, **backend_kwargs)
            backends[rig_arm.name] = backend
            return backend

        session, live_arms = cli.power_up(rig, backend_factory=factory)
        scene = viz.ArmSceneVisualizer.from_rig(rig, port=next(_PORT))
        created.append((session, scene))
        followers = {entry.name: entry.arm for entry in live_arms}
        return scene, followers, backends

    yield build
    for session, scene in created:
        scene.server.stop()
        session.close()


def sync_scene(scene, followers):
    """Do what live's mirror loop does: draw the newest measured pose."""
    for name, follower in followers.items():
        state = follower.latest_state
        if state is not None:
            scene.update(name, state.joints.position_rad)


def make_panel(scene, followers, name="left", **kwargs):
    sync_scene(scene, followers)
    return ArmControlPanel(followers[name], scene[name], log=lambda _: None, **kwargs)


def confirm_and_arm(panel):
    panel._confirm.value = True
    panel.arm()


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_panel_starts_disarmed_with_every_slider_locked(cell) -> None:
    scene, followers, _ = cell()
    panel = make_panel(scene, followers)

    assert not panel.armed
    assert all(slider.disabled for slider in panel._sliders)
    assert panel._effector_slider is not None and panel._effector_slider.disabled
    assert panel._arm_button.disabled


def test_confirmation_is_what_enables_the_arm_button(cell) -> None:
    scene, followers, _ = cell()
    panel = make_panel(scene, followers)

    assert panel._arm_button.disabled
    panel._confirm.value = True
    assert not panel._arm_button.disabled
    panel._confirm.value = False
    assert panel._arm_button.disabled


def test_disarmed_sliders_follow_the_arm(cell) -> None:
    """Someone pushes the arm by hand; the targets must not stay behind."""
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)

    backends["left"].position = 0.3
    backends["left"].effector_position = 0.6
    panel.step(DT)

    assert [pytest.approx(s.value, abs=1e-3) for s in panel._sliders] == [0.3] * 6
    assert panel._effector_slider.value == pytest.approx(0.6, abs=1e-3)
    assert backends["left"].commands == []


def test_refuses_to_arm_on_a_stale_state(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    backends["left"].state_age_s = 5.0

    with pytest.raises(AlignmentError, match="ms old"):
        confirm_and_arm(panel)
    assert not panel.armed


def test_refuses_to_arm_when_no_state_has_arrived(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    backends["left"].publishing = False

    with pytest.raises(AlignmentError, match="no state"):
        confirm_and_arm(panel)
    assert not panel.armed


def test_refuses_to_arm_when_the_render_disagrees_with_the_arm(cell) -> None:
    """The check i2rt leaves to a checkbox: is the screen actually the arm?"""
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    # The arm moved and nothing redrew it, so the render is a lie.
    backends["left"].position = 0.8

    with pytest.raises(AlignmentError, match="off the measured pose"):
        confirm_and_arm(panel)
    assert not panel.armed


def test_refuses_to_arm_a_node_that_takes_no_direct_commands(cell) -> None:
    scene, followers, _ = cell()
    panel = make_panel(scene, followers)
    panel._can_command = False

    with pytest.raises(AlignmentError, match="direct position commands"):
        confirm_and_arm(panel)


def test_refuses_a_urdf_that_disagrees_about_the_joint_count(cell) -> None:
    scene, followers, _ = cell()
    sync_scene(scene, followers)
    follower = followers["left"]
    follower._capabilities = dataclasses.replace(follower.capabilities, joint_names=("j1", "j2"))

    with pytest.raises(ConfigurationError, match="cannot map sliders to joints"):
        ArmControlPanel(follower, scene["left"], log=lambda _: None)


# --------------------------------------------------------------------------- #
# Arming and commanding
# --------------------------------------------------------------------------- #


def test_arming_re_engages_position_control_before_commanding(cell) -> None:
    """After a gravity float the node ignores commands until hold() lands."""
    scene, followers, backends = cell()
    panel = make_panel(scene, followers, float_mode=True)
    holds_before = backends["left"].holds

    confirm_and_arm(panel)

    assert panel.armed
    assert backends["left"].holds == holds_before + 1
    assert backends["left"].commands == []
    assert not any(slider.disabled for slider in panel._sliders)


def test_arming_seeds_the_target_at_the_measured_pose(cell) -> None:
    """The first command must be where the arm already is, not the rest pose."""
    scene, followers, backends = cell()
    backends["left"].position = 0.25
    panel = make_panel(scene, followers)

    confirm_and_arm(panel)
    panel.step(DT)

    sent = backends["left"].commands[-1]
    assert np.allclose(sent.position_rad, 0.25)


def test_a_full_throw_drag_is_rate_limited_not_teleported(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers, limits=ControlLimits(max_joint_rate_rad_s=0.5))
    confirm_and_arm(panel)

    for slider in panel._sliders:
        slider.value = slider.max
    panel.step(DT)

    sent = backends["left"].commands[-1]
    assert np.max(np.abs(np.asarray(sent.position_rad))) == pytest.approx(0.5 * DT, abs=1e-9)


def test_the_target_is_reached_once_enough_ticks_have_passed(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers, limits=ControlLimits(max_joint_rate_rad_s=1.0))
    confirm_and_arm(panel)

    for slider in panel._sliders:
        slider.value = 0.2
    for _ in range(50):
        panel.step(DT)

    assert np.allclose(backends["left"].commands[-1].position_rad, 0.2, atol=1e-6)


def test_commands_are_clamped_to_the_urdf_limits(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers, limits=ControlLimits(max_joint_rate_rad_s=1e6))
    confirm_and_arm(panel)

    upper = np.array([spec.upper for spec in scene["left"].joint_specs])
    for slider in panel._sliders:
        slider.value = slider.max
    panel.step(DT)

    assert np.all(np.asarray(backends["left"].commands[-1].position_rad) <= upper + 1e-9)


def test_the_gripper_is_commanded_as_a_normalized_effector(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers, limits=ControlLimits(max_effector_rate_s=1e6))
    confirm_and_arm(panel)

    panel._effector_slider.value = 1.0
    panel.step(DT)

    assert backends["left"].commands[-1].effector == pytest.approx(1.0)


def test_an_armed_panel_stops_mirroring_into_its_own_sliders(cell) -> None:
    """The operator owns the sliders while armed; nothing writes back."""
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    confirm_and_arm(panel)

    panel._sliders[0].value = 0.4
    backends["left"].position = -0.9
    panel.step(DT)

    assert panel._sliders[0].value == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Trips
# --------------------------------------------------------------------------- #


def test_a_state_that_goes_stale_disarms_the_panel(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    confirm_and_arm(panel)

    backends["left"].state_age_s = 5.0
    panel.step(DT)

    assert not panel.armed
    assert all(slider.disabled for slider in panel._sliders)


def test_a_state_that_stops_arriving_disarms_the_panel(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    confirm_and_arm(panel)

    backends["left"].publishing = False
    panel.step(DT)

    assert not panel.armed


def test_a_rejected_command_disarms_the_panel(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    confirm_and_arm(panel)

    def reject(command, *, live=False):
        raise CommandRejectedError("node said no")

    backends["left"].command = reject
    panel.step(DT)

    assert not panel.armed


def test_disarming_clears_the_confirmation(cell) -> None:
    """Re-arming after a trip means looking at the arm again, not one click."""
    scene, followers, _ = cell()
    panel = make_panel(scene, followers)
    confirm_and_arm(panel)

    panel.disarm("test")

    assert not panel._confirm.value
    assert panel._arm_button.disabled


def test_disarming_returns_a_floating_arm_to_float(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers, float_mode=True)
    confirm_and_arm(panel)
    floats_before = backends["left"].floats

    panel.disarm("test")

    assert backends["left"].floats == floats_before + 1


def test_disarming_a_holding_arm_holds(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)
    confirm_and_arm(panel)
    holds_before = backends["left"].holds

    panel.disarm("test")

    assert backends["left"].holds == holds_before + 1
    assert backends["left"].floats == 0


def test_disarm_is_idempotent(cell) -> None:
    scene, followers, backends = cell()
    panel = make_panel(scene, followers)

    panel.disarm("once")
    panel.disarm("twice")

    assert backends["left"].holds == 0


# --------------------------------------------------------------------------- #
# The whole rig
# --------------------------------------------------------------------------- #


def test_each_arm_of_the_rig_carries_its_own_gate(cell) -> None:
    scene, followers, backends = cell()
    sync_scene(scene, followers)
    control = RigControlPanel(scene, followers, log=lambda _: None)

    confirm_and_arm(control["left"])

    assert control.armed_names == ("left",)
    control.step(DT)
    assert backends["left"].commands
    assert backends["right"].commands == []


def test_disarm_all_puts_every_arm_down(cell) -> None:
    scene, followers, _ = cell()
    sync_scene(scene, followers)
    control = RigControlPanel(scene, followers, log=lambda _: None)
    confirm_and_arm(control["left"])
    confirm_and_arm(control["right"])
    assert set(control.armed_names) == {"left", "right"}

    control.disarm_all("stop")

    assert control.armed_names == ()


def test_an_unknown_arm_name_is_an_error_not_a_no_op(cell) -> None:
    scene, followers, _ = cell()
    sync_scene(scene, followers)
    control = RigControlPanel(scene, followers, log=lambda _: None)

    with pytest.raises(ConfigurationError, match="unknown arm"):
        control["middle"]


def test_control_needs_at_least_one_follower(cell) -> None:
    scene, followers, _ = cell()

    with pytest.raises(ConfigurationError, match="at least one follower"):
        RigControlPanel(scene, {})


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def test_control_without_the_browser_view_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="drop --no-viz"):
        cli.run_live(resolve_rig("yam_bimanual"), visualize=False, control=True)


def test_live_accepts_the_control_flag() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["live", "--control", "--rig", "yam_bimanual", "--help"])
    assert exit_info.value.code == 0


def test_run_live_builds_the_panel_but_arms_nothing_by_itself() -> None:
    """--control adds the gate; it never arms an arm on your behalf."""
    import threading

    backends: dict[str, FakeArmBackend] = {}

    def factory(rig_arm):
        backends[rig_arm.name] = FakeArmBackend(dof=6)
        return backends[rig_arm.name]

    stop = threading.Event()
    stop.set()
    status = cli.run_live(
        resolve_rig("yam_bimanual"),
        control=True,
        camera_preview=False,  # this is about the control panel, not the cell's cameras
        port=next(_PORT),
        backend_factory=factory,
        stop=stop,
    )

    assert status == 0
    assert backends
    assert all(not backend.commands for backend in backends.values())


def test_the_gripper_slider_is_labelled_the_way_the_node_reads_it(cell) -> None:
    # The slider value goes straight into PositionCommand.effector, where the
    # native effector treats normalized 1.0 as OPEN (its ready pose is
    # from_normalized(1.0)). A label with the ends reversed invites an operator
    # to close a gripper while believing they are opening it.
    scene, followers, _ = cell()
    panel = make_panel(scene, followers)

    assert panel._effector_slider is not None
    assert "0 closed" in panel._effector_slider.label
    assert "1 open" in panel._effector_slider.label

