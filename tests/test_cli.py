"""The operator CLI: doctor's preflight, and zero's servo writes.

Nothing here opens a real bus — ``buses.open_bus`` is replaced with a fake so
the zeroing path is exercised end to end without hardware.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import threading
from types import SimpleNamespace

import pytest
from fake_arm_backend import FakeArmBackend

from openpi_control import cli
from openpi_control.exceptions import ConfigurationError, NativeProcessError
from openpi_control.rigs import resolve_rig
from openpi_control.servos import buses


@pytest.fixture(autouse=True)
def quiet_run_logging(tmp_path, monkeypatch):
    """Keep run logs out of the developer's ~/openpi-data during tests."""
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path / "logs"))


@pytest.fixture
def no_mesh_cache(monkeypatch):
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: None)


class FakeCanBus:
    """Records the zero frames a driver sends, and acknowledges them."""

    def __init__(self, *, acknowledge: bool = True) -> None:
        self.sent: list[tuple[int, bytes]] = []
        self._acknowledge = acknowledge

    def send(self, message) -> None:  # noqa: ANN001 - can.Message
        self.sent.append((message.arbitration_id, bytes(message.data)))

    def recv(self, timeout=None):  # noqa: ANN001, ARG002
        return object() if self._acknowledge else None


@pytest.fixture
def fake_bus(monkeypatch):
    """Swap open_bus for a fake CAN bus; returns the bus for inspection."""

    def install(*, acknowledge: bool = True) -> FakeCanBus:
        bus = FakeCanBus(acknowledge=acknowledge)

        @contextlib.contextmanager
        def opener(port_type, interface, *, baudrate=None):  # noqa: ANN001, ARG001
            yield bus

        monkeypatch.setattr(buses, "open_bus", opener)
        # Zeroing sleeps 0.5s per servo in the real driver.
        monkeypatch.setattr("openpi_control.servos.dm_can.time.sleep", lambda _: None)
        return bus

    return install


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #


def test_yam_plan_covers_six_joints_and_the_gripper() -> None:
    plan = cli.build_plan("Yam", "E_Yam")
    assert [entry.joint_id for entry in plan] == [1, 2, 3, 4, 5, 6, 7]
    assert [entry.servo_id for entry in plan] == [1, 2, 3, 4, 5, 6, 7]
    assert {entry.servo_model for entry in plan} == {"DM J4340", "DM J4310"}
    assert all(not entry.read_only for entry in plan)
    assert cli.plan_port_type(plan) == buses.PORT_TYPE_CAN


def test_the_teaching_handle_encoder_is_read_only() -> None:
    """Its zero is fixed in hardware; the registry maps it to no driver."""
    plan = cli.build_plan("Yam", "E_Yam_Handle")
    handle = [entry for entry in plan if entry.source == "E_Yam_Handle"]
    assert len(handle) == 1
    assert handle[0].read_only
    assert handle[0].servo_model == "CAN Passive Encoder"
    # 0x50E, the handle's documented request id.
    assert handle[0].servo_id == 0x50E


def test_plan_is_ordered_by_joint_id() -> None:
    for model in ("Yam", "ARX_X5", "Trossen_wai_ctrl"):
        ids = [entry.joint_id for entry in cli.build_plan(model)]
        assert ids == sorted(ids), model


def test_an_unknown_servo_model_is_a_clear_error(tmp_path) -> None:
    catalog = tmp_path / "Bogus.json"
    catalog.write_text(
        json.dumps({"joints": [{"joint_id": 1, "servos": [{"servo_model": "Nope 9000"}]}]})
    )
    with pytest.raises(ConfigurationError, match="not in the servo registry"):
        cli.servo_entries(catalog, "arm")


def test_a_servo_without_a_model_is_rejected(tmp_path) -> None:
    catalog = tmp_path / "Bogus.json"
    catalog.write_text(json.dumps({"joints": [{"joint_id": 1, "servos": [{"servo_id": 3}]}]}))
    with pytest.raises(ConfigurationError, match="no servo_model"):
        cli.servo_entries(catalog, "arm")


def test_a_read_only_arm_has_no_bus_to_open() -> None:
    """ARX_ENC is a leader-only encoder arm: nothing to zero."""
    plan = cli.build_plan("ARX_ENC")
    assert all(entry.read_only for entry in plan)
    assert cli.plan_port_type(plan) is None


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def _by_name(results, name):
    return next(result for result in results if result.name == name)


def test_doctor_reports_a_missing_interface_as_a_failure(no_mesh_cache) -> None:
    results = cli.run_doctor("Yam", "definitely_not_a_can_iface", effector_model="E_Yam")
    interface = _by_name(results, "interface definitely_not_a_can_iface")
    assert interface.status == cli._FAIL
    assert "does not exist" in interface.detail
    # The static checks before it must still have passed.
    assert _by_name(results, "packaged assets").status == cli._OK
    assert _by_name(results, "servo registry").status == cli._OK
    assert _by_name(results, "bus type").detail == "can"


def test_doctor_names_the_command_that_lists_interfaces(no_mesh_cache) -> None:
    """The old message pointed at run/devices.sh, which this repo never had."""
    results = cli.run_doctor("Yam", "definitely_not_a_can_iface")
    detail = _by_name(results, "interface definitely_not_a_can_iface").detail
    assert "devices.sh" not in detail
    assert "ip -brief link show type can" in detail


def test_doctor_opens_no_bus_without_probe(monkeypatch, no_mesh_cache) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("doctor must not open a bus unless --probe is given")

    monkeypatch.setattr(buses, "open_bus", explode)
    cli.run_doctor("Yam", "definitely_not_a_can_iface", effector_model="E_Yam")


def test_doctor_flags_an_uncached_mesh_directory(no_mesh_cache) -> None:
    results = cli.run_doctor("Yam", "definitely_not_a_can_iface")
    meshes_check = _by_name(results, "visual meshes")
    assert meshes_check.status == cli._WARN
    assert "--fetch-meshes" in meshes_check.detail


def test_doctor_warns_when_a_model_ships_no_urdf(no_mesh_cache) -> None:
    results = cli.run_doctor("FR3", "192.168.1.10")
    assert _by_name(results, "urdf").status == cli._WARN


def test_doctor_reports_a_read_only_arm_has_nothing_to_zero(no_mesh_cache) -> None:
    results = cli.run_doctor("ARX_ENC", "can0")
    bus_type = _by_name(results, "bus type")
    assert bus_type.status == cli._WARN
    assert "nothing to zero" in bus_type.detail


def test_doctor_exit_code_is_nonzero_only_on_failure(capsys, no_mesh_cache) -> None:
    assert cli.main(["doctor", "--model", "Yam", "--interface", "definitely_not_a_can_iface"]) == 1
    assert "FAIL" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# zero
# --------------------------------------------------------------------------- #


def test_zero_sends_one_frame_per_servo(fake_bus) -> None:
    bus = fake_bus()
    plan = cli.build_plan("Yam", "E_Yam")
    outcomes = cli.zero_arm(plan, buses.PORT_TYPE_CAN, "can0")

    assert [entry.servo_id for entry, _ in outcomes] == [1, 2, 3, 4, 5, 6, 7]
    assert all(error is None for _, error in outcomes)
    # One zero frame per servo, addressed to the servo's own CAN id.
    assert [arbitration_id for arbitration_id, _ in bus.sent] == [1, 2, 3, 4, 5, 6, 7]
    assert {payload for _, payload in bus.sent} == {bytes([0xFF] * 7 + [0xFE])}


def test_zero_skips_read_only_servos_without_sending(fake_bus) -> None:
    bus = fake_bus()
    plan = cli.build_plan("Yam", "E_Yam_Handle")
    outcomes = cli.zero_arm(plan, buses.PORT_TYPE_CAN, "can0")

    assert [entry.servo_id for entry, _ in outcomes] == [1, 2, 3, 4, 5, 6]
    assert 0x50E not in [arbitration_id for arbitration_id, _ in bus.sent]


def test_zero_reports_an_unacknowledged_servo(fake_bus) -> None:
    fake_bus(acknowledge=False)
    plan = cli.build_plan("Yam")
    outcomes = cli.zero_arm(plan, buses.PORT_TYPE_CAN, "can0")
    assert all(error == "no acknowledgement" for _, error in outcomes)


def test_zero_on_a_read_only_arm_writes_nothing(fake_bus) -> None:
    bus = fake_bus()
    outcomes = cli.zero_arm(cli.build_plan("ARX_ENC"), buses.PORT_TYPE_CAN, "can0")
    assert outcomes == []
    assert bus.sent == []


def test_dry_run_touches_no_bus(monkeypatch, capsys) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not open a bus")

    monkeypatch.setattr(buses, "open_bus", explode)
    code = cli.main(
        ["zero", "--model", "Yam", "--interface", "can0", "--effector", "E_Yam", "--dry-run"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert out.count("zero") >= 7


def test_dry_run_marks_the_read_only_joint(capsys) -> None:
    cli.main(
        [
            "zero",
            "--model",
            "Yam",
            "--interface",
            "can0",
            "--effector",
            "E_Yam_Handle",
            "--dry-run",
        ]
    )
    assert "read-only, skipped" in capsys.readouterr().out


def test_joint_filter_selects_one_servo(capsys) -> None:
    cli.main(["zero", "--model", "Yam", "--interface", "can0", "--joint", "3", "--dry-run"])
    out = capsys.readouterr().out
    assert "joint 3" in out
    assert "joint 1" not in out


def test_an_unknown_joint_lists_the_real_ones(capsys) -> None:
    assert cli.main(["zero", "--model", "Yam", "--interface", "can0", "--joint", "99"]) == 2
    assert "joints are 1, 2, 3, 4, 5, 6" in capsys.readouterr().err


def test_zero_checks_the_interface_before_prompting(monkeypatch, capsys) -> None:
    """A missing bus must fail before the operator is asked to confirm."""

    def explode(_):
        raise AssertionError("must not prompt when the interface is absent")

    monkeypatch.setattr("builtins.input", explode)
    code = cli.main(
        [
            "zero",
            "--model",
            "Yam",
            "--interface",
            "definitely_not_a_can_iface",
            "--effector",
            "E_Yam",
        ]
    )
    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_declining_the_prompt_writes_nothing(monkeypatch, capsys, fake_bus) -> None:
    bus = fake_bus()
    monkeypatch.setattr(buses, "check_interface", lambda *a, **k: None)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    code = cli.main(["zero", "--model", "Yam", "--interface", "can0"])
    assert code == 1
    assert bus.sent == []
    assert "aborted" in capsys.readouterr().out


def test_only_the_exact_word_confirms(monkeypatch, fake_bus) -> None:
    bus = fake_bus()
    monkeypatch.setattr(buses, "check_interface", lambda *a, **k: None)
    for reply in ("y", "yes", "ZERO", ""):
        monkeypatch.setattr("builtins.input", lambda _, reply=reply: reply)
        assert cli.main(["zero", "--model", "Yam", "--interface", "can0"]) == 1
    assert bus.sent == [], "a near-miss reply must not zero the arm"

    monkeypatch.setattr("builtins.input", lambda _: "zero")
    assert cli.main(["zero", "--model", "Yam", "--interface", "can0"]) == 0
    assert len(bus.sent) == 6


def test_yes_skips_the_prompt(monkeypatch, fake_bus) -> None:
    bus = fake_bus()
    monkeypatch.setattr(buses, "check_interface", lambda *a, **k: None)

    def explode(_):
        raise AssertionError("--yes must not prompt")

    monkeypatch.setattr("builtins.input", explode)
    assert cli.main(["zero", "--model", "Yam", "--interface", "can0", "--yes"]) == 0
    assert len(bus.sent) == 6


def test_a_failed_servo_makes_the_command_fail(monkeypatch, fake_bus, capsys) -> None:
    fake_bus(acknowledge=False)
    monkeypatch.setattr(buses, "check_interface", lambda *a, **k: None)
    assert cli.main(["zero", "--model", "Yam", "--interface", "can0", "--yes"]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_every_run_leaves_a_log(capsys, no_mesh_cache) -> None:
    cli.main(["doctor", "--model", "Yam", "--interface", "definitely_not_a_can_iface"])
    out = capsys.readouterr().out
    assert "runtime/doctor.log" in out


# --------------------------------------------------------------------------- #
# live: powering a rig up and back down
# --------------------------------------------------------------------------- #


class _FakeScene:
    """Records what live pushed at the browser, and can stop the loop."""

    def __init__(self, stop_after: int | None = None, stop: threading.Event | None = None) -> None:
        self.updates: list[tuple[str, float]] = []
        self._stop_after = stop_after
        self._stop = stop

    def update(self, name, positions) -> None:
        self.updates.append((name, float(positions[0])))
        if self._stop is not None and self._stop_after is not None:
            if len(self.updates) >= self._stop_after:
                self._stop.set()


@pytest.fixture
def fakes():
    """A backend_factory plus the backends it handed out, keyed by arm name."""
    made: dict[str, FakeArmBackend] = {}

    def factory(**kwargs):
        def make(rig_arm):
            backend = FakeArmBackend(**kwargs)
            made[rig_arm.name] = backend
            return backend

        return make

    return factory, made


def test_power_up_energizes_every_arm_and_holds_by_default(fakes) -> None:
    # Connecting is the power-on, and a follower must come up stiff unless the
    # operator explicitly asked for a compliant arm.
    factory, made = fakes
    rig = resolve_rig("yam_bimanual")

    session, live_arms = cli.power_up(rig, backend_factory=factory())
    try:
        assert [entry.name for entry in live_arms] == ["left", "right"]
        assert set(made) == {"left", "right"}
        assert all(backend.connects == 1 for backend in made.values())
        assert all(backend.floats == 0 for backend in made.values())
    finally:
        session.close()


def test_float_mode_makes_the_followers_backdrivable(fakes) -> None:
    factory, made = fakes
    session, _ = cli.power_up(
        resolve_rig("yam_bimanual"), float_mode=True, backend_factory=factory()
    )
    try:
        assert [backend.floats for backend in made.values()] == [1, 1]
    finally:
        session.close()


def test_float_mode_leaves_an_arm_holding_when_the_node_cannot_float(fakes, capsys) -> None:
    # Silently pretending the arm is compliant would have someone tug on a
    # stiff arm wondering why it will not move.
    factory, _ = fakes
    session, _ = cli.power_up(
        resolve_rig("yam_bimanual"),
        float_mode=True,
        backend_factory=factory(supports_gravity_compensation=False),
    )
    try:
        out = capsys.readouterr().out
        assert "left: node has no gravity float; left holding" in out
    finally:
        session.close()


def test_a_failed_bring_up_leaves_nothing_energized(fakes) -> None:
    # The half-connected case is the dangerous one: the arm that did come up is
    # stiff, and this session is the only thing that can put it down again.
    made: dict[str, FakeArmBackend] = {}

    def factory(rig_arm):
        fails = rig_arm.name == "right"
        backend = FakeArmBackend(
            connect_error=NativeProcessError("no node") if fails else None
        )
        made[rig_arm.name] = backend
        return backend

    with pytest.raises(NativeProcessError):
        cli.power_up(resolve_rig("yam_bimanual"), backend_factory=factory)

    assert made["left"].connects == 1
    assert not made["left"].connected
    assert made["left"].closes  # closed on the way out, not orphaned


def test_power_down_parks_each_arm_at_home_before_de_energizing(fakes) -> None:
    # move_to_ready=True is the whole safeguard: it rides
    # MOVE_TO_READY_AND_SHUTDOWN, so the arm drives to home_pos and the node
    # exits at the end of that move instead of dropping it.
    factory, made = fakes
    session, live_arms = cli.power_up(resolve_rig("yam_bimanual"), backend_factory=factory())

    assert cli.power_down(session, live_arms, park=True) == 0

    for backend in made.values():
        assert backend.closes[0] is True
        assert not backend.connected


def test_no_park_de_energizes_in_place(fakes) -> None:
    factory, made = fakes
    session, live_arms = cli.power_up(resolve_rig("yam_bimanual"), backend_factory=factory())

    assert cli.power_down(session, live_arms, park=False) == 0

    for backend in made.values():
        assert True not in backend.closes


def test_an_arm_whose_node_has_no_ready_move_is_closed_in_place_out_loud(
    fakes, capsys
) -> None:
    factory, made = fakes
    session, live_arms = cli.power_up(
        resolve_rig("yam_bimanual"), backend_factory=factory(supports_move_to_ready=False)
    )

    assert cli.power_down(session, live_arms, park=True) == 0

    out = capsys.readouterr().out
    assert "node has no ready move; closing in place" in out
    for backend in made.values():
        assert True not in backend.closes
        assert not backend.connected


def test_a_ctrl_c_during_the_park_still_de_energizes_every_arm(fakes, capsys) -> None:
    # An impatient second ctrl-c must not unwind past session.close(): a node
    # left running holds an energized arm until the parent liveness pipe kills
    # it, which drops the arm exactly where the aborted park left it.
    factory, made = fakes
    session, live_arms = cli.power_up(resolve_rig("yam_bimanual"), backend_factory=factory())
    interrupted = made["left"]
    park_in_place = interrupted.close

    def interrupt_the_park(*, move_to_ready: bool = False) -> None:
        if move_to_ready:
            raise KeyboardInterrupt
        park_in_place(move_to_ready=move_to_ready)

    interrupted.close = interrupt_the_park

    assert cli.power_down(session, live_arms, park=True) > 0

    assert "park interrupted" in capsys.readouterr().out
    for backend in made.values():
        assert not backend.connected


def test_a_failed_park_still_de_energizes_every_arm(fakes) -> None:
    # A park that throws must not abandon an energized arm; the failure is
    # reported, and the session is closed regardless.
    factory, made = fakes
    session, live_arms = cli.power_up(
        resolve_rig("yam_bimanual"),
        backend_factory=factory(close_error=NativeProcessError("node gone")),
    )

    assert cli.power_down(session, live_arms, park=True) > 0

    for backend in made.values():
        assert not backend.connected


def test_mirror_pushes_each_arm_s_newest_pose_at_the_scene(fakes) -> None:
    factory, made = fakes
    session, live_arms = cli.power_up(resolve_rig("yam_bimanual"), backend_factory=factory())
    try:
        made["left"].position = 0.4
        made["right"].position = -0.4
        stop = threading.Event()
        scene = _FakeScene(stop_after=2, stop=stop)

        cli.mirror(scene, live_arms, stop=stop, rate_hz=1000.0)

        assert ("left", 0.4) in scene.updates
        assert ("right", -0.4) in scene.updates
    finally:
        session.close()


def test_mirror_leaves_the_last_pose_when_an_arm_goes_quiet(fakes) -> None:
    # latest_state() returning None must not raise: tearing down a session that
    # is holding two energized arms is far worse than a stale render.
    factory, made = fakes
    session, live_arms = cli.power_up(resolve_rig("yam_bimanual"), backend_factory=factory())
    try:
        for backend in made.values():
            backend.connected = False  # publishes nothing, stays energized
        stop = threading.Event()
        stop.set()

        cli.mirror(_FakeScene(), live_arms, stop=stop, rate_hz=1000.0)
    finally:
        session.close()


def test_run_live_powers_up_mirrors_and_parks_in_one_call(fakes) -> None:
    factory, made = fakes
    stop = threading.Event()
    stop.set()  # exit the mirror loop on the first check

    assert (
        cli.run_live(
            resolve_rig("yam_bimanual"),
            visualize=False,
            backend_factory=factory(),
            stop=stop,
        )
        == 0
    )

    for backend in made.values():
        assert backend.connects == 1
        assert backend.closes[0] is True  # parked on the way down
        assert not backend.connected


def test_run_live_reports_a_failed_power_down(fakes) -> None:
    factory, _ = fakes
    stop = threading.Event()
    stop.set()

    assert (
        cli.run_live(
            resolve_rig("yam_bimanual"),
            visualize=False,
            backend_factory=factory(close_error=NativeProcessError("node gone")),
            stop=stop,
        )
        == 1
    )


def test_interface_overrides_parse_as_arm_equals_interface() -> None:
    assert cli._parse_interface_overrides(["left=can2", "right=can3"]) == {
        "left": "can2",
        "right": "can3",
    }
    assert cli._parse_interface_overrides(None) == {}
    for bad in (["left"], ["=can2"], ["left="]):
        with pytest.raises(ConfigurationError, match="ARM=IFACE"):
            cli._parse_interface_overrides(bad)


def test_live_list_describes_the_rig_without_energizing_anything(capsys) -> None:
    assert cli.main(["live", "--list"]) == 0
    out = capsys.readouterr().out
    assert "yam_bimanual" in out
    assert "left" in out and "can0" in out
    assert "right" in out and "can1" in out


def test_live_refuses_to_energize_when_preflight_fails(capsys) -> None:
    # Nothing comes up unless every arm passes: a rig half up is a stiff arm
    # with no session left to put it down.
    assert cli.main(["live", "--interface", "left=definitely-not-an-interface"]) == 1
    err = capsys.readouterr().err
    assert "nothing was energized" in err


def test_preflight_rig_checks_every_arm() -> None:
    failures, reports = cli.preflight_rig(resolve_rig("yam_bimanual"))

    assert [name for name, _ in reports] == ["left", "right"]
    assert all(results for _, results in reports)
    assert failures >= 0


def test_doctor_rig_checks_both_arms_of_the_rig(capsys) -> None:
    cli.main(["doctor", "--rig", "yam_bimanual"])
    out = capsys.readouterr().out
    assert "left (Yam on can0)" in out
    assert "right (Yam on can1)" in out
    assert "2 arms" in out


def test_doctor_rig_takes_interface_overrides(capsys) -> None:
    cli.main(["doctor", "--rig", "yam_bimanual", "--interface-override", "right=can7"])
    assert "right (Yam on can7)" in capsys.readouterr().out


def test_doctor_rejects_rig_mixed_with_a_single_arm(capsys) -> None:
    assert cli.main(["doctor", "--rig", "yam_bimanual", "--model", "Yam"]) == 2
    assert "drop --model/--interface" in capsys.readouterr().err


def test_doctor_needs_either_a_rig_or_a_model_and_interface(capsys) -> None:
    assert cli.main(["doctor"]) == 2
    assert "either --rig, or both --model and --interface" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# cameras
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_camera_bus(monkeypatch, tmp_path):
    """Point camera discovery at a fake /dev/v4l/by-id holding given serials.

    Nothing here opens a camera: the CLI's camera checks are the udev-reading
    half of the module, which is the half that has to work on a box with no
    RealSense SDK installed.
    """
    from test_cameras import fake_by_id

    from openpi_control import cameras as cameras_mod

    def _install(serials):
        by_id = fake_by_id(tmp_path, serials)
        monkeypatch.setattr(cameras_mod, "BY_ID_DIR", by_id)
        return by_id

    return _install


def test_camera_checks_pass_when_every_declared_camera_is_present(
    fake_camera_bus,
) -> None:
    fake_camera_bus(["348523020354", "254623070863", "254623070417"])

    results = cli.run_camera_checks(resolve_rig("yam_bimanual"))

    assert [r.status for r in results] == [cli._OK] * 3
    # The resolution is in the detail line, so an operator can see at a glance
    # that a camera came up in the mode the rig asked for -- and the modes are
    # not uniform: the top D435 has no 848x480, so it runs 640x480 while the
    # D405 wrists keep their native mode.
    assert "640x480@30" in results[0].detail
    assert "848x480@30" in results[1].detail


def test_a_missing_camera_only_warns_for_doctor(fake_camera_bus) -> None:
    # Cameras are not needed to drive an arm, so an unplugged wrist camera must
    # not stop `doctor` from green-lighting the cell.
    fake_camera_bus(["348523020354"])

    results = cli.run_camera_checks(resolve_rig("yam_bimanual"))

    statuses = {r.name: r.status for r in results}
    assert statuses["camera top"] == cli._OK
    assert statuses["camera left_wrist"] == cli._WARN
    assert not any(r.status == cli._FAIL for r in results)


def test_a_missing_camera_is_fatal_when_the_caller_needs_it(fake_camera_bus) -> None:
    # The recorder passes required=True: writing an episode with a view
    # silently absent is worse than refusing to start.
    fake_camera_bus(["348523020354"])

    results = cli.run_camera_checks(resolve_rig("yam_bimanual"), required=True)

    assert sum(1 for r in results if r.status == cli._FAIL) == 2
    assert any("254623070863" in r.detail for r in results)


def test_camera_checks_flag_a_camera_the_rig_does_not_know(fake_camera_bus) -> None:
    # The useful half of "the top view is missing" is usually "and here is the
    # serial of the camera that replaced it".
    fake_camera_bus(["348523020354", "254623070863", "254623070417", "999999999999"])

    results = cli.run_camera_checks(resolve_rig("yam_bimanual"))

    unclaimed = [r for r in results if r.name == "unclaimed cameras"]
    assert len(unclaimed) == 1
    assert "999999999999" in unclaimed[0].detail


def test_narrowing_to_one_arm_stops_checking_the_other_wrist(fake_camera_bus) -> None:
    fake_camera_bus(["348523020354", "254623070417"])

    results = cli.run_camera_checks(resolve_rig("yam_bimanual").subset(["right"]))

    assert [r.name for r in results] == ["camera top", "camera right_wrist"]
    assert all(r.status == cli._OK for r in results)


def test_a_rig_with_no_cameras_says_so_instead_of_passing_vacuously(
    fake_camera_bus,
) -> None:
    fake_camera_bus([])

    results = cli.run_camera_checks(resolve_rig("yam_bimanual").without_cameras())

    assert len(results) == 1
    assert results[0].status == cli._OK
    assert "none" in results[0].detail


def test_the_cameras_command_exits_nonzero_only_on_a_failure(
    fake_camera_bus, capsys
) -> None:
    fake_camera_bus(["348523020354"])

    # Two cameras missing, but missing is a warning here, so this still passes.
    assert cli.main(["cameras"]) == 0
    assert "camera left_wrist" in capsys.readouterr().out


def test_pinning_a_camera_to_a_device_that_is_not_there_is_reported(
    fake_camera_bus, tmp_path, capsys
) -> None:
    fake_camera_bus(["348523020354", "254623070863", "254623070417"])

    cli.main(["cameras", "--camera", f"top={tmp_path / 'nope'}"])

    out = capsys.readouterr().out
    assert "pinned device" in out
    assert str(tmp_path / "nope") in out


def test_a_snapshot_without_a_probe_is_refused_before_anything_opens(
    fake_camera_bus, tmp_path
) -> None:
    # --snapshot alone would silently do nothing; saying so beats an empty dir.
    fake_camera_bus(["348523020354"])

    with pytest.raises(ConfigurationError, match="needs --probe"):
        cli._command_cameras(
            argparse.Namespace(
                rig="yam_bimanual",
                only=None,
                camera=None,
                probe=False,
                snapshot=tmp_path,
            ),
            tmp_path / "log",
        )


def test_doctor_on_a_rig_reports_its_cameras(fake_camera_bus, no_mesh_cache, capsys) -> None:
    # Cameras belong to the rig, not to an arm, so they are checked once rather
    # than repeated under every arm.
    fake_camera_bus(["348523020354", "254623070863", "254623070417"])

    cli.main(["doctor", "--rig", "yam_bimanual"])

    out = capsys.readouterr().out
    assert "cameras (3 declared)" in out
    assert out.count("camera top") == 1
    assert "2 arms, 3 cameras" in out


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #


def test_record_without_a_repo_id_says_what_to_do(capsys) -> None:
    # Defaulting the dataset id would scatter half-finished datasets under
    # whatever name happened to be generated.
    assert cli.main(["record", "--task", "fold the towel"]) == 2

    assert "--repo-id" in capsys.readouterr().err


def test_record_dry_run_needs_no_repo_id(fake_camera_bus, monkeypatch) -> None:
    # --dry-run rehearses a session with the arms live and writes nothing, so
    # there is no dataset to name.
    fake_camera_bus([])
    seen: dict[str, object] = {}

    def fake_run(rig, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_record", fake_run)

    assert cli.main(["record", "--dry-run", "--no-cameras", "--skip-preflight"]) == 0
    assert seen["dry_run"] is True
    assert seen["repo_id"] is None


def test_record_warns_when_the_task_is_unset(fake_camera_bus, monkeypatch, capsys) -> None:
    # The task string lands on every frame, and relabelling means rewriting the
    # dataset, so an unset --task deserves a word rather than a silent default.
    fake_camera_bus([])
    monkeypatch.setattr(cli, "run_record", lambda rig, **kwargs: 0)

    cli.main(["record", "--dry-run", "--no-cameras", "--skip-preflight"])

    assert "no --task given" in capsys.readouterr().err


def test_record_refuses_a_nonsense_frame_rate(capsys) -> None:
    assert cli.main(["record", "--dry-run", "--fps", "0"]) == 2

    assert "--fps must be positive" in capsys.readouterr().err


def test_record_preflight_treats_a_missing_camera_as_fatal(
    fake_camera_bus, no_mesh_cache, capsys
) -> None:
    # Unlike `doctor`, recording with a view silently absent produces a dataset
    # that is wrong rather than a cell that is merely unchecked.
    fake_camera_bus(["348523020354"])  # top only; both wrists missing

    status = cli.main(["record", "--dry-run", "--task", "t"])

    assert status == 1
    err = capsys.readouterr().err
    assert "failed check(s); nothing was energized" in err


def test_record_narrows_cameras_with_only(fake_camera_bus, monkeypatch, capsys) -> None:
    fake_camera_bus(["348523020354", "254623070417"])
    seen: dict[str, object] = {}

    def fake_run(rig, **kwargs):
        seen["cameras"] = rig.camera_names
        seen["arms"] = rig.names
        return 0

    monkeypatch.setattr(cli, "run_record", fake_run)
    cli.main(["record", "--dry-run", "--task", "t", "--only", "right"])

    assert seen["arms"] == ("right",)
    assert seen["cameras"] == ("top", "right_wrist")
    del capsys


def test_no_cameras_records_state_only(fake_camera_bus, monkeypatch) -> None:
    fake_camera_bus([])
    seen: dict[str, object] = {}

    def fake_run(rig, **kwargs):
        seen["cameras"] = rig.camera_names
        return 0

    monkeypatch.setattr(cli, "run_record", fake_run)
    cli.main(["record", "--dry-run", "--task", "t", "--no-cameras", "--skip-preflight"])

    assert seen["cameras"] == ()


def test_record_runs_a_whole_session_on_fake_backends(fakes, capsys) -> None:
    # The end-to-end path: power up, drive the arms, write episodes, park. The
    # `hold` source exists exactly so this is possible without a headset.
    factory, made = fakes
    rig = resolve_rig("yam_bimanual").without_cameras()

    status = cli.run_record(
        rig,
        task="a whole session",
        repo_id=None,
        teleop="hold",
        fps=200,
        hold_duration_s=0.05,
        dry_run=True,
        park=True,
        backend_factory=factory(),
    )

    assert status == 0
    out = capsys.readouterr().out
    assert "1 episode(s)" in out
    # Both arms were energized, commanded, and then parked at home_pos.
    for backend in made.values():
        assert backend.connects == 1
        assert backend.commands
        assert backend.closes[0] is True  # parked at home_pos on the way down


def test_a_session_that_saved_nothing_is_a_failure(fakes, capsys) -> None:
    # Exit status is what a wrapper script keys on: a session that produced no
    # episode must not look like a success.
    factory, made = fakes
    stop = threading.Event()
    stop.set()  # end before a single tick runs

    status = cli.run_record(
        resolve_rig("yam_bimanual").without_cameras(),
        task="nothing",
        repo_id=None,
        teleop="hold",
        dry_run=True,
        backend_factory=factory(),
        stop=stop,
    )

    assert status == 1
    assert "no episode was saved" in capsys.readouterr().err
    # And the arms still came down.
    for backend in made.values():
        assert not backend.connected


def test_an_unknown_teleop_source_is_refused(fakes) -> None:
    factory, _ = fakes

    with pytest.raises(ConfigurationError, match="unknown teleop source"):
        cli.run_record(
            resolve_rig("yam_bimanual").without_cameras(),
            task="t",
            repo_id=None,
            teleop="quest3",
            dry_run=True,
            backend_factory=factory(),
        )


def test_the_arms_come_down_even_when_the_session_raises(fakes) -> None:
    # A recording session holds two energized arms; an exception on the way
    # through must not leave them up.
    factory, made = fakes

    with pytest.raises(ConfigurationError):
        cli.run_record(
            resolve_rig("yam_bimanual").without_cameras(),
            task="t",
            repo_id=None,
            teleop="nope",
            dry_run=True,
            backend_factory=factory(),
        )

    assert made
    for backend in made.values():
        assert not backend.connected


# --------------------------------------------------------------------------- #
# live camera preview
# --------------------------------------------------------------------------- #


def test_preview_names_the_cameras_it_could_not_open(tmp_path, monkeypatch, capsys) -> None:
    """A missing camera is a line of output, not a refusal to energize.

    ``record`` opens all-or-none because a dataset with a view silently absent
    is corrupt. ``live`` exists to drive arms, so an unplugged wrist camera
    costs you a tile, not the session.
    """
    from tests_helpers_cameras import fake_by_id  # noqa: PLC0415

    from openpi_control import cameras as cameras_mod

    rig = resolve_rig("yam_bimanual")
    # Only the top camera is on this bus; both wrists are unplugged.
    monkeypatch.setattr(
        cameras_mod, "BY_ID_DIR", fake_by_id(tmp_path, [rig.cameras[0].serial])
    )
    opened = []

    class _Reader:
        def __init__(self, spec):
            opened.append(spec.name)
            self.spec = spec

    monkeypatch.setattr(cameras_mod, "CameraReader", _Reader)

    readers = cli.open_preview_cameras(rig)

    assert opened == [rig.cameras[0].name]
    assert set(readers) == {rig.cameras[0].name}
    out = capsys.readouterr().out
    for camera in rig.cameras[1:]:
        assert f"{camera.name}" in out
        assert "not on the bus" in out


def test_preview_collapses_one_failure_shared_by_every_camera(
    tmp_path, monkeypatch, capsys
) -> None:
    """No SDK installed is the same sentence three times; say it once."""
    from tests_helpers_cameras import fake_by_id  # noqa: PLC0415

    from openpi_control import cameras as cameras_mod

    rig = resolve_rig("yam_bimanual")
    monkeypatch.setattr(
        cameras_mod, "BY_ID_DIR", fake_by_id(tmp_path, [c.serial for c in rig.cameras])
    )

    def refuse(spec):
        raise ConfigurationError("reading a camera needs the RealSense SDK")

    monkeypatch.setattr(cameras_mod, "CameraReader", refuse)

    assert cli.open_preview_cameras(rig) == {}

    out = capsys.readouterr().out
    assert out.count("needs the RealSense SDK") == 1
    for camera in rig.cameras:
        assert camera.name in out


def test_preview_keeps_the_cameras_that_do_open(tmp_path, monkeypatch, capsys) -> None:
    """One busy camera must not cost the other two their tiles."""
    from tests_helpers_cameras import fake_by_id  # noqa: PLC0415

    from openpi_control import cameras as cameras_mod

    rig = resolve_rig("yam_bimanual")
    monkeypatch.setattr(
        cameras_mod, "BY_ID_DIR", fake_by_id(tmp_path, [c.serial for c in rig.cameras])
    )
    busy = rig.cameras[1].name

    class _Reader:
        def __init__(self, spec):
            if spec.name == busy:
                raise ConfigurationError(f"cannot start camera {spec.name!r}: device busy")
            self.spec = spec

    monkeypatch.setattr(cameras_mod, "CameraReader", _Reader)

    readers = cli.open_preview_cameras(rig)

    assert set(readers) == {c.name for c in rig.cameras} - {busy}
    assert "device busy" in capsys.readouterr().out


def test_mirror_steps_a_camera_panel_on_the_pose_clock(fakes) -> None:
    """One clock for poses, commands, and previews -- never three threads."""
    factory, _ = fakes
    rig = resolve_rig("yam_bimanual")
    session, live_arms = cli.power_up(rig, backend_factory=factory())
    steps: list[float] = []

    class _Panel:
        def step(self, dt: float) -> None:
            steps.append(dt)
            stop.set()

    stop = threading.Event()
    try:
        cli.mirror(None, live_arms, stop=stop, rate_hz=100.0, cameras=_Panel())
    finally:
        session.close()

    assert steps == [pytest.approx(0.01)]


# --------------------------------------------------------------------------- #
# CAN bitrate
# --------------------------------------------------------------------------- #


def test_bitrate_comes_from_sysfs_when_the_kernel_exposes_it(monkeypatch) -> None:
    # The cheap path: no subprocess at all.
    monkeypatch.setattr(cli, "_can_sysfs", lambda iface, leaf: "1000000")

    def forbidden(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("sysfs answered; ip should not have been run")

    monkeypatch.setattr(cli.subprocess, "run", forbidden)

    assert cli.can_bitrate("can0") == 1000000


def test_bitrate_falls_back_to_ip_when_sysfs_has_no_bittiming(monkeypatch) -> None:
    # /sys/class/net/<if>/can_bittiming does not exist on every kernel -- it is
    # absent on 6.8 -- which made this check warn on every run of a correctly
    # configured 1 Mbit bus. A preflight that always warns is one nobody reads.
    monkeypatch.setattr(cli, "_can_sysfs", lambda iface, leaf: None)
    payload = json.dumps(
        [{"linkinfo": {"info_kind": "can", "info_data": {"bittiming": {"bitrate": 1000000}}}}]
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=payload, stderr=""),
    )

    assert cli.can_bitrate("can0") == 1000000


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=1, stdout="", stderr="Device does not exist"),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="not json", stderr=""),
        SimpleNamespace(returncode=0, stdout="[{}]", stderr=""),
        SimpleNamespace(returncode=0, stdout='[{"linkinfo": {}}]', stderr=""),
    ],
)
def test_an_unreadable_bitrate_is_unknown_not_a_crash(monkeypatch, result) -> None:
    # doctor must survive a non-CAN interface, an absent one, and an `ip` whose
    # output shape changed -- reporting "unknown" rather than raising.
    monkeypatch.setattr(cli, "_can_sysfs", lambda iface, leaf: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: result)

    assert cli.can_bitrate("can0") is None


def test_a_missing_ip_binary_is_survivable(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_can_sysfs", lambda iface, leaf: None)

    def boom(*args, **kwargs):
        raise FileNotFoundError("no ip")

    monkeypatch.setattr(cli.subprocess, "run", boom)

    assert cli.can_bitrate("can0") is None
