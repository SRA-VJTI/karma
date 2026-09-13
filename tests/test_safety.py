"""Shared runtime safeguards, exercised without physical hardware."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from openpi_control import cli, safety, teleop_runtime
from openpi_control.exceptions import ConfigurationError
from openpi_control.inference import InferenceError
from openpi_control.record import ArmTarget, TeleopStep
from openpi_control.rigs import resolve_rig


def state(age=0.0, positions=None):
    return SimpleNamespace(
        age_s=age,
        is_fresh=lambda max_age: age <= max_age,
        joints=SimpleNamespace(position_rad=np.zeros(6) if positions is None else positions),
        effector=SimpleNamespace(position=1.0),
    )


def test_state_guard_pauses_recovers_and_stops_missing_state(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(safety.time, "monotonic", lambda: now[0])
    guard = safety.StateGuard(["left"])
    assert guard.check({"left": state(0.26)}) is None
    assert guard.check({"left": state()}) is not None
    assert guard.check({}) is None
    now[0] = 0.9
    assert guard.check({}) is None
    now[0] = 1.0
    with pytest.raises(InferenceError, match="left state is missing"):
        guard.check({})


def test_stale_received_state_stops_immediately():
    with pytest.raises(InferenceError, match="silent past the 1s limit"):
        safety.StateGuard(["left"]).check({"left": state(1.1)})


def test_nonfinite_state_is_rejected():
    with pytest.raises(InferenceError, match="non-finite"):
        safety.StateGuard(["left"]).check({"left": state(positions=[float("nan")] * 6)})


def test_headless_limits_follow_joint_chain_not_xml_order():
    limits = safety.joint_limits(resolve_rig("yam_bimanual"))
    for lower, upper in limits.values():
        np.testing.assert_allclose(lower, [-2.61799, 0, 0, -1.5708, -1.5708, -2.0944])
        np.testing.assert_allclose(upper, [3.13, 3.65, 3.13, 1.5708, 1.5708, 2.0944])


@pytest.mark.parametrize(
    "command,extra",
    [
        ("infer", ["--instruction", "test"]),
        ("rollout", ["--repo-id", "local/test"]),
        ("hitl", ["--repo-id", "local/test"]),
        ("teleop", []),
    ],
)
def test_all_commands_stop_at_failed_arm_preflight(monkeypatch, tmp_path, command, extra):
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "preflight_rig", lambda rig: (1, []))
    monkeypatch.setattr(cli, "run_camera_checks", lambda *a, **k: [])
    monkeypatch.setattr(cli, "check_camera_modes", lambda *a, **k: [])

    def forbidden(*args, **kwargs):
        pytest.fail("runtime started despite failed preflight")

    monkeypatch.setattr(cli, "run_infer", forbidden)
    monkeypatch.setattr(cli, "run_rollout", forbidden)
    monkeypatch.setattr(cli, "run_hitl", forbidden)
    monkeypatch.setattr(teleop_runtime, "run_teleop", forbidden)
    assert cli.main([command, *extra]) == 1


@pytest.mark.parametrize(
    "command,extra",
    [
        ("infer", ["--instruction", "test"]),
        ("rollout", ["--repo-id", "local/test"]),
        ("hitl", ["--repo-id", "local/test"]),
    ],
)
def test_policy_commands_require_camera_preflight(monkeypatch, tmp_path, command, extra):
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "preflight_rig", lambda rig: (0, []))
    observed = []

    def camera_check(rig, **kwargs):
        observed.append(kwargs["required"])
        return [cli.CheckResult(cli._FAIL, "camera", "missing")]

    monkeypatch.setattr(cli, "run_camera_checks", camera_check)
    monkeypatch.setattr(cli, "check_camera_modes", lambda rig: [])
    assert cli.main([command, *extra]) == 1
    assert observed == [True]


@pytest.mark.parametrize("fault", ["close", "stale", "invalid", "none"])
def test_teleop_guards_and_parks_even_if_source_close_fails(monkeypatch, fault):
    stop = threading.Event()
    sent = []
    parked = []
    arm = SimpleNamespace(latest_state=state(), command=sent.append)
    rig = resolve_rig("yam_bimanual").subset(["left"])
    live = SimpleNamespace(name="left", rig_arm=rig.arms[0], arm=arm)
    monkeypatch.setattr(cli, "power_up", lambda *a, **k: (object(), [live]))
    monkeypatch.setattr(cli, "power_down", lambda *a, **k: parked.append(True) or 0)

    class Source:
        def __init__(self, *a, **k):
            if fault == "stale":
                arm.latest_state = state(1.1)

        def describe(self):
            return "test"

        def poll(self, states):
            stop.set()
            value = float("nan") if fault == "invalid" else 100.0
            return TeleopStep(targets={"left": ArmTarget((value,) * 6, 0.5)})

        def close(self):
            if fault == "close":
                raise RuntimeError("close failed")

    monkeypatch.setattr(teleop_runtime, "QuestTeleopSource", Source)
    if fault == "none":
        assert teleop_runtime.run_teleop(rig, stop=stop) == 0
        np.testing.assert_allclose(sent[0].position_rad, safety.joint_limits(rig)["left"][1])
    else:
        error = {"close": RuntimeError, "stale": InferenceError, "invalid": ConfigurationError}[
            fault
        ]
        with pytest.raises(error):
            teleop_runtime.run_teleop(rig, stop=stop)
    assert parked == [True]
    if fault in {"stale", "invalid"}:
        assert not sent


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("command", ["infer", "rollout", "hitl", "teleop"])
def test_invalid_motion_setting_rejected_before_power_up(monkeypatch, command, value):
    def forbidden(*a, **k):
        pytest.fail("hardware was opened for invalid motion settings")

    monkeypatch.setattr(cli, "power_up", forbidden)
    rig = resolve_rig("yam_bimanual")
    with pytest.raises(ConfigurationError, match="finite and positive"):
        if command == "teleop":
            teleop_runtime.run_teleop(rig, rate_hz=value)
        elif command == "infer":
            cli.run_infer(rig, instruction="test", max_step_rad=value)
        else:
            getattr(cli, "run_" + command)(rig, repo_id="local/test", max_step_rad=value)


@pytest.mark.parametrize("fault", ["client", "cameras"])
def test_inference_parks_before_ancillary_cleanup_failure(monkeypatch, fault):
    from fake_arm_backend import FakeArmBackend

    stop = threading.Event()
    stop.set()
    backends = []

    def factory(rig_arm):
        backend = FakeArmBackend()
        backends.append(backend)
        return backend

    def close_client():
        if fault == "client":
            raise RuntimeError("client cleanup")

    def close_cameras(readers):
        if fault == "cameras":
            raise RuntimeError("camera cleanup")

    monkeypatch.setattr(cli, "open_inference_cameras", lambda *a, **k: {})
    monkeypatch.setattr(cli.cameras_mod, "close_readers", close_cameras)
    policy = SimpleNamespace(health=lambda: None, close=close_client, url="test")
    with pytest.raises(RuntimeError, match="cleanup"):
        cli.run_infer(
            resolve_rig("yam_bimanual"),
            instruction="test",
            visualize=False,
            prefetch=False,
            stop=stop,
            backend_factory=factory,
            policy=policy,
        )
    assert len(backends) == 2
    assert all(not backend.connected and backend.closes[0] for backend in backends)


def test_hitl_stale_guard_runs_even_without_policy_polling(monkeypatch, tmp_path):
    from openpi_control import dagger, record

    parked = []
    monkeypatch.setattr(cli, "power_up", lambda *a, **k: (object(), []))
    monkeypatch.setattr(cli, "settle_arm_states", lambda *a: {})
    monkeypatch.setattr(cli, "power_down", lambda *a, **k: parked.append(True) or 0)
    monkeypatch.setattr(cli, "InferenceRolloutSource", lambda **k: object())
    monkeypatch.setattr(
        dagger,
        "DaggerSource",
        lambda **k: SimpleNamespace(
            stats=dagger.InterventionStats(),
            describe=lambda: "human control",
            close=lambda: None,
            commanded_effector={},
        ),
    )

    def capture(**kwargs):
        # Exercise the recorder clock callback directly: no policy act() call
        # can supply the safeguard during human intervention.
        kwargs["on_tick"]({"left": state(1.1), "right": state()})
        pytest.fail("HITL accepted stale feedback during human intervention")

    monkeypatch.setattr(record, "record_session", capture)
    with pytest.raises(InferenceError, match="silent past"):
        cli._hitl_episodes(
            resolve_rig("yam_bimanual"),
            client=object(),
            human=object(),
            cameras={},
            sink=record.MemorySink(),
            scene=None,
            camera_panel=None,
            manifest={"episodes": []},
            manifest_path=tmp_path / "hitl.json",
            episodes=1,
            episode_seconds=1,
            fps=30,
            speed=1,
            chunk_size=1,
            max_step_rad=0.1,
            max_effector_step=0.1,
            carry_targets=False,
            prefetch=False,
            prefetch_margin_s=0,
            wait_between_episodes=False,
            backend_factory=None,
            input_fn=lambda _: "test",
        )
    assert parked == [True]
