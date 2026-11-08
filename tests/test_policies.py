"""Provider lifecycle and dispatch invariants without motors, GPUs or cloud calls."""

import dataclasses
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_inference_record import _Reader, _state

from openpi_control.exceptions import ConfigurationError
from openpi_control.inference import DEFAULT_CONTRACT, BimanualObservation, InferenceError
from openpi_control.inference_record import InferenceRolloutSource
from openpi_control.policies.config import PolicyConfig
from openpi_control.policies.control import PolicyControl
from openpi_control.policies.factory import create_client
from openpi_control.policies.session import PolicySession


@pytest.fixture(autouse=True)
def local_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path))


class FakePolicy:
    managed = True
    contract = DEFAULT_CONTRACT
    display_name = "molmoact2"
    action_hz = 20
    last_latency = {}
    config = PolicyConfig(action_hz=20)

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, obs, instruction):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(2)
        return np.full((10, 14), 0.1 * self.calls)


def observation():
    frame = np.zeros((2, 2, 3), np.uint8)
    return BimanualObservation(frame, frame, frame, np.zeros(14))


def test_profiles_strict_and_precedence(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"provider":"pi05", "action_hz":25}')
    config = PolicyConfig.load(path, action_hz=30)
    assert config.action_hz == 30
    assert config.norm_tag == "yam-bimanual-merged"
    path.write_text('{"unknown": true}')
    with pytest.raises(ConfigurationError, match="unknown"):
        PolicyConfig.load(path)
    with pytest.raises(ConfigurationError, match="checkpoint"):
        PolicyConfig.load(provider="dream")
    with pytest.raises(ConfigurationError):
        PolicyConfig(action_hz=float("nan"))


def test_dream_rejects_unsupported_rig_before_start():
    config = PolicyConfig.load(provider="dream", checkpoint="explicit/model")
    with pytest.raises(ConfigurationError, match="YAM"):
        create_client(config, dataclasses.replace(DEFAULT_CONTRACT, arm_dof=5))


def test_stop_discards_late_reply_and_resume_resets():
    client = FakePolicy()
    session = PolicySession(client, "fold")
    session.update_generation()
    session.request(observation())
    assert client.entered.wait(1)
    session.control.pause()
    session.update_generation()
    assert not session.control.permits(0)
    client.release.set()
    session._thread.join(1)
    assert session.take() is None
    session.control.resume()
    session.update_generation()
    session.request(observation())
    session._thread.join(1)
    assert np.allclose(session.take(), 0.2)
    assert client.resets == 2
    session.close()


def test_only_one_inference_during_reset():
    client = FakePolicy()
    session = PolicySession(client, "fold")
    session.update_generation()
    session.request(observation())
    assert client.entered.wait(1)
    session.control.pause()
    session.control.resume()
    session.update_generation()
    session.request(observation())
    assert client.calls == 1
    client.release.set()
    session._thread.join(1)
    session.close()


def test_controls_preserve_human_ownership():
    control = PolicyControl()
    control.intervention(True)
    control.pause()
    control.resume()
    assert control.human and control.paused
    control.home_and_stop()
    control.intervention(False)
    control.resume()
    assert control.home and control.paused


def source_for(client, on_chunk=None):
    arms = {n: SimpleNamespace(latest_state=_state(n)) for n in ("left", "right")}
    source = InferenceRolloutSource(
        arms=arms,
        readers={n: _Reader() for n in DEFAULT_CONTRACT.cameras},
        client=client,
        instruction="fold",
        episode_seconds=10,
        fps=30,
        speed=1,
        prefetch=False,
        on_chunk=on_chunk,
    )
    return source, arms


def tick(source, arms):
    states = {n: _state(n) for n in arms}
    for n, st in states.items():
        arms[n].latest_state = st
    return source.act(states)


def test_managed_source_nonblocking_preview_hold_resume_and_home():
    client = FakePolicy()
    previews = []
    source, arms = source_for(client, lambda chunk: previews.append(chunk))
    started = time.monotonic()
    tick(source, arms)
    assert time.monotonic() - started < 0.2
    assert client.entered.wait(1)
    client.release.set()
    source.policy_session._thread.join(1)
    targets = tick(source, arms)
    assert previews and targets
    control = source.policy_session.control
    control.pause()
    hold = tick(source, arms)
    assert all(np.allclose(t.position_rad, 0) for t in hold.values())
    assert hold["left"].effector == targets["left"].effector
    control.resume()
    tick(source, arms)
    source.policy_session._thread.join(1)
    assert client.resets == 2
    control.home_and_stop()
    with pytest.raises(KeyboardInterrupt):
        tick(source, arms)
    source.close()


def test_dispatch_guard_blocks_stop_race():
    client = FakePolicy()
    source, arms = source_for(client)
    tick(source, arms)
    source.policy_session.control.pause()
    with source.dispatch_guard() as allowed:
        assert not allowed
    client.release.set()
    source.policy_session._thread.join(1)
    source.close()


def test_provider_failure_pauses_without_actions():
    client = FakePolicy()

    def fail(*args):
        raise InferenceError("unavailable")

    client.infer = fail
    session = PolicySession(client, "fold")
    session.update_generation()
    session.request(observation())
    session._thread.join(1)
    assert session.control.paused
    assert session.take() is None
    session.close()


def test_home_and_stop_keeps_hitl_partial_episode(monkeypatch, tmp_path):
    from openpi_control import cli, dagger, record
    from openpi_control.rigs import resolve_rig

    writer = record.MemorySink()
    manifest = {"episodes": []}
    control = PolicyControl()
    policy = SimpleNamespace(policy_session=SimpleNamespace(control=control))
    monkeypatch.setattr(cli, "power_up", lambda *a, **k: (object(), []))
    monkeypatch.setattr(cli, "settle_arm_states", lambda *a: {})
    parked = []
    monkeypatch.setattr(cli, "power_down", lambda *a, **k: parked.append(True) or 0)
    monkeypatch.setattr(cli, "InferenceRolloutSource", lambda **k: policy)
    monkeypatch.setattr(
        dagger,
        "DaggerSource",
        lambda **k: SimpleNamespace(
            _policy=policy,
            stats=dagger.InterventionStats(),
            describe=lambda: "test",
            close=lambda: None,
        ),
    )

    def record_episode(**kwargs):
        kwargs["sink"].add_frame({"frame": 1})
        kwargs["sink"].save_episode()
        control.home_and_stop()
        return SimpleNamespace(ended_by="interrupted")

    monkeypatch.setattr(record, "record_session", record_episode)

    def prompt(message):
        assert message.startswith("Prompt"), "home must not prompt for another trial or review"
        return "fold"

    result = cli._hitl_episodes(
        resolve_rig("yam_bimanual"),
        client=object(),
        human=object(),
        cameras={},
        sink=writer,
        scene=None,
        camera_panel=None,
        manifest=manifest,
        manifest_path=tmp_path / "hitl.json",
        episodes=3,
        episode_seconds=1,
        fps=30,
        speed=1,
        chunk_size=None,
        max_step_rad=0.1,
        max_effector_step=0.1,
        carry_targets=False,
        prefetch=False,
        prefetch_margin_s=0,
        wait_between_episodes=True,
        backend_factory=None,
        input_fn=prompt,
    )
    assert result == 0 and parked == [True]
    assert len(writer.episodes) == 1
    assert manifest["episodes"][0]["home_and_stop"]
    assert manifest["episodes"][0]["label"] == "unlabeled"


def test_reset_uses_latest_observation_and_matching_anchor():
    client = FakePolicy()
    resetting = threading.Event()
    release_reset = threading.Event()
    seen = []

    def reset():
        resetting.set()
        assert release_reset.wait(1)

    def infer(obs, instruction):
        seen.append(obs.state.copy())
        return np.zeros((4, 14))

    client.reset, client.infer = reset, infer
    session = PolicySession(client, "fold")
    session.update_generation()
    session.request(observation())
    assert resetting.wait(1)
    newer = dataclasses.replace(observation(), state=np.ones(14))
    session.feed(newer)
    release_reset.set()
    session._thread.join(1)
    assert np.all(seen[0] == 1)
    assert session.take() is not None
    session.close()


def test_expired_prediction_is_rejected():
    client = FakePolicy()
    client.config = dataclasses.replace(client.config, max_prediction_age_s=0.01)

    def infer(obs, instruction):
        time.sleep(0.03)
        return np.zeros((4, 14))

    client.infer = infer
    session = PolicySession(client, "fold")
    session.update_generation()
    session.request(observation())
    session._thread.join(1)
    assert session.control.paused
    assert session.take() is None
    assert "expired" in session.last_error
    session.close()


def test_startup_feedback_gap_recovers_without_resume():
    client = FakePolicy()
    client.release.set()
    source, arms = source_for(client)
    stale = {
        n: dataclasses.replace(_state(n), monotonic_timestamp=time.monotonic() - 0.3) for n in arms
    }
    assert source.act(stale) == {}
    control = source.policy_session.control
    assert control.waiting and not control.paused
    assert not client.calls
    assert tick(source, arms) == {}  # fresh observation clears startup inhibition
    assert not control.waiting and not control.paused
    tick(source, arms)
    source.policy_session._thread.join(1)
    tick(source, arms)
    assert source._has_executed_policy
    assert "adaptive-prefetch" not in source.describe()  # source_for disables prefetch
    source.close()


def test_startup_recovery_never_clears_operator_stop():
    client = FakePolicy()
    source, arms = source_for(client)
    assert source.act({n: None for n in arms}) == {}
    control = source.policy_session.control
    control.pause()
    tick(source, arms)
    assert control.paused
    assert client.calls == 0
    source.close()


def test_feedback_gap_after_motion_still_requires_resume():
    client = FakePolicy()
    client.release.set()
    source, arms = source_for(client)
    tick(source, arms)
    source.policy_session._thread.join(1)
    tick(source, arms)
    assert source._has_executed_policy
    assert source.act({n: None for n in arms}) == {}
    assert source.policy_session.control.paused
    tick(source, arms)
    assert source.policy_session.control.paused
    assert "Resume" in source.policy_session.control.reason
    source.close()


def test_startup_camera_gap_recovers_without_resume(monkeypatch):
    from openpi_control import inference_record
    from openpi_control.exceptions import StaleStateError

    client = FakePolicy()
    client.release.set()
    source, arms = source_for(client)
    original = inference_record.build_observation

    def stale(*args, **kwargs):
        raise StaleStateError("camera warming up")

    monkeypatch.setattr(inference_record, "build_observation", stale)
    assert tick(source, arms) == {}
    assert source.policy_session.control.waiting
    monkeypatch.setattr(inference_record, "build_observation", original)
    tick(source, arms)
    tick(source, arms)
    source.policy_session._thread.join(1)
    tick(source, arms)
    assert source._has_executed_policy
    source.close()


def test_startup_wait_retains_feedback_timeout():
    client = FakePolicy()
    source, arms = source_for(client)
    states = {
        n: dataclasses.replace(_state(n), monotonic_timestamp=time.monotonic() - 1.1) for n in arms
    }
    with pytest.raises(InferenceError, match="silent past"):
        source.act(states)
    assert client.calls == 0
    source.close()
