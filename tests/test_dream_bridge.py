"""Exercise the real isolated worker against a fake SDK, never the cloud."""

import sys
import threading
import time

import numpy as np
import pytest

from openpi_control.inference import DEFAULT_CONTRACT, BimanualObservation
from openpi_control.policies.config import PolicyConfig
from openpi_control.policies.dream import DreamPolicy
from openpi_control.policies.session import PolicySession

FAKE_SDK = """
import time
class Window:
    def __init__(self, client): self.client = client
    def build_observation(self): return dict(self.client.latest)
class DreamClient:
    def __init__(self):
        self.latest = {}; self.ticks = 0; self.last_timing = {}; self.mode = "async"
        self.policy_calls = 0
    def policy(self, config):
        self._window = Window(self); self.config = config; self.policy_calls += 1
    def get_models(self): return [{"id": "fake/model", "servable": True}]
    def reset_episode(self): self.latest = {}
    def set_observation(self, frames, state, timestamp=None):
        self.latest = dict(state); self.ticks += 1
    def get_action(self, commanded=None):
        observation = self._window.build_observation()
        before = self.ticks
        time.sleep(0.15)
        self.last_timing = {"ticks_during_inference": self.ticks - before,
                           "serve_infer_ms": 100, "prompt": observation["prompt"],
                           "action_hz": self.config["action_hz"], "policy_calls": self.policy_calls}
        return {"q_l_chunk": [[0.1]*6 + [1.1]]*4, "q_r_chunk": [[0.2]*6 + [-0.1]]*4}
    def refill_due(self, executed, remaining): return remaining <= 2
    def close(self): pass
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "dream.py").write_text(FAKE_SDK)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path / "logs"))
    config = PolicyConfig.load(
        provider="dream", checkpoint="fake/model", worker_python=sys.executable
    )
    client = DreamPolicy(config, DEFAULT_CONTRACT)
    assert client.health()["status"] == "ok"
    yield client
    client.close()
    assert client.process.poll() is not None


def obs(value=0):
    image = np.full((8, 8, 3), value, np.uint8)
    return BimanualObservation(image, image, image, np.zeros(14))


def test_worker_feeds_during_inference_and_maps_actions(client):
    client.reset()
    results = []
    errors = []

    def infer():
        try:
            results.append(client.infer(obs(), "fold", commanded={"q_l_chunk": [0] * 7}))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=infer)
    thread.start()
    for _ in range(15):
        client.feed(obs(255), "newest camera tick")
        time.sleep(0.01)
    thread.join(2)
    assert not errors
    assert results and results[0].shape == (4, 14)
    assert np.allclose(results[0][:, :6], 0.1)
    assert np.allclose(results[0][:, 7:13], 0.2)
    assert np.all(results[0][:, 6] == 1)
    assert np.all(results[0][:, 13] == 0)
    assert client.last_timing["ticks_during_inference"] > 0
    assert client.last_timing["prompt"] == "fold"
    assert client.refill_due(1, 2)
    assert not client.refill_due(1, 3)
    # Preview is the request image, not the white frames fed afterwards.
    from openpi_control.inference import decode_wire_frame

    assert decode_wire_frame(client.last_wire_frames["top"]).max() == 0


def test_effective_rate_does_not_compound_between_episodes(client):
    for _ in range(2):
        session = PolicySession(client, "fold", speed=0.5)
        assert session.action_hz == 10
        session.update_generation()
        session.request(obs())
        session._thread.join(2)
        assert session.take() is not None
        assert client.last_timing["action_hz"] == 10
        session.close()


def test_worker_shutdown_is_bounded(client):
    started = time.monotonic()
    client.close()
    assert time.monotonic() - started < 3


def test_episode_reset_does_not_rebuild_unchanged_policy(client):
    for _ in range(2):
        client.reset()
        client.infer(obs(), "fold")
        assert client.last_timing["policy_calls"] == 1
