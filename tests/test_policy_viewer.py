"""Viser callbacks remain intents and preview rendering never drives an arm."""

import time
from types import SimpleNamespace

import numpy as np

from openpi_control import viz
from openpi_control.inference import DEFAULT_CONTRACT
from openpi_control.policies.control import PolicyControl
from openpi_control.policies.viewer import PolicyPanel


class Handle:
    visible = False
    disabled = False
    content = ""
    callback = None

    def on_click(self, callback):
        self.callback = callback

    def remove(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Gui:
    def __init__(self):
        self.buttons = {}

    def add_folder(self, label):
        return Handle()

    def add_markdown(self, text):
        return Handle()

    def add_button(self, label):
        self.buttons[label] = Handle()
        return self.buttons[label]


class Scene:
    def __init__(self, specs=(), **kwargs):
        self.arms = {n: SimpleNamespace(base_frame=Handle()) for n in specs}
        self.server = kwargs.get("server", SimpleNamespace(gui=Gui()))
        self.previews = []

    def clear_chunk(self):
        pass

    def update_chunk(self, chunks):
        self.previews.append(chunks)

    def update(self, *args):
        pass

    def set_chunk_progress(self, *args):
        pass


class Cameras:
    def __init__(self, *args, **kwargs):
        pass

    def push(self, *args):
        pass

    def step(self, *args):
        pass

    def remove(self):
        pass


def test_preview_controls_and_human_ownership(monkeypatch):
    monkeypatch.setattr(viz, "ArmSceneVisualizer", Scene)
    monkeypatch.setattr(viz, "CameraPanel", Cameras)
    scene = Scene()
    session = SimpleNamespace(
        control=PolicyControl(),
        _generation=0,
        preview=np.zeros((4, 14)),
        progress=1,
        preview_id=1,
        input_frames={},
        last_observation_at=time.monotonic(),
        action_hz=20,
        instruction="fold",
        client=SimpleNamespace(
            contract=DEFAULT_CONTRACT,
            config=SimpleNamespace(checkpoint="test"),
            last_latency={},
            display_name="dream",
        ),
    )
    source = SimpleNamespace(policy_session=session, _arms={})
    panel = PolicyPanel(scene, SimpleNamespace(arms=[]), source, {})
    try:
        assert set(scene.server.gui.buttons) == {"Stop policy", "Resume policy", "Home and stop"}
        deadline = time.monotonic() + 1
        while not scene.previews and time.monotonic() < deadline:
            time.sleep(0.01)
        assert scene.previews and not panel.done.is_set()
        panel.stop.callback(None)
        assert session.control.paused
        panel.resume.callback(None)
        assert not session.control.paused
        session.control.intervention(True)
        panel.resume.callback(None)
        assert session.control.human
        panel.home.callback(None)
        assert session.control.home
    finally:
        panel.close()
