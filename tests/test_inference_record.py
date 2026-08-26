"""Policy rollout source tests without cameras, HTTP, or robot hardware."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from openpi_control import inference
from openpi_control.inference_record import InferenceRolloutSource
from openpi_control.record import EpisodeEvent
from openpi_control.types import (
    ArmMode,
    ArmRole,
    ArmState,
    EffectorState,
    JointState,
)


def _state(name: str) -> ArmState:
    return ArmState(
        name=name,
        role=ArmRole.FOLLOWER,
        joints=JointState(
            names=tuple(f"joint_{index + 1}" for index in range(6)),
            position_rad=[0.0] * 6,
            velocity_rad_s=[0.0] * 6,
            effort_nm=[0.0] * 6,
            temperature_c=[25.0] * 6,
            current_a=[0.0] * 6,
        ),
        effector=EffectorState(position=0.5),
        monotonic_timestamp=time.monotonic(),
        wall_timestamp=0.0,
        sequence=1,
        mode=ArmMode.HOLD,
    )


class _Reader:
    pixel_format = "rgb8"

    def __init__(self) -> None:
        self.frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def latest(self) -> np.ndarray:
        return self.frame


class _Client:
    def __init__(self) -> None:
        self.calls = 0
        self.jpeg_quality = 95

    def infer(self, _observation, _instruction):
        self.calls += 1
        return np.arange(3 * 14, dtype=np.float64).reshape(3, 14) * 0.001


class _RawOnlyClient(_Client):
    def infer(self, _observation, _instruction):
        self.calls += 1
        if self.jpeg_quality > 0:
            raise inference.EncodedFramesUnsupported("raw frames required")
        return np.zeros((2, 14), dtype=np.float64)


def test_source_executes_a_configured_chunk_prefix() -> None:
    chunks: list[np.ndarray] = []
    client = _Client()
    source = InferenceRolloutSource(
        arms={
            "left": SimpleNamespace(latest_state=_state("left")),
            "right": SimpleNamespace(latest_state=_state("right")),
        },
        readers={
            "top": _Reader(),
            "left_wrist": _Reader(),
            "right_wrist": _Reader(),
        },
        client=client,  # type: ignore[arg-type]
        instruction="fold the towel",
        episode_seconds=10.0,
        speed=1.0,
        chunk_size=2,
        prefetch=False,
        on_chunk=chunks.append,
    )

    try:
        first = source.poll({"left": _state("left"), "right": _state("right")})
        second = source.poll({"left": _state("left"), "right": _state("right")})
    finally:
        source.close()

    assert first.event is EpisodeEvent.START
    assert set(first.targets) == {"left", "right"}
    assert second.event is EpisodeEvent.NONE
    assert client.calls == 1
    assert chunks[0].shape == (2, 14)


def test_source_saves_when_its_duration_expires() -> None:
    source = InferenceRolloutSource(
        arms={
            "left": SimpleNamespace(latest_state=_state("left")),
            "right": SimpleNamespace(latest_state=_state("right")),
        },
        readers={
            "top": _Reader(),
            "left_wrist": _Reader(),
            "right_wrist": _Reader(),
        },
        client=_Client(),  # type: ignore[arg-type]
        instruction="fold the towel",
        episode_seconds=0.001,
        speed=1.0,
        prefetch=False,
    )

    try:
        source.poll({"left": _state("left"), "right": _state("right")})
        time.sleep(0.01)
        step = source.poll({"left": _state("left"), "right": _state("right")})
    finally:
        source.close()

    assert step.event is EpisodeEvent.SAVE


def test_source_falls_back_to_raw_frames_for_an_old_server() -> None:
    client = _RawOnlyClient()
    source = InferenceRolloutSource(
        arms={
            "left": SimpleNamespace(latest_state=_state("left")),
            "right": SimpleNamespace(latest_state=_state("right")),
        },
        readers={
            "top": _Reader(),
            "left_wrist": _Reader(),
            "right_wrist": _Reader(),
        },
        client=client,  # type: ignore[arg-type]
        instruction="fold the towel",
        episode_seconds=10.0,
        speed=1.0,
        prefetch=False,
    )

    try:
        step = source.poll({"left": _state("left"), "right": _state("right")})
    finally:
        source.close()

    assert step.event is EpisodeEvent.START
    assert client.calls == 2
    assert client.jpeg_quality == 0
