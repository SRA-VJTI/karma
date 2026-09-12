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


def _source(client=None, **kwargs) -> InferenceRolloutSource:
    kwargs.setdefault("instruction", "fold the towel")
    kwargs.setdefault("episode_seconds", 10.0)
    kwargs.setdefault("speed", 1.0)
    kwargs.setdefault("prefetch", False)
    return InferenceRolloutSource(
        arms={
            "left": SimpleNamespace(latest_state=_state("left")),
            "right": SimpleNamespace(latest_state=_state("right")),
        },
        readers={"top": _Reader(), "left_wrist": _Reader(), "right_wrist": _Reader()},
        client=client if client is not None else _Client(),  # type: ignore[arg-type]
        **kwargs,
    )


def test_acting_without_polling_carries_no_episode_event() -> None:
    # The DAgger recorder owns the episode clock, because an episode has to end
    # on time while the human -- not this source -- is driving.
    source = _source()
    try:
        targets = source.act({"left": _state("left"), "right": _state("right")})
    finally:
        source.close()

    assert set(targets) == {"left", "right"}
    assert len(targets["left"].position_rad) == 6


def test_taking_back_the_arms_drops_the_chunk_planned_before_the_correction() -> None:
    # The queued actions continue a trajectory from before the human touched
    # the scene; replaying them undoes the correction at full speed.
    client = _Client()
    source = _source(client, chunk_size=3)
    states = {"left": _state("left"), "right": _state("right")}
    try:
        source.act(states)
        assert client.calls == 1, "one chunk fetched, two of its three actions still queued"

        source.resume_after_intervention(states, effector={"left": 0.2, "right": 0.2})
        source.act(states)
    finally:
        source.close()

    assert client.calls == 2, "the next tick re-planned instead of resuming the old chunk"


def test_the_policy_takes_the_arms_back_holding_what_the_human_was_holding() -> None:
    source = _source()
    # Measured 0.5 (the jaw stopped on an object), commanded 0.05 by the human.
    states = {"left": _state("left"), "right": _state("right")}
    try:
        source.resume_after_intervention(states, effector={"left": 0.05, "right": 0.05})
        targets = source.act(states)
    finally:
        source.close()

    # The executor steps from the commanded grip, not from the measurement, so
    # the first commanded effector is within one step of 0.05 rather than 0.5.
    assert targets["left"].effector is not None
    assert abs(targets["left"].effector - 0.05) <= inference.DEFAULT_MAX_EFFECTOR_STEP
