"""LeRobot episode source for policy-driven inference rollouts.

The normal ``infer`` command is intentionally a continuous controller. This
module adapts the same policy client and bounded executor to the recording
protocol so a rollout run can save fixed-duration policy trials in the same
LeRobot v3 sink used by teleoperation.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .inference import (
    DEFAULT_CHUNK_SPEED,
    DEFAULT_MAX_EFFECTOR_STEP,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_PREFETCH_MARGIN_S,
    BoundedChunkExecutor,
    ChunkPrefetcher,
    EncodedFramesUnsupported,
    InferenceError,
    MolmoActClient,
    build_observation,
    time_scale,
)
from .exceptions import ConfigurationError
from .record import ArmTarget, EpisodeEvent, TeleopSource, TeleopStep
from .types import ArmState, PositionCommand


class InferenceRolloutSource(TeleopSource):
    """Drive one timed episode from MolmoAct and expose it to ``record_session``.

    A source instance owns exactly one episode. The outer rollout runner can
    close the hardware between source instances while keeping one LeRobot sink
    open for the complete dataset.
    """

    def __init__(
        self,
        *,
        arms: Mapping[str, Any],
        readers: Mapping[str, Any],
        client: MolmoActClient,
        instruction: str,
        episode_seconds: float,
        fps: int = 30,
        speed: float = DEFAULT_CHUNK_SPEED,
        chunk_size: int | None = None,
        max_step_rad: float = DEFAULT_MAX_STEP_RAD,
        max_effector_step: float = DEFAULT_MAX_EFFECTOR_STEP,
        limits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
        prefetch: bool = True,
        prefetch_margin_s: float = DEFAULT_PREFETCH_MARGIN_S,
        carry_targets: bool = False,
        stop: threading.Event | None = None,
        on_chunk: Callable[[np.ndarray], None] | None = None,
        on_tick: Callable[
            [Mapping[str, ArmState], Mapping[str, PositionCommand], int], None
        ]
        | None = None,
    ) -> None:
        if not instruction.strip():
            raise ConfigurationError("inference instruction must not be empty")
        if episode_seconds <= 0:
            raise ConfigurationError("inference episode duration must be positive")
        if fps <= 0:
            raise ConfigurationError("inference recording fps must be positive")
        if speed <= 0:
            raise ConfigurationError("inference chunk speed must be positive")
        if chunk_size is not None and chunk_size <= 0:
            raise ConfigurationError("inference chunk size must be positive")
        if prefetch_margin_s < 0:
            raise ConfigurationError("inference prefetch margin must not be negative")
        self._arms = dict(arms)
        self._readers = dict(readers)
        self._client = client
        self._instruction = instruction
        self._episode_seconds = float(episode_seconds)
        self._period = 1.0 / fps
        self._speed = float(speed)
        self._chunk_size = chunk_size
        self._limits = limits
        self._prefetch_margin_s = float(prefetch_margin_s)
        self._stop = stop if stop is not None else threading.Event()
        self._on_chunk = on_chunk
        self._on_tick = on_tick
        self._executor = BoundedChunkExecutor(
            max_step_rad=max_step_rad,
            max_effector_step=max_effector_step,
            carry_targets=carry_targets,
        )
        self._prefetcher = ChunkPrefetcher(client) if prefetch else None
        self._episode_started_at: float | None = None
        self._saved = False
        self._plan = np.empty((0, 14), dtype=np.float64)
        self._plan_index = 0
        self._closed = False

    def describe(self) -> str:
        chunk = "full" if self._chunk_size is None else str(self._chunk_size)
        prefetch = "prefetch" if self._prefetcher is not None else "no-prefetch"
        return (
            f"MolmoAct({self._instruction!r}, {self._episode_seconds:g}s, "
            f"speed={self._speed:g}, chunk={chunk}, {prefetch})"
        )

    def poll(self, states: Mapping[str, ArmState | None]) -> TeleopStep:
        if self._stop.is_set():
            return TeleopStep(event=EpisodeEvent.STOP)
        if self._saved:
            return TeleopStep(event=EpisodeEvent.STOP)
        if self._episode_started_at is None:
            self._episode_started_at = time.monotonic()
            event = EpisodeEvent.START
        else:
            event = EpisodeEvent.NONE

        if time.monotonic() - self._episode_started_at >= self._episode_seconds:
            self._saved = True
            self._plan = np.empty((0, 14), dtype=np.float64)
            return TeleopStep(event=EpisodeEvent.SAVE)

        fresh_states = self._fresh_states(states)
        if self._plan_index >= len(self._plan):
            observation = build_observation(self._arms, self._readers)
            self._executor.reset(fresh_states)
            actions = self._request(observation)
            if self._chunk_size is not None:
                if self._chunk_size > len(actions):
                    raise InferenceError(
                        f"requested chunk size {self._chunk_size}, but server returned "
                        f"only {len(actions)} actions"
                    )
                actions = actions[: self._chunk_size]
            self._plan = time_scale(actions, self._speed)
            self._plan_index = 0
            if self._on_chunk is not None:
                self._on_chunk(self._plan.copy())

        action = self._plan[self._plan_index]
        self._plan_index += 1
        commands = self._executor.step(action, limits=self._limits)
        targets = {
            name: ArmTarget(
                position_rad=tuple(float(value) for value in command.position_rad),
                effector=command.effector,
            )
            for name, command in commands.items()
        }
        if self._on_tick is not None:
            self._on_tick(fresh_states, commands, self._plan_index)
        self._maybe_prefetch()
        return TeleopStep(targets=targets, event=event)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._prefetcher is not None:
            self._prefetcher.close()

    def _fresh_states(
        self, states: Mapping[str, ArmState | None]
    ) -> dict[str, ArmState]:
        fresh: dict[str, ArmState] = {}
        for name in self._arms:
            state = states.get(name)
            if state is None:
                raise InferenceError(f"{name} state is missing during inference recording")
            if not state.is_fresh(0.25):
                raise InferenceError(
                    f"{name} state is {state.age_s * 1e3:.0f} ms old during inference recording"
                )
            fresh[name] = state
        return fresh

    def _request(self, observation: Any) -> np.ndarray:
        try:
            if self._prefetcher is not None:
                return self._prefetcher.take(observation, self._instruction)
            return self._client.infer(observation, self._instruction)
        except EncodedFramesUnsupported as err:
            if int(getattr(self._client, "jpeg_quality", 0)) <= 0:
                raise
            print(f"  frames   {err}", file=sys.stderr)
            print(
                "  frames   sending raw frames for the rest of this run",
                file=sys.stderr,
            )
            self._client.jpeg_quality = 0
            if self._prefetcher is not None:
                self._prefetcher.drop()
            return self._client.infer(observation, self._instruction)

    def _maybe_prefetch(self) -> None:
        if self._prefetcher is None or self._prefetcher.busy:
            return
        queued_s = (len(self._plan) - self._plan_index) * self._period
        if queued_s > self._prefetcher.latency_s + self._prefetch_margin_s:
            return
        observation = build_observation(self._arms, self._readers)
        self._prefetcher.submit(observation, self._instruction)
