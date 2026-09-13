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

from .exceptions import ConfigurationError, StaleStateError
from .inference import (
    DEFAULT_CHUNK_SPEED,
    DEFAULT_MAX_EFFECTOR_STEP,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_PREFETCH_MARGIN_S,
    MOLMOACT_ACTION_DIM,
    BoundedChunkExecutor,
    ChunkPrefetcher,
    EncodedFramesUnsupported,
    InferenceError,
    MolmoActClient,
    build_observation,
    time_scale,
)
from .record import ArmTarget, EpisodeEvent, TeleopSource, TeleopStep
from .safety import MAX_STALE_S, MAX_STATE_AGE_S, StateGuard
from .types import ArmState, PositionCommand

DEFAULT_MAX_STALE_S = MAX_STALE_S


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
        max_stale_s: float = DEFAULT_MAX_STALE_S,
        carry_targets: bool = False,
        stop: threading.Event | None = None,
        on_chunk: Callable[[np.ndarray], None] | None = None,
        on_tick: Callable[[Mapping[str, ArmState], Mapping[str, PositionCommand], int], None]
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
        if max_stale_s <= MAX_STATE_AGE_S:
            raise ConfigurationError(
                f"max_stale_s must exceed the {MAX_STATE_AGE_S:g}s freshness limit"
            )
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
        self._max_stale_s = float(max_stale_s)
        # When the current run of stale ticks began; None while states are fresh.
        self._state_guard = StateGuard(self._arms, max_stale_s=max_stale_s)
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
        self._plan = np.empty((0, MOLMOACT_ACTION_DIM), dtype=np.float64)
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
            self._plan = np.empty((0, MOLMOACT_ACTION_DIM), dtype=np.float64)
            return TeleopStep(event=EpisodeEvent.SAVE)

        return TeleopStep(targets=self.act(states), event=event)

    def act(self, states: Mapping[str, ArmState | None]) -> dict[str, ArmTarget]:
        """One tick of policy-commanded targets, carrying no episode event.

        Split out of :meth:`poll` for the DAgger recorder, which owns the
        episode clock itself because the policy is not always what is driving:
        an episode has to end on time while a human is still correcting, and
        this source must not advance its plan or spend an inference call on a
        tick whose commands would be thrown away.
        """
        fresh_states = self._fresh_states(states)
        if fresh_states is None:
            # Nothing is planned, stepped, or requested on this tick. The
            # recorder sees the same stale state and neither commands the arms
            # nor writes the frame, so advancing the plan here would spend an
            # action nobody ever sends.
            return {}
        if self._plan_index >= len(self._plan):
            try:
                observation = build_observation(self._arms, self._readers)
            except StaleStateError:
                # The same gap, starting between the check above and this read.
                return {}
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
        return targets

    def resume_after_intervention(
        self,
        states: Mapping[str, ArmState],
        *,
        effector: Mapping[str, float] | None = None,
    ) -> None:
        """Take the arms back from a human without carrying anything stale.

        Everything queued here was computed for a world that no longer exists.
        The remaining actions continue a trajectory from before the correction,
        and a prefetched chunk was asked for from an observation the human has
        since invalidated -- replaying either would undo the correction at full
        speed. Both are dropped, so the next tick re-plans from what the
        cameras see now.

        The executor is re-seeded here rather than at that next chunk boundary
        because its own ``reset`` would either carry the human's motion into a
        bounded step from a stale target, or (with ``carry_targets``) skip
        re-seeding entirely. ``effector`` is the gripper the human last
        commanded; see :meth:`~openpi_control.inference.BoundedChunkExecutor.seed`
        for why it cannot come from the measurement.
        """
        self._plan = np.empty((0, MOLMOACT_ACTION_DIM), dtype=np.float64)
        self._plan_index = 0
        if self._prefetcher is not None:
            self._prefetcher.drop()
        self._executor.seed(states, effector=effector)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._prefetcher is not None:
            self._prefetcher.close()

    def _fresh_states(self, states: Mapping[str, ArmState | None]) -> dict[str, ArmState] | None:
        return self._state_guard.check(states)

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
        try:
            observation = build_observation(self._arms, self._readers)
        except StaleStateError:
            return  # retried next tick; a prefetch is an optimisation, not a step
        self._prefetcher.submit(observation, self._instruction)
