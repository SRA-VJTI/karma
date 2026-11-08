"""One serialized model worker and a nonblocking control-loop interface."""

from __future__ import annotations

import threading
import time

import numpy as np

from ..inference import EncodedFramesUnsupported, InferenceError
from .control import PolicyControl
from .trace import PolicyTrace


class PolicySession:
    def __init__(self, client, instruction, *, speed=1.0, prefetch=True):
        self.client, self.instruction = client, instruction
        if not hasattr(client, "operation_lock"):
            client.operation_lock = threading.Lock()
        self.control = PolicyControl()
        self.trace = getattr(client, "trace", None) or PolicyTrace(client.config)
        self.owns_trace = not hasattr(client, "trace")
        self.action_hz = client.config.action_hz * speed
        # SDK pacing uses actual row consumption rate.
        if client.display_name == "dream":
            client.action_hz = self.action_hz
        self.prefetch = prefetch
        self._thread = None
        self._result = None
        self._lock = threading.Lock()
        self._generation = -1
        self._reset_generation = -1
        self.request_id = 0
        self.latency_s = 0.35
        self.last_error = None
        self.preview = None
        self.preview_id = 0
        self.progress = 0
        self.commanded = {}
        self.last_observation_at = None
        self._latest = None
        self.input_frames = {}
        self.closed = False
        self.trace.event("session_start", instruction=instruction)
        print(f"  policy artifacts: {self.trace.path}", flush=True)

    def invalidate(self):
        with self._lock:
            self._result = None
        self.preview = None
        self.trace.event(
            "invalidated", generation=self.control.generation, reason=self.control.reason
        )

    def update_generation(self):
        if self._generation != self.control.generation:
            self._generation = self.control.generation
            self.invalidate()
            print(f"  policy: {self.control.reason}", flush=True)
            return True
        return False

    def feed(self, observation):
        with self._lock:
            self._latest = (
                observation,
                {k: list(v) for k, v in self.commanded.items()},
                time.monotonic(),
            )
        feed = getattr(self.client, "feed", None)
        if (
            feed
            and self.control.permits(self._generation)
            and self._reset_generation == self._generation
        ):
            feed(observation, self.instruction, self.commanded)

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def due(self, executed, remaining):
        if self.busy or self._result is not None:
            return False
        if not self.prefetch:
            return remaining == 0
        refill = getattr(self.client, "refill_due", None)
        return (
            refill(executed, remaining)
            if refill
            else remaining / self.action_hz <= self.latency_s + 0.1
        )

    def request(self, observation):
        if self.busy or self._result is not None or not self.control.permits(self._generation):
            return
        generation = self._generation
        self.request_id += 1
        request_id = self.request_id
        anchor = {k: list(v) for k, v in self.commanded.items()}
        self.last_observation_at = time.monotonic()
        submitted_at = self.last_observation_at

        def run_locked():
            started = time.monotonic()
            try:
                if self._reset_generation != generation:
                    self.client.reset()
                    self._reset_generation = generation
                    self.trace.event("reset", generation=generation)
                if self.closed or not self.control.permits(generation):
                    return
                # Reset/previous-episode teardown may have waited for an old call.
                # Select the latest observation and its matching anchor after that wait.
                with self._lock:
                    latest = self._latest
                request_observation, request_anchor, captured_at = (
                    latest if latest is not None else (observation, anchor, submitted_at)
                )
                self.last_observation_at = captured_at
                if time.monotonic() - captured_at > self.client.config.max_prediction_age_s:
                    raise InferenceError("observation expired while waiting for policy reset")
                if self.client.config.capture_payloads:
                    np.savez_compressed(
                        self.trace.path / f"request-{request_id:06d}.npz",
                        state=request_observation.state,
                        **{
                            name: getattr(request_observation, name)
                            for name in ("top_cam", "left_cam", "right_cam", "side_cam")
                            if getattr(request_observation, name) is not None
                        },
                    )
                self.trace.event(
                    "request",
                    request_id=request_id,
                    generation=generation,
                    commanded=request_anchor,
                )
                if self.client.display_name == "dream":
                    actions = self.client.infer(
                        request_observation, self.instruction, commanded=request_anchor
                    )
                else:
                    try:
                        actions = self.client.infer(request_observation, self.instruction)
                    except EncodedFramesUnsupported:
                        self.client.jpeg_quality = 0
                        actions = self.client.infer(request_observation, self.instruction)
                if time.monotonic() - captured_at > self.client.config.max_prediction_age_s:
                    raise InferenceError("prediction expired before execution")
                actions = np.asarray(actions, dtype=float)
                if (
                    actions.ndim != 2
                    or actions.shape[1] != self.client.contract.dimension
                    or not 1 <= len(actions) <= 256
                    or not np.isfinite(actions).all()
                ):
                    raise InferenceError("invalid policy action chunk")
                if self.client.config.capture_payloads:
                    np.save(self.trace.path / f"actions-{request_id:06d}.npy", actions)
                with self._lock:
                    if self.control.permits(generation) and not self.closed:
                        self._result = (generation, request_id, actions)
                        self.input_frames = dict(getattr(self.client, "last_wire_frames", {}))
                        self.latency_s = time.monotonic() - started
                self.trace.event(
                    "response",
                    request_id=request_id,
                    generation=generation,
                    rows=len(actions),
                    latency=getattr(self.client, "last_latency", {}),
                )
            except Exception as exc:
                if self.control.permits(generation) and not self.closed:
                    self.last_error = type(exc).__name__ + ": " + str(exc)
                    self.control.pause(self.last_error)
                    self.trace.event("policy_error", error_type=type(exc).__name__)

        def run():
            with self.client.operation_lock:
                if not self.closed and self.control.permits(generation):
                    run_locked()

        self._thread = threading.Thread(target=run, name="karma-policy", daemon=True)
        self._thread.start()

    def take(self):
        with self._lock:
            item, self._result = self._result, None
        if item and self.control.permits(item[0]):
            self.preview_id = item[1]
            self.preview = item[2]
            self.progress = 0
            with self.control.lock:
                if self.control.permits(item[0]):
                    self.control.reason = "Executing policy"
            self.trace.event("preview", request_id=item[1], generation=item[0], rows=len(item[2]))
            return item[2]
        return None

    def record_commands(self, commands):
        for name, command in commands.items():
            self.commanded["q_l_chunk" if name == "left" else "q_r_chunk"] = [
                *command.position_rad,
                command.effector,
            ]

    def close(self):
        self.closed = True
        self.control.pause("Episode closed")
        self.invalidate()
        if self.owns_trace:
            self.trace.close()
