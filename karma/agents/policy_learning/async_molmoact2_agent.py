"""Async MolmoAct2 bimanual YAM policy agent.

This client talks to the ``molmoact2_karma_server.py`` websocket server:

    metadata ::= msgpack dict sent by server on websocket accept
    request  ::= {
        "state": (14,) float32,
        "top_camera-images-rgb":   (3,H,W) uint8,
        "left_camera-images-rgb":  (3,H,W) uint8,
        "right_camera-images-rgb": (3,H,W) uint8,
        "task": str,  # optional
    }
    response ::= {"actions": (T,14) float32, "server_timing": {...}}

It mirrors the chunk-buffering behavior used by the existing OpenPI / ACT
agents, but deliberately does not send OpenPI RTC-only fields such as
``action_prefix`` because the MolmoAct2 server ignores them.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Literal, Tuple

import numpy as np
from dm_env.specs import Array

from karma.agents.agent import PolicyAgent
from karma.agents.constants import ActionSpec
from karma.robots.utils import Rate

InferenceMode = Literal["sync", "async", "async_rate_limited", "temporal_ensemble"]
ImagePreprocess = Literal["none", "center_crop", "resize", "pad"]
_ASYNC_MODES = ("async", "async_rate_limited")


@dataclass
class MolmoAct2ModelIOConfig:
    """Observation keys expected by the MolmoAct2-BimanualYAM server."""

    state_keys: Tuple[str, ...] = (
        "left-joint_pos",
        "left-gripper_pos",
        "right-joint_pos",
        "right-gripper_pos",
    )
    image_keys: Tuple[str, ...] = (
        "top_camera-images-rgb",
        "left_camera-images-rgb",
        "right_camera-images-rgb",
    )


def _recursive_flatten(obj: Any, prefix: str = "", sep: str = "-") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict):
                flat.update(_recursive_flatten(v, key, sep=sep))
            else:
                flat[key] = v
    else:
        flat[prefix] = obj
    return flat


def _parse_target_size(size: int | str | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    """Return (target_h, target_w). Strings may be "224" or "640x480"."""
    if size is None:
        return None
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("target_image_size must be positive")
        return (size, size)
    if isinstance(size, str):
        text = size.lower().replace(",", "x")
        if "x" not in text:
            val = int(text)
            return (val, val)
        a, b = text.split("x", 1)
        # Config side uses image convention HxW. The server's --resize flag uses
        # WIDTHxHEIGHT; keep the distinction explicit in this client docstring.
        h, w = int(a), int(b)
        if h <= 0 or w <= 0:
            raise ValueError("target_image_size dimensions must be positive")
        return (h, w)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        h, w = int(size[0]), int(size[1])
        if h <= 0 or w <= 0:
            raise ValueError("target_image_size dimensions must be positive")
        return (h, w)
    raise ValueError(f"invalid target_image_size: {size!r}")


def _as_hwc_rgb(img: Any, key: str) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"{key}: expected RGB image with 3 dimensions, got shape {arr.shape}")
    if arr.shape[-1] == 3:
        hwc = arr
    elif arr.shape[0] == 3:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"{key}: expected HWC or CHW RGB image, got shape {arr.shape}")
    if hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(hwc)


def _resize_direct(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _center_crop_and_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return _resize_direct(img[y0 : y0 + side, x0 : x0 + side], target_h, target_w)


def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


class _InferTimingReporter:
    def __init__(self, name: str, report_interval_s: float = 2.0) -> None:
        self._name = name
        self._report_interval_s = float(report_interval_s)
        self._calls = 0
        self._sum_ms = 0.0
        self._last_ms = 0.0
        self._window_start = time.monotonic()

    def record(self, infer_ms: float) -> None:
        self._last_ms = float(infer_ms)
        self._sum_ms += self._last_ms
        self._calls += 1
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed >= self._report_interval_s:
            hz = self._calls / elapsed if elapsed > 0 else 0.0
            avg = self._sum_ms / self._calls if self._calls else 0.0
            print(
                f"[{self._name}] server infer: {self._calls} calls in {elapsed:.2f}s "
                f"→ {hz:.2f} Hz, avg {avg:.1f} ms, last {self._last_ms:.1f} ms"
            )
            self._calls = 0
            self._sum_ms = 0.0
            self._window_start = now


class AsyncMolmoAct2Agent(PolicyAgent):
    """Bimanual YAM agent backed by a remote MolmoAct2 websocket server."""

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int | None = 8112,
        task: str | None = "",
        action_horizon: int | None = None,
        inference_mode: InferenceMode = "async",
        inference_interval_s: float | None = None,
        min_smoothed_actions: int = 1,
        max_smoothed_actions: int = 4,
        temporal_ensemble_k: float = 0.01,
        model_io_config: MolmoAct2ModelIOConfig | None = None,
        image_preprocess: ImagePreprocess = "none",
        target_image_size: int | str | tuple[int, int] | list[int] | None = None,
        publish_policy_images: bool = False,
        clip_gripper: bool = True,
        # Per-tick EMA on the selected action after chunk blending.
        # None / 1.0 disables. Smaller = smoother but laggier.
        output_smoothing_alpha: float | None = None,
        # Optional hardware-side safety filters. These operate on the final
        # absolute joint-position command emitted by the policy client, not on
        # the model's raw chunk. Units are radians/tick for arm joints and
        # normalized gripper units/tick for grippers.
        max_joint_step_per_tick: float | None = None,
        max_gripper_step_per_tick: float | None = None,
        # Optional cap on how far a commanded absolute setpoint may be from the
        # latest observed robot state. Useful when a raw model chunk begins with
        # an OOD jump; units are radians / normalized gripper units.
        max_action_delta_from_state: float | None = None,
        max_gripper_delta_from_state: float | None = None,
        debug_action_stats: bool = False,
        require_server_protocol: bool = True,
    ) -> None:
        valid_modes = ("sync", "async", "async_rate_limited", "temporal_ensemble")
        if inference_mode not in valid_modes:
            raise ValueError(f"inference_mode must be one of {valid_modes}; got {inference_mode!r}")
        if image_preprocess not in ("none", "center_crop", "resize", "pad"):
            raise ValueError("image_preprocess must be one of: none, center_crop, resize, pad")
        if inference_mode == "async_rate_limited" and (
            inference_interval_s is None or inference_interval_s <= 0
        ):
            raise ValueError("inference_mode='async_rate_limited' requires inference_interval_s > 0")
        if min_smoothed_actions < 0 or max_smoothed_actions < 0:
            raise ValueError("min_smoothed_actions and max_smoothed_actions must be non-negative")
        if min_smoothed_actions > max_smoothed_actions:
            raise ValueError("min_smoothed_actions cannot exceed max_smoothed_actions")

        try:
            from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "AsyncMolmoAct2Agent requires `openpi_client` for the websocket/msgpack wire format."
            ) from exc

        # openpi_client accepts either host+port or a full ws:// URI. Avoid
        # appending a second port when ip is already a websocket URL.
        client_port = None if ip.startswith("ws://") or ip.startswith("wss://") else port
        self._client = _websocket_client_policy.WebsocketClientPolicy(host=ip, port=client_port)
        self._server_metadata = dict(self._client.get_server_metadata() or {})
        protocol = self._server_metadata.get("server_protocol")
        if require_server_protocol and protocol not in (None, "molmoact2"):
            raise RuntimeError(
                f"connected server protocol is {protocol!r}, expected 'molmoact2'. "
                "Check the ip/port in the YAML config."
            )
        if protocol == "molmoact2":
            print(f"[AsyncMolmoAct2Agent] connected to MolmoAct2 server: {self._server_metadata}")

        self.task = task
        self.action_horizon = int(
            action_horizon
            or self._server_metadata.get("n_action_steps")
            or self._server_metadata.get("action_horizon")
            or 10
        )
        self.inference_mode: InferenceMode = inference_mode
        self.inference_interval_s = inference_interval_s
        self.min_smoothed_actions = int(min_smoothed_actions)
        self.max_smoothed_actions = int(max_smoothed_actions)
        self._te_k = float(temporal_ensemble_k)
        self.config = model_io_config or MolmoAct2ModelIOConfig()
        self._image_preprocess: ImagePreprocess = image_preprocess
        if image_preprocess != "none" and target_image_size is None:
            target_image_size = 224
        self._target_hw = _parse_target_size(target_image_size)
        self._publish_policy_images = bool(publish_policy_images)
        self._clip_gripper = bool(clip_gripper)
        if output_smoothing_alpha is not None and output_smoothing_alpha <= 0.0:
            output_smoothing_alpha = None
        if output_smoothing_alpha is not None and output_smoothing_alpha > 1.0:
            raise ValueError("output_smoothing_alpha must be in (0, 1], or null to disable")
        for name, value in (
            ("max_joint_step_per_tick", max_joint_step_per_tick),
            ("max_gripper_step_per_tick", max_gripper_step_per_tick),
            ("max_action_delta_from_state", max_action_delta_from_state),
            ("max_gripper_delta_from_state", max_gripper_delta_from_state),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive, or null to disable")
        self._output_smoothing_alpha = output_smoothing_alpha
        self._max_joint_step_per_tick = max_joint_step_per_tick
        self._max_gripper_step_per_tick = max_gripper_step_per_tick
        self._max_action_delta_from_state = max_action_delta_from_state
        self._max_gripper_delta_from_state = max_gripper_delta_from_state
        self._debug_action_stats = bool(debug_action_stats)
        self._last_action_stats_log_ts = 0.0
        self._last_output_action: np.ndarray | None = None
        self._last_state: np.ndarray | None = None
        self.use_joint_state_as_action = False

        self.inference_interval_rate = (
            Rate(1.0 / inference_interval_s, rate_name="molmoact2_inference_interval")
            if inference_interval_s is not None and inference_interval_s > 0
            else None
        )
        self._infer_timer = _InferTimingReporter(name=type(self).__name__)

        self.action_lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.last_actions: np.ndarray | None = None
        self.action_counter = 0
        self._obs: Dict[str, Any] | None = None
        self._stop = threading.Event()
        self._last_display_images: Dict[str, np.ndarray] = {}

        # Sync mode chunk state.
        self._sync_chunk: np.ndarray | None = None
        self._sync_index = 0

        # Temporal-ensemble state.
        self._te_chunks: Deque[Tuple[int, np.ndarray]] = deque()
        self._te_tick = 0
        self._te_latest_chunk: np.ndarray | None = None
        self._te_latest_emit_tick = 0

        if inference_mode in _ASYNC_MODES:
            self.action_thread = threading.Thread(
                target=self._action_loop,
                name="AsyncMolmoAct2Agent_inference",
                daemon=True,
            )
            self.action_thread.start()
        else:
            self.action_thread = None

    # ------------------------------------------------------------------ #
    # Metadata / specs
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "inference_mode": self.inference_mode,
            "inference_interval_s": self.inference_interval_s,
            "min_smoothed_actions": self.min_smoothed_actions,
            "max_smoothed_actions": self.max_smoothed_actions,
            "temporal_ensemble_k": self._te_k,
            "image_preprocess": self._image_preprocess,
            "target_image_size_hw": self._target_hw,
            "output_smoothing_alpha": self._output_smoothing_alpha,
            "max_joint_step_per_tick": self._max_joint_step_per_tick,
            "max_gripper_step_per_tick": self._max_gripper_step_per_tick,
            "max_action_delta_from_state": self._max_action_delta_from_state,
            "max_gripper_delta_from_state": self._max_gripper_delta_from_state,
            "debug_action_stats": self._debug_action_stats,
            "task": self.task,
            **self._server_metadata,
        }

    def action_spec(self) -> ActionSpec:
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }

    # ------------------------------------------------------------------ #
    # Observation preprocessing
    # ------------------------------------------------------------------ #

    def obs_to_model_input(self, obs: Dict[str, Any]) -> Dict[str, Any] | None:
        flat = _recursive_flatten(obs)
        required = list(self.config.state_keys) + list(self.config.image_keys)
        missing = [k for k in required if k not in flat]
        if missing:
            now = time.monotonic()
            if now - getattr(self, "_last_missing_log_ts", 0.0) > 2.0:
                preview = ", ".join(missing[:4]) + (" …" if len(missing) > 4 else "")
                print(f"[AsyncMolmoAct2Agent] obs not ready — waiting on: {preview}")
                self._last_missing_log_ts = now
            return None

        flat_state = [np.asarray(flat[k]).reshape(-1) for k in self.config.state_keys]
        state = np.ascontiguousarray(np.concatenate(flat_state, axis=-1), dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"MolmoAct2 state must be shape (14,), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("MolmoAct2 state contains non-finite values")
        self._last_state = state.copy()

        request: Dict[str, Any] = {"state": state}
        display_images: Dict[str, np.ndarray] = {}
        for key in self.config.image_keys:
            hwc = self._preprocess_image(flat[key], key)
            display_label = key.split("-", 1)[0]
            display_images[display_label] = hwc
            request[key] = np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))

        if self.task is not None:
            request["task"] = str(self.task)
        self._last_display_images = display_images
        return request

    def _preprocess_image(self, img: Any, key: str) -> np.ndarray:
        hwc = _as_hwc_rgb(img, key)
        if self._image_preprocess == "none":
            return hwc
        if self._target_hw is None:
            raise RuntimeError("target_image_size is required when image_preprocess != 'none'")
        target_h, target_w = self._target_hw
        if self._image_preprocess == "center_crop":
            return _center_crop_and_resize(hwc, target_h, target_w)
        if self._image_preprocess == "resize":
            return _resize_direct(hwc, target_h, target_w)
        return _resize_with_pad(hwc, target_h, target_w)

    # ------------------------------------------------------------------ #
    # Public act() — called by AgentNode.step()
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        raw = self(obs)
        if raw is None:
            return {}
        a = np.array(raw, dtype=np.float32, copy=True)
        if a.shape != (14,):
            raise ValueError(f"MolmoAct2 action must be shape (14,), got {a.shape}")
        a = self._smooth_output_action(a)
        left = a[:7]
        right = a[7:]
        if self._clip_gripper:
            left[-1] = np.clip(left[-1], 0.0, 1.0)
            right[-1] = np.clip(right[-1], 0.0, 1.0)

        action: Dict[str, Any] = {
            "left": {"pos": left},
            "right": {"pos": right},
            "_chunk": self._snapshot_chunk(),
        }
        if self._publish_policy_images:
            action["_images"] = dict(self._last_display_images)
        return action

    def _smooth_output_action(self, action: np.ndarray) -> np.ndarray:
        """Filter the one absolute joint-position command emitted this tick.

        The model/server can occasionally return a chunk whose first setpoint is
        too far from the robot's measured state, or whose consecutive setpoints
        have hardware-unfriendly jumps.  EMA reduces jitter, while the optional
        caps below make the final command stream physically smooth even when a
        raw chunk is noisy.
        """
        prev = self._last_output_action
        filtered = np.array(action, dtype=np.float32, copy=True)

        alpha = self._output_smoothing_alpha
        if alpha is not None and alpha < 1.0 and prev is not None and prev.shape == filtered.shape:
            filtered = (alpha * filtered + (1.0 - alpha) * prev).astype(np.float32)

        filtered = self._clip_action_step(filtered, prev)
        filtered = self._clip_action_from_state(filtered)
        self._last_output_action = np.array(filtered, dtype=np.float32, copy=True)
        return filtered

    def _clip_action_from_state(self, action: np.ndarray) -> np.ndarray:
        state = self._last_state
        if state is None or state.shape != action.shape:
            return action
        return self._clip_action_delta(
            action,
            state,
            joint_limit=self._max_action_delta_from_state,
            gripper_limit=self._max_gripper_delta_from_state,
        )

    def _clip_action_step(self, action: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        if prev is None or prev.shape != action.shape:
            return action
        return self._clip_action_delta(
            action,
            prev,
            joint_limit=self._max_joint_step_per_tick,
            gripper_limit=self._max_gripper_step_per_tick,
        )

    @staticmethod
    def _clip_action_delta(
        action: np.ndarray,
        reference: np.ndarray,
        joint_limit: float | None,
        gripper_limit: float | None,
    ) -> np.ndarray:
        if joint_limit is None and gripper_limit is None:
            return action
        out = np.array(action, dtype=np.float32, copy=True)
        if joint_limit is not None:
            arm_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            delta = np.clip(out[arm_idx] - reference[arm_idx], -joint_limit, joint_limit)
            out[arm_idx] = reference[arm_idx] + delta
        if gripper_limit is not None:
            grip_idx = [6, 13]
            delta = np.clip(out[grip_idx] - reference[grip_idx], -gripper_limit, gripper_limit)
            out[grip_idx] = reference[grip_idx] + delta
        return out.astype(np.float32)

    def __call__(self, obs: Dict[str, Any]) -> np.ndarray | None:
        model_input = self.obs_to_model_input(obs)
        if model_input is None:
            return None
        with self.obs_lock:
            self._obs = model_input
        if self.inference_mode == "sync":
            return self._step_sync(model_input)
        if self.inference_mode == "temporal_ensemble":
            return self._step_temporal_ensemble(model_input)
        if self.last_actions is None:
            return None
        return self.select_action()

    def _snapshot_chunk(self) -> Dict[str, Any] | None:
        if self.inference_mode == "temporal_ensemble":
            with self.action_lock:
                if self._te_latest_chunk is None:
                    return None
                offset = self._te_tick - self._te_latest_emit_tick
                if offset < 0 or offset >= self._te_latest_chunk.shape[0]:
                    return None
                remaining = self._te_latest_chunk[offset:]
        elif self.inference_mode == "sync":
            with self.action_lock:
                if self._sync_chunk is None:
                    return None
                remaining = self._sync_chunk[self._sync_index :]
        else:
            with self.action_lock:
                if self.last_actions is None:
                    return None
                remaining = self.last_actions[self.action_counter :]

        if remaining.ndim != 2 or remaining.shape[0] == 0 or remaining.shape[1] != 14:
            return None
        return {
            "left": np.ascontiguousarray(remaining[:, :7], dtype=np.float32),
            "right": np.ascontiguousarray(remaining[:, 7:], dtype=np.float32),
        }

    # ------------------------------------------------------------------ #
    # Inference / chunking
    # ------------------------------------------------------------------ #

    def _infer(self, obs: Dict[str, Any]) -> np.ndarray:
        t0 = time.monotonic()
        response = self._client.infer(obs)
        wall_ms = (time.monotonic() - t0) * 1000.0
        if "error" in response:
            raise RuntimeError(f"MolmoAct2 server error: {response['error']}")
        if "actions" not in response:
            raise RuntimeError(f"MolmoAct2 server response missing 'actions': {response.keys()}")
        timing = response.get("server_timing") or {}
        self._infer_timer.record(float(timing.get("infer_ms", wall_ms)))

        # msgpack_numpy deserializes arrays as read-only views over the websocket
        # bytes buffer. Make an explicit writable copy before clipping grippers.
        actions = np.array(response["actions"], dtype=np.float32, copy=True)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1 and actions.shape[0] == 14:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"MolmoAct2 returned actions with shape {actions.shape}; expected (T, 14)")
        if actions.shape[0] == 0:
            raise ValueError("MolmoAct2 returned an empty action chunk")
        if not np.isfinite(actions).all():
            raise ValueError("MolmoAct2 returned non-finite actions")
        actions = np.array(actions, dtype=np.float32, copy=True, order="C")
        if self._clip_gripper:
            actions[:, 6] = np.clip(actions[:, 6], 0.0, 1.0)
            actions[:, 13] = np.clip(actions[:, 13], 0.0, 1.0)
        self._maybe_report_action_stats(actions, obs)
        return actions

    def _maybe_report_action_stats(self, actions: np.ndarray, obs: Dict[str, Any]) -> None:
        if not self._debug_action_stats:
            return
        now = time.monotonic()
        if now - self._last_action_stats_log_ts < 2.0:
            return
        self._last_action_stats_log_ts = now
        state = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
        arm_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        grip_idx = [6, 13]
        first_delta = actions[0] - state if state.shape == (14,) else actions[0] * 0.0
        chunk_steps = np.diff(np.concatenate([state[None, :], actions], axis=0), axis=0) if state.shape == (14,) else np.diff(actions, axis=0)
        arm_first_max = float(np.max(np.abs(first_delta[arm_idx])))
        grip_first_max = float(np.max(np.abs(first_delta[grip_idx])))
        arm_step_max = float(np.max(np.abs(chunk_steps[:, arm_idx]))) if chunk_steps.size else 0.0
        grip_step_max = float(np.max(np.abs(chunk_steps[:, grip_idx]))) if chunk_steps.size else 0.0
        print(
            "[AsyncMolmoAct2Agent] raw action stats: "
            f"shape={tuple(actions.shape)}, "
            f"firstΔ arm={arm_first_max:.3f} rad grip={grip_first_max:.3f}, "
            f"max step arm={arm_step_max:.3f} rad grip={grip_step_max:.3f}"
        )

    def _step_sync(self, obs: Dict[str, Any]) -> np.ndarray:
        with self.action_lock:
            need_chunk = self._sync_chunk is None or self._sync_index >= self._sync_chunk.shape[0]
        if need_chunk:
            chunk = self._infer(obs)
            with self.action_lock:
                self._sync_chunk = chunk
                self._sync_index = 0
        with self.action_lock:
            assert self._sync_chunk is not None
            idx = min(self._sync_index, self._sync_chunk.shape[0] - 1)
            action = self._sync_chunk[idx]
            self._sync_index += 1
        return action

    def _action_loop(self) -> None:
        while not self._stop.is_set():
            if self._obs is None:
                time.sleep(0.01)
                continue
            with self.obs_lock:
                current_obs = dict(self._obs)
            with self.action_lock:
                start_inference_action_counter = self.action_counter

            inferred_action = self._infer(current_obs)
            self._blend_merge(inferred_action, start_inference_action_counter)

            if self.inference_interval_rate is not None:
                self.inference_interval_rate.sleep()

    def _blend_merge(self, inferred_action: np.ndarray, start_inference_action_counter: int) -> None:
        with self.action_lock:
            complete_inference_action_counter = self.action_counter
            consumed_during_inference = max(
                0,
                complete_inference_action_counter - start_inference_action_counter,
            )
            server_chunk_len = inferred_action.shape[0]
            skip = consumed_during_inference
            if skip >= server_chunk_len:
                print(
                    f"[AsyncMolmoAct2Agent] inference latency ({skip} ticks) >= chunk "
                    f"length ({server_chunk_len}); resetting to chunk head"
                )
                skip = 0
            new_action = inferred_action[skip:]

            if self.last_actions is None:
                self.last_actions = new_action
                self.action_counter = 0
            elif new_action.shape[0] < 2 and self.last_actions.shape[0] >= 2:
                print(
                    f"[AsyncMolmoAct2Agent] discarding length-{new_action.shape[0]} chunk "
                    f"(consumed_during_inference={consumed_during_inference}) — keeping old buffer"
                )
            else:
                remaining_actions = self.last_actions[self.action_counter :]
                target = min(consumed_during_inference, self.max_smoothed_actions)
                num_smoothed = max(self.min_smoothed_actions, target)
                num_smoothed = min(num_smoothed, remaining_actions.shape[0], new_action.shape[0])
                if num_smoothed > 0:
                    weights = np.linspace(1.0 / num_smoothed, 1.0, num_smoothed).reshape(-1, 1)
                    smoothed = (
                        weights * new_action[:num_smoothed]
                        + (1.0 - weights) * remaining_actions[:num_smoothed]
                    )
                    self.last_actions = np.concatenate([smoothed, new_action[num_smoothed:]], axis=0)
                else:
                    self.last_actions = new_action
                self.action_counter = 0

    def _step_temporal_ensemble(self, obs: Dict[str, Any]) -> np.ndarray:
        chunk = self._infer(obs)
        with self.action_lock:
            t = self._te_tick
            self._te_chunks.append((t, chunk))
            while self._te_chunks and self._te_chunks[0][0] + self._te_chunks[0][1].shape[0] <= t:
                self._te_chunks.popleft()
            self._te_latest_chunk = chunk
            self._te_latest_emit_tick = t

            contribs: list[np.ndarray] = []
            for emit_tick, c in self._te_chunks:
                offset = t - emit_tick
                if 0 <= offset < c.shape[0]:
                    contribs.append(c[offset])
            stacked = np.stack(contribs, axis=0)
            ages = np.arange(stacked.shape[0], dtype=np.float32)
            weights = np.exp(-self._te_k * ages)
            weights = weights / weights.sum()
            action = (stacked * weights[:, None]).sum(axis=0).astype(np.float32)
            self._te_tick += 1
        return action

    def select_action(self) -> np.ndarray:
        while self.last_actions is None and not self._stop.is_set():
            time.sleep(0.01)
        if self._stop.is_set():
            raise RuntimeError("AsyncMolmoAct2Agent was closed before the first action became available")
        with self.action_lock:
            assert self.last_actions is not None
            buf_len = self.last_actions.shape[0]
            idx = min(self.action_counter, buf_len - 1)
            action = self.last_actions[idx]
            if self.action_counter >= buf_len - 1:
                if self.action_counter == buf_len - 1:
                    print(
                        f"[AsyncMolmoAct2Agent] inference lag — repeating action at "
                        f"counter {self.action_counter} (buf_len={buf_len}, "
                        f"action_horizon={self.action_horizon})"
                    )
            else:
                self.action_counter += 1
        return action

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._stop.set()
        if self.action_thread is not None and self.action_thread.is_alive():
            self.action_thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            self._client._ws.close()

    def reset(self) -> None:
        with self.action_lock:
            self.last_actions = None
            self.action_counter = 0
            self._sync_chunk = None
            self._sync_index = 0
            self._last_output_action = None
            self._te_chunks.clear()
            self._te_tick = 0
            self._te_latest_chunk = None
            self._te_latest_emit_tick = 0
