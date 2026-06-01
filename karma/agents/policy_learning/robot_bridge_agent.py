"""Robot bridge agent for karma (RLT / DSRL online training).

Exposes the YAM as a *passive executor* over a length-prefixed pickle TCP
socket (default ``:6112``) so an external RL trainer (the robometer RLT/DSRL
loop, or ``rltoken/eval_baseline.py``) can drive the robot in a closed loop::

    trainer  --reset/step-->  RobotBridgeAgent  --commands-->  RobotNodes
             <--obs/done-----

Method-agnostic: it knows nothing about RLT (``ã + Δa`` chunk refinement) vs
DSRL (``z₀`` noise) — the algorithm lives entirely in the trainer; this agent
just moves the arms and reports obs.

It is a drop-in replacement for ``AsyncMolmoAct2Agent`` in a karma session:
same camera / joint_state / cmd topics. Instead of running a policy itself, it

  * caches the latest obs (state ``(14,)`` + 3 CHW-uint8 cameras) every
    ``act()`` tick,
  * plays out a trainer-sent action chunk one absolute setpoint per tick,
  * homes both arms on ``RESET`` (linear interp to a configurable home pose),
  * reports the resulting obs + ``done`` back to the trainer.

Wire protocol — matches ``rl_zknob_loop.BridgeClient`` /
``rltoken/eval_baseline.py::BridgeClient`` byte-for-byte::

    frame  = struct.pack(">I", len(payload)) + payload      # payload = pickle.dumps(msg)
    RESET  = {"type": "RESET", "task": str}                 -> obs response
    STEP   = {"type": "STEP", "chunk": (T,14) f32, "need_obs": True} -> obs response

    response = {
        "top_camera-images-rgb":  (3,H,W) uint8,
        "left_camera-images-rgb": (3,H,W) uint8,
        "right_camera-images-rgb":(3,H,W) uint8,
        "state":  (14,) float32,
        "prompt": task,
        "reward": 0.0,      # placeholder: the trainer scores via its own reward server
        "success": False,   # placeholder
        "done":    bool,    # True once step_count >= max_steps_per_episode
        "truncated": bool,
        "info":   {...},
    }

``reward``/``success`` are placeholders (the trainer queries RoboReward itself).
``done`` fires at ``max_steps_per_episode`` (default 43, matching the
``molmoact2-rlt`` trainer's ``MAX_STEPS_PER_EPISODE``).

Action layout (14-D): ``[left_joints(6), left_gripper(1), right_joints(6),
right_gripper(1)]`` -> ``{"left": {"pos": (7,)}, "right": {"pos": (7,)}}``.
"""
from __future__ import annotations

import logging
import pickle
import socket
import struct
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from dm_env.specs import Array

from karma.agents.agent import PolicyAgent
from karma.agents.constants import ActionSpec
from karma.agents.policy_learning.async_molmoact2_agent import (
    MolmoAct2ModelIOConfig,
    _as_hwc_rgb,
    _recursive_flatten,
)

log = logging.getLogger("robot_bridge_agent")

# Default home pose per arm: 6 zeroed joints + gripper open (1.0). Matches the
# molmoact2 session's RobotNode startup_joint_pos and CLAUDE.md's
# "linear interp to [0,...,1] per arm".
_DEFAULT_HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
_GRIPPER_IDX = (6, 13)


def _send_framed(sock: socket.socket, msg: Any) -> None:
    payload = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_framed(sock: socket.socket) -> Optional[Any]:
    header = _recvall(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = _recvall(sock, length)
    if payload is None:
        return None
    return pickle.loads(payload)


class RobotBridgeAgent(PolicyAgent):
    """Karma agent that turns the robot into a socket-driven executor for RL.

    Lives on the robot box. The trainer (GPU box / rltoken) connects to
    ``host:port`` and steps the robot via the protocol documented above.
    Method-agnostic — works for the RLT chunk-refinement loop, the DSRL noise
    loop, or any trainer speaking the same protocol.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 6112,
        task: str = "stack the cubes",
        max_steps_per_episode: int = 43,
        home_on_reset: bool = True,
        home_pose: Optional[Tuple[float, ...]] = None,
        home_interp_ticks: int = 30,
        clip_gripper: bool = True,
        model_io_config: Optional[MolmoAct2ModelIOConfig] = None,
        wait_timeout_s: float = 30.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.task = task
        self.max_steps = int(max_steps_per_episode)
        self.home_on_reset = bool(home_on_reset)
        self.home_pose = np.asarray(home_pose if home_pose is not None else _DEFAULT_HOME, dtype=np.float32)
        if self.home_pose.shape != (14,):
            raise ValueError(f"home_pose must be (14,), got {self.home_pose.shape}")
        self.home_interp_ticks = max(1, int(home_interp_ticks))
        self.clip_gripper = bool(clip_gripper)
        self.config = model_io_config or MolmoAct2ModelIOConfig()
        self.wait_timeout_s = float(wait_timeout_s)

        # Shared state guarded by `_cv`. act() (karma thread) is the producer of
        # obs and the consumer of the action queues; the TCP handler thread is
        # the producer of action queues and the consumer of obs.
        self._cv = threading.Condition()
        self._latest_state: Optional[np.ndarray] = None
        self._latest_images: Dict[str, np.ndarray] = {}
        self._obs_version = 0                       # bumped every act() tick with valid obs
        self._pending: Deque[np.ndarray] = deque()  # chunk setpoints awaiting execution
        self._home_queue: Deque[np.ndarray] = deque()
        self._last_cmd: Optional[np.ndarray] = None  # last 14-D command (held between chunks)
        self._step_count = 0
        self._missing_log_ts = 0.0

        self._stop = threading.Event()
        self._server_thread = threading.Thread(target=self._serve_forever, name="robot-bridge", daemon=True)
        self._server_thread.start()
        log.info("RobotBridgeAgent listening on %s:%d (max_steps=%d, home_on_reset=%s)",
                 self.host, self.port, self.max_steps, self.home_on_reset)

    # ------------------------------------------------------------------ #
    # karma Agent interface
    # ------------------------------------------------------------------ #
    def action_spec(self) -> ActionSpec:
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }

    def reset(self) -> None:
        """Session-level reset (karma TUI). Clears queues; trainer drives the rest."""
        with self._cv:
            self._pending.clear()
            self._home_queue.clear()
            self._step_count = 0
            self._cv.notify_all()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        extracted = self._extract_obs(obs)
        with self._cv:
            if extracted is not None:
                self._latest_state, self._latest_images = extracted
                self._obs_version += 1
            # Pick the command for this tick: homing first, then trainer chunk,
            # else hold the last absolute setpoint (no-op until trainer steps).
            cmd: Optional[np.ndarray] = None
            if self._home_queue:
                cmd = self._home_queue.popleft()
            elif self._pending:
                cmd = self._pending.popleft()
            if cmd is not None:
                self._last_cmd = cmd
            self._cv.notify_all()
            hold = self._last_cmd

        out_cmd = cmd if cmd is not None else hold
        if out_cmd is None:
            return {}  # obs not ready / nothing commanded yet -> robot holds
        return self._to_command(out_cmd)

    # ------------------------------------------------------------------ #
    # obs / action helpers
    # ------------------------------------------------------------------ #
    def _extract_obs(self, obs: Dict[str, Any]) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        flat = _recursive_flatten(obs)
        required = list(self.config.state_keys) + list(self.config.image_keys)
        missing = [k for k in required if k not in flat]
        if missing:
            now = time.monotonic()
            if now - self._missing_log_ts > 2.0:
                log.info("obs not ready — waiting on: %s", ", ".join(missing[:4]))
                self._missing_log_ts = now
            return None
        state = np.ascontiguousarray(
            np.concatenate([np.asarray(flat[k]).reshape(-1) for k in self.config.state_keys], axis=-1),
            dtype=np.float32,
        )
        if state.shape != (14,):
            raise ValueError(f"bridge state must be (14,), got {state.shape}")
        images: Dict[str, np.ndarray] = {}
        for key in self.config.image_keys:
            hwc = _as_hwc_rgb(flat[key], key)
            images[key] = np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))  # CHW uint8
        return state, images

    def _to_command(self, action14: np.ndarray) -> Dict[str, Any]:
        a = np.asarray(action14, dtype=np.float32).reshape(-1).copy()
        left, right = a[:7], a[7:]
        if self.clip_gripper:
            left[-1] = float(np.clip(left[-1], 0.0, 1.0))
            right[-1] = float(np.clip(right[-1], 0.0, 1.0))
        return {"left": {"pos": left}, "right": {"pos": right}}

    def _snapshot_response(self, *, done: bool, truncated: bool, extra_info: Optional[dict] = None) -> dict:
        """Build a trainer-facing obs response from the latest cached obs. Caller holds `_cv`."""
        resp: Dict[str, Any] = {key: self._latest_images.get(key) for key in self.config.image_keys}
        resp["state"] = None if self._latest_state is None else self._latest_state.copy()
        resp["prompt"] = self.task
        resp["reward"] = 0.0
        resp["success"] = False
        resp["done"] = bool(done)
        resp["truncated"] = bool(truncated)
        resp["info"] = {"step_count": self._step_count, "max_steps": self.max_steps}
        if extra_info:
            resp["info"].update(extra_info)
        return resp

    # ------------------------------------------------------------------ #
    # TCP server
    # ------------------------------------------------------------------ #
    def _serve_forever(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        srv.settimeout(1.0)
        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            log.info("trainer connected from %s", addr)
            try:
                self._handle_conn(conn)
            except Exception as exc:  # noqa: BLE001
                log.warning("trainer connection error: %s", exc)
            finally:
                conn.close()
                log.info("trainer disconnected (%s)", addr)
        srv.close()

    def _handle_conn(self, conn: socket.socket) -> None:
        while not self._stop.is_set():
            msg = _recv_framed(conn)
            if msg is None:
                return
            mtype = msg.get("type")
            if mtype == "RESET":
                resp = self._do_reset(msg.get("task"))
            elif mtype == "STEP":
                resp = self._do_step(np.asarray(msg["chunk"], dtype=np.float32))
            else:
                resp = {"error": f"unknown message type {mtype!r}"}
            _send_framed(conn, resp)

    def _do_reset(self, task: Optional[str]) -> dict:
        if task:
            self.task = str(task)
        with self._cv:
            self._pending.clear()
            self._step_count = 0
            if self.home_on_reset:
                start = self._last_cmd if self._last_cmd is not None else self._latest_state
                self._home_queue = deque(self._build_home_traj(start))
                # wait until homing is fully played out by act()
                self._cv.wait_for(lambda: len(self._home_queue) == 0, timeout=self.wait_timeout_s)
            # ensure at least one fresh obs after homing
            v = self._obs_version
            self._cv.wait_for(lambda: self._obs_version > v or self._latest_state is not None,
                              timeout=self.wait_timeout_s)
            return self._snapshot_response(done=False, truncated=False, extra_info={"event": "reset"})

    def _do_step(self, chunk: np.ndarray) -> dict:
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            return {"error": f"chunk must be (T,14), got {chunk.shape}"}
        with self._cv:
            self._pending.extend(np.ascontiguousarray(row, dtype=np.float32) for row in chunk)
            self._step_count += 1
            # wait for the whole chunk to be consumed by act()
            consumed = self._cv.wait_for(lambda: len(self._pending) == 0, timeout=self.wait_timeout_s)
            # then wait one more obs tick so the returned obs reflects the executed chunk
            v = self._obs_version
            self._cv.wait_for(lambda: self._obs_version > v, timeout=self.wait_timeout_s)
            done = self._step_count >= self.max_steps
            truncated = not consumed  # timed out waiting -> robot/karma not ticking
            return self._snapshot_response(done=done, truncated=truncated)

    def _build_home_traj(self, start: Optional[np.ndarray]) -> List[np.ndarray]:
        """Linear interp from `start` (or home if unknown) to the home pose."""
        if start is None:
            return [self.home_pose.copy()]
        start = np.asarray(start, dtype=np.float32).reshape(-1)
        if start.shape != (14,):
            return [self.home_pose.copy()]
        n = self.home_interp_ticks
        return [((1.0 - t) * start + t * self.home_pose).astype(np.float32)
                for t in np.linspace(0.0, 1.0, n)]

    def close(self) -> None:
        self._stop.set()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
