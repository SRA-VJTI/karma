"""DSRL bridge agent for the bimanual YAM.

Replaces the MolmoAct2 client when an off-board RL trainer wants to drive the
robot via chunked action proposals. The trainer (running on the GPU box; see
``robometer-policy-learning``) opens a long-lived TCP socket to this agent
and exchanges length-prefixed pickle messages:

    Wire framing: ``struct.pack(">I", len(body)) + pickle.dumps(body)``

    RESET  {"type": "RESET", "task": <str>}
           -> formatted obs dict (5 top-level keys: 3 images + state + prompt)

    STEP   {"type": "STEP", "chunk": (10, 14) float32, "need_obs": True}
           -> obs dict + {"reward", "success", "done", "truncated", "info"}

    PING   {"type": "PING"}
           -> {"pong": True}

The agent owns a state machine driven by Karma's ``AgentNode`` tick loop
(10 Hz). TCP requests mutate the phase; ``act(obs)`` produces one (14,)
action per tick by reading the current phase:

    IDLE -> HOMING -> WAIT_RESET -> EXECUTING -> POST_CHUNK_WAIT -> HOLDING

The TCP handler is the only thread calling ``_handle_reset`` / ``_handle_step``;
Karma's main loop is the only thread calling ``act``. A single ``_lock`` guards
the shared state; two ``Event`` objects signal phase completion across threads.

SECURITY: pickle deserialization is intentional (matches the trainer's wire
format). Bind to the lab LAN only; never expose port 6112 to untrusted networks
without first swapping in a signed framing.
"""

from __future__ import annotations

import contextlib
import pickle
import socket
import socketserver
import struct
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from dm_env.specs import Array

from karma.agents.agent import PolicyAgent
from karma.agents.constants import ActionSpec
from karma.agents.policy_learning.async_molmoact2_agent import (
    MolmoAct2ModelIOConfig,
    _as_hwc_rgb,
    _recursive_flatten,
    _resize_with_pad,
)

# State-machine phases.
PHASE_IDLE = "IDLE"
PHASE_HOMING = "HOMING"
PHASE_WAIT_RESET = "WAIT_RESET"
PHASE_EXECUTING = "EXECUTING"
PHASE_POST_CHUNK_WAIT = "POST_CHUNK_WAIT"
PHASE_HOLDING = "HOLDING"

# Defaults. Per-arm home pose mirrors ``startup_joint_pos`` from the prod
# MolmoAct2 YAM config (gripper convention: 0 = open, 1 = closed).
HOME_POSE_PER_ARM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
ACTION_DIM = 14
CHUNK_HORIZON = 10
IMAGE_HW: Tuple[int, int] = (224, 224)
TICK_HZ = 10.0
HOME_TICKS = 20  # 2.0 s at 10 Hz
DEFAULT_MAX_STEPS = 6

# ---------------------------------------------------------------------------
# Safety bounds — empirically derived from the 44 stack-cube demos (10 Hz, the
# `action` column; analysis 2026-06-02). Demos and the bridge share the joint
# layout [L_j0..L_j5, L_grip, R_j0..R_j5, R_grip]. These cap how FAR (position
# envelope) and how FAST (per-tick rate limit) the arm can be commanded,
# independent of what the RL trainer sends — the seatbelt for online training.
# ---------------------------------------------------------------------------
# Per-tick |Δ| demo max over ALL arm joints = 0.265 rad; 0.35 = ~30% headroom,
# so a normal trajectory never trips but a single-tick slam is caught.
DEFAULT_MAX_JOINT_STEP = 0.35  # rad per 10 Hz tick (arm joints only)
# Arm-joint indices (grippers 6,13 excluded — release must be free to commit).
ARM_IDX = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
# Per-joint position envelope = demo action [min,max] padded ±0.15 rad.
# Grippers fixed to [0,1] (0=open, 1=closed); their padded entries below are the
# raw [0,1] and are also enforced explicitly in _clip.
JOINT_POS_LO = np.array(
    [-1.38, -0.17, -0.18, -1.51, -1.32, -2.04, 0.0,
     -0.70, -0.18, -0.15, -1.48, -0.94, -1.44, 0.0], dtype=np.float32)
JOINT_POS_HI = np.array(
    [0.82, 2.96, 2.85, 1.23, 0.57, 0.98, 1.0,
     1.24, 2.99, 2.94, 1.54, 0.59, 1.22, 1.0], dtype=np.float32)
# A clamp larger than this margin counts as the trainer genuinely fighting a
# limit (not float noise); CLAMP_TRIP_TICKS consecutive trips -> hw_fault, which
# the STEP handler already turns into a clean episode truncation + reset.
CLAMP_TRIP_MARGIN = 0.05  # rad
CLAMP_TRIP_TICKS = 5      # consecutive violating ticks (0.5 s at 10 Hz)


@dataclass
class _SharedState:
    phase: str = PHASE_IDLE
    pending_chunk: Optional[np.ndarray] = None
    chunk_pos: int = 0
    last_command: Optional[np.ndarray] = None
    last_obs: Optional[Dict[str, Any]] = None
    obs_ready: bool = False
    home_start_state: Optional[np.ndarray] = None
    home_tick: int = 0
    episode_step: int = 0
    task: str = ""
    hw_fault: bool = False
    clamp_trip_count: int = 0  # consecutive safety-clamp violations


class RobotBridgeAgent(PolicyAgent):
    """Karma ``PolicyAgent`` that proxies the YAM to a remote RL trainer."""

    use_joint_state_as_action: bool = False

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 6112,
        task: str = "",
        max_steps_per_episode: int = DEFAULT_MAX_STEPS,
        chunk_horizon: int = CHUNK_HORIZON,
        action_dim: int = ACTION_DIM,
        image_hw: Tuple[int, int] | list[int] = IMAGE_HW,
        home_ticks: int = HOME_TICKS,
        max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
        reward_hook: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self._max_steps = int(max_steps_per_episode)
        self._chunk_horizon = int(chunk_horizon)
        self._action_dim = int(action_dim)
        self._image_hw = (int(image_hw[0]), int(image_hw[1]))
        self._home_ticks = int(home_ticks)
        self._max_joint_step = float(max_joint_step)
        self._io = MolmoAct2ModelIOConfig()
        self._reward_hook = reward_hook or self._default_reward_hook

        self._lock = threading.Lock()
        # Serializes RPC handling across all open sockets. The trainer
        # (KarmaYAMRemoteEnv) opens a fresh socket per call rather than
        # pipelining on a single long-lived connection, so we accept
        # concurrent connections and only serialize the actual handler calls.
        self._op_lock = threading.Lock()
        self._homing_done = threading.Event()
        self._chunk_done = threading.Event()
        self._stop = threading.Event()
        self._state = _SharedState(task=str(task))

        self._server = _make_tcp_server((self.host, self.port), self)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="RobotBridgeAgent_tcp",
            daemon=True,
        )
        self._server_thread.start()
        print(f"[RobotBridgeAgent] listening on {self.host}:{self.port}")

    # ------------------------------------------------------------------ #
    # PolicyAgent API
    # ------------------------------------------------------------------ #

    def action_spec(self) -> ActionSpec:
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "max_steps_per_episode": self._max_steps,
            "chunk_horizon": self._chunk_horizon,
            "action_dim": self._action_dim,
            "image_hw": self._image_hw,
            "home_ticks": self._home_ticks,
            "max_joint_step": self._max_joint_step,
        }

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        formatted = self._format_obs(obs)
        if formatted is None:
            # Obs missing camera or state keys — emit no new command this tick.
            # RobotNode will hold whatever it last received.
            return {}

        with self._lock:
            self._state.last_obs = formatted
            self._state.obs_ready = True
            phase = self._state.phase
            current_state = formatted["state"]

            if phase == PHASE_HOMING:
                action_vec = self._step_homing(current_state)
            elif phase == PHASE_EXECUTING:
                action_vec = self._step_executing(current_state)
            elif phase == PHASE_POST_CHUNK_WAIT:
                action_vec = self._step_post_chunk_wait(current_state)
            else:
                # IDLE, WAIT_RESET, HOLDING — freeze.
                action_vec = self._hold(current_state)

            action_vec = self._clip(action_vec)
            self._state.last_command = action_vec.copy()

        return self._to_command_dict(action_vec)

    def reset(self) -> None:
        # rr-session may call this between recordings. Clear chunk + phase
        # only; leave the TCP listener and the trainer's long-lived connection
        # alone — the trainer's next RESET re-homes.
        with self._lock:
            self._state.phase = PHASE_IDLE
            self._state.pending_chunk = None
            self._state.chunk_pos = 0
            self._state.last_command = None
            self._state.home_start_state = None
            self._state.home_tick = 0
            self._state.clamp_trip_count = 0
        self._homing_done.clear()
        self._chunk_done.clear()

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self._server.shutdown()
        with contextlib.suppress(Exception):
            self._server.server_close()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

    # ------------------------------------------------------------------ #
    # Per-tick action production
    # ------------------------------------------------------------------ #

    def _hold(self, current_state: np.ndarray) -> np.ndarray:
        if self._state.last_command is not None:
            return self._state.last_command.copy()
        return current_state.astype(np.float32, copy=True)

    def _step_homing(self, current_state: np.ndarray) -> np.ndarray:
        if self._state.home_start_state is None:
            self._state.home_start_state = current_state.astype(np.float32, copy=True)
            self._state.home_tick = 0
        target = np.concatenate([HOME_POSE_PER_ARM, HOME_POSE_PER_ARM], axis=0)
        denom = max(self._home_ticks - 1, 1)
        alpha = min(1.0, (self._state.home_tick + 1) / denom)
        action = (1.0 - alpha) * self._state.home_start_state + alpha * target
        self._state.home_tick += 1
        if self._state.home_tick >= self._home_ticks:
            self._state.phase = PHASE_WAIT_RESET
            self._homing_done.set()
        return action.astype(np.float32)

    def _step_executing(self, current_state: np.ndarray) -> np.ndarray:
        chunk = self._state.pending_chunk
        if chunk is None or self._state.chunk_pos >= chunk.shape[0]:
            # Defensive — shouldn't happen if state machine is intact.
            self._state.phase = PHASE_POST_CHUNK_WAIT
            return self._hold(current_state)
        action = chunk[self._state.chunk_pos].astype(np.float32, copy=True)
        self._state.chunk_pos += 1
        if self._state.chunk_pos >= chunk.shape[0]:
            self._state.phase = PHASE_POST_CHUNK_WAIT
        return action

    def _step_post_chunk_wait(self, current_state: np.ndarray) -> np.ndarray:
        # One-tick state: obs captured here reflects the post-chunk pose,
        # then we signal the TCP thread and drop to HOLDING.
        action = self._hold(current_state)
        self._state.phase = PHASE_HOLDING
        self._chunk_done.set()
        return action

    def _clip(self, action: np.ndarray) -> np.ndarray:
        """Hard safety floor: every command passes through here before it is
        sent. Bounds position (envelope) and speed (per-tick rate limit) on the
        arm joints, leaves the grippers free to commit, and trips ``hw_fault``
        when the trainer keeps fighting a limit. Called under ``self._lock``.
        """
        a = action.astype(np.float32, copy=True)
        if a.shape != (self._action_dim,):
            raise ValueError(f"action must be ({self._action_dim},); got {a.shape}")
        raw = a.copy()

        # (1) Gripper position bound: 6 (left), 13 (right). 0=open, 1=closed.
        a[6] = float(np.clip(a[6], 0.0, 1.0))
        a[13] = float(np.clip(a[13], 0.0, 1.0))
        # (2) Arm-joint position envelope (demo-derived, padded). Gripper entries
        #     in the arrays are [0,1] so this is a no-op there after step (1).
        a = np.clip(a, JOINT_POS_LO, JOINT_POS_HI).astype(np.float32, copy=False)
        # (3) Per-tick rate limit on arm joints — caps commanded speed no matter
        #     what the trainer sends. Grippers exempt (release must commit fast).
        last = self._state.last_command
        if last is not None:
            delta = a[ARM_IDX] - last[ARM_IDX]
            delta = np.clip(delta, -self._max_joint_step, self._max_joint_step)
            a[ARM_IDX] = last[ARM_IDX] + delta
        # (4) If the command was meaningfully clamped for several ticks running,
        #     the trainer is grinding against a limit — trip hw_fault so the STEP
        #     handler truncates the episode (clean reset beats a silent grind).
        if float(np.abs(a - raw).max()) > CLAMP_TRIP_MARGIN:
            self._state.clamp_trip_count += 1
            if (self._state.clamp_trip_count >= CLAMP_TRIP_TICKS
                    and not self._state.hw_fault):
                self._state.hw_fault = True
                print(
                    f"[bridge] SAFETY: {self._state.clamp_trip_count} consecutive "
                    f"clamp trips -> hw_fault; episode will truncate",
                    flush=True,
                )
        else:
            self._state.clamp_trip_count = 0
        return a

    def _to_command_dict(self, action_vec: np.ndarray) -> Dict[str, Any]:
        left = np.ascontiguousarray(action_vec[:7], dtype=np.float32)
        right = np.ascontiguousarray(action_vec[7:], dtype=np.float32)
        return {"left": {"pos": left}, "right": {"pos": right}}

    # ------------------------------------------------------------------ #
    # Obs formatting: Karma bus -> trainer wire schema
    # ------------------------------------------------------------------ #

    def _format_obs(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        flat = _recursive_flatten(obs)
        required = list(self._io.state_keys) + list(self._io.image_keys)
        if any(k not in flat for k in required):
            return None

        state_parts = [np.asarray(flat[k]).reshape(-1) for k in self._io.state_keys]
        state = np.ascontiguousarray(np.concatenate(state_parts, axis=-1), dtype=np.float32)
        if state.shape != (self._action_dim,):
            raise ValueError(f"state must be ({self._action_dim},); got {state.shape}")
        if not np.isfinite(state).all():
            return None

        formatted: Dict[str, Any] = {"state": state, "prompt": self._state.task}
        target_h, target_w = self._image_hw
        for key in self._io.image_keys:
            hwc = _as_hwc_rgb(flat[key], key)
            padded = _resize_with_pad(hwc, target_h, target_w)
            chw = np.ascontiguousarray(np.transpose(padded, (2, 0, 1)), dtype=np.uint8)
            formatted[key] = chw
        return formatted

    # ------------------------------------------------------------------ #
    # TCP message handlers (called from the TCP server thread)
    # ------------------------------------------------------------------ #

    def _handle_ping(self, _msg: Dict[str, Any]) -> Dict[str, Any]:
        return {"pong": True}

    def _handle_reset(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        task = str(msg.get("task", "") or "")
        # Clear chunk_done BEFORE state mutations so a stale set() from the
        # prior episode can't unblock the next chunk wait.
        self._chunk_done.clear()
        self._homing_done.clear()
        with self._lock:
            self._state.task = task
            self._state.episode_step = 0
            self._state.pending_chunk = None
            self._state.chunk_pos = 0
            self._state.home_start_state = None
            self._state.home_tick = 0
            self._state.hw_fault = False
            self._state.clamp_trip_count = 0
            self._state.phase = PHASE_HOMING

        if not self._homing_done.wait(timeout=10.0):
            raise RuntimeError("homing did not complete within 10 s")

        # Pause for the operator to restage the workspace between episodes.
        self._prompt_tty("[dsrl] restaged, press enter to start episode... ", default="")

        with self._lock:
            obs = dict(self._state.last_obs) if self._state.last_obs is not None else None
        if obs is None:
            raise RuntimeError("no formatted obs available after homing")
        return obs

    def _handle_step(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        chunk = np.asarray(msg["chunk"], dtype=np.float32)
        if chunk.shape != (self._chunk_horizon, self._action_dim):
            raise ValueError(
                f"chunk must be ({self._chunk_horizon}, {self._action_dim}); got {chunk.shape}"
            )
        if not np.isfinite(chunk).all():
            raise ValueError("chunk contains non-finite values")

        # Log motion the chunk is asking for, vs. the robot's current state.
        # If end-vs-current delta is near zero, the trainer is asking for a
        # hold — arms won't visibly move regardless of bridge correctness.
        with self._lock:
            cur_state = (
                self._state.last_obs["state"].copy()
                if self._state.last_obs is not None
                else None
            )
            step_idx_for_log = self._state.episode_step
        if cur_state is not None:
            end_delta = chunk[-1] - cur_state
            within_chunk = chunk[-1] - chunk[0]
            print(
                f"[bridge] STEP {step_idx_for_log}: |chunk[-1]-cur|_max={np.abs(end_delta).max():.4f} "
                f"|chunk[-1]-chunk[0]|_max={np.abs(within_chunk).max():.4f} "
                f"chunk_abs_max={np.abs(chunk).max():.4f}",
                flush=True,
            )

        self._chunk_done.clear()
        with self._lock:
            self._state.pending_chunk = chunk
            self._state.chunk_pos = 0
            self._state.phase = PHASE_EXECUTING
            step_idx = self._state.episode_step

        timeout_s = max(2.0, (self._chunk_horizon + 2) / TICK_HZ * 3.0)
        if not self._chunk_done.wait(timeout=timeout_s):
            raise RuntimeError(f"chunk execution did not complete within {timeout_s:.1f} s")

        with self._lock:
            obs_after = dict(self._state.last_obs) if self._state.last_obs is not None else None
            hw_fault = self._state.hw_fault
            self._state.episode_step += 1
            step_idx_after = self._state.episode_step
        if obs_after is None:
            raise RuntimeError("no formatted obs available after chunk execution")

        reward_payload = self._reward_hook(
            obs_after=obs_after,
            chunk=chunk,
            last_info={},
            step_idx=step_idx,
            max_steps=self._max_steps,
        ) or {}
        reward = float(reward_payload.get("reward", 0.0))
        success = bool(reward_payload.get("success", False))
        done = bool(reward_payload.get("done", success))
        truncated = bool(hw_fault) or step_idx_after >= self._max_steps
        info: Dict[str, Any] = dict(reward_payload.get("info", {}))

        response: Dict[str, Any] = dict(obs_after)
        response["reward"] = reward
        response["success"] = success
        response["done"] = done
        response["truncated"] = truncated
        response["info"] = info
        return response

    # ------------------------------------------------------------------ #
    # Default reward + operator prompt
    # ------------------------------------------------------------------ #

    def _default_reward_hook(
        self,
        *,
        obs_after: Dict[str, Any],
        chunk: np.ndarray,
        last_info: Dict[str, Any],
        step_idx: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        if step_idx < max_steps - 1:
            return {"reward": 0.0, "success": False, "done": False}
        answer = self._prompt_tty("[dsrl] episode complete — success? [y/N] ", default="n")
        success = answer.lower().startswith("y")
        return {
            "reward": 1.0 if success else 0.0,
            "success": success,
            "done": True,
        }

    def _prompt_tty(self, message: str, *, default: str) -> str:
        # Bypass rr-session's curses grab of sys.stdin/sys.stdout by talking
        # directly to /dev/tty. Falls back to the default if the controlling
        # terminal is unavailable (headless / nohup runs).
        try:
            with open("/dev/tty", "r+") as tty:
                tty.write(message)
                tty.flush()
                line = tty.readline()
                return line.strip() if line else default
        except OSError:
            print(f"[RobotBridgeAgent] /dev/tty unavailable; defaulting to {default!r}")
            return default


# ---------------------------------------------------------------------- #
# TCP server
# ---------------------------------------------------------------------- #


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        b = sock.recv(remaining)
        if not b:
            raise ConnectionError("socket closed while reading")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


def _read_message(sock: socket.socket) -> Dict[str, Any]:
    header = _recv_exactly(sock, 4)
    (length,) = struct.unpack(">I", header)
    body = _recv_exactly(sock, length)
    return pickle.loads(body)


def _send_message(sock: socket.socket, msg: Dict[str, Any]) -> None:
    body = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(body)) + body)


class _BridgeRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        agent: RobotBridgeAgent = self.server.agent  # type: ignore[attr-defined]
        client = self.client_address
        print(f"[bridge] {client} connected", flush=True)
        # Loop until the client closes — supports both pipelined (one
        # long-lived socket) and one-shot (new socket per RPC) trainers.
        while not agent._stop.is_set():
            try:
                msg = _read_message(self.request)
            except (ConnectionError, EOFError, OSError) as exc:
                print(f"[bridge] {client} handler exit awaiting next msg: {exc}", flush=True)
                return
            mtype = msg.get("type")
            print(f"[bridge] {client} got {mtype}", flush=True)
            with agent._op_lock:
                try:
                    if mtype == "RESET":
                        response = agent._handle_reset(msg)
                    elif mtype == "STEP":
                        response = agent._handle_step(msg)
                    elif mtype == "PING":
                        response = agent._handle_ping(msg)
                    else:
                        print(f"[bridge] {client} unknown type {mtype!r}; closing", flush=True)
                        return
                except Exception as exc:
                    # Fatal: close the socket. Trainer fills info["disconnected"]
                    # via its own socket-error path. Never send {"error": ...} —
                    # the trainer doesn't parse it and will crash on the next obs.
                    print(f"[bridge] {client} fatal during {mtype}: {exc}; closing", flush=True)
                    return
            # Bracket the actual send so we can pin partial-send vs clean-send
            # failures against the trainer-side _recv_msg logs.
            try:
                body = pickle.dumps(response)
            except Exception as exc:
                print(f"[bridge] {client} pickle failed for {mtype}: {exc}; closing", flush=True)
                return
            print(f"[bridge] {client} {mtype} resp built, {len(body)} bytes", flush=True)
            try:
                self.request.sendall(struct.pack(">I", len(body)) + body)
            except OSError as exc:
                print(f"[bridge] {client} send failed mid-{mtype}: {exc}; closing", flush=True)
                return
            print(f"[bridge] {client} {mtype} resp sent, looping", flush=True)


class _BridgeTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _make_tcp_server(addr: Tuple[str, int], agent: RobotBridgeAgent) -> _BridgeTCPServer:
    server = _BridgeTCPServer(addr, _BridgeRequestHandler)
    server.agent = agent  # type: ignore[attr-defined]
    return server
