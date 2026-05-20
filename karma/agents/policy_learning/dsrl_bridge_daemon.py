"""DSRL bridge daemon — robot-side counterpart for KarmaYAMRemoteEnv.

This is a **scaffold**, not a finished agent. It handles the trainer-facing
wire protocol (length-prefixed pickle over TCP) and exposes the four
hooks the daemon needs to provide:

    home_robot()                 — drive both arms back to a known pose
    prompt_operator_reset(task)  — block until the operator re-stages objects
    capture_observation()        — read joint state + 3 RGB cameras
    execute_chunk(chunk)         — push a (10, 14) action chunk through the bus

Each hook is a TODO that you fill in against your live Karma session. The
trainer in ``robometer-policy-learning`` does not care *how* you implement
them, only that they return the expected shapes (documented in each TODO).

Run alongside your Karma session::

    # on the robot box, in the activated Karma session shell:
    python -m karma.agents.policy_learning.dsrl_bridge_daemon \
        --host 0.0.0.0 --port 6112

Then point the trainer's ``remote_robot.host``/``remote_robot.port`` at this
box and the same port.

Wire protocol (matches ``robometer_policy_learning/envs/karma_yam_remote_env.py``):

    request  ::= pickle.dumps({"type": "RESET"|"STEP", ...})
                  framed by uint32-big-endian length prefix
    response ::= pickle.dumps({
                     "top_camera-images-rgb":   uint8 (3,H,W),
                     "left_camera-images-rgb":  uint8 (3,H,W),
                     "right_camera-images-rgb": uint8 (3,H,W),
                     "state":                   float32 (14,),
                     "prompt":                  str,
                     # STEP only:
                     "done":                    bool (optional, default False),
                     "truncated":               bool (optional, default False),
                     "info":                    dict (optional),
                 })

The trainer never reads ``done`` from this daemon for normal episodes —
episode termination is decided by the Robometer voting window upstream. Set
``done=True`` here only for unrecoverable hardware faults.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import socket
import socketserver
import struct
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


log = logging.getLogger("dsrl_bridge")


# -- TODO: Karma integration hooks -------------------------------------------
#
# Replace each `raise NotImplementedError` with the appropriate Karma call.
# A reasonable starting point: write a small client that connects to the
# already-running Karma session's pub/sub bus and snapshots state on demand.
# Look at /home/sra/yam/karma/karma/agents/policy_learning/async_molmoact2_agent.py
# for the established pattern of reading observations off the bus.


def home_robot() -> None:
    """Drive both YAM arms to the standard home pose and open both grippers.

    Should block until the motion completes. A good default home is whatever
    the existing Karma teleop reset routine uses.
    """
    # TODO: e.g. karma_session.call("robot.home", blocking=True)
    raise NotImplementedError("wire up home_robot() against your Karma session")


def prompt_operator_reset(task_prompt: str) -> None:
    """Print a re-staging prompt and block on operator <Enter>.

    For v1 (manual reset) we just stop the world until the human says go.
    """
    print(
        f"\n[dsrl_bridge] *** STAGE FOR: {task_prompt!r} ***\n"
        f"[dsrl_bridge] press <Enter> when the scene is reset...",
        flush=True,
    )
    try:
        input()
    except EOFError:
        # Daemon mode: no stdin. Fall back to a short pause; the trainer's
        # reset socket timeout is generous enough to absorb it.
        time.sleep(2.0)


def capture_observation() -> Dict[str, Any]:
    """Read one observation from the robot/cameras.

    Returns a dict with EXACTLY these keys (the trainer's env wrapper validates):
      * ``top_camera-images-rgb``    uint8 (3, H, W)
      * ``left_camera-images-rgb``   uint8 (3, H, W)
      * ``right_camera-images-rgb``  uint8 (3, H, W)
      * ``state``                    float32 (14,)
        — concat of [left arm joint pos (6), left gripper (1),
                     right arm joint pos (6), right gripper (1)]
      * ``prompt``                   str
    """
    # TODO: pull the latest frames from the Karma camera bus and the latest
    # joint state from the robot node. Match the same flattening + ordering
    # the MolmoAct2 client uses (see karma/agents/policy_learning/
    # async_molmoact2_agent.py:_recursive_flatten and MolmoAct2ModelIOConfig).
    raise NotImplementedError("wire up capture_observation() against your Karma session")


def execute_chunk(chunk: np.ndarray) -> Dict[str, Any]:
    """Execute a ``(T, 14)`` action chunk and return ``info`` once it finishes.

    The chunk is exactly what MolmoAct2 produces today and what
    ``async_molmoact2_agent.py`` pushes through the bus — slice it into
    per-tick joint setpoints at 10 Hz (or the rate Karma is configured for).

    Returns an info dict — may be empty. The trainer never reads its env
    reward (Robometer relabels asynchronously); ``done`` defaults to False
    unless a hardware fault is detected.
    """
    # TODO: e.g. for i in range(chunk.shape[0]):
    #            karma_session.send_action(chunk[i])
    #            karma_session.wait_one_tick()
    raise NotImplementedError("wire up execute_chunk() against your Karma session")


# -- wire helpers (mirror env wrapper exactly) -------------------------------


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock: socket.socket) -> Any:
    header = _recvall(sock, 4)
    if not header:
        return None
    n = struct.unpack(">I", header)[0]
    body = _recvall(sock, n)
    if body is None:
        return None
    return pickle.loads(body)


def _send_msg(sock: socket.socket, msg: Any) -> None:
    payload = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _validate_obs(obs: Dict[str, Any]) -> None:
    for key in (
        "top_camera-images-rgb",
        "left_camera-images-rgb",
        "right_camera-images-rgb",
    ):
        arr = obs.get(key)
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError(f"obs[{key!r}] must be a 3D ndarray, got {type(arr).__name__}")
    state = obs.get("state")
    if not isinstance(state, np.ndarray) or state.shape != (14,):
        raise ValueError(f"obs['state'] must be float32 (14,), got {getattr(state, 'shape', None)}")
    if "prompt" not in obs:
        raise ValueError("obs['prompt'] is required (the language instruction)")


# -- TCP server --------------------------------------------------------------


class DSRLBridgeHandler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        peer = self.client_address
        log.info("trainer connected from %s", peer)

    def finish(self) -> None:
        log.info("trainer disconnected from %s", self.client_address)

    def handle(self) -> None:
        sock: socket.socket = self.request
        sock.settimeout(None)
        while True:
            try:
                msg = _recv_msg(sock)
            except (ConnectionError, OSError) as exc:
                log.warning("recv failed: %s", exc)
                return
            if msg is None:
                return
            try:
                if msg.get("type") == "RESET":
                    task = msg.get("task") or ""
                    log.info("RESET (task=%r)", task)
                    home_robot()
                    prompt_operator_reset(task or "<no task>")
                    obs = capture_observation()
                    _validate_obs(obs)
                    obs.setdefault("prompt", task)
                    _send_msg(sock, obs)
                elif msg.get("type") == "STEP":
                    chunk = np.asarray(msg.get("chunk"), dtype=np.float32)
                    if chunk.ndim != 2 or chunk.shape[1] != 14:
                        raise ValueError(
                            f"STEP chunk must be (T, 14), got {chunk.shape}"
                        )
                    info = execute_chunk(chunk) or {}
                    obs = capture_observation()
                    _validate_obs(obs)
                    response: Dict[str, Any] = dict(obs)
                    response["info"] = info
                    response.setdefault("done", False)
                    response.setdefault("truncated", False)
                    _send_msg(sock, response)
                else:
                    _send_msg(sock, {"error": f"unknown message type: {msg.get('type')!r}"})
            except NotImplementedError as exc:
                log.error("hook not wired: %s", exc)
                _send_msg(sock, {"error": str(exc)})
                # Don't kill the connection — let the user fix and reconnect.
            except Exception as exc:  # noqa: BLE001
                log.exception("handler error")
                _send_msg(sock, {"error": str(exc)})


class _ReusingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str, port: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log.info("DSRL bridge listening on %s:%d", host, port)
    log.info(
        "(scaffold mode — the four hooks at the top of this file need to be "
        "wired into your Karma session before training will work)"
    )
    with _ReusingTCPServer((host, port), DSRLBridgeHandler) as srv:
        srv.serve_forever()


def main(argv: Optional[Tuple[str, ...]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0", help="bind address")
    p.add_argument("--port", type=int, default=6112, help="bind port")
    args = p.parse_args(argv)
    try:
        serve(args.host, args.port)
    except KeyboardInterrupt:
        log.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
