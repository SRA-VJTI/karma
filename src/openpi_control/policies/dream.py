"""Dream SDK bridge with independent observation and reply channels."""

from __future__ import annotations

import base64
import json
import os
import queue
import select
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..inference import InferenceError, encode_frame
from .trace import PolicyTrace


class DreamPolicy:
    managed = True
    display_name = "dream"

    def __init__(self, config, contract):
        self.config, self.contract = config, contract
        self.action_hz = config.action_hz
        self.url = "dream:" + config.checkpoint
        self.jpeg_quality = config.jpeg_quality or 95
        self.last_latency = {}
        self.last_wire_frames = {}
        self.trace = PolicyTrace(config)
        self.process = None
        self.generation = 0
        self.refill_table = []
        self.mode = "async"
        self._write_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._replies = queue.Queue()
        self._ticks = queue.Queue(maxsize=2)
        self._closed = threading.Event()
        self._serial = 0
        self._prompt = ""
        self._last_timestamp = 0.0

    def _start(self):
        if self.process is not None:
            return
        root = Path(__file__).resolve().parents[3]
        python = self.config.worker_python or str(root / "runtime/dream/.venv/bin/python")
        if not Path(python).is_file():
            raise InferenceError(
                "Dream runtime missing: uv sync --locked --project runtime/dream; "
                "or set worker_python"
            )
        self._log = (self.trace.path / "dream-worker.log").open("a")
        self.process = subprocess.Popen(
            [python, "-P", str(Path(__file__).with_name("dream_worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
            env={**os.environ, "DREAM_LOG_DIR": str(self.trace.path / "sdk")},
        )
        os.set_blocking(self.process.stdin.fileno(), False)

        def read():
            try:
                for line in self.process.stdout:
                    self._replies.put(json.loads(line))
            except Exception:
                pass
            self._replies.put({"error": "worker exited"})

        threading.Thread(target=read, daemon=True).start()
        settings = dict(
            checkpoint=self.config.checkpoint,
            strict=True,
            action_hz=self.action_hz,
            execution_horizon=self.config.execution_horizon,
            timeout_ms=round(self.config.timeout_s * 1000),
            obs_window=self.config.obs_window,
            obs_fps=self.config.obs_fps,
            embodiment={
                "state": {
                    "right_arm": [0, 6],
                    "right_gripper": [6, 7],
                    "left_arm": [7, 13],
                    "left_gripper": [13, 14],
                },
                "action": {
                    "right_arm": [0, 6],
                    "right_gripper": [6, 7],
                    "left_arm": [7, 13],
                    "left_gripper": [13, 14],
                },
                "video": {r: "observation.images." + r for r in ("overhead", "wrist_l", "wrist_r")},
            },
        )
        if self.config.ik:
            settings["ik"] = {"arm": "yam", "head": "joint"}
        self._settings = settings
        self._call("policy", config=settings)
        threading.Thread(target=self._send_ticks, daemon=True).start()

    def _write(self, body):
        with self._write_lock:
            if self._closed.is_set() or self.process.poll() is not None:
                raise InferenceError("Dream worker is closed; inspect dream-worker.log")
            if body.get("op") == "infer":
                body["tick"]["timestamp"] = max(body["tick"]["timestamp"], self._last_timestamp)
                self._last_timestamp = body["tick"]["timestamp"]
            self._write_bytes(json.dumps(body, allow_nan=False) + "\n")

    def _write_bytes(self, text):
        pending = memoryview(text.encode())
        deadline = time.monotonic() + self.config.timeout_s + 5
        fd = self.process.stdin.fileno()
        while pending:
            if self._closed.is_set() or time.monotonic() >= deadline:
                raise InferenceError("Dream worker input stalled")
            if not select.select([], [fd], [], 0.1)[1]:
                continue
            try:
                written = os.write(fd, pending)
                pending = pending[written:]
            except BlockingIOError:
                continue

    def _call(self, op, **args):
        with self._rpc_lock:
            self._serial += 1
            serial = self._serial
            self._write(dict(id=serial, op=op, **args))
            deadline = time.monotonic() + self.config.timeout_s + 5
            while True:
                try:
                    result = self._replies.get(timeout=max(0.001, deadline - time.monotonic()))
                except queue.Empty as exc:
                    self.process.kill()
                    raise InferenceError(
                        "Dream worker deadline exceeded; inspect dream-worker.log"
                    ) from exc
                if result.get("error"):
                    raise InferenceError("Dream: " + result["error"] + "; inspect dream-worker.log")
                if result.get("id") == serial:
                    return result["result"]
                if time.monotonic() >= deadline:
                    raise InferenceError("Dream worker reply deadline exceeded")

    def health(self):
        self._start()
        return self._call("health", checkpoint=self.config.checkpoint)

    def reset(self):
        self.generation += 1
        self._settings["action_hz"] = self.action_hz
        self._call("reset", generation=self.generation, config=self._settings)
        self.refill_table = []

    def _tick(self, observation, instruction, timestamp):
        from PIL import Image

        frames = {}
        for _name, role, image in zip(
            self.contract.cameras,
            ("overhead", "wrist_l", "wrist_r"),
            (observation.top_cam, observation.left_cam, observation.right_cam),
            strict=True,
        ):
            rgb = np.asarray(Image.fromarray(image).resize((320, 176)))
            encoded = encode_frame(rgb, jpeg_quality=self.jpeg_quality)
            frames["video." + role] = base64.b64encode(encoded.tobytes()).decode()
        state = observation.state
        return dict(
            frames=frames,
            state={
                "state.left_arm": state[:6].tolist(),
                "state.left_gripper": state[6:7].tolist(),
                "state.right_arm": state[7:13].tolist(),
                "state.right_gripper": state[13:14].tolist(),
                "prompt": instruction,
            },
            timestamp=timestamp,
        )

    def feed(self, observation, instruction, commanded=None):
        item = (self.generation, observation, instruction, time.monotonic())
        try:
            self._ticks.put_nowait(item)
        except queue.Full:
            try:
                self._ticks.get_nowait()
            except queue.Empty:
                pass
            self.trace.event("observation_dropped")
            try:
                self._ticks.put_nowait(item)
            except queue.Full:
                pass

    def _send_ticks(self):
        while not self._closed.is_set():
            try:
                generation, obs, prompt, stamp = self._ticks.get(timeout=0.1)
                tick = self._tick(obs, prompt, stamp)
                with self._write_lock:
                    if stamp <= self._last_timestamp or generation != self.generation:
                        continue
                    self._last_timestamp = stamp
                    # _write is serialized with infer so timestamps never travel backwards.
                    self._write_bytes(
                        json.dumps(dict(id=0, op="tick", generation=generation, tick=tick)) + "\n"
                    )
            except queue.Empty:
                continue
            except Exception:
                return

    def infer(self, observation, instruction, *, commanded=None):
        tick = self._tick(observation, instruction, time.monotonic())
        self.last_wire_frames = {
            name: np.frombuffer(base64.b64decode(tick["frames"]["video." + role]), dtype=np.uint8)
            for name, role in zip(
                self.contract.cameras, ("overhead", "wrist_l", "wrist_r"), strict=True
            )
        }
        # Freeze tick sending until the infer message is written, not until inference completes.
        with self._write_lock:
            tick["timestamp"] = max(tick["timestamp"], self._last_timestamp)
            self._last_timestamp = tick["timestamp"]
        started = time.monotonic()
        result = self._call(
            "infer", generation=self.generation, tick=tick, commanded=commanded or {}
        )
        arrays = [np.asarray(result[n], dtype=float) for n in ("left", "right")]
        if (
            any(
                a.ndim != 2 or a.shape[1] != 7 or not 1 <= len(a) <= 256 or not np.isfinite(a).all()
                for a in arrays
            )
            or arrays[0].shape != arrays[1].shape
        ):
            raise InferenceError("Dream returned invalid YAM action chunks")
        for a in arrays:
            a[:, 6] = np.clip(a[:, 6], 0, 1)
        self.refill_table = result["refill"]
        self.mode = result["mode"]
        timing = result.get("timing") or {}
        self.last_timing = timing
        self.last_latency = {
            "round_trip_s": time.monotonic() - started,
            "gpu_s": (timing.get("serve_infer_ms") or 0) / 1000,
        }
        return np.concatenate(arrays, axis=1)

    def refill_due(self, executed, remaining):
        if self.config.pacing == "sync" or not self.refill_table:
            return remaining == 0
        n = len(self.refill_table) - 1
        return self.refill_table[min(executed, n)][min(remaining, n)]

    def close(self):
        self._closed.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            for stream in (self.process.stdin, self.process.stdout, self._log):
                stream.close()
        self.trace.close()
