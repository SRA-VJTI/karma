"""Hardware-free, serialized /act server shared by the GPU adapters."""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import json_numpy
import numpy as np
from PIL import Image

IMAGE_KEYS = {
    "top": "top_cam",
    "left_wrist": "left_cam",
    "right_wrist": "right_cam",
    "side": "side_cam",
}


def decode_image(value):
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == np.uint8:
        with Image.open(io.BytesIO(array.tobytes())) as image:
            return np.asarray(image.convert("RGB"))
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be uint8 RGB HxWx3 or JPEG bytes")
    return array


class ActService:
    def __init__(self, predict, metadata):
        self.predict = predict
        self.metadata = dict(
            metadata,
            status="ok",
            contract_version=1,
            action_kind="absolute",
            image_encodings=["jpeg", "rgb8"],
        )
        self.lock = threading.Lock()

    def act(self, payload):
        if payload.get("normalization_tag") != self.metadata["norm_tag"]:
            raise ValueError("normalization_tag does not match loaded checkpoint")
        state = np.asarray(payload["state"], dtype=np.float32)
        if state.shape != (self.metadata["state_dim"],) or not np.isfinite(state).all():
            raise ValueError("invalid state dimensions or nonfinite state")
        instruction = payload["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        images = {
            role: decode_image(payload[IMAGE_KEYS[role]]) for role in self.metadata["camera_roles"]
        }
        steps = payload.get("num_steps", 10)
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError("num_steps must be an integer between 1 and 1000")
        with self.lock:
            start = time.perf_counter()
            actions = np.asarray(
                self.predict(images, state, instruction, payload), dtype=np.float32
            )
            elapsed = (time.perf_counter() - start) * 1000
        if (
            actions.ndim != 2
            or actions.shape[1] != len(state)
            or not 1 <= len(actions) <= 256
            or not np.isfinite(actions).all()
        ):
            raise RuntimeError("model returned invalid actions")
        return {
            "actions": actions,
            "inference_ms": elapsed,
            "dt_ms": elapsed,
            "action_hz": self.metadata["action_hz"],
        }


def serve(service, host, port):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data):
            body = json_numpy.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(200, service.metadata) if self.path.rstrip("/") in {
                "/act",
                "/healthz",
            } else self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/act":
                return self.send(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 32 * 1024 * 1024:
                    raise ValueError("body must be between 1 byte and 32 MiB")
                payload = json_numpy.loads(self.rfile.read(size).decode())
                result = service.act(payload)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self.send(400, {"error": str(exc)})
            except Exception:
                logging.exception("model inference failed")
                return self.send(500, {"error": "model inference failed; inspect server log"})
            self.send(200, result)

    print(json.dumps(service.metadata, indent=2), flush=True)
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
