"""Serve a Karma Flash ACT checkpoint on the existing /act wire protocol."""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .flash_policy import CAMERAS, TAG, FlashConfig, flip_grippers, prepare_image
from .inference import MOLMOACT_IMAGE_KEYS


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        import torch
        from lerobot.policies.act import ACTPolicy

        checkpoint = Path(checkpoint)
        self.config = FlashConfig(**json.loads((checkpoint / "flash.json").read_text()))
        self.device = device
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
        self.policy.reset()
        with np.load(checkpoint / "normalization.npz", allow_pickle=False) as archive:
            self.stats = {
                key: torch.as_tensor(archive[key], device=device) for key in archive.files
            }
        expected = {
            f"{name}.{kind}" for name in ("observation.state", "action") for kind in ("mean", "std")
        }
        if (
            set(self.stats) != expected
            or any(v.shape != (14,) or not torch.isfinite(v).all() for v in self.stats.values())
            or any(
                (self.stats[f"{name}.std"] <= 0).any() for name in ("observation.state", "action")
            )
        ):
            raise ValueError("invalid checkpoint normalization statistics")
        cfg = self.policy.config
        if (
            cfg.chunk_size != self.config.chunk_size
            or cfg.input_features["observation.state"].shape != (14,)
            or cfg.output_features["action"].shape != (14,)
            or set(cfg.image_features) != {f"observation.images.{c}" for c in CAMERAS}
            or any(
                feature.shape != (3, *self.config.image_size)
                for feature in cfg.image_features.values()
            )
        ):
            raise ValueError("ACT checkpoint and Flash manifest disagree")
        self.lock = threading.Lock()

    def predict(self, payload):
        import torch

        if payload.get("normalization_tag") != TAG:
            raise ValueError("wrong normalization tag")
        if payload.get("instruction") != self.config.task:
            raise ValueError("ACT instruction must exactly match the trained task")
        state = np.asarray(payload["state"])
        if state.shape != (14,):
            raise ValueError("state must have shape (14,)")
        state = torch.as_tensor(flip_grippers(state), device=self.device)
        batch = {
            "observation.state": (
                (state - self.stats["observation.state.mean"]) / self.stats["observation.state.std"]
            )[None]
        }
        for camera in CAMERAS:
            image = prepare_image(payload[MOLMOACT_IMAGE_KEYS[camera]], camera, self.config)
            batch[f"observation.images.{camera}"] = torch.from_numpy(image)[None].to(self.device)
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)[0]
            actions = actions * self.stats["action.std"] + self.stats["action.mean"]
            actions = actions.cpu().numpy()
        if actions.shape != (self.config.chunk_size, 14) or not np.isfinite(actions).all():
            raise RuntimeError("ACT returned invalid actions")
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0, 1)
        return {"actions": flip_grippers(actions), "dt_ms": (time.perf_counter() - started) * 1000}


def create_app(predictor):
    import json_numpy
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI(title="Karma Flash YAM ACT")

    @app.get("/act")
    def health():
        return {
            "status": "ok",
            "norm_tag": TAG,
            "state_dim": 14,
            "num_cameras": 3,
            "cameras": list(CAMERAS),
            "gripper_open": 1,
            "flash_config": asdict(predictor.config),
        }

    # Request is imported here to keep FastAPI optional for pure contract tests.
    # Explicit annotation avoids deferred lookup in module globals.
    async def act(request):
        try:
            payload = json_numpy.loads(await request.body())
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            # The server accepts one robot session at a time; serialize GPU use.
            result = predictor.predict(payload)
            return Response(json_numpy.dumps(result), media_type="application/json")
        except (KeyError, TypeError, ValueError) as err:
            return Response(
                json.dumps({"error": str(err)}), status_code=400, media_type="application/json"
            )

    act.__annotations__["request"] = Request
    app.post("/act")(act)
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8202)
    args = parser.parse_args()
    uvicorn.run(create_app(Predictor(args.checkpoint, args.device)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
