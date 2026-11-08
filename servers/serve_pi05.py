"""π0.5 YAM adapter. Transforms adapted from inspect-robots' pi05 example.

Run in OpenPI at 15a9616. No OpenPI or GPU dependency is installed into Karma.
"""

import argparse
import dataclasses

import einops
import numpy as np
from act_server import ActService, serve
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.training import config as _config


def _parse_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": _parse_image(data["images"]["top"]),
                "left_wrist_0_rgb": _parse_image(data["images"]["left"]),
                "right_wrist_0_rgb": _parse_image(data["images"]["right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Model action dim is 32; the robot uses the first 14.
        return {"actions": np.asarray(data["actions"][:, :14])}


@dataclasses.dataclass(frozen=True)
class YamInferenceConfig(_config.DataConfigFactory):
    def create(self, assets_dirs, model_config) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=_transforms.Group(
                inputs=[YamInputs(model_type=model_config.model_type)],
                outputs=[YamOutputs()],
            ),
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def load_policy(ckpt_dir: str, num_steps: int):
    train_config = _config.TrainConfig(
        name="yam_pi05",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16),
        # asset_id must match the norm-stats dir shipped in the checkpoint:
        # <ckpt>/assets/yam-bimanual-merged/norm_stats.json
        data=YamInferenceConfig(assets=_config.AssetsConfig(asset_id="yam-bimanual-merged")),
    )
    return policy_config.create_trained_policy(
        train_config, ckpt_dir, sample_kwargs={"num_steps": num_steps}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8204)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    policy = load_policy(args.ckpt, args.num_steps)

    def predict(images, state, instruction, payload):
        if payload.get("num_steps", args.num_steps) != args.num_steps:
            raise ValueError("pi05 num_steps is fixed at startup; match the client profile")
        return policy.infer(
            {
                "images": {
                    "top": images["top"],
                    "left": images["left_wrist"],
                    "right": images["right_wrist"],
                },
                "state": state,
                "prompt": instruction,
            }
        )["actions"]

    metadata = dict(
        provider="pi05",
        checkpoint=args.ckpt,
        norm_tag="yam-bimanual-merged",
        state_dim=14,
        num_cameras=3,
        camera_roles=["top", "left_wrist", "right_wrist"],
        gripper_open=1.0,
        joint_units="radians",
        action_hz=30.0,
        action_horizon=16,
        num_steps=args.num_steps,
    )
    service = ActService(predict, metadata)
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    service.act(
        dict(
            state=np.zeros(14),
            instruction="warmup",
            normalization_tag=metadata["norm_tag"],
            top_cam=dummy,
            left_cam=dummy,
            right_cam=dummy,
            num_steps=args.num_steps,
        )
    )
    serve(service, args.host, args.port)


if __name__ == "__main__":
    main()
