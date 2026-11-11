"""MolmoAct2 YAM/SO101 adapter using checkpoint-owned predict_action transforms."""

import argparse
import json
from pathlib import Path

import numpy as np
from act_server import ActService, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="local Hugging Face snapshot directory")
    parser.add_argument("--rig", choices=["yam_bimanual", "so101"], default="yam_bimanual")
    parser.add_argument("--norm-tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8202)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--dataset-gripper-open",
        type=float,
        choices=[0, 1],
        default=0,
        help="YAM checkpoint training convention; Karma wire is 1=open",
    )
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint = str(Path(args.checkpoint).resolve())
    stats = json.loads((Path(checkpoint) / "norm_stats.json").read_text())
    if args.norm_tag not in stats:
        raise ValueError(f"norm tag {args.norm_tag!r} not in checkpoint statistics")
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, extra_special_tokens={}
    )
    model = (
        AutoModelForImageTextToText.from_pretrained(
            checkpoint, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
        )
        .to(args.device)
        .eval()
    )

    # Match floating processor inputs to model weights without rewriting cached source files.
    def move(inputs, device):
        return {
            k: v.to(device=device, dtype=model.dtype if v.is_floating_point() else v.dtype)
            if torch.is_tensor(v)
            else v
            for k, v in inputs.items()
        }

    model._move_inputs_to_device = move
    yam = args.rig == "yam_bimanual"
    roles = ["top", "left_wrist", "right_wrist"] if yam else ["top", "side"]
    dimension = 14 if yam else 6

    def predict(images, state, instruction, payload):
        state = state.copy()
        if yam and args.dataset_gripper_open == 0:
            state[[6, 13]] = 1 - state[[6, 13]]
        with torch.inference_mode():
            result = model.predict_action(
                processor=processor,
                images=[Image.fromarray(images[r]) for r in roles],
                task=instruction,
                state=state,
                norm_tag=args.norm_tag,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=payload.get("num_steps", args.num_steps),
                normalize_language=True,
                enable_cuda_graph=payload.get("enable_cuda_graph", args.cuda_graph),
            )
        actions = result.actions
        if torch.is_tensor(actions):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if yam and args.dataset_gripper_open == 0:
            actions = actions.copy()
            actions[:, [6, 13]] = 1 - actions[:, [6, 13]]
        return actions

    metadata = dict(
        provider="molmoact2",
        checkpoint=checkpoint,
        norm_tag=args.norm_tag,
        state_dim=dimension,
        num_cameras=len(roles),
        camera_roles=roles,
        gripper_open=1.0 if yam else 0.0,
        joint_units="radians" if yam else "policy_frame",
        action_hz=30.0,
        action_horizon=getattr(model.config, "max_action_horizon", 30),
        dataset_gripper_open=args.dataset_gripper_open if yam else None,
    )
    service = ActService(predict, metadata)
    from act_server import IMAGE_KEYS

    service.act(
        dict(
            state=np.zeros(dimension),
            instruction="warmup",
            normalization_tag=args.norm_tag,
            num_steps=args.num_steps,
            enable_cuda_graph=args.cuda_graph,
            **{IMAGE_KEYS[r]: np.zeros((360, 640, 3), dtype=np.uint8) for r in roles},
        )
    )
    serve(service, args.host, args.port)


if __name__ == "__main__":
    main()
