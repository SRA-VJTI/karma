"""Train LeRobot 0.6.1 ACT on Karma's bimanual kinesthetic recordings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .flash_policy import CAMERAS, FlashConfig, prepare_image


def make_policy(config, device, *, tiny=False):
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act import ACTConfig, ACTPolicy

    features = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    features.update(
        {
            f"observation.images.{c}": PolicyFeature(FeatureType.VISUAL, (3, *config.image_size))
            for c in CAMERAS
        }
    )
    kwargs = (
        dict(
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
        )
        if tiny
        else {}
    )
    return ACTPolicy(
        ACTConfig(
            input_features=features,
            output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
            device=device,
            chunk_size=config.chunk_size,
            n_action_steps=config.execute_steps,
            pretrained_backbone_weights=None if tiny else "ResNet18_Weights.IMAGENET1K_V1",
            normalization_mapping={
                key: NormalizationMode.IDENTITY for key in ("STATE", "ACTION", "VISUAL")
            },
            **kwargs,
        )
    ).to(device)


def load_stats(stats):
    result = {}
    for name in ("observation.state", "action"):
        for kind in ("mean", "std"):
            value = np.asarray(stats[name][kind], dtype=np.float32).reshape(-1)
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"invalid dataset stats {name}.{kind}")
            result[f"{name}.{kind}"] = np.maximum(value, 1e-6) if kind == "std" else value
    return result


def prepare_sample(sample, config, stats):
    import torch

    result = {}
    for name in ("observation.state", "action"):
        value = torch.as_tensor(sample[name], dtype=torch.float32)
        if value.shape[-1] != 14 or not torch.isfinite(value).all():
            raise ValueError(f"invalid {name}")
        result[name] = (value - torch.from_numpy(stats[f"{name}.mean"])) / torch.from_numpy(
            stats[f"{name}.std"]
        )
    result["action_is_pad"] = sample["action_is_pad"]
    dt = float(sample["action.dt_s"].reshape(-1)[0])
    if not 0.5 / config.fps <= dt <= 1.5 / config.fps:
        raise ValueError("irregular teaching frame interval; inspect/re-record this episode")
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        # LeRobot decodes RGB float CHW [0,1]. Match serving's uint8 boundary.
        pixels = sample[key].detach().cpu().numpy()
        pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        result[key] = torch.from_numpy(prepare_image(pixels, camera, config))
    return result


def train(args):
    import torch
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    config = FlashConfig(**json.loads(args.config.read_text()))
    teaching = json.loads((args.root / "meta" / "teaching.json").read_text())
    expected = {
        "schema": "karma.yam_teaching.v1",
        "action_alignment": "next_measured_state",
        "gripper": "0=open, 1=closed",
        "arms": ["left", "right"],
        "fps": config.fps,
    }
    if any(teaching.get(k) != v for k, v in expected.items()):
        raise ValueError("dataset is not the matching bimanual teaching contract")
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    if meta.fps != config.fps:
        raise ValueError("checkpoint fps must match dataset fps")
    stats = load_stats(meta.stats)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps={"action": [i / config.fps for i in range(config.chunk_size)]},
    )
    if len(dataset) < 2 or args.steps < 1 or args.batch_size < 1:
        raise ValueError("need a nonempty dataset, positive steps and batch size")
    # ACT has no language conditioning. Refuse silently mixing unrelated tasks.
    if set(meta.tasks.index) != {config.task}:
        raise ValueError("dataset tasks must contain only the configured ACT task")
    args.output.mkdir(parents=True, exist_ok=False)
    policy = make_policy(config, args.device)
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    policy.train()

    class PreparedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            return prepare_sample(dataset[index], config, stats)

    loader = torch.utils.data.DataLoader(
        PreparedDataset(), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite ACT training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"step {step}/{args.steps}: loss={loss.item():.5f}", flush=True)
            if step >= args.steps:
                break
    policy.save_pretrained(args.output)
    np.savez(args.output / "normalization.npz", **stats)
    (args.output / "flash.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (args.output / "training.json").write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "steps": step,
                "frames": len(dataset),
                "lerobot": "0.6.1",
                "action_alignment": "next_measured_state",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
