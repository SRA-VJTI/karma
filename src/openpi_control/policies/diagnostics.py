"""Policy-only preflight and saved-observation probes. Never open robot interfaces."""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from ..exceptions import ConfigurationError
from ..inference import BimanualObservation, InferenceError, PolicyContract
from ..rigs import resolve_rig
from .config import PolicyConfig
from .factory import create_client


def main(argv=None):
    parser = argparse.ArgumentParser(prog="karma policy")
    parser.add_argument("operation", choices=["doctor", "probe"])
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--rig", default="yam_bimanual")
    parser.add_argument(
        "--observation",
        type=Path,
        help="NPZ with state, top_cam, left_cam, right_cam (or side_cam)",
    )
    parser.add_argument("--instruction", default="")
    args = parser.parse_args(argv)
    client = None
    try:
        config = PolicyConfig.load(args.policy_config)
        rig = resolve_rig(args.rig)
        rig = dataclasses.replace(
            rig, policy_norm_tag=config.norm_tag, policy_frame=config.policy_frame
        )
        contract = PolicyContract.from_rig(rig)
        if config.camera_roles:
            contract = dataclasses.replace(contract, cameras=config.camera_roles)
        client = create_client(config, contract)
        print(json.dumps({"config": config.as_dict(), "health": client.health()}, indent=2))
        if args.operation == "probe":
            if not args.observation or not args.instruction.strip():
                raise ConfigurationError("probe requires --observation and --instruction")
            with np.load(args.observation, allow_pickle=False) as data:
                obs = BimanualObservation(
                    state=data["state"],
                    top_cam=data["top_cam"],
                    left_cam=data.get("left_cam"),
                    right_cam=data.get("right_cam"),
                    side_cam=data.get("side_cam"),
                    contract=client.contract,
                )
            client.reset()
            actions = client.infer(obs, args.instruction)
            print(
                json.dumps(
                    {
                        "shape": list(actions.shape),
                        "timing": client.last_latency,
                        "actions": actions.tolist(),
                    }
                )
            )
        return 0
    except (ConfigurationError, InferenceError, OSError, ValueError, KeyError) as exc:
        print(f"policy check failed: {exc}")
        return 2
    finally:
        if client:
            client.close()
