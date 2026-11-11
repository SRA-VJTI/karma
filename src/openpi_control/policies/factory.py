"""Build clients without importing GPU model dependencies."""

from __future__ import annotations

import dataclasses

from ..exceptions import ConfigurationError
from ..inference import InferenceError, MolmoActClient


class HTTPPolicy(MolmoActClient):
    managed = True

    def __init__(self, config, contract):
        super().__init__(
            config.server,
            contract=contract,
            timeout_s=config.timeout_s,
            num_steps=config.num_steps,
            jpeg_quality=config.jpeg_quality,
            enable_cuda_graph=config.enable_cuda_graph,
        )
        self.config = config
        self.display_name = config.provider
        self.action_hz = config.action_hz

    def health(self):
        metadata = super().health()
        if self.config.provider == "pi05" and metadata.get("provider") != "pi05":
            raise InferenceError("pi05 requires the Karma pi05 server adapter with health metadata")
        if metadata.get("contract_version") is not None:
            expected = {
                "contract_version": 1,
                "provider": self.config.provider,
                "action_hz": self.action_hz,
                "action_kind": "absolute",
                "joint_units": "policy_frame" if self.contract.frame else "radians",
                "camera_roles": list(self.contract.cameras),
                "gripper_open": self.contract.gripper_open,
            }
            if self.config.provider == "pi05":
                expected["num_steps"] = self.config.num_steps
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise InferenceError(f"server {key}={metadata.get(key)!r}; expected {value!r}")
            if self.config.checkpoint and metadata.get("checkpoint") != self.config.checkpoint:
                raise InferenceError("server checkpoint differs from configured checkpoint")
        self.metadata = metadata
        return metadata

    def reset(self):
        pass


def create_client(config, contract):
    if config.camera_roles and tuple(config.camera_roles) != contract.cameras:
        raise ConfigurationError(
            f"profile requires camera roles {config.camera_roles}; rig provides {contract.cameras}"
        )
    if config.norm_tag:
        contract = dataclasses.replace(contract, norm_tag=config.norm_tag)
    if config.provider == "dream":
        if (
            contract.arms != ("left", "right")
            or contract.arm_dof != 6
            or contract.frame is not None
        ):
            raise ConfigurationError(
                "Dream currently supports bimanual YAM in its native joint frame only"
            )
        if contract.cameras != ("top", "left_wrist", "right_wrist"):
            raise ConfigurationError("Dream requires top, left_wrist and right_wrist cameras")
        from .dream import DreamPolicy

        return DreamPolicy(config, contract)
    return HTTPPolicy(config, contract)
