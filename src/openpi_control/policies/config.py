"""Serializable, strict policy profiles shared by commands and diagnostics."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from ..exceptions import ConfigurationError


@dataclasses.dataclass(frozen=True)
class PolicyConfig:
    provider: str = "molmoact2"
    server: str | None = None
    checkpoint: str | None = None
    norm_tag: str | None = None
    policy_frame: str | None = None
    action_hz: float = 30.0
    timeout_s: float = 60.0
    max_prediction_age_s: float = 10.0
    num_steps: int = 10
    jpeg_quality: int = 95
    enable_cuda_graph: bool = True
    execution_horizon: int = 30
    obs_window: int = 9
    obs_fps: float = 5.0
    ik: bool = True
    pacing: str = "adaptive"
    worker_python: str | None = None
    capture_payloads: bool = False
    camera_roles: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.provider not in {"molmoact2", "pi05", "dream"}:
            raise ConfigurationError(f"unknown policy provider: {self.provider}")
        for key in ("action_hz", "timeout_s", "obs_fps", "max_prediction_age_s"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigurationError(f"{key} must be finite and positive")
        for key in ("num_steps", "execution_horizon", "obs_window"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"{key} must be a positive integer")
        if type(self.jpeg_quality) is not int or not 0 <= self.jpeg_quality <= 100:
            raise ConfigurationError("jpeg_quality must be an integer between 0 and 100")
        if self.pacing not in {"adaptive", "sync"}:
            raise ConfigurationError("pacing must be adaptive or sync")
        for key in ("ik", "enable_cuda_graph", "capture_payloads"):
            if type(getattr(self, key)) is not bool:
                raise ConfigurationError(f"{key} must be a boolean")
        for key in ("server", "checkpoint", "norm_tag", "policy_frame", "worker_python"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigurationError(f"{key} must be a nonempty string")
        if self.server:
            url = urlsplit(self.server if "://" in self.server else "http://" + self.server)
            if url.username or url.password or url.query or url.fragment:
                raise ConfigurationError(
                    "server URL must not contain credentials, query or fragment"
                )
        if self.camera_roles is not None:
            roles = self.camera_roles
            if (
                not isinstance(roles, (list, tuple))
                or not roles
                or any(not isinstance(role, str) for role in roles)
                or len(set(roles)) != len(roles)
                or not set(roles) <= {"top", "left_wrist", "right_wrist", "side"}
            ):
                raise ConfigurationError("camera_roles must contain unique supported camera roles")
            object.__setattr__(self, "camera_roles", tuple(roles))
        if self.provider == "dream":
            if not self.checkpoint or not self.checkpoint.strip():
                raise ConfigurationError("Dream requires an explicit checkpoint in --policy-config")
            if self.server or self.norm_tag or self.policy_frame:
                raise ConfigurationError(
                    "Dream uses SDK endpoints and its YAM frame, not server/norm_tag/policy_frame"
                )

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        values = {}
        if path:
            try:
                values = json.loads(Path(path).read_text())
            except (OSError, ValueError) as exc:
                raise ConfigurationError(f"cannot read policy profile {path}: {exc}") from exc
            if not isinstance(values, dict):
                raise ConfigurationError("policy profile must be a JSON object")
        unknown = values.keys() - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ConfigurationError(f"unknown policy fields: {sorted(unknown)}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        if values.get("provider") == "dream":
            values.setdefault("action_hz", 20.0)
            values.setdefault("timeout_s", 15.0)
        if values.get("provider") == "pi05":
            values.setdefault("server", "http://localhost:8204")
            values.setdefault("norm_tag", "yam-bimanual-merged")
            values.setdefault("enable_cuda_graph", False)
        for key in ("policy_frame", "worker_python"):
            if path and values.get(key) and not Path(values[key]).is_absolute():
                values[key] = str((Path(path).resolve().parent / values[key]).resolve())
        try:
            return cls(**values)
        except (TypeError, AttributeError) as exc:
            raise ConfigurationError(f"invalid policy profile: {exc}") from exc
