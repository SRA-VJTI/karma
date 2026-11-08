"""The structural interface for robot-side policy adapters."""

from typing import Any, Protocol

import numpy as np

from ..inference import BimanualObservation, PolicyContract
from .config import PolicyConfig


class PolicyClient(Protocol):
    """One provider; calls are serialized by PolicySession, never by the controller."""

    managed: bool
    config: PolicyConfig
    contract: PolicyContract
    display_name: str
    action_hz: float
    jpeg_quality: int
    last_latency: dict[str, float]
    last_wire_frames: dict[str, np.ndarray]

    def health(self) -> dict[str, Any]:
        """Validate model availability and contract before hardware startup."""
        ...

    def reset(self) -> None:
        """Start a new model episode; called only between inference requests."""
        ...

    def infer(self, observation: BimanualObservation, instruction: str) -> np.ndarray:
        """Return finite absolute targets in the contract's canonical robot frame."""
        ...

    def close(self) -> None:
        """Release provider connections and workers with bounded waiting."""
        ...
