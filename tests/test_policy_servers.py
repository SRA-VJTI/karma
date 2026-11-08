"""GPU adapter contract tests using an in-memory model."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from openpi_control.inference import encode_frame

spec = importlib.util.spec_from_file_location(
    "karma_act_server", Path(__file__).parents[1] / "servers/act_server.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("provider", ["molmoact2", "pi05"])
@pytest.mark.parametrize("jpeg", [0, 90])
def test_adapter_validates_and_decodes(provider, jpeg):
    metadata = dict(
        provider=provider,
        norm_tag="test",
        state_dim=14,
        camera_roles=["top", "left_wrist", "right_wrist"],
        action_hz=20,
    )

    def predict(images, state, instruction, payload):
        assert all(v.shape == (4, 4, 3) for v in images.values())
        return np.zeros((3, 14))

    service = module.ActService(predict, metadata)
    frame = encode_frame(np.zeros((4, 4, 3), np.uint8), jpeg_quality=jpeg)
    payload = dict(
        state=np.zeros(14),
        instruction="fold",
        normalization_tag="test",
        top_cam=frame,
        left_cam=frame,
        right_cam=frame,
    )
    result = service.act(payload)
    assert result["actions"].shape == (3, 14)
    assert result["inference_ms"] == result["dt_ms"]
    assert result["action_hz"] == 20
    assert service.metadata["contract_version"] == 1
    with pytest.raises(ValueError, match="normalization"):
        service.act(dict(payload, normalization_tag="wrong"))
    with pytest.raises(ValueError, match="state"):
        service.act(dict(payload, state=np.full(14, np.nan)))
    with pytest.raises(KeyError):
        service.act({k: v for k, v in payload.items() if k != "left_cam"})
    service.predict = lambda *a: np.full((3, 14), np.nan)
    with pytest.raises(RuntimeError, match="invalid actions"):
        service.act(payload)
