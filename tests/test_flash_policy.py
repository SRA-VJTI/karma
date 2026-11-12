import json
from dataclasses import asdict

import numpy as np
import pytest

from openpi_control.exceptions import ConfigurationError
from openpi_control.flash_policy import (
    CAMERAS,
    CONTRACT,
    FlashChunkClient,
    FlashConfig,
    blend_chunks,
    flip_grippers,
    prepare_image,
)
from openpi_control.inference import BimanualObservation, InferenceError


def config(**kwargs):
    return FlashConfig(
        task="pick cup", image_size=(32, 32), chunk_size=4, execute_steps=2, blend_steps=2, **kwargs
    )


def test_mask_is_shared_rgb_and_does_not_mutate_raw():
    raw = np.full((64, 64, 3), 255, np.uint8)
    cfg = config(masks={"top": [[0, 0, 1, 1]]})
    transformed = prepare_image(raw, "top", cfg)
    expected = (np.array([124, 116, 104]) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(transformed[:, 0, 0], expected, atol=1e-6)
    assert raw.min() == 255
    assert transformed.shape == (3, 32, 32)
    assert prepare_image(raw, "left_wrist", cfg).mean() > 2


def test_raw_and_lossless_encoded_images_preprocess_identically():
    import io

    from PIL import Image

    raw = np.random.default_rng(1).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    data = io.BytesIO()
    Image.fromarray(raw).save(data, format="PNG")
    np.testing.assert_array_equal(
        prepare_image(raw, "top", config()),
        prepare_image(np.frombuffer(data.getvalue(), np.uint8), "top", config()),
    )


@pytest.mark.parametrize(
    "kwargs", [dict(fps=0), dict(masks={"left": []}), dict(masks={"top": [[0, 0, 2, 1]]})]
)
def test_invalid_checkpoint_contract(kwargs):
    with pytest.raises(ConfigurationError):
        config(**kwargs)


def test_polarity_roundtrip_changes_only_grippers():
    state = np.linspace(0, 1, 14)
    converted = flip_grippers(state)
    np.testing.assert_allclose(flip_grippers(converted), state, atol=1e-7)
    np.testing.assert_allclose(converted[[6, 13]], 1 - state[[6, 13]])
    np.testing.assert_allclose(converted[:6], state[:6])
    with pytest.raises(ConfigurationError):
        flip_grippers(np.full(14, np.nan))


def test_overlap_zero_actions_are_valid_and_inputs_are_untouched():
    old, new = np.zeros((4, 14)), np.ones((4, 14))
    blended = blend_chunks(old, new, executed=2, overlap=2)
    np.testing.assert_array_equal(blended[0], 0)
    np.testing.assert_array_equal(blended[1:], 1)
    assert new.min() == 1
    with pytest.raises(InferenceError):
        blend_chunks(old, new[:2], executed=2, overlap=2)


class FakeHTTP:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bad = False

    def get(self, *args, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            text=json.dumps(
                {
                    "status": "ok",
                    "norm_tag": CONTRACT.norm_tag,
                    "state_dim": 14,
                    "num_cameras": 3,
                    "cameras": list(CAMERAS),
                    "gripper_open": 1,
                    "flash_config": asdict(self.cfg),
                }
            )
        )

    def post(self, *args, **kwargs):
        from types import SimpleNamespace

        import json_numpy

        actions = np.zeros((4, 14)) if not self.bad else np.zeros((2, 14))
        return SimpleNamespace(text=json_numpy.dumps({"actions": actions}), status_code=200)


def test_client_contract_prefix_and_error_reset():
    session = FakeHTTP(config())
    client = FlashChunkClient(session=session, jpeg_quality=0)
    client.health()
    raw = np.zeros((32, 32, 3), np.uint8)
    obs = BimanualObservation(raw, raw, raw, np.zeros(14), contract=CONTRACT)
    assert client.infer(obs, "pick cup").shape == (2, 14)
    session.bad = True
    with pytest.raises(InferenceError):
        client.infer(obs, "pick cup")
    assert client.previous is None
    with pytest.raises(InferenceError):
        FlashChunkClient(session=session, fps=60).health()
    with pytest.raises(InferenceError):
        client.infer(obs, "another task")


def test_act_training_save_reload_and_http_contract(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    pytest.importorskip("fastapi")
    import json_numpy
    from fastapi.testclient import TestClient

    from openpi_control.flash_serve import Predictor, create_app
    from openpi_control.flash_train import make_policy, prepare_sample

    torch.set_num_threads(1)
    cfg = config()
    stats = {
        f"{name}.{kind}": np.full(14, 0 if kind == "mean" else 1, np.float32)
        for name in ("observation.state", "action")
        for kind in ("mean", "std")
    }
    sample = {
        "observation.state": torch.zeros(14),
        "action": torch.zeros((4, 14)),
        "action_is_pad": torch.tensor([False, False, False, True]),
        "action.dt_s": torch.tensor([1 / 30]),
    }
    sample.update({f"observation.images.{c}": torch.zeros((3, 32, 32)) for c in CAMERAS})
    batch = {
        k: v.unsqueeze(0).repeat(2, *([1] * v.ndim))
        for k, v in prepare_sample(sample, cfg, stats).items()
    }
    policy = make_policy(cfg, "cpu", tiny=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    # Serving must use its requested device and the complete checkpoint, without
    # fetching the backbone initialization saved in a training configuration.
    policy.config.device = "cuda"
    policy.config.pretrained_backbone_weights = "not-a-real-backbone-weight"
    policy.save_pretrained(tmp_path)
    np.savez(tmp_path / "normalization.npz", **stats)
    (tmp_path / "flash.json").write_text(json.dumps(asdict(cfg)))
    app = create_app(Predictor(tmp_path))
    client = TestClient(app)
    assert client.get("/act").json()["state_dim"] == 14
    payload = {
        "state": np.zeros(14, np.float32),
        "instruction": cfg.task,
        "normalization_tag": CONTRACT.norm_tag,
    }
    payload.update(
        {key: np.zeros((32, 32, 3), np.uint8) for key in ("top_cam", "left_cam", "right_cam")}
    )
    result = client.post("/act", content=json_numpy.dumps(payload))
    assert result.status_code == 200, result.text
    actions = json_numpy.loads(result.text)["actions"]
    assert actions.shape == (4, 14) and np.isfinite(actions).all()
    assert np.all((0 <= actions[:, [6, 13]]) & (actions[:, [6, 13]] <= 1))
    payload["normalization_tag"] = "arx"
    assert client.post("/act", content=json_numpy.dumps(payload)).status_code == 400
    sample["action.dt_s"] = torch.tensor([0.4])
    with pytest.raises(ValueError, match="irregular"):
        prepare_sample(sample, cfg, stats)
