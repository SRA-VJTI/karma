"""SO101 policy, recorder and CLI contracts without moving hardware."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_inference import _Reader, _state

from openpi_control import cameras as cameras_mod
from openpi_control import cli, inference
from openpi_control.inference_record import InferenceRolloutSource
from openpi_control.rigs import resolve_rig


def states_for(contract):
    states = {}
    for name in contract.arms:
        state = _state(name, effector=0.8)
        joints = dataclasses.replace(
            state.joints,
            names=state.joints.names[:5],
            frame_age_ms=np.zeros(5),
            position_rad=np.zeros(5),
            velocity_rad_s=np.zeros(5),
            effort_nm=np.zeros(5),
            temperature_c=np.ones(5) * 25,
            current_a=np.zeros(5),
        )
        states[name] = dataclasses.replace(state, joints=joints)
    return states


@pytest.mark.parametrize("rig_name,width", [("so101", 6), ("so101_bimanual", 12)])
def test_so101_observation_http_and_bounded_playback(rig_name, width):
    rig = dataclasses.replace(resolve_rig(rig_name), policy_norm_tag="test_so101")
    contract = inference.PolicyContract.from_rig(rig)
    states = states_for(contract)
    arms = {name: SimpleNamespace(latest_state=state) for name, state in states.items()}
    readers = {"top": _Reader(np.zeros((2, 3, 3), dtype=np.uint8))}
    observation = inference.build_observation(arms, readers, contract=contract)
    assert observation.state.shape == (width,)
    np.testing.assert_allclose(observation.state[5::6], 0.2)

    class Session:
        def get(self, *a, **kw):
            return SimpleNamespace(
                text=json.dumps(
                    dict(status="ok", norm_tag="test_so101", state_dim=width, num_cameras=1)
                )
            )

        def post(self, *a, **kw):
            import json_numpy

            request = json_numpy.loads(kw["data"])
            assert "top_cam" in request and "left_cam" not in request and "right_cam" not in request
            action = np.zeros((2, width))
            action[:, 5::6] = 1.0  # dataset/policy closed -> native 0
            return SimpleNamespace(status_code=200, text=json_numpy.dumps({"actions": action}))

    client = inference.MolmoActClient(contract=contract, session=Session(), jpeg_quality=0)
    client.health()
    actions = client.infer(observation, "pick")
    executor = inference.BoundedChunkExecutor(contract=contract, max_effector_step=0.2)
    executor.reset(states)
    commands = executor.step(actions[0])
    for command in commands.values():
        assert len(command.position_rad) == 5
        assert command.effector == pytest.approx(0.6)
    source = InferenceRolloutSource(
        arms=arms,
        readers=readers,
        client=client,
        instruction="pick",
        episode_seconds=30,
        prefetch=False,
    )
    step = source.poll(states)
    assert set(step.targets) == set(contract.arms)
    source.close()


def test_so101_rejects_yam_server_and_missing_normalization():
    with pytest.raises(inference.ConfigurationError, match="norm-tag"):
        inference.PolicyContract.from_rig(resolve_rig("so101"))
    contract = inference.PolicyContract(("right",), 5, ("top",), "so101", 0.0)
    session = SimpleNamespace(
        get=lambda *a, **kw: SimpleNamespace(
            text=json.dumps(
                dict(status="ok", norm_tag="yam_dual_molmoact2", state_dim=14, num_cameras=3)
            )
        )
    )
    with pytest.raises(inference.InferenceError):
        inference.MolmoActClient(contract=contract, session=session).health()


@pytest.mark.parametrize("command", ["inference", "rollout", "hitl"])
def test_policy_cli_carries_so101_camera_and_norm(command, monkeypatch):
    seen = []
    target = "run_infer" if command == "inference" else "run_" + command
    monkeypatch.setattr(cli, target, lambda rig, **kw: seen.append(rig) or 0)
    argv = [
        command,
        "--rig",
        "so101",
        "--norm-tag",
        "trained",
        "--camera-serial",
        "top=123456",
        "--skip-preflight",
    ]
    argv += ["--instruction", "pick"] if command == "inference" else ["--repo-id", "local/test"]
    assert cli.main(argv) == 0
    assert seen[0].camera_names == ("top",)
    assert seen[0].cameras[0].serial == "123456"
    assert seen[0].policy_norm_tag == "trained"


def test_policy_cli_turns_camera_paths_into_webcams_on_so101(monkeypatch, tmp_path):
    # SO101 ships a placeholder serial, so `--camera ROLE=/dev/videoN` with no
    # `--camera-serial` is a webcam declaration, not a RealSense device pin.
    seen = []
    monkeypatch.setattr(cli, "run_infer", lambda rig, **kw: seen.append(rig) or 0)
    top, side = tmp_path / "video5", tmp_path / "video7"
    argv = [
        "inference",
        "--rig",
        "so101",
        "--norm-tag",
        "trained",
        "--camera",
        f"top={top}",
        "--camera",
        f"right_wrist={side}",
        "--skip-preflight",
        "--instruction",
        "pick",
    ]
    assert cli.main(argv) == 0
    rig = seen[0]
    assert rig.camera_names == ("top", "right_wrist")
    assert [c.backend for c in rig.cameras] == ["opencv", "opencv"]
    assert [c.device for c in rig.cameras] == [str(top), str(side)]
    assert rig.cameras[1].arm == "right"


def test_policy_cli_keeps_camera_as_a_realsense_pin_when_a_serial_is_given(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "run_infer", lambda rig, **kw: seen.append(rig) or 0)
    argv = [
        "inference",
        "--rig",
        "so101",
        "--norm-tag",
        "trained",
        "--camera-serial",
        "top=123456",
        "--camera",
        f"top={tmp_path / 'video4'}",
        "--skip-preflight",
        "--instruction",
        "pick",
    ]
    assert cli.main(argv) == 0
    assert seen[0].cameras[0].backend == "realsense"
    assert seen[0].cameras[0].serial == "123456"


def test_so101_side_view_goes_on_the_wire_as_side_cam():
    # The SO100/SO101 checkpoints take a fixed second view named ``side``; it
    # must reach the server as ``side_cam``, not disguised as a wrist.
    rig = resolve_rig("so101")
    rig = dataclasses.replace(
        rig,
        policy_norm_tag="test_so101",
        cameras=rig.cameras
        + (
            cameras_mod.RigCamera(
                name="side", serial="0", label="side", backend="opencv", device="/dev/video7"
            ),
        ),
    )
    contract = inference.PolicyContract.from_rig(rig)
    assert contract.cameras == ("top", "side")
    states = states_for(contract)
    arms = {name: SimpleNamespace(latest_state=state) for name, state in states.items()}
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    readers = {"top": _Reader(frame), "side": _Reader(frame)}
    observation = inference.build_observation(arms, readers, contract=contract)
    assert observation.side_cam is not None

    seen = {}

    class Session:
        def post(self, *a, **kw):
            import json_numpy

            seen.update(json_numpy.loads(kw["data"]))
            return SimpleNamespace(
                status_code=200, text=json_numpy.dumps({"actions": np.zeros((2, 6))})
            )

    client = inference.MolmoActClient(contract=contract, session=Session(), jpeg_quality=0)
    client.infer(observation, "pick")
    assert "top_cam" in seen and "side_cam" in seen
    assert "right_cam" not in seen


def test_policy_cli_accepts_a_side_webcam(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "run_infer", lambda rig, **kw: seen.append(rig) or 0)
    argv = [
        "inference", "--rig", "so101", "--norm-tag", "trained",
        "--camera", f"top={tmp_path / 'video5'}", "--camera", f"side={tmp_path / 'video7'}",
        "--skip-preflight", "--instruction", "pick",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    assert seen[0].camera_names == ("top", "side")
    assert seen[0].cameras[1].arm is None
