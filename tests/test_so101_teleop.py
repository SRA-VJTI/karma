"""SO101 kinematics, controller lifecycle, and native bridge regressions."""
# ruff: noqa: E402

import dataclasses
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("websockets")

from openpi_control import cli
from openpi_control.exceptions import ConfigurationError
from openpi_control.rigs import resolve_rig
from openpi_control.teleop_vr import QuestTeleopSource
from openpi_control.types import ArmMode, ArmRole, ArmState, EffectorState, JointState
from vr_teleop_kit.ik.so101_ik import JOINT_NAMES, SO101IKSolver
from vr_teleop_kit.lerobot.bi_quest_teleop import BiQuestTeleoperator, BiQuestTeleoperatorConfig


def measured(q):
    return ArmState(
        name="right",
        role=ArmRole.FOLLOWER,
        joints=JointState(
            names=JOINT_NAMES,
            position_rad=q,
            velocity_rad_s=[0.0] * 5,
            effort_nm=[0.0] * 5,
            temperature_c=[25.0] * 5,
            current_a=[0.0] * 5,
        ),
        effector=EffectorState(position=0.7),
        mode=ArmMode.HOLD,
        monotonic_timestamp=time.monotonic(),
        wall_timestamp=0.0,
        sequence=1,
    )


def operator():
    return BiQuestTeleoperator(
        BiQuestTeleoperatorConfig(robot_model="SO101", publish_ik_state=False)
    )


def controller(*, x=0.0, grip=True, trigger=0.3, rest=False):
    buttons = [{"p": False, "v": 0.0} for _ in range(6)]
    buttons[0]["v"] = trigger
    buttons[1]["p"] = grip
    buttons[3]["p"] = rest
    return {"position": [x, 1.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0], "buttons": buttons}


def test_fk_matches_mujoco_urdf_and_jacobian():
    path = Path(__file__).parents[1] / "src/openpi_control/models/arms/SO101/SO101.urdf"
    root = ET.parse(path).getroot()
    for link in root.findall("link"):
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", balanceinertia="true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    solver = SO101IKSolver()
    for q in (np.zeros(5), np.array([0.2, -0.4, 0.5, -0.3, 0.6])):
        for name, value in zip(JOINT_NAMES, q, strict=True):
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        pos, _, jac = solver._kinematics(q)
        np.testing.assert_allclose(pos, data.body("end_link").xpos, atol=1e-8)
        for i in range(5):
            perturbed = q.copy()
            perturbed[i] += 1e-6
            np.testing.assert_allclose(
                (solver.fk(perturbed)[0] - pos) / 1e-6, jac[:3, i], atol=1e-6
            )


def test_reachable_target_converges_and_unreachable_target_stays_bounded():
    solver = SO101IKSolver()
    q = np.array([0.1, -0.2, 0.3, 0.2, -0.1])
    target, quat = solver.fk(q + np.array([0.1, 0.1, -0.1, 0.05, 0.15]))
    for _ in range(300):
        q = solver.solve(target, quat, q)
    assert np.linalg.norm(solver.fk(q)[0] - target) < 1e-4
    for _ in range(200):
        next_q = solver.solve([10.0, -10.0, 10.0], [0.0, 1.0, 0.0, 0.0], q)
        assert np.all(np.isfinite(next_q))
        assert np.max(np.abs(next_q - q)) <= 0.02000001
        assert np.all(next_q >= solver.lower) and np.all(next_q <= solver.upper)
        q = next_q
    with pytest.raises(ValueError, match="finite"):
        solver.solve([float("nan"), 0.0, 0.0], quat, q)


def test_bridge_seeds_five_joints_and_preserves_gripper_without_controller():
    teleop = operator()
    bridge = QuestTeleopSource(["right"], robot_model="SO101", teleoperator=teleop)
    q = np.array([0.2, -0.3, 0.4, -0.1, 0.5])
    assert not bridge.poll({"right": None}).targets
    result = bridge.poll({"right": measured(q)}).targets["right"]
    np.testing.assert_allclose(result.position_rad, q)
    assert result.effector == pytest.approx(0.7)
    assert teleop._arms["right"]["qpos"].shape == (6,)
    assert "right_joint_6.pos" not in teleop.get_action()
    wrong = dataclasses.replace(
        measured(q),
        joints=dataclasses.replace(
            measured(q).joints,
            names=("a",),
            position_rad=[0.0],
            velocity_rad_s=[0.0],
            effort_nm=[0.0],
            temperature_c=[0.0],
            current_a=[0.0],
            frame_age_ms=None,
        ),
    )
    with pytest.raises(ConfigurationError, match="expects 5"):
        bridge.seed_from({"right": wrong})


def test_clutch_motion_release_stale_reanchor_and_gripper():
    teleop = operator()
    q = np.array([0.1, -0.3, 0.4, -0.1, 0.2])
    teleop.seed_qpos_from_obs({f"right_joint_{i + 1}.pos": v for i, v in enumerate(q)})
    arm = teleop._arms["right"]
    teleop._update_arm("right", controller(), None, 0.0)
    np.testing.assert_allclose(arm["qpos"][:5], q, atol=1e-12)
    for _ in range(20):
        teleop._update_arm("right", controller(x=0.03), None, 0.0)
    assert np.linalg.norm(arm["qpos"][:5] - q) > 1e-3
    held = arm["qpos"].copy()
    teleop._update_arm("right", controller(x=1.0), None, 1.0)
    np.testing.assert_array_equal(arm["qpos"], held)
    teleop._update_arm("right", controller(x=1.0), None, 0.0)
    np.testing.assert_allclose(arm["qpos"], held, atol=1e-12)
    teleop._update_arm("right", controller(grip=False, trigger=1.0), None, 0.0)
    np.testing.assert_allclose(arm["qpos"], held, atol=1e-12)
    assert arm["trigger"] == 0.3
    assert not arm["engaged"]
    teleop._apply_config_update({"max_dq_per_joint_scalar_pos": 0.01})
    assert arm["solver"].max_dq_per_joint.shape == (5,)
    assert teleop._settings_profile()["robot_model"] == "SO101"


def test_so101_cli_presets_and_rest_pose(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path))
    seen = {}
    monkeypatch.setattr(cli, "_command_teleop", lambda args, path: seen.update(args=args) or 0)
    assert (
        cli.main(
            ["teleop", "--rig", "so101", "--rest-pose-right", "0,0,0,0,0", "--max-dq-pos", "0.01"]
        )
        == 0
    )
    config = cli._teleop_config(seen["args"])
    assert config["rest_qpos_right"] == [0.0] * 5
    assert config["max_dq_per_joint"] == [0.01] * 3 + [0.02] * 2
    assert resolve_rig("so101").names == ("right",)
    assert resolve_rig("so101_bimanual").names == ("left", "right")
    with pytest.raises(ConfigurationError, match="YAM-only"):
        QuestTeleopSource(["right"], robot_model="SO101", model_path="yam.xml")


def test_so101_native_loop_selects_solver_and_sends_five_joints(monkeypatch):
    import threading
    from types import SimpleNamespace

    from openpi_control import teleop_runtime

    rig = resolve_rig("so101")
    stop = threading.Event()
    sent, closed = [], []
    q = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
    arm = SimpleNamespace(latest_state=measured(q))

    def command(target):
        sent.append(target)
        stop.set()

    arm.command = command
    live = SimpleNamespace(name="right", rig_arm=rig.arms[0], arm=arm)
    monkeypatch.setattr(cli, "power_up", lambda *a, **k: (object(), [live]))
    monkeypatch.setattr(cli, "power_down", lambda *a, **k: closed.append(True) or 0)

    def source(names, **kwargs):
        assert kwargs["robot_model"] == "SO101"
        return QuestTeleopSource(names, teleoperator=operator(), **kwargs)

    monkeypatch.setattr(teleop_runtime, "QuestTeleopSource", source)
    assert teleop_runtime.run_teleop(rig, rate_hz=50, stop=stop) == 0
    np.testing.assert_allclose(sent[0].position_rad, q)
    assert sent[0].effector == pytest.approx(0.7)
    assert closed == [True]


def test_profile_defaults_keep_explicit_overrides_and_yam_defaults():
    yam = BiQuestTeleoperatorConfig()
    assert len(yam.rest_qpos_left) == 6
    assert yam.max_dq_per_joint == [0.06] * 3 + [0.24] * 3
    so101 = BiQuestTeleoperatorConfig(robot_model="SO101", scale_translation=1.5)
    assert so101.scale_translation == 1.5
    assert so101.max_dq_per_joint == [0.02] * 5
    with pytest.raises(ValueError, match="rest_qpos"):
        BiQuestTeleoperator(
            BiQuestTeleoperatorConfig(robot_model="SO101", rest_qpos_right=[0.0] * 6)
        )


def test_single_arm_wrapper_inherits_profile_defaults():
    from vr_teleop_kit.lerobot.single_arm_quest_teleop import (
        SingleArmQuestTeleoperator,
        SingleArmQuestTeleoperatorConfig,
    )

    config = SingleArmQuestTeleoperatorConfig(robot_model="SO101")
    teleop = SingleArmQuestTeleoperator(config)
    assert len(config.rest_qpos_right) == 5
    assert set(teleop.action_features) == set(teleop.get_action())
    assert len(teleop.action_features) == 6
    assert len(SingleArmQuestTeleoperatorConfig().rest_qpos_right) == 6


def test_yam_teleoperator_retains_six_joints_and_two_finger_slots(monkeypatch):
    from vr_teleop_kit.lerobot import bi_quest_teleop

    # Geometry-independent regression for the generalized controller plumbing.
    monkeypatch.setattr(bi_quest_teleop, "DecoupledIKSolver", lambda **kw: object())
    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig())
    teleop.seed_qpos_from_obs({"right_joint_6.pos": 0.4, "right_gripper.pos": 1.0})
    assert teleop.get_action()["right_joint_6.pos"] == 0.4
    np.testing.assert_allclose(teleop._arms["right"]["qpos"][6:], [0.0475, 0.0475])
    assert len(teleop.action_features) == 14


def test_so101_rest_ramp_reaches_five_joint_target(monkeypatch):
    teleop = operator()
    q = np.array([0.1, -0.3, 0.4, -0.1, 0.2])
    teleop.seed_qpos_from_obs({f"right_joint_{i + 1}.pos": v for i, v in enumerate(q)})
    now = [10.0]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])
    teleop._update_arm("right", controller(grip=False, rest=True), None, 0.0)
    now[0] += 2.1
    teleop._update_arm("right", controller(grip=False), None, 0.0)
    np.testing.assert_allclose(teleop._arms["right"]["qpos"][:5], np.zeros(5))
    assert not teleop._arms["right"]["ramp_active"]


def test_hardware_folded_pose_uses_native_tolerance_and_bounded_idle_target():
    teleop = operator()
    bridge = QuestTeleopSource(["right"], robot_model="SO101", teleoperator=teleop)
    q = (np.array([1965, 905, 3199, 2159, 2086]) - 2048) * 2 * np.pi / 4096
    solver = teleop._arms["right"]["solver"]
    bounded = np.clip(q, solver.lower, solver.upper)
    target = bridge.poll({"right": measured(q)}).targets["right"]
    np.testing.assert_allclose(target.position_rad, bounded)
    # Gripping at an arbitrary controller pose must not add any movement.
    teleop._update_arm("right", controller(x=0.7), None, 0.0)
    np.testing.assert_allclose(teleop._arms["right"]["qpos"][:5], bounded, atol=1e-12)
    for _ in range(20):
        old = teleop._arms["right"]["qpos"][:5].copy()
        teleop._update_arm("right", controller(x=0.71), None, 0.0)
        new = teleop._arms["right"]["qpos"][:5]
        assert np.all(new >= solver.lower) and np.all(new <= solver.upper)
        assert np.max(np.abs(new - old)) <= 0.0200001


def test_excessive_calibration_error_still_rejected_atomically():
    teleop = operator()
    bridge = QuestTeleopSource(["right"], robot_model="SO101", teleoperator=teleop)
    solver = teleop._arms["right"]["solver"]
    q = np.zeros(5)
    q[2] = solver.upper[2] + solver.seed_margin[2] + 0.01
    original = teleop._arms["right"]["qpos"].copy()
    with pytest.raises(ConfigurationError, match="native tracking tolerance.*elbow_flex"):
        bridge.poll({"right": measured(q)})
    assert not bridge._seeded
    np.testing.assert_array_equal(teleop._arms["right"]["qpos"], original)


def test_float32_and_encoder_overrun_match_native_margin_without_expanding_limits():
    solver = SO101IKSolver()
    np.testing.assert_allclose(solver.seed_margin, [0.1] * 5)
    for overrun in (0.0, 2 * np.pi / 4096, 0.075, 0.1):
        q = solver.upper + overrun
        np.testing.assert_allclose(
            solver.validate_seed(q.astype(np.float32)), solver.upper, atol=1e-6
        )
    with pytest.raises(ValueError, match="elbow_flex"):
        solver.validate_seed(solver.upper + solver.seed_margin + 0.01)
