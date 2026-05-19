"""YAM bimanual VR teleoperation agent using Meta Quest + WebXR."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
import viser
import viser.extras
import viser.transforms as vtf
from dm_env.specs import Array
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from robots_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_pad

# ---------------------------------------------------------------------------
# Coordinate frame transform
# ---------------------------------------------------------------------------
# WebXR / Meta Quest tracking space: X right, Y up, Z toward viewer (right-handed).
# YAM robot base frame: X forward (toward user), Y left, Z up.
# For a user standing in front of the robot and facing it:
#   - VR +X (user's right) -> robot -Y (robot's right side)
#   - VR +Y (up)           -> robot +Z (up)
#   - VR +Z (toward user)  -> robot -X (away from robot)
#
# Adjust this matrix (or expose as a YAML parameter) if the robot is oriented
# differently relative to the operator.
_R_VR_TO_ROBOT: np.ndarray = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

_SO3_R_VR_TO_ROBOT = vtf.SO3.from_matrix(_R_VR_TO_ROBOT)


def _rotate_pos(vr_delta: np.ndarray) -> np.ndarray:
    return _R_VR_TO_ROBOT @ vr_delta


def _rotate_quat(vr_wxyz: np.ndarray) -> np.ndarray:
    """Express a VR-frame quaternion in the robot base frame."""
    return (_SO3_R_VR_TO_ROBOT @ vtf.SO3(vr_wxyz) @ _SO3_R_VR_TO_ROBOT.inverse()).wxyz


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class YamVRTeleopAgent(Agent):
    """Bimanual YAM VR teleoperation via Meta Quest WebXR.

    Hosts a FastAPI server with:
      - GET /    -> serves the WebXR HTML page (open in Quest browser)
      - WS  /ws  -> receives controller pose messages from the Quest

    The Quest controller positions are mapped to YAM arm IK targets in a
    *relative* mode: pressing both grips simultaneously for ~0.5 s sets the
    reference pose (VR calibration point → current IK handle position).
    Subsequent movements are applied as deltas from that reference.

    Grip-button values (0–1) drive the left / right gripper sliders.

    Args:
        bimanual:            Control both arms. Default True.
        right_arm_extrinsic: {"position": [x,y,z], "rotation": [w,x,y,z]}
                             of the right arm base in the left arm base frame.
                             Required when bimanual=True.
        viser_port:          Port for the Viser 3-D viewer.  Default 8080.
        vr_port:             Port for the WebXR WebSocket server.  Default 8766.
        workspace_scale:     Scale VR metre-delta to robot metre-delta.
                             Shrink (<1) to limit robot excursion when the
                             operator's reach exceeds the robot workspace.
    """

    def __init__(
        self,
        *,
        bimanual: bool = True,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
        viser_port: int = 8080,
        vr_port: int = 8766,
        workspace_scale: float = 0.8,
    ) -> None:
        self.bimanual = bimanual
        self.right_arm_extrinsic = right_arm_extrinsic
        self.vr_port = vr_port
        self.workspace_scale = workspace_scale

        if bimanual:
            assert right_arm_extrinsic is not None, (
                "right_arm_extrinsic must be provided for bimanual YAM VR teleop"
            )

        # ── Viser + IK ────────────────────────────────────────────────────
        self.viser_server = viser.ViserServer(port=viser_port)
        self.ik = YamPyroki(viser_server=self.viser_server, bimanual=bimanual)
        self.ik_thread = threading.Thread(target=self.ik.run, name="yam_vr_ik", daemon=True)
        self.ik_thread.start()

        # ── Visualization ─────────────────────────────────────────────────
        self.obs: Optional[Dict[str, Any]] = None
        self._setup_visualization()
        self._vis_thread = threading.Thread(
            target=self._update_visualization, name="yam_vr_vis", daemon=True
        )
        self._vis_thread.start()

        # ── Calibration state ─────────────────────────────────────────────
        # Populated when the user sends a "calibrate" message.
        self._cal_vr_left: Optional[np.ndarray] = None    # [w,x,y,z, px,py,pz]
        self._cal_vr_right: Optional[np.ndarray] = None
        self._cal_ik_left: Optional[np.ndarray] = None    # IK handle [w,x,y,z, px,py,pz]
        self._cal_ik_right: Optional[np.ndarray] = None
        self._calibrated: bool = False

        # Grip-hold tracking for in-VR calibration gesture (both grips > 0.8 for 0.5 s)
        self._grip_hold_start: Optional[float] = None
        self._GRIP_CALIBRATE_THRESHOLD: float = 0.8
        self._GRIP_CALIBRATE_DURATION: float = 0.5

        # ── WebXR server ──────────────────────────────────────────────────
        self._app = self._build_fastapi()
        self._ws_thread = threading.Thread(
            target=self._run_vr_server, name="yam_vr_ws", daemon=True
        )
        self._ws_thread.start()

    # ── Visualization ──────────────────────────────────────────────────────

    def _setup_visualization(self) -> None:
        self.base_frame_left_real = self.viser_server.scene.add_frame(
            "/base_left_real", show_axes=False
        )
        self.urdf_vis_left_real = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/base_left_real",
            mesh_color_override=(0.55, 0.75, 0.95),
        )
        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.25  # type: ignore[attr-defined]

        self.left_gripper_slider = self.viser_server.gui.add_slider(
            "Left Gripper", min=0.0, max=2.4, step=0.01, initial_value=0.0
        )

        if self.bimanual and self.right_arm_extrinsic is not None:
            self.ik.base_frame_right.position = np.array(
                self.right_arm_extrinsic["position"]
            )
            self.ik.base_frame_right.wxyz = np.array(
                self.right_arm_extrinsic["rotation"]
            )
            self.base_frame_right_real = self.viser_server.scene.add_frame(
                "/base_left_real/base_right_real", show_axes=False
            )
            self.base_frame_right_real.position = self.ik.base_frame_right.position
            self.urdf_vis_right_real = viser.extras.ViserUrdf(
                self.viser_server,
                deepcopy(self.ik.urdf),
                root_node_name="/base_left_real/base_right_real",
                mesh_color_override=(0.55, 0.75, 0.95),
            )
            for mesh in self.urdf_vis_right_real._meshes:
                mesh.opacity = 0.25  # type: ignore[attr-defined]
            self.right_gripper_slider = self.viser_server.gui.add_slider(
                "Right Gripper", min=0.0, max=2.4, step=0.01, initial_value=0.0
            )

        self._viser_cam_handles: dict = {}
        self._vr_status = self.viser_server.gui.add_text(
            "VR Status", initial_value="Waiting for Quest connection…"
        )

    def _update_visualization(self) -> None:
        while self.obs is None:
            time.sleep(0.025)
        while True:
            obs_copy = self.obs
            if obs_copy is None:
                time.sleep(0.02)
                continue
            try:
                self.urdf_vis_left_real.update_cfg(
                    np.flip(obs_copy["left"]["joint_pos"])
                )
                if self.bimanual:
                    self.urdf_vis_right_real.update_cfg(  # type: ignore[attr-defined]
                        np.flip(obs_copy["right"]["joint_pos"])
                    )
                rgb_images = obs_get_rgb(obs_copy)
                for key, image in rgb_images.items():
                    if key not in self._viser_cam_handles:
                        self._viser_cam_handles[key] = self.viser_server.gui.add_image(
                            image, label=key
                        )
                    self._viser_cam_handles[key].image = resize_with_pad(image, 224, 224)
            except Exception:
                pass
            time.sleep(0.02)

    # ── WebXR / FastAPI server ─────────────────────────────────────────────

    def _build_fastapi(self) -> FastAPI:
        app = FastAPI()
        html_path = Path(__file__).parent / "vr_client.html"

        @app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse(html_path.read_text())

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self._vr_status.value = "Quest connected"
            try:
                while True:
                    raw = await websocket.receive_text()
                    msg = json.loads(raw)
                    self._handle_vr_message(msg)
            except WebSocketDisconnect:
                self._vr_status.value = "Quest disconnected"

        return app

    def _run_vr_server(self) -> None:
        uvicorn.run(
            self._app,
            host="0.0.0.0",
            port=self.vr_port,
            log_level="warning",
        )

    # ── VR message handling ────────────────────────────────────────────────

    def _handle_vr_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "calibrate":
            self._do_calibrate(msg)
        elif kind == "poses":
            self._check_grip_calibrate(msg)
            if self._calibrated:
                self._apply_vr_poses(msg)

    def _check_grip_calibrate(self, msg: dict) -> None:
        """Calibrate automatically when both grips are held for 0.5 s."""
        left = msg.get("left") or {}
        right = msg.get("right") or {}
        both_gripped = (
            float(left.get("grip", 0.0)) > self._GRIP_CALIBRATE_THRESHOLD
            and float(right.get("grip", 0.0)) > self._GRIP_CALIBRATE_THRESHOLD
        )
        now = time.monotonic()
        if both_gripped:
            if self._grip_hold_start is None:
                self._grip_hold_start = now
            elif now - self._grip_hold_start >= self._GRIP_CALIBRATE_DURATION:
                self._do_calibrate(msg)
                self._grip_hold_start = None  # reset so it doesn't re-fire
        else:
            self._grip_hold_start = None

    def _do_calibrate(self, msg: dict) -> None:
        """Snapshot VR reference poses and current IK handle positions."""
        left = msg.get("left")
        right = msg.get("right")

        if left:
            self._cal_vr_left = np.array(
                left["quat"] + left["pos"], dtype=np.float64
            )
            ctl = self.ik.transform_handles["left"].control
            if ctl is not None:
                self._cal_ik_left = np.concatenate(
                    [np.array(ctl.wxyz), np.array(ctl.position)], dtype=np.float64
                )

        if right and self.bimanual:
            self._cal_vr_right = np.array(
                right["quat"] + right["pos"], dtype=np.float64
            )
            ctl = self.ik.transform_handles["right"].control
            if ctl is not None:
                self._cal_ik_right = np.concatenate(
                    [np.array(ctl.wxyz), np.array(ctl.position)], dtype=np.float64
                )

        self._calibrated = (
            self._cal_vr_left is not None
            and self._cal_ik_left is not None
            and (
                not self.bimanual
                or (
                    self._cal_vr_right is not None
                    and self._cal_ik_right is not None
                )
            )
        )
        self._vr_status.value = (
            "Calibrated — tracking active" if self._calibrated else "Calibration incomplete"
        )

    def _apply_vr_poses(self, msg: dict) -> None:
        """Convert incoming VR poses to IK targets and update gripper sliders."""
        left = msg.get("left")
        right = msg.get("right")

        if left and self._cal_vr_left is not None and self._cal_ik_left is not None:
            self._update_arm(
                side="left",
                vr_pos=np.array(left["pos"], dtype=np.float64),
                vr_wxyz=np.array(left["quat"], dtype=np.float64),
                grip=float(left.get("grip", 0.0)),
                cal_vr=self._cal_vr_left,
                cal_ik=self._cal_ik_left,
                gripper_slider=self.left_gripper_slider,
            )

        if (
            right
            and self.bimanual
            and self._cal_vr_right is not None
            and self._cal_ik_right is not None
        ):
            self._update_arm(
                side="right",
                vr_pos=np.array(right["pos"], dtype=np.float64),
                vr_wxyz=np.array(right["quat"], dtype=np.float64),
                grip=float(right.get("grip", 0.0)),
                cal_vr=self._cal_vr_right,
                cal_ik=self._cal_ik_right,
                gripper_slider=self.right_gripper_slider,  # type: ignore[attr-defined]
            )

    def _update_arm(
        self,
        side: str,
        vr_pos: np.ndarray,
        vr_wxyz: np.ndarray,
        grip: float,
        cal_vr: np.ndarray,
        cal_ik: np.ndarray,
        gripper_slider: Any,
    ) -> None:
        # Position: delta in VR space → rotate to robot space → add to calibration IK target
        delta_vr = vr_pos - cal_vr[4:]
        new_pos = cal_ik[4:] + _rotate_pos(delta_vr) * self.workspace_scale

        # Orientation: rotate calibration delta into robot base frame
        cal_vr_rot = vtf.SO3(cal_vr[:4])
        cur_vr_rot = vtf.SO3(vr_wxyz)
        delta_vr_rot = cal_vr_rot.inverse() @ cur_vr_rot
        delta_robot_rot = (
            _SO3_R_VR_TO_ROBOT @ delta_vr_rot @ _SO3_R_VR_TO_ROBOT.inverse()
        )
        new_wxyz = (vtf.SO3(cal_ik[:4]) @ delta_robot_rot).wxyz

        ctl = self.ik.transform_handles[side].control
        if ctl is not None:
            ctl.position = tuple(new_pos)  # type: ignore[assignment]
            ctl.wxyz = tuple(new_wxyz)  # type: ignore[assignment]

        # Grip (0–1) → gripper range (0–2.4 for YAM)
        gripper_slider.value = grip * 2.4

    # ── Agent interface ────────────────────────────────────────────────────

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)
        action: Dict[str, Dict[str, np.ndarray]] = {
            "left": {
                "pos": np.concatenate(
                    [np.flip(self.ik.joints["left"]), [self.left_gripper_slider.value]]
                ).astype(np.float32),
            }
        }
        if self.bimanual:
            action["right"] = {
                "pos": np.concatenate(
                    [
                        np.flip(self.ik.joints["right"]),
                        [self.right_gripper_slider.value],  # type: ignore[attr-defined]
                    ]
                ).astype(np.float32),
            }
        return action

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        spec: Dict[str, Dict[str, Array]] = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)}
        }
        if self.bimanual:
            spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return spec


__all__ = ["YamVRTeleopAgent"]
