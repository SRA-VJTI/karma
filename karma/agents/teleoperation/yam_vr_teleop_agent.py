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

from karma.agents.agent import Agent
from karma.robots.inverse_kinematics.yam_pyroki import YamPyroki
from karma.sensors.cameras.camera_utils import obs_get_rgb, resize_with_pad

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
# Helpers
# ---------------------------------------------------------------------------

def _fit_sphere(points: np.ndarray) -> np.ndarray:
    """Fit a sphere to N×3 points via linear least squares. Returns centre."""
    # |p - c|^2 = r^2  →  2p·c - (|c|^2 - r^2) = |p|^2
    A = np.column_stack([2.0 * points, np.ones(len(points))])
    b = np.sum(points ** 2, axis=1)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3]  # centre


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
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        position_smoothing: float = 0.7,
        dead_zone_m: float = 0.003,
        speed_damping: float = 4.0,
    ) -> None:
        self.bimanual = bimanual
        self.right_arm_extrinsic = right_arm_extrinsic
        self.vr_port = vr_port
        self.workspace_scale = workspace_scale
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.position_smoothing = position_smoothing
        self.dead_zone_m = dead_zone_m
        self.speed_damping = speed_damping

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

        # ── Wrist pivot calibration ───────────────────────────────────────
        # WebXR reports grip pose at the palm centre; the true wrist pivot is
        # offset from that.  We find the offset by having the user twist their
        # wrist in place for 5 s — the palm traces a sphere whose centre is
        # the pivot.  Store the offset in controller frame (VR space).
        self._wrist_offset: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._cal_positions: Dict[str, list] = {"left": [], "right": []}
        self._cal_active: Dict[str, bool] = {"left": False, "right": False}

        # ── Input smoothing state ─────────────────────────────────────────
        # EMA-smoothed VR position and orientation — kills hand-tremor jitter.
        # Reset to None on each grip rising edge so re-grip always starts fresh.
        self._smooth_pos: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._smooth_wxyz: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

        # ── Adaptive scaling state ────────────────────────────────────────
        self._speed_ema: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self._prev_vr_pos: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

        # ── Per-arm grip-to-activate state ────────────────────────────────
        # Squeeze grip to start tracking that arm; release to freeze it.
        # Each arm is independent — no global calibration step needed.
        self._gripping: Dict[str, bool] = {"left": False, "right": False}
        self._cal_vr: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._cal_ik: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._GRIP_THRESHOLD: float = 0.5

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
            self.ik._load_fresh_urdf(),
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
                self.ik._load_fresh_urdf(),
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
            ssl_certfile=self.ssl_certfile,
            ssl_keyfile=self.ssl_keyfile,
        )

    # ── VR message handling ────────────────────────────────────────────────

    def _handle_vr_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "poses":
            self._apply_vr_poses(msg)
        elif kind == "wrist_cal_start":
            side = msg.get("side")
            if side in self._cal_positions:
                self._cal_positions[side] = []
                self._cal_active[side] = True
                self._vr_status.value = f"Wrist cal {side}: twist for 5s…"
        elif kind == "wrist_cal_end":
            side = msg.get("side")
            if side in self._cal_positions and self._cal_active[side]:
                self._cal_active[side] = False
                pts = np.array(self._cal_positions[side])
                if len(pts) >= 10:
                    pivot = _fit_sphere(pts)
                    self._wrist_offset[side] = pivot - np.mean(pts, axis=0)
                    self._vr_status.value = f"Wrist cal {side}: done (offset={self._wrist_offset[side].round(3)})"
                else:
                    self._vr_status.value = f"Wrist cal {side}: not enough data"

    def _apply_vr_poses(self, msg: dict) -> None:
        sides = ["left", "right"] if self.bimanual else ["left"]
        for side in sides:
            data = msg.get(side)
            if not data:
                continue

            raw_pos = np.array(data["pos"], dtype=np.float64)
            vr_wxyz = np.array(data["quat"], dtype=np.float64)

            # Collect positions during wrist calibration
            if self._cal_active[side]:
                self._cal_positions[side].append(raw_pos.tolist())

            # Apply wrist pivot offset: shift from palm centre to true pivot
            if self._wrist_offset[side] is not None:
                R = vtf.SO3(vr_wxyz).as_matrix()
                vr_pos = raw_pos + R @ self._wrist_offset[side]
            else:
                vr_pos = raw_pos

            # EMA smoothing — damps high-frequency hand tremor.
            # Always runs (even when disarmed) so re-arming never has stale state.
            # position_smoothing: 0 = raw pass-through, ~0.9 = very smooth/laggy.
            # Quaternion uses linear blend + renorm (NLERP) for orientation.
            a = self.position_smoothing
            if self._smooth_pos[side] is None:
                self._smooth_pos[side] = vr_pos.copy()
                self._smooth_wxyz[side] = vr_wxyz.copy()
            else:
                self._smooth_pos[side] = a * self._smooth_pos[side] + (1.0 - a) * vr_pos
                q0, q1 = self._smooth_wxyz[side], vr_wxyz
                if np.dot(q0, q1) < 0.0:  # shortest-path flip
                    q1 = -q1
                q = a * q0 + (1.0 - a) * q1
                self._smooth_wxyz[side] = q / np.linalg.norm(q)
            vr_pos = self._smooth_pos[side]
            vr_wxyz = self._smooth_wxyz[side]

            # Snapshot origin on first pose for this arm
            if self._cal_vr[side] is None:
                self._cal_vr[side] = np.concatenate([vr_wxyz, vr_pos])
                ctl = self.ik.transform_handles[side].control
                if ctl is not None:
                    self._cal_ik[side] = np.concatenate(
                        [np.array(ctl.wxyz), np.array(ctl.position)]
                    ).astype(np.float64)

            if self._cal_vr[side] is None or self._cal_ik[side] is None:
                continue

            # Only track while grip (middle finger) is held.
            # Release to freeze arm in place; re-grip to resume from current position.
            grip = float(data.get("grip", 0.0))
            was_gripping = self._gripping[side]
            now_gripping = grip > self._GRIP_THRESHOLD
            if now_gripping and not was_gripping:
                # Re-snapshot origin so re-gripping from a new hand position doesn't jump.
                # Also reset smoothing state so the EMA starts clean at the new grip point.
                self._smooth_pos[side] = vr_pos.copy()
                self._smooth_wxyz[side] = vr_wxyz.copy()
                self._prev_vr_pos[side] = None
                self._speed_ema[side] = 0.0
                self._cal_vr[side] = np.concatenate([vr_wxyz, vr_pos])
                ctl = self.ik.transform_handles[side].control
                if ctl is not None:
                    self._cal_ik[side] = np.concatenate(
                        [np.array(ctl.wxyz), np.array(ctl.position)]
                    ).astype(np.float64)
            self._gripping[side] = now_gripping
            if not now_gripping:
                continue

            gripper_slider = (
                self.left_gripper_slider if side == "left"
                else self.right_gripper_slider  # type: ignore[attr-defined]
            )
            self._update_arm(
                side=side,
                vr_pos=vr_pos,
                vr_wxyz=vr_wxyz,
                trigger=float(data.get("trigger", 0.0)),
                cal_vr=self._cal_vr[side],
                cal_ik=self._cal_ik[side],
                gripper_slider=gripper_slider,
            )

    def _update_arm(
        self,
        side: str,
        vr_pos: np.ndarray,
        vr_wxyz: np.ndarray,
        trigger: float,
        cal_vr: np.ndarray,
        cal_ik: np.ndarray,
        gripper_slider: Any,
    ) -> None:
        # Adaptive scaling: slow movements get full scale, fast movements are
        # damped to prevent unsafe robot excursions.
        if self._prev_vr_pos[side] is not None:
            speed = float(np.linalg.norm(vr_pos - self._prev_vr_pos[side])) * 60.0  # m/s at ~60 Hz
            self._speed_ema[side] = 0.85 * self._speed_ema[side] + 0.15 * speed
        self._prev_vr_pos[side] = vr_pos.copy()
        adaptive_scale = self.workspace_scale / (1.0 + self.speed_damping * self._speed_ema[side])

        # Position: delta in VR space → rotate to robot space → add to calibration IK target.
        # Smooth dead zone: linearly ramp from 0 at the threshold to full above it.
        # Avoids the step discontinuity that a hard cutoff causes at the boundary.
        delta_vr  = vr_pos - cal_vr[4:]
        delta_norm = float(np.linalg.norm(delta_vr))
        if delta_norm > 1e-9:
            effective = max(0.0, delta_norm - self.dead_zone_m)
            delta_vr  = delta_vr * (effective / delta_norm)
        new_pos = cal_ik[4:] + _rotate_pos(delta_vr) * adaptive_scale

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

        # Trigger (0–1) → gripper range (0–2.4 for YAM)
        gripper_slider.value = trigger * 2.4

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
