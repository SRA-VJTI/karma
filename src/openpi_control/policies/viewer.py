"""Viser policy preview and intent controls; this module never commands hardware."""

import logging
import threading
import time

from ..inference import decode_wire_frame, split_chunk


class PolicyPanel:
    def __init__(self, scene, rig, source, readers, *, mesh_dir=None):
        from ..viz import ArmSceneVisualizer, ArmSpec, CameraPanel

        self.scene, self.source = scene, source
        self.session = source.policy_session
        self.control = self.session.control
        self.done = threading.Event()
        self.ghost = ArmSceneVisualizer(
            {
                "prediction_" + a.name: ArmSpec(
                    a.model,
                    effector_model=a.effector_model,
                    instance_config=a.instance_config,
                    mesh_dir=mesh_dir,
                    base_position=a.base_position,
                    base_rotation_wxyz=a.base_rotation_wxyz,
                    color=(130, 160, 180),
                )
                for a in rig.arms
            },
            server=scene.server,
            show_grid=False,
        )
        for arm in self.ghost.arms.values():
            arm.base_frame.visible = False
        with scene.server.gui.add_folder("Policy controls"):
            self.info = scene.server.gui.add_markdown("Waiting for prediction")
            self.stop = scene.server.gui.add_button("Stop policy")
            self.resume = scene.server.gui.add_button("Resume policy")
            self.home = scene.server.gui.add_button("Home and stop")
            self.stop.on_click(lambda _: self.control.pause())
            self.resume.on_click(lambda _: self.control.resume())
            self.home.on_click(lambda _: self.control.home_and_stop())
        self.live = CameraPanel(scene.server, readers, folder="Live cameras")
        self.inputs = CameraPanel(scene.server, readers, folder="Model input", max_width=None)
        self.thread = threading.Thread(target=self._loop, name="policy-viser", daemon=True)
        self.thread.start()

    def _loop(self):
        previous = None
        while not self.done.wait(0.125):
            try:
                session, control = self.session, self.control
                preview = session.preview if control.permits(session._generation) else None
                for name, arm in self.source._arms.items():
                    state = arm.latest_state
                    if state is not None:
                        self.scene.update(name, state.joints.position_rad)
                if preview is not previous:
                    self.scene.clear_chunk()
                    if preview is not None:
                        chunks = split_chunk(preview, session.client.contract)
                        self.scene.update_chunk(chunks)
                        for name, chunk in chunks.items():
                            self.ghost.update(
                                "prediction_" + name, chunk[-1, : session.client.contract.arm_dof]
                            )
                        self.inputs.push(
                            {k: decode_wire_frame(v) for k, v in session.input_frames.items()}
                        )
                    for arm in self.ghost.arms.values():
                        arm.base_frame.visible = preview is not None
                    previous = preview
                self.scene.set_chunk_progress(session.progress)
                self.live.step(0.125)
                self.resume.disabled = not control.paused or control.human or control.home
                self.stop.disabled = control.paused or control.home
                self.home.disabled = control.home
                latency = getattr(session.client, "last_latency", {})
                age = (
                    time.monotonic() - session.last_observation_at
                    if session.last_observation_at
                    else 0
                )
                checkpoint = session.client.config.checkpoint or session.client.contract.norm_tag
                self.info.content = (
                    f"**{session.client.display_name}** · {control.reason}\n\n"
                    f"Checkpoint: {checkpoint}\n\nTask: {session.instruction}\n\n"
                    f"Chunk {session.preview_id}, "
                    f"row {session.progress}/{len(preview) if preview is not None else 0}; "
                    f"{session.action_hz:g} actions/s; "
                    f"{getattr(session.client, 'mode', 'prefetch')}; "
                    f"round trip {latency.get('round_trip_s', 0):.3f}s\n\n"
                    f"Observation age {age:.2f}s\n\n"
                    "Trails and grey endpoint ghosts are model targets. "
                    "Measured arms show hardware feedback; bounded commands may differ."
                )
            except Exception:
                logging.getLogger(__name__).exception("Policy preview update failed")
                self.done.set()

    def close(self):
        self.done.set()
        self.thread.join(timeout=1)
        self.scene.clear_chunk()
        self.live.remove()
        self.inputs.remove()
        for arm in self.ghost.arms.values():
            arm.base_frame.visible = False
        for handle in (self.stop, self.resume, self.home, self.info):
            handle.remove()
