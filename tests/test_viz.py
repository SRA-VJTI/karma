"""Viser visualization of the packaged models. Never touches a bus."""

from __future__ import annotations

import numpy as np
import pytest

from openpi_control.config import SUPPORTED_MODELS, resolve_model_assets
from openpi_control.exceptions import ConfigurationError
from openpi_control.rigs import resolve_rig, rig_names

viz = pytest.importorskip("openpi_control.viz", reason="needs the 'viz' extra")
pytest.importorskip("viser", reason="needs the 'viz' extra")
pytest.importorskip("yourdfpy", reason="needs the 'viz' extra")

# Models that ship a URDF; FR3 defers its model to the vendor controller.
URDF_MODELS = [m for m in SUPPORTED_MODELS if resolve_model_assets(m).urdf is not None]

# Ports are per-test so a leaked server cannot wedge the next case.
_PORT = iter(range(8500, 8700))


@pytest.fixture(autouse=True)
def isolate_mesh_cache(monkeypatch):
    """Hide any real ~/openpi-data/meshes cache.

    Without this, a developer who has run --fetch-meshes gets different render
    modes than CI. Tests that want a cache patch this again themselves.
    """
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: None)


@pytest.fixture
def make_viz():
    created = []

    def factory(model: str, **kwargs):
        instance = viz.ArmVisualizer(model, port=next(_PORT), **kwargs)
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.server.stop()


@pytest.fixture
def make_scene():
    created = []

    def factory(rig, **kwargs):
        scene = viz.ArmSceneVisualizer.from_rig(rig, port=next(_PORT), **kwargs)
        created.append(scene)
        return scene

    yield factory
    for scene in created:
        scene.server.stop()


def _load_urdf(model: str):
    yourdfpy = viz.import_viz_modules().yourdfpy
    return yourdfpy.URDF.load(
        str(resolve_model_assets(model).urdf),
        load_meshes=False,
        build_collision_scene_graph=False,
    )


@pytest.mark.parametrize("model", URDF_MODELS)
def test_every_packaged_urdf_renders_without_meshes(model: str, make_viz) -> None:
    """The wheel ships no assets/*.stl, so the mesh-free path must carry all models."""
    instance = make_viz(model)
    assert instance.render_mode == "skeleton"
    assert instance.joint_names
    assert len(instance._link_frames) == len(instance._urdf.link_map)


@pytest.mark.parametrize("model", URDF_MODELS)
def test_joint_order_follows_the_kinematic_chain(model: str) -> None:
    """Arm state is indexed along the chain, not by URDF document order.

    Yam.urdf and SO101.urdf both declare their joints tip-first, so trusting
    yourdfpy's document order would drive joint 1 from joint 6's command.
    """
    urdf = _load_urdf(model)
    chain = viz.chain_ordered_actuated_joints(urdf)
    assert sorted(chain) == sorted(urdf.actuated_joint_names)

    # Depth from the base link, counting fixed joints too (ARX_L5 reaches its
    # base_link from a "world" root through a fixed joint).
    depth = {urdf.base_link: 0}
    pending = list(urdf.robot.joints)
    while pending:
        remaining = [j for j in pending if j.parent not in depth]
        for joint in pending:
            if joint.parent in depth:
                depth[joint.child] = depth[joint.parent] + 1
        if len(remaining) == len(pending):
            break
        pending = remaining

    depths = [depth[urdf.joint_map[name].child] for name in chain]
    assert depths == sorted(depths), f"{model} chain is not ordered base-to-tip: {depths}"


def test_yam_chain_order_is_base_to_tip() -> None:
    urdf = _load_urdf("Yam")
    assert tuple(urdf.actuated_joint_names) == (
        "joint6",
        "joint5",
        "joint4",
        "joint3",
        "joint2",
        "joint1",
    )
    assert viz.chain_ordered_actuated_joints(urdf) == (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )


def test_update_matches_a_by_name_reference(make_viz) -> None:
    """A positional update must land on the joints its index names."""
    instance = make_viz("Yam")
    pose = [0.3, 0.9, 0.7, -0.2, 0.4, -0.6]

    reference = _load_urdf("Yam")
    reference.update_cfg(dict(zip(instance.joint_names, pose, strict=True)))
    expected = reference.get_transform("end_link", reference.base_link)

    instance.update(pose)
    actual = instance._urdf.get_transform("end_link", instance._urdf.base_link)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_first_joint_swings_the_tip_about_the_base_axis(make_viz) -> None:
    instance = make_viz("Yam")
    instance.update([0.0] * 6)
    rest = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3].copy()
    instance.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    swung = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3]
    assert abs(swung[2] - rest[2]) < 1e-6
    assert np.linalg.norm(swung[:2] - rest[:2]) > 1e-3


@pytest.mark.parametrize("model", URDF_MODELS)
def test_positions_are_clamped_to_the_urdf_limits(model: str, make_viz) -> None:
    instance = make_viz(model)
    count = len(instance.joint_specs)
    lower = np.array([spec.lower for spec in instance.joint_specs])
    upper = np.array([spec.upper for spec in instance.joint_specs])

    instance.update([1e3] * count)
    np.testing.assert_array_less(instance.positions, upper + 1e-9)
    instance.update([-1e3] * count)
    np.testing.assert_array_less(lower - 1e-9, instance.positions)


def test_rest_pose_is_inside_the_limits(make_viz) -> None:
    for model in URDF_MODELS:
        instance = make_viz(model)
        for spec in instance.joint_specs:
            assert spec.lower <= spec.rest <= spec.upper


def test_update_accepts_a_partial_mapping(make_viz) -> None:
    instance = make_viz("Yam")
    instance.update([0.1] * 6)
    instance.update({"joint3": 0.5})
    assert instance.positions[2] == pytest.approx(0.5)
    assert instance.positions[0] == pytest.approx(0.1)


def test_update_rejects_bad_input(make_viz) -> None:
    instance = make_viz("Yam")
    with pytest.raises(ConfigurationError, match="expects 6 joint positions"):
        instance.update([0.0, 0.0])
    with pytest.raises(ConfigurationError, match="finite"):
        instance.update([float("nan")] * 6)
    with pytest.raises(ConfigurationError, match="unknown joint names"):
        instance.update({"not_a_joint": 0.0})


def test_fr3_reports_that_it_ships_no_urdf() -> None:
    with pytest.raises(ConfigurationError, match="ships no URDF"):
        viz.ArmVisualizer("FR3", port=next(_PORT))


def test_unsupported_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unsupported model"):
        viz.ArmVisualizer("NotAnArm", port=next(_PORT))


def test_missing_mesh_dir_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="not a directory"):
        viz.ArmVisualizer("Yam", mesh_dir="/nonexistent/assets", port=next(_PORT))


def test_mesh_mode_permutes_into_viser_urdf_order(tmp_path, make_viz) -> None:
    """ViserUrdf indexes by its own actuated order; ours is chain order."""
    trimesh = viz.import_viz_modules().trimesh
    assets = tmp_path / "assets"
    assets.mkdir()
    for link in ("base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"):
        for kind in ("visual", "collision"):
            trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(
                assets / f"{link}_{kind}.stl"
            )

    instance = make_viz("Yam", mesh_dir=assets)
    assert instance.render_mode == "mesh"
    vis_order = list(instance._urdf_vis.get_actuated_joint_names())
    permuted = [vis_order[i] for i in instance._vis_permutation]
    assert tuple(permuted) == instance.joint_names

    instance.update([0.0] * 6)
    rest = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3].copy()
    instance.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    swung = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3]
    assert abs(swung[2] - rest[2]) < 1e-6, "joint1 must rotate about the base axis"


def test_gui_builds_one_slider_per_joint(make_viz) -> None:
    instance = make_viz("Yam", effector_model="E_Yam")
    instance.add_gui()
    assert len(instance._sliders) == len(instance.joint_specs)


def test_cli_list_reports_every_model(capsys) -> None:
    assert viz.main(["--list"]) == 0
    out = capsys.readouterr().out
    for model in SUPPORTED_MODELS:
        assert model in out
    assert "no URDF" in out  # FR3


# --------------------------------------------------------------------------- #
# Mesh discovery
# --------------------------------------------------------------------------- #


def test_mesh_dir_falls_back_to_the_local_cache(tmp_path, monkeypatch, make_viz) -> None:
    """With no mesh_dir given, a filled cache should render the real arm."""
    from openpi_control import meshes

    cache = tmp_path / "Yam"
    cache.mkdir()
    trimesh = viz.import_viz_modules().trimesh
    for name in meshes.urdf_mesh_names("Yam"):
        trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(cache / name)

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: cache if model == "Yam" else None)
    instance = make_viz("Yam")
    assert instance.render_mode == "mesh"
    assert instance.mesh_dir == cache


def test_an_empty_cache_leaves_the_skeleton(monkeypatch, make_viz) -> None:
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: None)
    instance = make_viz("Yam")
    assert instance.render_mode == "skeleton"
    assert instance.missing_meshes == ()
    assert "--fetch-meshes" in instance.mesh_status()


def test_missing_meshes_are_reported_not_hidden(tmp_path, monkeypatch, make_viz) -> None:
    """A mesh dir short of a file renders that link bare; say so, don't hide it."""
    from openpi_control import meshes

    cache = tmp_path / "Yam"
    cache.mkdir()
    trimesh = viz.import_viz_modules().trimesh
    for name in meshes.urdf_mesh_names("Yam"):
        if "link_6" in name:
            continue
        trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(cache / name)

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: cache)
    instance = make_viz("Yam")
    assert set(instance.missing_meshes) == {"link_6_visual.stl", "link_6_collision.stl"}
    assert "missing upstream (2)" in instance.mesh_status()
    instance.add_gui()  # must not raise with a missing-mesh row present


def test_explicit_mesh_dir_beats_the_cache(tmp_path, monkeypatch, make_viz) -> None:
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: tmp_path / "cache")
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    trimesh = viz.import_viz_modules().trimesh
    trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(explicit / "base_link_visual.stl")

    instance = make_viz("Yam", mesh_dir=explicit)
    assert instance.mesh_dir == explicit


# --------------------------------------------------------------------------- #
# Rigs in one scene
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rig_name", rig_names())
def test_every_packaged_rig_renders_as_one_scene(rig_name: str, make_scene) -> None:
    """A rig has to reach the browser, not just resolve: this is what --rig runs."""
    rig = resolve_rig(rig_name)
    scene = make_scene(rig)

    assert scene.names == rig.names
    assert scene.label == rig.name
    # One shared server, so the arms land in a single browser scene.
    for name in scene.names:
        assert scene[name].server is scene.server


def test_from_rig_places_each_arm_at_its_rig_base_pose(make_scene) -> None:
    # Without this the two YAM arms render on top of each other and a bimanual
    # scene is unreadable.
    rig = resolve_rig("yam_bimanual")
    scene = make_scene(rig)

    for arm in rig.arms:
        rendered = np.asarray(scene[arm.name].base_frame.position, dtype=np.float64)
        assert rendered == pytest.approx(np.asarray(arm.base_position), abs=1e-6)


def test_from_rig_carries_the_effector_and_tints_the_arms_apart(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"))

    assert [scene[n].effector_model for n in scene.names] == ["E_Yam", "E_Yam"]
    colors = {scene[n]._bone_color for n in scene.names}
    assert len(colors) == len(scene.names)


def test_from_rig_arm_names_are_the_rig_s_own(make_scene) -> None:
    """The CLI, the session, and the scene must share one vocabulary for an arm."""
    scene = make_scene(resolve_rig("yam_bimanual"))

    scene.update("left", [0.1] * 6)
    assert scene["left"].positions[0] == pytest.approx(0.1)
    with pytest.raises(ConfigurationError, match="unknown arm"):
        scene.update("left_follower", [0.0] * 6)


def test_from_rig_takes_an_explicit_label_over_the_rig_name(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"), label="bench")
    assert scene.label == "bench"


def test_scene_update_all_drives_both_arms_independently(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"))

    scene.update_all({"left": [0.2] * 6, "right": [-0.2] * 6})

    assert scene["left"].positions[0] == pytest.approx(0.2)
    assert scene["right"].positions[0] == pytest.approx(-0.2)


def test_cli_list_rigs_describes_each_rig_and_its_arms(capsys) -> None:
    assert viz.main(["--list-rigs"]) == 0
    out = capsys.readouterr().out
    for name in rig_names():
        rig = resolve_rig(name)
        assert name in out
        for arm in rig.arms:
            assert arm.name in out
            assert arm.interface in out


def test_cli_refuses_rig_and_model_together() -> None:
    # --rig takes its models from the rig; silently ignoring --model would be
    # the kind of quiet surprise that costs an hour.
    with pytest.raises(SystemExit):
        viz.main(["--rig", "yam_bimanual", "--model", "Yam"])


def test_live_drives_the_rig_scene_from_the_arms(monkeypatch) -> None:
    """cli.run_live's viser path: the seam test_cli covers only with visualize=False.

    Uses the CLI's own fake backends, so no bus is opened and no node is
    started -- this checks the wiring between run_live and from_rig.
    """
    import threading

    from fake_arm_backend import FakeArmBackend

    from openpi_control import cli

    rig = resolve_rig("yam_bimanual")
    made: dict[str, object] = {}

    def backend_factory(rig_arm):
        backend = FakeArmBackend()
        backend.position = 0.3 if rig_arm.name == "left" else -0.3
        made[rig_arm.name] = backend
        return backend

    scenes = []
    real_from_rig = viz.ArmSceneVisualizer.from_rig

    def capturing_from_rig(cls_rig, **kwargs):
        kwargs["port"] = next(_PORT)
        scene = real_from_rig(cls_rig, **kwargs)
        scenes.append(scene)
        return scene

    monkeypatch.setattr(viz.ArmSceneVisualizer, "from_rig", capturing_from_rig)

    stop = threading.Event()
    stop.set()  # one pass through the mirror loop, then unwind

    # camera_preview=False on purpose: this test is about the arm-to-scene
    # wiring, and a developer with the cell's cameras plugged in should not
    # have the suite take them away from whatever is streaming them.
    assert (
        cli.run_live(
            rig,
            visualize=True,
            camera_preview=False,
            backend_factory=backend_factory,
            stop=stop,
        )
        == 0
    )

    assert len(scenes) == 1
    assert scenes[0].names == rig.names
    # Parked on the way down, and the scene's server was stopped with it.
    for backend in made.values():
        assert backend.closes[0] is True
        assert not backend.connected


# --------------------------------------------------------------------------- #
# camera preview
# --------------------------------------------------------------------------- #


class _FakeReader:
    """A CameraReader's preview surface: a spec, a frame count, and latest().

    Fake rather than real on purpose -- a suite that opens the cell's cameras
    takes them away from whatever is streaming them, and the panel's job is
    pushing frames, not capturing them.
    """

    def __init__(self, name: str, *, label: str = "", frame=None) -> None:
        from openpi_control.cameras import CameraSpec

        self.spec = CameraSpec(
            name=name, label=label, serial="254623070531", device=f"/dev/{name}"
        )
        self.frames_read = 0
        self._frame = frame

    def latest(self):
        return self._frame

    def publish(self, frame) -> None:
        """What the capture thread does: newest frame wins, count goes up."""
        self._frame = frame
        self.frames_read += 1


def _frame(width: int = 848, height: int = 480) -> np.ndarray:
    """A BGR frame whose channels are tellable apart, so a flip is visible."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 10  # blue
    frame[..., 1] = 120  # green
    frame[..., 2] = 240  # red
    return frame


@pytest.fixture
def preview_server():
    import viser

    created = []

    def factory():
        server = viser.ViserServer(port=next(_PORT))
        created.append(server)
        return server

    yield factory
    for server in created:
        server.stop()


def test_a_preview_tile_exists_before_any_frame_arrives(preview_server) -> None:
    """The layout is final when the page opens, not after the first frame."""
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader})

    assert panel.names == ("top",)
    panel.step(1.0)  # nothing published yet
    assert reader.latest() is None  # and the panel did not invent one


def test_a_published_frame_reaches_its_tile_as_rgb(preview_server) -> None:
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader})
    reader.publish(_frame())

    panel.step(1.0)

    image = panel._tiles["top"].image
    # BGR in, RGB out: the red channel of the source has to come out first.
    assert tuple(image[0, 0]) == (240, 120, 10)
    assert image.dtype == np.uint8
    assert image.flags["C_CONTIGUOUS"]  # encoders want contiguous memory


def test_a_preview_is_shrunk_to_the_requested_width(preview_server) -> None:
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader}, max_width=400)
    reader.publish(_frame(848, 480))

    panel.step(1.0)

    height, width, _ = panel._tiles["top"].image.shape
    assert width <= 400
    # 848 -> stride 3 -> 283x160, and the aspect ratio survives the subsample.
    assert (width, height) == (283, 160)


def test_a_frame_already_smaller_than_the_cap_is_left_alone(preview_server) -> None:
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader}, max_width=400)
    reader.publish(_frame(320, 240))

    panel.step(1.0)

    assert panel._tiles["top"].image.shape == (240, 320, 3)


def test_a_preview_pushes_at_its_own_rate_not_the_callers(preview_server) -> None:
    """The mirror runs at 30 Hz; a preview has no reason to."""
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader}, rate_hz=10.0)
    reader.publish(_frame())
    panel.step(1.0)
    first = panel._tiles["top"].image

    reader.publish(_frame() // 2)  # a new, different frame
    panel.step(1.0 / 30.0)  # one mirror tick: inside the preview period

    assert np.array_equal(panel._tiles["top"].image, first)  # not pushed yet
    panel.step(1.0 / 30.0)
    panel.step(1.0 / 30.0)  # three mirror ticks add up to a preview period
    assert not np.array_equal(panel._tiles["top"].image, first)


def test_a_camera_that_stopped_delivering_is_not_pushed_again(preview_server) -> None:
    """A frozen stream holds its last image instead of re-encoding it."""
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader})
    reader.publish(_frame())

    panel.step(1.0)
    assert panel._pushed["top"] == 1
    panel.step(1.0)  # a full period later, still the same frame
    assert panel._pushed["top"] == 1


def test_a_tile_is_labelled_by_camera_name_first(preview_server) -> None:
    """``name`` is the word the rig, the CLI, and the dataset all use."""
    panel = viz.CameraPanel(
        preview_server(), {"left_wrist": _FakeReader("left_wrist", label="left gripper view")}
    )

    assert panel._tiles["left_wrist"].label == "left_wrist — left gripper view"


def test_a_panel_with_no_cameras_is_allowed(preview_server) -> None:
    """``--only left`` on a rig whose one camera is unplugged still runs."""
    panel = viz.CameraPanel(preview_server(), {})

    assert panel.names == ()
    panel.step(1.0)  # and stepping it is a no-op, not a crash


def test_a_preview_refuses_nonsense_settings(preview_server) -> None:
    for kwargs in (
        {"max_width": 0},
        {"rate_hz": 0.0},
        {"rate_hz": -1.0},
        {"image_format": "webp"},
    ):
        with pytest.raises(ConfigurationError):
            viz.CameraPanel(preview_server(), {}, **kwargs)


def _rgb_frame(width: int = 848, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 240  # red first: already RGB
    frame[..., 1] = 120
    frame[..., 2] = 10
    return frame


def test_pushed_frames_keep_their_capture_resolution(preview_server) -> None:
    """`infer` shows what the model was handed, so nothing may be subsampled."""
    panel = viz.CameraPanel(
        preview_server(), {"top": _FakeReader("top")}, max_width=None, image_format="png"
    )

    panel.push({"top": _rgb_frame(848, 480)})

    image = panel._tiles["top"].image
    assert image.shape == (480, 848, 3)
    assert tuple(image[0, 0]) == (240, 120, 10)  # rgb in, rgb out
    assert image.flags["C_CONTIGUOUS"]


def test_pushing_bypasses_the_readers_and_the_clock(preview_server) -> None:
    """The tile is the observation, not a fresh picture taken beside it."""
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader}, max_width=None)
    reader.publish(_frame())  # what the camera has now, and must not be shown

    panel.push({"top": _rgb_frame(320, 240)})

    assert panel._tiles["top"].image.shape == (240, 320, 3)
    assert reader.frames_read == 1  # the panel never pulled


def test_pushing_skips_names_with_no_tile_and_missing_frames(preview_server) -> None:
    panel = viz.CameraPanel(preview_server(), {"top": _FakeReader("top")}, max_width=None)

    panel.push({"top": None, "no_such_camera": _rgb_frame(8, 8)})

    assert panel._tiles["top"].image.shape == (3, 4, 3)  # still the placeholder


def test_a_preview_refuses_a_frame_that_is_not_a_colour_image(preview_server) -> None:
    reader = _FakeReader("top")
    panel = viz.CameraPanel(preview_server(), {"top": reader})
    reader.publish(np.zeros((480, 848), dtype=np.uint8))  # a depth frame, say

    with pytest.raises(ConfigurationError, match="HxWx3"):
        panel.step(1.0)


def test_removing_a_panel_drops_its_tiles(preview_server) -> None:
    panel = viz.CameraPanel(preview_server(), {"top": _FakeReader("top")})

    panel.remove()

    assert panel._tiles == {}


# --- predicted action-chunk overlay -------------------------------------------
#
# The overlay is built once per chunk and advanced per tick. Two failure modes
# are guarded here, both of which happened: rebuilding it per tick cost 61 ms
# against the 33 ms tick budget at the default 30 Hz, and hiding executed
# segments drained the trail to nothing over each chunk.


def _chunk(steps: int = 6, *, dof: int = 7, drift: float = 0.05) -> np.ndarray:
    """A bimanual chunk whose flange actually moves on every step."""
    actions = np.zeros((steps, 2 * dof))
    for index in range(steps):
        pose = np.array([0.0, -0.4, 0.6, 0.0, 0.3, 0.0, 1.0])[:dof] + index * drift
        actions[index, :dof] = pose
        actions[index, dof:] = pose
    return actions


def _split(actions: np.ndarray) -> dict[str, np.ndarray]:
    return {"left": actions[:, :7], "right": actions[:, 7:]}


def _drawn(scene) -> dict[str, int]:
    return {
        name: sum(1 for h in segments if h is not None and h.visible)
        for name, segments in scene._chunk_segments.items()
    }


def test_a_chunk_draws_one_segment_per_predicted_step(make_scene) -> None:
    """Plus the leading segment from the measured pose into the first action."""
    scene = make_scene(resolve_rig("yam_bimanual"))
    actions = _chunk(steps=6)

    scene.update_chunk(_split(actions))

    for name in ("left", "right"):
        assert len(scene._chunk_segments[name]) == 6
        assert len(scene._chunk_points[name]) == 7  # measured pose + 6 actions
    assert _drawn(scene) == {"left": 6, "right": 6}


def test_the_whole_trail_stays_on_screen_for_the_life_of_the_chunk(make_scene) -> None:
    """The regression this replaced: retiring executed segments emptied the scene.

    Hiding each segment as its action ran drained 60 segments to 0 over a chunk
    -- 1.0 s at the default 30 Hz -- and left nothing at all on screen through
    every inference round trip, which reads as no overlay.
    """
    scene = make_scene(resolve_rig("yam_bimanual"))
    actions = _chunk(steps=8)
    scene.update_chunk(_split(actions))

    for index in range(len(actions)):
        scene.set_chunk_progress(index + 1)
        assert _drawn(scene) == {"left": 8, "right": 8}


def test_advancing_the_chunk_moves_the_progress_marker_along_the_trail(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"))
    actions = _chunk(steps=6)
    scene.update_chunk(_split(actions))
    points = scene._chunk_points["left"]
    np.testing.assert_allclose(scene._chunk_markers["left"].position, points[0], atol=1e-6)

    scene.set_chunk_progress(3)

    np.testing.assert_allclose(scene._chunk_markers["left"].position, points[3], atol=1e-6)
    np.testing.assert_allclose(
        scene._chunk_markers["right"].position, scene._chunk_points["right"][3], atol=1e-6
    )


def test_the_final_tick_parks_the_marker_at_the_end_of_the_chunk(make_scene) -> None:
    """``set_chunk_progress(len(actions))`` indexes one past the last action."""
    scene = make_scene(resolve_rig("yam_bimanual"))
    actions = _chunk(steps=5)
    scene.update_chunk(_split(actions))

    scene.set_chunk_progress(len(actions))

    points = scene._chunk_points["left"]
    np.testing.assert_allclose(scene._chunk_markers["left"].position, points[-1], atol=1e-6)
    assert _drawn(scene) == {"left": 5, "right": 5}


def test_advancing_the_chunk_adds_and_removes_no_scene_geometry(make_scene) -> None:
    """The per-tick path must not rebuild the overlay: that is the whole fix.

    A rebuild is 60 mesh removes, 62 forward-kinematics evaluations and 60 mesh
    adds; measured at 61 ms, against 33 ms of tick budget at the default 30 Hz.
    """
    scene = make_scene(resolve_rig("yam_bimanual"))
    actions = _chunk(steps=8)
    scene.update_chunk(_split(actions))
    handles = list(scene._chunk_handles)
    added = 0
    original = scene.server.scene.add_mesh_trimesh

    def counting(*args, **kwargs):
        nonlocal added
        added += 1
        return original(*args, **kwargs)

    scene.server.scene.add_mesh_trimesh = counting  # type: ignore[method-assign]
    for index in range(len(actions)):
        scene.set_chunk_progress(index + 1)

    assert added == 0
    assert scene._chunk_handles == handles


def test_a_chunk_the_policy_predicted_as_a_hold_collapses_to_its_approach(
    make_scene,
) -> None:
    """Identical actions have no motion between them, only into the first one."""
    scene = make_scene(resolve_rig("yam_bimanual"))
    held = np.tile(_chunk(steps=1)[0], (4, 1))

    scene.update_chunk(_split(held))

    segments = scene._chunk_segments["left"]
    assert segments[0] is not None  # measured pose -> the held action
    assert segments[1:] == [None, None, None]  # slots kept, nothing to draw
    assert scene._chunk_markers["left"].visible


def test_a_chunk_that_moves_nothing_at_all_still_shows_the_marker(make_scene) -> None:
    """Every segment degenerate: the marker is the only thing left to draw.

    Without it a policy predicting the pose the arm already holds would put
    nothing on screen, which is indistinguishable from a broken overlay.
    """
    scene = make_scene(resolve_rig("yam_bimanual"))
    resting = np.concatenate([scene["left"].positions, [1.0]])
    chunk = np.tile(np.concatenate([resting, resting]), (3, 1))

    scene.update_chunk(_split(chunk))

    assert scene._chunk_segments["left"] == [None, None, None]
    assert scene._chunk_handles  # the markers, and nothing else
    assert scene._chunk_markers["left"].visible


def test_stepping_back_through_a_chunk_moves_the_marker_back(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"))
    actions = _chunk(steps=5)
    scene.update_chunk(_split(actions))
    scene.set_chunk_progress(4)

    scene.set_chunk_progress(1)

    np.testing.assert_allclose(
        scene._chunk_markers["left"].position, scene._chunk_points["left"][1], atol=1e-6
    )


def test_the_next_chunk_starts_over_at_its_own_beginning(make_scene) -> None:
    """Progress belongs to the displayed chunk, not to the scene."""
    scene = make_scene(resolve_rig("yam_bimanual"))
    scene.update_chunk(_split(_chunk(steps=5)))
    scene.set_chunk_progress(4)

    scene.update_chunk(_split(_chunk(steps=5, drift=0.07)))

    assert scene._chunk_consumed == 0
    np.testing.assert_allclose(
        scene._chunk_markers["left"].position, scene._chunk_points["left"][0], atol=1e-6
    )


def test_advancing_with_no_chunk_on_screen_is_harmless(make_scene) -> None:
    """ctrl-c between the clear and the next prediction must not raise."""
    scene = make_scene(resolve_rig("yam_bimanual"))
    scene.update_chunk(_split(_chunk(steps=3)))
    scene.clear_chunk()

    scene.set_chunk_progress(2)

    assert scene._chunk_segments == {}
    assert scene._chunk_markers == {}
    assert scene._chunk_consumed == 0


def test_the_overlay_is_anchored_to_the_pose_it_was_predicted_from(make_scene) -> None:
    """Moving the arm after the prediction must not redraw the trail.

    Re-anchoring per tick is what made the old overlay stretch its first segment
    toward wherever the arm had already travelled.
    """
    scene = make_scene(resolve_rig("yam_bimanual"))
    scene.update_chunk(_split(_chunk(steps=4)))
    before = [p.copy() for p in scene._chunk_points["left"]]

    scene.update("left", np.array([0.4, -0.8, 0.9, 0.2, 0.1, 0.3]))
    scene.set_chunk_progress(1)

    for was, now in zip(before, scene._chunk_points["left"], strict=True):
        np.testing.assert_allclose(was, now, atol=1e-9)


def test_a_chunk_narrower_than_the_arm_is_rejected(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"))

    with pytest.raises(ConfigurationError, match="at least 6"):
        scene.update_chunk({"left": np.zeros((4, 3)), "right": np.zeros((4, 3))})


# --- page theme and palette ---------------------------------------------------


def _greyscale(color) -> float:
    """WCAG relative luminance, scaled to 0-255: what a hue-blind viewer sees."""

    def channel(value: float) -> float:
        value /= 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = color
    return 255 * (0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue))


@pytest.fixture
def theme_calls(monkeypatch):
    """Record configure_theme kwargs; it queues a message with no readable state."""
    calls: list[dict] = []
    gui_api = pytest.importorskip("viser._gui_api")
    original = gui_api.GuiApi.configure_theme

    def recording(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(gui_api.GuiApi, "configure_theme", recording)
    return calls


def test_a_page_we_serve_asks_for_the_dark_canvas_its_palette_assumes(
    make_scene, theme_calls
) -> None:
    """The amber axis markers sit at 2.0:1 on viser's default white canvas."""
    make_scene(resolve_rig("yam_bimanual"))

    assert len(theme_calls) == 1
    assert theme_calls[0]["dark_mode"] is True


def test_a_page_we_serve_does_not_offer_to_publish_the_cell(
    make_scene, theme_calls
) -> None:
    """The share button relays the page; a live robot cell is not one stray click."""
    make_scene(resolve_rig("yam_bimanual"))

    assert theme_calls[0]["show_share_button"] is False


def test_a_single_arm_page_is_themed_the_same_way(make_viz, theme_calls) -> None:
    make_viz("Yam")

    assert len(theme_calls) == 1
    assert theme_calls[0]["dark_mode"] is True


def test_a_server_handed_in_keeps_whatever_theme_its_owner_chose(theme_calls) -> None:
    """A caller's page is a caller's page, chrome included."""
    server = viz.import_viz_modules().viser.ViserServer(port=next(_PORT))
    try:
        viz.ArmSceneVisualizer({"only": viz.ArmSpec("Yam")}, server=server)
        assert theme_calls == []
    finally:
        server.stop()


def test_the_scene_arms_never_share_the_gui_accent_colour() -> None:
    """A highlighted slider must not read as one of the arms."""
    assert viz._PAGE_BRAND_COLOR not in viz._ARM_COLORS


def test_the_first_two_arm_tints_differ_in_lightness_not_only_hue() -> None:
    """Translucent trails overlap, and hue alone is not a distinction then.

    The earlier amber sat 13/255 from the blue in greyscale -- one colour to
    anyone not seeing hue, and to anyone reading a screenshot in greyscale.
    """
    left, right = viz._ARM_COLORS[0], viz._ARM_COLORS[1]

    assert abs(_greyscale(left) - _greyscale(right)) > 40
