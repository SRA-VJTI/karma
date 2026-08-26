"""MolmoAct2 protocol, mapping, and hardware-loop tests without a robot."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fake_arm_backend import FakeArmBackend

from openpi_control import cli, inference
from openpi_control.exceptions import ConfigurationError
from openpi_control.rigs import resolve_rig
from openpi_control.types import (
    ArmMode,
    ArmRole,
    ArmState,
    EffectorState,
    JointState,
    PositionCommand,
)


def _state(name: str, *, position: float = 0.0, effector: float = 0.25) -> ArmState:
    return ArmState(
        name=name,
        role=ArmRole.FOLLOWER,
        joints=JointState(
            names=tuple(f"joint_{i + 1}" for i in range(6)),
            position_rad=[position] * 6,
            velocity_rad_s=[0.0] * 6,
            effort_nm=[0.0] * 6,
            temperature_c=[25.0] * 6,
            current_a=[0.0] * 6,
        ),
        effector=EffectorState(position=effector),
        monotonic_timestamp=time.monotonic(),
        wall_timestamp=0.0,
        sequence=1,
        mode=ArmMode.HOLD,
    )


class _Reader:
    pixel_format = "bgr8"

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def latest(self) -> np.ndarray:
        return self._frame


def test_server_url_normalization() -> None:
    assert inference.normalize_server_url("10.0.0.5:8202") == "http://10.0.0.5:8202/act"
    assert inference.normalize_server_url("https://example.test/base/act") == (
        "https://example.test/base/act"
    )
    assert inference.normalize_server_url(None).endswith("127.0.0.1:8202/act")


def test_observation_uses_model_camera_and_state_order() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)  # BGR
    observation = inference.build_observation(
        {
            "left": SimpleNamespace(latest_state=_state("left", position=0.1)),
            "right": SimpleNamespace(latest_state=_state("right", position=-0.2)),
        },
        {
            "top": _Reader(frame),
            "left_wrist": _Reader(frame),
            "right_wrist": _Reader(frame),
        },
    )

    assert tuple(observation.top_cam[0, 0]) == (30, 20, 10)
    assert observation.state.shape == (14,)
    np.testing.assert_allclose(observation.state[:6], 0.1)
    assert observation.state[6] == pytest.approx(0.25)
    np.testing.assert_allclose(observation.state[7:13], -0.2)
    assert observation.state[13] == pytest.approx(0.25)


def test_action_split_and_bounded_executor() -> None:
    states = {"left": _state("left"), "right": _state("right")}
    executor = inference.BoundedChunkExecutor(max_step_rad=0.1, max_effector_step=0.2)
    executor.reset(states)
    commands = executor.step(
        np.array([1.0] * 7 + [-1.0] * 7),
        limits={
            "left": (np.full(6, -0.5), np.full(6, 0.5)),
            "right": (np.full(6, -0.5), np.full(6, 0.5)),
        },
    )
    np.testing.assert_allclose(commands["left"].position_rad, 0.1)
    np.testing.assert_allclose(commands["right"].position_rad, -0.1)
    assert commands["left"].effector == pytest.approx(0.45)
    assert commands["right"].effector == pytest.approx(0.05)
    with pytest.raises(ConfigurationError, match=r"shape \(14,"):
        inference.split_action([0.0] * 13)


def test_a_chunk_splits_into_the_same_arms_its_rows_do() -> None:
    """The Viser overlay takes the chunk; the executor takes rows. One layout."""
    actions = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)

    split = inference.split_chunk(actions)

    assert set(split) == {"left", "right"}
    for index, row in enumerate(actions):
        per_row = inference.split_action(row)
        np.testing.assert_allclose(split["left"][index], per_row["left"])
        np.testing.assert_allclose(split["right"][index], per_row["right"])


def test_a_chunk_of_the_wrong_width_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match=r"shape \(N, 14\)"):
        inference.split_chunk(np.zeros((3, 13)))
    with pytest.raises(ConfigurationError, match=r"shape \(N, 14\)"):
        inference.split_chunk(np.zeros(14))


def test_chunk_plan_walks_to_the_action_in_sub_steps() -> None:
    """The opt-in reference executor reaches each action rather than approaching it."""
    states = {"left": _state("left"), "right": _state("right")}
    executor = inference.ReachingChunkExecutor()
    plan = executor.plan(np.array([0.2] * 6 + [0.25] + [0.0] * 6 + [0.25]), states)

    # 0.2 rad of travel at 0.01 rad per sub-step.
    assert len(plan) == 20
    assert executor.clamped == 0
    np.testing.assert_allclose(plan[0]["left"].position_rad, 0.0, atol=1e-9)
    np.testing.assert_allclose(plan[-1]["left"].position_rad, 0.2)
    np.testing.assert_allclose(plan[-1]["right"].position_rad, 0.0, atol=1e-9)
    # A move inside one sub-step is commanded directly rather than split.
    assert len(executor.plan(np.array([0.001] * 6 + [0.25] + [0.0] * 6 + [0.25]), states)) == 1


def test_chunk_plan_clamps_from_the_measured_pose_every_time() -> None:
    """The clamp cannot drift: it is re-anchored on measurement, not on the
    previous command, so a far-away action converges over the chunk."""
    executor = inference.ReachingChunkExecutor(max_joint_step_rad=0.3)
    action = np.array([1.0] * 6 + [1.0] + [1.0] * 6 + [1.0])
    reached = []
    for position in (0.0, 0.3, 0.6):
        states = {
            "left": _state("left", position=position),
            "right": _state("right", position=position),
        }
        reached.append(executor.plan(action, states)[-1]["left"].position_rad[0])
    np.testing.assert_allclose(reached, [0.3, 0.6, 0.9])


def test_start_pose_plan_ends_at_the_training_start_pose() -> None:
    states = {"left": _state("left"), "right": _state("right")}
    plan = inference.start_pose_plan(states)
    np.testing.assert_allclose(
        plan[-1]["left"].position_rad, inference.MOLMOACT_START_JOINTS["left"]
    )
    np.testing.assert_allclose(
        plan[-1]["right"].position_rad, inference.MOLMOACT_START_JOINTS["right"]
    )
    assert plan[-1]["left"].effector == pytest.approx(inference.MOLMOACT_START_EFFECTOR)


class _FakeResponse:
    def __init__(self, body: str, status_code: int = 200) -> None:
        self.text = body
        self.status_code = status_code


class _FakeSession:
    """Stands in for requests.Session, and records that one is reused."""

    def __init__(self, *responses: _FakeResponse) -> None:
        self.responses = list(responses)
        self.gets: list[tuple[str, float]] = []
        self.posts: list[tuple[str, bytes, dict]] = []
        self.closed = False

    def get(self, url, timeout=None):
        self.gets.append((url, timeout))
        return self.responses.pop(0)

    def post(self, url, data=None, headers=None, timeout=None):
        self.posts.append((url, data, headers or {}))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _observation() -> inference.BimanualObservation:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    return inference.BimanualObservation(
        frame, frame, frame, np.zeros(14, dtype=np.float32)
    )


def test_http_client_posts_json_numpy_payload(monkeypatch) -> None:
    payloads = []

    class FakeJsonNumpy:
        @staticmethod
        def dumps(payload):
            payloads.append(payload)
            return "{}"

        @staticmethod
        def loads(_body):
            return {"actions": np.zeros((2, 14), dtype=np.float32), "dt_ms": 180.0}

    monkeypatch.setattr(inference, "_json_numpy", lambda: FakeJsonNumpy)
    session = _FakeSession(
        _FakeResponse('{"status":"ok","norm_tag":"yam_dual_molmoact2","state_dim":14}'),
        _FakeResponse("{}"),
    )

    client = inference.MolmoActClient("127.0.0.1:8202", jpeg_quality=0, session=session)
    client.health()
    actions = client.infer(_observation(), "pick up the object")

    assert actions.shape == (2, 14)
    assert session.gets[0][0] == "http://127.0.0.1:8202/act"
    url, body, headers = session.posts[0]
    assert url == "http://127.0.0.1:8202/act"
    assert headers["Content-Type"] == "application/json"
    assert isinstance(body, bytes)
    payload = payloads[-1]
    assert payload["instruction"] == "pick up the object"
    assert payload["top_cam"].shape == (2, 2, 3)
    # The runtime defaults the reference deployment runs.
    assert payload["num_steps"] == inference.DEFAULT_MOLMOACT_NUM_STEPS
    assert payload["enable_cuda_graph"] is True
    assert payload["normalization_tag"] == inference.MOLMOACT_NORM_TAG
    # dt_ms is what separates a slow GPU from a slow link.
    assert client.last_latency["gpu_s"] == pytest.approx(0.18)
    client.close()
    assert session.closed


def test_health_rejects_a_server_with_another_contract(monkeypatch) -> None:
    session = _FakeSession(_FakeResponse('{"status":"ok","state_dim":8,"num_cameras":1}'))
    client = inference.MolmoActClient(session=session)
    with pytest.raises(inference.InferenceError, match="state_dim"):
        client.health()


def test_frames_go_out_jpeg_encoded_by_default(monkeypatch) -> None:
    pytest.importorskip("PIL")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    encoded = inference.encode_frame(
        frame, jpeg_quality=inference.DEFAULT_MOLMOACT_JPEG_QUALITY
    )
    # A 1-D uint8 array is what the server reads as an encoded image.
    assert encoded.ndim == 1
    assert encoded.dtype == np.uint8
    assert encoded.nbytes < frame.nbytes
    np.testing.assert_array_equal(inference.encode_frame(frame, jpeg_quality=0), frame)


# Verbatim from the two generations of host_server_yam.py's ``_to_pil``. Both
# say "HxWx3", which is why matching that alone cannot tell them apart.
RAW_ONLY_SERVER_ERROR = (
    '{"error": "inference failed: image must be HxWx3, got shape (271814,)"}'
)
CURRENT_SERVER_ERROR = (
    '{"error": "inference failed: image must be HxWx3 (raw) or 1-D uint8 '
    '(encoded), got shape (360, 640, 4)"}'
)


def test_only_a_raw_only_server_reads_as_unable_to_decode_jpeg() -> None:
    assert inference.server_rejects_encoded_frames(RAW_ONLY_SERVER_ERROR)
    assert not inference.server_rejects_encoded_frames(CURRENT_SERVER_ERROR)
    # A corrupt JPEG is Pillow's complaint, not a contract mismatch.
    assert not inference.server_rejects_encoded_frames(
        '{"error": "inference failed: cannot identify image file"}'
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (RAW_ONLY_SERVER_ERROR, inference.EncodedFramesUnsupported),
        (CURRENT_SERVER_ERROR, inference.InferenceError),
    ],
)
def test_a_bad_frame_is_not_mistaken_for_an_out_of_date_server(
    monkeypatch, body, expected
) -> None:
    """Only the raw-only server may cost the run its JPEG transport.

    A current server rejecting a genuinely malformed frame used to trip the
    same branch, so a camera handing over RGBA silently dropped the rest of the
    episode to raw frames -- 3.7x the latency, for a fault raw frames share.
    """

    class FakeJsonNumpy:
        @staticmethod
        def dumps(_payload):
            return "{}"

    monkeypatch.setattr(inference, "_json_numpy", lambda: FakeJsonNumpy)
    client = inference.MolmoActClient(
        jpeg_quality=inference.DEFAULT_MOLMOACT_JPEG_QUALITY,
        session=_FakeSession(_FakeResponse(body, status_code=500)),
    )
    with pytest.raises(expected) as caught:
        client.infer(_observation(), "task")
    # The narrower type must not surface for the current server.
    assert isinstance(caught.value, inference.EncodedFramesUnsupported) == (
        expected is inference.EncodedFramesUnsupported
    )


def test_both_transports_carry_the_same_picture(monkeypatch) -> None:
    """JPEG must not fail on a frame raw transport delivers fine."""
    pytest.importorskip("PIL")
    frame = np.full((8, 8, 3), 300.0, dtype=np.float64)  # over-range, not uint8

    raw = inference.encode_frame(frame, jpeg_quality=0)
    assert raw.dtype == np.uint8 and raw.max() == 255

    encoded = inference.encode_frame(frame, jpeg_quality=95)
    assert encoded.ndim == 1
    np.testing.assert_allclose(
        inference.decode_wire_frame(encoded), raw, atol=2
    )

    with pytest.raises(inference.InferenceError, match="HxWx3"):
        inference.encode_frame(np.zeros((8, 8, 4), dtype=np.uint8), jpeg_quality=95)


def test_http_client_rejects_wrong_action_width(monkeypatch) -> None:
    class FakeJsonNumpy:
        @staticmethod
        def loads(_body):
            return {"actions": np.zeros((3, 13), dtype=np.float32)}

        @staticmethod
        def dumps(_payload):
            return "{}"

    monkeypatch.setattr(inference, "_json_numpy", lambda: FakeJsonNumpy)
    client = inference.MolmoActClient(
        jpeg_quality=0, session=_FakeSession(_FakeResponse("{}"))
    )
    with pytest.raises(inference.InferenceError, match=r"shape \(N, 14\)"):
        client.infer(_observation(), "task")


def test_wire_frames_are_kept_and_decode_to_what_the_policy_saw(monkeypatch) -> None:
    """The Viser panel shows the model's own picture, so the client keeps it."""
    pytest.importorskip("PIL")

    class FakeJsonNumpy:
        @staticmethod
        def dumps(_payload):
            return "{}"

        @staticmethod
        def loads(_body):
            return {"actions": np.zeros((30, 14), dtype=np.float32)}

    monkeypatch.setattr(inference, "_json_numpy", lambda: FakeJsonNumpy)
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 3)).astype(np.uint8)
    observation = inference.BimanualObservation(
        frame, frame, frame, np.zeros(14, dtype=np.float32)
    )
    client = inference.MolmoActClient(session=_FakeSession(_FakeResponse("{}")))
    client.infer(observation, "task")

    # Keyed by the rig's camera names, not the model's payload keys.
    assert sorted(client.last_wire_frames) == ["left_wrist", "right_wrist", "top"]
    assert client.last_wire_frames["top"].ndim == 1  # JPEG on the wire
    served = inference.served_frames(client)
    assert served["top"].shape == frame.shape
    assert served["top"].dtype == np.uint8
    # Decoded from the wire, so it carries the compression the model saw.
    assert not np.array_equal(served["top"], frame)


def test_raw_wire_frames_decode_to_themselves() -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(inference.decode_wire_frame(frame), frame)
    with pytest.raises(inference.InferenceError, match="1-D encoded or HxWx3 raw"):
        inference.decode_wire_frame(np.zeros((4, 4)))


def test_prefetcher_overlaps_one_call_and_tracks_latency() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def infer(self, _observation, _instruction):
            self.calls += 1
            return np.full((30, 14), float(self.calls))

    client = FakeClient()
    prefetcher = inference.ChunkPrefetcher(client)  # type: ignore[arg-type]
    observation = _observation()

    assert not prefetcher.busy
    prefetcher.submit(observation, "task")
    prefetcher.submit(observation, "task")  # a second is a no-op while pending
    chunk = prefetcher.take(observation, "task")

    assert client.calls == 1
    assert prefetcher.used_pending is True
    np.testing.assert_allclose(chunk, 1.0)
    # Falling back to a synchronous call is the no-prefetch path.
    np.testing.assert_allclose(prefetcher.take(observation, "task"), 2.0)
    assert prefetcher.used_pending is False
    prefetcher.close()


def test_prefetcher_reraises_in_the_caller() -> None:
    class FailingClient:
        def infer(self, _observation, _instruction):
            raise inference.InferenceError("server said no")

    prefetcher = inference.ChunkPrefetcher(FailingClient())  # type: ignore[arg-type]
    prefetcher.submit(_observation(), "task")
    with pytest.raises(inference.InferenceError, match="server said no"):
        prefetcher.take(_observation(), "task")


def test_run_infer_commands_and_parks_with_fake_hardware(monkeypatch) -> None:
    class FakePolicy:
        url = "http://policy/act"
        last_latency: dict[str, float] = {}

        def __init__(self):
            self.calls = 0
            self.closed = False

        def health(self):
            return {"status": "ok"}

        def infer(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return np.ones((1, 14), dtype=np.float64) * 0.1
            raise inference.InferenceError("test stop")

        def close(self):
            self.closed = True

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    readers = {
        name: SimpleNamespace(latest=lambda frame=frame: frame, pixel_format="rgb8")
        for name in ("top", "left_wrist", "right_wrist")
    }
    monkeypatch.setattr(cli, "open_inference_cameras", lambda *_args, **_kwargs: readers)
    made = {}

    def factory(rig_arm):
        backend = FakeArmBackend()
        made[rig_arm.name] = backend
        return backend

    policy = FakePolicy()
    status = cli.run_infer(
        resolve_rig("yam_bimanual"),
        instruction="test task",
        visualize=False,
        control_rate_hz=1000.0,
        backend_factory=factory,
        policy=policy,
    )

    assert status == 1
    assert all(backend.commands for backend in made.values())
    assert all(backend.closes[0] is True for backend in made.values())
    assert all(not backend.connected for backend in made.values())
    assert policy.closed


def test_speed_spends_more_ticks_on_the_same_chunk(monkeypatch) -> None:
    """--speed 0.5 commands a four-action chunk over seven ticks, not four.

    Counted as a difference between two otherwise identical runs, so the park
    at teardown -- which both pay -- cancels out.
    """

    def run(speed: float) -> int:
        class FakePolicy:
            url = "http://policy/act"
            last_latency: dict[str, float] = {}

            def __init__(self) -> None:
                self.calls = 0

            def health(self) -> dict[str, str]:
                return {"status": "ok"}

            def infer(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return np.stack([np.full(14, 0.01 * i) for i in range(4)])
                raise inference.InferenceError("test stop")

            def close(self) -> None:
                pass

        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        readers = {
            name: SimpleNamespace(latest=lambda frame=frame: frame, pixel_format="rgb8")
            for name in ("top", "left_wrist", "right_wrist")
        }
        monkeypatch.setattr(cli, "open_inference_cameras", lambda *_a, **_k: readers)
        made: dict[str, FakeArmBackend] = {}

        def factory(rig_arm):
            backend = FakeArmBackend()
            made[rig_arm.name] = backend
            return backend

        cli.run_infer(
            resolve_rig("yam_bimanual"),
            instruction="test task",
            visualize=False,
            control_rate_hz=1000.0,
            speed=speed,
            backend_factory=factory,
            policy=FakePolicy(),
        )
        return len(made["left"].commands)

    assert run(0.5) - run(1.0) == 3


def test_run_infer_reset_is_the_first_thing_commanded(monkeypatch) -> None:
    """The start-pose ramp runs before any policy action reaches the arms."""

    class FakePolicy:
        url = "http://policy/act"
        last_latency: dict[str, float] = {}

        def health(self):
            return {"status": "ok"}

        def infer(self, *_args, **_kwargs):
            raise inference.InferenceError("stop before the first chunk")

        def close(self):
            return None

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    readers = {
        name: SimpleNamespace(latest=lambda frame=frame: frame, pixel_format="rgb8")
        for name in ("top", "left_wrist", "right_wrist")
    }
    monkeypatch.setattr(cli, "open_inference_cameras", lambda *_args, **_kwargs: readers)
    made = {}

    def factory(rig_arm):
        backend = FakeArmBackend()
        made[rig_arm.name] = backend
        return backend

    cli.run_infer(
        resolve_rig("yam_bimanual"),
        instruction="test task",
        visualize=False,
        reset_start_pose=True,
        backend_factory=factory,
        policy=FakePolicy(),
    )

    # The inference call fails, so every command seen came from the ramp, and
    # the last one is the pose the training demonstrations start from.
    for name, backend in made.items():
        assert backend.commands
        np.testing.assert_allclose(
            backend.commands[-1].position_rad,
            inference.MOLMOACT_START_JOINTS[name],
            atol=1e-6,
        )


def test_time_scale_stretches_a_chunk_without_shortening_its_path() -> None:
    """Half speed is twice the ticks over the same path, ends included.

    This is the property that separates ``--speed`` from a tighter clamp: the
    clamp files each action down and moves on, so the chunk is executed as a
    shrunken copy of itself, while time-scaling spends longer on every
    millimetre of the original.
    """
    chunk = np.stack([np.full(14, float(i)) for i in range(4)])
    scaled = inference.time_scale(chunk, 0.5)

    assert len(scaled) == 7
    np.testing.assert_allclose(scaled[0], chunk[0])
    np.testing.assert_allclose(scaled[-1], chunk[-1])
    # Every odd tick lands halfway between two of the policy's own actions.
    np.testing.assert_allclose(scaled[1], np.full(14, 0.5))
    np.testing.assert_allclose(scaled[3], np.full(14, 1.5))
    # And the total distance travelled is unchanged.
    assert float(np.abs(np.diff(scaled, axis=0)).sum()) == pytest.approx(
        float(np.abs(np.diff(chunk, axis=0)).sum())
    )


def test_time_scale_at_full_speed_is_the_chunk_itself() -> None:
    chunk = np.stack([np.full(14, float(i)) for i in range(4)])
    np.testing.assert_array_equal(inference.time_scale(chunk, 1.0), chunk)
    with pytest.raises(ConfigurationError):
        inference.time_scale(chunk, 0.0)


def test_carry_targets_keeps_the_joint_integrator_across_chunks() -> None:
    """With --carry-targets a chunk boundary is not a discontinuity.

    The default pulls the joint target back to the measured pose every chunk,
    so an arm running behind has its command re-seeded to the lag and then
    re-accelerates -- a sawtooth in commanded velocity once a chunk. Carrying
    the target makes the boundary invisible to the arm.
    """
    states = {"left": _state("left"), "right": _state("right")}

    default = inference.BoundedChunkExecutor(max_step_rad=0.1, max_effector_step=0.2)
    default.reset(states)
    default.step(np.ones(14))
    # The measured pose is still 0.0 -- the arm has not followed -- so the
    # default gives the commanded 0.1 back on the next chunk.
    default.reset(states)
    np.testing.assert_allclose(default.targets["left"][:6], 0.0)

    carried = inference.BoundedChunkExecutor(
        max_step_rad=0.1, max_effector_step=0.2, carry_targets=True
    )
    carried.reset(states)
    carried.step(np.ones(14))
    carried.reset(states)
    np.testing.assert_allclose(carried.targets["left"][:6], 0.1)


def test_command_lag_is_the_worst_joint_gap_to_the_measured_pose() -> None:
    """The number that says the hardware, not the clamp, is the limit."""
    states = {"left": _state("left", position=0.0), "right": _state("right", position=0.0)}
    commands = {
        "left": PositionCommand(np.full(6, 0.05)),
        "right": PositionCommand(np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])),
    }
    assert inference.command_lag(commands, states) == pytest.approx(0.2)


def test_bounded_executor_counts_the_actions_it_files_down() -> None:
    """The clamp count is per action, and only rises when the limit bites."""
    states = {"left": _state("left"), "right": _state("right")}
    executor = inference.BoundedChunkExecutor(max_step_rad=0.1, max_effector_step=0.2)
    executor.reset(states)
    assert executor.clamped == 0

    executor.step(np.array([1.0] * 14))
    assert executor.clamped == 1

    # Commanding the target the arm is already walking to asks for no move at
    # all, so nothing is filed down and the count holds.
    executor.step(np.concatenate([executor.targets["left"], executor.targets["right"]]))
    assert executor.clamped == 1


def test_run_infer_drops_to_raw_frames_when_the_server_cannot_decode(
    monkeypatch, capsys
) -> None:
    """A server that rejects JPEG costs the payload format, not the run."""

    class FakePolicy:
        url = "http://policy/act"
        last_latency: dict[str, float] = {}

        def __init__(self) -> None:
            self.jpeg_quality = inference.DEFAULT_MOLMOACT_JPEG_QUALITY
            self.qualities: list[int] = []
            self.calls = 0

        def health(self):
            return {"status": "ok"}

        def infer(self, *_args, **_kwargs):
            self.calls += 1
            self.qualities.append(self.jpeg_quality)
            if self.calls == 1:
                raise inference.EncodedFramesUnsupported("image must be HxWx3")
            if self.calls == 2:
                return np.ones((1, 14), dtype=np.float64) * 0.1
            raise inference.InferenceError("test stop")

        def close(self):
            return None

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    readers = {
        name: SimpleNamespace(latest=lambda frame=frame: frame, pixel_format="rgb8")
        for name in ("top", "left_wrist", "right_wrist")
    }
    monkeypatch.setattr(cli, "open_inference_cameras", lambda *_args, **_kwargs: readers)
    made = {}

    def factory(rig_arm):
        backend = FakeArmBackend()
        made[rig_arm.name] = backend
        return backend

    policy = FakePolicy()
    cli.run_infer(
        resolve_rig("yam_bimanual"),
        instruction="test task",
        visualize=False,
        control_rate_hz=1000.0,
        backend_factory=factory,
        policy=policy,
    )

    # Encoded once, then raw for good: the retry and every later call.
    assert policy.qualities == [inference.DEFAULT_MOLMOACT_JPEG_QUALITY, 0, 0]
    assert "raw frames" in capsys.readouterr().err
    assert all(backend.commands for backend in made.values())


def test_infer_flags_reach_the_runtime(monkeypatch) -> None:
    """Every documented infer flag lands on run_infer, not only in the docs."""
    seen: dict[str, object] = {}

    def fake_run_infer(_rig, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_infer", fake_run_infer)
    assert cli.main(["infer", "--instruction", "fold the cloth", "--skip-preflight"]) == 0
    # The wire defaults are the reference deployment's.
    assert seen["jpeg_quality"] == inference.DEFAULT_MOLMOACT_JPEG_QUALITY
    assert seen["enable_cuda_graph"] is True
    assert seen["request_timeout_s"] == inference.DEFAULT_REQUEST_TIMEOUT_S
    # The reference *runtime* is opt-in, because it behaved worse on this rig.
    assert seen["reach_actions"] is False
    assert seen["reset_start_pose"] is False
    # Except prefetching, which is on: it changes only when the next call is
    # made, and inferring between chunks parks the arms for a whole round trip
    # at every chunk boundary.
    assert seen["prefetch"] is True
    # The gripper's clamp is looser than the joints', so a full-stroke command
    # lands in four ticks rather than ten.
    assert seen["max_step_rad"] == inference.DEFAULT_MAX_STEP_RAD
    assert seen["max_effector_step"] == inference.DEFAULT_MAX_EFFECTOR_STEP
    assert inference.DEFAULT_MAX_EFFECTOR_STEP > inference.DEFAULT_MAX_STEP_RAD
    # Chunks play at the rate they were recorded at until asked otherwise.
    assert seen["speed"] == inference.DEFAULT_CHUNK_SPEED
    assert seen["carry_targets"] is False

    seen.clear()
    assert (
        cli.main(
            [
                "infer",
                "--instruction", "fold the cloth",
                "--skip-preflight",
                "--raw-frames",
                "--no-cuda-graph",
                "--reach-actions",
                "--max-step-rad", "0.3",
                "--max-effector-step", "0.5",
                "--no-prefetch",
                "--prefetch-margin-s", "0.2",
                "--reset-start-pose",
                "--speed", "0.5",
                "--carry-targets",
            ]
        )
        == 0
    )
    assert seen["jpeg_quality"] == 0
    assert seen["enable_cuda_graph"] is False
    assert seen["reach_actions"] is True
    assert seen["max_step_rad"] == 0.3
    assert seen["max_effector_step"] == 0.5
    assert seen["prefetch"] is False
    assert seen["prefetch_margin_s"] == 0.2
    assert seen["reset_start_pose"] is True
    assert seen["speed"] == 0.5
    assert seen["carry_targets"] is True


def test_run_infer_routes_chunks_through_the_prefetcher(monkeypatch) -> None:
    """--prefetch puts every inference on the worker, off the control thread.

    That the worker actually overlaps a call with arm motion is
    :func:`test_prefetcher_overlaps_one_call_and_tracks_latency`; what matters
    here is that ``run_infer`` wires the loop through it at all.
    """

    class FakePolicy:
        url = "http://policy/act"
        last_latency: dict[str, float] = {}
        jpeg_quality = 0

        def __init__(self) -> None:
            self.threads: list[str] = []
            self.calls = 0

        def health(self):
            return {"status": "ok"}

        def infer(self, *_args, **_kwargs):
            self.calls += 1
            self.threads.append(threading.current_thread().name)
            if self.calls <= 2:
                return np.zeros((1, 14), dtype=np.float64)
            raise inference.InferenceError("test stop")

        def close(self):
            return None

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    readers = {
        name: SimpleNamespace(latest=lambda frame=frame: frame, pixel_format="rgb8")
        for name in ("top", "left_wrist", "right_wrist")
    }
    monkeypatch.setattr(cli, "open_inference_cameras", lambda *_args, **_kwargs: readers)
    policy = FakePolicy()
    cli.run_infer(
        resolve_rig("yam_bimanual"),
        instruction="test task",
        visualize=False,
        control_rate_hz=1000.0,
        prefetch=True,
        backend_factory=lambda _rig_arm: FakeArmBackend(),
        policy=policy,
    )

    assert policy.calls >= 3  # two chunks executed, then the stop
    assert all(name.startswith("molmoact-prefetch") for name in policy.threads)


def test_reset_carries_the_commanded_gripper_across_chunks() -> None:
    """Joints re-seed from measurement every chunk; the gripper target does not.

    A gripper holding an object always reads short of the value it was
    commanded, so re-seeding the effector target from the measurement would
    hand back a slice of the grip on every chunk boundary.
    """
    states = {"left": _state("left"), "right": _state("right")}
    executor = inference.BoundedChunkExecutor(max_step_rad=0.1, max_effector_step=0.2)
    executor.reset(states)
    # The first reset has nothing to carry, so it seeds from the measurement.
    assert executor.targets["left"][-1] == pytest.approx(0.25)

    executor.step(np.zeros(14))
    commanded = executor.targets["left"][-1]
    assert commanded == pytest.approx(0.05)

    # The measured gripper is still 0.25 (it did not follow), but the next
    # chunk resumes from what was actually commanded, not from the lag.
    executor.reset(states)
    assert executor.targets["left"][-1] == pytest.approx(commanded)
    np.testing.assert_allclose(executor.targets["left"][:6], 0.0)


def test_gripper_watch_flags_a_gripper_that_never_moves() -> None:
    """A swept command with a pinned measurement is a gripper that is inert."""
    watch = inference.GripperWatch()
    states = {"left": _state("left"), "right": _state("right")}
    for value in (1.0, 0.75, 0.5, 0.25, 0.0):
        commands = {
            name: PositionCommand(np.zeros(6), effector=value) for name in ("left", "right")
        }
        watch.observe(commands, states)
    # _state pins both grippers at 0.25 however far the command sweeps.
    assert watch.stalled() == ["left", "right"]
    assert "l0.00/0.25" in watch.render()

    # A gripper that follows its command is not a stall.
    following = inference.GripperWatch()
    for value in (1.0, 0.5, 0.0):
        following.observe(
            {"left": PositionCommand(np.zeros(6), effector=value)},
            {"left": _state("left", effector=value)},
        )
    assert following.stalled() == []


def test_gripper_watch_ignores_a_gripper_holding_an_object() -> None:
    """Clamped on something is not inert: it had to travel to get there."""
    watch = inference.GripperWatch()
    # Close onto the object -- the measurement follows until the jaws meet it.
    for value in (1.0, 0.8, 0.6, 0.45, 0.35, 0.30):
        watch.observe(
            {"left": PositionCommand(np.zeros(6), effector=value)},
            {"left": _state("left", effector=value)},
        )
    # Then hold: commanded shut, measured stuck on the object for a long time.
    for _ in range(200):
        watch.observe(
            {"left": PositionCommand(np.zeros(6), effector=0.0)},
            {"left": _state("left", effector=0.30)},
        )
    assert watch.stalled() == []


def test_gripper_watch_needs_a_real_sweep_before_it_accuses() -> None:
    """A gripper the policy has barely touched is not evidence of anything."""
    watch = inference.GripperWatch()
    for value in (0.30, 0.32, 0.29):
        watch.observe(
            {"left": PositionCommand(np.zeros(6), effector=value)},
            {"left": _state("left")},
        )
    assert watch.stalled() == []


def test_gripper_watch_catches_a_slow_walk_no_single_chunk_would_show() -> None:
    """The regression that let this run a whole session unreported.

    The policy walked the gripper open in steps far smaller than any per-chunk
    threshold, so a window that reset each chunk never saw a sweep worth
    flagging. Cumulatively the command covers most of the range.
    """
    watch = inference.GripperWatch()
    for step in range(40):
        watch.observe(
            {"left": PositionCommand(np.zeros(6), effector=step * 0.02)},
            {"left": _state("left", effector=0.0)},
        )
    assert watch.stalled() == ["left"]
