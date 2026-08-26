"""The record loop and its schema: no hardware, no headset, no lerobot.

That the whole session is testable is the point of the three seams in
``openpi_control.record`` -- a recording run is the one thing here you cannot
rehearse cheaply on real arms, so the episode state machine, the stall handling,
and the gripper polarity all have to be pinned down here instead.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fake_arm_backend import FakeArmBackend

from openpi_control.exceptions import ConfigurationError
from openpi_control.record import (
    DATASET_GRIPPER_CLOSED,
    DATASET_GRIPPER_OPEN,
    NATIVE_GRIPPER_OPEN,
    ArmTarget,
    EpisodeControlHints,
    EpisodeEvent,
    HoldSource,
    MemorySink,
    ScriptedSource,
    TeleopStep,
    action_vector,
    arm_feature_names,
    build_features,
    camera_shapes,
    hold_targets,
    needs_rgb_conversion,
    observation_vector,
    record_session,
    rig_robot_type,
    to_dataset_gripper,
    to_native_gripper,
    to_rgb,
)
from openpi_control.rigs import resolve_rig
from openpi_control.types import (
    ArmCapabilities,
    ArmMode,
    ArmRole,
    ArmState,
    EffectorState,
    JointState,
)

FAST = {"fps": 1000}  # keep the loop's sleeps out of the test runtime


class FakeArm:
    """The seam the record loop actually depends on: capabilities, state, command.

    Deliberately not a FollowerArm -- the loop is documented to accept anything
    with ``latest_state`` and ``command``, and one test below proves a real
    session satisfies that.
    """

    def __init__(self, *, dof: int = 6, position: float = 0.25, effector: float | None = 1.0):
        self.capabilities = ArmCapabilities(
            protocol_version=(1, 0),
            model="Yam",
            joint_names=tuple(f"joint_{i + 1}" for i in range(dof)),
            has_effector=effector is not None,
            supports_direct_commands=True,
            supports_live_input=True,
            supports_gravity_compensation=True,
            supports_force_feedback=False,
            supports_move_to_ready=True,
        )
        self.dof = dof
        self.position = position
        self.effector = effector
        self.age_s = 0.0
        self.publishing = True
        self.commands: list = []

    @property
    def latest_state(self) -> ArmState | None:
        if not self.publishing:
            return None
        return ArmState(
            name="fake",
            role=ArmRole.FOLLOWER,
            joints=JointState(
                names=self.capabilities.joint_names,
                position_rad=[self.position] * self.dof,
                velocity_rad_s=[0.0] * self.dof,
                effort_nm=[0.0] * self.dof,
                temperature_c=[25.0] * self.dof,
                current_a=[0.0] * self.dof,
            ),
            effector=EffectorState(position=self.effector) if self.effector is not None else None,
            monotonic_timestamp=time.monotonic() - self.age_s,
            wall_timestamp=0.0,
            sequence=1,
            mode=ArmMode.HOLD,
        )

    def command(self, command) -> None:
        self.commands.append(command)


class FakeCamera:
    """Returns a scripted sequence of frames, one per ``latest()`` call.

    Unlike a real reader -- whose ``latest()`` is idempotent until a new frame
    lands -- this consumes an entry per call, which makes "the grab failed on
    this tick" expressible. It also means the loop's camera warm-up poll spends
    the first entry, so a script should lead with the frame it wants recorded.
    """

    pixel_format = "bgr8"

    def __init__(self, frames: list[np.ndarray | None]):
        self._frames = list(frames)
        self.index = 0

    def latest(self) -> np.ndarray | None:
        if self.index < len(self._frames):
            frame = self._frames[self.index]
            self.index += 1
            return frame
        return None

    def wait_for_frame(self, timeout_s: float = 5.0) -> np.ndarray | None:
        del timeout_s
        return self._frames[0] if self._frames else None


def target(dof: int = 6, value: float = 0.5, effector: float | None = None) -> ArmTarget:
    return ArmTarget(position_rad=tuple([value] * dof), effector=effector)


def bgr(height: int = 2, width: int = 3) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 10  # B
    frame[..., 1] = 20  # G
    frame[..., 2] = 30  # R
    return frame


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_state_names_are_joints_then_gripper_per_arm() -> None:
    # A single-arm subset has to be a strict prefix of the bimanual layout, or
    # the two are different schemas that look alike.
    names = arm_feature_names(["left", "right"], {"left": 6, "right": 6})

    assert names[:7] == [
        "left_joint_1",
        "left_joint_2",
        "left_joint_3",
        "left_joint_4",
        "left_joint_5",
        "left_joint_6",
        "left_gripper",
    ]
    assert names[7:] == [n.replace("left", "right") for n in names[:7]]
    assert arm_feature_names(["left"], {"left": 6}) == names[:7]


def test_cameras_are_video_features_not_image_features() -> None:
    # `image` features land as a directory of PNGs per episode; `video` lands as
    # one mp4. The difference is about two orders of magnitude on disk.
    features = build_features(["left_gripper"], {"top": (480, 848, 3)})

    assert features["observation.images.top"]["dtype"] == "video"
    assert features["observation.images.top"]["shape"] == (480, 848, 3)
    assert features["observation.state"]["shape"] == (1,)
    assert features["action"]["names"] == ["left_gripper"]


def test_camera_shapes_come_from_a_real_frame() -> None:
    # A rotated camera emits the transposed shape, so a schema built from the
    # requested capture size would disagree with the frames.
    shapes = camera_shapes({"top": FakeCamera([bgr(480, 848)])})

    assert shapes == {"top": (480, 848, 3)}


def test_a_camera_that_never_delivers_is_refused_not_skipped() -> None:
    # Recording without it would silently leave that view out of the dataset.
    with pytest.raises(ConfigurationError, match="no frame"):
        camera_shapes({"top": FakeCamera([])}, warmup_s=0.01)


def test_robot_type_distinguishes_a_one_armed_subset() -> None:
    rig = resolve_rig("yam_bimanual")

    assert rig_robot_type(rig) == "yam_bimanual"
    assert rig_robot_type(rig.subset(["right"])) == "yam_bimanual_right"


# --------------------------------------------------------------------------- #
# gripper polarity
# --------------------------------------------------------------------------- #


def test_the_two_gripper_conventions_are_inverses() -> None:
    # This package: 1.0 is open. LeRobot: 0.0 is open. Getting it backwards
    # makes the trigger open the gripper, and makes a recorded dataset teach the
    # inverse of what the operator did.
    assert to_dataset_gripper(NATIVE_GRIPPER_OPEN) == DATASET_GRIPPER_OPEN
    assert to_dataset_gripper(0.0) == DATASET_GRIPPER_CLOSED
    assert to_native_gripper(DATASET_GRIPPER_OPEN) == NATIVE_GRIPPER_OPEN
    for value in (0.0, 0.25, 0.5, 1.0):
        assert to_native_gripper(to_dataset_gripper(value)) == pytest.approx(value)


def test_an_open_gripper_is_recorded_as_zero() -> None:
    arm = FakeArm(effector=NATIVE_GRIPPER_OPEN)

    observation = observation_vector(["left"], {"left": arm.latest_state}, {"left": 6})

    assert observation[-1] == pytest.approx(DATASET_GRIPPER_OPEN)


def test_a_commanded_gripper_is_recorded_in_dataset_units() -> None:
    # The target is native (1 = open); the column must come out inverted.
    observation = np.zeros(7, dtype=np.float32)

    action = action_vector(["left"], {"left": target(effector=1.0)}, observation, {"left": 6})

    assert action[-1] == pytest.approx(DATASET_GRIPPER_OPEN)


# --------------------------------------------------------------------------- #
# observation and action vectors
# --------------------------------------------------------------------------- #


def test_an_uncommanded_arm_records_where_it_already_is() -> None:
    # A zero action would read as a command to fold the arm up; "stay put" is
    # the only honest action for a tick nobody commanded.
    arm = FakeArm(position=0.4, effector=NATIVE_GRIPPER_OPEN)
    states = {"left": arm.latest_state}
    observation = observation_vector(["left"], states, {"left": 6})

    action = action_vector(["left"], {}, observation, {"left": 6})

    assert action == pytest.approx(observation)


def test_an_uncommanded_gripper_holds_while_the_joints_move() -> None:
    arm = FakeArm(position=0.4, effector=0.25)
    observation = observation_vector(["left"], {"left": arm.latest_state}, {"left": 6})

    action = action_vector(
        ["left"], {"left": target(value=0.9, effector=None)}, observation, {"left": 6}
    )

    assert action[:6] == pytest.approx([0.9] * 6)
    assert action[6] == pytest.approx(observation[6])


def test_a_target_with_the_wrong_joint_count_is_refused() -> None:
    # Silently padding or truncating would command an arm from half a pose.
    with pytest.raises(ConfigurationError, match="commanded 3 joints"):
        action_vector(["left"], {"left": target(dof=3)}, np.zeros(7, np.float32), {"left": 6})


def test_a_gripper_target_outside_the_range_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="normalized"):
        ArmTarget(position_rad=(0.0,), effector=1.5)


def test_hold_targets_carry_the_native_gripper_reading() -> None:
    arm = FakeArm(position=0.3, effector=0.75)

    targets = hold_targets({"left": arm.latest_state}, {"left": 6})

    assert targets["left"].position_rad == pytest.approx((0.3,) * 6)
    # Native, not dataset units: this goes straight into a PositionCommand.
    assert targets["left"].effector == pytest.approx(0.75)


def test_hold_targets_skip_an_arm_that_has_not_reported() -> None:
    arm = FakeArm()
    arm.publishing = False

    assert hold_targets({"left": arm.latest_state}, {"left": 6}) == {}


# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #


def test_frames_are_converted_from_bgr_to_rgb() -> None:
    # LeRobot treats a 3-channel frame as RGB; cameras give BGR. Skipping this
    # yields a dataset where every red object is blue.
    converted = to_rgb(bgr())

    assert converted[0, 0].tolist() == [30, 20, 10]
    assert converted.flags["C_CONTIGUOUS"]  # video encoders need contiguous memory


# --------------------------------------------------------------------------- #
# the episode state machine
# --------------------------------------------------------------------------- #


def run(steps, *, arms=None, cameras=None, sink=None, **kwargs):
    arms = arms if arms is not None else {"left": FakeArm()}
    sink = sink if sink is not None else MemorySink()
    source = ScriptedSource(steps)
    result = record_session(
        arms=arms,
        source=source,
        sink=sink,
        cameras=cameras,
        task="fold the towel",
        report=lambda message: None,
        **{**FAST, **kwargs},
    )
    return result, sink, arms


def test_only_frames_between_start_and_save_are_written() -> None:
    # Teleop keeps driving between takes -- that is how the scene gets reset --
    # so ticks outside an episode must command the arm and write nothing.
    result, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}),  # before: not recorded
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
            TeleopStep(targets={"left": target()}),  # after: not recorded
        ]
    )

    assert sink.num_episodes == 1
    # The START tick is part of the take; the SAVE tick closes it and is not.
    assert len(sink.episodes[0]) == 2
    assert result.frames == 2
    assert result.discarded == 0


def test_every_frame_carries_the_task() -> None:
    # Relabelling afterwards means rewriting the dataset.
    _, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ]
    )

    assert {frame["task"] for frame in sink.episodes[0]} == {"fold the towel"}


def test_arms_are_commanded_every_tick_including_between_episodes() -> None:
    _, _, arms = run([TeleopStep(targets={"left": target(value=0.7, effector=0.2)})] * 3)

    assert len(arms["left"].commands) == 3
    command = arms["left"].commands[0]
    assert list(command.position_rad) == pytest.approx([0.7] * 6)
    # Native units reach the arm untouched -- the inversion is only for the file.
    assert command.effector == pytest.approx(0.2)


def test_restarting_a_take_discards_the_open_one() -> None:
    # Right B means "start this take"; a botched one is redone with one press.
    result, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ]
    )

    assert result.discarded == 1
    assert sink.num_episodes == 1
    assert len(sink.episodes[0]) == 1  # only the restart tick; SAVE closes it


def test_quest_episode_messages_are_numbered_and_actionable() -> None:
    messages: list[str] = []
    sink = MemorySink()

    record_session(
        arms={"left": FakeArm()},
        source=ScriptedSource(
            [
                TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
                TeleopStep(targets={"left": target()}),
                TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
                TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
            ]
        ),
        sink=sink,
        task="t",
        report=messages.append,
        episode_controls=EpisodeControlHints(start="Right B", save="Left Y"),
        **FAST,
    )

    assert messages[0] == "▶  EPISODE 1 STARTED — RECORDING"
    assert messages[1] == "   Left Y saves · Right B restarts this episode"
    assert any(message.startswith("↻  EPISODE 1 RESTARTED") for message in messages)
    assert any(message.startswith("✓  EPISODE 1 SAVED — 1 frame · ") for message in messages)
    assert messages[-1] == "○  READY FOR EPISODE 2 — press Right B to start"


def test_save_while_idle_tells_the_operator_how_to_start() -> None:
    messages: list[str] = []

    record_session(
        arms={"left": FakeArm()},
        source=ScriptedSource([TeleopStep(event=EpisodeEvent.SAVE)]),
        sink=MemorySink(),
        task="t",
        report=messages.append,
        episode_controls=EpisodeControlHints(start="Right B", save="Left Y"),
        **FAST,
    )

    assert messages == ["!  SAVE IGNORED — no episode is recording; press Right B to start"]


def test_saving_an_empty_buffer_leaves_the_episode_open() -> None:
    # LeRobot raises on an empty buffer, and a save pressed before any frame
    # landed is a fumble, not the end of a take. The buffer is genuinely empty
    # here because the only tick that could have written one was stalled.
    arm = FakeArm()

    class StallFirst(ScriptedSource):
        def poll(self, states):
            arm.age_s = 5.0 if self.polls == 0 else 0.0
            return super().poll(states)

    sink = MemorySink()
    result = record_session(
        arms={"left": arm},
        source=StallFirst(
            [
                TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
                TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
                TeleopStep(targets={"left": target()}),
                TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
            ]
        ),
        sink=sink,
        task="t",
        report=lambda message: None,
        max_stall_s=1.0,
        **FAST,
    )

    # The premature save was ignored, the take stayed open, and the second save
    # -- with a frame in the buffer by then -- is the one that landed.
    assert result.stalled_ticks == 1
    assert sink.num_episodes == 1
    assert len(sink.episodes[0]) == 1


def test_a_save_while_idle_does_nothing() -> None:
    result, sink, _ = run([TeleopStep(event=EpisodeEvent.SAVE)] * 2)

    assert sink.num_episodes == 0
    assert result.discarded == 0


def test_discard_throws_the_take_away_without_starting_another() -> None:
    result, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.DISCARD),
            TeleopStep(targets={"left": target()}),
        ]
    )

    assert sink.num_episodes == 0
    assert result.discarded == 1
    assert result.frames == 0


def test_the_session_ends_after_the_requested_episode_count() -> None:
    steps = []
    for _ in range(3):
        steps += [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ]

    result, sink, _ = run(steps, num_episodes=2)

    assert sink.num_episodes == 2
    assert "2 episode(s)" in result.ended_by


def test_an_open_episode_is_discarded_when_the_session_ends() -> None:
    # A partial take stops mid-motion; saving it would put that in the dataset.
    result, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}),
            TeleopStep(event=EpisodeEvent.STOP),
        ]
    )

    assert sink.num_episodes == 0
    assert result.discarded == 1
    assert sink.finalized


def test_an_interrupted_episode_can_be_saved_as_partial_data() -> None:
    """Rollout recording keeps frames captured before Ctrl-C."""
    sink = MemorySink()
    source = ScriptedSource(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}),
        ]
    )
    original_poll = source.poll

    def interrupt_after_two_polls(states):
        if source.polls >= 2:
            raise KeyboardInterrupt
        return original_poll(states)

    source.poll = interrupt_after_two_polls  # type: ignore[method-assign]
    result = record_session(
        arms={"left": FakeArm()},
        source=source,
        sink=sink,
        task="partial rollout",
        report=lambda message: None,
        save_on_interrupt=True,
        **FAST,
    )

    assert result.ended_by == "interrupted"
    assert result.discarded == 0
    assert sink.num_episodes == 1
    assert len(sink.episodes[0]) == 2


def test_the_dataset_is_finalized_even_when_nothing_was_recorded() -> None:
    _, sink, _ = run([TeleopStep(event=EpisodeEvent.STOP)])

    assert sink.finalized


def test_a_stop_event_set_externally_ends_the_session() -> None:
    stop = threading.Event()
    stop.set()

    result, sink, _ = run([TeleopStep(targets={"left": target()})], stop=stop)

    assert result.frames == 0
    assert sink.finalized


def test_a_passive_tick_observer_sees_the_same_states_as_the_recorder() -> None:
    arm = FakeArm()
    seen = []

    run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ],
        arms={"left": arm},
        on_tick=lambda states: seen.append(states["left"]),
    )

    # The scripted source contributes one final STOP iteration, which is still
    # a real preview tick even though it writes no dataset frame.
    assert len(seen) == 3
    assert all(state is not None for state in seen)


def test_a_passive_tick_observer_keeps_previewing_during_an_arm_stall() -> None:
    arm = FakeArm()
    arm.age_s = 5.0
    seen = []

    run(
        [TeleopStep()] * 3,
        arms={"left": arm},
        on_tick=lambda states: seen.append(states["left"]),
        max_stall_s=10.0,
    )

    assert len(seen) == 4  # three scripted iterations plus the final STOP


# --------------------------------------------------------------------------- #
# stalls
# --------------------------------------------------------------------------- #


def test_a_stalled_arm_is_not_recorded_as_a_still_one() -> None:
    # The node republishes its last cached pose when CAN frames stop, so writing
    # through a stall teaches a policy the arm was motionless while the operator
    # was moving it.
    arm = FakeArm()
    arm.age_s = 5.0  # every state is far too old

    result, sink, _ = run(
        [TeleopStep(targets={"left": target()}, event=EpisodeEvent.START)]
        + [TeleopStep(targets={"left": target()})] * 5,
        arms={"left": arm},
        max_stall_s=10.0,  # long enough that the episode is not discarded here
    )

    assert result.frames == 0
    assert result.stalled_ticks == 6
    assert not arm.commands  # a stalled arm is not commanded either


def test_a_long_stall_discards_the_open_episode() -> None:
    arm = FakeArm()

    result, sink, _ = run(
        [TeleopStep(targets={"left": target()}, event=EpisodeEvent.START)]
        + [TeleopStep(targets={"left": target()})] * 6,
        arms={"left": arm},
        max_stall_s=0.003,  # 3 ticks at fps=1000
    )

    assert result.discarded == 1
    assert sink.num_episodes == 0


def test_an_arm_that_never_reported_counts_as_stalled() -> None:
    arm = FakeArm()
    arm.publishing = False

    result, _, _ = run([TeleopStep(targets={"left": target()})] * 3, arms={"left": arm})

    assert result.stalled_ticks == 3


def test_a_recovered_arm_resumes_recording() -> None:
    # A momentary stall must not end the take: only a sustained one does.
    arm = FakeArm()
    source_steps = [
        TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
        TeleopStep(targets={"left": target()}),
        TeleopStep(targets={"left": target()}),
        TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
    ]

    class Flaky(ScriptedSource):
        def poll(self, states):
            # Stale for exactly one tick, in the middle of the take.
            arm.age_s = 5.0 if self.polls == 1 else 0.0
            return super().poll(states)

    sink = MemorySink()
    result = record_session(
        arms={"left": arm},
        source=Flaky(source_steps),
        sink=sink,
        task="t",
        report=lambda message: None,
        max_stall_s=1.0,
        **FAST,
    )

    assert result.stalled_ticks == 1
    assert sink.num_episodes == 1
    # Four ticks: START writes, one stalls, one writes, SAVE closes.
    assert len(sink.episodes[0]) == 2


# --------------------------------------------------------------------------- #
# cameras in the loop
# --------------------------------------------------------------------------- #


def test_camera_frames_land_under_their_own_key_in_rgb() -> None:
    _, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ],
        cameras={"top": FakeCamera([bgr(), bgr()])},
    )

    frame = sink.episodes[0][0]
    assert "observation.images.top" in frame
    assert frame["observation.images.top"][0, 0].tolist() == [30, 20, 10]


def test_a_dropped_grab_repeats_the_last_frame() -> None:
    # LeRobot needs every feature on every frame, so a camera that misses one
    # grab must not drop out of the schema mid-episode.
    # Leading frame is spent by the warm-up poll; the loop then sees one good
    # frame followed by two failed grabs.
    first = bgr()
    camera = FakeCamera([first, first, None, None])

    _, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ],
        cameras={"top": camera},
    )

    assert all("observation.images.top" in frame for frame in sink.episodes[0])


# --------------------------------------------------------------------------- #
# misuse
# --------------------------------------------------------------------------- #


def test_a_source_that_names_an_unknown_arm_is_an_error() -> None:
    with pytest.raises(ConfigurationError, match="unknown arm"):
        run([TeleopStep(targets={"rihgt": target()})])


def test_recording_needs_at_least_one_arm() -> None:
    with pytest.raises(ConfigurationError, match="at least one arm"):
        run([TeleopStep()], arms={})


def test_a_nonpositive_frame_rate_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="fps must be positive"):
        record_session(
            arms={"left": FakeArm()},
            source=ScriptedSource([]),
            sink=MemorySink(),
            task="t",
            fps=0,
        )


def test_an_unconnected_arm_says_so_rather_than_recording_nothing() -> None:
    class Unconnected:
        latest_state = None

        def command(self, command) -> None: ...

    with pytest.raises(ConfigurationError, match="not connected"):
        run([TeleopStep()], arms={"left": Unconnected()})


def test_the_memory_sink_refuses_an_empty_episode() -> None:
    with pytest.raises(ConfigurationError, match="no frames"):
        MemorySink().save_episode()


# --------------------------------------------------------------------------- #
# the hold source
# --------------------------------------------------------------------------- #


def test_the_hold_source_records_one_episode_and_stops() -> None:
    arm = FakeArm()
    sink = MemorySink()

    result = record_session(
        arms={"left": arm},
        source=HoldSource({"left": 6}, duration_s=0.01),
        sink=sink,
        task="pipeline check",
        report=lambda message: None,
        **FAST,
    )

    assert sink.num_episodes == 1
    assert result.frames > 0
    assert "source stopped" in result.ended_by
    # Holding means commanding the measured pose, not drifting.
    assert list(arm.commands[0].position_rad) == pytest.approx([arm.position] * 6)


def test_the_hold_source_waits_for_a_first_state() -> None:
    arm = FakeArm()
    arm.publishing = False

    step = HoldSource({"left": 6}).poll({"left": arm.latest_state})

    assert step.targets == {}
    assert step.event is EpisodeEvent.NONE


# --------------------------------------------------------------------------- #
# the seam really fits a real arm
# --------------------------------------------------------------------------- #


def test_a_real_session_arm_satisfies_the_loop_s_expectations() -> None:
    # The loop is documented to take "anything with latest_state and command".
    # This is the case that matters: a FollowerArm from a real ArmSession, on a
    # fake backend, driven through a whole episode.
    from openpi_control.cli import power_up

    rig = resolve_rig("yam_bimanual").subset(["left"]).without_cameras()
    backends: dict[str, FakeArmBackend] = {}

    def factory(rig_arm):
        backends[rig_arm.name] = FakeArmBackend()
        return backends[rig_arm.name]

    session, live_arms = power_up(rig, backend_factory=factory)
    try:
        arms = {entry.name: entry.arm for entry in live_arms}
        sink = MemorySink()
        result = record_session(
            arms=arms,
            source=ScriptedSource(
                [
                    TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
                    TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
                ]
            ),
            sink=sink,
            task="real session",
            report=lambda message: None,
            **FAST,
        )
    finally:
        session.close()

    assert sink.num_episodes == 1
    assert result.frames == 1  # START writes, SAVE closes
    assert len(backends["left"].commands) == 2


# --------------------------------------------------------------------------- #
# camera warm-up and pixel format
# --------------------------------------------------------------------------- #


def test_recording_refuses_to_start_until_every_camera_has_a_frame() -> None:
    # LeRobot needs every feature on every frame, so a camera that is only slow
    # to start would otherwise leave its view off the front of an episode.
    with pytest.raises(ConfigurationError, match="produced no frame"):
        run(
            [TeleopStep(targets={"left": target()}, event=EpisodeEvent.START)],
            cameras={"top": FakeCamera([])},
            camera_warmup_s=0.05,
        )


def test_a_slow_camera_is_waited_for_rather_than_dropped() -> None:
    class Slow:
        pixel_format = "bgr8"

        def __init__(self):
            self.calls = 0

        def latest(self):
            self.calls += 1
            return bgr() if self.calls > 3 else None

    _, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ],
        cameras={"top": Slow()},
    )

    assert "observation.images.top" in sink.episodes[0][0]


def test_an_rgb_camera_is_not_converted_again() -> None:
    # Converting an already-RGB frame swaps the channels back the wrong way, so
    # every red object in the dataset would come out blue.
    class RgbCamera(FakeCamera):
        pixel_format = "rgb8"

    frame = bgr()  # bytes are (10, 20, 30); already-RGB means store them as-is
    _, sink, _ = run(
        [
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.START),
            TeleopStep(targets={"left": target()}, event=EpisodeEvent.SAVE),
        ],
        cameras={"top": RgbCamera([frame, frame])},
    )

    assert sink.episodes[0][0]["observation.images.top"][0, 0].tolist() == [10, 20, 30]


def test_a_camera_that_does_not_say_its_format_is_assumed_bgr() -> None:
    # BGR is the older assumption, here and in OpenCV. Guessing "already RGB"
    # for an unknown camera would silently store swapped channels.
    class Nameless:
        def latest(self):
            return bgr()

    assert needs_rgb_conversion(Nameless())
    assert not needs_rgb_conversion(type("C", (), {"pixel_format": "rgb8"})())
