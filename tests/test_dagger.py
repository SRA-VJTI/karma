"""Human-gated DAgger, without a headset, a policy server, or arms.

The record loop and the two drivers are each tested elsewhere. What is only
testable here is the seam between them: who is driving on a given tick, what
each side is told at a handoff, and what ends up in the frame that says so.
Getting any of those wrong produces a dataset that looks fine and trains a
policy on the wrong actions, which is the failure this file exists to refuse.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from openpi_control.dagger import INTERVENTION_FEATURE, DaggerSource, InterventionStats
from openpi_control.inference_record import InferenceRolloutSource
from openpi_control.record import (
    ArmTarget,
    EpisodeEvent,
    MemorySink,
    TeleopStep,
    record_session,
)
from openpi_control.types import (
    ArmMode,
    ArmRole,
    ArmState,
    EffectorState,
    JointState,
)

DOFS = 6


def state(position: float = 0.25, effector: float | None = 1.0) -> ArmState:
    return ArmState(
        name="fake",
        role=ArmRole.FOLLOWER,
        joints=JointState(
            names=tuple(f"joint_{index + 1}" for index in range(DOFS)),
            position_rad=[position] * DOFS,
            velocity_rad_s=[0.0] * DOFS,
            effort_nm=[0.0] * DOFS,
            temperature_c=[25.0] * DOFS,
            current_a=[0.0] * DOFS,
        ),
        effector=EffectorState(position=effector) if effector is not None else None,
        monotonic_timestamp=time.monotonic(),
        wall_timestamp=0.0,
        sequence=1,
        mode=ArmMode.HOLD,
    )


def states(position: float = 0.25, effector: float | None = 1.0) -> dict[str, ArmState]:
    return {name: state(position, effector) for name in ("left", "right")}


class FakePolicy:
    """Stands in for InferenceRolloutSource: acts, and can be told to re-plan."""

    def __init__(self, *, joint: float = 0.1, effector: float = 0.9) -> None:
        self.joint = joint
        self.effector = effector
        self.acts = 0
        self.resumes: list[dict] = []
        self.closed = False

    def describe(self) -> str:
        return "FakePolicy"

    def act(self, _states) -> dict[str, ArmTarget]:
        self.acts += 1
        return {
            name: ArmTarget(position_rad=(self.joint,) * DOFS, effector=self.effector)
            for name in ("left", "right")
        }

    def resume_after_intervention(self, states_in, *, effector=None) -> None:
        self.resumes.append({"states": states_in, "effector": effector})

    def close(self) -> None:
        self.closed = True


class FakeHuman:
    """Stands in for QuestTeleopSource: two buttons as levels, and a seed."""

    def __init__(self, *, joint: float = 0.7, effector: float = 0.2) -> None:
        self.joint = joint
        self.effector = effector
        self.take_pressed = False
        self.give_pressed = False
        self.polls = 0
        self.seeds: list[dict] = []
        self.seedable = True
        self.closed = False

    def describe(self) -> str:
        return "FakeHuman"

    def right_b(self) -> bool:
        return self.take_pressed

    def left_y(self) -> bool:
        return self.give_pressed

    def seed_from(self, states_in, *, effector=None) -> bool:
        self.seeds.append({"states": states_in, "effector": effector})
        return self.seedable

    def poll(self, _states) -> TeleopStep:
        self.polls += 1
        return TeleopStep(
            targets={
                name: ArmTarget(position_rad=(self.joint,) * DOFS, effector=self.effector)
                for name in ("left", "right")
            }
        )

    def close(self) -> None:
        self.closed = True


def build(**kwargs):
    policy = kwargs.pop("policy", None) or FakePolicy()
    human = kwargs.pop("human", None) or FakeHuman()
    kwargs.setdefault("episode_seconds", 60.0)
    # A real 1 s hold in every test would be a minute of sleeping; the gesture
    # under test is "held past the threshold", not the threshold itself.
    kwargs.setdefault("end_hold_s", 0.005)
    kwargs.setdefault("report", lambda _message: None)
    return DaggerSource(policy=policy, human=human, **kwargs), policy, human


def label(step: TeleopStep) -> float:
    return float(np.asarray(step.extras["intervention"])[0])


def tap_give(source, human, at=None) -> TeleopStep:
    """Press and release Left Y -- the gesture that hands control back.

    Resolved on the release, so it takes two ticks. A test that only presses
    would be asserting against the hold gesture instead.
    """
    at = at if at is not None else states()
    human.give_pressed = True
    source.poll(at)
    human.give_pressed = False
    return source.poll(at)


def hold_give(source, human, at=None) -> TeleopStep:
    """Hold Left Y past the threshold -- the gesture that ends the episode."""
    at = at if at is not None else states()
    human.give_pressed = True
    source.poll(at)
    time.sleep(source.end_hold_s * 2)
    return source.poll(at)


# --------------------------------------------------------------------------- #
# the two buttons
# --------------------------------------------------------------------------- #


def test_right_b_hands_the_arms_to_the_operator() -> None:
    source, policy, human = build()

    autonomous = source.poll(states())
    human.take_pressed = True
    corrected = source.poll(states())

    assert source.intervening
    assert policy.acts == 1, "the policy drove exactly the first tick"
    assert human.polls == 1, "and the operator drove the second"
    assert label(autonomous) == 0.0
    assert label(corrected) == 1.0
    assert corrected.targets["left"].position_rad == (human.joint,) * DOFS


def test_left_y_gives_the_arms_back() -> None:
    source, policy, human = build()
    source.poll(states())
    human.take_pressed = True
    source.poll(states())
    human.take_pressed = False

    returned = tap_give(source, human)

    assert not source.intervening
    assert label(returned) == 0.0
    assert policy.acts == 2, "the policy is driving again on the tick Y is released"
    assert len(policy.resumes) == 1


def test_holding_the_take_over_button_takes_over_once() -> None:
    # The kit reports levels. Without edge detection a held thumb would re-seed
    # the operator's solver thirty times a second, each time re-anchoring the
    # mapper at their current hand pose.
    source, _policy, human = build()
    human.take_pressed = True

    for _ in range(5):
        source.poll(states())

    assert source.stats.interventions == 1
    assert len(human.seeds) == 1


def test_releasing_and_pressing_again_is_a_second_intervention() -> None:
    source, _policy, human = build()

    human.take_pressed = True
    source.poll(states())
    human.take_pressed = False
    human.give_pressed = True
    source.poll(states())
    human.give_pressed = False
    source.poll(states())
    human.take_pressed = True
    source.poll(states())

    assert source.stats.interventions == 2


def test_the_give_back_button_does_nothing_while_the_policy_is_driving() -> None:
    source, policy, human = build()
    human.give_pressed = True

    source.poll(states())

    assert not source.intervening
    assert policy.resumes == [], "nothing was handed back; nothing was re-planned"


def test_the_take_over_button_does_nothing_while_the_operator_is_driving() -> None:
    # Right B means "take the arms". Pressed again mid-correction it must not
    # re-seed, which would re-anchor the mapper and jump the arm.
    source, _policy, human = build()
    human.take_pressed = True
    source.poll(states())
    human.take_pressed = False
    source.poll(states())
    human.take_pressed = True
    source.poll(states())

    assert source.intervening
    assert len(human.seeds) == 1


# --------------------------------------------------------------------------- #
# ending the attempt on demand
# --------------------------------------------------------------------------- #


def test_holding_the_give_button_ends_the_episode() -> None:
    # An attempt is over when the operator says it is -- the towel is folded,
    # or it is on the floor -- and waiting out the remaining ninety seconds
    # records neither.
    source, _policy, human = build()
    source.poll(states())

    step = hold_give(source, human)

    assert step.event is EpisodeEvent.SAVE
    assert not step.targets, "nothing is commanded on the ending tick"
    assert source.poll(states()).event is EpisodeEvent.STOP


def test_the_episode_ends_from_under_the_operator_too() -> None:
    # Held while correcting, it still ends: the runner parks the arms either
    # way, and having to hand back first would be a rule to remember at the
    # exact moment there is nothing to hand back to.
    source, _policy, human = build()
    human.take_pressed = True
    source.poll(states())

    assert hold_give(source, human).event is EpisodeEvent.SAVE


def test_a_tap_hands_back_rather_than_ending_the_episode() -> None:
    # The whole reason hand-back is resolved on the release: a tap must not be
    # the beginning of an end-episode hold.
    source, _policy, human = build()
    human.take_pressed = True
    source.poll(states())
    human.take_pressed = False

    step = tap_give(source, human)

    assert step.event is EpisodeEvent.NONE
    assert not source.intervening


def test_holding_through_the_threshold_ends_once_not_every_tick() -> None:
    source, _policy, human = build()
    source.poll(states())
    hold_give(source, human)

    # Still held. The episode is already over; this must not re-fire.
    assert source.poll(states()).event is EpisodeEvent.STOP


def test_releasing_after_an_end_does_not_also_hand_back() -> None:
    # The press was spent on ending the episode. Releasing it is the operator
    # letting go, not a second instruction.
    source, policy, human = build()
    human.take_pressed = True
    source.poll(states())
    human.take_pressed = False
    hold_give(source, human)
    human.give_pressed = False
    source.poll(states())

    assert policy.resumes == []


# --------------------------------------------------------------------------- #
# what each side is told at a handoff
# --------------------------------------------------------------------------- #


def test_the_operator_inherits_the_gripper_the_policy_commanded() -> None:
    # A gripper holding something reads short of what closed it. Seeding the
    # operator from the measurement hands them a grip already giving way, and
    # the object drops on the first tick they take over.
    source, policy, human = build(policy=FakePolicy(effector=0.05))
    source.poll(states(effector=0.4))  # measured: the jaw stopped on the object
    human.take_pressed = True
    source.poll(states(effector=0.4))

    assert human.seeds[0]["effector"] == {"left": 0.05, "right": 0.05}


def test_the_policy_inherits_the_gripper_the_operator_commanded() -> None:
    source, policy, human = build(human=FakeHuman(effector=0.05))
    human.take_pressed = True
    source.poll(states(effector=0.4))
    human.take_pressed = False

    tap_give(source, human, states(effector=0.4))

    assert policy.resumes[0]["effector"] == {"left": 0.05, "right": 0.05}


def test_the_policy_replans_rather_than_resuming_its_old_chunk() -> None:
    source, policy, human = build()
    source.poll(states())
    human.take_pressed = True
    source.poll(states())
    human.take_pressed = False

    tap_give(source, human)

    assert len(policy.resumes) == 1
    assert set(policy.resumes[0]["states"]) == {"left", "right"}


def test_a_take_over_with_no_published_pose_is_refused() -> None:
    # An unseeded solver sits at its configured rest pose, so the first
    # commanded tick would be a full-speed move to park with the operator's
    # hand nowhere near it.
    said: list[str] = []
    source, policy, human = build(report=said.append)
    human.seedable = False
    human.take_pressed = True

    step = source.poll(states())

    assert not source.intervening
    assert label(step) == 0.0
    assert policy.acts == 1, "the policy kept the arms"
    assert any("TAKE-OVER IGNORED" in message for message in said)


def test_a_hand_back_without_a_measured_pose_keeps_the_operator_driving() -> None:
    # The executor can only be seeded from a measured pose. Handing back
    # without one would have the policy step from a target nobody has
    # confirmed the arm is near.
    said: list[str] = []
    source, policy, human = build(report=said.append)
    human.take_pressed = True
    source.poll(states())
    human.take_pressed = False

    step = tap_give(source, human, {"left": state(), "right": None})

    assert source.intervening
    assert label(step) == 1.0
    assert policy.resumes == []
    assert any("HAND-BACK IGNORED" in message for message in said)


# --------------------------------------------------------------------------- #
# the episode
# --------------------------------------------------------------------------- #


def test_the_first_tick_starts_the_episode() -> None:
    source, _policy, _human = build()
    assert source.poll(states()).event is EpisodeEvent.START
    assert source.poll(states()).event is EpisodeEvent.NONE


def test_an_episode_ends_on_time_even_mid_intervention() -> None:
    # The clock is here rather than in the policy source precisely because a
    # correcting human is the state in which the policy is never polled.
    source, _policy, human = build(episode_seconds=0.05)
    source.poll(states())
    human.take_pressed = True
    source.poll(states())
    time.sleep(0.06)

    assert source.poll(states()).event is EpisodeEvent.SAVE
    assert source.poll(states()).event is EpisodeEvent.STOP


def test_closing_leaves_the_headset_connection_alone() -> None:
    # One WebSocket serves the whole run. Reconnecting between attempts would
    # drop the operator out of teleoperation during the scene reset.
    source, policy, human = build()
    source.close()

    assert policy.closed
    assert not human.closed


def test_the_stats_count_both_sides() -> None:
    source, _policy, human = build()
    source.poll(states())
    source.poll(states())
    human.take_pressed = True
    source.poll(states())

    stats = source.stats
    assert (stats.policy_frames, stats.human_frames, stats.interventions) == (2, 1, 1)
    assert stats.frames == 3
    assert stats.as_dict()["human_fraction"] == pytest.approx(1 / 3, abs=1e-4)


def test_a_fully_autonomous_attempt_says_so() -> None:
    assert "fully autonomous" in InterventionStats(0, 0, 90).summary()


# --------------------------------------------------------------------------- #
# the column
# --------------------------------------------------------------------------- #


def test_the_intervention_column_is_declared_the_way_it_is_written() -> None:
    # LeRobot needs every feature on every frame with the declared shape; a
    # column whose spec and values disagree fails inside the video writer.
    source, _policy, _human = build()
    spec = INTERVENTION_FEATURE["intervention"]
    value = np.asarray(source.poll(states()).extras["intervention"])

    assert value.shape == tuple(spec["shape"])
    assert value.dtype == np.dtype(spec["dtype"])


# --------------------------------------------------------------------------- #
# through the real record loop
# --------------------------------------------------------------------------- #


class RecordingArm:
    """The seam ``record_session`` needs, and nothing more."""

    def __init__(self) -> None:
        self.capabilities = SimpleNamespace(dof=DOFS)
        self.commands: list = []

    @property
    def latest_state(self) -> ArmState:
        return state()

    def command(self, command) -> None:
        self.commands.append(command)


class StillCamera:
    pixel_format = "rgb8"

    def latest(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)


class TranscriptSink(MemorySink):
    """A MemorySink that also keeps every frame it was handed.

    The loop discards an episode left open when the session ends, which is the
    right default and the wrong one for reading back what was written on each
    tick. Episode boundaries are ``test_record``'s subject; this file's is the
    contents of the frames.
    """

    def __init__(self) -> None:
        super().__init__()
        self.transcript: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.transcript.append(frame)
        super().add_frame(frame)


class StillClient:
    jpeg_quality = 95

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, _observation, _instruction):
        self.calls += 1
        return np.zeros((4, 14), dtype=np.float64)


def test_a_session_records_who_was_driving_on_every_frame() -> None:
    # The end the whole module is for: one episode, the arms driven by both in
    # turn, and a dataset that says which frames are which.
    arms = {"left": RecordingArm(), "right": RecordingArm()}
    cameras = {"top": StillCamera(), "left_wrist": StillCamera(), "right_wrist": StillCamera()}
    client = StillClient()
    human = FakeHuman()
    policy = InferenceRolloutSource(
        arms=arms,
        readers=cameras,
        client=client,  # type: ignore[arg-type]
        instruction="fold the towel",
        episode_seconds=60.0,
        speed=1.0,
        prefetch=False,
    )
    source = DaggerSource(
        policy=policy,
        human=human,
        episode_seconds=60.0,
        report=lambda _message: None,
    )
    sink = TranscriptSink()

    # One autonomous tick, the operator takes the arms for three, the policy
    # gets them back for one. ``on_tick`` runs before the source is polled, so
    # a button set on tick N is acted on by tick N. Y is pressed on tick 4 and
    # released on tick 5, because handing back is resolved on the release.
    class Gate:
        """Presses the buttons from inside the loop, on the tick counts above."""

        def __init__(self) -> None:
            self.ticks = 0

        def __call__(self, _states) -> None:
            self.ticks += 1
            human.take_pressed = self.ticks == 2
            human.give_pressed = self.ticks == 4
            if self.ticks >= 5:
                stop.set()

    stop = threading.Event()
    record_session(
        arms=arms,
        source=source,
        sink=sink,
        cameras=cameras,
        task="fold the towel",
        fps=1000,
        stop=stop,
        on_tick=Gate(),
        report=lambda _message: None,
    )

    frames = sink.transcript
    assert [float(frame["intervention"][0]) for frame in frames] == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert source.stats.interventions == 1
    # The recorded action is whoever's hand was on the arm, never the other's.
    # The policy asks for zero from a pose measured at 0.25 and gets there one
    # bounded 0.1 step at a time; the operator commands 0.7 outright.
    assert frames[0]["action"][0] == pytest.approx(0.15), "policy, one step in"
    assert frames[1]["action"][0] == pytest.approx(human.joint), "operator"
    assert frames[3]["action"][0] == pytest.approx(human.joint), "operator, Y down but not released"
    assert frames[4]["action"][0] == pytest.approx(0.15), "policy, re-seeded from the measurement"
    # Handing back re-planned rather than resuming the four-action chunk.
    assert client.calls == 2


def test_commanded_effector_is_public_and_tracks_whoever_is_driving() -> None:
    """The gripper-stall watch reads this, and it must follow the handoff.

    The session watches for a gripper that is commanded across its range and
    never moves. That check needs the *commanded* side, and this source is the
    only object that has it for every tick: the two drivers are polled
    alternately, so neither one sees the whole episode.
    """
    source, policy, human = build()
    policy.effector = 0.4
    human.effector = 0.9

    # Nothing commanded yet -- no gripper to report, and no exception either.
    assert source.commanded_effector == {}

    source.poll(states())
    assert source.commanded_effector == {"left": 0.4, "right": 0.4}

    # After a takeover it is the operator's gripper that is being commanded,
    # which is exactly the value the stall watch has to compare against.
    human.take_pressed = True
    source.poll(states())
    assert source.intervening
    assert source.commanded_effector == {"left": 0.9, "right": 0.9}


def test_commanded_effector_skips_an_arm_with_no_gripper() -> None:
    """An arm commanded without an effector contributes no gripper reading."""
    source, policy, _human = build()
    policy.effector = None
    source.poll(states())
    assert source.commanded_effector == {}


def test_a_skipped_policy_tick_does_not_erase_the_commanded_gripper() -> None:
    """The policy returns nothing on a tick it skips for a stale state.

    That empty map must not become "what was last commanded", or a takeover on
    the very next tick seeds the operator with no gripper at all -- and a
    gripper seeded from the measurement gives back part of the grip.
    """
    source, policy, human = build()
    policy.effector = 0.3
    source.poll(states())
    assert source.commanded_effector == {"left": 0.3, "right": 0.3}

    policy.act = lambda _states: {}  # type: ignore[method-assign]
    source.poll(states())
    assert source.commanded_effector == {"left": 0.3, "right": 0.3}

    human.take_pressed = True
    source.poll(states())
    assert human.seeds[-1]["effector"] == {"left": 0.3, "right": 0.3}
