"""Recording teleoperated demonstrations into a LeRobot dataset.

The loop here is the boring part, and deliberately so: observe every arm, ask
the teleop source what to do, command the arms, write a frame. What makes it
worth its own module is everything around that -- the episode state machine an
operator drives from the headset, the feature schema derived from the rig, and
the several ways a dataset can come out quietly wrong.

Three seams keep this testable without hardware, without a headset, and without
``lerobot`` installed, which matters because a recording session is the one
thing here you cannot rehearse cheaply on real arms:

:class:`TeleopSource`
    Where actions and episode events come from. The Quest bridge lives in
    :mod:`openpi_control.teleop_vr`; :class:`ScriptedSource` is the stand-in.

:class:`EpisodeSink`
    Where frames go. :class:`LeRobotSink` is the real one; a fake sink lets the
    loop be tested frame by frame.

The arms themselves
    Anything with ``latest_state`` and ``command`` -- so the same
    ``FakeArmBackend`` the ``live`` tests use drives a whole recording session.

Three ways a dataset comes out quietly wrong -- a stale pose recorded as a still
arm, BGR stored where RGB is assumed, a discarded take leaking into the next
episode -- are refused rather than written. Each is argued at the place it is
handled; ``docs/recording.md`` collects them for the operator.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .exceptions import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from .rigs import Rig
    from .types import ArmState

# How stale an arm's published state may be before a tick is dropped. The node
# publishes at 200 Hz, so at a 30 Hz record rate anything past ~3 ticks of
# silence is a real stall rather than jitter.
DEFAULT_MAX_STATE_AGE_S = 0.1

# How long every arm may be stalled before an open episode is thrown away. Half
# a second of a frozen arm is already enough bad data to matter, and an episode
# is cheap to redo while a poisoned dataset is not.
DEFAULT_MAX_STALL_S = 0.5

# How long a camera gets to produce its first frame before recording refuses to
# start. A RealSense needs a few hundred ms after its pipeline starts; five
# seconds means the camera is not working, not that it is slow.
DEFAULT_CAMERA_WARMUP_S = 5.0

# Gripper conventions, plural, because the two worlds this module joins disagree
# and the inversion has to live in exactly one place or it lives in five.
#
#   native (this package, `EffectorState.position` and `PositionCommand.effector`)
#       1.0 = fully open, 0.0 = fully closed. See docs/fr3.md and the "semantic
#       normalized position 1.0 (fully open)" ready move in the native effector.
#   dataset (LeRobot, and every dataset already recorded on this cell)
#       0.0 = fully open, 1.0 = fully closed.
#
# Recorded columns use the *dataset* convention, because the entire point of
# writing a LeRobotDataset is that LeRobot and openpi tooling can read it, and
# because the datasets this cell already produced use it -- one collection with
# two polarities in it would be worse than either choice. `ArmTarget.effector`
# stays *native*, since it goes straight into a `PositionCommand`.
NATIVE_GRIPPER_OPEN = 1.0
NATIVE_GRIPPER_CLOSED = 0.0
DATASET_GRIPPER_OPEN = 0.0
DATASET_GRIPPER_CLOSED = 1.0


def to_dataset_gripper(native: float) -> float:
    """Native gripper position (1 = open) -> dataset column (0 = open)."""
    return 1.0 - float(native)


def to_native_gripper(dataset: float) -> float:
    """Dataset column (0 = open) -> native gripper position (1 = open).

    Same arithmetic as its inverse, under its own name so a call site reads the
    direction instead of leaving a bare ``1.0 - x`` to be interpreted.
    """
    return 1.0 - float(dataset)


class EpisodeEvent(StrEnum):
    """What the operator just asked for, between one tick and the next."""

    #: Nothing happened this tick.
    NONE = "none"
    #: Begin a take. If one is already open it is thrown away first, because
    #: the operator's button means "start this take" and a botched one should be
    #: redone with a single press rather than stop-then-start.
    START = "start"
    #: Keep the open take.
    SAVE = "save"
    #: Throw the open take away without starting another.
    DISCARD = "discard"
    #: End the session.
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ArmTarget:
    """Where one arm should go this tick.

    ``position_rad`` is the arm's own joints, gripper excluded -- the gripper is
    ``effector``, in this package's *native* convention (1.0 open, 0.0 closed),
    because that is what a ``PositionCommand`` takes. ``None`` means "leave the
    gripper where it is", which is not the same as 0.0 -- that would slam it
    shut.
    """

    position_rad: tuple[float, ...]
    effector: float | None = None

    def __post_init__(self) -> None:
        if not self.position_rad:
            raise ConfigurationError("an arm target needs at least one joint")
        if self.effector is not None and not 0.0 <= self.effector <= 1.0:
            raise ConfigurationError(
                f"effector target {self.effector} is not normalized to [0.0, 1.0]"
            )


@dataclass(frozen=True, slots=True)
class TeleopStep:
    """One tick's worth of intent from the teleop source."""

    targets: Mapping[str, ArmTarget] = field(default_factory=dict)
    event: EpisodeEvent = EpisodeEvent.NONE


class TeleopSource(Protocol):
    """Where actions and episode control come from.

    Implementations own their own transport and threading. ``poll`` is called
    once per tick with every arm's newest state and must not block for long: it
    sits directly in the record loop, so a source that waits on a network round
    trip sets the dataset's real frame rate.
    """

    def describe(self) -> str:
        """One line naming the source, for the session log."""
        ...

    def poll(self, states: Mapping[str, ArmState | None]) -> TeleopStep:
        """Targets and any episode event for this tick."""
        ...

    def close(self) -> None: ...


class EpisodeSink(Protocol):
    """Where recorded frames go."""

    @property
    def num_episodes(self) -> int: ...

    def add_frame(self, frame: dict[str, object]) -> None: ...

    def save_episode(self) -> None: ...

    def discard_episode(self) -> None: ...

    def finalize(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RecordResult:
    """What a session produced."""

    episodes: int
    frames: int
    discarded: int
    stalled_ticks: int
    ended_by: str

    def summary(self) -> str:
        return (
            f"{self.episodes} episode(s), {self.frames} frames, "
            f"{self.discarded} discarded, {self.stalled_ticks} stalled tick(s) "
            f"({self.ended_by})"
        )


# --------------------------------------------------------------------------- #
# feature schema
# --------------------------------------------------------------------------- #


def arm_feature_names(arm_names: Sequence[str], dofs: Mapping[str, int]) -> list[str]:
    """State/action column names: each arm's joints, then its gripper.

    Flat and prefixed (``left_joint_1`` ... ``left_gripper``, then ``right_``)
    because that is the layout LeRobot datasets from bimanual cells use, and it
    keeps a single-arm subset a strict prefix of the bimanual one rather than a
    different schema.
    """
    names: list[str] = []
    for name in arm_names:
        names.extend(f"{name}_joint_{index + 1}" for index in range(dofs[name]))
        names.append(f"{name}_gripper")
    return names


def build_features(
    state_names: Sequence[str],
    camera_shapes: Mapping[str, tuple[int, int, int]],
) -> dict[str, dict]:
    """The LeRobot feature dict for a session.

    Cameras are typed ``video`` rather than ``image`` so episodes land as one
    mp4 per camera instead of a directory of PNGs -- the difference is roughly
    two orders of magnitude on disk. Shapes come from a real frame, not from
    the requested capture size, so a rotated camera is described correctly.
    """
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
    }
    for name, shape in camera_shapes.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


def observation_vector(
    arm_names: Sequence[str],
    states: Mapping[str, ArmState | None],
    dofs: Mapping[str, int],
) -> np.ndarray:
    """Measured state, laid out to match :func:`arm_feature_names`.

    Gripper values are converted to the dataset convention on the way out, so
    every number in a recorded row shares one polarity.

    An arm with no state yet contributes zeros. That only happens before the
    first frame arrives, and the record loop refuses to write a tick where any
    state is missing, so those zeros never reach a dataset.
    """
    values: list[float] = []
    for name in arm_names:
        state = states.get(name)
        dof = dofs[name]
        if state is None:
            values.extend([0.0] * (dof + 1))
            continue
        positions = np.asarray(state.joints.position_rad, dtype=np.float64)
        values.extend(float(value) for value in positions[:dof])
        values.append(
            to_dataset_gripper(state.effector.position)
            if state.effector is not None
            else DATASET_GRIPPER_OPEN
        )
    return np.asarray(values, dtype=np.float32)


def action_vector(
    arm_names: Sequence[str],
    targets: Mapping[str, ArmTarget],
    observation: np.ndarray,
    dofs: Mapping[str, int],
) -> np.ndarray:
    """Commanded target, laid out to match :func:`arm_feature_names`.

    Falls back to the measured value for anything the source did not command --
    an arm it is not driving, or a gripper it left alone. "Where the arm already
    is" is the only honest action for a tick nobody commanded; a zero there
    would read as a command to fold the arm up.
    """
    values: list[float] = []
    offset = 0
    for name in arm_names:
        dof = dofs[name]
        target = targets.get(name)
        if target is None:
            values.extend(float(value) for value in observation[offset : offset + dof + 1])
        else:
            if len(target.position_rad) != dof:
                raise ConfigurationError(
                    f"teleop source commanded {len(target.position_rad)} joints for arm "
                    f"{name!r}, which has {dof}"
                )
            values.extend(float(value) for value in target.position_rad)
            values.append(
                to_dataset_gripper(target.effector)
                if target.effector is not None
                # The observation is already in dataset units, so an
                # uncommanded gripper carries straight over without a round trip
                # through the native convention.
                else float(observation[offset + dof])
            )
        offset += dof + 1
    return np.asarray(values, dtype=np.float32)


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR (what a camera gives by default) -> RGB (what LeRobot assumes).

    Done with a numpy reverse rather than ``cv2.cvtColor`` so the record loop
    needs no OpenCV, and done here rather than in the sink because the stored
    dataset really is RGB -- that is a property of the data, not of the writer.

    Prefer not needing it. A channel reverse plus the contiguous copy an encoder
    requires costs ~1.4 ms for an 848x480 frame, against ~0.03 ms for a plain
    copy: three cameras at 90 Hz would spend 4.1 ms of an 11.1 ms tick just
    swapping bytes. Asking the camera for ``rgb8`` up front moves that work into
    the SDK's native conversion, where it is free -- which is why
    :func:`record_session` skips this for readers that already emit RGB.
    """
    return np.ascontiguousarray(frame[:, :, ::-1])


def needs_rgb_conversion(camera: object) -> bool:
    """Whether this camera's frames still have to be flipped to RGB.

    Defaults to True for anything that does not declare a ``pixel_format``:
    BGR is the older assumption here and in OpenCV, so guessing "already RGB"
    for an unknown camera would silently store every dataset with red and blue
    swapped.
    """
    return str(getattr(camera, "pixel_format", "bgr8")).lower() != "rgb8"


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


def record_session(
    *,
    arms: Mapping[str, object],
    source: TeleopSource,
    sink: EpisodeSink,
    task: str,
    fps: int = 30,
    cameras: Mapping[str, object] | None = None,
    num_episodes: int = 0,
    stop: threading.Event | None = None,
    max_state_age_s: float = DEFAULT_MAX_STATE_AGE_S,
    max_stall_s: float = DEFAULT_MAX_STALL_S,
    camera_warmup_s: float = DEFAULT_CAMERA_WARMUP_S,
    report: object | None = None,
    finalize: bool = True,
    save_on_interrupt: bool = False,
) -> RecordResult:
    """Drive the arms from ``source`` and write episodes to ``sink``.

    Teleop keeps driving between episodes on purpose: resetting the scene with
    the arm is most of what happens between takes, and an operator should not
    have to think about whether the arm is live. Only frames between a START and
    its SAVE are written, except that ``save_on_interrupt`` also keeps the
    already-captured prefix.

    ``num_episodes`` of 0 runs until the source says STOP or ``stop`` is set.

    By default, Ctrl-C discards an open episode because that is the safe
    default for teleoperation. ``save_on_interrupt`` keeps all frames already
    captured in a partial episode instead; the policy rollout recorder uses
    that mode so an operator can stop a trial without losing its data.

    The source does not have to track whether an episode is open: START while
    recording discards and restarts, SAVE while idle does nothing. That keeps a
    headset bridge to plain edge detection on two buttons.

    Events are applied before the frame is written, so a take spans the START
    tick through the tick before its SAVE. The tick carrying SAVE is the
    operator reaching for a button rather than doing the task, and it is the
    same boundary vr-teleop-kit's recorder uses.
    """
    from .types import PositionCommand

    if fps <= 0:
        raise ConfigurationError(f"fps must be positive, got {fps}")
    arm_names = list(arms)
    if not arm_names:
        raise ConfigurationError("recording needs at least one arm")
    cameras = dict(cameras or {})
    stop = stop if stop is not None else threading.Event()
    say = report or (lambda message: print(f"  {message}"))

    dofs = {name: _arm_dof(arm) for name, arm in arms.items()}
    # No tick runs until every camera has produced a frame. A camera that is
    # merely slow to start would otherwise leave its feature off the front of an
    # episode, and LeRobot needs every feature on every frame -- that is a
    # broken dataset, not a few dim images.
    _await_first_frames(cameras, camera_warmup_s)
    period = 1.0 / fps
    max_stall_ticks = max(1, int(round(max_stall_s * fps)))

    recording = False
    frames_in_episode = 0
    total_frames = 0
    discarded = 0
    stalled_ticks = 0
    consecutive_stalls = 0
    ended_by = "stop requested"
    interrupted = False
    # Last good frame per camera: LeRobot needs every feature on every frame, so
    # a camera that misses a grab repeats rather than dropping out of the schema.
    last_frames: dict[str, np.ndarray] = {}
    # Decided once, not per frame: at 90 Hz this branch runs 270 times a second.
    convert = {name: needs_rgb_conversion(camera) for name, camera in cameras.items()}

    def drop(reason: str) -> None:
        """Throw the open take away. The only place that counts a discard."""
        nonlocal recording, discarded
        sink.discard_episode()
        recording, discarded = False, discarded + 1
        say(f"episode discarded: {reason}")

    next_tick = time.perf_counter()
    try:
        while not stop.is_set():
            states: dict[str, ArmState | None] = {
                name: getattr(arm, "latest_state", None) for name, arm in arms.items()
            }
            step = source.poll(states)

            if step.event is EpisodeEvent.STOP:
                ended_by = "source stopped"
                break
            if step.event is EpisodeEvent.START:
                if recording:
                    # Re-pressing start means "that take was no good, again".
                    drop(f"restarted, {frames_in_episode} frames thrown away")
                recording, frames_in_episode = True, 0
                say(f"episode {sink.num_episodes} recording")
            elif step.event is EpisodeEvent.SAVE and recording:
                if frames_in_episode == 0:
                    # LeRobot raises on saving an empty buffer, and a save
                    # pressed before any frame landed is a fumble, not an end.
                    say("save with 0 frames — episode stays open")
                else:
                    sink.save_episode()
                    recording = False
                    total_frames += frames_in_episode
                    say(f"episode {sink.num_episodes - 1} saved ({frames_in_episode} frames)")
                    if num_episodes and sink.num_episodes >= num_episodes:
                        ended_by = f"reached {num_episodes} episode(s)"
                        break
            elif step.event is EpisodeEvent.DISCARD and recording:
                drop(f"{frames_in_episode} frames thrown away")

            stale = _stale_arms(states, max_state_age_s)
            if stale:
                stalled_ticks += 1
                consecutive_stalls += 1
                if recording and consecutive_stalls >= max_stall_ticks:
                    # The node republishes its last cached pose when frames stop
                    # arriving, so continuing would write a motionless arm as
                    # ground truth for however long the stall lasts.
                    drop(f"{', '.join(stale)} stalled for {consecutive_stalls} ticks")
                next_tick = _wait(next_tick, period)
                continue
            consecutive_stalls = 0

            for name, target in step.targets.items():
                arm = arms.get(name)
                if arm is None:
                    raise ConfigurationError(
                        f"teleop source commanded unknown arm {name!r}; "
                        f"this session holds {', '.join(arm_names)}"
                    )
                arm.command(  # type: ignore[attr-defined]
                    PositionCommand(
                        position_rad=np.asarray(target.position_rad, dtype=np.float64),
                        effector=target.effector,
                    )
                )

            if recording:
                observation = observation_vector(arm_names, states, dofs)
                frame: dict[str, object] = {
                    "observation.state": observation,
                    "action": action_vector(arm_names, step.targets, observation, dofs),
                    "task": task,
                }
                for name, camera in cameras.items():
                    image = camera.latest()  # type: ignore[attr-defined]
                    if image is None:
                        image = last_frames.get(name)
                    if image is None:
                        continue
                    last_frames[name] = image
                    frame[f"observation.images.{name}"] = (
                        to_rgb(image) if convert[name] else image
                    )
                sink.add_frame(frame)
                frames_in_episode += 1

            next_tick = _wait(next_tick, period)
    except KeyboardInterrupt:
        ended_by = "interrupted"
        interrupted = True
    finally:
        if recording:
            if save_on_interrupt and interrupted and frames_in_episode:
                sink.save_episode()
                recording = False
                total_frames += frames_in_episode
                say(
                    f"episode {sink.num_episodes - 1} saved partially "
                    f"({frames_in_episode} frames; interrupted)"
                )
            else:
                # An interrupted take is discarded by default. Rollout
                # recording opts into saving partial data above.
                drop("session ended mid-episode")
        if finalize:
            sink.finalize()

    return RecordResult(
        episodes=sink.num_episodes,
        frames=total_frames,
        discarded=discarded,
        stalled_ticks=stalled_ticks,
        ended_by=ended_by,
    )


def _wait(next_tick: float, period: float) -> float:
    """Sleep until the next tick; resynchronize if we already overran it."""
    next_tick += period
    remaining = next_tick - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
        return next_tick
    # Behind schedule: chasing the missed ticks would burn CPU catching up and
    # still not recover the frames, so restart the clock from now.
    return time.perf_counter()


def _await_first_frames(
    cameras: Mapping[str, object], timeout_s: float
) -> dict[str, np.ndarray]:
    """Block until every camera has a frame; return the frames it saw.

    Returning them matters: a caller that needs the frame (to read its shape)
    would otherwise ask ``latest()`` a second time, and that is two chances to
    get ``None`` for one guarantee.

    Polls ``latest()`` rather than calling ``wait_for_frame``, because the loop
    is documented to accept any object with ``latest()``; requiring more of a
    camera here than the loop itself uses would narrow that seam for no reason.
    """
    first: dict[str, np.ndarray] = {}
    if not cameras:
        return first
    deadline = time.monotonic() + timeout_s
    pending = dict(cameras)
    while pending and time.monotonic() < deadline:
        for name, camera in list(pending.items()):
            frame = camera.latest()  # type: ignore[attr-defined]
            if frame is not None:
                first[name] = frame
                del pending[name]
        if pending:
            time.sleep(0.02)
    if pending:
        raise ConfigurationError(
            f"camera(s) {', '.join(sorted(pending))} produced no frame in "
            f"{timeout_s:g}s; recording would leave those views out of the dataset"
        )
    return first


def _stale_arms(states: Mapping[str, ArmState | None], max_age_s: float) -> list[str]:
    return sorted(
        name
        for name, state in states.items()
        if state is None or not state.is_fresh(max_age_s)
    )


def _arm_dof(arm: object) -> int:
    capabilities = getattr(arm, "capabilities", None)
    dof = getattr(capabilities, "dof", None)
    if not isinstance(dof, int) or dof <= 0:
        raise ConfigurationError(
            f"cannot record from {arm!r}: it reports no joint count, so it is "
            "probably not connected yet"
        )
    return dof


# --------------------------------------------------------------------------- #
# sinks
# --------------------------------------------------------------------------- #


class LeRobotSink:
    """An :class:`EpisodeSink` backed by a real ``LeRobotDataset``.

    ``lerobot`` is imported lazily: it pulls in torch, and a robot box that only
    drives arms has no use for that. Everything else in this module works
    without it, which is what lets the record loop be tested at all.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        fps: int,
        features: Mapping[str, dict],
        robot_type: str,
        root: Path | None = None,
        image_writer_threads: int = 0,
        streaming_encoding: bool = True,
        encoder_queue_maxsize: int = 90,
    ) -> None:
        dataset_module = _require_lerobot()
        self.repo_id = repo_id
        self._dataset = dataset_module.LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=robot_type,
            use_videos=any(
                spec.get("dtype") == "video" for spec in features.values()
            ),
            # Streaming encoding, because the alternative does not survive this
            # cell's frame rate. LeRobot's default stages every frame as a PNG
            # and encodes the lot inside save_episode(): at three cameras and
            # 90 Hz that is 270 PNG encodes a second (~37 ms each on a noisy
            # 848x480 frame), several gigabytes of temporary files per take, and
            # a save_episode that blocks this process for tens of seconds --
            # inside the control loop, with the arms holding their last command.
            # Streaming encodes during capture instead, so save_episode is
            # near-instant and nothing is staged.
            #
            # The cost is a bounded queue: when the encoder falls behind, LeRobot
            # drops a frame with a warning rather than blocking. That is the
            # right trade for a teleop loop -- a stalled loop is worse than a
            # dropped frame -- but it is a real behaviour, so a session reports
            # its frame count and the dataset records the rate it actually kept.
            streaming_encoding=streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            # Only meaningful without streaming; kept so a caller can opt back
            # into the staged path.
            image_writer_threads=0 if streaming_encoding else image_writer_threads,
            features=dict(features),
        )

    @property
    def root(self) -> Path:
        return self._dataset.root

    @property
    def num_episodes(self) -> int:
        return int(self._dataset.num_episodes)

    def add_frame(self, frame: dict[str, object]) -> None:
        self._dataset.add_frame(frame)

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def discard_episode(self) -> None:
        """Drop the open episode's buffer, frames and all.

        On lerobot 3.0 ``clear_episode_buffer`` is enough, and that is worth
        recording because it was not always: older versions only deleted image
        directories for features typed ``image``, so ``video`` cameras like ours
        left the async writer's per-frame PNGs on disk -- gigabytes for one
        thrown-away take -- and callers had to reach past the public API for
        ``writer.cleanup_interrupted_episode``. That workaround is gone from
        here because it no longer applies and no longer exists: 0.6 has no
        ``writer`` attribute at all, so the code was silently doing nothing.
        ``test_a_discarded_take_leaves_nothing_behind`` is what keeps this
        honest against a future version.
        """
        self._dataset.clear_episode_buffer()

    def finalize(self) -> None:
        self._dataset.finalize()

    def push_to_hub(self, *, private: bool = False) -> None:
        """Upload the finished dataset. The caller runs this after the arms are
        down (see :func:`openpi_control.cli.run_record`), so a failed upload
        costs a retry and nothing else."""
        if self.num_episodes == 0:
            raise ConfigurationError("nothing to push: no episode was saved")
        self._dataset.push_to_hub(
            private=private, tags=["robotics", "lerobot", "openpi-control"]
        )


class MemorySink:
    """An :class:`EpisodeSink` that keeps everything in memory.

    Not a test double -- it is what ``--dry-run`` records into, so the whole
    session (episode transitions, camera frames, the fps loop) can be rehearsed
    with the arms live and nothing written to disk.
    """

    def __init__(self) -> None:
        self.episodes: list[list[dict[str, object]]] = []
        self.open_frames: list[dict[str, object]] = []
        self.discarded = 0
        self.finalized = False

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def add_frame(self, frame: dict[str, object]) -> None:
        self.open_frames.append(frame)

    def save_episode(self) -> None:
        if not self.open_frames:
            raise ConfigurationError("cannot save an episode with no frames")
        self.episodes.append(self.open_frames)
        self.open_frames = []

    def discard_episode(self) -> None:
        self.open_frames = []
        self.discarded += 1

    def finalize(self) -> None:
        self.finalized = True


# --------------------------------------------------------------------------- #
# a source you can drive from a script
# --------------------------------------------------------------------------- #


class ScriptedSource:
    """A :class:`TeleopSource` that replays a fixed list of steps.

    How the record loop is tested, and the shortest complete example of the
    protocol: a source owes the loop targets and, occasionally, an episode event.
    """

    def __init__(self, steps: Sequence[TeleopStep]) -> None:
        self._steps = list(steps)
        self._index = 0
        self.polls = 0
        self.closed = False

    def describe(self) -> str:
        return f"scripted source ({len(self._steps)} steps)"

    def poll(self, states: Mapping[str, ArmState | None]) -> TeleopStep:
        del states
        self.polls += 1
        if self._index < len(self._steps):
            step = self._steps[self._index]
            self._index += 1
            return step
        return TeleopStep(event=EpisodeEvent.STOP)

    def close(self) -> None:
        self.closed = True


class HoldSource:
    """A :class:`TeleopSource` that holds every arm still for a fixed time.

    Not a way to collect data -- nothing moves. It is how you verify the rest of
    the pipeline on real hardware without a headset: that the arms come up, that
    every camera lands in the dataset with the right shape and the right colours,
    that the loop actually holds the requested frame rate. One episode of
    ``duration_s``, then STOP.
    """

    def __init__(self, dofs: Mapping[str, int], *, duration_s: float = 10.0) -> None:
        self._dofs = dict(dofs)
        self._duration_s = duration_s
        self._started_at: float | None = None
        self._saved = False

    def describe(self) -> str:
        return f"hold ({self._duration_s:g}s, arms stationary — a pipeline check, not data)"

    def poll(self, states: Mapping[str, ArmState | None]) -> TeleopStep:
        targets = hold_targets(states, self._dofs)
        if self._started_at is None:
            # Wait for a first state: holding an arm at a pose nobody has
            # reported yet is not something to guess at.
            if not targets:
                return TeleopStep()
            self._started_at = time.monotonic()
            return TeleopStep(targets=targets, event=EpisodeEvent.START)
        if self._saved:
            # One episode is the whole point; asking to save again every tick
            # would spin here forever when --num-episodes was not given.
            return TeleopStep(targets=targets, event=EpisodeEvent.STOP)
        if time.monotonic() - self._started_at >= self._duration_s:
            self._saved = True
            return TeleopStep(targets=targets, event=EpisodeEvent.SAVE)
        return TeleopStep(targets=targets)

    def close(self) -> None:
        return None


def hold_targets(
    states: Mapping[str, ArmState | None], dofs: Mapping[str, int]
) -> dict[str, ArmTarget]:
    """Targets that ask every arm to stay exactly where it is.

    An arm with no state yet is left out rather than guessed at: commanding a
    pose nobody has reported is the one thing worse than commanding nothing.
    """
    targets: dict[str, ArmTarget] = {}
    for name, state in states.items():
        if state is None:
            continue
        positions = tuple(float(value) for value in state.joints.position_rad[: dofs[name]])
        effector = float(state.effector.position) if state.effector is not None else None
        targets[name] = ArmTarget(position_rad=positions, effector=effector)
    return targets


def camera_shapes(
    cameras: Mapping[str, object], *, warmup_s: float = DEFAULT_CAMERA_WARMUP_S
) -> dict[str, tuple[int, int, int]]:
    """Frame shape of each camera, taken from a real frame.

    From a frame rather than from the requested capture size, because a rotated
    camera emits the transposed shape and a feature schema that disagrees with
    the frames fails deep inside the video encoder.
    """
    return {
        name: (*frame.shape[:2], 3)
        for name, frame in _await_first_frames(cameras, warmup_s).items()
    }


def rig_robot_type(rig: Rig) -> str:
    """A stable ``robot_type`` string for the dataset metadata."""
    return rig.name if len(rig.arms) > 1 else f"{rig.name}_{rig.arms[0].name}"


def _require_lerobot():  # noqa: ANN202 - the lerobot dataset module, Any by design
    try:
        from lerobot.datasets import lerobot_dataset
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            "recording a dataset needs lerobot, which is not importable here "
            f"({err}). Install it with `uv sync --extra lerobot` -- note that "
            "needs Python 3.12 or newer, so on 3.11 the extra resolves to "
            "nothing and this is the error you get."
        ) from err
    return lerobot_dataset
