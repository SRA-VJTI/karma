"""YAM kinesthetic teaching, bounded friction profiles, and measured-action recording."""

from __future__ import annotations

import json
import math
import select
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .exceptions import ConfigurationError, StaleStateError
from .record import EpisodeEvent, TeleopStep
from .types import ArmMode

SCHEMA = "karma.yam_teaching.v1"


@dataclass(frozen=True)
class FrictionJoint:
    coulomb_nm: float = 0.0
    viscous_nm_s: float = 0.0
    transition_rad_s: float = 0.05
    compensation_limit_nm: float = 0.0
    velocity_limit_rad_s: float = 0.5
    damping_nm_s: float = 0.1

    def validate(self, *, gripper=False):
        values = tuple(asdict(self).values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ConfigurationError("friction parameters must be numbers")
        if not all(math.isfinite(v) for v in values):
            raise ConfigurationError("friction parameters must be finite")
        maximum = 0.5 if gripper else 2.0
        if not (
            0 <= self.coulomb_nm <= self.compensation_limit_nm <= maximum
            and 0 <= self.viscous_nm_s <= maximum
            and self.transition_rad_s >= 0.01
            and 0 < self.velocity_limit_rad_s <= 3.0
            and 0 <= self.damping_nm_s <= 5.0
        ):
            raise ConfigurationError(f"invalid friction profile (compensation cap <= {maximum} Nm)")

    def torque(self, velocity):
        assist = self.coulomb_nm * np.tanh(velocity / self.transition_rad_s)
        assist += self.viscous_nm_s * velocity
        return np.clip(assist, -self.compensation_limit_nm, self.compensation_limit_nm) - (
            self.damping_nm_s * velocity
        )


@dataclass(frozen=True)
class TeachingProfile:
    joints: tuple[FrictionJoint, ...]

    def __post_init__(self):
        if len(self.joints) != 7:
            raise ConfigurationError("YAM teaching needs six arm joints and one E_Yam gripper")
        for index, joint in enumerate(self.joints):
            joint.validate(gripper=index == 6)

    def wire_values(self):
        return tuple(tuple(float(v) for v in asdict(joint).values()) for joint in self.joints)


def example_config():
    return {
        "schema": SCHEMA,
        "note": "Uncalibrated: gravity plus damping only. Tune each physical arm independently.",
        "arms": {
            name: {"joints": [asdict(FrictionJoint()) for _ in range(7)]}
            for name in ("left", "right")
        },
    }


def load_profiles(path, names):
    try:
        data = json.loads(Path(path).read_text())
        if data["schema"] != SCHEMA or set(data) - {"schema", "note", "arms"}:
            raise ValueError("unknown schema or top-level fields")
        if set(data["arms"]) - {"left", "right"}:
            raise ValueError("unknown arm")
        result = {}
        for name in names:
            arm = data["arms"][name]
            if set(arm) != {"joints"}:
                raise ValueError(f"unknown fields for {name}")
            result[name] = TeachingProfile(tuple(FrictionJoint(**j) for j in arm["joints"]))
        return result
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise ConfigurationError(f"invalid teaching config {path}: {err}") from err


class NextStateActionSink:
    """Label observation t with measured q(t+1), never send those labels to motors.

    Keep one frame pending; drop the unpaired terminal frame on save. Each
    episode is isolated and images are copied because camera buffers may be reused.
    """

    def __init__(self, sink):
        self.sink = sink
        self.pending = None
        self.written = 0
        self.last_timestamp = None

    @property
    def num_episodes(self):
        return self.sink.num_episodes

    def add_frame(self, frame):
        timestamp = float(np.asarray(frame["observation.capture_monotonic_s"])[0])
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ConfigurationError("teaching capture timestamps must increase")
        if self.pending is not None:
            self.pending["action"] = np.asarray(frame["observation.state"], dtype=np.float32).copy()
            self.pending["action.dt_s"] = np.array([timestamp - self.last_timestamp], np.float32)
            self.sink.add_frame(self.pending)
            self.written += 1
        self.pending = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in frame.items()}
        self.last_timestamp = timestamp

    def save_episode(self):
        if self.written:
            self.sink.save_episode()
        else:
            self.sink.discard_episode()
        self.pending, self.written, self.last_timestamp = None, 0, None

    def discard_episode(self):
        self.pending, self.written, self.last_timestamp = None, 0, None
        self.sink.discard_episode()

    def finalize(self):
        self.pending = None
        self.sink.finalize()


class TeachingSource:
    """Explicit console events, with no position commands and loop-owned renewal."""

    def __init__(self, arms, profiles, event_reader, *, allow_record=False):
        self.arms = arms
        self.event_reader = event_reader
        self.allow_record = allow_record
        self.active = False
        # Verify every arm's capability before configuring either.
        if any(not arm.capabilities.supports_teaching for arm in arms.values()):
            raise ConfigurationError("native node lacks YAM teaching support; rebuild Karma")
        for name, arm in arms.items():
            arm.configure_teaching(profiles[name])

    def describe(self):
        return "YAM kinesthetic teaching (measured next-state actions; 250 ms native lease)"

    def hold(self):
        self.active = False
        errors = []
        for arm in self.arms.values():
            try:
                arm.hold()
            except Exception as err:  # finish stopping the other arm
                errors.append(err)
        if errors:
            raise errors[0]

    def poll(self, states):
        event = self.event_reader()
        if event in {"h", "s", "d", "q"}:
            self.hold()
            return TeleopStep(
                event={
                    "h": EpisodeEvent.DISCARD,
                    "s": EpisodeEvent.SAVE,
                    "d": EpisodeEvent.DISCARD,
                    "q": EpisodeEvent.STOP,
                }[event]
            )
        if event == "r" and not self.allow_record:
            raise ConfigurationError("r requires --record --repo-id and --task")
        if self.active or event in {"t", "r"}:
            try:
                for name in self.arms:
                    state = states.get(name)
                    if state is None or not state.is_fresh(0.1) or state.effector is None:
                        raise StaleStateError(f"{name}: missing/stale teaching feedback")
                    if self.active and state.mode is not ArmMode.TEACHING:
                        raise StaleStateError(f"{name}: native teaching guard returned to HOLD")
                if not self.active:
                    for arm in self.arms.values():
                        arm.enter_teaching()
                    self.active = True
                for arm in self.arms.values():
                    arm.renew_teaching()
            except BaseException:
                self.hold()
                raise
        return TeleopStep(
            event=EpisodeEvent.START if event == "r" else EpisodeEvent.NONE,
            extras={"observation.capture_monotonic_s": np.array([time.monotonic()], np.float64)},
        )

    def close(self):
        self.hold()


def console_event():
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        return "q" if line == "" else line.strip().lower()
    return ""


def run_cli(args, log_path):
    from . import cameras, cli, record

    if args.write_config:
        try:
            args.write_config.parent.mkdir(parents=True, exist_ok=True)
            # Never silently replace a tuned profile with zero compensation.
            with args.write_config.open("x") as file:
                json.dump(example_config(), file, indent=2)
                file.write("\n")
        except OSError as err:
            raise ConfigurationError(f"cannot write teaching profile: {err}") from err
        print(f"Wrote uncalibrated profile: {args.write_config}")
        return 0
    if args.config is None:
        raise ConfigurationError("teach needs --config; generate one with --write-config PATH")
    if not 20 <= args.fps <= 100:
        raise ConfigurationError("teaching fps must be between 20 and 100")
    if args.record and (not args.repo_id or not args.task or not args.task.strip()):
        raise ConfigurationError("recording requires --repo-id and --task")
    rig = cli._configured_rig(args)
    if args.record:
        rig = rig.with_camera_capture(fps=args.fps)
    if args.arm != "both":
        rig = rig.subset((args.arm,))
    profiles = load_profiles(args.config, rig.names)
    failures, reports = cli.preflight_rig(rig)
    for name, checks in reports:
        print(name)
        for check in checks:
            print(check.render())
    if failures:
        return 1
    if args.record:
        record.require_lerobot()
    readers, session, source, sink = {}, None, None, None
    live_arms = []
    try:
        if args.record:
            discovery = cameras.discover(
                rig.cameras, overrides=cameras.parse_camera_overrides(args.camera)
            )
            if not discovery.complete:
                raise ConfigurationError(f"missing teaching cameras: {discovery.missing}")
            readers = dict(cameras.open_readers(discovery.specs()))
        shapes = record.camera_shapes(readers)
        if args.record:
            features = record.build_features(
                record.arm_feature_names(rig.names, dict.fromkeys(rig.names, 6)),
                shapes,
                extra={
                    "observation.capture_monotonic_s": {
                        "dtype": "float64",
                        "shape": (1,),
                        "names": ["seconds"],
                    },
                    "action.dt_s": {"dtype": "float32", "shape": (1,), "names": ["seconds"]},
                },
            )
            raw_sink = record.LeRobotSink(
                repo_id=args.repo_id,
                fps=args.fps,
                features=features,
                robot_type="yam_kinesthetic_next_state_v1",
                root=args.root,
            )
            (raw_sink.root / "meta" / "teaching.json").write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "action_alignment": "next_measured_state",
                        "terminal_frame": "dropped",
                        "gripper": "0=open, 1=closed",
                        "fps": args.fps,
                        "arms": list(rig.names),
                        "profiles": {name: asdict(profile) for name, profile in profiles.items()},
                    },
                    indent=2,
                )
                + "\n"
            )
            sink = NextStateActionSink(raw_sink)
        else:
            sink = record.MemorySink()
        session, live_arms = cli.power_up(rig)
        arms = {entry.name: entry.arm for entry in live_arms}
        source = TeachingSource(arms, profiles, console_event, allow_record=args.record)
        print("Ready, holding. Type a key then Enter:")
        print("t: practice | r: record | s: hold + save | h/d: hold + discard | q: exit")
        print("Ctrl+C de-energizes in place; support the arms. No automatic parking.")
        record.record_session(
            arms=arms,
            source=source,
            sink=sink,
            task=args.task or "practice",
            fps=args.fps,
            cameras=readers,
            finalize=False,
        )
        return 0
    finally:
        # Stop/hold before any potentially slow dataset finalization.
        try:
            if source is not None:
                source.close()
        finally:
            try:
                if session is not None:
                    cli.power_down(session, live_arms, park=False)
            finally:
                for reader in readers.values():
                    reader.close()
                if sink is not None:
                    sink.finalize()
        print(f"log: {log_path}")
