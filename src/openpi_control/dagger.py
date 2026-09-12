"""Human-gated DAgger: a policy drives, an operator takes over, both are recorded.

``rollout`` records what a policy does. ``collect`` records what a human does.
This module records the thing in between, which is the one that produces
training data a policy can improve from: the policy drives, and when it is
about to fail the operator presses Right B, corrects the scene by hand, and
taps Left Y to give it back. Holding Left Y instead ends the attempt there and
then, because an episode is over when the operator says so and the ninety
seconds left on the clock record nothing. Every frame is written either way,
labelled with who was driving.

That labelling is the whole point, so it is worth being exact about what the
dataset means. Each frame pairs the *measured* state with the action actually
commanded on that tick -- the policy's action while it drives, the operator's
while they correct -- and an ``intervention`` column saying which. Training on
the intervention frames alone is HG-DAgger (Kelly et al., 2019): expert actions,
labelled on states the *policy* chose to visit, which is the distribution shift
that plain behaviour cloning never sees. The autonomous frames are kept too,
because they are what makes an episode reconstructible and what the manifest's
success label is about, and because filtering on one column is easy while
recovering frames nobody wrote is not.

What this deliberately does not do is query the policy during an intervention.
Original DAgger labels every state with the expert while the *policy* keeps
acting; here the policy is stopped the moment the human takes over, because on
real hardware there is only one set of arms and only one of them can have it.

Three things have to be true at a handoff or the data is worse than useless,
and each is argued where it is handled:

  * the operator's solver is anchored where the arm actually is, with the
    gripper the policy last *commanded* (:meth:`QuestTeleopSource.seed_from`);
  * the policy's queued chunk and any in-flight inference are dropped, because
    they describe a world the correction just invalidated
    (:meth:`InferenceRolloutSource.resume_after_intervention`);
  * the recorded action is whoever's hand was actually on the arm, never the
    other one's intent.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .inference import MOLMOACT_ARM_NAMES
from .record import ArmTarget, EpisodeEvent, TeleopSource, TeleopStep

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

    from .inference_record import InferenceRolloutSource
    from .teleop_vr import QuestTeleopSource
    from .types import ArmState

#: The dataset column that makes this a DAgger dataset rather than a rollout
#: with a human occasionally in shot. 1.0 on frames the operator commanded.
#:
#: ``float32`` rather than a boolean because it is what every LeRobot version
#: this cell has run accepts without argument, and because the consumers -- a
#: filter, a loss weight -- want a number anyway.
INTERVENTION_FEATURE: dict[str, dict] = {
    "intervention": {"dtype": "float32", "shape": (1,), "names": ["intervention"]}
}

#: How long Left Y must be held to end the episode instead of handing control
#: back. Long enough that the tap which hands back cannot reach it by accident,
#: short enough to still be a gesture rather than a wait.
END_EPISODE_HOLD_S = 1.0

_HUMAN = np.asarray([1.0], dtype=np.float32)
_POLICY = np.asarray([0.0], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class InterventionStats:
    """How much of one episode the operator drove."""

    interventions: int = 0
    human_frames: int = 0
    policy_frames: int = 0

    @property
    def frames(self) -> int:
        return self.human_frames + self.policy_frames

    @property
    def human_fraction(self) -> float:
        return self.human_frames / self.frames if self.frames else 0.0

    def summary(self) -> str:
        if not self.interventions:
            return "no intervention — fully autonomous"
        takeover = "takeover" if self.interventions == 1 else "takeovers"
        return (
            f"{self.interventions} {takeover}, {self.human_frames} human frames "
            f"({self.human_fraction:.0%} of {self.frames})"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "interventions": self.interventions,
            "human_frames": self.human_frames,
            "policy_frames": self.policy_frames,
            "human_fraction": round(self.human_fraction, 4),
        }


class DaggerSource(TeleopSource):
    """One episode in which control passes back and forth between the two.

    Owns the episode clock, which the policy source normally owns. It has to
    for two reasons: an episode that ends when the policy's timer expires would
    never end at all while a human is correcting, because a correcting human is
    exactly the state in which the policy is not being polled; and the operator
    can end an attempt early, which is not a thing the policy can be asked
    about.

    Only one driver is polled per tick. The other is not idling in the
    background -- the policy is not spending inference calls on commands that
    would be discarded, and the operator's solver is not integrating hand
    motion nobody asked for. Both are what makes the button press feel like a
    switch rather than a blend.
    """

    #: Operator-facing names for the two buttons, for the messages this prints.
    TAKE = "Right B"
    GIVE = "Left Y"

    def __init__(
        self,
        *,
        policy: InferenceRolloutSource,
        human: QuestTeleopSource,
        episode_seconds: float,
        end_hold_s: float = END_EPISODE_HOLD_S,
        stop: threading.Event | None = None,
        report: Callable[[str], None] | None = None,
        on_mode_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._policy = policy
        self._human = human
        self._episode_seconds = float(episode_seconds)
        self.end_hold_s = float(end_hold_s)
        self._stop = stop if stop is not None else threading.Event()
        self._say = report or (lambda message: print(f"  {message}", flush=True))
        self._on_mode_change = on_mode_change

        self._intervening = False
        # Levels, like everywhere else the kit's buttons are read: a held
        # button must take control once, not thirty times a second.
        self._last_take = False
        self._last_give = False
        # When the current press of GIVE began, and whether the hold has
        # already been spent on ending the episode. Both reset on release.
        self._give_pressed_at: float | None = None
        self._give_consumed = False
        self._episode_started_at: float | None = None
        self._saved = False
        # What was last commanded, whoever commanded it. Only the gripper is
        # ever read back out, and only at a handoff.
        self._last_targets: dict[str, ArmTarget] = {}
        self._interventions = 0
        self._human_frames = 0
        self._policy_frames = 0

    # ----------------------------------------------------------------- #
    # TeleopSource
    # ----------------------------------------------------------------- #

    def describe(self) -> str:
        return (
            f"DAgger over {self._policy.describe()}; {self.TAKE} takes over, "
            f"{self.GIVE} hands back, held {self.GIVE} ends the episode"
        )

    def poll(self, states: Mapping[str, ArmState | None]) -> TeleopStep:
        if self._stop.is_set() or self._saved:
            return TeleopStep(event=EpisodeEvent.STOP)

        if self._episode_started_at is None:
            self._episode_started_at = time.monotonic()
            event = EpisodeEvent.START
        else:
            event = EpisodeEvent.NONE

        if time.monotonic() - self._episode_started_at >= self._episode_seconds:
            # The backstop, for the attempt nobody ended by hand. It runs
            # during an intervention too: the clock measures a trial of the
            # scene, not of the policy, so two attempts stay comparable
            # however much help one of them needed.
            self._saved = True
            return TeleopStep(event=EpisodeEvent.SAVE)

        if self._apply_handoff(states):
            # Ends the attempt where it stands, whoever was driving. No targets
            # on this tick, exactly as the clock's own expiry: the arms hold
            # their last command until the runner parks them.
            self._saved = True
            self._say(
                f"■  EPISODE ENDED — {self.GIVE} held; saving what was captured and parking"
            )
            return TeleopStep(event=EpisodeEvent.SAVE)

        if self._intervening:
            targets = dict(self._human.poll(states).targets)
            self._human_frames += 1
        else:
            targets = self._policy.act(states)
            self._policy_frames += 1
        self._last_targets = targets
        return TeleopStep(
            targets=targets,
            event=event,
            extras={"intervention": _HUMAN if self._intervening else _POLICY},
        )

    def close(self) -> None:
        """Close the policy client only.

        The headset bridge outlives this source. It holds one WebSocket to the
        relay for the whole session, and reconnecting it between episodes would
        drop the operator out of teleoperation during the scene reset -- when
        they are most likely to be using it. Its owner closes it.
        """
        self._policy.close()

    # ----------------------------------------------------------------- #
    # who is driving
    # ----------------------------------------------------------------- #

    @property
    def intervening(self) -> bool:
        return self._intervening

    @property
    def stats(self) -> InterventionStats:
        return InterventionStats(
            interventions=self._interventions,
            human_frames=self._human_frames,
            policy_frames=self._policy_frames,
        )

    def _apply_handoff(self, states: Mapping[str, ArmState | None]) -> bool:
        """Act on the two buttons; True when the operator asked to end the episode.

        The two are resolved differently, on purpose.

        ``TAKE`` acts on the press. You reach for it because the policy is
        about to do something you want stopped, and that is the one place in
        this loop where a delay costs something real.

        ``GIVE`` carries two meanings, so it is resolved on the release: a tap
        hands control back, a hold past :attr:`end_hold_s` ends the
        episode instead. That costs the hand-back the length of the tap --
        which is nothing, because handing control *to* the policy is a
        deliberate act and never a rescue -- and buys an end-episode gesture
        that no tap can reach by accident.
        """
        take = self._human.right_b()
        # Each button is only live in the mode it means something in, so a
        # thumb resting on both is not an ambiguity to resolve.
        if take and not self._last_take and not self._intervening:
            self._take_over(states)
        self._last_take = take

        give, now = self._human.left_y(), time.monotonic()
        if give and not self._last_give:
            self._give_pressed_at = now
            self._give_consumed = False

        ended = False
        if (
            give
            and not self._give_consumed
            and self._give_pressed_at is not None
            and now - self._give_pressed_at >= self.end_hold_s
        ):
            # Spent here rather than on release, so the episode ends under the
            # operator's thumb rather than whenever they happen to let go.
            self._give_consumed = True
            ended = True

        if self._last_give and not give:
            if not self._give_consumed and self._intervening:
                self._hand_back(states)
            self._give_pressed_at = None
        self._last_give = give
        return ended

    def _take_over(self, states: Mapping[str, ArmState | None]) -> None:
        if not self._human.seed_from(states, effector=self._commanded_effector()):
            # Refusing is the safe half of this: an unseeded solver starts at
            # its configured rest pose, so the first commanded tick would be a
            # full-speed move to the park pose with the operator's hand
            # nowhere near it.
            self._say("!  TAKE-OVER IGNORED — no arm has published a pose yet")
            return
        self._intervening = True
        self._interventions += 1
        if self._on_mode_change is not None:
            self._on_mode_change(True)
        self._say(f"✋  HUMAN IN CONTROL — policy paused · {self.GIVE} hands it back")

    def _hand_back(self, states: Mapping[str, ArmState | None]) -> None:
        fresh = {name: states.get(name) for name in MOLMOACT_ARM_NAMES}
        missing = sorted(name for name, state in fresh.items() if state is None)
        if missing:
            # Staying in human control is the safe answer. The policy's
            # executor can only be seeded from a measured pose, and handing
            # back without one would have it step from a target nobody has
            # confirmed the arm is anywhere near.
            self._say(
                f"!  HAND-BACK IGNORED — no state from {', '.join(missing)}; "
                "you still have the arms"
            )
            return
        self._policy.resume_after_intervention(
            {name: state for name, state in fresh.items() if state is not None},
            effector=self._commanded_effector(),
        )
        self._intervening = False
        if self._on_mode_change is not None:
            self._on_mode_change(False)
        self._say("▶  POLICY IN CONTROL — re-planning from the corrected scene")

    @property
    def commanded_effector(self) -> dict[str, float]:
        """The gripper each arm was last commanded to, whoever commanded it.

        Public because the session's gripper-stall watch wants the commanded
        side of that comparison and this is the only object that knows it: the
        two drivers are polled alternately, so neither one sees every tick.
        """
        return self._commanded_effector()

    def _commanded_effector(self) -> dict[str, float]:
        """The gripper each arm was last *commanded* to, in native units.

        Both sides of a handoff want this and neither can use the measurement:
        a gripper holding something reads short of what closed it, so seeding
        the receiving side from the measurement gives back a slice of the grip
        at every handoff, and an object changes hands by being dropped.
        """
        return {
            name: target.effector
            for name, target in self._last_targets.items()
            if target.effector is not None
        }
