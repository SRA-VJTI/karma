"""Driving the arms from the browser: a control surface over the viser scene.

:mod:`openpi_control.viz` draws and never opens a bus; this module is the half
that commands. Keeping them apart is deliberate -- the visualizer stays safe to
run against a dead cell, and everything that can move an arm lives here.

The design point that lets sliders and a live hardware feed coexist -- which
``live`` previously refused to attempt, because they fight over the pose -- is
that they are not both allowed to own it:

* The **render always follows the hardware.** Every frame draws the newest
  published state, armed or not, so the screen stays a measurement and never
  becomes a wish.
* The **sliders are the target**, not the drawing. While disarmed they are
  slaved to the measured pose, so they cannot go stale behind an arm that
  someone pushed by hand; while armed the operator owns them and nothing
  writes back.

Around that sits a safety gate modelled on i2rt's ``ViserControlInterface``,
with the parts it leaves to the operator's judgment made explicit. A panel

* refuses to arm on a state that is stale, missing, or disagrees with what is
  on screen, rather than trusting a checkbox that says it matches,
* walks toward the slider target at a bounded joint rate instead of sending
  wherever the slider landed, so a fast drag is a move you can watch,
* disarms itself when the state goes quiet, when a command is rejected, or
  when the browser disconnects -- an unattended tab must not keep an arm
  tracking a target nobody is watching.

Each arm carries its own gate, so a bimanual cell can be driven one side at a
time from a single page::

    from openpi_control.viser_control import RigControlPanel

    control = RigControlPanel(scene, {"left": left_follower})
    while running:
        control.step(dt)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .exceptions import AlignmentError, ConfigurationError, PiControlError
from .types import PositionCommand

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

    from .arms import FollowerArm
    from .types import ArmState
    from .viz import ArmSceneVisualizer, ArmVisualizer

# Slider traffic is pushed every frame while disarmed; skip the write when the
# value has not meaningfully moved so an idle arm is not a websocket firehose.
_SLIDER_EPSILON = 1e-4

_STATUS_DISARMED = "**DISARMED** — mirroring only, nothing is sent"
_STATUS_ARMED = "**ARMED** — sliders command the arm"


@dataclass(frozen=True, slots=True)
class ControlLimits:
    """The bounds every browser-authored command passes through.

    ``max_joint_rate_rad_s`` is the one that matters: it turns a slider from a
    teleport into a velocity. Dragging a joint from stop to stop then takes
    seconds rather than a single control tick, which is the difference between
    a move you can interrupt and a lurch you watch happen.

    ``stale_state_s`` is both the arming precondition and the running
    watchdog -- an arm that stops publishing is an arm we have stopped being
    able to see, and commanding blind is exactly the case worth refusing.
    """

    max_joint_rate_rad_s: float = 0.5
    max_effector_rate_s: float = 1.0
    stale_state_s: float = 0.25
    pose_tolerance_rad: float = 0.05


class ArmControlPanel:
    """One follower's gate, sliders, and per-tick command.

    The panel owns no thread. :meth:`step` is called by whoever is already
    pumping the scene -- ``live``'s mirror loop -- so the render and the
    commands stay on one clock and cannot interleave from two.
    """

    def __init__(
        self,
        follower: FollowerArm,
        viz: ArmVisualizer,
        *,
        folder: str | None = None,
        limits: ControlLimits | None = None,
        float_mode: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.follower = follower
        self.viz = viz
        self.name = viz.name
        self.limits = limits or ControlLimits()
        self.float_mode = float_mode
        self._log = log

        specs = viz.joint_specs
        capabilities = follower.capabilities
        if capabilities.dof != len(specs):
            # Refuse rather than pad or truncate: a URDF that disagrees with the
            # node about how many joints there are cannot produce a command whose
            # indices mean what the sliders say they mean.
            raise ConfigurationError(
                f"{self.name}: URDF has {len(specs)} actuated joints but the node "
                f"reports {capabilities.dof} — cannot map sliders to joints"
            )
        self._has_effector = capabilities.has_effector
        self._can_command = capabilities.supports_direct_commands
        self._can_float = capabilities.supports_gravity_compensation

        self._lower = np.array([spec.lower for spec in specs], dtype=np.float64)
        self._upper = np.array([spec.upper for spec in specs], dtype=np.float64)

        # Guards the armed flag and the target against the viser callback
        # thread, which arms and disarms while the loop thread is mid-step.
        self._lock = threading.RLock()
        self._armed = False
        self._target: np.ndarray | None = None
        self._effector_target: float | None = None

        self._build_gui(folder or f"{self.name} — control")

    # ------------------------------------------------------------------ #
    # GUI
    # ------------------------------------------------------------------ #

    def _build_gui(self, folder: str) -> None:
        gui = self.viz.server.gui
        with gui.add_folder(folder):
            self._status = gui.add_markdown(_STATUS_DISARMED)
            # The checkbox is the "look at the actual arm" step, and it is the
            # only part of the gate a machine cannot do. Everything it claims
            # in i2rt -- that the render matches, that the state is live -- is
            # checked in _verify_ready instead of taken on trust.
            self._confirm = gui.add_checkbox("Pose on screen matches the arm", False)
            self._arm_button = gui.add_button("Arm")
            self._arm_button.disabled = True
            self._disarm_button = gui.add_button("Disarm")
            self._disarm_button.disabled = True

            self._sliders: list[Any] = []
            for index, spec in enumerate(self.viz.joint_specs):
                slider = gui.add_slider(
                    spec.name,
                    min=spec.lower,
                    max=spec.upper,
                    step=0.002,
                    initial_value=float(self.viz.positions[index]),
                )
                slider.disabled = True
                self._sliders.append(slider)

            self._effector_slider: Any | None = None
            if self._has_effector:
                # Normalized, because that is what PositionCommand takes: the
                # node owns the mapping from [0, 1] to this gripper's travel.
                # 1.0 is OPEN, not closed -- the native effector's ready pose is
                # normalized 1.0 and it opens the gripper. This slider's value
                # goes straight into a PositionCommand, so a label that had the
                # two ends the wrong way round would invite an operator to close
                # a gripper while believing they were opening it. The initial
                # value is open for the same reason: it is what shows before the
                # first state arrives, and closed is the wrong way to guess.
                self._effector_slider = gui.add_slider(
                    "gripper (0 closed — 1 open)", min=0.0, max=1.0, step=0.01, initial_value=1.0
                )
                self._effector_slider.disabled = True

        @self._confirm.on_update
        def _(_: Any) -> None:
            with self._lock:
                self._arm_button.disabled = self._armed or not self._confirm.value

        @self._arm_button.on_click
        def _(_: Any) -> None:
            try:
                self.arm()
            except PiControlError as err:
                # A refused arming is the gate working, not a crash: say why on
                # the page and in the run log, and stay disarmed.
                self._status.content = f"**REFUSED** — {err}"
                self._log(f"  {self.name}: refused to arm — {err}")

        @self._disarm_button.on_click
        def _(_: Any) -> None:
            self.disarm("operator")

    # ------------------------------------------------------------------ #
    # Arming
    # ------------------------------------------------------------------ #

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def _verify_ready(self, state: ArmState | None) -> np.ndarray:
        """Check every precondition for commanding; return the measured pose.

        Raises :class:`AlignmentError` naming the one that failed, so the
        browser can show the operator what to fix instead of a dead button.
        """
        if not self._can_command:
            raise AlignmentError(f"{self.name} does not accept direct position commands")
        if state is None:
            raise AlignmentError(f"{self.name} has published no state yet")
        if not state.is_fresh(self.limits.stale_state_s):
            raise AlignmentError(f"{self.name} state is {state.age_s * 1e3:.0f} ms old")
        measured = np.asarray(state.joints.position_rad, dtype=np.float64)
        shown = self.viz.positions
        if measured.size != shown.size:
            raise AlignmentError(
                f"{self.name} publishes {measured.size} joints, the render has {shown.size}"
            )
        drift = float(np.max(np.abs(measured - shown))) if measured.size else 0.0
        if drift > self.limits.pose_tolerance_rad:
            # The render clamps to the URDF limits, so this also catches an arm
            # standing outside the model's own joint range -- arming there would
            # snap it back to the limit on the first command.
            raise AlignmentError(
                f"{self.name} render is {drift:.3f} rad off the measured pose "
                f"(limit {self.limits.pose_tolerance_rad:.3f})"
            )
        if self._has_effector and state.effector is None:
            raise AlignmentError(f"{self.name} has an effector but publishes no effector state")
        return measured

    def arm(self) -> None:
        """Take control: verify, re-engage position control, seed the target."""
        with self._lock:
            if self._armed:
                return
            state = self.follower.latest_state
            measured = self._verify_ready(state)
            assert state is not None  # _verify_ready rejects None

            # After a gravity float the node has suspended position control, and
            # a direct command would land on a mode that is not listening. hold()
            # re-engages it at the pose the arm is standing in, which is also the
            # pose we are about to command -- so this is a no-op move, not a jump.
            self.follower.hold()

            self._target = np.clip(measured, self._lower, self._upper)
            self._effector_target = (
                float(state.effector.position) if state.effector is not None else None
            )
            self._push_sliders(self._target, self._effector_target)
            self._armed = True
            for slider in self._sliders:
                slider.disabled = False
            if self._effector_slider is not None:
                self._effector_slider.disabled = False
            self._arm_button.disabled = True
            self._disarm_button.disabled = False
            self._status.content = _STATUS_ARMED
        self._log(f"  {self.name}: ARMED — browser sliders are commanding")

    def disarm(self, reason: str = "operator") -> None:
        """Stop commanding and put the arm back in its resting mode.

        Safe to call on an already-disarmed panel and from any thread: this is
        the path both the Disarm button and every automatic trip go through.
        """
        with self._lock:
            if not self._armed:
                return
            self._armed = False
            self._target = None
            self._effector_target = None
            for slider in self._sliders:
                slider.disabled = True
            if self._effector_slider is not None:
                self._effector_slider.disabled = True
            # Clear the confirmation too: re-arming should mean looking at the
            # arm again, not clicking one button after a trip nobody read.
            self._confirm.value = False
            self._arm_button.disabled = True
            self._disarm_button.disabled = True
            self._status.content = f"{_STATUS_DISARMED} — disarmed: {reason}"

        self._log(f"  {self.name}: DISARMED — {reason}")
        try:
            if self.float_mode and self._can_float:
                self.follower.enter_gravity_compensation()
            else:
                self.follower.hold()
        except PiControlError as err:
            # Report and continue: the arm has already stopped receiving new
            # targets, which is the part that mattered, and the session's own
            # power-down still owns putting it down.
            self._log(f"  {self.name}: could not restore resting mode — {err}")

    # ------------------------------------------------------------------ #
    # Per-tick
    # ------------------------------------------------------------------ #

    def step(self, dt: float) -> None:
        """One tick: mirror into the sliders, or send one bounded command."""
        state = self.follower.latest_state
        with self._lock:
            if not self._armed:
                self._follow_state(state)
                return
            if state is None or not state.is_fresh(self.limits.stale_state_s):
                trip = "state went stale" if state is not None else "state stopped arriving"
                command = None
            else:
                trip = None
                command = self._advance(dt)
        if trip is not None:
            self.disarm(trip)
            return
        if command is None:
            return
        try:
            self.follower.command(command)
        except PiControlError as err:
            self.disarm(f"command rejected: {err}")

    def _advance(self, dt: float) -> PositionCommand:
        """Step the held target toward the sliders, at most one tick's travel."""
        assert self._target is not None
        goal = np.clip(
            np.array([float(slider.value) for slider in self._sliders], dtype=np.float64),
            self._lower,
            self._upper,
        )
        span = self.limits.max_joint_rate_rad_s * dt
        self._target = self._target + np.clip(goal - self._target, -span, span)

        effector: float | None = None
        if self._effector_slider is not None and self._effector_target is not None:
            reach = self.limits.max_effector_rate_s * dt
            delta = float(self._effector_slider.value) - self._effector_target
            self._effector_target += float(np.clip(delta, -reach, reach))
            effector = float(np.clip(self._effector_target, 0.0, 1.0))
        return PositionCommand(self._target.copy(), effector=effector)

    def _follow_state(self, state: ArmState | None) -> None:
        """Slave the disarmed sliders to the arm, so arming never jumps."""
        if state is None:
            return
        measured = np.asarray(state.joints.position_rad, dtype=np.float64)
        if measured.size != len(self._sliders):
            return
        self._push_sliders(
            np.clip(measured, self._lower, self._upper),
            None if state.effector is None else float(state.effector.position),
        )

    def _push_sliders(self, positions: np.ndarray, effector: float | None) -> None:
        for slider, value in zip(self._sliders, positions.tolist(), strict=True):
            if abs(float(slider.value) - value) > _SLIDER_EPSILON:
                slider.value = float(value)
        if self._effector_slider is not None and effector is not None:
            if abs(float(self._effector_slider.value) - effector) > _SLIDER_EPSILON:
                self._effector_slider.value = float(np.clip(effector, 0.0, 1.0))


class RigControlPanel:
    """Every named follower of a rig, each behind its own gate, on one page.

    One process, one port, one world frame -- so a bimanual cell is driven from
    a single browser tab with both arms drawn where the rig says their bases
    are, rather than one tab per arm with no shared frame between them.
    """

    def __init__(
        self,
        scene: ArmSceneVisualizer,
        followers: Mapping[str, FollowerArm],
        *,
        limits: ControlLimits | None = None,
        float_mode: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        if not followers:
            raise ConfigurationError("browser control needs at least one follower arm")
        self.scene = scene
        self._log = log
        server = scene.server

        # Built before the per-arm folders so it lands at the top of the panel:
        # the control that stops everything should not be somewhere you scroll to.
        with server.gui.add_folder("Safety"):
            disarm_all = server.gui.add_button("Disarm all")

        self.panels: dict[str, ArmControlPanel] = {
            name: ArmControlPanel(
                follower,
                scene[name],
                limits=limits,
                float_mode=float_mode,
                log=log,
            )
            for name, follower in followers.items()
        }

        @disarm_all.on_click
        def _(_: Any) -> None:
            self.disarm_all("operator stop")

        @server.on_client_disconnect
        def _(_: Any) -> None:
            # Any disconnect disarms everything, not just the last one. With two
            # people watching, disarming when one tab closes costs a re-arm; the
            # other reading -- keep commanding because someone else is still
            # connected -- risks an arm tracking a slider nobody has hold of.
            self.disarm_all("browser disconnected")

    def __getitem__(self, name: str) -> ArmControlPanel:
        try:
            return self.panels[name]
        except KeyError:
            raise ConfigurationError(
                f"unknown arm {name!r}; this panel drives: {', '.join(self.panels)}"
            ) from None

    @property
    def armed_names(self) -> tuple[str, ...]:
        return tuple(name for name, panel in self.panels.items() if panel.armed)

    def step(self, dt: float) -> None:
        for panel in self.panels.values():
            panel.step(dt)

    def disarm_all(self, reason: str = "operator") -> None:
        for panel in self.panels.values():
            panel.disarm(reason)
