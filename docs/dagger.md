# Human-in-the-loop recording (DAgger)

`openpi hitl` records the case `collect` and `rollout` each miss half of: the
policy drives, and when it is about to fail the operator takes the arms,
corrects the scene, and gives them back. Every frame is written either way,
labelled with who was driving.

```bash
uv run openpi hitl \
    --repo-id you/yam-fold-towel-dagger-v1 \
    --root ~/openpi-data/dagger/fold-pink-towel-v1 \
    --episodes 5 \
    --episode-seconds 120 \
    --server http://192.168.0.107:4090 \
    --interface left=can_left \
    --interface right=can_right \
    --speed 0.5 \
    --open-quest
```

One command starts the Quest relay, creates the `adb reverse` tunnel, opens the
three cameras once and lends them to the policy, the dataset, Viser, and the
headset, then runs the attempts.

## The two buttons

| In VR | Does |
| --- | --- |
| **Right B** | Take the arms. The policy stops: no chunk is executed, no inference is requested. |
| **Left Y**, tapped | Hand back. The policy re-plans from what the cameras see now and continues. |
| **Left Y**, held 1 s | End this attempt now: save what was captured, park both arms, go to the `y/n` prompt, then the next attempt. |

They are the same two buttons `collect` uses to start and save an episode,
because they are the two a gloved thumb finds without looking.

Pressing B does not move anything by itself. It hands you an armed teleoperator
seeded where the arm already is; squeeze the grip to start driving, exactly as
in `teleop`. A press that would be ambiguous is ignored rather than guessed at:
B while you are already driving does nothing.

The two are resolved differently, and it is worth knowing which way round.
**B acts on the press** — you reach for it because the policy is about to do
something you want stopped, and that is the one place in this loop where a
delay costs something real. **Y acts on the release**, because it carries two
meanings: a tap hands back, a hold past a second ends the attempt. That costs
the hand-back the length of your tap, which is nothing — handing control *to*
the policy is a deliberate act and never a rescue — and it buys an end-episode
gesture that no tap can reach by accident.

Holding Y works whether you or the policy is driving. The arms park either way,
so having to hand back first would be a rule to remember at the exact moment
there is nothing to hand back to.

Ctrl-C does the same thing from the keyboard, if you have the headset off.

## What a handoff actually does

Three things have to be true at the moment control changes hands, and each is
one of the ways this can silently produce a dataset worse than no dataset.

**The receiving side inherits the commanded gripper, not the measured one.** A
jaw holding a towel reads short of the value that closed it — that is what
holding looks like. Seed the operator's trigger (or the policy's integrator)
from that measurement and the grip is handed over already giving way; do it at
every handoff and the object is put down by degrees. Joints still come from the
measurement, because that is genuinely where the arm is.

**The policy's queued chunk is dropped.** The remaining actions in it continue a
trajectory planned before the human touched anything, and replaying them
executes, at full speed, the motion the correction just undid. Any inference
still in flight is dropped with it — it was asked for from an observation that
is no longer true, and it cannot land in a later request's slot.

**The recorded action is whoever's hand was on the arm.** Never the other's
intent.

## What lands on disk

LeRobot v3 under `--root`, with the schema `collect` writes plus one column:

| Column | Is |
| --- | --- |
| `observation.state` | the measured pose, as always |
| `action` | what was actually commanded this tick, by whoever was driving |
| `intervention` | `1.0` while the operator was driving, `0.0` while the policy was |
| `task` | that attempt's prompt, on every frame |

Beside it, `openpi_control_hitl.json` carries per-attempt prompt, `y/n` success
label, whether Ctrl-C cut it short, and the takeover counts:

```json
{
  "attempt": 2,
  "episode_index": 1,
  "prompt": "fold the towel in half",
  "success": true,
  "saved": true,
  "aborted": false,
  "interventions": 2,
  "human_frames": 340,
  "policy_frames": 3260,
  "human_fraction": 0.0944
}
```

`human_fraction` trending down across sessions is the thing you are actually
trying to achieve, and it is why the counts are in the manifest rather than
left to be re-derived from the parquet.

## Training on it

Filtering to `intervention == 1.0` gives you HG-DAgger (Kelly et al., 2019):
expert actions labelled on states the *policy* chose to visit, which is exactly
the distribution that behaviour cloning on `collect` data never sees. The
autonomous frames are kept anyway — they make an episode reconstructible, they
are what the success label is about, and filtering one column is easy while
recovering frames nobody wrote is not.

What this deliberately does not do is query the policy during an intervention.
Original DAgger labels every visited state with the expert while the policy
keeps acting; on real hardware there is one set of arms and only one of you can
have it. Stopping the policy on takeover is the honest version, and it is also
the one the operator can reason about while reaching into a moving cell.

## Session shape

Same as `rollout`: a prompt per attempt, a timed run, both arms parked, then
`Episode N successful? [y/n]`. The arms de-energize between attempts so the
scene can be reset by hand.

The relay, the USB tunnel, and the headset connection are opened once around
the whole run instead of per attempt — dropping the operator's WebSocket every
time would put the headset through its reconnect exactly when they are reaching
into the cell to reset the towel.

`--episode-seconds` is a backstop, not the normal way an attempt ends: hold Y
when the towel is folded or on the floor, and the clock only catches the
attempt you forgot about. The clock runs during interventions too, so two
attempts stay comparable however much help one of them needed.

Ctrl-C during an attempt does exactly what holding Y does — saves the frames
already captured, parks both arms, and continues to the label prompt — the same
as `rollout` and deliberately unlike `collect`. A second Ctrl-C at a prompt
exits the run.

## What it checks before it drives

`hitl` runs the same guards the continuous `infer` command does, because an
attempt that fails on one of them wastes a scene reset as surely as a bad fold.

**Preflight.** The doctor checks for both arms, plus all three cameras — which
are *required* here, not optional, since the policy is handed every view on
every request. Nothing is energized if a check fails. `--skip-preflight` opts
out.

**It waits for the first arm state.** `power_up` returns when the nodes are
energized, which is before either has necessarily published. The policy treats
a stale state as fatal, so the session used to die on a state a millisecond
over the 250 ms limit, one tick after energizing. It now waits for both arms to
start streaming and says so; ten seconds of silence is still a hard failure,
because a node that has not published by then is not late, it is not running.

**It states the gripper before anything moves.** The measured opening of each
gripper is printed at the top of every attempt:

```
  gripper  left 0.998, right 1.000 (1.0 = open) — check the jaws agree
```

Look at the jaws and check they agree with the number. Viser cannot show you
this — the packaged YAM URDF bakes the gripper into `link_6`, so the render
shows the same jaws whatever the gripper is doing — and a reading that
disagrees with the hardware means the gripper servo is zeroed at the wrong
stop. This is the one place you find that out before it tries to grip a towel.

**It watches for a gripper that never moves.** An inert gripper is invisible to
every other check: the node keeps publishing, the joints keep tracking, only
the jaws are dead. If one is commanded across its range without the measurement
moving once, the run says so — once per arm, on stderr. It is watched on the
record clock rather than the policy's, so a gripper that dies while the
operator is correcting is caught too, and `--no-viz` does not switch it off.

## Options worth knowing

| Flag | Does |
| --- | --- |
| `--episode-seconds` | backstop duration for an attempt you do not end by hand; the clock runs during interventions too |
| `--speed` | policy chunk playback speed; `0.5` is this cell's conservative default |
| `--no-reset-pause` | do not wait for the manual scene reset between attempts |
| `--quest-transport lan` | direct HTTPS instead of the USB tunnel; needs `--ssl-keyfile` and `--ssl-certfile` |
| `--no-relay` | teleoperate through an already-running `openpi relay` |
| `--push-to-hub` | upload after the arms are down |

Everything `rollout` accepts for the policy wire (`--num-steps`, `--chunk-size`,
`--raw-frames`, `--no-prefetch`, …) is accepted here and means the same thing;
see [inference.md](inference.md).
