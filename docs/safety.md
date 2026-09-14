# Runtime safety checks

`uv run karma infer`, `rollout`, `hitl`, and `teleop` share Python-side
safety checks. Native motor fault handling, torque/thermal protection, command
limits, and shutdown remain active beneath these checks.

| Check | Behavior |
| --- | --- |
| Arm preflight | A failed check prevents startup. `--skip-preflight` explicitly bypasses preflight, not runtime checks. |
| Required policy cameras | Infer, rollout, and HITL check camera presence and capture modes; cameras must deliver initial frames before powering up. Teleop does not require policy cameras. |
| Motion settings | Non-finite or non-positive rates and step limits are rejected before hardware startup. |
| Initial arm states | Wait up to 10 seconds for fresh state before sending policy or VR targets. |
| State freshness | Pause commands above 250 ms state age; stop on 1 second of sustained stale/missing state. |
| Joint bounds | Load packaged URDF limits in joint-chain order, with or without visualization. Both policy and human commands are bounded. Native instance limits continue to apply independently. |
| Gripper readings | Print measured gripper position at startup; report a gripper that remains stationary across substantial commanded travel. This is a warning, not an automatic stop. |
| Teardown | Attempt parking/de-energizing even when source cleanup fails. `--no-park`, where offered, still shuts down hardware without a parking move. |

HITL checks state freshness during human takeover as well as policy control.
Its recorder and rollout use the same 250 ms freshness threshold. A prolonged
recording stall may discard the open take to avoid retaining a broken trajectory.
The recorder writes the bounded joint targets actually issued, including human
corrections.

The policy's incremental action limits and VR's IK/clutch limits remain
controller-specific; a per-tick step at 30 Hz is not equivalent to one at 200 Hz.
These checks do not provide collision avoidance or certify physical safety.

Regression tests in `tests/test_safety.py` exercise shared preflight, stale-state
recovery/failure, headless limits, invalid configuration, and shutdown failures.
`tests/test_record.py` checks that bounded commands match saved actions and that
an invalid command for one arm prevents commands to either arm on that tick.
